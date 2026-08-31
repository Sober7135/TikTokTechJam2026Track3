"""Lazy CUDA binding for the exact Case-6 shared-CTA attention path."""

from __future__ import annotations

import os
import subprocess
import threading
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Iterator

import torch


_EXTENSION: ModuleType | None = None
_EXTENSION_LOCK = threading.Lock()


def _extension_tool_directory(
    tool_directory: Path = Path("/run/current-system/sw/bin"),
) -> Path:
    missing = []
    for tool in ("ninja", "c++", "nvcc"):
        executable = tool_directory / tool
        try:
            resolved = executable.resolve(strict=True)
        except FileNotFoundError:
            missing.append(tool)
            continue
        if not resolved.is_file() or not os.access(resolved, os.X_OK):
            missing.append(tool)
    if missing:
        raise RuntimeError(
            "the Nix extension build tools are unavailable: " + ", ".join(missing)
        )
    return tool_directory


@contextmanager
def _extension_build_path(tool_directory: Path) -> Iterator[None]:
    original_path = os.environ.get("PATH")
    original_cc = os.environ.pop("CC", None)
    os.environ["PATH"] = (
        str(tool_directory)
        if original_path is None
        else str(tool_directory) + os.pathsep + original_path
    )
    try:
        yield
    finally:
        if original_path is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = original_path
        if original_cc is None:
            os.environ.pop("CC", None)
        else:
            os.environ["CC"] = original_cc


def _cuda_runtime_include() -> Path:
    cudart = Path("/run/current-system/sw/lib/libcudart.so")
    if not cudart.exists():
        raise RuntimeError("the Nix CUDA runtime symlink is unavailable")
    include = cudart.resolve().parent.parent / "include"
    if not (include / "cuda_runtime.h").is_file():
        raise RuntimeError("the Nix CUDA runtime headers are unavailable")
    return include


def _cuda_cccl_include() -> Path:
    cudart_root = Path("/run/current-system/sw/lib/libcudart.so").resolve().parent.parent
    nix_store = Path("/run/current-system/sw/bin/nix-store")
    if not nix_store.is_file():
        raise RuntimeError("the Nix store query tool is unavailable")
    query = subprocess.run(
        [str(nix_store), "-q", "--references", str(cudart_root)],
        check=True,
        capture_output=True,
        text=True,
    )
    candidates = [
        Path(reference) / "include"
        for reference in query.stdout.splitlines()
        if "cuda_cccl" in Path(reference).name
    ]
    valid = [include for include in candidates if (include / "nv/target").is_file()]
    if len(valid) != 1:
        raise RuntimeError("the active Nix CUDA runtime has no unique CCCL headers")
    return valid[0]


def _load_extension() -> ModuleType:
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION

    with _EXTENSION_LOCK:
        if _EXTENSION is None:
            source = Path(__file__).with_suffix(".cu")
            tool_directory = _extension_tool_directory()
            with _extension_build_path(tool_directory):
                from torch.utils.cpp_extension import load

                _EXTENSION = load(
                    name="techjam_case6_exact_attention_v1",
                    sources=[str(source)],
                    extra_cuda_cflags=[
                        "-O3",
                        f"-I{_cuda_runtime_include()}",
                        f"-I{_cuda_cccl_include()}",
                    ],
                    verbose=False,
                )
    return _EXTENSION


def case6_exact_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    context: torch.Tensor,
    scale: float,
) -> None:
    """Write exact Case-6 causal attention for one <=512 batch slice."""
    tensors = (query, key, value, context)
    if query.ndim != 4 or not 0 < query.shape[0] <= 512:
        raise ValueError("Case-6 attention requires a non-empty <=512 batch slice")
    expected_shape = (query.shape[0], 4, 128, 32)
    if any(tuple(tensor.shape) != expected_shape for tensor in tensors):
        raise ValueError("Case-6 attention requires exact matching BHSD shapes")
    if any(tensor.device.type != "cuda" for tensor in tensors):
        raise ValueError("Case-6 attention requires CUDA tensors")
    if any(tensor.device != query.device for tensor in tensors):
        raise ValueError("Case-6 attention tensors must share one CUDA device")
    if any(tensor.dtype != torch.bfloat16 for tensor in tensors):
        raise ValueError("Case-6 attention requires BF16 tensors")
    if any(tensor.stride(-1) != 1 for tensor in tensors):
        raise ValueError("Case-6 attention requires unit-stride head dimensions")

    _load_extension().case6_exact_attention(
        query,
        key,
        value,
        context,
        float(scale),
    )
