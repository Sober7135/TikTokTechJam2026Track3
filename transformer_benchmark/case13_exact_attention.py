"""Lazy CUDA binding for exact Case-13 shared-CTA attention."""

from __future__ import annotations

import threading
from pathlib import Path
from types import ModuleType

import torch

from .native_softmax_bf16 import (
    _cuda_cccl_include,
    _cuda_runtime_include,
    _extension_build_path,
    _extension_tool_directory,
)


_EXTENSION: ModuleType | None = None
_EXTENSION_LOCK = threading.Lock()


def _load_extension() -> ModuleType:
    """Compile the fixed-shape extension once before benchmark timing."""
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
                    name="techjam_case13_exact_attention_v1",
                    sources=[str(source)],
                    extra_cuda_cflags=[
                        "-O3",
                        "-lineinfo",
                        f"-I{_cuda_runtime_include()}",
                        f"-I{_cuda_cccl_include()}",
                    ],
                    verbose=False,
                )
    return _EXTENSION


def case13_exact_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    context: torch.Tensor,
    scale: float,
) -> None:
    """Write exact Case-13 causal attention directly into ``context``."""
    expected_shape = (64, 4, 1024, 32)
    tensors = (query, key, value, context)
    if any(tuple(tensor.shape) != expected_shape for tensor in tensors):
        raise ValueError("Case-13 exact attention requires 64x4x1024x32 tensors")
    if any(tensor.device.type != "cuda" for tensor in tensors):
        raise ValueError("Case-13 exact attention requires CUDA tensors")
    if any(tensor.device != query.device for tensor in tensors):
        raise ValueError("Case-13 exact attention tensors must share one device")
    if any(tensor.dtype != torch.bfloat16 for tensor in tensors):
        raise ValueError("Case-13 exact attention requires BF16 tensors")
    if any(tensor.stride(-1) != 1 for tensor in tensors):
        raise ValueError("Case-13 exact attention requires unit-stride head dims")

    _load_extension().case13_exact_attention(
        query,
        key,
        value,
        context,
        float(scale),
    )
