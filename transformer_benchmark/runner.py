"""Command-line parsing, result serialization, and benchmark orchestration."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .cases import OFFICIAL_TEST_CASES, TransformerConfig
from .correctness import AccuracySummary, run_accuracy_tests
from .models import (
    BaselineTransformer,
    UserOptimizedTransformer,
    copy_model_weights,
)
from .timing import benchmark_models


NIXOS_LIBCUDA_DIRECTORY = Path("/run/opengl-driver/lib")


def configure_triton_driver_path(
    driver_directory: Path = NIXOS_LIBCUDA_DIRECTORY,
) -> None:
    """Point Triton's supported driver lookup at the NixOS driver directory."""
    if "TRITON_LIBCUDA_PATH" in os.environ:
        return
    if driver_directory.joinpath("libcuda.so.1").is_file():
        os.environ["TRITON_LIBCUDA_PATH"] = str(driver_directory)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return device


def resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[dtype_name]


def release_case_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def maybe_compile(model: nn.Module, enabled: bool, mode: str) -> nn.Module:
    if not enabled:
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("this PyTorch build does not provide torch.compile")
    return torch.compile(model, mode=mode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a baseline and optimized PyTorch Transformer"
    )
    parser.add_argument(
        "--official-matrix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run the declared 14-case matrix (enabled by default)",
    )
    parser.add_argument(
        "--official-cases",
        type=int,
        nargs="+",
        metavar="ID",
        help="run only these one-based official case IDs",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--causal", action="store_true")

    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, cuda:0, ..."
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--padding-ratio", type=float, default=0.0)
    parser.add_argument("--input-scale", type=float, default=1.0)

    parser.add_argument("--accuracy-trials", type=int, default=5)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--benchmark-rounds", type=int, default=3)
    parser.add_argument("--benchmark-on-failure", action="store_true")

    parser.add_argument("--compile-baseline", action="store_true")
    parser.add_argument("--compile-user", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    parser.add_argument("--non-strict-weight-copy", action="store_true")
    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium"),
        default="high",
    )
    parser.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable/disable TF32 on CUDA for both implementations",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="write a versioned machine-readable result to this path",
    )
    return parser.parse_args()


def validate_args(
    args: argparse.Namespace, device: torch.device, dtype: torch.dtype
) -> None:
    if args.official_cases and not args.official_matrix:
        raise ValueError("--official-cases requires --official-matrix")
    if args.official_cases:
        invalid_case_ids = sorted(
            {case_id for case_id in args.official_cases if not 1 <= case_id <= 14}
        )
        if invalid_case_ids:
            raise ValueError(f"invalid official case IDs: {invalid_case_ids}")
        if len(set(args.official_cases)) != len(args.official_cases):
            raise ValueError("--official-cases must not contain duplicates")
    if args.official_matrix and args.official_cases is None and device.type != "cuda":
        raise ValueError(
            "the full official matrix requires CUDA; use --official-cases for a "
            "CPU subset or --no-official-matrix for a custom CPU smoke test"
        )
    if not 0.0 <= args.padding_ratio < 1.0:
        raise ValueError("padding_ratio must be in [0, 1)")
    if args.input_scale <= 0:
        raise ValueError("input_scale must be positive")
    if args.accuracy_trials <= 0:
        raise ValueError("accuracy_trials must be positive")
    if args.rtol < 0 or args.atol < 0:
        raise ValueError("rtol and atol must be non-negative")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if args.repeats <= 0 or args.benchmark_rounds <= 0:
        raise ValueError("repeats and benchmark_rounds must be positive")
    if device.type == "cpu" and dtype == torch.float16:
        print("[warning] float16 CPU kernels may be unsupported or slow")


def environment_metadata(device: torch.device, dtype: torch.dtype) -> Dict[str, object]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "device": str(device),
        "dtype": str(dtype),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    }


def settings_metadata(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "seed": args.seed,
        "padding_ratio": args.padding_ratio,
        "input_scale": args.input_scale,
        "accuracy_trials": args.accuracy_trials,
        "rtol": args.rtol,
        "atol": args.atol,
        "accuracy_rule": "strict_less_than_or",
        "warmup": args.warmup,
        "repeats": args.repeats,
        "benchmark_rounds": args.benchmark_rounds,
        "benchmark_on_failure": args.benchmark_on_failure,
        "matmul_precision": args.matmul_precision,
        "allow_tf32": args.allow_tf32,
        "compile_baseline": args.compile_baseline,
        "compile_user": args.compile_user,
    }


def json_safe(value: Any) -> Any:
    """Convert non-finite diagnostics into strict JSON null values."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json_document(path: Optional[Path], result: Dict[str, object]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(json_safe(result), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def write_json_result(
    path: Optional[Path],
    config: TransformerConfig,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    accuracy: AccuracySummary,
    performance: Optional[Dict[str, object]],
) -> None:
    write_json_document(
        path,
        {
            "schema_version": 1,
            "config": asdict(config),
            "environment": environment_metadata(device, dtype),
            "settings": settings_metadata(args),
            "correctness_passed": accuracy.passed,
            "accuracy": asdict(accuracy),
            "performance": performance,
        },
    )


def configure_runtime(args: argparse.Namespace, device: torch.device) -> None:
    torch.set_float32_matmul_precision(args.matmul_precision)
    if device.type == "cuda":
        configure_triton_driver_path()
        torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.allow_tf32 = args.allow_tf32


def run_benchmark_case(
    config: TransformerConfig,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[AccuracySummary, Optional[Dict[str, object]]]:
    config.validate()

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    baseline = BaselineTransformer(config)
    optimized = UserOptimizedTransformer(config)
    copy_model_weights(
        baseline,
        optimized,
        strict=not args.non_strict_weight_copy,
    )

    baseline = baseline.to(device=device, dtype=dtype).eval()
    optimized = optimized.to(device=device, dtype=dtype).eval()

    # Compile only after model construction, weight copy, device transfer, and eval().
    baseline = maybe_compile(baseline, args.compile_baseline, args.compile_mode)
    optimized = maybe_compile(optimized, args.compile_user, args.compile_mode)

    print("=== Configuration ===")
    print(config)
    print(f"device={device}, dtype={dtype}, torch={torch.__version__}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")

    accuracy = run_accuracy_tests(
        baseline=baseline,
        optimized=optimized,
        config=config,
        device=device,
        dtype=dtype,
        trials=args.accuracy_trials,
        seed=args.seed,
        padding_ratio=args.padding_ratio,
        input_scale=args.input_scale,
        rtol=args.rtol,
        atol=args.atol,
    )

    if not accuracy.passed and not args.benchmark_on_failure:
        print("\nPerformance benchmark skipped because accuracy validation failed.")
        print(
            "Use --benchmark-on-failure to benchmark an incorrect "
            "implementation anyway."
        )
        return accuracy, None

    # Accuracy tensors are no longer needed. Release their cached CUDA blocks
    # before allocating the fixed timing input, outside the measured region.
    release_case_memory(device)

    performance = benchmark_models(
        baseline=baseline,
        optimized=optimized,
        config=config,
        device=device,
        dtype=dtype,
        seed=args.seed,
        padding_ratio=args.padding_ratio,
        input_scale=args.input_scale,
        warmup=args.warmup,
        repeats=args.repeats,
        rounds=args.benchmark_rounds,
    )
    return accuracy, performance


def config_from_args(args: argparse.Namespace) -> TransformerConfig:
    return TransformerConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        num_heads=args.heads,
        ffn_dim=args.ffn_dim,
        num_layers=args.layers,
        causal=args.causal,
    )


def official_case_result(
    case_id: int,
    config: TransformerConfig,
    accuracy: AccuracySummary,
    performance: Optional[Dict[str, object]],
) -> Dict[str, object]:
    return {
        "case_id": case_id,
        "config": asdict(config),
        "status": "succeeded" if accuracy.passed else "correctness_failed",
        "failure_category": None if accuracy.passed else "correctness",
        "correctness_passed": accuracy.passed,
        "accuracy": asdict(accuracy),
        "performance": performance,
    }


def execution_failed_result(
    case_id: int,
    config: TransformerConfig,
    failure_category: str,
    error: RuntimeError,
) -> Dict[str, object]:
    return {
        "case_id": case_id,
        "config": asdict(config),
        "status": "execution_failed",
        "failure_category": failure_category,
        "correctness_passed": None,
        "accuracy": None,
        "performance": None,
        "error": str(error),
    }


def write_matrix_result(
    path: Optional[Path],
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    requested_case_ids: List[int],
    case_results: List[Dict[str, object]],
) -> None:
    complete = len(case_results) == len(requested_case_ids)
    all_cases_executed = complete and all(
        result["status"] != "execution_failed" for result in case_results
    )
    if not all_cases_executed:
        correctness_passed: Optional[bool] = None
        failure_category: Optional[str] = "execution" if complete else None
    elif all(result["correctness_passed"] is True for result in case_results):
        correctness_passed = True
        failure_category = None
    else:
        correctness_passed = False
        failure_category = "correctness"
    write_json_document(
        path,
        {
            "schema_version": 2,
            "mode": "official_matrix",
            "case_set": "competition_appendix_3_7",
            "requested_case_ids": requested_case_ids,
            "complete": complete,
            "all_cases_executed": all_cases_executed,
            "environment": environment_metadata(device, dtype),
            "settings": settings_metadata(args),
            "failure_category": failure_category,
            "correctness_passed": correctness_passed,
            "cases": case_results,
        },
    )


def is_out_of_memory_error(error: RuntimeError) -> bool:
    return "out of memory" in str(error).lower()


def run_official_matrix(
    args: argparse.Namespace, device: torch.device, dtype: torch.dtype
) -> int:
    requested_case_ids = args.official_cases or list(
        range(1, len(OFFICIAL_TEST_CASES) + 1)
    )
    case_results: List[Dict[str, object]] = []

    for position, case_id in enumerate(requested_case_ids, start=1):
        config = OFFICIAL_TEST_CASES[case_id - 1]
        print(
            f"\n######## Official case {case_id} "
            f"({position}/{len(requested_case_ids)}) ########"
        )
        try:
            accuracy, performance = run_benchmark_case(config, args, device, dtype)
            result = official_case_result(case_id, config, accuracy, performance)
        except RuntimeError as error:
            if not is_out_of_memory_error(error):
                raise
            print(f"[out of memory] case {case_id}: {error}")
            result = execution_failed_result(case_id, config, "out_of_memory", error)

        case_results.append(result)
        release_case_memory(device)
        write_matrix_result(
            args.json_output,
            args,
            device,
            dtype,
            requested_case_ids,
            case_results,
        )

    if any(result["status"] == "execution_failed" for result in case_results):
        return 3
    if any(result["correctness_passed"] is not True for result in case_results):
        return 2
    return 0


def run_single_case(
    args: argparse.Namespace, device: torch.device, dtype: torch.dtype
) -> int:
    config = config_from_args(args)
    accuracy, performance = run_benchmark_case(config, args, device, dtype)
    write_json_result(
        args.json_output, config, args, device, dtype, accuracy, performance
    )
    return 0 if accuracy.passed else 2


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    validate_args(args, device, dtype)
    configure_runtime(args, device)

    if args.official_matrix:
        return run_official_matrix(args, device, dtype)
    return run_single_case(args, device, dtype)
