#!/usr/bin/env python3
"""
Compare numerical accuracy and inference latency between a baseline Transformer
and a user-optimized implementation.

Correctness rule for every finite output element:
    abs(user - ref) < atol
    OR
    abs(user - ref) < rtol * abs(ref)

The default thresholds are atol=0.002 and rtol=0.02 (2%).
The CLI runs the declared official shape matrix by default. Use
--no-official-matrix for one custom shape. The implementation lives in the
``transformer_benchmark`` package; this module remains the stable CLI entry
point and import-compatible facade.
"""

from transformer_benchmark.cases import (
    OFFICIAL_TEST_CASES,
    TransformerConfig,
    generate_random_case,
)
from transformer_benchmark.correctness import (
    AccuracyResult,
    AccuracySummary,
    compare_outputs,
    run_accuracy_tests,
)
from transformer_benchmark.models import (
    BaselineSelfAttention,
    BaselineTransformer,
    BaselineTransformerBlock,
    UserOptimizedTransformer,
    copy_model_weights,
)
from transformer_benchmark.runner import (
    config_from_args,
    configure_runtime,
    environment_metadata,
    execution_failed_result,
    is_out_of_memory_error,
    main,
    maybe_compile,
    official_case_result,
    parse_args,
    release_case_memory,
    resolve_device,
    resolve_dtype,
    run_benchmark_case,
    run_official_matrix,
    run_single_case,
    settings_metadata,
    validate_args,
    write_json_document,
    write_json_result,
    write_matrix_result,
)
from transformer_benchmark.timing import (
    TimingResult,
    benchmark_models,
    benchmark_once,
    percentile,
    warmup_model,
)

__all__ = [
    "AccuracyResult",
    "AccuracySummary",
    "BaselineSelfAttention",
    "BaselineTransformer",
    "BaselineTransformerBlock",
    "OFFICIAL_TEST_CASES",
    "TimingResult",
    "TransformerConfig",
    "UserOptimizedTransformer",
    "benchmark_models",
    "benchmark_once",
    "compare_outputs",
    "config_from_args",
    "configure_runtime",
    "copy_model_weights",
    "environment_metadata",
    "execution_failed_result",
    "generate_random_case",
    "is_out_of_memory_error",
    "main",
    "maybe_compile",
    "official_case_result",
    "parse_args",
    "percentile",
    "release_case_memory",
    "resolve_device",
    "resolve_dtype",
    "run_accuracy_tests",
    "run_benchmark_case",
    "run_official_matrix",
    "run_single_case",
    "settings_metadata",
    "validate_args",
    "warmup_model",
    "write_json_document",
    "write_json_result",
    "write_matrix_result",
]


if __name__ == "__main__":
    raise SystemExit(main())
