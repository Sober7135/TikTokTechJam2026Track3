from __future__ import annotations

import importlib.util
import inspect
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F

from transformer_benchmark.case6_exact_layer_norm import (
    _PINNED_CUDA_VERSION,
    _PINNED_DEVICE_CAPABILITY,
    _PINNED_TORCH_GIT_VERSION,
    _PINNED_TORCH_VERSION,
    case6_exact_layer_norm,
    case6_exact_layer_norm_eligible,
)
from transformer_benchmark.models import (
    TransformerConfig,
    UserOptimizedTransformer,
)


ORACLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "research"
    / "case6_layer_norm_oracle.py"
)
SPEC = importlib.util.spec_from_file_location("case6_layer_norm_oracle", ORACLE_PATH)
assert SPEC is not None and SPEC.loader is not None
ORACLE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ORACLE
SPEC.loader.exec_module(ORACLE)


class Case6LayerNormOracleTest(unittest.TestCase):
    def test_four_row_coordinate_map_preserves_native_row_order(self) -> None:
        report = ORACLE.verify_coordinate_map()
        self.assertEqual(report["coordinates_per_row"], 128)
        self.assertEqual(report["native_data_warps_per_row"], 1)
        self.assertEqual(report["native_empty_warps_per_row"], 3)
        self.assertEqual(report["specialized_rows_per_block"], 4)
        self.assertIs(report["rows_are_disjoint"], True)

    def test_native_empty_warp_tree_is_bitwise_identity(self) -> None:
        report = ORACLE.verify_welford_schedule()
        self.assertEqual(report["online_updates_per_lane"], 4)
        self.assertEqual(report["shuffle_offsets"], [16, 8, 4, 2, 1])
        self.assertEqual(report["native_interwarp_offsets"], [2, 1])
        self.assertEqual(len(report["state_bits"]), 4)
        self.assertIs(report["bitwise_equal"], True)

    def test_cuda_source_retains_pinned_expression_order(self) -> None:
        report = ORACLE.verify_source_guards()
        self.assertEqual(report["pinned_git_version"], _PINNED_TORCH_GIT_VERSION)
        self.assertGreaterEqual(report["ordered_fragments"], 15)
        self.assertIs(report["no_block_barrier"], True)
        self.assertIs(report["no_fast_math"], True)

    def test_runtime_guard_is_pinned_to_wheel_cuda_and_sm89(self) -> None:
        self.assertEqual(_PINNED_TORCH_VERSION, "2.13.0+cu130")
        self.assertEqual(
            _PINNED_TORCH_GIT_VERSION,
            "cf30153c4c131c8164ee7798e5022d810682e2cb",
        )
        self.assertEqual(_PINNED_CUDA_VERSION, "13.0")
        self.assertEqual(_PINNED_DEVICE_CAPABILITY, (8, 9))
        source = inspect.getsource(case6_exact_layer_norm_eligible)
        self.assertEqual(source.count("data_ptr() % 8 == 0"), 3)

    def test_cpu_and_non_case6_paths_use_unchanged_native_fallback(self) -> None:
        input_tensor = torch.randn(2, 8, 16, dtype=torch.bfloat16)
        weight = torch.randn(16, dtype=torch.bfloat16)
        bias = torch.randn(16, dtype=torch.bfloat16)
        expected = F.layer_norm(input_tensor, (16,), weight, bias, 1e-5)
        with mock.patch(
            "transformer_benchmark.case6_exact_layer_norm._load_extension"
        ) as load_extension:
            actual = case6_exact_layer_norm(
                input_tensor,
                weight,
                bias,
                1e-5,
            )
        load_extension.assert_not_called()
        self.assertTrue(torch.equal(actual, expected))
        self.assertFalse(
            case6_exact_layer_norm_eligible(input_tensor, weight, bias)
        )

    def test_extension_load_failure_uses_native_fallback(self) -> None:
        input_tensor = torch.empty(1, dtype=torch.bfloat16)
        with mock.patch(
            "transformer_benchmark.case6_exact_layer_norm."
            "case6_exact_layer_norm_eligible",
            return_value=True,
        ), mock.patch(
            "transformer_benchmark.case6_exact_layer_norm._load_extension",
            side_effect=RuntimeError("synthetic build failure"),
        ), mock.patch(
            "transformer_benchmark.case6_exact_layer_norm.F.layer_norm",
            return_value=input_tensor,
        ) as native_layer_norm:
            actual = case6_exact_layer_norm(
                input_tensor,
                input_tensor,
                input_tensor,
                1e-5,
            )
        self.assertIs(actual, input_tensor)
        native_layer_norm.assert_called_once_with(
            input_tensor,
            (128,),
            input_tensor,
            input_tensor,
            1e-5,
        )

    def test_nonpositive_epsilon_uses_native_fallback(self) -> None:
        input_tensor = torch.empty(1, dtype=torch.bfloat16)
        with mock.patch(
            "transformer_benchmark.case6_exact_layer_norm."
            "case6_exact_layer_norm_eligible",
            return_value=True,
        ) as eligible, mock.patch(
            "transformer_benchmark.case6_exact_layer_norm._load_extension"
        ) as load_extension, mock.patch(
            "transformer_benchmark.case6_exact_layer_norm.F.layer_norm",
            return_value=input_tensor,
        ) as native_layer_norm:
            actual = case6_exact_layer_norm(
                input_tensor,
                input_tensor,
                input_tensor,
                0.0,
            )
        self.assertIs(actual, input_tensor)
        eligible.assert_not_called()
        load_extension.assert_not_called()
        native_layer_norm.assert_called_once_with(
            input_tensor,
            (1,),
            input_tensor,
            input_tensor,
            0.0,
        )

    def test_model_dispatch_is_exact_case6_only_with_native_else_path(self) -> None:
        source = inspect.getsource(
            UserOptimizedTransformer._case6_exact_layer_norm_eligible
        )
        self.assertIn("(10000, 128, 128, 4, 128, 4, True)", source)
        self.assertIn("valid_token_mask is not None", source)
        self.assertIn("torch.is_inference_mode_enabled()", source)
        self.assertIn("_PINNED_DEVICE_CAPABILITY", source)
        eager_source = inspect.getsource(UserOptimizedTransformer._forward_eager)
        self.assertIn("self.final_norm(x)", eager_source)

        model = UserOptimizedTransformer(
            TransformerConfig(
                batch_size=1,
                seq_len=8,
                d_model=16,
                num_heads=4,
                ffn_dim=32,
                num_layers=1,
                causal=True,
            )
        ).eval()
        x = torch.randn(1, 8, 16)
        with torch.inference_mode():
            self.assertFalse(model._case6_exact_layer_norm_eligible(x, None))


if __name__ == "__main__":
    unittest.main()
