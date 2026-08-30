import copy
import inspect
import unittest

import torch

from transformer_benchmark.cases import TransformerConfig
from transformer_benchmark.correctness import compare_outputs
from transformer_benchmark.models import (
    BaselineTransformerBlock,
    BaselineTransformer,
    UserOptimizedTransformerBlock,
    UserOptimizedTransformer,
    copy_model_weights,
)


class UserOptimizedTransformerTests(unittest.TestCase):
    def test_candidate_dispatch_does_not_modify_baseline_blocks(self) -> None:
        config = TransformerConfig(1, 8, 16, 4, 32, 1, True)
        baseline = BaselineTransformer(config)
        optimized = UserOptimizedTransformer(config)

        self.assertIs(type(baseline.layers[0]), BaselineTransformerBlock)
        self.assertNotIn("_candidate_cublaslt_linear", vars(baseline.layers[0]))
        self.assertIs(type(optimized.layers[0]), UserOptimizedTransformerBlock)

    def test_case2_cpu_fallback_preserves_strict_contract(self) -> None:
        config = TransformerConfig(1, 128, 128, 4, 128, 4, True)
        torch.manual_seed(1234)
        baseline = BaselineTransformer(config).eval()
        optimized = UserOptimizedTransformer(config).eval()
        copy_model_weights(baseline, optimized)

        x = torch.randn(1, 128, 128)
        valid_token_mask = torch.ones(1, 128, dtype=torch.bool)
        valid_token_mask[:, -32:] = False
        x = x.masked_fill(~valid_token_mask[..., None], 0)
        with torch.inference_mode():
            reference = baseline(x, valid_token_mask)
            candidate = optimized(x, valid_token_mask)

        result = compare_outputs(reference, candidate, rtol=0.02, atol=0.002)
        self.assertTrue(result.passed)
        self.assertIsNone(optimized._cuda_graph)
        for layer in optimized.layers:
            self.assertEqual(
                tuple(layer.attention._cached_causal_mask.shape), (128, 128)
            )

    def test_attention_residual_cpu_fallback_preserves_operation_order(self) -> None:
        config = TransformerConfig(4, 8, 16, 4, 32, 1, True)
        torch.manual_seed(4321)
        baseline = BaselineTransformer(config).eval()
        optimized = UserOptimizedTransformer(config).eval()
        copy_model_weights(baseline, optimized)

        x = torch.randn(4, 8, 16)
        with torch.inference_mode():
            reference = baseline(x, None)
            candidate = optimized(x, None)

        result = compare_outputs(reference, candidate, rtol=0.02, atol=0.002)
        self.assertTrue(result.passed)
        self.assertTrue(torch.equal(reference, candidate))

    def test_legacy_qkv_weights_are_packed_in_order(self) -> None:
        config = TransformerConfig(1, 8, 16, 4, 32, 1, True)
        baseline = BaselineTransformer(config).eval()
        optimized = UserOptimizedTransformer(config).eval()
        copy_model_weights(baseline, optimized)

        source = baseline.layers[0].attention
        packed = optimized.layers[0].attention.qkv_proj
        width = config.d_model
        self.assertTrue(torch.equal(packed.weight[:width], source.q_proj.weight))
        self.assertTrue(
            torch.equal(packed.weight[width : 2 * width], source.k_proj.weight)
        )
        self.assertTrue(torch.equal(packed.weight[2 * width :], source.v_proj.weight))
        self.assertTrue(torch.equal(packed.bias[:width], source.q_proj.bias))
        self.assertTrue(
            torch.equal(packed.bias[width : 2 * width], source.k_proj.bias)
        )
        self.assertTrue(torch.equal(packed.bias[2 * width :], source.v_proj.bias))

    def test_packed_state_dict_round_trip(self) -> None:
        config = TransformerConfig(1, 8, 16, 4, 32, 1, True)
        baseline = BaselineTransformer(config).eval()
        optimized = UserOptimizedTransformer(config).eval()
        copy_model_weights(baseline, optimized)
        saved = copy.deepcopy(optimized.state_dict())

        reloaded = UserOptimizedTransformer(config).eval()
        reloaded.load_state_dict(saved, strict=True)
        actual = reloaded.state_dict()
        self.assertEqual(saved.keys(), actual.keys())
        for key in saved:
            self.assertTrue(torch.equal(saved[key], actual[key]), key)

    def test_cases6_and10_keep_packed_value_views(self) -> None:
        source = inspect.getsource(
            UserOptimizedTransformer(TransformerConfig(1, 8, 16, 4, 32, 1, True))
            .layers[0]
            .attention._project_qkv
        )
        all_view_block, qk_only_block = source.split("use_packed_qk_views", 1)
        self.assertIn("(10000, 128, 128, 4)", all_view_block)
        self.assertIn("(64, 128, 128, 2)", all_view_block)
        self.assertNotIn("(10000, 128, 128, 4)", qk_only_block)
        self.assertNotIn("(64, 128, 128, 2)", qk_only_block)

    def test_packed_value_kernel_accepts_strided_value_layout(self) -> None:
        from transformer_benchmark.pv_context import bf16_probability_value

        source = inspect.getsource(bf16_probability_value)
        self.assertNotIn("value.is_contiguous", source)
        self.assertIn("value.stride(-1) != 1", source)
        self.assertIn("(64, 128, 64)", source)

    def test_cases1_and5_use_direct_pv_prefix_tiles(self) -> None:
        from transformer_benchmark.pv_context import (
            _bf16_probability_value_kernel,
            bf16_probability_value,
        )

        config = TransformerConfig(64, 128, 128, 4, 128, 4, True)
        attention = UserOptimizedTransformer(config).layers[0].attention
        chunk_source = inspect.getsource(attention._chunked_triangular_context)
        forward_source = inspect.getsource(attention.forward)
        wrapper_source = inspect.getsource(bf16_probability_value)
        kernel_source = inspect.getsource(_bf16_probability_value_kernel.fn)

        self.assertIn("(64, 4, 128, 32)", chunk_source)
        self.assertIn("(128, 4, 128, 32)", chunk_source)
        self.assertIn("(64, 128, 128, 4)", forward_source)
        self.assertIn("(128, 128, 128, 4)", forward_source)
        self.assertIn("(32, 96, 32)", wrapper_source)
        self.assertIn("key_count != row_start + row_count", wrapper_source)
        self.assertIn("block_key_count", kernel_source)
        self.assertIn("mask=valid_keys", kernel_source)

    def test_case11_dispatches_hd8_pv_into_sequence_major_backing(self) -> None:
        attention = UserOptimizedTransformer(
            TransformerConfig(1, 8, 16, 4, 32, 1, True)
        ).layers[0].attention
        chunk_source = inspect.getsource(attention._chunked_triangular_context)
        self.assertIn(
            "use_hd8_pv_kernel = tuple(query.shape) == (64, 16, 128, 8)",
            chunk_source,
        )
        self.assertIn("context_sequence_major.permute(0, 2, 1, 3)", chunk_source)
        self.assertIn("bf16_probability_value_hd8", chunk_source)

        from transformer_benchmark.pv_context import bf16_probability_value_hd8

        wrapper_source = inspect.getsource(bf16_probability_value_hd8)
        self.assertIn("(batch, heads, row_count) != (64, 16, 16)", wrapper_source)
        self.assertIn("key_count not in range(16, 129, 16)", wrapper_source)
        self.assertIn("tuple(value.shape) != (64, 16, 128, 8)", wrapper_source)

        forward_source = inspect.getsource(attention.forward)
        direct_write_block = forward_source.split("direct_context_write =", 1)[
            1
        ].split("context = self._chunked_triangular_context", 1)[0]
        self.assertIn("(64, 128, 128, 16)", direct_write_block)

    def test_cases4_and12_dispatch_full_hd32_pv_to_final_layout(self) -> None:
        attention = UserOptimizedTransformer(
            TransformerConfig(1, 8, 16, 4, 32, 1, True)
        ).layers[0].attention
        forward_source = inspect.getsource(attention.forward)
        full_pv_block = forward_source.split(
            "use_full_hd32_pv_kernel =", 1
        )[1].split("else:", 1)[0]

        self.assertIn("(16, 4, 128, 32)", full_pv_block)
        self.assertIn("(64, 4, 32, 32)", full_pv_block)
        self.assertIn("context_sequence_major.permute(0, 2, 1, 3)", forward_source)
        self.assertIn(
            "bf16_probability_value(probs_float32, v, context, 0)",
            forward_source,
        )

        from transformer_benchmark.pv_context import bf16_probability_value

        wrapper_source = inspect.getsource(bf16_probability_value)
        self.assertIn("(128, 128, 32)", wrapper_source)
        self.assertIn("(32, 32, 32)", wrapper_source)
        self.assertIn("num_warps=8 if row_count == 128 else 4", wrapper_source)

    def test_cases4_and12_extend_exact_gelu_fusion_by_exact_shape(self) -> None:
        block_source = inspect.getsource(UserOptimizedTransformerBlock.forward)
        fused_gate = block_source.split("use_fused_ffn_in =", 1)[1].split(
            "if use_fused_ffn_in:", 1
        )[0]

        self.assertIn("(16, 128, 128)", fused_gate)
        self.assertIn("(64, 32, 128)", fused_gate)
        self.assertIn("x.device.type == \"cuda\"", fused_gate)
        self.assertIn("x.dtype == torch.bfloat16", fused_gate)
        self.assertIn("valid_token_mask is None", fused_gate)
        self.assertNotIn("x.numel()", fused_gate)

    def test_full_hd32_pv_sequence_major_layout_is_transpose_contiguous(self) -> None:
        for batch, seq_len in ((16, 128), (64, 32)):
            backing = torch.empty(batch, seq_len, 4, 32)
            context = backing.permute(0, 2, 1, 3)
            self.assertFalse(context.is_contiguous())
            self.assertTrue(context.transpose(1, 2).is_contiguous())
            self.assertEqual(
                tuple(context.transpose(1, 2).view(batch, seq_len, 128).shape),
                (batch, seq_len, 128),
            )

    def test_mask_classification_tracks_mutation_and_inference_tensors(self) -> None:
        config = TransformerConfig(1, 128, 128, 4, 128, 4, True)
        optimized = UserOptimizedTransformer(config).eval()

        mask = torch.ones(1, 128, dtype=torch.bool)
        self.assertTrue(optimized._mask_is_all_true(mask))
        mask[0, -1] = False
        self.assertFalse(optimized._mask_is_all_true(mask))

        with torch.inference_mode():
            inference_mask = torch.ones(1, 128, dtype=torch.bool)
            self.assertTrue(optimized._mask_is_all_true(inference_mask))
            inference_mask[0, -1] = False
            self.assertFalse(optimized._mask_is_all_true(inference_mask))

    def test_cuda_graph_replay_returns_independent_outputs(self) -> None:
        config = TransformerConfig(1, 8, 16, 4, 32, 1, True)
        optimized = UserOptimizedTransformer(config).eval()
        static_output = torch.zeros(1)

        class FakeGraph:
            def replay(self) -> None:
                static_output.add_(1)

        optimized._cuda_graph = FakeGraph()
        optimized._cuda_graph_output = static_output
        optimized._cuda_graph_signature = ("stable",)
        optimized._cuda_graph_eligible = lambda _x, _mask: True
        optimized._mask_is_all_true = lambda _mask: True
        optimized._cuda_graph_live_signature = lambda _x, _mask, _all_true: (
            "stable",
        )

        x = torch.zeros(1, 8, 16)
        valid_token_mask = torch.ones(1, 8, dtype=torch.bool)
        first = optimized(x, valid_token_mask)
        second = optimized(x, valid_token_mask)

        self.assertIsNot(first, second)
        self.assertEqual(first.item(), 1.0)
        self.assertEqual(second.item(), 2.0)


if __name__ == "__main__":
    unittest.main()
