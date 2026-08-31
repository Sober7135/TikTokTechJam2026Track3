import copy
import inspect
import unittest

import torch

from transformer_benchmark.cases import TransformerConfig
from transformer_benchmark.correctness import compare_outputs
from transformer_benchmark.models import (
    BaselineTransformerBlock,
    BaselineTransformer,
    UserOptimizedSelfAttention,
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

    def test_direct_qkv_dispatch_is_exact_shape_and_inference_only(self) -> None:
        attention = UserOptimizedTransformer(
            TransformerConfig(1, 8, 16, 4, 32, 1, True)
        ).layers[0].attention
        source = inspect.getsource(attention._project_qkv)
        direct_block = source.split("use_direct_layout =", 1)[1].split(
            "if use_direct_layout:", 1
        )[0]

        for declared_shape in (
            "(16, 128, 128, 4)",
            "(64, 32, 128, 4)",
            "(64, 128, 128, 1)",
            "(64, 128, 128, 16)",
            "(64, 1024, 128, 4)",
        ):
            self.assertIn(declared_shape, direct_block)
        for fallback_shape in (
            "(64, 128, 32, 4)",
            "(64, 128, 128, 2)",
            "(64, 128, 128, 4)",
            "(128, 128, 128, 4)",
        ):
            self.assertNotIn(fallback_shape, direct_block)
        self.assertIn("torch.is_inference_mode_enabled()", direct_block)
        self.assertIn("not torch.is_grad_enabled()", direct_block)
        self.assertIn('x.device.type == "cuda"', direct_block)
        self.assertIn("x.dtype == torch.bfloat16", direct_block)
        self.assertIn("x.is_contiguous()", direct_block)

        forward_source = inspect.getsource(attention.forward)
        self.assertIn(
            "direct_layout=causal and valid_token_mask is None",
            forward_source,
        )

    def test_direct_qkv_backing_exposes_three_contiguous_bhsd_views(self) -> None:
        output = torch.empty(3, 2, 4, 8, 4)
        query, key, value = output.unbind(dim=0)

        for projected in (query, key, value):
            self.assertEqual(tuple(projected.shape), (2, 4, 8, 4))
            self.assertEqual(tuple(projected.stride()), (128, 32, 4, 1))
            self.assertTrue(projected.is_contiguous())
        self.assertEqual(
            len(
                {
                    projected.untyped_storage().data_ptr()
                    for projected in (query, key, value)
                }
            ),
            1,
        )
        self.assertEqual(
            [projected.storage_offset() for projected in (query, key, value)],
            [0, 256, 512],
        )

    def test_direct_qkv_preserves_linear_weight_and_bf16_boundaries(self) -> None:
        from transformer_benchmark.direct_qkv import (
            _bf16_qkv_direct_layout_kernel,
            bf16_qkv_direct_layout,
        )

        kernel_source = inspect.getsource(_bf16_qkv_direct_layout_kernel.fn.fn)
        wrapper_source = inspect.getsource(bf16_qkv_direct_layout)
        self.assertIn(
            "output_columns[None, :] * width",
            kernel_source,
        )
        self.assertIn("tl.dot(activation, weight, out_dtype=tl.float32)", kernel_source)
        self.assertIn(".to(tl.bfloat16)", kernel_source)
        self.assertIn("(3, batch, num_heads, sequence_length, head_dimension)", wrapper_source)
        self.assertIn("output.unbind(dim=0)", wrapper_source)

    def test_case11_direct_qkv_uses_auditable_fixed_launch(self) -> None:
        from transformer_benchmark.direct_qkv import bf16_qkv_direct_layout

        wrapper_source = inspect.getsource(bf16_qkv_direct_layout)
        fixed_block = wrapper_source.split(
            "if (batch, sequence_length, width, num_heads) == "
            "(64, 128, 128, 16):",
            1,
        )[1].split("else:", 1)[0]

        self.assertIn("_bf16_qkv_direct_layout_kernel.fn[fixed_grid]", fixed_block)
        self.assertIn("block_rows=128", fixed_block)
        self.assertIn("block_columns=128", fixed_block)
        self.assertIn("block_reduction=32", fixed_block)
        self.assertIn("num_warps=8", fixed_block)
        self.assertIn("num_stages=3", fixed_block)

        fallback_block = wrapper_source.split("else:", 1)[1]
        self.assertIn("_bf16_qkv_direct_layout_kernel[grid]", fallback_block)

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

    def test_case6_attention_batches_preserve_existing_prefix_kernels(self) -> None:
        attention = UserOptimizedTransformer(
            TransformerConfig(10000, 128, 128, 4, 128, 4, True)
        ).layers[0].attention
        chunk_source = inspect.getsource(attention._chunked_triangular_context)

        self.assertIn(
            "use_case6_batch_chunks = tuple(query.shape) == (10000, 4, 128, 32)",
            chunk_source,
        )
        self.assertIn("batch_chunk_size = 512", chunk_source)
        self.assertIn(
            "consolidate_key_tile=use_case6_batch_chunks",
            chunk_source,
        )
        self.assertIn("value_batch", chunk_source)
        self.assertIn("context_batch", chunk_source)

        from transformer_benchmark.triangular_scores import (
            triangular_causal_score_chunk,
        )

        score_source = inspect.getsource(triangular_causal_score_chunk)
        self.assertIn(
            "use_consolidated_key_tile = consolidate_key_tile or",
            score_source,
        )

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
        self.assertIn("num_warps=2", wrapper_source)
        self.assertIn("num_stages=2", wrapper_source)

        forward_source = inspect.getsource(attention.forward)
        direct_write_block = forward_source.split("direct_context_write =", 1)[
            1
        ].split("context = self._chunked_triangular_context", 1)[0]
        self.assertIn("(64, 128, 128, 16)", direct_write_block)

    def test_case7_dispatches_independent_hd8_pv_into_final_layout(self) -> None:
        attention = UserOptimizedTransformer(
            TransformerConfig(64, 128, 32, 4, 32, 4, True)
        ).layers[0].attention
        chunk_source = inspect.getsource(attention._chunked_triangular_context)
        self.assertIn(
            "use_case7_hd8_pv_kernel = tuple(query.shape) == (64, 4, 128, 8)",
            chunk_source,
        )
        self.assertIn("use_hd8_pv_kernel", chunk_source)
        self.assertIn("use_case7_hd8_pv_kernel", chunk_source)
        self.assertIn("use_case13_pv_kernel", chunk_source)
        self.assertIn("bf16_probability_value_hd8_case7", chunk_source)

        from transformer_benchmark.pv_context import (
            _bf16_probability_value_hd8_case7_kernel,
            bf16_probability_value_hd8_case7,
        )

        wrapper_source = inspect.getsource(bf16_probability_value_hd8_case7)
        kernel_source = inspect.getsource(
            _bf16_probability_value_hd8_case7_kernel.fn
        )
        self.assertIn("(batch, heads, row_count) != (64, 4, 64)", wrapper_source)
        self.assertIn("key_count not in (64, 128)", wrapper_source)
        self.assertIn("tuple(value.shape) != (64, 4, 128, 8)", wrapper_source)
        self.assertIn("num_warps=4", wrapper_source)
        self.assertIn("num_stages=2", wrapper_source)
        self.assertIn(".to(tl.bfloat16)", kernel_source)
        self.assertIn("tl.dot(probabilities, values, out_dtype=tl.float32)", kernel_source)
        self.assertIn("mask=columns[None, :] < 8", kernel_source)

        sequence_major = torch.empty(64, 128, 4, 8)
        context = sequence_major.permute(0, 2, 1, 3)
        self.assertFalse(context.is_contiguous())
        self.assertTrue(context.transpose(1, 2).is_contiguous())
        self.assertEqual(
            tuple(context.transpose(1, 2).view(64, 128, 32).shape),
            (64, 128, 32),
        )

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
        self.assertIn("is_full_hd32_tile", wrapper_source)
        self.assertIn("block_row_count = 64", wrapper_source)
        self.assertIn("num_warps = 2", wrapper_source)
        self.assertIn(
            "triton.cdiv(row_count, block_row_count)", wrapper_source
        )
        self.assertIn("num_warps = 8 if row_count == 128 else 4", wrapper_source)

    def test_cases6_and11_use_one_adaptive_score_key_tile_per_prefix(self) -> None:
        from transformer_benchmark.triangular_scores import (
            triangular_causal_score_chunk,
        )

        source = inspect.getsource(triangular_causal_score_chunk)
        self.assertIn("(10000, 4, 128, 32)", source)
        self.assertIn("(64, 16, 128, 8)", source)
        self.assertIn("if row_stop <= 16", source)
        self.assertIn("block_key_size = 16", source)
        self.assertIn("elif row_stop <= 32", source)
        self.assertIn("block_key_size = 32", source)
        self.assertIn("elif row_stop <= 64", source)
        self.assertIn("block_key_size = 64", source)
        self.assertIn("block_key_size = 128", source)
        self.assertIn("block_key_size = block_query_size", source)
        self.assertIn("triton.cdiv(row_stop, block_key_size)", source)
        self.assertIn("block_query_size=block_query_size", source)
        self.assertIn("block_key_size=block_key_size", source)
        self.assertIn(
            "num_warps=4 if block_query_size <= 32 else 8",
            source,
        )

    def test_cases6_and13_reduce_masked_future_chunk_geometry(self) -> None:
        attention = UserOptimizedTransformer(
            TransformerConfig(1, 8, 16, 4, 32, 1, True)
        ).layers[0].attention
        source = inspect.getsource(attention.forward)
        chunk_block = source.split(
            "if use_chunked_triangular_attention:", 1
        )[1].split("direct_context_write =", 1)[0]

        self.assertIn("(batch, seq_len, self.d_model, self.num_heads)", chunk_block)
        self.assertIn("64,\n                1024,\n                128,\n                4", chunk_block)
        self.assertIn("10000,\n                128,\n                128,\n                4", chunk_block)
        self.assertIn("chunk_size = 256", chunk_block)
        self.assertIn("chunk_size = 32", chunk_block)
        self.assertNotIn("chunk_size = 128", chunk_block)

    def test_cases4_and9_use_prefix_attention_with_exact_boundaries(self) -> None:
        attention = UserOptimizedTransformer(
            TransformerConfig(1, 8, 16, 4, 32, 1, True)
        ).layers[0].attention
        forward_source = inspect.getsource(attention.forward)
        chunk_gate = forward_source.split(
            "use_chunked_triangular_attention =", 1
        )[1].split("if use_chunked_triangular_attention:", 1)[0]
        direct_gate = forward_source.split("direct_context_write =", 1)[1].split(
            "context = self._chunked_triangular_context", 1
        )[0]
        chunk_source = inspect.getsource(attention._chunked_triangular_context)

        self.assertIn("(16, 128, 128, 4)", chunk_gate)
        self.assertIn("(64, 128, 128, 1)", chunk_gate)
        self.assertIn("(16, 128, 128, 4)", direct_gate)
        self.assertIn("(64, 128, 128, 1)", direct_gate)
        self.assertIn("(16, 4, 128, 32)", chunk_source)

        from transformer_benchmark.triangular_scores import (
            triangular_causal_score_chunk,
        )

        score_source = inspect.getsource(triangular_causal_score_chunk)
        self.assertIn("head_dim not in (8, 32, 64, 128)", score_source)
        self.assertIn("block_query_size = row_count", score_source)
        self.assertIn("block_key_size = block_query_size", score_source)

    def test_case13_dispatches_tiled_k_direct_pv(self) -> None:
        attention = UserOptimizedTransformer(
            TransformerConfig(1, 8, 16, 4, 32, 1, True)
        ).layers[0].attention
        chunk_source = inspect.getsource(attention._chunked_triangular_context)
        self.assertIn(
            "use_case13_pv_kernel = tuple(query.shape) == (64, 4, 1024, 32)",
            chunk_source,
        )
        self.assertIn("bf16_probability_value_case13", chunk_source)
        self.assertIn("context_sequence_major.permute(0, 2, 1, 3)", chunk_source)

        from transformer_benchmark.pv_context import (
            _bf16_probability_value_case13_kernel,
            bf16_probability_value_case13,
        )

        wrapper_source = inspect.getsource(bf16_probability_value_case13)
        kernel_source = inspect.getsource(_bf16_probability_value_case13_kernel.fn)
        self.assertIn("(batch, heads, row_count) != (64, 4, 256)", wrapper_source)
        self.assertIn("key_count not in (256, 512, 768, 1024)", wrapper_source)
        self.assertIn("tuple(value.shape) != (64, 4, 1024, 32)", wrapper_source)
        self.assertIn("range(0, block_key_count, 128)", kernel_source)
        self.assertIn("accumulator += tl.dot", kernel_source)
        self.assertIn(".to(tl.bfloat16)", kernel_source)
        self.assertIn("num_warps=4", wrapper_source)

    def test_case13_keeps_bf16_scores_until_native_softmax_load(self) -> None:
        attention = UserOptimizedTransformer(
            TransformerConfig(1, 8, 16, 4, 32, 1, True)
        ).layers[0].attention
        chunk_source = inspect.getsource(attention._chunked_triangular_context)

        self.assertIn(
            "use_case13_bf16_score_transport = use_case13_pv_kernel",
            chunk_source,
        )
        self.assertIn(
            "output_float32=not use_case13_bf16_score_transport",
            chunk_source,
        )
        self.assertIn(
            "output_float16=use_case13_bf16_score_transport",
            chunk_source,
        )
        self.assertIn(
            "torch._softmax(prefix_scores, -1, True)",
            chunk_source,
        )
        self.assertIn(
            "prefix_probs_float32 = torch.softmax(prefix_scores, dim=-1)",
            chunk_source,
        )

        from transformer_benchmark.triangular_scores import (
            _triangular_scores_kernel,
            triangular_causal_score_chunk,
        )

        score_wrapper_source = inspect.getsource(triangular_causal_score_chunk)
        score_kernel_source = inspect.getsource(_triangular_scores_kernel.fn)
        self.assertIn("output_float16: bool = False", score_wrapper_source)
        self.assertIn("torch.float16 if output_float16", score_wrapper_source)
        self.assertIn("output_float16=output_float16", score_wrapper_source)
        self.assertIn("scores = scores.to(tl.float16)", score_kernel_source)

    def test_exact_gelu_fusion_includes_declared_d128_ffn128_shapes(self) -> None:
        block_source = inspect.getsource(UserOptimizedTransformerBlock.forward)
        fused_gate = block_source.split("use_fused_ffn_in =", 1)[1].split(
            "if use_fused_ffn_in:", 1
        )[0]

        self.assertIn("(16, 128, 128)", fused_gate)
        self.assertIn("(64, 32, 128)", fused_gate)
        self.assertIn("(64, 1024, 128)", fused_gate)
        self.assertIn("(10000, 128, 128)", fused_gate)
        self.assertIn("x.device.type == \"cuda\"", fused_gate)
        self.assertIn("x.dtype == torch.bfloat16", fused_gate)
        self.assertIn("valid_token_mask is None", fused_gate)
        self.assertNotIn("x.numel()", fused_gate)

    def test_ffn_out_residual_fusion_includes_large_d128_shapes(self) -> None:
        block_source = inspect.getsource(UserOptimizedTransformerBlock.forward)
        fused_gate = block_source.split("use_fused_ffn_out =", 1)[1].split(
            "if use_fused_ffn_out:", 1
        )[0]

        self.assertIn("(64, 1024, 128)", fused_gate)
        self.assertIn("(10000, 128, 128)", fused_gate)
        self.assertIn("valid_token_mask is None", fused_gate)
        self.assertIn("hidden.is_contiguous()", fused_gate)
        self.assertIn("x.is_contiguous()", fused_gate)

    def test_attention_out_residual_fusion_includes_large_d128_shapes(self) -> None:
        attention_source = inspect.getsource(
            UserOptimizedSelfAttention._project_output_with_residual
        )
        fused_gate = attention_source.split("use_fused_attention_out =", 1)[
            1
        ].split("if use_fused_attention_out:", 1)[0]

        self.assertIn("(64, 1024, 128)", fused_gate)
        self.assertIn("(10000, 128, 128)", fused_gate)
        self.assertIn("context.is_contiguous()", fused_gate)
        self.assertIn("residual.is_contiguous()", fused_gate)

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
