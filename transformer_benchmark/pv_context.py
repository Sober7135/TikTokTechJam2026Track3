"""Exact BF16 probability/value boundary for packed-value attention paths."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _bf16_probability_value_kernel(
    probability_ptr,
    value_ptr,
    context_ptr,
    probability_stride_batch: tl.constexpr,
    probability_stride_head: tl.constexpr,
    probability_stride_row: tl.constexpr,
    value_stride_batch: tl.constexpr,
    value_stride_head: tl.constexpr,
    value_stride_row: tl.constexpr,
    context_stride_batch: tl.constexpr,
    context_stride_head: tl.constexpr,
    context_stride_row: tl.constexpr,
    heads: tl.constexpr,
    row_start: tl.constexpr,
    row_count: tl.constexpr,
    key_count: tl.constexpr,
    head_dim: tl.constexpr,
):
    block_bh = tl.program_id(0)
    batch_index = block_bh // heads
    head_index = block_bh % heads

    rows = tl.arange(0, row_count)
    keys = tl.arange(0, key_count)
    columns = tl.arange(0, head_dim)
    probability_offsets = (
        batch_index * probability_stride_batch
        + head_index * probability_stride_head
        + rows[:, None] * probability_stride_row
        + keys[None, :]
    )
    value_offsets = (
        batch_index * value_stride_batch
        + head_index * value_stride_head
        + keys[:, None] * value_stride_row
        + columns[None, :]
    )

    # This conversion is the exact boundary being fused: native ATen softmax
    # produces FP32 probabilities, while the reference rounds them to BF16
    # before the probability/value matrix multiplication.
    probabilities = tl.load(probability_ptr + probability_offsets).to(tl.bfloat16)
    values = tl.load(value_ptr + value_offsets)
    accumulator = tl.dot(probabilities, values, out_dtype=tl.float32)
    context = accumulator.to(tl.bfloat16)

    context_offsets = (
        batch_index * context_stride_batch
        + head_index * context_stride_head
        + (row_start + rows[:, None]) * context_stride_row
        + columns[None, :]
    )
    tl.store(context_ptr + context_offsets, context)


def bf16_probability_value(
    probabilities: torch.Tensor,
    value: torch.Tensor,
    context: torch.Tensor,
    row_start: int,
) -> None:
    """Write ``BF16(probabilities) @ value`` into a packed-value context slice."""
    if probabilities.device.type != "cuda" or probabilities.dtype != torch.float32:
        raise ValueError("probabilities must be a CUDA FP32 tensor")
    if value.device != probabilities.device or value.dtype != torch.bfloat16:
        raise ValueError("value must be a CUDA BF16 tensor on the same device")
    if context.device != value.device or context.dtype != torch.bfloat16:
        raise ValueError("context must be a CUDA BF16 tensor on the same device")
    if probabilities.ndim != 4 or value.ndim != 4 or context.ndim != 4:
        raise ValueError("probabilities, value, and context must be rank-4 tensors")

    batch, heads, row_count, key_count = probabilities.shape
    expected_value_shape = (batch, heads, context.shape[-2], context.shape[-1])
    if tuple(value.shape) != expected_value_shape:
        raise ValueError("value and context batch/head/sequence dimensions must match")
    if tuple(context.shape) != tuple(value.shape):
        raise ValueError("context and value shapes must match")
    if (row_count, key_count, value.shape[-1]) not in {
        (64, 64, 32),
        (64, 128, 32),
        (64, 64, 64),
        (64, 128, 64),
    }:
        raise ValueError("PV kernel is specialized to case-6/10 prefix chunks")
    if row_start not in (0, 64) or row_start + row_count > context.shape[-2]:
        raise ValueError("row range is outside the case-6/10 context")
    if not probabilities.is_contiguous():
        raise ValueError("PV kernel requires contiguous probabilities")
    if value.stride(-1) != 1:
        raise ValueError("PV kernel requires unit-stride value columns")
    if context.stride(-1) != 1:
        raise ValueError("PV kernel requires unit-stride context columns")

    _bf16_probability_value_kernel[(batch * heads,)](
        probabilities,
        value,
        context,
        probability_stride_batch=probabilities.stride(0),
        probability_stride_head=probabilities.stride(1),
        probability_stride_row=probabilities.stride(2),
        value_stride_batch=value.stride(0),
        value_stride_head=value.stride(1),
        value_stride_row=value.stride(2),
        context_stride_batch=context.stride(0),
        context_stride_head=context.stride(1),
        context_stride_row=context.stride(2),
        heads=heads,
        row_start=row_start,
        row_count=row_count,
        key_count=key_count,
        head_dim=value.shape[-1],
        num_warps=4,
        num_stages=2,
    )
