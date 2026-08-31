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
    block_row_count: tl.constexpr,
    key_count: tl.constexpr,
    block_key_count: tl.constexpr,
    head_dim: tl.constexpr,
):
    block_bh = tl.program_id(0)
    batch_index = block_bh // heads
    head_index = block_bh % heads

    block_row = tl.program_id(1)
    rows = block_row * block_row_count + tl.arange(0, block_row_count)
    keys = tl.arange(0, block_key_count)
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
    valid_keys = keys < key_count
    probabilities = tl.load(
        probability_ptr + probability_offsets,
        mask=valid_keys[None, :],
        other=0.0,
    ).to(tl.bfloat16)
    values = tl.load(
        value_ptr + value_offsets,
        mask=valid_keys[:, None],
        other=0.0,
    )
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
    prefix_tile = (row_count, key_count, value.shape[-1])
    if prefix_tile not in {
        (32, 32, 32),
        (32, 64, 32),
        (32, 96, 32),
        (32, 128, 32),
        (64, 64, 32),
        (64, 128, 32),
        (64, 64, 64),
        (64, 128, 64),
        (128, 128, 32),
    }:
        raise ValueError("PV kernel is specialized to declared PV tiles")
    allowed_row_starts = (0, 32, 64, 96) if row_count == 32 else (0, 64)
    if (
        row_start not in allowed_row_starts
        or key_count != row_start + row_count
        or row_start + row_count > context.shape[-2]
    ):
        raise ValueError("row/key range is outside the declared PV context")
    if not probabilities.is_contiguous():
        raise ValueError("PV kernel requires contiguous probabilities")
    if value.stride(-1) != 1:
        raise ValueError("PV kernel requires unit-stride value columns")
    if context.stride(-1) != 1:
        raise ValueError("PV kernel requires unit-stride context columns")

    is_full_hd32_tile = (
        row_start == 0
        and row_count == key_count == context.shape[-2]
        and value.shape[-1] == 32
    )
    if is_full_hd32_tile and row_count == 128:
        # Case 4 has only 64 batch/head tiles. Split its independent output
        # rows into two proven 64-row PV programs so Ada can schedule 128
        # smaller four-warp blocks while each block still performs the full
        # K=128 reduction in one tl.dot.
        block_row_count = 64
        num_warps = 4
    elif is_full_hd32_tile and row_count == 32:
        # Case 12 already exposes 256 independent batch/head tiles. Its
        # 32x32 accumulator needs only two warps; prefix row-32 calls for
        # cases 1/5 retain the historical four-warp launch below.
        block_row_count = 32
        num_warps = 2
    else:
        block_row_count = row_count
        num_warps = 8 if row_count == 128 else 4

    _bf16_probability_value_kernel[
        (batch * heads, triton.cdiv(row_count, block_row_count))
    ](
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
        block_row_count=block_row_count,
        key_count=key_count,
        block_key_count=triton.next_power_of_2(key_count),
        head_dim=value.shape[-1],
        num_warps=num_warps,
        num_stages=2,
    )


@triton.jit
def _bf16_probability_value_case13_kernel(
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
    key_count: tl.constexpr,
    block_key_count: tl.constexpr,
):
    block_bh = tl.program_id(0)
    batch_index = block_bh // heads
    head_index = block_bh % heads
    rows = tl.program_id(1) * 64 + tl.arange(0, 64)
    columns = tl.arange(0, 32)
    reduction_offsets = tl.arange(0, 128)
    accumulator = tl.zeros((64, 32), dtype=tl.float32)

    # Keep shared-memory use bounded by reducing K in 128-wide tensor-core
    # tiles. Probability rounding still occurs before every dot, while the
    # partial dot results accumulate in FP32 in increasing key order.
    for key_start in range(0, block_key_count, 128):
        keys = key_start + reduction_offsets
        probabilities = tl.load(
            probability_ptr
            + batch_index * probability_stride_batch
            + head_index * probability_stride_head
            + rows[:, None] * probability_stride_row
            + keys[None, :],
            mask=keys[None, :] < key_count,
            other=0.0,
        ).to(tl.bfloat16)
        values = tl.load(
            value_ptr
            + batch_index * value_stride_batch
            + head_index * value_stride_head
            + keys[:, None] * value_stride_row
            + columns[None, :],
            mask=keys[:, None] < key_count,
            other=0.0,
        )
        accumulator += tl.dot(probabilities, values, out_dtype=tl.float32)

    context_offsets = (
        batch_index * context_stride_batch
        + head_index * context_stride_head
        + (row_start + rows[:, None]) * context_stride_row
        + columns[None, :]
    )
    tl.store(context_ptr + context_offsets, accumulator.to(tl.bfloat16))


def bf16_probability_value_case13(
    probabilities: torch.Tensor,
    value: torch.Tensor,
    context: torch.Tensor,
    row_start: int,
) -> None:
    """Write one Case13 BF16 PV prefix directly into final context backing."""
    if probabilities.device.type != "cuda" or probabilities.dtype != torch.float32:
        raise ValueError("case13 probabilities must be a CUDA FP32 tensor")
    if value.device != probabilities.device or value.dtype != torch.bfloat16:
        raise ValueError("case13 value must be CUDA BF16 on the same device")
    if context.device != value.device or context.dtype != torch.bfloat16:
        raise ValueError("case13 context must be CUDA BF16 on the same device")
    if probabilities.ndim != 4 or value.ndim != 4 or context.ndim != 4:
        raise ValueError("case13 PV tensors must be rank 4")

    batch, heads, row_count, key_count = probabilities.shape
    if (batch, heads, row_count) != (64, 4, 256):
        raise ValueError("case13 PV requires a 64x4x256 probability prefix")
    if key_count not in (256, 512, 768, 1024):
        raise ValueError("case13 PV key count must be 256, 512, 768, or 1024")
    if tuple(value.shape) != (64, 4, 1024, 32):
        raise ValueError("case13 PV requires the exact value shape")
    if tuple(context.shape) != tuple(value.shape):
        raise ValueError("case13 context and value shapes must match")
    if row_start not in (0, 256, 512, 768) or key_count != row_start + 256:
        raise ValueError("case13 PV row and key prefixes must end together")
    if not probabilities.is_contiguous():
        raise ValueError("case13 probabilities must be contiguous")
    if value.stride(-1) != 1 or context.stride(-1) != 1:
        raise ValueError("case13 value and context columns must be unit stride")

    _bf16_probability_value_case13_kernel[(batch * heads, 4)](
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
        key_count=key_count,
        block_key_count=triton.next_power_of_2(key_count),
        num_warps=4,
        num_stages=2,
    )


@triton.jit
def _bf16_probability_value_hd8_kernel(
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
    key_count: tl.constexpr,
    block_key_count: tl.constexpr,
):
    block_bh = tl.program_id(0)
    batch_index = block_bh // heads
    head_index = block_bh % heads

    rows = tl.arange(0, 16)
    keys = tl.arange(0, block_key_count)
    columns = tl.arange(0, 16)
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

    # Native softmax remains outside this kernel. Round its FP32 output to the
    # same BF16 materialization boundary before the BF16 PV tensor-core dot.
    probabilities = tl.load(
        probability_ptr + probability_offsets,
        mask=keys[None, :] < key_count,
        other=0.0,
    ).to(tl.bfloat16)
    values = tl.load(
        value_ptr + value_offsets,
        mask=(keys[:, None] < key_count) & (columns[None, :] < 8),
        other=0.0,
    )
    accumulator = tl.dot(probabilities, values, out_dtype=tl.float32)
    context = accumulator.to(tl.bfloat16)

    context_offsets = (
        batch_index * context_stride_batch
        + head_index * context_stride_head
        + (row_start + rows[:, None]) * context_stride_row
        + columns[None, :]
    )
    tl.store(
        context_ptr + context_offsets,
        context,
        mask=columns[None, :] < 8,
    )


def bf16_probability_value_hd8(
    probabilities: torch.Tensor,
    value: torch.Tensor,
    context: torch.Tensor,
    row_start: int,
) -> None:
    """Write one exact case-11 BF16 PV prefix into final context layout."""
    if probabilities.device.type != "cuda" or probabilities.dtype != torch.float32:
        raise ValueError("probabilities must be a CUDA FP32 tensor")
    if value.device != probabilities.device or value.dtype != torch.bfloat16:
        raise ValueError("value must be a CUDA BF16 tensor on the same device")
    if context.device != value.device or context.dtype != torch.bfloat16:
        raise ValueError("context must be a CUDA BF16 tensor on the same device")
    if probabilities.ndim != 4 or value.ndim != 4 or context.ndim != 4:
        raise ValueError("probabilities, value, and context must be rank-4 tensors")

    batch, heads, row_count, key_count = probabilities.shape
    if (batch, heads, row_count) != (64, 16, 16):
        raise ValueError("HD8 PV kernel requires the exact case-11 prefix shape")
    if key_count not in range(16, 129, 16):
        raise ValueError("case-11 prefix key count must be 16 through 128")
    if tuple(value.shape) != (64, 16, 128, 8):
        raise ValueError("HD8 PV kernel requires the exact case-11 value shape")
    if tuple(context.shape) != tuple(value.shape):
        raise ValueError("context and value shapes must match")
    if row_start + row_count != key_count:
        raise ValueError("case-11 row and key prefixes must end together")
    if not probabilities.is_contiguous():
        raise ValueError("HD8 PV kernel requires contiguous probabilities")
    if value.stride(-1) != 1 or context.stride(-1) != 1:
        raise ValueError("HD8 PV kernel requires unit-stride columns")

    _bf16_probability_value_hd8_kernel[(batch * heads,)](
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
        key_count=key_count,
        block_key_count=triton.next_power_of_2(key_count),
        # The 16x16 accumulator has only eight live output columns, while the
        # exact case-11 grid already exposes 1,024 independent batch/head
        # programs. Two warps reduce per-program scheduling and register
        # pressure without reducing grid-level parallelism.
        num_warps=2,
        num_stages=2,
    )


@triton.jit
def _bf16_probability_value_hd8_case7_kernel(
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
    key_count: tl.constexpr,
    block_key_count: tl.constexpr,
):
    block_bh = tl.program_id(0)
    batch_index = block_bh // heads
    head_index = block_bh % heads

    rows = tl.arange(0, 64)
    keys = tl.arange(0, block_key_count)
    columns = tl.arange(0, 16)
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

    # Keep the established attention boundary: native softmax produces FP32,
    # probabilities round to BF16 before PV, the tensor-core dot accumulates
    # in FP32, and the context rounds to BF16 before the output projection.
    probabilities = tl.load(
        probability_ptr + probability_offsets,
        mask=keys[None, :] < key_count,
        other=0.0,
    ).to(tl.bfloat16)
    values = tl.load(
        value_ptr + value_offsets,
        mask=(keys[:, None] < key_count) & (columns[None, :] < 8),
        other=0.0,
    )
    accumulator = tl.dot(probabilities, values, out_dtype=tl.float32)
    context = accumulator.to(tl.bfloat16)

    context_offsets = (
        batch_index * context_stride_batch
        + head_index * context_stride_head
        + (row_start + rows[:, None]) * context_stride_row
        + columns[None, :]
    )
    tl.store(
        context_ptr + context_offsets,
        context,
        mask=columns[None, :] < 8,
    )


def bf16_probability_value_hd8_case7(
    probabilities: torch.Tensor,
    value: torch.Tensor,
    context: torch.Tensor,
    row_start: int,
) -> None:
    """Write one exact case-7 BF16 PV prefix into final context layout."""
    if probabilities.device.type != "cuda" or probabilities.dtype != torch.float32:
        raise ValueError("probabilities must be a CUDA FP32 tensor")
    if value.device != probabilities.device or value.dtype != torch.bfloat16:
        raise ValueError("value must be a CUDA BF16 tensor on the same device")
    if context.device != value.device or context.dtype != torch.bfloat16:
        raise ValueError("context must be a CUDA BF16 tensor on the same device")
    if probabilities.ndim != 4 or value.ndim != 4 or context.ndim != 4:
        raise ValueError("probabilities, value, and context must be rank-4 tensors")

    batch, heads, row_count, key_count = probabilities.shape
    if (batch, heads, row_count) != (64, 4, 64):
        raise ValueError("case-7 HD8 PV kernel requires a 64x4x64 prefix")
    if key_count not in (64, 128):
        raise ValueError("case-7 prefix key count must be 64 or 128")
    if tuple(value.shape) != (64, 4, 128, 8):
        raise ValueError("case-7 HD8 PV kernel requires the exact value shape")
    if tuple(context.shape) != tuple(value.shape):
        raise ValueError("context and value shapes must match")
    if row_start not in (0, 64) or row_start + row_count != key_count:
        raise ValueError("case-7 row and key prefixes must end together")
    if not probabilities.is_contiguous():
        raise ValueError("case-7 HD8 PV kernel requires contiguous probabilities")
    if value.stride(-1) != 1 or context.stride(-1) != 1:
        raise ValueError("case-7 HD8 PV kernel requires unit-stride columns")

    _bf16_probability_value_hd8_case7_kernel[(batch * heads,)](
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
        key_count=key_count,
        block_key_count=triton.next_power_of_2(key_count),
        num_warps=4,
        num_stages=2,
    )
