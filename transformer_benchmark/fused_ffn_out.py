"""Candidate-only fused FFN output projection and residual addition."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


_WIDTH = 128


@triton.autotune(
    configs=[
        triton.Config(
            {"block_rows": 64, "block_columns": 64, "block_reduction": 32},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"block_rows": 64, "block_columns": 128, "block_reduction": 32},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"block_rows": 128, "block_columns": 64, "block_reduction": 32},
            num_warps=8,
            num_stages=3,
        ),
    ],
    key=["row_count"],
)
@triton.jit
def _bf16_linear_residual_kernel(
    activation_ptr,
    weight_ptr,
    bias_ptr,
    residual_ptr,
    output_ptr,
    row_count,
    width: tl.constexpr,
    block_rows: tl.constexpr,
    block_columns: tl.constexpr,
    block_reduction: tl.constexpr,
):
    row_offsets = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    column_offsets = (
        tl.program_id(1) * block_columns + tl.arange(0, block_columns)
    )
    reduction_offsets = tl.arange(0, block_reduction)
    accumulator = tl.zeros((block_rows, block_columns), dtype=tl.float32)

    for reduction_start in range(0, width, block_reduction):
        reduction_indices = reduction_start + reduction_offsets
        activation = tl.load(
            activation_ptr
            + row_offsets[:, None] * width
            + reduction_indices[None, :],
            mask=row_offsets[:, None] < row_count,
            other=0.0,
        )
        weight = tl.load(
            weight_ptr
            + column_offsets[None, :] * width
            + reduction_indices[:, None],
            mask=column_offsets[None, :] < width,
            other=0.0,
        )
        accumulator += tl.dot(activation, weight, out_dtype=tl.float32)

    bias = tl.load(
        bias_ptr + column_offsets,
        mask=column_offsets < width,
        other=0.0,
    )
    offsets = row_offsets[:, None] * width + column_offsets[None, :]
    mask = (row_offsets[:, None] < row_count) & (
        column_offsets[None, :] < width
    )
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)

    # nn.Linear first materializes its post-bias result in BF16. Preserve that
    # rounding boundary before evaluating the separate BF16 residual add.
    projection_bf16 = (accumulator + bias[None, :]).to(tl.bfloat16)
    output_fp32 = projection_bf16.to(tl.float32) + residual.to(tl.float32)
    tl.store(output_ptr + offsets, output_fp32, mask=mask)


def bf16_linear_residual(
    activation: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    """Fuse a 128-wide BF16 linear projection with its residual addition."""
    if activation.device.type != "cuda" or activation.dtype != torch.bfloat16:
        raise ValueError("fused FFN output requires a CUDA BF16 activation")
    if activation.ndim != 3 or activation.shape[-1] != _WIDTH:
        raise ValueError("fused FFN activation must have shape [B, S, 128]")
    if tuple(residual.shape) != tuple(activation.shape):
        raise ValueError("fused FFN residual must match the activation shape")
    if tuple(weight.shape) != (_WIDTH, _WIDTH):
        raise ValueError("fused FFN weight must have shape [128, 128]")
    if tuple(bias.shape) != (_WIDTH,):
        raise ValueError("fused FFN bias must have shape [128]")
    if any(
        tensor.device != activation.device for tensor in (weight, bias, residual)
    ):
        raise ValueError("fused FFN tensors must share one CUDA device")
    if any(
        tensor.dtype != torch.bfloat16 for tensor in (weight, bias, residual)
    ):
        raise ValueError("fused FFN weight, bias, and residual must be BF16")
    if not all(tensor.is_contiguous() for tensor in (activation, weight, residual)):
        raise ValueError("fused FFN activation, weight, and residual must be contiguous")

    output = torch.empty_like(residual)
    row_count = activation.numel() // _WIDTH
    grid = lambda meta: (
        triton.cdiv(row_count, meta["block_rows"]),
        triton.cdiv(_WIDTH, meta["block_columns"]),
    )
    _bf16_linear_residual_kernel[grid](
        activation,
        weight,
        bias,
        residual,
        output,
        row_count,
        width=_WIDTH,
    )
    return output
