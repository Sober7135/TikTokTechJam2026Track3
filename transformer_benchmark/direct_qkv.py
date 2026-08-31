"""Candidate-only exact BF16 QKV projection into attention-native layout."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"block_rows": 32, "block_columns": 64, "block_reduction": 32},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"block_rows": 64, "block_columns": 64, "block_reduction": 32},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"block_rows": 64, "block_columns": 128, "block_reduction": 32},
            num_warps=8,
            num_stages=3,
        ),
    ],
    key=["row_count", "width", "head_count"],
)
@triton.jit
def _bf16_qkv_direct_layout_kernel(
    activation_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    row_count,
    width: tl.constexpr,
    packed_width: tl.constexpr,
    sequence_length: tl.constexpr,
    head_count: tl.constexpr,
    head_dimension: tl.constexpr,
    block_rows: tl.constexpr,
    block_columns: tl.constexpr,
    block_reduction: tl.constexpr,
):
    row_offsets = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    output_columns = (
        tl.program_id(1) * block_columns + tl.arange(0, block_columns)
    )
    reduction_offsets = tl.arange(0, block_reduction)
    valid_rows = row_offsets < row_count
    valid_columns = output_columns < packed_width
    accumulator = tl.zeros((block_rows, block_columns), dtype=tl.float32)

    for reduction_start in range(0, width, block_reduction):
        reduction_indices = reduction_start + reduction_offsets
        activation = tl.load(
            activation_ptr
            + row_offsets[:, None] * width
            + reduction_indices[None, :],
            mask=valid_rows[:, None],
            other=0.0,
        )
        # nn.Linear.weight is [out_features, in_features].  Loading the output
        # feature on the second tile axis presents [K, N] to tl.dot without
        # changing the reduction order used by the existing exact FFN kernels.
        weight = tl.load(
            weight_ptr
            + output_columns[None, :] * width
            + reduction_indices[:, None],
            mask=valid_columns[None, :],
            other=0.0,
        )
        accumulator += tl.dot(activation, weight, out_dtype=tl.float32)

    bias = tl.load(
        bias_ptr + output_columns,
        mask=valid_columns,
        other=0.0,
    )
    projected_bf16 = (accumulator + bias[None, :]).to(tl.bfloat16)

    # Convert packed output feature p*D+h*HD+d and flattened input row b*S+s
    # directly to contiguous [P, B, H, S, HD].  Q, K, and V therefore share
    # one allocation while each exposed [B, H, S, HD] slice is contiguous.
    projection = output_columns // width
    feature = output_columns - projection * width
    head = feature // head_dimension
    dimension = feature - head * head_dimension
    batch = row_offsets // sequence_length
    sequence = row_offsets - batch * sequence_length
    batch_count = row_count // sequence_length
    projection_stride = batch_count * head_count * sequence_length * head_dimension
    batch_stride = head_count * sequence_length * head_dimension
    head_stride = sequence_length * head_dimension
    output_offsets = (
        projection[None, :] * projection_stride
        + batch[:, None] * batch_stride
        + head[None, :] * head_stride
        + sequence[:, None] * head_dimension
        + dimension[None, :]
    )
    tl.store(
        output_ptr + output_offsets,
        projected_bf16,
        mask=valid_rows[:, None] & valid_columns[None, :],
    )


def bf16_qkv_direct_layout(
    activation: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_heads: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project packed Q/K/V directly into three contiguous BHSD views."""
    if activation.device.type != "cuda" or activation.dtype != torch.bfloat16:
        raise ValueError("direct QKV requires a CUDA BF16 activation")
    if activation.ndim != 3 or activation.shape[-1] not in (32, 128):
        raise ValueError("direct QKV activation width must be 32 or 128")
    batch, sequence_length, width = activation.shape
    if num_heads <= 0 or width % num_heads != 0:
        raise ValueError("direct QKV heads must divide the activation width")
    packed_width = 3 * width
    if tuple(weight.shape) != (packed_width, width):
        raise ValueError("direct QKV weight must have shape [3D, D]")
    if tuple(bias.shape) != (packed_width,):
        raise ValueError("direct QKV bias must have shape [3D]")
    if weight.device != activation.device or bias.device != activation.device:
        raise ValueError("direct QKV tensors must share one CUDA device")
    if weight.dtype != torch.bfloat16 or bias.dtype != torch.bfloat16:
        raise ValueError("direct QKV weight and bias must be BF16")
    if not activation.is_contiguous() or not weight.is_contiguous():
        raise ValueError("direct QKV activation and weight must be contiguous")

    head_dimension = width // num_heads
    output = torch.empty(
        (3, batch, num_heads, sequence_length, head_dimension),
        device=activation.device,
        dtype=activation.dtype,
    )
    row_count = batch * sequence_length
    grid = lambda meta: (
        triton.cdiv(row_count, meta["block_rows"]),
        triton.cdiv(packed_width, meta["block_columns"]),
    )
    _bf16_qkv_direct_layout_kernel[grid](
        activation,
        weight,
        bias,
        output,
        row_count,
        width=width,
        packed_width=packed_width,
        sequence_length=sequence_length,
        head_count=num_heads,
        head_dimension=head_dimension,
    )
    query, key, value = output.unbind(dim=0)
    return query, key, value
