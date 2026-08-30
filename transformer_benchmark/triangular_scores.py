"""Causal score generation that preserves the benchmark's BF16 boundaries."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _triangular_scores_kernel(
    query_ptr,
    key_ptr,
    scores_ptr,
    scale_value,
    query_stride_batch: tl.constexpr,
    query_stride_head: tl.constexpr,
    query_stride_seq: tl.constexpr,
    key_stride_batch: tl.constexpr,
    key_stride_head: tl.constexpr,
    key_stride_seq: tl.constexpr,
    scores_stride_bh: tl.constexpr,
    scores_stride_row: tl.constexpr,
    heads: tl.constexpr,
    query_row_start,
    query_count: tl.constexpr,
    key_count: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    block_head_dim: tl.constexpr,
    block_size: tl.constexpr,
    skip_fully_future: tl.constexpr,
    output_float32: tl.constexpr,
):
    block_bh = tl.program_id(0)
    batch_index = block_bh // heads
    head_index = block_bh % heads
    query_block = tl.program_id(1)
    key_block = tl.program_id(2)
    output_query_rows = query_block * block_size + tl.arange(0, block_size)
    query_rows = query_row_start + output_query_rows
    key_rows = key_block * block_size + tl.arange(0, block_size)
    score_offsets = (
        block_bh * scores_stride_bh
        + output_query_rows[:, None] * scores_stride_row
        + key_rows[None, :]
    )
    output_mask = (output_query_rows[:, None] < query_count) & (
        key_rows[None, :] < key_count
    )

    tile_is_fully_future = skip_fully_future & (
        key_block * block_size
        >= query_row_start + (query_block + 1) * block_size
    )
    if tile_is_fully_future:
        future_scores = tl.full(
            (block_size, block_size), -float("inf"), tl.float32
        )
        if not output_float32:
            future_scores = future_scores.to(tl.bfloat16)
        tl.store(scores_ptr + score_offsets, future_scores, mask=output_mask)
    else:
        columns = tl.arange(0, block_head_dim)
        query_offsets = (
            batch_index * query_stride_batch
            + head_index * query_stride_head
            + query_rows[:, None] * query_stride_seq
            + columns[None, :]
        )
        key_offsets = (
            batch_index * key_stride_batch
            + head_index * key_stride_head
            + key_rows[:, None] * key_stride_seq
            + columns[None, :]
        )
        query_tile = tl.load(
            query_ptr + query_offsets,
            mask=(query_rows[:, None] < seq_len)
            & (columns[None, :] < head_dim),
            other=0.0,
        )
        key_tile = tl.load(
            key_ptr + key_offsets,
            mask=(key_rows[:, None] < seq_len)
            & (columns[None, :] < head_dim),
            other=0.0,
        )
        scores = tl.dot(query_tile, tl.trans(key_tile), out_dtype=tl.float32)
        scores = scores.to(tl.bfloat16)
        scores = (scores * scale_value).to(tl.bfloat16)
        if output_float32:
            scores = scores.to(tl.float32)
        causal_mask = key_rows[None, :] <= query_rows[:, None]
        scores = tl.where(causal_mask, scores, -float("inf"))
        tl.store(scores_ptr + score_offsets, scores, mask=output_mask)


def triangular_causal_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    scale: float,
    output_float32: bool = False,
) -> torch.Tensor:
    """Compute lower-triangular BF16 scores and materialize the native boundary."""
    if query.shape != key.shape:
        raise ValueError("query and key shapes must match")
    if query.ndim != 4 or query.shape[-1] not in (8, 32, 64, 128, 256):
        raise ValueError(
            "triangular scores require head_dim in {8, 32, 64, 128, 256}"
        )
    if query.shape[-2] not in (32, 128, 1024):
        raise ValueError(
            "triangular scores currently require seq_len=32, 128, or 1024"
        )
    if query.device.type != "cuda" or query.dtype != torch.bfloat16:
        raise ValueError("triangular scores require CUDA BF16 tensors")
    if query.stride(-1) != 1 or key.stride(-1) != 1:
        raise ValueError("triangular scores require unit-stride head dimensions")

    batch, heads, seq_len, head_dim = query.shape
    scores = torch.empty(
        (batch, heads, seq_len, seq_len),
        device=query.device,
        dtype=torch.float32 if output_float32 else query.dtype,
    )
    if seq_len == 32:
        block_size = 32
    elif seq_len == 128:
        small_batch_head32 = batch <= 4 and heads == 4 and head_dim == 32
        block_size = 32 if small_batch_head32 else 64
    else:
        block_size = 128
    block_head_dim = max(16, head_dim)
    blocks = triton.cdiv(seq_len, block_size)
    grid = (batch * heads, blocks, blocks)
    _triangular_scores_kernel[grid](
        query,
        key,
        scores,
        scale,
        query_stride_batch=query.stride(0),
        query_stride_head=query.stride(1),
        query_stride_seq=query.stride(2),
        key_stride_batch=key.stride(0),
        key_stride_head=key.stride(1),
        key_stride_seq=key.stride(2),
        scores_stride_bh=scores.stride(1),
        scores_stride_row=scores.stride(2),
        heads=heads,
        query_row_start=0,
        query_count=seq_len,
        key_count=seq_len,
        seq_len=seq_len,
        head_dim=head_dim,
        block_head_dim=block_head_dim,
        block_size=block_size,
        skip_fully_future=(batch, heads, seq_len, head_dim)
        == (64, 1, 128, 128),
        output_float32=output_float32,
        num_warps=4 if block_size == 32 else 8,
        num_stages=2,
    )
    return scores


def triangular_causal_score_chunk(
    query: torch.Tensor,
    key: torch.Tensor,
    scale: float,
    row_start: int,
    row_stop: int,
    output_float32: bool = False,
) -> torch.Tensor:
    """Compute a compact causal score block using the exact tiled QK path."""
    if query.shape != key.shape or query.ndim != 4:
        raise ValueError("query and key must have the same rank-4 shape")
    if query.device.type != "cuda" or query.dtype != torch.bfloat16:
        raise ValueError("score chunks require CUDA BF16 tensors")
    if query.stride(-1) != 1 or key.stride(-1) != 1:
        raise ValueError("score chunks require unit-stride head dimensions")

    batch, heads, seq_len, head_dim = query.shape
    if not (0 <= row_start < row_stop <= seq_len):
        raise ValueError("row range must be within the sequence")
    row_count = row_stop - row_start
    if seq_len == 1024:
        block_size = 128
        if head_dim != 32:
            raise ValueError("seq_len=1024 score chunks require head_dim=32")
    elif seq_len == 128:
        block_size = row_count
        if block_size not in (16, 32, 64):
            raise ValueError("seq_len=128 score chunk rows must be 16, 32, or 64")
        if head_dim not in (8, 32, 64):
            raise ValueError(
                "seq_len=128 score chunks require head_dim in {8, 32, 64}"
            )
    else:
        raise ValueError("score chunks currently require seq_len=128 or 1024")
    if row_start % block_size or row_stop % block_size:
        raise ValueError("score chunk row boundaries must be tile-aligned")

    scores = torch.empty(
        (batch, heads, row_count, row_stop),
        device=query.device,
        dtype=torch.float32 if output_float32 else query.dtype,
    )
    grid = (
        batch * heads,
        triton.cdiv(row_count, block_size),
        triton.cdiv(row_stop, block_size),
    )
    _triangular_scores_kernel[grid](
        query,
        key,
        scores,
        scale,
        query_stride_batch=query.stride(0),
        query_stride_head=query.stride(1),
        query_stride_seq=query.stride(2),
        key_stride_batch=key.stride(0),
        key_stride_head=key.stride(1),
        key_stride_seq=key.stride(2),
        scores_stride_bh=scores.stride(1),
        scores_stride_row=scores.stride(2),
        heads=heads,
        query_row_start=row_start,
        query_count=row_count,
        key_count=row_stop,
        seq_len=seq_len,
        head_dim=head_dim,
        block_head_dim=max(16, head_dim),
        block_size=block_size,
        skip_fully_future=seq_len == 1024,
        output_float32=output_float32,
        num_warps=4 if block_size <= 32 else 8,
        num_stages=2,
    )
    return scores
