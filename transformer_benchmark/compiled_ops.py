"""Narrow compiled operations that preserve the explicit benchmark semantics."""

from __future__ import annotations

import torch


@torch.compile(fullgraph=True, dynamic=False)
def causal_scale_mask(
    scores: torch.Tensor,
    causal_mask: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Fuse BF16 score scaling and causal masking without fusing reductions."""
    return (scores * scale).masked_fill(causal_mask, float("-inf"))
