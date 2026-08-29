"""Benchmark case definitions and deterministic input generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(frozen=True)
class TransformerConfig:
    batch_size: int
    seq_len: int
    d_model: int
    num_heads: int
    ffn_dim: int
    num_layers: int
    causal: bool

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if self.ffn_dim <= 0:
            raise ValueError("ffn_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")


# Competition appendix 3.7. Case IDs are one-based and match the published table.
OFFICIAL_TEST_CASES: Tuple[TransformerConfig, ...] = (
    TransformerConfig(64, 128, 128, 4, 128, 4, True),
    TransformerConfig(1, 128, 128, 4, 128, 4, True),
    TransformerConfig(4, 128, 128, 4, 128, 4, True),
    TransformerConfig(16, 128, 128, 4, 128, 4, True),
    TransformerConfig(128, 128, 128, 4, 128, 4, True),
    TransformerConfig(10000, 128, 128, 4, 128, 4, True),
    TransformerConfig(64, 128, 32, 4, 32, 4, True),
    TransformerConfig(64, 128, 1024, 4, 1024, 4, True),
    TransformerConfig(64, 128, 128, 1, 128, 4, True),
    TransformerConfig(64, 128, 128, 2, 128, 4, True),
    TransformerConfig(64, 128, 128, 16, 128, 4, True),
    TransformerConfig(64, 32, 128, 4, 128, 4, True),
    TransformerConfig(64, 1024, 128, 4, 128, 4, True),
    TransformerConfig(32, 100000, 1024, 16, 1024, 2, True),
)


def generate_random_case(
    config: TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    padding_ratio: float,
    input_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    x = torch.randn(
        config.batch_size,
        config.seq_len,
        config.d_model,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    x = x * input_scale

    if padding_ratio <= 0:
        valid_token_mask = torch.ones(
            config.batch_size, config.seq_len, device=device, dtype=torch.bool
        )
        return x, valid_token_mask

    min_valid = max(1, int(round(config.seq_len * (1.0 - padding_ratio))))
    lengths = torch.randint(
        low=min_valid,
        high=config.seq_len + 1,
        size=(config.batch_size,),
        generator=generator,
        device=device,
    )
    positions = torch.arange(config.seq_len, device=device)[None, :]
    valid_token_mask = positions < lengths[:, None]
    x = x.masked_fill(~valid_token_mask[..., None], 0)
    return x, valid_token_mask
