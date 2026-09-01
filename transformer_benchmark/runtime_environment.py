"""Process environment required before importing PyTorch or CUDA libraries."""

from __future__ import annotations

import os


PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"


def configure_pre_torch_environment() -> None:
    """Use a fragmentation-resistant CUDA allocator unless explicitly set."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", PYTORCH_CUDA_ALLOC_CONF)
