#!/usr/bin/env python3
"""Executable schedule oracle for the exact Case-6 LayerNorm prototype.

This does not claim to emulate CUDA floating-point instructions.  It proves the
structural fact on which exactness depends: for N=128, PyTorch's only nonempty
native warp and each specialized row warp consume identical vec4 coordinates
and execute the same ordered Welford tree.  The native inter-warp tree then
combines that state only with zero-count identities.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


WIDTH = 128
VEC_SIZE = 4
WARP_SIZE = 32
ROWS_PER_BLOCK = 4
SHUFFLE_OFFSETS = (16, 8, 4, 2, 1)
PINNED_GIT_VERSION = "cf30153c4c131c8164ee7798e5022d810682e2cb"


@dataclass(frozen=True)
class WelfordState:
    mean: np.float32
    sigma2: np.float32
    count: np.float32


def _f32(value: object) -> np.float32:
    return np.float32(value)


def empty_state() -> WelfordState:
    return WelfordState(_f32(0), _f32(0), _f32(0))


def online_sum(value: np.float32, state: WelfordState) -> WelfordState:
    """Pinned source expression order, rounded after each scalar operation."""
    delta = _f32(value - state.mean)
    new_count = _f32(state.count + _f32(1))
    reciprocal = _f32(_f32(1) / new_count)
    new_mean = _f32(state.mean + _f32(delta * reciprocal))
    sigma2 = _f32(
        state.sigma2 + _f32(delta * _f32(value - new_mean))
    )
    return WelfordState(new_mean, sigma2, new_count)


def combine(data_b: WelfordState, data_a: WelfordState) -> WelfordState:
    """Pinned cuWelfordCombine argument and expression order."""
    delta = _f32(data_b.mean - data_a.mean)
    count = _f32(data_a.count + data_b.count)
    if count > _f32(0):
        coef = _f32(_f32(1) / count)
        n_a = _f32(data_a.count * coef)
        n_b = _f32(data_b.count * coef)
        mean = _f32(_f32(n_a * data_a.mean) + _f32(n_b * data_b.mean))
        sigma2 = _f32(
            _f32(data_a.sigma2 + data_b.sigma2)
            + _f32(
                _f32(_f32(delta * delta) * data_a.count) * n_b
            )
        )
    else:
        mean = _f32(0)
        sigma2 = _f32(0)
    return WelfordState(mean, sigma2, count)


def _warp_reduce(states: list[WelfordState]) -> WelfordState:
    assert len(states) == WARP_SIZE
    current = states
    for offset in SHUFFLE_OFFSETS:
        before = current
        current = list(before)
        for lane in range(WARP_SIZE - offset):
            current[lane] = combine(before[lane], before[lane + offset])
    return current[0]


def active_warp_state(row: np.ndarray) -> WelfordState:
    assert row.shape == (WIDTH,)
    lane_states: list[WelfordState] = []
    for lane in range(WARP_SIZE):
        state = empty_state()
        for coordinate in range(lane * VEC_SIZE, (lane + 1) * VEC_SIZE):
            state = online_sum(_f32(row[coordinate]), state)
        lane_states.append(state)
    return _warp_reduce(lane_states)


def native_state(row: np.ndarray) -> WelfordState:
    """Native blockDim=(32,4): one active warp plus three empty warps."""
    warp_states = [active_warp_state(row)]
    empty_warp = _warp_reduce([empty_state() for _ in range(WARP_SIZE)])
    warp_states.extend([empty_warp] * 3)
    for offset in (2, 1):
        before = warp_states
        warp_states = list(before)
        for warp in range(offset):
            warp_states[warp] = combine(before[warp], before[warp + offset])
    return warp_states[0]


def specialized_state(row: np.ndarray) -> WelfordState:
    """Specialized blockDim=(32,4): this row's owning warp only."""
    return active_warp_state(row)


def _state_bits(state: WelfordState) -> tuple[int, int, int]:
    values = np.asarray(
        [state.mean, state.sigma2, state.count], dtype=np.float32
    )
    return tuple(int(value) for value in values.view(np.uint32))


def verify_coordinate_map() -> dict[str, object]:
    native = {
        (0, lane, element): lane * VEC_SIZE + element
        for lane in range(WARP_SIZE)
        for element in range(VEC_SIZE)
    }
    specialized = {
        (row, lane, element): lane * VEC_SIZE + element
        for row in range(ROWS_PER_BLOCK)
        for lane in range(WARP_SIZE)
        for element in range(VEC_SIZE)
    }
    expected = list(range(WIDTH))
    assert sorted(native.values()) == expected
    for row in range(ROWS_PER_BLOCK):
        assert sorted(
            coordinate
            for (owner, _lane, _element), coordinate in specialized.items()
            if owner == row
        ) == expected
    assert len(specialized) == ROWS_PER_BLOCK * WIDTH
    return {
        "native_data_warps_per_row": 1,
        "native_empty_warps_per_row": 3,
        "specialized_rows_per_block": ROWS_PER_BLOCK,
        "coordinates_per_row": WIDTH,
        "rows_are_disjoint": True,
    }


def verify_welford_schedule() -> dict[str, object]:
    generator = torch.Generator(device="cpu").manual_seed(9206)
    rows = torch.randn(
        ROWS_PER_BLOCK,
        WIDTH,
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16).to(torch.float32).numpy()
    state_bits = []
    for row in rows:
        native = native_state(row)
        specialized = specialized_state(row)
        assert _state_bits(native) == _state_bits(specialized)
        state_bits.append(_state_bits(native))
    return {
        "bf16_rows_checked": ROWS_PER_BLOCK,
        "online_updates_per_lane": VEC_SIZE,
        "shuffle_offsets": list(SHUFFLE_OFFSETS),
        "native_interwarp_offsets": [2, 1],
        "state_bits": state_bits,
        "bitwise_equal": True,
    }


def verify_source_guards() -> dict[str, object]:
    source = (
        Path(__file__).resolve().parents[1]
        / "transformer_benchmark"
        / "case6_exact_layer_norm.cu"
    ).read_text(encoding="utf-8")
    ordered_fragments = (
        "float delta = val - curr_sum.mean;",
        "float new_count = curr_sum.count + 1.0f;",
        "float new_mean = curr_sum.mean + fn_rcp_mul(delta, new_count);",
        "curr_sum.sigma2 + delta * (val - new_mean)",
        "U delta = dataB.mean - dataA.mean;",
        "U count = dataA.count + dataB.count;",
        "auto nA = dataA.count * coef;",
        "auto nB = dataB.count * coef;",
        "mean = nA * dataA.mean + nB * dataB.mean;",
        "delta * delta * dataA.count * nB;",
        "for (int offset = (kWarpSize >> 1); offset > 0; offset >>= 1)",
        "wd = cuWelfordCombine(wd, wdB);",
        "wd.sigma2 = WARP_SHFL(wd.sigma2, 0) / float(kWidth);",
        "c10::cuda::compat::rsqrt(wd.sigma2 + eps)",
        "(rstd_val * (static_cast<float>(data.val[ii]) - wd.mean))",
    )
    positions = [source.index(fragment) for fragment in ordered_fragments]
    assert positions == sorted(positions)
    assert "__syncthreads" not in source
    assert "--use_fast_math" not in source
    assert "cf30153c" in source
    return {
        "pinned_git_version": PINNED_GIT_VERSION,
        "ordered_fragments": len(ordered_fragments),
        "no_block_barrier": True,
        "no_fast_math": True,
    }


def main() -> None:
    report = {
        "coordinates": verify_coordinate_map(),
        "welford": verify_welford_schedule(),
        "source": verify_source_guards(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
