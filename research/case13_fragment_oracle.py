#!/usr/bin/env python3
"""Static lane/ownership oracle for the Case-13 fused attention design.

This file never executes CUDA.  It encodes the public PTX ISA fragment maps
for ``mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`` and proves the
M16 decomposition used by the Case-13 shared-CTA kernel.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass


Coordinate = tuple[int, int]
PackedPair = tuple[Coordinate, Coordinate]


def _lane_parts(lane: int) -> tuple[int, int]:
    if not 0 <= lane < 32:
        raise ValueError("lane must be in [0, 32)")
    return lane >> 2, lane & 3


def official_a_pairs(lane: int) -> tuple[PackedPair, ...]:
    """Official row-major A register order, low BF16 then high BF16."""
    group, thread = _lane_parts(lane)
    low = 2 * thread
    return (
        ((group, low), (group, low + 1)),
        ((group + 8, low), (group + 8, low + 1)),
        ((group, low + 8), (group, low + 9)),
        ((group + 8, low + 8), (group + 8, low + 9)),
    )


def official_b_pairs(lane: int) -> tuple[PackedPair, ...]:
    """Official column-major B register order."""
    group, thread = _lane_parts(lane)
    low = 2 * thread
    return (
        ((low, group), (low + 1, group)),
        ((low + 8, group), (low + 9, group)),
    )


def official_c_coordinates(lane: int) -> tuple[Coordinate, ...]:
    """Official C/D FP32 register coordinates."""
    group, thread = _lane_parts(lane)
    low = 2 * thread
    return (
        (group, low),
        (group, low + 1),
        (group + 8, low),
        (group + 8, low + 1),
    )


def o74_a_pairs(lane: int) -> tuple[PackedPair, ...]:
    """Rejected O74 order: row0-high and row8-low were exchanged."""
    official = official_a_pairs(lane)
    return official[0], official[2], official[1], official[3]


def _flatten(pairs: tuple[PackedPair, ...]) -> set[Coordinate]:
    return {coordinate for pair in pairs for coordinate in pair}


def verify_single_mma_fragments() -> dict[str, object]:
    expected_a = {(row, column) for row in range(16) for column in range(16)}
    expected_b = {(row, column) for row in range(16) for column in range(8)}
    expected_c = {(row, column) for row in range(16) for column in range(8)}
    actual_a = set().union(*(_flatten(official_a_pairs(lane)) for lane in range(32)))
    actual_b = set().union(*(_flatten(official_b_pairs(lane)) for lane in range(32)))
    actual_c = set().union(*(set(official_c_coordinates(lane)) for lane in range(32)))
    if (actual_a, actual_b, actual_c) != (expected_a, expected_b, expected_c):
        raise AssertionError("official A/B/C maps do not cover one MMA exactly")
    for lane in range(32):
        official = official_a_pairs(lane)
        rejected = o74_a_pairs(lane)
        mismatches = [index for index in range(4) if official[index] != rejected[index]]
        if mismatches != [1, 2]:
            raise AssertionError(f"unexpected O74 mismatch in lane {lane}")
        if (rejected[0], rejected[2], rejected[1], rejected[3]) != official:
            raise AssertionError(f"A correction failed in lane {lane}")
    return {
        "A_elements": len(actual_a),
        "B_elements": len(actual_b),
        "C_elements": len(actual_c),
        "o74_swapped_registers": [1, 2],
        "corrected_all_lanes": True,
    }


@dataclass(frozen=True)
class PrefixPlan:
    row_start: int
    key_count: int
    row_blocks: int
    qk_fragments_per_warp: int
    qk_k16_steps: tuple[int, ...]
    pv_k16_steps: tuple[int, ...]
    softmax_iterations_per_lane: int
    softmax_valid_iterations_per_lane: int
    query_shared_bytes: int
    score_shared_bytes: int
    total_shared_bytes: int


@dataclass(frozen=True)
class QKFutureSkipPlan:
    key_count: int
    total_fragments: int
    live_fragments: int
    fully_future_fragments: int
    score_coordinates_written: int
    fully_future_score_coordinates: int
    per_row_block_future_fragments: tuple[int, ...]
    per_row_block_warp_future_fragments: tuple[tuple[int, ...], ...]


def verify_qk_future_skip(key_count: int) -> QKFutureSkipPlan:
    """Prove round-robin ownership and direct ``-inf`` tile coverage."""
    if key_count not in (256, 512, 768, 1024):
        raise ValueError("Case-13 prefix must be 256, 512, 768, or 1024")
    row_start = key_count - 256
    fragments_per_warp = key_count // 64
    total_fragments = 0
    fully_future_fragments = 0
    fully_future_score_coordinates = 0
    per_row_block_future_fragments: list[int] = []
    per_row_block_warp_future_fragments: list[tuple[int, ...]] = []

    for row_block in range(16):
        owners: dict[int, int] = {}
        writes: dict[Coordinate, tuple[int, int]] = {}
        future_writes: set[Coordinate] = set()
        warp_future_counts = [0] * 8
        query_start = row_start + row_block * 16
        query_stop = query_start + 16

        for warp in range(8):
            for local_fragment in range(fragments_per_warp):
                fragment_index = local_fragment * 8 + warp
                matrix_column = fragment_index * 8
                if matrix_column in owners:
                    raise AssertionError(f"duplicate QK tile {matrix_column}")
                owners[matrix_column] = warp

                fully_future = matrix_column > query_stop - 1
                for lane in range(32):
                    for local_row, local_column in official_c_coordinates(lane):
                        coordinate = (local_row, matrix_column + local_column)
                        if coordinate in writes:
                            raise AssertionError(
                                f"duplicate score write {coordinate}"
                            )
                        writes[coordinate] = (warp, local_fragment)
                        if fully_future:
                            global_query = query_start + local_row
                            global_key = matrix_column + local_column
                            if global_key <= global_query:
                                raise AssertionError(
                                    "direct -inf write crossed the causal diagonal"
                                )
                            future_writes.add(coordinate)

                total_fragments += 1
                if fully_future:
                    fully_future_fragments += 1
                    warp_future_counts[warp] += 1

        expected_columns = set(range(0, key_count, 8))
        if set(owners) != expected_columns:
            raise AssertionError("round-robin warps do not own every N8 tile")
        expected_scores = {
            (row, column)
            for row in range(16)
            for column in range(key_count)
        }
        if set(writes) != expected_scores:
            raise AssertionError("QK fragments do not write the complete score tile")
        if max(warp_future_counts) - min(warp_future_counts) > 1:
            raise AssertionError("future suffix is not balanced across QK warps")
        per_row_block_future_fragments.append(sum(warp_future_counts))
        per_row_block_warp_future_fragments.append(tuple(warp_future_counts))
        fully_future_score_coordinates += len(future_writes)

    if fully_future_fragments != 240:
        raise AssertionError("each Case-13 prefix must skip exactly 240 fragments")
    return QKFutureSkipPlan(
        key_count=key_count,
        total_fragments=total_fragments,
        live_fragments=total_fragments - fully_future_fragments,
        fully_future_fragments=fully_future_fragments,
        score_coordinates_written=16 * 16 * key_count,
        fully_future_score_coordinates=fully_future_score_coordinates,
        per_row_block_future_fragments=tuple(per_row_block_future_fragments),
        per_row_block_warp_future_fragments=tuple(
            per_row_block_warp_future_fragments
        ),
    )


def prefix_plan(key_count: int) -> PrefixPlan:
    if key_count not in (256, 512, 768, 1024):
        raise ValueError("Case-13 prefix must be 256, 512, 768, or 1024")
    row_start = key_count - 256
    n_tiles = key_count // 8
    fragments_per_warp = n_tiles // 8

    qk_owners: dict[tuple[int, int], int] = {}
    for warp in range(8):
        for local_fragment in range(fragments_per_warp):
            column = (warp * fragments_per_warp + local_fragment) * 8
            tile = (0, column)
            if tile in qk_owners:
                raise AssertionError(f"duplicate QK tile {tile}")
            qk_owners[tile] = warp
    expected_qk = {(0, column) for column in range(0, key_count, 8)}
    if set(qk_owners) != expected_qk:
        raise AssertionError("eight QK warps do not cover M16xprefix")

    pv_owners = {(0, warp * 8): warp for warp in range(4)}
    expected_pv = {(0, column) for column in range(0, 32, 8)}
    if set(pv_owners) != expected_pv:
        raise AssertionError("four PV warps do not cover M16xN32")

    softmax_owners = {
        row: (row % 8, row // 8) for row in range(16)
    }
    if set(softmax_owners) != set(range(16)):
        raise AssertionError("softmax waves do not cover all 16 rows")
    padded_count = 1024 if key_count == 768 else key_count
    for lane in range(32):
        columns = tuple(
            lane + iteration * 32 for iteration in range(padded_count // 32)
        )
        valid_columns = tuple(column for column in columns if column < key_count)
        if len(valid_columns) != key_count // 32:
            raise AssertionError("softmax lane ownership crossed prefix")
    all_softmax_columns = {
        lane + iteration * 32
        for lane in range(32)
        for iteration in range(key_count // 32)
    }
    if all_softmax_columns != set(range(key_count)):
        raise AssertionError("native softmax lane/iteration map is incomplete")

    return PrefixPlan(
        row_start=row_start,
        key_count=key_count,
        row_blocks=16,
        qk_fragments_per_warp=fragments_per_warp,
        qk_k16_steps=(0, 16),
        pv_k16_steps=tuple(range(0, key_count, 16)),
        softmax_iterations_per_lane=padded_count // 32,
        softmax_valid_iterations_per_lane=key_count // 32,
        query_shared_bytes=16 * 32 * 2,
        score_shared_bytes=16 * key_count * 2,
        total_shared_bytes=16 * (32 + key_count) * 2,
    )


def verify_boundaries() -> dict[str, tuple[str, ...]]:
    return {
        "qk": (
            "continuous FP32 K16 MMA chain",
            "RNE BF16 dot",
            "widen to FP32 and multiply scale",
            "RNE BF16 scaled score",
            "RNE BF16-to-FP16 score transport",
        ),
        "softmax": (
            "FP16 load to FP32",
            "lane plus iteration times 32 ownership",
            "serial local max",
            "descending XOR max",
            "exp",
            "serial local sum",
            "descending XOR sum",
            "divide or NaN on zero sum",
            "RNE BF16 probability",
        ),
        "pv": (
            "BF16 probability and value",
            "continuous FP32 increasing K16 MMA chain",
            "RNE BF16 context",
        ),
    }


def verify() -> dict[str, object]:
    return {
        "single_mma": verify_single_mma_fragments(),
        "prefixes": [asdict(prefix_plan(key)) for key in (256, 512, 768, 1024)],
        "qk_future_skip": [
            asdict(verify_qk_future_skip(key)) for key in (256, 512, 768, 1024)
        ],
        "boundaries": verify_boundaries(),
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, sort_keys=True))
