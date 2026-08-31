#!/usr/bin/env python3
"""Static lane/fragment oracle for the rejected O74 Case-6 MMA kernel.

This module does not execute CUDA.  It encodes the public PTX ISA mapping for
``mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`` and checks the direct
register loaders/stores used by O74.  With ``--historical-cache-root`` it also
audits the immutable Triton PTX captured by the fixed profiling job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


Coordinate = tuple[int, int]
PackedPair = tuple[Coordinate, Coordinate]


def _lane_parts(lane: int) -> tuple[int, int]:
    if not 0 <= lane < 32:
        raise ValueError("lane must be in [0, 32)")
    return lane >> 2, lane & 3


def official_a_pairs(lane: int) -> tuple[PackedPair, ...]:
    """PTX A register order; each tuple is one low/high BF16x2 register."""
    group, thread = _lane_parts(lane)
    low = 2 * thread
    return (
        ((group, low), (group, low + 1)),
        ((group + 8, low), (group + 8, low + 1)),
        ((group, low + 8), (group, low + 9)),
        ((group + 8, low + 8), (group + 8, low + 9)),
    )


def o74_a_pairs(lane: int) -> tuple[PackedPair, ...]:
    """Register order in rejected O74 commit 7f5c86f."""
    group, thread = _lane_parts(lane)
    low = 2 * thread
    return (
        ((group, low), (group, low + 1)),
        ((group, low + 8), (group, low + 9)),
        ((group + 8, low), (group + 8, low + 1)),
        ((group + 8, low + 8), (group + 8, low + 9)),
    )


def corrected_a_pairs(lane: int) -> tuple[PackedPair, ...]:
    """O74 load order after swapping packed registers a[1] and a[2]."""
    broken = o74_a_pairs(lane)
    return broken[0], broken[2], broken[1], broken[3]


def official_b_pairs(lane: int) -> tuple[PackedPair, ...]:
    """PTX B register order for column-major B."""
    group, thread = _lane_parts(lane)
    low = 2 * thread
    return (
        ((low, group), (low + 1, group)),
        ((low + 8, group), (low + 9, group)),
    )


def o74_qk_b_pairs(lane: int) -> tuple[PackedPair, ...]:
    """O74 K loads expressed as logical B=K^T coordinates."""
    return official_b_pairs(lane)


def o74_pv_b_pairs(lane: int) -> tuple[PackedPair, ...]:
    """O74 V loads expressed as logical B=V coordinates."""
    return official_b_pairs(lane)


def official_c_coordinates(lane: int) -> tuple[Coordinate, ...]:
    group, thread = _lane_parts(lane)
    low = 2 * thread
    return (
        (group, low),
        (group, low + 1),
        (group + 8, low),
        (group + 8, low + 1),
    )


def o74_store_coordinates(lane: int) -> tuple[Coordinate, ...]:
    return official_c_coordinates(lane)


def _flatten_pairs(pairs: tuple[PackedPair, ...]) -> set[Coordinate]:
    return {coordinate for pair in pairs for coordinate in pair}


def verify_fragment_maps() -> dict[str, object]:
    expected_a = {(row, column) for row in range(16) for column in range(16)}
    expected_b = {(row, column) for row in range(16) for column in range(8)}
    expected_c = {(row, column) for row in range(16) for column in range(8)}
    all_a = set().union(*(_flatten_pairs(official_a_pairs(lane)) for lane in range(32)))
    all_b = set().union(*(_flatten_pairs(official_b_pairs(lane)) for lane in range(32)))
    all_c = set().union(*(set(official_c_coordinates(lane)) for lane in range(32)))
    if all_a != expected_a or all_b != expected_b or all_c != expected_c:
        raise AssertionError("official PTX fragment maps do not cover their matrices")

    broken_lanes: dict[int, list[int]] = {}
    for lane in range(32):
        official = official_a_pairs(lane)
        broken = o74_a_pairs(lane)
        mismatches = [index for index in range(4) if official[index] != broken[index]]
        if mismatches:
            broken_lanes[lane] = mismatches
        if corrected_a_pairs(lane) != official:
            raise AssertionError(f"corrected A mapping still differs at lane {lane}")
        if o74_qk_b_pairs(lane) != official_b_pairs(lane):
            raise AssertionError(f"QK B mapping differs at lane {lane}")
        if o74_pv_b_pairs(lane) != official_b_pairs(lane):
            raise AssertionError(f"PV B mapping differs at lane {lane}")
        if o74_store_coordinates(lane) != official_c_coordinates(lane):
            raise AssertionError(f"C store mapping differs at lane {lane}")

    expected_broken = {lane: [1, 2] for lane in range(32)}
    if broken_lanes != expected_broken:
        raise AssertionError("O74 mismatch is not exactly a[1]/a[2] in every lane")
    return {
        "official_coverage": {"A": 256, "B": 128, "C": 128},
        "o74_broken_lanes": len(broken_lanes),
        "o74_mismatched_registers_per_lane": [1, 2],
        "corrected_qk": True,
        "corrected_pv": True,
    }


@dataclass(frozen=True)
class Case6Plan:
    key_count: int
    padded_key_count: int
    qk_fragments_per_warp: int
    qk_k16_steps: tuple[int, ...]
    pv_k16_steps: tuple[int, ...]


def case6_plan(key_count: int) -> Case6Plan:
    if key_count not in (32, 64, 96, 128):
        raise ValueError("Case-6 key_count must be 32, 64, 96, or 128")
    padded = 128 if key_count in (96, 128) else key_count
    n_tiles = padded // 8
    fragments_per_warp = (2 * n_tiles) // 4

    owners: dict[tuple[int, int], int] = {}
    for warp in range(4):
        fragment_base = warp * fragments_per_warp
        for fragment in range(fragments_per_warp):
            index = fragment_base + fragment
            tile = ((index // n_tiles) * 16, (index % n_tiles) * 8)
            if tile in owners:
                raise AssertionError(f"duplicate QK fragment {tile}")
            owners[tile] = warp
    expected_tiles = {
        (row, column)
        for row in (0, 16)
        for column in range(0, padded, 8)
    }
    if set(owners) != expected_tiles:
        raise AssertionError("Case-6 QK warp plan does not cover 32xN")

    pv_owners: dict[tuple[int, int], int] = {}
    for warp in range(4):
        for fragment in range(2):
            index = warp * 2 + fragment
            tile = ((index // 4) * 16, (index % 4) * 8)
            if tile in pv_owners:
                raise AssertionError(f"duplicate PV fragment {tile}")
            pv_owners[tile] = warp
    expected_pv = {
        (row, column) for row in (0, 16) for column in range(0, 32, 8)
    }
    if set(pv_owners) != expected_pv:
        raise AssertionError("Case-6 PV warp plan does not cover 32x32")

    return Case6Plan(
        key_count=key_count,
        padded_key_count=padded,
        qk_fragments_per_warp=fragments_per_warp,
        qk_k16_steps=(0, 16),
        pv_k16_steps=tuple(range(0, padded, 16)),
    )


PTX_OPCODE = "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
PTX_LINE = re.compile(
    r"mma\.sync\.aligned\.m16n8k16\.row\.col\.f32\.bf16\.bf16\.f32\s+"
    r"\{\s*([^}]+)\s*\},\s*\{[^}]+\},\s*\{[^}]+\},\s*\{\s*([^}]+)\s*\}"
)


@dataclass(frozen=True)
class HistoricalSpecialization:
    kind: str
    key_count: int
    directory: str
    sha256: str
    mma_count: int
    k16_steps: int


HISTORICAL_SPECIALIZATIONS = (
    HistoricalSpecialization("qk", 32, "CGVV54QBIBEYY27PVCAOAKAX7CRROJH3B4UHWINH3KMWQFBR6GFQ", "3aa839c5019bb6ef08338f57fc3b226d274b9afe42850526cef8f4e0e1cd1758", 4, 2),
    HistoricalSpecialization("pv", 32, "5HEW7J65PQ5XS25IX55XNQT2VCLVYUHW4JURSIJ7VRHJXEGBGW2A", "11d07126248f7100c9d9eedabfd2ab2353009681c61155a16d2ff5a7ef5cc6dc", 4, 2),
    HistoricalSpecialization("qk", 64, "PKD77XUZUBQQITK4LJNNYDYDFNEBU6C4GEJ32CU3HVPL2CRKOPNA", "a46679f78378e7dc779fad1b3d68763adabd42fbbce960c86cfea577e9679ccf", 8, 2),
    HistoricalSpecialization("pv", 64, "LFRLQC54YPWFV3PYWB3HZ4ZVWY7LMED4VC4X3WSJDYIP7M4U3U7Q", "bcc84921778c2fc4ce21be6d207d77d7406c71d7fe90e351b5dbdf463d194c16", 8, 4),
    HistoricalSpecialization("qk", 96, "ODWXJMDPIPPHEZWBQRWIG4G3YZO7JKQET5RDZ3UQZNCA3FLYVRHQ", "1a91c477c5057a8c4ea0622bc74971ee1778fc1c4b5a2b44bf4a73deb28aad22", 16, 2),
    HistoricalSpecialization("pv", 96, "AD7BZQC5IBBDWU2YOVU4GKG5NTWADQFTR2GQY6DSBGNPYBWYJZRA", "d7f57d212e9d171696d9570cdb4c962934db5db403a54c934c468bd506906dc0", 16, 8),
    HistoricalSpecialization("qk", 128, "EUUSVGJSYE3UMVBKDXKLRLDXTHVISAO72GBBVN2ZI64I4SG43HGA", "3304f33a63543bb82e62261502ef9c3b7c1470bb197b10bf8f481b3efe9204b9", 16, 2),
    HistoricalSpecialization("pv", 128, "DO3XOE46CLOZATXRL6NJCK6WQKIPVVKNRNOU67NIBOQQPTYKAT2A", "e0a5665db922a58dec5ea58ddb47bc168ae35165c7869fda37a4c9964c964849", 16, 8),
)


def verify_historical_ptx(cache_root: Path) -> list[dict[str, object]]:
    results = []
    for item in HISTORICAL_SPECIALIZATIONS:
        filename = (
            "_triangular_scores_kernel.ptx"
            if item.kind == "qk"
            else "_bf16_probability_value_kernel.ptx"
        )
        path = cache_root / item.directory / filename
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != item.sha256:
            raise AssertionError(f"historical PTX digest changed: {path}")
        text = payload.decode("utf-8")
        matches = list(PTX_LINE.finditer(text))
        if len(matches) != item.mma_count:
            raise AssertionError(f"unexpected MMA count in {path}")

        positions: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for index, match in enumerate(matches):
            destination = tuple(part.strip() for part in match.group(1).split(","))
            accumulator = tuple(part.strip() for part in match.group(2).split(","))
            if destination != accumulator:
                raise AssertionError(f"non-self-chained MMA in {path}")
            positions[destination].append(index)
        if any(len(chain) != item.k16_steps for chain in positions.values()):
            raise AssertionError(f"wrong K16 chain length in {path}")
        group_count = len(positions)
        for chain in positions.values():
            if chain != list(range(chain[0], item.mma_count, group_count)):
                raise AssertionError(f"K16 chain order changed in {path}")
        results.append(
            {
                "kind": item.kind,
                "key_count": item.key_count,
                "sha256": digest,
                "mma_count": len(matches),
                "output_fragments": group_count,
                "k16_steps": item.k16_steps,
                "self_chained": True,
            }
        )
    return results


def verify_rounding_boundaries() -> dict[str, tuple[str, ...]]:
    qk = ("FP32 MMA", "RNE BF16 score", "FP32 scale", "RNE BF16 score")
    softmax = (
        "BF16/FP16 load to FP32",
        "serial local max",
        "descending XOR max",
        "exp",
        "serial local sum",
        "descending XOR sum",
        "divide",
        "RNE BF16 probability",
    )
    pv = ("BF16 MMA inputs", "continuous FP32 K16 accumulation", "RNE BF16 context")
    return {"qk": qk, "softmax": softmax, "pv": pv}


def verify(cache_root: Path | None = None) -> dict[str, object]:
    report: dict[str, object] = {
        "fragment_maps": verify_fragment_maps(),
        "case6_plans": [case6_plan(key).__dict__ for key in (32, 64, 96, 128)],
        "rounding_boundaries": verify_rounding_boundaries(),
    }
    if cache_root is not None:
        report["historical_ptx"] = verify_historical_ptx(cache_root)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-cache-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.historical_cache_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
