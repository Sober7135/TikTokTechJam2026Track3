from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ORACLE_PATH = (
    Path(__file__).resolve().parents[1] / "research" / "case13_fragment_oracle.py"
)
CUDA_SOURCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "transformer_benchmark"
    / "case13_exact_attention.cu"
)
SPEC = importlib.util.spec_from_file_location("case13_fragment_oracle", ORACLE_PATH)
assert SPEC is not None and SPEC.loader is not None
ORACLE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ORACLE
SPEC.loader.exec_module(ORACLE)


class Case13FragmentOracleTest(unittest.TestCase):
    def test_official_fragment_maps_and_o74_correction(self) -> None:
        report = ORACLE.verify_single_mma_fragments()
        self.assertEqual(report["A_elements"], 256)
        self.assertEqual(report["B_elements"], 128)
        self.assertEqual(report["C_elements"], 128)
        self.assertEqual(report["o74_swapped_registers"], [1, 2])
        self.assertIs(report["corrected_all_lanes"], True)

    def test_all_case13_prefixes_have_exact_m16_ownership(self) -> None:
        for key_count, fragments, pv_steps, score_shared_bytes in (
            (256, 4, 16, 8192),
            (512, 8, 32, 16384),
            (768, 12, 48, 24576),
            (1024, 16, 64, 32768),
        ):
            plan = ORACLE.prefix_plan(key_count)
            self.assertEqual(plan.row_start, key_count - 256)
            self.assertEqual(plan.row_blocks, 16)
            self.assertEqual(plan.qk_fragments_per_warp, fragments)
            self.assertEqual(plan.qk_k16_steps, (0, 16))
            self.assertEqual(len(plan.pv_k16_steps), pv_steps)
            self.assertEqual(
                plan.softmax_iterations_per_lane,
                32 if key_count == 768 else key_count // 32,
            )
            self.assertEqual(
                plan.softmax_valid_iterations_per_lane,
                key_count // 32,
            )
            self.assertEqual(plan.query_shared_bytes, 1024)
            self.assertEqual(plan.score_shared_bytes, score_shared_bytes)
            self.assertEqual(plan.total_shared_bytes, score_shared_bytes + 1024)

    def test_native_boundaries_are_explicit(self) -> None:
        boundaries = ORACLE.verify_boundaries()
        self.assertIn("RNE BF16-to-FP16 score transport", boundaries["qk"])
        self.assertIn("descending XOR sum", boundaries["softmax"])
        self.assertEqual(boundaries["softmax"][-1], "RNE BF16 probability")
        self.assertEqual(boundaries["pv"][-1], "RNE BF16 context")

    def test_round_robin_future_skip_writes_every_score_once(self) -> None:
        total_fragments = 0
        total_future_fragments = 0
        for key_count in (256, 512, 768, 1024):
            plan = ORACLE.verify_qk_future_skip(key_count)
            self.assertEqual(plan.total_fragments, 2 * key_count)
            self.assertEqual(plan.fully_future_fragments, 240)
            self.assertEqual(plan.live_fragments, 2 * key_count - 240)
            self.assertEqual(
                plan.score_coordinates_written,
                16 * 16 * key_count,
            )
            self.assertEqual(
                plan.fully_future_score_coordinates,
                240 * 16 * 8,
            )
            self.assertEqual(
                plan.per_row_block_future_fragments,
                tuple(range(30, -1, -2)),
            )
            for warp_counts in plan.per_row_block_warp_future_fragments:
                self.assertLessEqual(max(warp_counts) - min(warp_counts), 1)
            total_fragments += plan.total_fragments
            total_future_fragments += plan.fully_future_fragments
        self.assertEqual(total_fragments, 5120)
        self.assertEqual(total_future_fragments, 960)

    def test_cuda_source_uses_the_proved_skip_schedule(self) -> None:
        source = CUDA_SOURCE_PATH.read_text()
        self.assertIn("(fragment * kWarps + warp) * 8", source)
        self.assertIn("matrix_column > maximum_query_row", source)
        self.assertIn("store_fully_future_score_fragment<KeyCount>", source)
        self.assertIn("reduction_start += 16", source)
        self.assertIn("mma_m16n8k16(score_acc, a, b)", source)
        self.assertIn("native_softmax_rows<KeyCount>", source)


if __name__ == "__main__":
    unittest.main()
