from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ORACLE_PATH = (
    Path(__file__).resolve().parents[1] / "research" / "case13_fragment_oracle.py"
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


if __name__ == "__main__":
    unittest.main()
