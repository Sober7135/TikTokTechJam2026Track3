from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ORACLE_PATH = (
    Path(__file__).resolve().parents[1] / "research" / "case6_fragment_oracle.py"
)
SPEC = importlib.util.spec_from_file_location("case6_fragment_oracle", ORACLE_PATH)
assert SPEC is not None and SPEC.loader is not None
ORACLE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ORACLE
SPEC.loader.exec_module(ORACLE)


class Case6FragmentOracleTest(unittest.TestCase):
    def test_o74_a_pair_swap_is_the_only_fragment_map_mismatch(self) -> None:
        report = ORACLE.verify_fragment_maps()
        self.assertEqual(report["o74_broken_lanes"], 32)
        self.assertEqual(report["o74_mismatched_registers_per_lane"], [1, 2])
        self.assertIs(report["corrected_qk"], True)
        self.assertIs(report["corrected_pv"], True)

    def test_case6_prefix_plans_cover_qk_and_pv_without_reassociation(self) -> None:
        for key_count, padded, pv_steps in (
            (32, 32, 2),
            (64, 64, 4),
            (96, 128, 8),
            (128, 128, 8),
        ):
            plan = ORACLE.case6_plan(key_count)
            self.assertEqual(plan.padded_key_count, padded)
            self.assertEqual(plan.qk_k16_steps, (0, 16))
            self.assertEqual(len(plan.pv_k16_steps), pv_steps)

    def test_required_bf16_boundaries_are_explicit(self) -> None:
        boundaries = ORACLE.verify_rounding_boundaries()
        self.assertEqual(boundaries["qk"].count("RNE BF16 score"), 2)
        self.assertEqual(boundaries["softmax"][-1], "RNE BF16 probability")
        self.assertEqual(boundaries["pv"][-1], "RNE BF16 context")


if __name__ == "__main__":
    unittest.main()
