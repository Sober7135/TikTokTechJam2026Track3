import unittest
from types import SimpleNamespace

import torch

from transformer_benchmark.correctness import AccuracySummary
from transformer_benchmark.profiling import (
    PROFILE_CANDIDATE_REPLAYS,
    PROFILE_MAX_CASES,
    PROFILE_MAX_EVENTS,
    PROFILE_WARMUP_REPLAYS,
    summarize_profile_events,
)
from transformer_benchmark.runner import (
    official_case_result,
    parse_args,
    settings_metadata,
    validate_args,
)
from transformer_benchmark.cases import OFFICIAL_TEST_CASES


class CandidateProfilingTests(unittest.TestCase):
    def test_profile_requires_cuda_and_an_explicit_bounded_case_list(self) -> None:
        default_profile = parse_args(["--profile-candidate", "--device", "cuda:0"])
        with self.assertRaisesRegex(ValueError, "explicit focused"):
            validate_args(default_profile, torch.device("cuda:0"), torch.bfloat16)

        custom_profile = parse_args(
            ["--profile-candidate", "--no-official-matrix", "--device", "cuda:0"]
        )
        with self.assertRaisesRegex(ValueError, "explicit focused"):
            validate_args(custom_profile, torch.device("cuda:0"), torch.bfloat16)

        cpu_profile = parse_args(
            ["--profile-candidate", "--official-cases", "6", "--device", "cpu"]
        )
        with self.assertRaisesRegex(ValueError, "CUDA"):
            validate_args(cpu_profile, torch.device("cpu"), torch.bfloat16)

        too_many_cases = parse_args(
            [
                "--profile-candidate",
                "--official-cases",
                *[str(case_id) for case_id in range(1, PROFILE_MAX_CASES + 2)],
                "--device",
                "cuda:0",
            ]
        )
        with self.assertRaisesRegex(ValueError, f"at most {PROFILE_MAX_CASES}"):
            validate_args(too_many_cases, torch.device("cuda:0"), torch.bfloat16)

        invalid_failure_mode = parse_args(
            [
                "--profile-candidate",
                "--official-cases",
                "13",
                "--benchmark-on-failure",
                "--device",
                "cuda:0",
            ]
        )
        with self.assertRaisesRegex(ValueError, "benchmark-on-failure"):
            validate_args(
                invalid_failure_mode, torch.device("cuda:0"), torch.bfloat16
            )

        valid_profile = parse_args(
            ["--profile-candidate", "--official-cases", "6", "13", "--device", "cuda:0"]
        )
        validate_args(valid_profile, torch.device("cuda:0"), torch.bfloat16)

    def test_profile_serialization_is_bounded_and_reports_attribution(self) -> None:
        events = [
            SimpleNamespace(
                key=("cudaGraphLaunch" if index == 0 else f"event-{index}") + "x" * 300,
                count=index + 1,
                self_device_time_total=1000.0,
                device_time_total=2000.0,
                self_device_memory_usage=index,
                device_memory_usage=index * 2,
            )
            for index in range(PROFILE_MAX_EVENTS + 7)
        ]
        summary = summarize_profile_events(events, [10.0, 10.0, 20.0], 50.0)

        self.assertEqual(summary["warmup_replays"], PROFILE_WARMUP_REPLAYS)
        self.assertEqual(summary["profiled_replays"], PROFILE_CANDIDATE_REPLAYS)
        self.assertEqual(summary["event_count_before_limit"], PROFILE_MAX_EVENTS + 7)
        self.assertEqual(len(summary["events"]), PROFILE_MAX_EVENTS)
        self.assertTrue(
            all(len(event["name"]) <= 256 for event in summary["events"])
        )
        self.assertTrue(summary["graph_replay_observed"])
        self.assertEqual(summary["attribution"]["status"], "approximately_complete")
        self.assertEqual(summary["attribution"]["unattributed_device_time_ms"], 3.0)

    def test_default_results_and_settings_omit_profile_fields(self) -> None:
        args = parse_args(["--official-cases", "1", "--device", "cpu"])
        settings = settings_metadata(args)
        self.assertNotIn("candidate_profile", settings)

        accuracy = AccuracySummary(
            passed=True,
            rtol=0.02,
            atol=0.002,
            max_abs_error=0.0,
            max_relative_error=0.0,
            failed_elements=0,
            total_elements=1,
            trials=[],
        )
        result = official_case_result(
            1, OFFICIAL_TEST_CASES[0], accuracy, {"speedup": 1.0}
        )
        self.assertNotIn("candidate_profile", result)


if __name__ == "__main__":
    unittest.main()
