"""Bounded candidate-only profiling for focused CUDA diagnostics."""

from __future__ import annotations

import math
import statistics
import time
from typing import Any, Dict, Iterable, List

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile

from .cases import TransformerConfig, generate_random_case
from .timing import warmup_model


PROFILE_WARMUP_REPLAYS = 2
PROFILE_CANDIDATE_REPLAYS = 3
PROFILE_MAX_CASES = 3
PROFILE_MAX_EVENTS = 30
PROFILE_MAX_EVENT_NAME_CHARS = 256
PROFILE_MAX_MEMORY_BYTES = (1 << 63) - 1


def _finite_nonnegative(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) and number > 0.0 else 0.0


def _bounded_memory_bytes(value: object) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return max(-PROFILE_MAX_MEMORY_BYTES, min(PROFILE_MAX_MEMORY_BYTES, number))


def summarize_profile_events(
    events: Iterable[Any], replay_cuda_samples_ms: List[float], region_wall_ms: float
) -> Dict[str, object]:
    """Serialize profiler aggregates into a small, stable machine-readable form."""
    serialized_events: List[Dict[str, object]] = []
    all_self_device_time_ms = 0.0
    graph_replay_observed = False

    for event in events:
        name = str(getattr(event, "key", "<unnamed>"))[:PROFILE_MAX_EVENT_NAME_CHARS]
        self_device_time_ms = _finite_nonnegative(
            getattr(event, "self_device_time_total", 0.0)
        ) / 1000.0
        device_time_ms = _finite_nonnegative(
            getattr(event, "device_time_total", 0.0)
        ) / 1000.0
        all_self_device_time_ms += self_device_time_ms
        lowered_name = name.lower()
        graph_replay_observed |= "cudagraph" in lowered_name or "cuda graph" in lowered_name
        serialized_events.append(
            {
                "name": name,
                "count": max(0, int(getattr(event, "count", 0))),
                "self_device_time_ms": self_device_time_ms,
                "device_time_ms": device_time_ms,
                "self_device_memory_bytes": _bounded_memory_bytes(
                    getattr(event, "self_device_memory_usage", 0)
                ),
                "device_memory_bytes": _bounded_memory_bytes(
                    getattr(event, "device_memory_usage", 0)
                ),
            }
        )

    serialized_events.sort(
        key=lambda event: (
            float(event["self_device_time_ms"]),
            float(event["device_time_ms"]),
            int(event["count"]),
        ),
        reverse=True,
    )
    top_events = serialized_events[:PROFILE_MAX_EVENTS]

    replay_cuda_total_ms = sum(
        sample
        for sample in replay_cuda_samples_ms
        if math.isfinite(sample) and sample >= 0.0
    )
    if replay_cuda_total_ms <= 0.0:
        attribution_ratio = None
        unattributed_device_time_ms = None
        attribution_status = "unavailable"
    else:
        attribution_ratio = all_self_device_time_ms / replay_cuda_total_ms
        unattributed_device_time_ms = max(
            0.0, replay_cuda_total_ms - all_self_device_time_ms
        )
        if all_self_device_time_ms == 0.0:
            attribution_status = "unattributed"
        elif attribution_ratio < 0.90:
            attribution_status = "partial"
        elif attribution_ratio <= 1.10:
            attribution_status = "approximately_complete"
        else:
            attribution_status = "over_attributed"

    return {
        "schema_version": 1,
        "activities": ["cpu", "cuda"],
        "warmup_replays": PROFILE_WARMUP_REPLAYS,
        "profiled_replays": PROFILE_CANDIDATE_REPLAYS,
        "event_limit": PROFILE_MAX_EVENTS,
        "event_count_before_limit": len(serialized_events),
        "record_shapes": False,
        "with_stack": False,
        "trace_exported": False,
        "memory_accounting": "torch_profiler_operator_allocation_deltas",
        "replay_wall_time": {
            "measurement": "cuda_event_elapsed_per_candidate_call",
            "samples_ms": replay_cuda_samples_ms,
            "median_ms": statistics.median(replay_cuda_samples_ms),
            "total_ms": replay_cuda_total_ms,
            "profile_region_host_wall_ms": region_wall_ms,
        },
        "graph_replay_observed": graph_replay_observed,
        "attribution": {
            "status": attribution_status,
            "self_device_time_ms": all_self_device_time_ms,
            "ratio_to_replay_wall_time": attribution_ratio,
            "unattributed_device_time_ms": unattributed_device_time_ms,
            "caveat": "operator device times may overlap or be hidden by CUDA Graph replay",
        },
        "events": top_events,
    }


def profile_candidate(
    optimized: nn.Module,
    config: TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    padding_ratio: float,
    input_scale: float,
) -> Dict[str, object]:
    """Profile three candidate calls after two unprofiled warmup calls."""
    if device.type != "cuda":
        raise ValueError("candidate profiling requires CUDA")

    x, valid_mask = generate_random_case(
        config=config,
        device=device,
        dtype=dtype,
        seed=seed + 200000,
        padding_ratio=padding_ratio,
        input_scale=input_scale,
    )
    warmup_model(
        optimized,
        x,
        valid_mask,
        PROFILE_WARMUP_REPLAYS,
        device,
    )

    starts = [
        torch.cuda.Event(enable_timing=True) for _ in range(PROFILE_CANDIDATE_REPLAYS)
    ]
    ends = [
        torch.cuda.Event(enable_timing=True) for _ in range(PROFILE_CANDIDATE_REPLAYS)
    ]
    torch.cuda.synchronize(device)
    region_start_ns = time.perf_counter_ns()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=True,
        with_stack=False,
    ) as profiler:
        with torch.inference_mode():
            for index in range(PROFILE_CANDIDATE_REPLAYS):
                starts[index].record()
                optimized(x, valid_mask)
                ends[index].record()
    torch.cuda.synchronize(device)
    region_wall_ms = (time.perf_counter_ns() - region_start_ns) / 1e6
    replay_cuda_samples_ms = [
        start.elapsed_time(end) for start, end in zip(starts, ends)
    ]
    return summarize_profile_events(
        profiler.key_averages(), replay_cuda_samples_ms, region_wall_ms
    )
