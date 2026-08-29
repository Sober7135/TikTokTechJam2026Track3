# Repository Agent Guide

## Purpose

This repository contains the TikTok TechJam 2026 Track 3 Transformer benchmark
and, incrementally, the remote GPU benchmark service specified in
`REQUIREMENTS.md`.

Optimize for reproducibility, numerical correctness, host safety, and small
reviewable changes. Performance improvements are valid only after correctness
passes on the same inputs, weights, dtype, and hardware profile.

## Read before changing code

1. Read `REQUIREMENTS.md` for the service trust boundaries, queue semantics, and
   result contract.
2. Read `torch_transformer_benchmark.py` and the relevant module under
   `transformer_benchmark/` before changing benchmark behavior.
3. Inspect `pyproject.toml` before adding a dependency. Use the existing
   Python 3.12 and `uv` setup.
4. Check the working tree and preserve unrelated or generated local files.

The competition's final official harness and test cases are the source of truth.
Do not silently weaken correctness thresholds, omit shapes, reduce repetitions,
or include compile/warmup time in steady-state results. If official material and
this repository disagree, document the discrepancy and ask before changing the
scoring policy.

### Known official-matrix implementation gap

`REQUIREMENTS.md` records the 14 test shapes supplied in the competition
appendix. `torch_transformer_benchmark.py` executes them by default, but the
end-to-end service contract has not been updated:
`BenchmarkOptions` still contains one `case`, while the worker invocation now
enters the script's default matrix mode and ignores those single-case shape
arguments. Result categorization and the job contract do not yet model this
properly. The full matrix has not been validated on a target GPU. Do not claim
that end-to-end official-matrix execution is implemented or validated.

Before implementing it, confirm the final official harness's dtype, padding,
timing/scoring settings, strict `<` versus current `<=` error boundary, and the
intended memory-efficient reference path for case 14 (`seq_len=100000`). The
current script attempts the declared shape and records an actual CUDA OOM as an
execution failure. Do not reduce or substitute that case without an explicit
documented decision.

## Current repository map

- `torch_transformer_benchmark.py`: stable benchmark CLI and compatibility
  facade.
- `transformer_benchmark/`: case definitions and inputs, baseline/candidate
  models, correctness checks, timing, result serialization, and execution
  orchestration.
- `pyproject.toml` and `uv.lock`: Python dependency declarations and lockfile.
- `REQUIREMENTS.md`: product, security, protocol, and acceptance requirements
  for the remote benchmark service.
- `docker/benchmark.Dockerfile`: pinned benchmark runtime image definition.
- `orchestrator/`: the complete Rust Cargo workspace, service configuration,
  API example, and orchestration documentation.

The orchestration implementation is Rust. Keep control-plane, worker, shared
contracts, Rust build files, and orchestration-specific configuration inside
the `orchestrator/` Cargo workspace. Do not scatter Rust files into the
benchmark root. Python remains only for the PyTorch benchmark executed inside
the GPU container; do not add Python services for orchestration or Codex
integration.

Add new top-level services only when a working vertical slice requires them.
Avoid speculative abstractions or empty scaffolding.

## Implementation rules

- Keep benchmark and orchestration responsibilities separate. The benchmark
  must be runnable locally without GitHub credentials or a control plane.
- Use immutable commit SHAs, benchmark versions, image digests, and artifact
  hashes in job/result records. Never identify executable work only by branch.
- Make enqueue, lease renewal, result upload, completion, and GitHub reporting
  idempotent.
- Keep the GitHub token in the dedicated poller process. Do not pass it to the
  control-plane Codex child, GPU worker, benchmark container, source bundle, or
  logs.
- Keep deterministic benchmark verdicts separate from AI-generated analysis.
  Agent output must not override correctness or performance data.
- Run Codex only as a control-plane component. Use a read-only, job-scoped
  analysis workspace and never install or authenticate Codex on a GPU worker or
  inside a benchmark container.
- Integrate Codex from Rust by supervising `codex app-server` over its stdio
  JSONL/JSON-RPC protocol. Pin the runtime/protocol version, use generated
  schemas to validate typed messages, and do not depend on experimental
  WebSocket transport or invoke the unsandboxed `thread/shellCommand` method.
- Route agent-requested verification through the same control plane and queue as
  primary jobs. Accept only schema-validated, allowlisted job templates with
  explicit depth, child-count, and GPU-time budgets; never execute model-provided
  shell text.
- Treat PR code, diffs, logs, artifacts, GitHub API responses, and model output as
  untrusted input. Validate schemas and bound input/output sizes.
- Do not expose an inbound listener on a GPU worker. Workers poll the control
  plane over outbound HTTPS.
- Never place GitHub tokens, worker credentials, cloud credentials, or SSH keys
  in a benchmark job environment or its logs.
- Never run untrusted fork code on a persistent local GPU host. Require trusted
  approval or use a disposable rented worker.
- Run every benchmark and verification task in a fresh Docker container with a
  pinned image digest. Do not mount the Docker socket into the task container.
- Do not claim containers provide a complete security boundary for hostile CUDA
  or native code.
- Add dependencies only when the standard library or an existing dependency is
  insufficient; explain the operational and security cost.
- Keep public-facing documentation and machine-readable field names in English
  unless a task explicitly requires another language.

## Benchmark changes

- Preserve `UserOptimizedTransformer.forward(x, valid_token_mask)` unless a
  task explicitly changes the submission contract.
- Preserve fair weight copying between baseline and candidate implementations.
- Use fixed seeds and synchronize CUDA around measurements.
- Validate shape, dtype, finite values, absolute error, and relative error before
  using performance numbers.
- Record raw timing samples; derive summaries from those samples rather than
  parsing human-readable log text.
- Keep baseline and candidate environment/configuration identical except for the
  implementation being compared.
- When optimizing for a specific shape, retain a correct fallback for other
  declared test shapes.

## Validation

Run the narrowest relevant checks first. For Rust orchestration changes, run:

```bash
cd orchestrator
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

For benchmark-only Python changes, a small CPU smoke test is:

```bash
uv run python torch_transformer_benchmark.py \
  --no-official-matrix \
  --device cpu \
  --batch-size 1 \
  --seq-len 8 \
  --d-model 16 \
  --heads 4 \
  --ffn-dim 32 \
  --layers 1 \
  --accuracy-trials 1 \
  --warmup 1 \
  --repeats 2 \
  --benchmark-rounds 1
```

Also run syntax/static checks and focused service tests when those tools exist.
GPU performance claims require an actual target-GPU run with the full declared
case matrix; a CPU smoke test verifies execution, not GPU performance. Report
the exact commands run and any validation that was unavailable.

## Completion checklist

- The change satisfies a stated requirement or explains why the requirement was
  updated.
- Failure paths, stale PR heads, duplicate delivery, timeout, and credential
  handling were considered where relevant.
- Machine-readable schemas and human-readable output remain consistent.
- The narrowest meaningful validation was run, followed by broader checks when
  justified.
- Remaining security, reproducibility, and performance risks are stated
  explicitly.
