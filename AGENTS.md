# Repository Agent Guide

## Purpose

This repository contains the TikTok TechJam 2026 Track 3 Transformer benchmark
and the local `benchmarkctl` GPU queue.

The only orchestration loop is agent plus `benchmarkctl`: an agent submits an
immutable local snapshot, `benchmarkctl` serializes GPU execution, and the same
agent reads the deterministic result and log. Do not add a second orchestration
path or a verification child-job system.

Optimize for reproducibility, numerical correctness, host safety, and small
reviewable changes. Performance improvements are valid only after correctness
passes on the same inputs, weights, dtype, and hardware profile.

## Read before changing code

1. Read `REQUIREMENTS.md` for the local queue semantics and result contract.
2. Read `torch_transformer_benchmark.py` and the relevant module under
   `transformer_benchmark/` before changing benchmark behavior.
3. Inspect `pyproject.toml` before adding a dependency. Use the existing Python
   3.12 and `uv` setup.
4. Check the working tree and preserve unrelated or generated local files.

The competition's final official harness and test cases are the source of truth.
Do not silently weaken correctness thresholds, omit shapes, reduce repetitions,
or include compile/warmup time in steady-state results. If official material and
this repository disagree, document the discrepancy and ask before changing the
scoring policy.

The declared matrix includes case 14 (`seq_len=100000`). The current script
attempts that shape and records an actual CUDA OOM as an execution failure. Do
not reduce or substitute the case without an explicit documented decision. GPU
performance claims require a full target-GPU matrix run.

## Repository map

- `torch_transformer_benchmark.py`: stable benchmark CLI and compatibility
  facade.
- `transformer_benchmark/`: case definitions and inputs, baseline/candidate
  models, correctness checks, timing, result serialization, and execution.
- `pyproject.toml` and `uv.lock`: Python dependency declarations and lockfile.
- `REQUIREMENTS.md`: local benchmark and queue contract.
- `orchestrator/`: Rust Cargo workspace containing only `benchmarkctl`.
- `orchestrator/crates/benchmarkctl/`: local snapshot, FIFO queue, exclusive GPU
  lock, native execution, Stop-hook integration, and inspection CLI.

Python remains the benchmark implementation. Rust remains the local queue and
process supervisor. Do not add services, network APIs, shared service contracts,
or Codex App Server integration.

## Agent workflow

- Agents must invoke all local GPU benchmarks through `benchmarkctl`; never run
  the Python GPU benchmark directly.
- Submit every local GPU benchmark with `benchmarkctl submit`. It inherits
  `CODEX_SESSION_ID`, snapshots immediately, and returns. Finish the turn so the
  project Stop hook can run the job asynchronously.
- When the Stop hook continues the turn, read its reported `result.json` and
  `benchmark.log`, verify numerical correctness first, and only then interpret
  performance.
- Use `benchmarkctl cancel <job-id>` only to cancel an `awaiting_hook` job that
  should no longer run. It does not delete the snapshot or stop a claimed job.
- Do not launch another `codex exec` process as a callback.
- Every execution is an ordinary benchmark job. Do not introduce parent/child
  verification jobs or allow model output to create another job type.
- A new commit is validated by submitting a new ordinary snapshot through
  `benchmarkctl`.
- Agents in separate Git worktrees must use the same absolute
  `BENCHMARKCTL_STATE_DIR` when they target the same GPU.

## Local execution rules

- Snapshot tracked and non-ignored untracked regular files immediately at
  submission; reject symbolic links and submodules.
- Use immutable commit SHAs, snapshot digests, lockfile digests, Python binary
  digests, and package-inventory digests in job records.
- Keep the GPU lock held by the benchmark wrapper so a surviving child cannot
  overlap another job after its submitting agent exits.
- Run each job from its fresh snapshot with a cleared environment, explicit GPU,
  fresh home/cache directories, and a bounded wall timeout.
- Never place Codex credentials, cloud credentials, SSH keys, or unrelated
  service variables in the benchmark environment or logs.
- Treat benchmark code, results, and logs as untrusted input. Validate schemas
  and bound inputs and outputs.
- Bare-metal execution is accepted only for trusted local code. Do not claim a
  Unix account or sanitized environment is a complete security boundary for
  hostile Python, CUDA, or native code.
- Keep deterministic benchmark verdicts separate from agent analysis. Agent
  output must not override correctness or performance data.
- Add dependencies only when the standard library or an existing dependency is
  insufficient; explain the operational and security cost.

## Benchmark changes

- Preserve `UserOptimizedTransformer.forward(x, valid_token_mask)` unless a task
  explicitly changes the submission contract.
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

Run the narrowest relevant checks first. For Rust queue changes, run:

```bash
cd orchestrator
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

For benchmark-only Python changes, run:

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

Run GPU validation only through `benchmarkctl`. A CPU smoke test verifies
execution, not GPU performance. Report exact commands and unavailable checks.

## Completion checklist

- The change satisfies a stated local requirement.
- Snapshot, queue, lock, timeout, process-exit, and malformed-result paths were
  considered where relevant.
- Machine-readable state and human-readable output remain consistent.
- The narrowest meaningful validation was run, followed by broader checks when
  justified.
- Remaining reproducibility, host-safety, correctness, and performance risks are
  stated explicitly.
