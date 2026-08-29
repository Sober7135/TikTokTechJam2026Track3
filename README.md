# TikTok TechJam 2026 Track 3

This repository keeps the GPU benchmark surface separate from its local Rust
queue and process supervisor.

## GPU benchmark

- `torch_transformer_benchmark.py`: stable CLI entry point for local benchmark
  invocations.
- `transformer_benchmark/`: benchmark implementation, split into case/input,
  model, correctness, timing, and runner modules. Candidate implementations
  belong in `transformer_benchmark/models.py`.
- `pyproject.toml` and `uv.lock`: the pinned Python 3.12 benchmark environment.
- `.venv/`: the host benchmark environment created once with `uv sync --frozen`.
  Jobs reuse it without installing dependencies.

Small CPU smoke test:

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

The declared competition matrix is the default GPU run:

```bash
uv run python torch_transformer_benchmark.py \
  --device cuda:0 \
  --dtype bfloat16 \
  --json-output result.json
```

The benchmark defaults to BF16 for inputs, weights, baseline, and candidate.
Pass `--dtype float32` explicitly when a full-precision reference run is needed.

Use `--official-cases 2 7 12` to run a selected subset while developing. Use
`--no-official-matrix` for one custom shape. Matrix output uses schema version 2
and records every case separately, including raw timing samples. The script
never shrinks an official shape. An actual CUDA out-of-memory error is recorded
as `execution_failed` with `failure_category: out_of_memory`; the process exits
with status 3.

### Accuracy rule

Baseline and candidate receive identical weights, inputs, masks, dtype, and
device configuration. For every output element, the script computes
`abs_error = abs(candidate - baseline)` and accepts it when both values are
finite and either:

- `abs_error < atol`; or
- `abs_error < rtol * abs(baseline)`.

The defaults are `atol=0.002` and `rtol=0.02`. Every element in every accuracy
trial must pass before performance is accepted. Relative error reported near a
zero baseline is clamped only for summary display; the pass/fail rule above
uses the unclamped baseline value.

## Local benchmark queue

Use `benchmarkctl` whenever an agent runs a benchmark. It snapshots tracked and
non-ignored untracked files at submission time, queues jobs in FIFO order, and
holds an exclusive cross-process GPU lock for the complete benchmark process.
The host `.venv` is reused; jobs never install dependencies. Snapshots reject
symbolic links and submodules. Submission records the `uv.lock`, Python binary,
version, and package-inventory digests, and execution refuses an environment
that changed while the job was queued.

```bash
cd orchestrator
cargo build --release -p benchmarkctl
cd ..
orchestrator/target/release/benchmarkctl submit -- \
  --device cuda:0 \
  --dtype bfloat16
```

The official matrix is already the benchmark default. Arguments after `--` are
passed to `torch_transformer_benchmark.py`; `benchmarkctl` owns
`--json-output`. Each job is stored under `.benchmarkctl/jobs/<job-id>/` with:

- `source/`: the exact local snapshot used for execution;
- `job.json`: queue state, base commit, snapshot digest, arguments, and timing;
- `result.json`: the benchmark's structured result;
- `benchmark.log`: captured stdout and stderr.

`submit` reads `CODEX_SESSION_ID` automatically; `--session-id <id>` is
available for explicit callers. It snapshots the source and returns immediately
with state `awaiting_hook`. The project `Stop` hook in `.codex/hooks.json` then
claims that session's oldest submitted job, waits through the shared FIFO queue,
and runs it. On completion the hook returns a Codex continuation prompt with the
exact result and log paths. The same agent verifies the deterministic result;
no second `codex exec resume` process is launched. Codex asks you to trust the
project hook the first time its definition is used.

To keep the Stop hook's queue wait bounded, `benchmarkctl` accepts at most four
unfinished jobs and caps each job timeout at 3600 seconds. A fifth submission is
rejected with backpressure and can be retried after an earlier job finishes.

Inspect prior jobs with:

```bash
orchestrator/target/release/benchmarkctl list
orchestrator/target/release/benchmarkctl show <job-id>
orchestrator/target/release/benchmarkctl cancel <job-id>
```

`cancel` releases queue capacity by moving an unclaimed `awaiting_hook` job to
`cancelled`. It retains the snapshot and job record for auditability and refuses
to cancel a job after the Stop hook has claimed it.

The default `.benchmarkctl/` queue is shared by agents using this worktree. If
agents use separate Git worktrees, set the same absolute
`BENCHMARKCTL_STATE_DIR` for all of them.

This local agent-plus-`benchmarkctl` loop is the complete orchestration model.
It has no auxiliary service, source upload, or special verification job type.
Each new commit is validated through another ordinary `benchmarkctl` job. See
[`orchestrator/README.md`](orchestrator/README.md) and
[`REQUIREMENTS.md`](REQUIREMENTS.md) for the complete contract.
