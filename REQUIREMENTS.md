# Benchmark Harness Requirements

Status: Local MVP

Scope: TikTok TechJam 2026 Track 3 benchmark and the local agent-driven GPU
queue.

## 1. Purpose

The repository has one orchestration loop:

1. An agent changes or reviews the benchmark implementation.
2. The agent invokes `benchmarkctl`; it never starts the Python GPU benchmark
   directly.
3. `benchmarkctl` snapshots the current worktree and serializes execution on the
   local GPU.
4. The benchmark writes deterministic structured results and a log.
5. The same agent reads those artifacts, verifies numerical correctness first,
   and only then interprets performance or prepares the next commit.

Every run is an ordinary benchmark job. A new commit is validated by submitting
a new ordinary job for its worktree snapshot. There are no verification child
jobs or auxiliary orchestration services.

The competition's final official harness and test cases are the source of truth.
Do not silently weaken correctness thresholds, omit shapes, reduce repetitions,
or include compile or warmup time in steady-state results.

## 2. Local queue contract

The Rust `benchmarkctl` CLI shall:

1. Snapshot tracked and non-ignored untracked regular files when a job is
   submitted, so later worktree edits cannot change queued work. Reject symbolic
   links and submodules rather than preserving mutable references.
2. Record the base commit and a SHA-256 digest of the copied snapshot.
3. Queue jobs persistently in FIFO order, allow at most one benchmark process to
   hold the shared GPU lock, and reject submissions above four unfinished jobs
   as explicit backpressure.
4. Keep the GPU lock held by the benchmark wrapper itself, so killing the
   submitting agent cannot allow a second benchmark to overlap a surviving
   child process.
5. Run the fixed `torch_transformer_benchmark.py` entry point from a fresh
   per-job snapshot with the host's pinned `.venv`, a cleared environment, an
   explicit GPU selection, and a wall timeout of at most 3600 seconds.
6. Record the snapshot `uv.lock` digest, Python executable digest, Python
   version, and installed-package inventory digest at submission, then refuse
   execution if that identity changes before the job starts.
7. Own the `--json-output` path and retain `job.json`, `result.json`, and
   `benchmark.log` under the queue state directory.
8. Validate the benchmark result schema and require
   `correctness_passed: true` before marking a job successful. Agent output must
   never override this deterministic verdict.
9. Support asynchronous `submit` with an explicit Codex session ID or the
    inherited `CODEX_SESSION_ID`. Submission snapshots immediately and enters
    `awaiting_hook` without occupying the GPU.
10. Provide a project Codex `Stop` hook that claims the oldest submitted job for
    its session, runs it through the same FIFO/GPU lock, and returns a trusted
    continuation prompt containing job, result, and log paths. It shall not
    launch another Codex process.
11. Support `cancel <job-id>` for an unclaimed `awaiting_hook` job. Cancellation
    shall be serialized against Stop-hook claiming, retain the job artifacts,
    and refuse to stop a `queued` or `running` job.
12. Use `.benchmarkctl/` by default. Separate Git worktrees targeting the same
    GPU shall use the same absolute `BENCHMARKCTL_STATE_DIR`.

No source upload or network service is part of the queue.

## 3. Execution and trust model

- Local jobs are limited to code trusted to run under the current Unix account.
- Each job runs from its immutable snapshot with a fresh home and cache
  directory.
- The child environment is cleared and receives only the minimum runtime values,
  including the selected GPU.
- The benchmark process must not inherit Codex credentials, SSH agent variables,
  cloud credentials, or unrelated service secrets.
- Bare-metal Python and CUDA execution is not a security boundary for hostile
  code. Untrusted fork code must not run on a persistent local GPU host.
- Queue state and result artifacts remain local and are not an upload boundary.

## 4. Benchmark correctness and timing

The benchmark shall:

- Preserve `UserOptimizedTransformer.forward(x, valid_token_mask)` unless the
  competition submission contract changes explicitly.
- Give baseline and candidate identical weights, inputs, masks, dtype, and
  device configuration.
- Use fixed seeds and synchronize CUDA around measurements.
- Validate shape, dtype, finite values, absolute error, and relative error before
  accepting performance numbers.
- Accept an output element only when relative error is strictly `< 0.02` OR
  absolute error is strictly `< 0.002`.
- Skip performance scoring after correctness failure unless an explicit
  diagnostic mode is selected.
- Exclude data generation, compilation, and warmup from steady-state latency.
- Alternate baseline/candidate measurement order to reduce clock and thermal
  bias.
- Record raw samples and derive median, mean, p90, minimum, throughput, and
  speedup from those samples.
- Exit nonzero on correctness or execution failure and write the corresponding
  machine-readable failure state.

The declared official matrix contains the 14 cases in
`transformer_benchmark/cases.py`. The script must attempt the declared shapes
without shrinking or substituting them. Case 14 (`seq_len=100000`) may exceed
the available GPU memory; an actual CUDA OOM must be recorded as an execution
failure rather than converted into a smaller workload.

GPU performance claims require an actual target-GPU run over the full declared
matrix. A CPU smoke test verifies execution only.

## 5. Agent loop

Every benchmark uses asynchronous submission:

```bash
orchestrator/target/release/benchmarkctl submit -- \
  --device cuda:0 \
  --dtype bfloat16
```

After `submit`, the agent finishes its turn. The project `Stop` hook runs the
queued job and continues the same session with trusted artifact paths. On
continuation, the agent reads `result.json` and `benchmark.log`, checks
correctness, and reports performance only when correctness passed.

Additional evidence is obtained through another ordinary `benchmarkctl`
submission. An obsolete unclaimed submission may be cancelled with:

```bash
orchestrator/target/release/benchmarkctl cancel <job-id>
```

The system has no special verification job type, parent/child lineage, or
model-controlled job template.

## 6. Validation

For Rust queue changes, run:

```bash
cd orchestrator
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

For benchmark-only Python changes, run the narrow CPU smoke test:

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

## 7. Acceptance criteria

The local loop is accepted when:

1. Concurrent callers execute in persistent FIFO order with one GPU holder.
2. Killing a submitting agent cannot overlap a surviving benchmark child with a
   second job.
3. A queued job executes the exact snapshot and environment identity recorded at
   submission.
4. A successful process with a missing, malformed, incomplete, or incorrect
   result is not marked successful.
5. `submit` returns after snapshotting, and the Stop hook resumes the same agent
   session with trusted result and log paths.
6. Cancelling an `awaiting_hook` job prevents later hook claiming and releases
   unfinished-job capacity without deleting its snapshot.
7. A new commit is tested through a new ordinary benchmark job; no
   verification-child workflow is required.
8. No benchmark dispatch or execution step depends on GitHub Actions, a network
   service, or another Codex process.

## 8. Non-goals

The repository intentionally does not implement or retain:

- network services or external CI integrations for benchmark dispatch;
- source upload or external artifact storage;
- leases, heartbeats, service credentials, or distributed retries;
- Codex App Server orchestration;
- verification child jobs or agent-created queue entries outside
  `benchmarkctl`;
- a general-purpose CI or multi-tenant GPU scheduler.
