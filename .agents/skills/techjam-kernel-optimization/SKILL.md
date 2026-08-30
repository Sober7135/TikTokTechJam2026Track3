---
name: techjam-kernel-optimization
description: Optimize the candidate Transformer implementation in this repository through correctness-first, evidence-driven iterations. Use when implementing, profiling, comparing, or refining GPU performance changes; use techjam-local-benchmark instead for benchmark submission or result review without optimization work.
---

# TechJam kernel optimization

Improve the candidate against the unchanged benchmark contract. This skill
organizes hypotheses, code changes, and measured feedback; it does not add or
replace a GPU execution path.

## Establish the contract

Before changing the candidate, read the root `AGENTS.md`, `REQUIREMENTS.md`,
`pyproject.toml`, `torch_transformer_benchmark.py`, and the relevant modules
under `transformer_benchmark/`. Check the working tree and preserve unrelated
changes.

Treat the official harness as authoritative. In particular:

- Preserve `UserOptimizedTransformer.forward(x, valid_token_mask)` and output
  shape `[B, S, D]` unless the task explicitly changes the submission contract.
- Compare identical weights, inputs, masks, dtype, device configuration, and
  benchmark settings.
- Require finite outputs with matching shape and dtype. Each element passes
  only when `abs_error < 0.002 OR abs_error < 0.02 * abs(reference)`; both
  inequalities are strict.
- Keep every declared shape and a correct fallback. Do not shrink, substitute,
  or silently skip case 14.
- Do not use performance from an incorrect candidate.

Reject shortcuts that mutate inputs, reuse reference computations or cached
outputs, hide work on an unsynchronized stream, specialize to undeclared input
values, weaken trials or tolerances, or move compilation/warmup into the timed
region.

## Run an optimization iteration

1. Identify the exact target cases and the best comparable snapshot. Reuse
   existing measurements only when commit or snapshot, arguments, dtype,
   software environment, and hardware profile match.
2. State one evidence-backed hypothesis: the suspected bottleneck, the proposed
   change, the cases it should help, its measurable effect, and the fallback or
   rollback condition. Keep the implementation route open until evidence
   supports PyTorch, Triton, TileLang, CUDA/CUTLASS, or another allowed path.
3. Implement one coherent optimization category. Closely coupled edits may be
   grouped, but do not mix unrelated experiments that cannot be attributed.
4. Run the narrowest non-GPU checks first. Add a dependency only after reading
   `pyproject.toml` and explaining its build, runtime, reproducibility, and host
   cost.
5. For every local GPU diagnostic or performance run, invoke
   `$techjam-local-benchmark` and submit an ordinary immutable snapshot with
   `benchmarkctl submit`. Finish the turn after submission so the Stop hook can
   run it. Never run the Python GPU benchmark, NCU, or a standalone kernel
   benchmark directly.
6. On continuation, let `$techjam-local-benchmark` verify the job and structured
   result. Check numerical correctness before reading latency. Compare raw
   samples and relevant per-case results, not a single best timing or
   human-readable log summary.
7. Promote a candidate to the historical best only when correctness passes and
   the measured target improves without an unacceptable declared-case
   regression. Otherwise retain the prior best and record the failure or noisy
   result as evidence.

If deeper profiling is justified, it must still execute inside an ordinary
`benchmarkctl` job using a repository-supported benchmark mode. Do not create a
profiler side path or second orchestrator. If no such mode exists, report the
missing evidence and ask before expanding the harness.

## Preserve experiment history

Create one independent Git commit for each evaluable optimization. Do not mix
unrelated optimizations, shared-winner seeding, benchmark-control changes, or
contract repairs into that commit. A rejected experiment may remain committed
when preserving it helps prevent the same route from being retried; label its
outcome clearly.

The optimization commit body must record:

- target cases and the measured or hypothesized bottleneck;
- the implementation and why its numerical-equivalence boundary is valid;
- exact dispatch conditions and the correct fallback;
- relevant static, test, and CPU validation;
- the ordinary GPU benchmark job ID, snapshot digest, correctness result, raw
  timing summary, and promote/retain/reject decision when available.

It is acceptable to create the commit before GPU execution and amend only its
message after the deterministic result arrives. Commit signing is not required;
do not enable or require GPG or SSH signing for this workflow.

## Feed evidence into the next iteration

For a multi-iteration campaign, read
[references/feedback-loop.md](references/feedback-loop.md). Feed the next
decision the contract, current implementation, historical best, latest
generation, compiler/execution diagnostics, correctness result, raw timings,
and validated profiler evidence when available. Prefer workload-specific
examples and measured RTX 4070 behavior over generic GPU advice or parameters
copied from a different architecture.

Stop when the requested target or budget is reached, no new evidence-backed
hypothesis remains, the same route repeatedly fails its rollback condition, or
an external decision is required. Report the best verified snapshot and the
remaining uncertainty; do not equate a CPU smoke test or partial-case run with
official target-GPU performance.

## Worktree and queue isolation

Do not repurpose a dirty or unrelated worktree. When separate worktrees target
the same GPU, use the same absolute `BENCHMARKCTL_STATE_DIR`. A worktree changes
source isolation only; `benchmarkctl` remains the sole GPU serialization and
execution mechanism.
