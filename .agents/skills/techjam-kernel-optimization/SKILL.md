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

## Invest only with bottleneck evidence

Do not start parameter sweeps or tile/launch micro-tuning from intuition alone.
Before editing code or consuming a GPU queue slot, record:

- a measured profile or timing/materialization decomposition tied to the current
  historical-best snapshot and target shape;
- the bottleneck's estimated fraction of case latency;
- the part of that bottleneck the proposed change removes; and
- the resulting predicted end-to-end case-latency improvement.

Normally require a predicted 5--10% case improvement. Do not start a route whose
prediction is below 2%; predictions between 2% and 5% need unusually strong
cross-case leverage before they are worth an experiment. If the repository has
no supported profiling mode, do not bypass `benchmarkctl`; first report the
missing evidence and decide whether adding a bounded repository-supported mode
is justified.

Prefer changes that alter complexity, eliminate a large materialized tensor,
or fuse a whole execution segment. A new block size, warp count, or allowlist
entry is not an optimization hypothesis without profile evidence showing that
its affected work is large enough to clear the investment gate.

## Assign structural bottleneck owners

Organize parallel work by bottleneck rather than by individual case:

- Cases 6 and 13: one attention owner for online softmax, Flash-style
  attention, score/PV pipelining, and working-set restructuring.
- Cases 1, 4, 5, 7, 9, and 10: one whole-model owner for persistent or
  whole-layer fusion that removes launches and intermediate tensors.
- Cases 11 and 12: one owner for an execution-segment fusion large enough to
  cross the remaining target gap.
- Freeze Cases 2 and 3 unless a shared structural change needs regression
  validation. Treat Case 8 as a 1.3--1.5x closeout target rather than a 7x
  campaign.

Use realistic per-case targets: Cases 11/12 at 7.5--10x, Case 13 at 7--10x,
Case 6 at 5--8x, Cases 1/4/5/7 at 4--7x, and Cases 8/9/10 at approximately
1.5x/4x/5x. These targets guide investment; correctness and the official
harness remain unchanged.

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
7. Apply the campaign gates after correctness passes. A focused candidate gain
   below 3% stops. A 3--5% gain continues only when it spans multiple owned
   cases and negative controls are stable. A gain above 5% advances to a unified
   Cases 1--13 run. Otherwise retain the prior best and record the result.
8. Promote a unified candidate only when the geometric mean of candidate
   latency ratios versus the fixed historical-best snapshot improves by at
   least 1%, with paired-speedup evidence not contradicting attribution and no
   unacceptable declared-case regression. Do not use baseline drift to turn a
   sub-threshold candidate change into a promotion.

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
