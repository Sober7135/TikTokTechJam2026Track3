# Measured optimization feedback loop

Use this reference for campaigns with more than one candidate iteration. It
adapts KernelBench's previous-generation plus execution plus profiling feedback
pattern to this repository's deterministic harness.

## Initial packet

Give the optimizer only the information that changes its decisions:

- immutable interface, output, correctness, timing, and case requirements;
- relevant baseline and candidate source;
- target cases and the reason they were selected;
- exact GPU, architecture, dtype, PyTorch/CUDA versions, and dependency limits;
- comparable baseline and current-best snapshot identifiers;
- raw per-case timing summaries and samples already measured on that profile;
- measured bottleneck share, the removable fraction, and the predicted
  end-to-end case-latency improvement;
- allowed implementation freedom and an explicit experiment budget, if one was
  supplied.

Do not anchor the first decision on a preferred technique unless repository
evidence already supports it. Hardware specifications and kernel parameters
from another GPU are hypotheses, not facts for the RTX 4070.

## Follow-up packet

After each completed ordinary benchmark job, provide:

```text
Contract and target cases:
<unchanged concise contract>

Historical best:
<commit or snapshot, implementation summary, correctness, raw timings>

Latest candidate:
<snapshot and focused diff or implementation summary>

Deterministic evaluation:
<job state, requested and completed cases, correctness verdict>

Compiler or execution diagnostics:
<bounded exact errors, or none>

Performance evidence, only when correct:
<raw samples and derived summaries for baseline and candidate>

Validated profiler evidence, when available:
<metrics tied to the same implementation, shape, dtype, and GPU>

Investment gate:
<bottleneck latency share * removable fraction = predicted case improvement;
normally 5-10%, and never start below 2%>

Decision requested:
Choose one next hypothesis, its expected measurable effect, and rollback rule.
Do not repeat a disproven route without new evidence.
```

The structured `result.json` verdict outranks model analysis and log prose.
Include only bounded relevant log sections. Keep a compact history of rejected
routes so later iterations do not rediscover them, but avoid pasting every old
generation into every prompt.

## Candidate decision

Classify every evaluated candidate as one of:

- `promote`: correctness passed and the declared target improved with acceptable
  regression risk;
- `retain-best`: correctness passed, but improvement is absent, noisy, or offset
  by a material regression;
- `reject-incorrect`: shape, dtype, finite-value, or strict error validation
  failed;
- `reject-execution`: compilation, timeout, OOM, malformed result, or another
  execution failure prevented a valid comparison.

Record the snapshot digest, exact arguments, environment identity, raw samples,
decision, and reason. A rerun for timing noise is another ordinary job, not a
special verification job.

For this campaign, apply the promotion thresholds after correctness:

- focused gain below 3%: stop;
- 3--5%: continue only for a multi-case owner with stable negative controls;
- above 5%: advance to unified Cases 1--13;
- unified candidate-latency geometric-mean improvement below 1% versus the
  fixed historical best: do not enter the winner.

Report paired speedup as an attribution check, but calculate the primary
promotion aggregate from candidate latencies so baseline timing drift cannot
manufacture a gain.

## Profiling gate

Request profiling only when timing evidence cannot distinguish plausible next
steps and the expected information can change the implementation decision.
Profile the smallest representative declared case that exhibits the bottleneck,
then confirm any resulting optimization through the normal correctness and
timing harness. Profiler output never substitutes for end-to-end benchmark
results.

## Pattern sources

The feedback packet is adapted to this repository from KernelBench's measured
iteration pattern, not from its standalone executor:

- prompt composition: <https://github.com/ScalingIntelligence/KernelBench/blob/main/src/kernelbench/prompts/prompts.toml>
- previous-generation plus execution and profiler feedback: <https://arxiv.org/html/2502.10517v1#S11.SS3>
