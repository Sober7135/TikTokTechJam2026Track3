# AI-Driven GPU Optimization for Exact Transformer Inference

**TikTok TechJam 2026 — Track 3**

**Sole entrant:** Jinye Wu<br>
**Repository:** [github.com/Sober7135/TikTokTechJam2026Track3](https://github.com/Sober7135/TikTokTechJam2026Track3)<br>
**Demo:** [youtu.be/_yS3vhezGsk](https://youtu.be/_yS3vhezGsk)<br>
**Target platform:** NVIDIA GeForce RTX 4070 Desktop (`sm_89`)

**Evaluation dtype:** BF16<br>
**Validated performance scope:** Track 3 Cases 1–13<br>
**Declared but hardware-infeasible case:** Case 14, reported separately

## Executive Summary

This project used AI coding agents to optimize the Transformer inference
workload defined by TikTok TechJam 2026 Track 3. The main contribution is not a
single hand-selected kernel. It is a complete AI optimization system in which
agents can propose, implement, benchmark, reject, and integrate GPU
optimizations while preserving numerical correctness and producing an
auditable evidence trail.

To make that process reliable, we built a local GPU control plane around
`benchmarkctl`, connected it to the agent lifecycle through a Codex Stop hook,
and encoded the optimization and evaluation policy in project-specific skills.
Multiple agents could research and implement ideas in isolated Git worktrees,
while all GPU jobs were serialized through one immutable-snapshot queue. Each
experiment therefore had a reproducible source snapshot, environment identity,
structured correctness result, raw timing samples, and an explicit
promote/retain/reject decision.

The final validated candidate achieved the following results over Cases 1–13:

- **3.900295x geometric-mean speedup** across equally weighted cases;
- **5.367318x reduction in aggregate latency**, from `555.428304 ms` to
  `103.483388 ms`;
- **supplemental dense-equivalent MFU improvement from 6.24% to 33.47%**;
- **65/65 accuracy trials passed bitwise exactly**;
- **0 failed elements out of 938,885,120 evaluated outputs**;
- maximum absolute error and maximum relative error both equal to zero.

The optimization portfolio covered the entire matrix rather than only the two
largest cases. It included CUDA Graph replay, exact projection/GELU fusion,
projection-residual epilogues, packed and direct-layout QKV, direct-write PV
kernels, causal key-prefix reduction, exact native-order softmax, shared-CTA
attention, and an exact multi-row LayerNorm specialization. Cases 6 and 13
receive deeper treatment because together they represented approximately 94.5%
of baseline aggregate latency, not because the remaining cases were ignored.

Using a dense-equivalent Transformer FLOP model and a `58.25 TFLOP/s` BF16
reference peak, the Cases 1–13 suite has an idealized compute time of
`34.638543 ms`. The current candidate is therefore `2.9875x` above this
idealized reference. This is a roofline reference rather than a promised
remaining speedup: softmax, LayerNorm, synchronization, memory traffic, and
launch overhead prevent full tensor-core utilization, while causal pruning can
also eliminate operations counted by the dense-equivalent numerator.

The final validated source is commit
`803ea145c796702357af4b1b75528dd701fa472a`, with Git tree
`a456810a8d6b139f3c3959816865c3bcb3278a65`. The final unified result is
`job-1788222492773-5dd44f6e492d883e`.

Case 14 remains part of the declared benchmark matrix and was not silently
reduced or substituted. Its original `S=100000` contract is not included in
the Cases 1–13 speedup or supplemental MFU claims. An ordinary immutable
benchmark job attempted the exact shape and recorded a CUDA out-of-memory
failure. Section 8 gives both the measured failure and the memory lower bound.

## 1. Objective and Benchmark Contract

The validated Cases 1–13 use a four-layer Transformer inference implementation
with shape variations in batch size, sequence length, model width, head count,
and FFN width. Case 14 uses two layers and is discussed separately because it
does not complete under the current local hardware contract. The public
submission interface remains unchanged:

```python
forward(x, valid_token_mask) -> Tensor[B, S, D]
```

AI-driven optimization is useful only if the search cannot improve its score
by changing the problem. The following invariants were therefore treated as
hard constraints:

- baseline and candidate receive identical weights, inputs, masks, dtype, and
  device configuration;
- random inputs use fixed seeds;
- official shapes are not reduced, replaced, or omitted silently;
- data generation, extension compilation, and warm-up are excluded from
  steady-state timing;
- CUDA is synchronized around measurements;
- output shape, dtype, and finite values are validated before performance is
  interpreted;
- the baseline implementation and weight-copying behavior remain unchanged;
- every shape-specific optimization retains a correct fallback;
- an incorrect candidate has no valid performance result.

### 1.1 Numerical Validation Rule

The Track 3 problem statement gives two error limits: relative error below
`0.02` and absolute error below `0.002`. The initially supplied Torch script
expressed its elementwise acceptance check as absolute **or** relative error,
with non-strict comparison and tighter defaults. To avoid weakening either
source, our local harness uses finite values and the following strict OR gate:

```text
absolute_error < 0.002
OR
absolute_error < 0.02 * abs(reference)
```

The final candidate did not rely on any interpretation of the tolerance. Every
evaluated output element in Cases 1–13 was bitwise identical to the reference,
so it passes strict or non-strict and AND or OR readings of the stated limits.

### 1.2 Alignment with the Track 3 Brief

We selected the PyTorch benchmark path, as permitted by the brief. The final
solution uses the in-scope methods explicitly contemplated by Track 3:
shape-specialized dispatch, GPU kernel fusion, memory-layout optimization,
BF16 and tensor-core execution, softmax optimization, Triton kernels, and
custom CUDA extensions. Production deployment is intentionally out of scope.

The brief also encourages AI-assisted optimization and requests details of the
AI tools and skills used. Section 2 therefore treats the agent infrastructure
as a first-class technical contribution rather than presenting only the final
CUDA kernels.

### 1.3 Performance Metrics

We use three complementary metrics.

For each case:

```text
case_speedup = baseline_median_latency / candidate_median_latency
```

The official-matrix summary gives every case equal weight:

```text
speedup_GM = geometric_mean(case_speedup)
```

We also report aggregate latency:

```text
aggregate_speedup = sum(baseline_latency) / sum(candidate_latency)
```

The geometric mean describes consistency across shapes. The aggregate ratio
describes total workload throughput and is dominated by the most expensive
cases. Dense-equivalent MFU is derived from the same aggregate latency and is
defined explicitly in Section 6. MFU is a supplemental analysis in this report,
not a metric specified by the Track 3 judging brief.

When deciding whether a new candidate should replace an existing winner, we do
not rely on baseline timing drift. The primary incremental metric is:

```text
incremental_gain =
    1 - geometric_mean(new_candidate_latency / winner_candidate_latency)
```

The last integrated change improved candidate-latency geometric mean by
`1.981750%` over the previous unified winner.

## 2. Building an AI Optimization System

GPU kernel optimization requires many short hypothesis-and-measurement cycles.
An AI agent can generate and implement those hypotheses quickly, but local GPU
execution introduces three control problems:

1. concurrent agents can interfere with one another on a shared GPU;
2. queued source can change before execution;
3. a language-model interpretation must never override deterministic
   correctness evidence.

We addressed these problems with a small local control plane rather than a
network service or external CI system.

### 2.1 Division of Responsibility

The project used AI for the majority of optimization research and
implementation, with explicit separation between policy, engineering, and
verdict:

| Layer | Responsibility |
|---|---|
| Human supervisor | Defined the competition goal, correctness contract, resource budget, and strategic priorities |
| Supervisor agent | Decomposed structural bottlenecks, assigned independent investigations, compared evidence, and integrated winners |
| Optimization agents | Inspected source and profiles, formed hypotheses, implemented kernels, ran checks, and interpreted deterministic results |
| Benchmark harness and `benchmarkctl` | Produced immutable execution identity, numerical verdicts, raw timings, and GPU serialization |

The human did not need to select every tile size or write every kernel, while
the AI was not allowed to decide that an incorrect result was acceptable. This
division made the search fast without making the benchmark verdict subjective.

### 2.2 AI Tools, Development Stack, and Assets

| Category | Tool or resource | How it was used |
|---|---|---|
| AI coding system | OpenAI Codex | Supervisor and worker agents analyzed bottlenecks, generated and reviewed implementations, interpreted deterministic evidence, and documented experiments |
| Agent policy | Repository-local Codex skills | Encoded correctness gates, benchmark submission rules, promotion criteria, fallback requirements, and experiment reporting |
| Agent lifecycle | Codex Stop hook | Resumed the same agent after an asynchronous GPU job completed |
| Source isolation | Git branches and worktrees | Allowed independent optimization directions without contaminating the current winner |
| GPU control plane | Rust and Cargo, through `benchmarkctl` | Snapshotted source, serialized the shared GPU, isolated the process environment, and retained results |
| Benchmark runtime | Python 3.12, PyTorch, NumPy, and `uv` | Implemented the official PyTorch path, reproducible dependency environment, correctness checks, and timing |
| GPU kernels | Triton and PyTorch C++/CUDA extensions built with `nvcc` | Implemented fused projections, layout-specialized kernels, attention, softmax, and LayerNorm |
| Development environment | NixOS/Nix, shell tools, PTX and SASS inspection | Supplied a reproducible compiler/runtime environment and low-level validation |

No external training dataset was used. Inputs, weights, masks, and seeds come
from the supplied benchmark contract and the repository's deterministic case
generator. The only external software assets are authorized open-source tools
and libraries in the pinned environment; the implementation does not use a
third-party model checkpoint, media asset, or proprietary dataset.

### 2.3 `benchmarkctl`: Immutable and Serialized GPU Evaluation

`benchmarkctl` is a Rust CLI that is the only local entry point for GPU
benchmark execution. On submission it immediately snapshots tracked and
non-ignored untracked regular files. Later edits in the worktree cannot change
the queued job.

For each job it records:

- base Git commit and snapshot SHA-256;
- `uv.lock` digest;
- Python executable digest and version;
- installed-package inventory digest;
- benchmark arguments and requested cases;
- GPU device, timeout, process exit, and terminal state;
- structured `result.json` and human-readable `benchmark.log`.

Jobs enter a persistent FIFO queue. The benchmark wrapper holds an exclusive
GPU lock for the complete child-process lifetime, so terminating the submitting
agent cannot cause a surviving benchmark to overlap another job. Queue
backpressure limits unfinished jobs instead of allowing an unbounded set of
experiments to consume host resources.

Every job runs from a fresh snapshot with a cleared environment, explicit GPU
selection, and isolated home/cache directories. Codex credentials, SSH agent
variables, cloud credentials, and unrelated service variables are not passed
to benchmark code. Before execution, `benchmarkctl` verifies that the pinned
Python and dependency identity still match the submission record.

Most importantly, success is determined by the structured benchmark result.
A zero process exit with a missing, malformed, incomplete, or incorrect result
is not accepted as a successful performance experiment.

### 2.4 Stop Hook: Asynchronous Agent Continuation

GPU experiments are submitted asynchronously:

```bash
orchestrator/target/release/benchmarkctl submit -- \
  --device cuda:0 \
  --dtype bfloat16
```

The agent finishes its turn after submission. A project-level Codex Stop hook
then claims the oldest job for that session, executes it through the same FIFO
queue and GPU lock, and resumes the same agent with trusted paths to
`job.json`, `result.json`, and `benchmark.log`.

This produces the following closed loop:

```text
AI agent proposes one optimization
              ↓
implements a small attributable change
              ↓
runs static checks and CPU fallback smoke
              ↓
submits an immutable snapshot with benchmarkctl
              ↓
agent turn ends; the GPU job runs asynchronously
              ↓
Stop hook resumes the same session
              ↓
agent verifies correctness before reading latency
              ↓
promote, reject, or refine
```

The Stop hook avoids synchronous GPU waiting and avoids polling, while keeping
the evidence attached to the agent that proposed the experiment. It does not
launch another agent process and does not create a second orchestration path.

### 2.5 Project Skills as Optimization Policy

Two repository-local skills convert project knowledge into reusable agent
behavior.

`techjam-kernel-optimization` defines the search policy:

- begin with measured bottleneck evidence;
- predict the end-to-end effect before consuming a GPU job;
- change one attributable optimization category at a time;
- preserve exact dispatch conditions and a native fallback;
- use focused targets and negative controls first;
- promote only after a unified Cases 1–13 run;
- record accepted and rejected experiments.

`techjam-local-benchmark` defines the evaluation policy:

- never run the Python GPU benchmark directly;
- submit ordinary immutable jobs through `benchmarkctl`;
- inspect structured results before logs;
- require every requested case and correctness trial to pass;
- interpret raw timing only after correctness;
- distinguish focused diagnostics from full-matrix evidence.

The skills act as an operating manual for the AI system. They reduce repeated
prompting, prevent an agent from forgetting the numerical contract during a
long campaign, and make later agents aware of rejected routes and promotion
criteria.

### 2.6 Parallel Research, Isolated Worktrees, One GPU Queue

Optimization work was divided by structural bottleneck rather than by asking
every agent to tune the same kernel. Agents explored attention, projection and
FFN fusion, layout removal, shape-specific execution, and difficult cases in
independent Git worktrees.

Worktrees isolated source changes, while all worktrees shared the same absolute
`BENCHMARKCTL_STATE_DIR`. Research and implementation could therefore proceed
in parallel, but GPU measurements remained serial and comparable.

Each evaluable optimization was represented by one Git commit. Commit bodies
recorded:

- target cases and bottleneck hypothesis;
- implementation and numerical-equivalence argument;
- exact dispatch and fallback conditions;
- static and CPU checks;
- benchmark job and snapshot identity;
- correctness and raw timing summary;
- promote, retain, or reject decision.

Commit signing was intentionally not required for this local experimental
workflow. The immutable snapshot digest and deterministic benchmark artifacts
provided the execution identity.

### 2.7 Correctness-Gated Search Strategy

The system did not perform unrestricted tile sweeps. Each iteration needed a
measured bottleneck, an estimated affected fraction, a predicted end-to-end
gain, and a rollback rule.

The normal decision sequence was:

1. static/source/resource validation;
2. unit tests and CPU fallback smoke;
3. focused GPU cases with negative controls;
4. strict correctness gate;
5. candidate-latency and raw-sample analysis;
6. unified Cases 1–13 validation;
7. integration only when the unified candidate improved the historical best.

This process is important to interpreting the final result. The final tree is
not a collection of the best number ever observed for each case. It is one
source snapshot that passed the entire unified matrix.

## 3. Optimization Portfolio

The final candidate combines shared optimizations with shape-specific kernels.
The strategy was to first remove common launch and memory overhead, then invest
deeper engineering effort in cases with large absolute latency.

### 3.1 Shared Optimization Families

#### CUDA Graph replay

Small and medium cases execute many short kernels across four Transformer
layers. Capturing the stable inference graph amortized Python and launch
overhead, especially for Cases 2, 3, 11, and 12. Graph replay retains an
independent output clone so the benchmark does not observe aliased or stale
storage.

#### Exact FFN input fusion

For selected width-128 shapes, the first FFN projection and exact-erf GELU were
fused. The dot product and bias accumulate in FP32, the projection is explicitly
rounded to BF16 at the same point as `nn.Linear`, and exact GELU is evaluated
from that rounded value. This removes an intermediate BF16 write/read pair and
a separate GELU launch without replacing exact GELU with a tanh approximation.

#### Projection-residual epilogues

Attention-output and FFN-output projections were specialized so their
epilogues consume the BF16 residual directly. The projection is still rounded
to BF16 before residual addition, preserving the original materialization
boundary. An earlier `beta*C` shortcut that skipped this boundary was not used.

#### Packed and direct-layout QKV

The candidate reduces redundant QKV launch and layout work. For supported
shapes, QKV is projected into a head-oriented layout, and packed V can be
consumed through explicit strides instead of copied into a second tensor.
Unsupported modes and layouts retain the native projection path.

#### Direct-write PV and context layouts

HD8 and HD32 PV kernels write directly into sequence-major backing storage.
This removes combinations of probability casts, `torch.cat`, transpose, and
contiguous-copy kernels while retaining BF16 inputs, FP32 accumulation, BF16
context output, and the same native FP32 softmax boundary.

#### Causal prefix reduction

Several exact shapes avoid processing future keys whose masked probabilities
are structurally zero. The candidate preserves QK arithmetic and native FP32
softmax for the live prefix while reducing score, softmax, and PV work that
cannot affect causal output.

### 3.2 Per-Case Optimization Map

The following table summarizes the principal retained paths. It is not intended
to list every low-level dispatch condition.

| Case | Principal retained optimization path | Final speedup |
|---:|---|---:|
| 1 | CUDA Graph, exact FFN fusion, projection-residual fusion, direct-write HD32 prefix PV | 2.461x |
| 2 | Small-shape CUDA Graph and launch amortization; custom FFN/PV variants that regressed were not retained | 9.501x |
| 3 | CUDA Graph and projection-residual epilogues | 7.368x |
| 4 | CUDA Graph, exact FFN fusion, full HD32 PV, projection-residual fusion, causal key-prefix reduction | 4.641x |
| 5 | CUDA Graph, width-128 fusion, direct-write HD32 prefix PV | 2.753x |
| 6 | Packed V, large-shape projection/FFN epilogues, exact shared-CTA attention, exact four-row LayerNorm | 5.481x |
| 7 | CUDA Graph and Case-7 HD8 PV written directly into final context layout | 2.404x |
| 8 | CUDA Graph and cached cuBLASLt FFN output; large D=1024 GEMMs remain dominant | 1.066x |
| 9 | CUDA Graph, shared width-128 fusion, causal key-prefix reduction | 1.784x |
| 10 | CUDA Graph, packed-V consumption, shared projection and FFN paths | 2.217x |
| 11 | CUDA Graph, HD8 PV specialization, compact QK prefixes, Case-11 QKV launch tuning | 7.079x |
| 12 | CUDA Graph, exact FFN fusion, full HD32 PV, projection-residual fusion | 6.127x |
| 13 | CUDA Graph, native-order softmax output, exact shared-CTA attention, causal QK fragment elimination | 9.115x |

### 3.3 Why Cases 6 and 13 Received Deeper Specialization

The baseline latency of Case 6 was `414.119934 ms`, and Case 13 was
`110.874306 ms`. Together they contributed approximately 94.5% of the
`555.428304 ms` baseline latency sum. Improving those two cases had much more
effect on aggregate throughput and MFU than another sub-microsecond change to a
small case.

This prioritization complements rather than replaces the equal-case geometric
mean. The final report therefore includes the complete matrix while using Cases
6 and 13 as detailed examples of the deepest AI-guided kernel work.

### 3.4 Exact Shared-CTA Attention

For Cases 6 and 13, the final attention path fuses exact QK, native-order
softmax, the required BF16 probability boundary, and PV within a shared-memory
CTA:

```text
QK MMA -> native-order softmax -> BF16 probability -> PV MMA
```

The implementation preserves the official MMA fragment mapping, increasing
K16 accumulator chains, native lane/XOR reduction order, and BF16 rounding
boundaries. Executable lane-coordinate oracles were used because matching HMMA
instruction counts alone was not sufficient to prove logical fragment
ownership.

An early Case-6 kernel failed because two packed A-fragment registers were
reversed. The oracle identified the correct logical order:

```text
row0/K0, row8/K0, row0/K8, row8/K8
```

After correction, the exact fused Case-6 attention substantially reduced
global intermediates while remaining bitwise identical.

Case 13 uses `S=1024`. Its shared-CTA path keeps an M16 score/probability tile
in shared memory, stages the query tile once, and eliminates global score and
probability intermediates. A later optimization skips an M16xN8 QK fragment
when the first key position is greater than the last query position in the
query block. Every value in that fragment is strictly above the causal
diagonal and would become negative infinity. The optimization skips 18.75% of
the Case-13 QK fragments while storing the identical masked representation.

### 3.5 Native-Order Softmax with Direct BF16 Output

The original path produced FP32 softmax probabilities, stored them globally,
and then converted them to BF16 before PV. The optimized path instantiates the
same native softmax schedule but stores the already-required BF16 value
directly:

```text
Before: FP32 softmax -> FP32 global store -> BF16 conversion -> PV
After:  FP32 softmax -> BF16 global store -> PV
```

Lane ownership, max/sum reduction order, exponential, division, and NaN
behavior remain unchanged. This halves global probability storage and removes
a conversion launch without changing the BF16 values observed by PV.

### 3.6 Exact Four-Row LayerNorm for Case 6

Profiling showed that Case 6 spent approximately `31.669290 ms` in nine native
BF16 width-128 LayerNorm calls. PyTorch's vectorized kernel launches four warps
per row, but width 128 with vector width four gives only 32 live lane vectors.
The first warp owns the input and output data while the remaining warps
participate in control and identity-state merging.

The specialized kernel assigns one independent row to each warp:

```text
Native:      one CTA -> one row  -> four warps, one data-owning warp
Specialized: one CTA -> four rows -> four warps, four useful warps
```

Each warp preserves the native online Welford updates, shuffle order, FP32
variance division, reciprocal square root, affine operation, and BF16 rounding.
The removed inter-warp merges combine the live state only with zero-count
identity states for this exact width. An executable Welford-state oracle and
source-order checks verified that the transformation is bitwise neutral for
the official finite inputs.

The specialization reduced focused Case-6 latency by approximately 19%, from
`93.314049 ms` to `75.597824 ms`, before final unified validation measured
`75.557884 ms`.

### 3.7 Case 8: A Deliberate Closeout Rather Than a Forced Win

Case 8 uses `D=1024` and `FFN=1024`. Its execution is dominated by large vendor
GEMMs rather than only launch and layout overhead. The final path retains a
cached cuBLASLt FFN output projection, but several additional cuBLASLt,
compiled-softmax, final-normalization, and compact-prefix experiments either
failed strict correctness or produced gains that did not reproduce in unified
validation.

The final speedup is `1.066x`. Retaining this result is preferable to combining
non-reproducible focused measurements into an artificial winner. Case 8 also
demonstrates an important property of the AI loop: stopping and retaining the
historical best is a valid outcome.

## 4. Correctness and Verification

### 4.1 Exact Dispatch and Fallback

Every custom route is guarded by the exact workload, runtime, device, dtype,
layout, inference, alignment, and mask requirements that were validated.
Unsupported shapes, training, gradients, CPU execution, non-BF16 inputs,
effective masks, and incompatible layouts use a baseline-compatible native
fallback.

### 4.2 Executable Oracles

Workload-specific checks cover:

- MMA lane and register mapping;
- fragment ownership and coordinate coverage;
- causal-mask safety and disjoint skipped/live tiles;
- absence of overlapping writes;
- Welford state construction and merge order;
- BF16 and FP16 rounding boundaries.

### 4.3 PTX and SASS Inspection

Static compilation checks verified expected MMA dependencies and instruction
counts, register and shared-memory use, shuffle/barrier structure, and absence
of local-memory spills for promoted kernels. Static inspection was treated as
a precondition, never as a replacement for end-to-end correctness.

### 4.4 Evaluation Environment

| Component | Configuration |
|---|---|
| CPU | AMD Ryzen 9 7950X, 16 cores / 32 threads |
| System memory | 61 GiB usable RAM |
| GPU | NVIDIA GeForce RTX 4070 Desktop, 12,282 MiB reported memory |
| GPU architecture | Ada Lovelace, `sm_89` |
| NVIDIA driver | 595.91.07 |
| Storage devices | ZHITAI TiPlus7100 4 TB NVMe and 2 TB NVMe; ST4000VX015 4 TB HDD |
| Benchmark filesystem | Encrypted Btrfs `/home`; approximately 3.5 TiB available during final audit |
| Operating system | Linux 7.2.0, glibc 2.42 |
| Python | 3.12.14 |
| PyTorch | `2.13.0+cu130` |
| CUDA runtime | 13.0 |
| Candidate dtype | `torch.bfloat16` |
| Matmul precision | `high` |
| TF32 | Enabled by benchmark configuration |
| CUDA allocator | `expandable_segments:True` |

Compiler and runtime inspection was useful for rejecting kernels with excessive
register or shared-memory use, but only end-to-end execution in this environment
could promote an optimization.

### 4.5 Unified Correctness Result

The final result was:

| Correctness metric | Result |
|---|---:|
| Cases executed | 13 / 13 |
| Accuracy trials | 65 / 65 passed |
| Output elements | 938,885,120 |
| Failed elements | 0 |
| Maximum absolute error | 0 |
| Maximum relative error | 0 |

Case 14 was executed as a separate ordinary benchmark job because a failed
execution has no meaningful latency or correctness value:

| Case | Shape | Job outcome | Correctness | Performance |
|---:|---|---|---|---|
| 14 | `B=32, S=100000, D=1024, H=16, FFN=1024, L=2`, causal BF16 | `execution_failed / out_of_memory` | Not reached | Not reported |

The result is retained as
`job-1788129496983-10b29b153d336914`, snapshot
`a3e3d22039259e96b64b5ee8470d87de4cdc007666ab3368223c53072cbae110`.
This is an explicit failed test result, not an omitted case or a zero score
silently folded into the Cases 1–13 aggregate.

## 5. Performance Results

The following measurements come from
`job-1788222492773-5dd44f6e492d883e`. Each case used 20 warm-up iterations and
three benchmark rounds of 100 steady-state samples. The table reports the
median derived from the structured raw samples.

| Case | B | S | D | Heads | FFN | Layers | Baseline (ms) | Candidate (ms) | Speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 128 | 128 | 4 | 128 | 4 | 1.421312 | 0.577536 | 2.461x |
| 2 | 1 | 128 | 128 | 4 | 128 | 4 | 0.933968 | 0.098304 | 9.501x |
| 3 | 4 | 128 | 128 | 4 | 128 | 4 | 0.935584 | 0.126976 | 7.368x |
| 4 | 16 | 128 | 128 | 4 | 128 | 4 | 0.941056 | 0.202752 | 4.641x |
| 5 | 128 | 128 | 128 | 4 | 128 | 4 | 3.222528 | 1.170432 | 2.753x |
| 6 | 10000 | 128 | 128 | 4 | 128 | 4 | 414.119934 | 75.557884 | 5.481x |
| 7 | 64 | 128 | 32 | 4 | 32 | 4 | 1.102848 | 0.458752 | 2.404x |
| 8 | 64 | 128 | 1024 | 4 | 1024 | 4 | 11.676672 | 10.955776 | 1.066x |
| 9 | 64 | 128 | 128 | 1 | 128 | 4 | 0.871424 | 0.488448 | 1.784x |
| 10 | 64 | 128 | 128 | 2 | 128 | 4 | 1.110016 | 0.500736 | 2.217x |
| 11 | 64 | 128 | 128 | 16 | 128 | 4 | 7.277568 | 1.028096 | 7.079x |
| 12 | 64 | 32 | 128 | 4 | 128 | 4 | 0.941088 | 0.153600 | 6.127x |
| 13 | 64 | 1024 | 128 | 4 | 128 | 4 | 110.874306 | 12.164096 | 9.115x |

### 5.1 Aggregate Results

| Metric | Result |
|---|---:|
| Baseline summed median latency | 555.428304 ms |
| Candidate summed median latency | 103.483388 ms |
| Summed latency reduction | 81.37% |
| Aggregate speedup | 5.367318x |
| Per-case speedup geometric mean | 3.900295x |
| Final incremental latency-GM improvement | 1.981750% |

The geometric mean and aggregate ratio answer different questions. Cases 2 and
3 demonstrate large launch-overhead gains and contribute strongly to the
geometric mean. Cases 6 and 13 dominate aggregate latency and MFU. Reporting
both prevents either small-case consistency or total throughput from being
hidden.

## 6. Supplemental Dense-Equivalent MFU and Theoretical Limits

### 6.1 FLOP Model

Track 3 does not define MFU as an official scoring metric. We report a
supplemental dense-equivalent Model FLOPs Utilization value to connect latency
to a transparent theoretical reference. For a Transformer
layer with batch `B`, sequence length `S`, model width `D`, and FFN width `F`,
the modeled operation count is:

\[
F_{\text{layer}}
=
8BSD^2 + 4BSDF + 4BS^2D
\]

The terms represent:

- Q, K, V and attention-output projections: `8BSD²` FLOPs;
- the two FFN projections: `4BSDF` FLOPs;
- dense QK and PV attention products: `4BS²D` FLOPs.

Elementwise activation, softmax, normalization, masking, and residual
operations are not included. Summed over Cases 1–13, this model gives:

\[
F_{\text{suite}} = 2.017695105024\ \text{TFLOP}
\]

The dense-equivalent MFU is:

\[
\text{MFU}
=
\frac{F_{\text{suite}} / T_{\text{suite}}}
     {58.25\ \text{TFLOP/s}}
\]

where `58.25 TFLOP/s` is the BF16 dense tensor-core reference peak used for
this RTX 4070 and accumulation mode.

### 6.2 Measured MFU

| Metric | Baseline | Final candidate |
|---|---:|---:|
| Aggregate latency | 555.428304 ms | 103.483388 ms |
| Dense-equivalent throughput | 3.632683 TFLOP/s | 19.497768 TFLOP/s |
| Dense-equivalent MFU | 6.236% | 33.473% |

The supplemental MFU improvement is `5.367318x`, equal to the aggregate latency ratio
because the modeled workload is fixed. This is the most direct summary of how
much additional model work the optimized implementation completes per unit
time over the entire Cases 1–13 suite.

### 6.3 Idealized Dense-Compute Roofline

If all modeled FLOPs executed at the `58.25 TFLOP/s` reference peak and every
other cost were free, the suite would take:

\[
T_{\text{roofline}}
=
\frac{2.017695105024\ \text{TFLOP}}
     {58.25\ \text{TFLOP/s}}
=
34.638543\ \text{ms}
\]

This produces the following idealized ratios:

| Comparison | Idealized ratio |
|---|---:|
| Baseline latency / dense roofline | 16.034979x |
| Candidate latency / dense roofline | 2.987521x |

The first number is an idealized baseline-to-roofline ceiling under this FLOP
definition. The second is the remaining distance between the current candidate
and the same reference.

### 6.4 Why `2.99x` Is Not a Promised Remaining Speedup

The roofline assumes all counted operations run at peak tensor-core throughput
and all uncounted work is free. A real Transformer violates both assumptions:

- softmax and LayerNorm contain reductions and transcendental operations;
- small shapes are limited by launch and synchronization latency;
- memory traffic, shared-memory barriers, register pressure, and output stores
  remain necessary;
- not every projection shape reaches peak GEMM efficiency;
- exact BF16/FP16 materialization boundaries restrict fusion choices;
- strict numerical equivalence prevents changing reduction order freely.

The metric is also intentionally dense-equivalent. Some promoted kernels skip
future causal work that appears in the `4BS²D` numerator. Consequently, the
metric measures effective model throughput rather than literal tensor-core
active cycles. An implementation that algebraically removes dense work can
move faster than the corresponding dense-work time without violating hardware
limits.

For these reasons, the report separates:

1. **measured result:** 33.47% dense-equivalent MFU;
2. **mathematical reference:** 34.64 ms and 2.99x remaining to the dense
   roofline;
3. **practical frontier:** lower and case-dependent because of non-GEMM work,
   exactness, memory traffic, and launch constraints;
4. **contract feasibility:** Case 14 cannot be included in this Cases 1–13
   calculation without changing the workload or execution capability.

Earlier planning used a `13.4x` roofline estimate for a five-case
representative subset. That number served as a search-direction estimate and
is not directly comparable with the final `16.03x` Cases 1–13 aggregate
baseline-to-roofline reference. Scope and FLOP accounting must accompany every
ceiling claim.

## 7. Rejected Experiments and What the AI System Learned

The optimization process produced useful negative evidence as well as promoted
kernels.

### 7.1 Packing Alone Was Not the Main Leverage Point

An early packed-QKV diagnostic produced only a small aggregate improvement.
Packing reduced launches and copies but did not remove the dominant attention,
normalization, and large-GEMM costs. The AI campaign therefore shifted toward
complete execution segments rather than continuing to tune packing alone.

### 7.2 Mathematically Equivalent LayerNorm Was Not Exact LayerNorm

Earlier custom LayerNorm implementations used mathematically equivalent mean
and variance formulas but changed PyTorch's online Welford update and merge
order. The difference amplified through four layers and failed strict
correctness. The promoted kernel specializes the native arithmetic order
instead of substituting another reduction.

### 7.3 Lower Launch Count Can Lose to Resource Pressure

A merged Case-6 attention-prefix kernel reduced launch and staging redundancy
but increased shared memory and register use. Occupancy fell and the exact
candidate became 2.61% slower. A subsequent lower-shared-memory proposal was
stopped before GPU execution when static compilation violated its resource
gate.

These results show why the agent policy requires compiler/resource evidence in
addition to a memory-traffic argument.

### 7.4 Focused Improvements Must Reproduce in the Unified Matrix

Several Case-8 and small-case experiments appeared positive in focused jobs but
failed their unified promotion threshold or moved with negative controls.
Those changes were reverted or retained only as rejected history. Final results
are therefore not assembled from unrelated best-case runs.

### 7.5 Instruction Counts Do Not Prove Logical Correctness

The failed Case-6 MMA mapping contained the expected instruction family and
counts but assigned two logical fragments incorrectly. Executable
lane-coordinate oracles became mandatory for direct-register MMA paths.

## 8. Impact and Relevance Beyond the Benchmark

The immediate result is faster execution for the supplied Transformer shapes,
but the reusable contribution is the optimization workflow. GPU engineering
normally alternates between source changes, long compilation, scarce-device
measurement, numerical debugging, and rollback. AI can accelerate hypothesis
generation and implementation, but only if the measurement loop makes false
promotions difficult. The combination of immutable snapshots, one shared GPU
queue, machine-readable correctness, agent skills, and auditable commits is
applicable to other operator, compiler, and inference optimization projects.

For model-serving stakeholders, the measured reduction from `555.43 ms` to
`103.48 ms` across the fixed suite represents `5.37x` more aggregate model work
per unit time under the report's workload weighting. Depending on the serving
regime, this kind of improvement can reduce accelerator demand, batch latency,
or cost per inference. The result is hardware- and shape-specific, so those
benefits must be remeasured on the deployment workload rather than extrapolated
from this benchmark.

For engineering teams, the system also makes AI-assisted performance work more
practical. Research can proceed in parallel across isolated worktrees, while
expensive GPU evidence remains serialized, reproducible, and attributable.
Rejected attempts remain useful because their compiler diagnostics,
correctness failures, and timings become structured feedback for later agents.
This turns AI from a one-shot code generator into a controlled experimental
collaborator.

The next practical step is not production deployment, which the Track 3 brief
places out of scope. It is portability validation: reproduce the immutable
winner on the organizer environment, recompile architecture-specific kernels,
and tune or disable each specialization when its dispatch assumptions no
longer hold.

## 9. Limitations

1. **Case 14 is not included in the performance or supplemental MFU claim.** Its declared
   configuration is `B=32`, `S=100000`, `D=1024`, 16 heads, and two layers.
   One full BF16 input occupies `6.1035 GiB`; a distinct full output raises the
   unavoidable input-plus-output floor to `12.2070 GiB`, already `627.7 MiB`
   above the GPU's measured `11.5941 GiB` capacity before weights, workspace,
   or attention state. The actual run failed while the unchanged baseline tried
   to allocate another `6.10 GiB`, with only `5.21 GiB` free. Because baseline
   execution precedes candidate execution, a candidate-only streaming kernel
   cannot make the official comparison start on this GPU. The case was not
   resized, aliased, offloaded, or timed candidate-only.
2. **The exact LayerNorm and direct-MMA kernels are runtime-specific.** A
   framework, compiler, CUDA, or GPU-architecture change requires renewed
   source, resource, SASS, oracle, and end-to-end validation.
3. **Small cases are sensitive to timer quantization.** Raw rounds and absolute
   latency changes are necessary when interpreting sub-0.2 ms measurements.
4. **MFU is supplemental and dense-equivalent, not an official score or a
   hardware profiler counter.** It is suitable
   for fixed-workload before/after comparison but does not claim that tensor
   cores were active for 33.47% of wall time.
5. **Local validation does not replace the organizer harness.** The final
   TikTok evaluation environment and official test cases remain authoritative.

6. **The benchmark source and final winner must be reviewed together.** The
   final measured source lives on the auditable integration branch and is
   identified by commit and tree hashes in Appendix A; publication must expose
   that exact tree rather than an earlier branch tip.

## 10. Conclusion

This work demonstrates that AI agents can perform sustained, low-level GPU
optimization when they are given a deterministic and correctness-gated
feedback system.

`benchmarkctl` provided immutable snapshots, environment identity, FIFO GPU
serialization, exclusive locking, and structured verdicts. The Stop hook
turned long GPU runs into asynchronous agent continuations. Project skills
encoded the numerical contract, investment rules, fallback requirements, and
promotion policy. Git worktrees and one-optimization commits allowed parallel
research without sacrificing final-source auditability.

Within that system, AI agents developed a portfolio of shared and
shape-specific optimizations across Cases 1–13. The final candidate reduced
aggregate latency from `555.428304 ms` to `103.483388 ms`, achieved a
`3.900295x` equal-case geometric-mean speedup, and increased dense-equivalent
supplemental dense-equivalent MFU from `6.24%` to `33.47%`. All 938,885,120 evaluated output elements were
bitwise identical to the reference.

The remaining `2.99x` distance to the idealized dense-compute reference should
not be interpreted as easy or fully attainable headroom. It identifies the
mathematical scale of the remaining opportunity; practical progress now
depends on removing non-GEMM work, memory traffic, and synchronization while
continuing to reproduce exact numerical boundaries.

The central lesson is that the productivity of AI optimization comes from the
quality of its feedback loop. Fast code generation alone is insufficient.
Immutable evaluation, deterministic correctness, explicit rollback rules, and
auditable experiment history are what made rapid iteration produce a credible
final result.

## Appendix A: Reproducibility Identifiers

| Item | Identifier |
|---|---|
| Final source commit | `803ea145c796702357af4b1b75528dd701fa472a` |
| Final Git tree | `a456810a8d6b139f3c3959816865c3bcb3278a65` |
| Benchmarked base commit | `5adff70f5536e578aaf6b3b41d10563082342b90` |
| Final snapshot SHA-256 | `f71eab84a92027add68bd86a20a895e4d3bcc4a5bbe074991fd6fedb3dd42b71` |
| Unified benchmark job | `job-1788222492773-5dd44f6e492d883e` |
| Python executable SHA-256 | `8f9781a98200d9ecda7e00464e4c64b1327abae788ae8e6979d5c859311410c7` |
| Python inventory SHA-256 | `d0ab19aadfb58a9228978b0c4f778b2e2736236b5c671cdb76f835645d1c49d2` |
| `uv.lock` SHA-256 | `8e9c61178f0c40779a5e3e4eaee4a15adfffd4ea7e72e3bcb51a8a1cfe39cc39` |

The final source commit differs from the benchmarked base commit only by a
message-only amendment; both reference the same validated Git tree.

## Appendix B: Representative Optimization Commits

| Commit | Contribution |
|---|---|
| `65dcfab` | Exact linear-plus-GELU FFN fusion |
| `68d4f50` | FFN output projection and residual epilogue fusion |
| `29279dd` | Attention output projection and residual epilogue fusion |
| `f157bcd` | Packed-V consumption for Cases 6 and 10 |
| `fd54468` | Direct-write HD32 prefix PV for Cases 1 and 5 |
| `002f9ce` | Specialized HD8 PV for Case 11 |
| `2204240` | Full HD32 PV for Cases 4 and 12 |
| `cbd17ed` | Direct final-layout HD8 PV for Case 7 |
| `753e85a` | Causal key-prefix reduction for Cases 4 and 9 |
| `0aa3258` | Native-order softmax with direct BF16 output |
| `226bc2e` | Corrected exact fused Case-6 attention |
| `f1d7140` | Exact shared-CTA Case-13 attention |
| `9120d58` | Fully future Case-13 QK fragment elimination |
| `803ea14` | Exact four-row Case-6 LayerNorm and final winner |

## Appendix C: Key Benchmark Jobs

| Job | Scope | Outcome |
|---|---|---|
| `job-1788209304650-9c14bb178d3871f7` | Cases 1–13 | Direct BF16 softmax-store winner; bitwise exact |
| `job-1788214517564-ff7c175156684913` | Cases 1–13 | Exact fused Case-6 attention winner; bitwise exact |
| `job-1788216718414-2790b7270e7fdc63` | Cases 1–13 | Shared-CTA Case-13 winner; bitwise exact |
| `job-1788219125730-051ece633911d073` | Cases 6/12/13 | Future-fragment focused result; bitwise exact |
| `job-1788221338443-3a7f296f33b57f42` | Cases 6/12/13 | Four-row LayerNorm focused result; bitwise exact |
| `job-1788222081389-bd0972bee7932ec3` | Cases 6/12/13 | Composite focused result; bitwise exact |
| `job-1788222492773-5dd44f6e492d883e` | Cases 1–13 | Final unified promotion; bitwise exact |
| `job-1788129496983-10b29b153d336914` | Case 14 exact declared shape | CUDA OOM before correctness or timing; no performance claim |

## Appendix D: Official Submission Alignment

This report is one component of the required submission package. The repository
also contains:

- `README.md` for overview, installation, result reproduction, limitations,
  and contribution disclosure;
- `DEVPOST_SUBMISSION.md` as paste-ready English project-description copy;
- `DEMO_SCRIPT.md` as a three-minute end-to-end walkthrough plan.

The package was checked against the
[official TechJam overview and deliverables](https://tiktoktechjam2026.devpost.com/)
and the [official rules](https://tiktoktechjam2026.devpost.com/rules), in
addition to the Track 3 problem statement supplied by the organizer.

Before Devpost submission, the repository must be switched to public
visibility, the demo must remain public on YouTube, and the published source
must contain the exact winner tree identified in Appendix A.
