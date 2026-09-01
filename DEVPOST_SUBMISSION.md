# Devpost Submission Draft

## Project Name

AI-Driven GPU Optimization for Exact Transformer Inference

## Elevator Pitch

We built an AI-assisted GPU optimization system that made the TikTok TechJam
Track 3 PyTorch Transformer benchmark 3.90x faster by geometric mean and 5.37x
faster in aggregate latency across Cases 1–13 on an RTX 4070, while producing
bitwise-exact outputs in all 65 accuracy trials.

## Inspiration

AI coding systems can generate CUDA and Triton code quickly, but performance
optimization is an experimental discipline. A plausible kernel can be slower,
numerically wrong, tied to one timing artifact, or measured against a changed
baseline. Our goal was therefore larger than producing one fast kernel: build a
feedback system in which AI agents can explore many low-level optimizations
without losing correctness, reproducibility, or attribution.

## What It Does

The project optimizes the supplied PyTorch Transformer inference workload with
a portfolio of common and shape-specialized GPU paths. It preserves the public
`forward(x, valid_token_mask)` interface and gives baseline and candidate the
same weights, inputs, masks, dtype, and device configuration.

The final candidate combines CUDA Graph replay, exact FFN and projection fusion,
residual epilogues, packed and direct-layout QKV, direct-write PV/context
kernels, causal key-prefix reduction, native-order BF16 softmax, shared-CTA
attention, and an exact multi-row LayerNorm specialization. Guarded dispatch
keeps a baseline-compatible fallback for unsupported inputs.

## How We Built It

OpenAI Codex performed most of the optimization research and implementation
under human-defined correctness, resource, and promotion policies. A supervisor
agent split the problem into independent bottlenecks; worker agents investigated
attention, FFN/projection fusion, layout removal, launch reduction, and difficult
shapes in isolated Git worktrees.

We built `benchmarkctl`, a local Rust control plane, to make those iterations
trustworthy. Each submitted experiment is snapshotted immediately, assigned a
content digest, and placed in a persistent FIFO queue. One exclusive GPU lock
prevents overlapping measurements. Jobs run with a cleared environment and
record the commit, source snapshot, Python executable, lockfile, package
inventory, requested cases, raw timing samples, and structured correctness
result.

A Codex Stop hook turns a long GPU run into an asynchronous continuation. The
agent submits the job and ends its turn; after the queued run finishes, the hook
resumes the same session with paths to `result.json` and `benchmark.log`. Two
repository-local Codex skills encode the optimization feedback loop and the
correctness-first benchmark review policy.

Every optimization was kept as an attributable commit whose message explains
the bottleneck, implementation, numerical-equivalence argument, dispatch scope,
validation evidence, and decision. A focused improvement was promoted only if
it reproduced in a unified Cases 1–13 snapshot. Incorrect or unstable attempts
were rejected and retained as feedback for later agents.

## Results

The final immutable run on an NVIDIA GeForce RTX 4070 used BF16, 20 warm-ups,
three rounds of 100 steady-state samples per case, and five accuracy trials per
case.

| Metric | Result |
|---|---:|
| Cases with valid performance | 13 / 13 for Cases 1–13 |
| Speedup geometric mean | 3.900295x |
| Aggregate baseline latency | 555.428304 ms |
| Aggregate candidate latency | 103.483388 ms |
| Aggregate speedup | 5.367318x |
| Accuracy trials | 65 / 65 passed |
| Failed output elements | 0 / 938,885,120 |
| Maximum absolute and relative error | 0 and 0 |

We also report supplemental dense-equivalent Model FLOPs Utilization. Under a
documented 2.017695 TFLOP suite model and 58.25 TFLOP/s reference peak, MFU rises
from 6.236% to 33.473%. The idealized dense-compute time is 34.639 ms, placing
the current candidate 2.99x above that mathematical reference. This is not an
official Track 3 metric or a promised remaining speedup: reductions, memory
traffic, synchronization, launch overhead, and strict numerical boundaries
make the practical limit lower and case-dependent.

Case 14 was attempted at its exact declared shape and not resized. The ordinary
job recorded CUDA OOM before correctness or timing. One full BF16 input plus an
independent output requires 12.207 GiB, already above the GPU's measured 11.594
GiB capacity before weights and workspace. We therefore report Cases 1–13 and
the Case 14 execution failure separately rather than hiding the limitation.

## Challenges We Ran Into

The hardest part was exactness. Mathematically equivalent LayerNorm or softmax
code can change reduction order or BF16 materialization and fail elementwise
comparison. A kernel can also contain the expected tensor-core instructions but
map logical fragments incorrectly. We added executable lane/coordinate oracles,
resource checks, fallback guards, and end-to-end comparisons; static PTX/SASS
inspection never replaced numerical validation.

Measurement integrity was the other major challenge. Parallel agents are useful
for research but unsafe for concurrent benchmarking on one GPU. Immutable
snapshots, a shared serialized queue, negative controls, and unified reruns kept
the final result comparable.

## Accomplishments We Are Proud Of

- One exact source snapshot improves all 13 measured cases; the result is not a
  collage of unrelated best-case runs.
- Every evaluated output element is bitwise identical to the reference.
- AI agents performed sustained low-level CUDA/Triton engineering through a
  deterministic feedback loop rather than one-shot code generation.
- The experiment history records both successful and rejected ideas, making the
  final result auditable and future work more efficient.

## What We Learned

The leverage did not come from packing QKV alone. Large gains required removing
complete execution segments: intermediate layouts, redundant materialization,
future causal work, framework launch overhead, and separate normalization or
epilogue passes. We also learned that the best AI optimization infrastructure
separates creative hypothesis generation from deterministic verdicts.

## Impact and Future Work

For fixed inference workloads, the measured aggregate result represents 5.37x
more modeled work per unit time and can translate into lower latency, lower
accelerator demand, or lower cost per inference. The reusable agent-plus-queue
workflow applies beyond this benchmark to compiler, operator, and model-serving
optimization where hardware is scarce and correctness is non-negotiable.

With more time, we would validate the exact winner on the organizer hardware,
retune architecture-specific routes, automate more portability guards, profile
the remaining non-GEMM work, and evaluate Case 14 on a larger-memory device
without changing its contract. Production deployment is outside the Track 3
scope.

## Development Tools, APIs, Libraries, and Assets

- **AI tool/API:** OpenAI Codex, used for supervisor and worker agents
- **Development tools:** Git worktrees, Rust/Cargo, `uv`, Nix, CUDA toolkit,
  `nvcc`, Ninja, PTX/SASS inspection, shell tooling
- **Libraries/frameworks:** PyTorch, NumPy, Triton
- **Datasets:** None
- **Assets:** Supplied Track 3 benchmark definition and deterministic synthetic
  inputs generated from fixed seeds; no external checkpoint or third-party media

## Links

- **Public repository:** https://github.com/Sober7135/TikTokTechJam2026Track3
- **Public YouTube demo:** https://youtu.be/_yS3vhezGsk
- **Technical report:** `TECHNICAL_REPORT.md` in the public repository

## Entrant and AI Disclosure

**Sole entrant:** Jinye Wu — problem framing, correctness and resource policy,
experiment supervision, review, integration decisions, and submission.

OpenAI Codex is disclosed as an AI development tool, not a team member. Human
oversight defined the problem, correctness and resource policy, supervised the
experiments, reviewed evidence, and made integration decisions. Jinye Wu owns
the final submission.
