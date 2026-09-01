# AI-Driven GPU Optimization for Exact Transformer Inference

TikTok TechJam 2026 — Track 3

This project uses AI coding agents, Triton, and custom CUDA kernels to optimize
the PyTorch Transformer benchmark supplied for Track 3. The output contract is
preserved: baseline and candidate receive the same weights, inputs, masks,
dtype, and device configuration, and performance is accepted only after
elementwise numerical validation.

The project is also an experiment in AI-assisted systems engineering. OpenAI
Codex agents explore independent optimization directions in isolated Git
worktrees, while a local Rust control plane (`benchmarkctl`) snapshots source,
serializes the shared GPU, records deterministic results, and resumes the same
agent through a Stop hook. Repository-local skills encode the correctness and
promotion policy.

## Results

The final unified RTX 4070 BF16 run for Cases 1–13 produced:

| Metric | Result |
|---|---:|
| Per-case speedup geometric mean | **3.900295x** |
| Aggregate latency | **555.428304 ms → 103.483388 ms** |
| Aggregate speedup | **5.367318x** |
| Supplemental dense-equivalent MFU | **6.236% → 33.473%** |
| Accuracy trials | **65 / 65 passed** |
| Failed output elements | **0 / 938,885,120** |
| Maximum absolute / relative error | **0 / 0** |

MFU is a supplemental efficiency analysis, not an official Track 3 scoring
metric. Its FLOP model and the theoretical roofline are defined in
[TECHNICAL_REPORT.md](TECHNICAL_REPORT.md).

Case 14 was not resized or silently omitted. Its exact
`B=32, S=100000, D=1024, H=16, FFN=1024, L=2` BF16 workload failed with CUDA
OOM on the 12 GB RTX 4070 before correctness or timing. Input plus an independent
output alone requires 12.207 GiB, above the measured 11.594 GiB device capacity,
before weights or workspace. The failed run is retained as
`job-1788129496983-10b29b153d336914`.

The validated Cases 1–13 source is commit
`803ea145c796702357af4b1b75528dd701fa472a`, Git tree
`a456810a8d6b139f3c3959816865c3bcb3278a65`, and immutable snapshot
`f71eab84a92027add68bd86a20a895e4d3bcc4a5bbe074991fd6fedb3dd42b71`.
The final result is `job-1788222492773-5dd44f6e492d883e`.

## Optimization Approach

The final candidate combines shared and shape-specialized routes:

- CUDA Graph replay for stable small-shape launch reduction;
- exact linear-plus-GELU FFN fusion;
- attention/FFN projection and residual epilogue fusion;
- packed and direct-layout QKV kernels;
- direct-write PV/context layouts;
- causal key-prefix and future-fragment elimination;
- native-order softmax with direct BF16 output;
- exact shared-CTA attention for Case 13;
- exact fused attention and four-row LayerNorm for Case 6;
- guarded dispatch with a baseline-compatible fallback for unsupported inputs.

Each optimization was correctness-gated and represented by an attributable Git
commit. Focused wins had to reproduce in one unified Cases 1–13 snapshot before
promotion. Regressions and numerically invalid approaches were retained as
negative evidence rather than combined into a synthetic best-of result.

## AI-Assisted Engineering Workflow

```text
Codex agent proposes a measured optimization
                 ↓
isolated worktree + small attributable commit
                 ↓
static checks and CPU fallback smoke
                 ↓
benchmarkctl snapshots source and queues one GPU job
                 ↓
Stop hook resumes the same agent with result.json and benchmark.log
                 ↓
correctness first, then latency: promote / retain / reject
```

The two repository-local skills are:

- `.agents/skills/techjam-kernel-optimization/`: hypothesis, implementation,
  feedback, and promotion policy;
- `.agents/skills/techjam-local-benchmark/`: immutable GPU submission and
  deterministic result-review policy.

`benchmarkctl` is intentionally local. It is a Rust CLI and process supervisor,
not a network service. Research can run in parallel, but all worktrees share one
absolute `BENCHMARKCTL_STATE_DIR`, FIFO queue, and exclusive GPU lock.

## Repository Structure

- `torch_transformer_benchmark.py`: stable benchmark CLI and compatibility
  facade;
- `transformer_benchmark/`: cases, models, custom kernels, correctness, timing,
  execution, and result serialization;
- `orchestrator/crates/benchmarkctl/`: immutable snapshot, FIFO queue, GPU lock,
  clean environment, Stop-hook integration, and result inspection;
- `pyproject.toml` and `uv.lock`: pinned Python 3.12 environment;
- `TECHNICAL_REPORT.md`: full AI workflow, optimization rationale, per-case
  results, MFU, theoretical limits, and reproducibility identifiers;
- `DEVPOST_SUBMISSION.md`: paste-ready Devpost project description;
- `DEMO_SCRIPT.md`: three-minute YouTube walkthrough script.

## Environment

The final measurements used:

| Component | Configuration |
|---|---|
| CPU | AMD Ryzen 9 7950X, 16 cores / 32 threads |
| RAM | 61 GiB |
| GPU | NVIDIA GeForce RTX 4070, 12,282 MiB, Ada `sm_89` |
| Driver / CUDA | NVIDIA 595.91.07 / CUDA 13.0 |
| OS | Linux 7.2.0, glibc 2.42, NixOS environment |
| Python / PyTorch | Python 3.12.14 / PyTorch `2.13.0+cu130` |
| Kernel stack | Triton 3.7.1 and PyTorch C++/CUDA extensions built with `nvcc` |
| Storage | ZHITAI TiPlus7100 4 TB and 2 TB NVMe; encrypted Btrfs `/home` |

## Setup and Installation

Prerequisites:

- Linux with an NVIDIA GPU and compatible driver;
- Python 3.12;
- `uv`;
- Rust/Cargo for `benchmarkctl`;
- CUDA toolkit, `nvcc`, a C++ compiler, and Ninja for custom extensions.

Install the pinned Python environment and build the local queue:

```bash
uv sync --frozen
cd orchestrator
cargo build --release -p benchmarkctl
cd ..
```

The validated kernel set targets RTX 4070 (`sm_89`). Other GPUs may use native
fallbacks or require recompilation and retuning.

## Reproduce the Results

### 1. CPU smoke test

This verifies the benchmark, correctness path, and fallback behavior. It is not
a GPU-performance result.

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

### 2. Immutable Cases 1–13 GPU run

Repository agents must submit GPU work through `benchmarkctl`:

```bash
orchestrator/target/release/benchmarkctl submit -- \
  --device cuda:0 \
  --dtype bfloat16 \
  --official-cases 1 2 3 4 5 6 7 8 9 10 11 12 13
```

The command snapshots immediately and returns. The project Stop hook claims the
job and runs it asynchronously. Inspect artifacts with:

```bash
orchestrator/target/release/benchmarkctl list
orchestrator/target/release/benchmarkctl show <job-id>
```

Each job is stored under `.benchmarkctl/jobs/<job-id>/`:

- `source/`: exact immutable source snapshot;
- `job.json`: commit, snapshot digest, arguments, environment identity, state;
- `result.json`: machine-readable correctness and raw timing samples;
- `benchmark.log`: captured stdout and stderr.

Correctness must be checked before performance. The local strict elementwise
gate accepts finite values when:

```text
absolute_error < 0.002
OR
absolute_error < 0.02 * abs(reference)
```

The final candidate was bitwise exact and therefore does not depend on the
tolerance interpretation.

### 3. Case 14 feasibility run

```bash
orchestrator/target/release/benchmarkctl submit -- \
  --device cuda:0 \
  --dtype bfloat16 \
  --official-cases 14
```

On the documented 12 GB GPU this is expected to produce a structured
`execution_failed / out_of_memory` result. Do not reduce the shape, alias the
required output, or report candidate-only timing as an official comparison.

## Development Tools, APIs, Libraries, and Assets

- AI tool/API: OpenAI Codex, used as supervisor and optimization agents;
- development tools: Git worktrees, Cargo/Rust, `uv`, Nix, CUDA toolchain,
  PTX/SASS inspection, shell tooling;
- libraries/frameworks: PyTorch, NumPy, Triton;
- datasets: none;
- assets: the supplied Track 3 benchmark definition and deterministic synthetic
  inputs generated from fixed seeds; no third-party model checkpoint or media.

## Limitations and Future Work

- Case 14 cannot execute under the unchanged comparison contract on the local
  12 GB GPU; validation requires a larger-memory device or an organizer-approved
  contract change.
- Custom CUDA paths are compiler-, runtime-, architecture-, dtype-, mask-, and
  shape-specific. They require renewed correctness and performance validation
  on another environment.
- Small-case timings are sensitive to timer quantization and launch noise;
  raw samples and unified reruns are retained for that reason.
- Dense-equivalent MFU is a transparent derived comparison, not a profiler
  counter or official competition metric.
- With more time, we would validate the exact winner on the organizer hardware,
  automate portability guards, profile remaining non-GEMM work, and investigate
  a contract-compliant route for Case 14 on a larger-memory GPU.

## Entrant Contribution and AI Disclosure

| Contributor | Contribution |
|---|---|
| Jinye Wu — sole entrant | Problem framing, correctness and resource policy, experiment supervision, review, integration decisions, and submission |
| OpenAI Codex (AI tool, not a team member) | AI-assisted research, code generation, kernel implementation, testing, result analysis, and documentation under deterministic human-defined gates |

## Submission Links

- Technical report: [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)
- Devpost description draft: [DEVPOST_SUBMISSION.md](DEVPOST_SUBMISSION.md)
- Demo script: [DEMO_SCRIPT.md](DEMO_SCRIPT.md)
- Public repository: [github.com/Sober7135/TikTokTechJam2026Track3](https://github.com/Sober7135/TikTokTechJam2026Track3) — set visibility to public before submission
- Public YouTube demo: [youtu.be/_yS3vhezGsk](https://youtu.be/_yS3vhezGsk)

The public repository must expose the exact validated winner tree identified
above. See [REQUIREMENTS.md](REQUIREMENTS.md) and
[orchestrator/README.md](orchestrator/README.md) for the complete local queue
contract.
