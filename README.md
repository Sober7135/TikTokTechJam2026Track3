# TikTok TechJam 2026 Track 3

This repository keeps the GPU benchmark surface separate from its remote
orchestration service.

## GPU benchmark

- `torch_transformer_benchmark.py`: stable CLI entry point for local and worker
  invocations.
- `transformer_benchmark/`: benchmark implementation, split into case/input,
  model, correctness, timing, and runner modules. Candidate implementations
  belong in `transformer_benchmark/models.py`.
- `pyproject.toml` and `uv.lock`: the pinned Python 3.12 benchmark environment.
- `docker/benchmark.Dockerfile`: the reproducible GPU runtime image. Dependency
  downloads are reused through a BuildKit uv cache, while benchmark containers
  run offline.

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

- `abs_error <= atol`; or
- `abs_error <= rtol * abs(baseline)`.

The defaults are `atol=0.002` and `rtol=0.02`. Every element in every accuracy
trial must pass before performance is accepted. Relative error reported near a
zero baseline is clamped only for summary display; the pass/fail rule above
uses the unclamped baseline value.

## Remote benchmark orchestrator

All Rust orchestration code and its configuration live under
[`orchestrator/`](orchestrator/):

- shared job/result contracts;
- persistent control-plane queue;
- a GitHub-token poller that enqueues exact PR head SHAs and updates one PR
  comment;
- outbound-polling Docker GPU worker;
- optional Codex App Server analysis and bounded verification jobs.

See [`orchestrator/README.md`](orchestrator/README.md) for build and operation
instructions, and [`REQUIREMENTS.md`](REQUIREMENTS.md) for the target polling
architecture.
