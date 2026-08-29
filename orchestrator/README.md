# Remote benchmark orchestrator

This directory contains a small Rust vertical slice for serializing GPU
benchmarks without GitHub Actions. A control plane persists the queue and owns
Codex analysis; a GitHub-token poller reconciles private pull requests; and an
outbound-only worker leases one job and runs it in a fresh Docker container.

```mermaid
flowchart LR
    P[GitHub poller<br/>fine-grained token] -->|poll PR head + update comment| PR[Private GitHub PR]
    P -->|upload exact-SHA archive + enqueue| CP[Rust control plane]
    ADMIN[Authenticated enqueue API] --> CP
    CP --> Q[(JSON queue state)]
    W[Rust GPU worker] -->|outbound HTTPS poll / heartbeat| CP
    W --> D[Fresh Docker container]
    D -->|versioned result.json| W
    CP --> A[Codex app-server<br/>read-only analysis]
    A -->|allowlisted verification request| Q
    CP -->|job + analysis state| P
```

The poller discovers non-draft open pull requests, downloads an archive for the
exact head SHA, uploads the content-addressed artifact to the control plane,
and idempotently enqueues it. It then updates one marked PR comment with queue
state, deterministic results, verification state, and separately labelled Codex
interpretation. This MVP intentionally has no webhook, GitHub App, Check Run,
or GitHub Actions dependency.

## Components

- `benchmark-contracts`: versioned, validated job/lease/result types.
- `benchmark-github-poller`: GitHub REST polling, immutable source handoff,
  enqueue reconciliation, and one updateable PR comment per pull request.
- `benchmark-control-plane`: Axum/Tokio authenticated HTTP API, persistent FIFO
  queue, content-addressed artifact handoff, lease expiry/retry, SHA
  superseding, and optional Codex App Server analysis. Requests are capped and
  shutdown drains in-flight requests.
- `benchmark-worker`: outbound polling, digest verification, heartbeat,
  timeout enforcement, and Docker execution with no network or secrets.
- `torch_transformer_benchmark.py`: the PyTorch harness. It now writes raw
  samples and correctness details using `--json-output`.

The JSON state backend is intentionally scoped to one control-plane process and
one GPU worker. It provides crash persistence, but not multi-process
transactions. Replace it with SQLite/PostgreSQL before running multiple control
plane replicas.

## Build and test

```bash
cd orchestrator
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Copy `.env.example` values into the service manager of your choice. Do not
commit the real tokens.

Start the control plane:

```bash
CONTROL_PLANE_ADMIN_TOKEN='replace-with-a-long-random-token' \
CONTROL_PLANE_WORKER_TOKEN='replace-with-another-long-token' \
cargo run -p benchmark-control-plane
```

The artifact URL returned by `CONTROL_PLANE_ARTIFACT_BASE_URL` must be reachable
by the worker. Use the same origin for `WORKER_ARTIFACT_BASE_URL`. Loopback is
only suitable when both processes run on the same host; use HTTPS otherwise.

Start the GitHub poller on the control-plane host:

```bash
GITHUB_TOKEN='github_pat_...' \
GITHUB_REPOSITORY='owner/repository' \
CONTROL_PLANE_URL='http://127.0.0.1:8080' \
CONTROL_PLANE_ADMIN_TOKEN='replace-with-a-long-random-token' \
BENCHMARK_VERSION='track3-v1' \
BENCHMARK_HARDWARE_PROFILE='rtx-4070' \
BENCHMARK_IMAGE='registry/image@sha256:<digest>' \
cargo run -p benchmark-github-poller
```

Use a fine-grained personal access token restricted to this repository, with
`Contents: read` and `Pull requests: read and write` permissions. The latter is
needed to read PR metadata and create/update its issue comment. Do not grant
code write, administration, workflow, or organization permissions. Store the
token in the poller's service-manager secret/environment, never in `.env`, the
repository, command-line arguments, or generated job files. Give it an expiry
and rotate it by replacing `GITHUB_TOKEN` and restarting only the poller.

The poller must be a separate process from the control plane. Only it receives
`GITHUB_TOKEN`; the control plane's Codex child, GPU worker, and Docker task do
not. `CONTROL_PLANE_ADMIN_TOKEN` authorizes the poller to upload and enqueue, so
it must also remain control-plane-side.

Enqueue a job after replacing every placeholder in
`examples/enqueue-job.json`:

```bash
curl --fail-with-body \
  -H 'Authorization: Bearer replace-with-a-long-random-token' \
  -H 'Content-Type: application/json' \
  --data-binary @examples/enqueue-job.json \
  http://127.0.0.1:8080/v1/jobs
```

Start a worker on the GPU host:

```bash
CONTROL_PLANE_URL='https://control.example' \
CONTROL_PLANE_WORKER_TOKEN='replace-with-another-long-token' \
WORKER_ID='local-4070-1' \
WORKER_HARDWARE_PROFILE='rtx-4070' \
WORKER_BENCHMARK_IMAGE='registry/image@sha256:<digest>' \
cargo run -p benchmark-worker
```

Loopback HTTP is accepted for development. Any non-loopback control-plane URL
must use HTTPS. The worker uses `curl` for TLS so Rust code never handles TLS
keys or certificate verification itself.

For a local development bundle, create a tarball whose files are at archive
root, set its exact SHA-256 in the job, use a `file:///absolute/path` URL, and
set `WORKER_ALLOW_FILE_SOURCE=true`. Production jobs must use short-lived HTTPS
artifact URLs.

## GitHub polling and PR comments

For every pull request, the poller stores the current head SHA, root job ID,
comment ID, and rendered-comment digest in `GITHUB_POLLER_STATE`. It creates a
timeline comment with the Issue Comments API and updates that comment as the job,
analysis, and verification state changes. If the stored comment ID is stale, it
searches for its hidden marker before creating a replacement. Process restarts
do not discard reconciliation state, and repeated polls do not enqueue or post
again when neither the PR head nor rendered content changed.

The enqueue request must use the exact GitHub `owner/repository` and PR number.
A newer head SHA takes ownership of the existing comment, while late completion
of an older superseded job cannot switch the report back to that job. Comment
text derived from PRs, results, or Codex is bounded, Markdown-escaped, and
prevented from triggering `@mentions`. Deterministic pass/fail data remains
independent of the AI interpretation.

## Docker contract

The worker accepts only the exact image reference configured in
`WORKER_BENCHMARK_IMAGE`, and the contracts require `@sha256:<digest>`. It runs:

- one fresh `docker run --rm` container per job;
- the assigned GPU only;
- no container network;
- a read-only source mount and separate result mount;
- no Docker socket or control-plane credentials;
- explicit memory, PID, shared-memory, and wall-time bounds.

[`../docker/benchmark.Dockerfile`](../docker/benchmark.Dockerfile) is a minimal
runtime-image template. Supply a pinned Python 3.12 base-image digest and a
pinned uv-image digest; the project intentionally does not embed moving
`latest` tags.

Dependency downloads are reusable across image builds. The Dockerfile copies
only `pyproject.toml` and `uv.lock` before running `uv sync`, so Docker reuses
the completed environment layer while the lockfile is unchanged. It also uses
a locked BuildKit cache mount named `techjam-uv-cache`; if an earlier layer is
invalidated, uv can reuse previously downloaded PyTorch/CUDA wheels instead of
fetching them again.

```bash
docker buildx build \
  --build-arg BASE_IMAGE='registry/python@sha256:<digest>' \
  --build-arg UV_IMAGE='ghcr.io/astral-sh/uv@sha256:<digest>' \
  --tag 'registry/techjam-benchmark:<version>' \
  --file ../docker/benchmark.Dockerfile \
  --load \
  ..
```

The named cache belongs to the BuildKit builder, not to a benchmark container.
For an ephemeral or rented builder, additionally export/import BuildKit cache
with `--cache-to` and `--cache-from`, or build once and push the resulting
benchmark image to a nearby registry. The runtime worker should only pull/run
the pinned resulting image; it must not run `uv sync` for each PR.

## Codex analysis

Set `CODEX_ANALYSIS_ENABLED=true` only on the control-plane host after `codex`
is authenticated. The control plane supervises `codex app-server --stdio`, uses
the initialize handshake, creates a read-only/no-network thread, and constrains
the final response to JSON. It passes bounded PR context and the validated root
result to the first review; if Codex requests one allowlisted verification job,
the control plane passes both root and child results to a final review. A
requested verification can only double timing samples or tighten accuracy
thresholds. The queue enforces one child and one verification level, so model
output can never become an arbitrary Docker command or recursive job loop.

The repository skill in
`../.agents/skills/techjam-benchmark-review/SKILL.md` defines the review policy.
Codex has no GitHub token and never posts comments itself; the poller renders its
stored structured output.

Codex output is interpretation only. The stored deterministic result remains
the source of truth for pass/fail.

See [`../REQUIREMENTS.md`](../REQUIREMENTS.md) for the full polling and reporting
target.
