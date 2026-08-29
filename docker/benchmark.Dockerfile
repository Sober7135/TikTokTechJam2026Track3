# syntax=docker/dockerfile:1.7

# Both build inputs must be immutable digest references. The base image must
# contain Python 3.12; CUDA/PyTorch packages are installed from uv.lock.
#
# docker buildx build \
#   --build-arg BASE_IMAGE=registry/python@sha256:<digest> \
#   --build-arg UV_IMAGE=ghcr.io/astral-sh/uv@sha256:<digest> \
#   --file docker/benchmark.Dockerfile .
ARG BASE_IMAGE
ARG UV_IMAGE

FROM ${UV_IMAGE} AS uv

FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/benchmark-venv

COPY --from=uv /uv /usr/local/bin/uv

# This layer changes only when the dependency contract changes. The BuildKit
# cache mount survives layer invalidation, so large PyTorch/CUDA wheels do not
# need to be downloaded again by the same builder.
WORKDIR /opt/benchmark
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,id=techjam-uv-cache,target=/root/.cache/uv,sharing=locked \
    uv sync --frozen --no-install-project

ENV PATH="/opt/benchmark-venv/bin:${PATH}"

# Candidate source is mounted read-only at /workspace for each job. The image
# receives neither the build cache nor worker credentials at runtime.
RUN python -c "import numpy, torch; print(torch.__version__)"

WORKDIR /workspace
