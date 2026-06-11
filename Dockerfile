# Kairon web app — runs the full stack (UI + analysis engine + verifier thread).
# Build with:  docker compose build
# Run with:    docker compose up
# Then visit:  http://localhost:47829/upload
#
# Layered for fast rebuilds: dependencies are installed in a separate layer
# from the project source. Bumping the source code only invalidates the
# bottom layer.

FROM python:3.12-slim AS base

# System deps: curl is used by the entrypoint healthcheck; gcc + libffi are
# needed by some scientific wheels (numpy/scipy/plotly). The slim image keeps
# the final layer small.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy

# Install uv (the package manager) — pinned to a known-good version.
COPY --from=ghcr.io/astral-sh/uv:0.5.10 /uv /uvx /usr/local/bin/

WORKDIR /app

# ---- 1. Dependency layer (invalidated only when pyproject.toml/uv.lock changes) ----
COPY pyproject.toml uv.lock ./
RUN uv sync --extra api --extra web --extra cli --extra ml --no-install-project

# ---- 2. Source layer (invalidated when the project source changes) ----
COPY src ./src
COPY assets ./assets
COPY README.md ./
COPY entrypoint.sh ./entrypoint.sh
# Strip any CRLF that checked out on Windows — shebang with \r breaks exec().
RUN sed -i 's/\r$//' /app/entrypoint.sh && chmod +x /app/entrypoint.sh
RUN uv sync --extra api --extra web --extra cli --extra ml

# The kairon CLI is installed via [project.scripts] in pyproject.toml.
# Make sure `kairon` is on PATH for the entrypoint smoke test.
ENV PATH="/app/.venv/bin:${PATH}"

# ---- Runtime dirs ----
# `runs/` holds the per-run CSV inputs and chart PNGs (writable).
# `data/`  holds the RunStore sqlite DB.
# `assets/` holds the earth background (mounted at /static/earth.png in the app).
RUN mkdir -p /app/runs /app/data /app/reports

# Drop privileges — run as a non-root user.
RUN useradd --create-home --shell /bin/bash kairon && \
    chown -R kairon:kairon /app
USER kairon

EXPOSE 47829

# Tiny entrypoint: ensure the persistent dirs exist (in case the volume is
# freshly mounted with odd perms), then exec uvicorn.
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "kairon.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "47829", "--workers", "1"]
