# ==========================================
# Stage 1: Builder with uv
# ==========================================
FROM python:3.12-slim AS builder

# Install uv from official binary
COPY --from=ghcr.io/astral-sh/uv:0.6 /uv /uvx /bin/

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /src

# 1. Cache dependencies layer separately from application code
COPY pyproject.toml .
RUN uv venv /opt/venv && \
    VIRTUAL_ENV=/opt/venv uv pip install -r pyproject.toml

# 2. Install application package into virtual environment
COPY melchior/ ./melchior/
COPY README.md .
RUN VIRTUAL_ENV=/opt/venv uv pip install --no-deps .

# ==========================================
# Stage 2: Minimal, secure runtime image
# ==========================================
FROM python:3.12-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create non-root user for secure sandbox execution
RUN groupadd -g 1000 melchior && \
    useradd -u 1000 -g melchior -m -s /bin/bash melchior && \
    mkdir -p /workspace/artifacts && \
    chown -R melchior:melchior /workspace

# Copy pre-built virtual environment from builder
COPY --from=builder --chown=melchior:melchior /opt/venv /opt/venv

# Copy example tasks and configurations
COPY --chown=melchior:melchior examples/ /workspace/examples/

USER melchior
WORKDIR /workspace

ENTRYPOINT ["melchior"]
CMD ["--help"]
