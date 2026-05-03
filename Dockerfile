# ── Stage 1: builder ──────────────────────────────────────────────────────────
# Install all Python dependencies into an isolated venv so only the venv
# (not build tools) gets copied to the final image.
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a venv for the final image
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Install only the API-runtime subset (no jupyter, optuna, mlflow, dvc…)
# Copying pyproject.toml first lets Docker cache this layer until deps change.
COPY pyproject.toml README.md ./
# Stub src so setuptools can resolve the editable metadata without the full tree
RUN mkdir -p src && touch src/__init__.py
RUN pip install --upgrade pip \
    && pip install --no-cache-dir ".[api]"


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="Expresso Churn Prediction API"
LABEL org.opencontainers.image.description="FastAPI service for prepaid subscriber churn scoring"

WORKDIR /app

# Copy the pre-built venv from the builder stage
COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy only what the API needs at runtime
COPY src/      ./src/
COPY configs/  ./configs/

# Create artifact mount-points so Docker volumes attach cleanly
RUN mkdir -p models data/features

# Run as a non-root user
RUN useradd --no-create-home --shell /bin/false appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Workers default to 1 so a single container is safe; override with
# API_WORKERS env var (e.g. 4 for production behind a load balancer).
CMD ["sh", "-c", \
     "uvicorn src.api.app:app \
        --host 0.0.0.0 \
        --port 8000 \
        --workers ${API_WORKERS:-1} \
        --log-level ${LOG_LEVEL:-info}"]
