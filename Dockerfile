# ── Stage 1: builder ──────────────────────────────────────────────────────────
# Install all Python dependencies into an isolated venv so only the venv
# (not build tools) gets copied to the final image.
FROM python:3.11-slim AS builder

# Which extras to install. The default ships every boosting library so any
# trained winner can be unpickled. To slim the image, install only the library
# of the model being served, e.g.
#   docker build --build-arg MODEL_EXTRAS=api,serve-xgboost .
ARG MODEL_EXTRAS=api,serve-all

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
    && pip install --no-cache-dir ".[${MODEL_EXTRAS}]"


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="Prepaid Churn Prediction API"
LABEL org.opencontainers.image.description="FastAPI service for prepaid subscriber churn scoring"

WORKDIR /app

# Copy the pre-built venv from the builder stage
COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CHURN_PREDICTIONS_DB=/app/data/db/predictions.db

# Copy only what the API needs at runtime
COPY src/      ./src/
COPY configs/  ./configs/

# Mount points: models/ and data/features/ are read-only volumes; data/db holds
# the prediction log on a writable volume so it survives restarts.
RUN mkdir -p models data/features data/db logs

# Run as a non-root user
RUN useradd --no-create-home --shell /bin/false appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Workers default to 1 so a single container is safe; override with
# API_WORKERS env var (e.g. 4 for production behind a load balancer). The
# rate limiter divides its limits by API_WORKERS (see src/api/rate_limit.py).
CMD ["sh", "-c", \
     "uvicorn src.api.app:app \
        --host 0.0.0.0 \
        --port 8000 \
        --workers ${API_WORKERS:-1} \
        --log-level ${LOG_LEVEL:-info}"]
