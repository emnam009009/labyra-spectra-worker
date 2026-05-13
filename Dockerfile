# syntax=docker/dockerfile:1.7

# ---------- Stage 1: builder ----------
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install --prefix=/install .

# ---------- Stage 2: runtime ----------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# Runtime deps only (no build-essential)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgfortran5 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY src/ /app/src/

# Non-root user for security
RUN useradd --create-home --uid 1000 worker && chown -R worker:worker /app
USER worker

EXPOSE 8080

# Cloud Run sends HTTP requests; FastAPI receives Pub/Sub push messages
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
