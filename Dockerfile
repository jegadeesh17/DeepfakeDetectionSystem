# Multi-stage Production Dockerfile for Deepfake Detection System
FROM python:3.11-slim AS builder

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgl1-mesa-glx libglib2.0-0 curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    /opt/venv/bin/pip install --no-cache-dir fastapi uvicorn pydantic pydantic-settings python-multipart pillow opencv-python-headless

FROM python:3.11-slim AS runner

WORKDIR /app
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8004

RUN apt-get update && apt-get install -y --no-install-recommends libgl1-mesa-glx libglib2.0-0 curl && rm -rf /var/lib/apt/lists/*

RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

COPY --from=builder /opt/venv /opt/venv

COPY --chown=appuser:appgroup api/ api/
COPY --chown=appuser:appgroup src/ src/
COPY --chown=appuser:appgroup configs/ configs/
COPY --chown=appuser:appgroup models/ models/

USER appuser

EXPOSE 8004

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
