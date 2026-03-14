# ── Build stage: install dependencies ─────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build tools needed by some Python packages (scipy, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Optional S3 upload support — installed separately so the base image
# works without it (boto3 adds ~30 MB)
RUN pip install --no-cache-dir --prefix=/install boto3


# ── Runtime stage ──────────────────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# LightGBM requires libgomp (OpenMP runtime)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY main.py .
COPY app/ app/

# /data is the mount point for persistent storage (EFS or local volume):
#   /data/energy.db   — SQLite database
#   /data/charts/     — generated PNG charts
#   /data/index.html — generated dashboard (also uploaded to S3 if configured)
RUN mkdir -p /data/charts

ENV DB_PATH=/data/energy.db \
    CHARTS_DIR=/data/charts \
    DASHBOARD_PATH=/data/index.html \
    PYTHONUNBUFFERED=1

# Default command — runs full update + analyse cycle
CMD ["python", "main.py", "all"]
