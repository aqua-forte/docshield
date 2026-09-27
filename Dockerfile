# ==============================================================================
# DocShield: Enterprise Image Anonymization & PII Redactor
# Production Container Image
# ==============================================================================

FROM python:3.11-slim AS base

# System configuration
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

# Install runtime dependencies (OpenCV headless support)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY pyproject.toml .
COPY README.md .
COPY LICENSE .
COPY docshield/ docshield/

# Install docshield in editable/production mode
RUN pip install --no-cache-dir -e .

# Security hardening: Run as non-privileged user
RUN useradd -m -u 10001 docshield && \
    chown -R docshield:docshield /app
USER docshield

# Expose microservice HTTP port
EXPOSE 8000

# Container health probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch production ASGI server
CMD ["uvicorn", "docshield.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
