# Lightweight HLE no-tools evaluation image.
# Reuses the existing hle_lite runtime and dependency set.
FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/haolpku/DataLite-RSI"
LABEL org.opencontainers.image.description="DataLite-RSI HLE no-tools evaluator"

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Python dependencies (ultra minimal)
RUN pip install --no-cache-dir \
    httpx>=0.20.0 \
    pyarrow>=10.0.0 \
    pyyaml>=5.1 \
    rich>=13.3.3 \
    python-dotenv>=0.16.0

# Copy application files
COPY benchmarks/hle/notools/evaluator.py /app/evaluator.py
COPY benchmarks/hle/notools/configs /app/configs

# Create directories
RUN mkdir -p /app/outputs /app/.hf-eval

# Environment
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_OFFLINE=1

# Default command
ENTRYPOINT ["python", "evaluator.py"]
CMD ["--help"]
