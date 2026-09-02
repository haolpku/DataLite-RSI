FROM python:3.12-slim@sha256:2c941e860699f878900b0edc2403613c234d4b32eda3cc9fa7036991a2a63c4a

LABEL org.opencontainers.image.source="https://github.com/haolpku/DataLite-RSI"
LABEL org.opencontainers.image.description="DataLite-RSI HLE with-tools evaluator"
LABEL org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN python -m pip install \
    "httpx==0.28.1" \
    "markdownify==0.14.1" \
    "readabilipy==0.3.0" \
    "numpy==2.2.6" \
    "scipy==1.15.3" \
    "sympy==1.14.0"

COPY benchmarks/hle/withtools/src/hle_eval /app/hle_eval

RUN install -d -m 0755 /config /data /output

# The harness retains access to provider secrets; generated Python is dropped
# to uid/gid 65534 by hle_eval.tools.execute_python.
ENTRYPOINT ["python", "-m", "hle_eval"]
CMD ["--config", "/config/eval.json"]
