FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime@sha256:77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee

LABEL org.opencontainers.image.source="https://github.com/haolpku/DataLite-RSI"
LABEL org.opencontainers.image.description="DataLite-RSI generative track: instruction-guided image editing, LoRA training and benchmark evaluation"
LABEL org.opencontainers.image.licenses="Apache-2.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/hf

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1=1.4.0-1 \
        libglib2.0-0=2.72.4-0ubuntu2.6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install \
        diffusers==0.39.0 \
        transformers==5.12.1 \
        accelerate==1.14.0 \
        peft==0.19.1 \
        safetensors==0.8.0 \
        tokenizers==0.22.2 \
        huggingface-hub==1.21.0 \
        datasets==5.0.0 \
        numpy==2.4.4 \
        pillow==12.2.0 \
        sentencepiece==0.2.1 \
        protobuf==7.35.1 \
        bitsandbytes==0.49.2 \
        ftfy==6.3.1 \
        regex==2026.5.9 \
        pyyaml==6.0.3 \
        pyarrow==24.0.0 \
        pandas==3.0.3 \
        tensorboard==2.20.0 \
        pydantic==2.12.5

RUN mkdir -p /data/hf /workspace /outputs \
    && chown -R 65532:65532 /data /outputs

WORKDIR /workspace
USER 65532:65532

HEALTHCHECK --interval=1m --timeout=30s --retries=2 \
    CMD python -c "import torch, diffusers, transformers, peft; print('ok')" || exit 1

CMD ["python", "-c", "import torch, diffusers, transformers, peft, accelerate; \
print('torch       ', torch.__version__); \
print('cuda        ', torch.version.cuda); \
print('gpus        ', torch.cuda.device_count()); \
print('diffusers   ', diffusers.__version__); \
print('transformers', transformers.__version__); \
print('peft        ', peft.__version__)"]
