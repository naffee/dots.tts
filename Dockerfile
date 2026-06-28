FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/runpod-volume/hf \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/hf/hub \
    DOTS_TTS_MODEL=rednote-hilab/dots.tts-soar \
    DOTS_TTS_PRECISION=bfloat16 \
    DOTS_TTS_OPTIMIZE=0 \
    DOTS_TTS_MAX_GENERATE_LENGTH=500

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install runpod && \
    python -m pip install -c constraints/recommended.txt -e .

CMD ["python", "-u", "handler.py"]
