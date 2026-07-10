FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        curl \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3.11-distutils \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src

RUN python3.11 -m pip install --no-cache-dir .

# Pre-download the MediaPipe hand landmarker model at build time so it's baked
# into the image rather than fetched on every container run.
RUN python3.11 -c "from nurvai_pipeline.hand_tracking import ensure_model; ensure_model()"

ENTRYPOINT ["nurvai-pipeline"]
CMD ["--help"]
