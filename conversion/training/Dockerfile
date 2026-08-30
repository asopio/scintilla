FROM nvidia/cuda:12.9.2-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/root/.pixi/bin:${PATH} \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/mnt/persistent-storage/hf_home \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/compat/lib.real

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    ca-certificates \
    bash \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://pixi.sh/install.sh | sh

WORKDIR /opt/llm-doc-converter

COPY . /opt/llm-doc-converter

RUN pixi install -m /opt/llm-doc-converter/training/pixi.toml

CMD ["/bin/bash"]
