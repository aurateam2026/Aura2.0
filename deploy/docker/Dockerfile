# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

FROM python:3.12-slim-bookworm

ARG UV_VERSION=0.9.8
ARG VLLM_WHEEL=https://wheels.vllm.ai/4ba787ec9959e566d3b587a796dff07673324da5/vllm-0.33.0-cp38-abi3-manylinux_2_28_x86_64.whl
ENV AURA_HOME=/var/lib/aura \
    AURA_VENV=/opt/aura-venv \
    HF_HOME=/var/lib/aura/cache/huggingface \
    UV_LINK_MODE=copy \
    PATH=/opt/aura-venv/bin:$PATH \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git ffmpeg libgomp1 libglib2.0-0 \
    build-essential pkg-config tini util-linux && \
    rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "uv==${UV_VERSION}" && \
    uv venv --python /usr/local/bin/python "$AURA_VENV" --seed && \
    uv pip install --no-cache --python "$AURA_VENV/bin/python" \
    --torch-backend=cu130 "vllm @ ${VLLM_WHEEL}"

WORKDIR /opt/aura
COPY . .
RUN VLLM_OMNI_VERSION_OVERRIDE=0.33.0.dev0 \
    uv pip install --no-cache --python "$AURA_VENV/bin/python" -e /opt/aura && \
    uv pip check --python "$AURA_VENV/bin/python"

# Model weights are mounted/downloaded at runtime. Kernel downloads and CUDA
# compatibility checks also run at startup, when the host accelerator is available.
EXPOSE 9310
ENTRYPOINT ["/usr/bin/tini", "-g", "--", "bash", "/opt/aura/deploy/docker/entrypoint.sh"]
CMD ["--aura", "/models/aura", "--gpus", "0,1"]
