#!/usr/bin/env bash
# Persistent root container for interactive work on current main (dev-cu13 = ffac53d779):
# GPUs, local checkpoint, workspace, the shared JIT cache, and CAP_SYS_NICE so SGLang's
# NUMA binding applies. Enter with: docker exec -it dsv41-workbench bash
# Measurement arms tolerate this container (label dsv41.role=workbench) only while the
# GPUs are idle, so stop any server started in it before running a queue.
set -euo pipefail
work=$GB300_WORK
source "$work/scripts/cells_cu13.sh"
name=dsv41-workbench

if docker inspect "$name" >/dev/null 2>&1; then
  docker start "$name" >/dev/null
else
  docker run -d --init --name "$name" --label dsv41.role=workbench \
    --gpus all --ipc=host --shm-size 32g --network host --cap-add SYS_NICE \
    -v $GB300_MODEL_DIR:/models/DeepSeek-V4.1-Flash:ro \
    -v "$work:/work" -v "$work/cache/cu13-ffac53d7:/root/.cache" -w /work \
    -e TRITON_CACHE_DIR=/root/.cache/triton -e DG_JIT_CACHE_DIR=/root/.cache/deep_gemm \
    -e TILELANG_CACHE_DIR=/root/.cache/tilelang -e CUPY_CACHE_DIR=/root/.cache/cupy \
    -e MODEL_PATH=/models/DeepSeek-V4.1-Flash \
    "$img" sleep infinity >/dev/null
fi

# Tools the image lacks: the nvtx package that SGLANG_ENABLE_NVTX_* needs, and jq/nano.
# Outbound port 80 is blocked here, so apt must use the HTTPS mirror.
docker exec "$name" bash -c '
  set -e
  python3 -c "import nvtx" 2>/dev/null || pip install -q nvtx
  sed -i -e "s|http://ports.ubuntu.com|https://ports.ubuntu.com|g" \
         -e "s|http://developer.download.nvidia.com|https://developer.download.nvidia.com|g" \
         /etc/apt/sources.list.d/*
  if ! command -v jq >/dev/null || ! command -v nano >/dev/null; then
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq jq nano >/dev/null
  fi'
docker ps --filter "name=^/$name\$" --format '{{.Names}}  {{.Status}}  {{.Image}}'
