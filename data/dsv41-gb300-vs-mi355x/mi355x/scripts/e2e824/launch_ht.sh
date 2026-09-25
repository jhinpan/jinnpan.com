#!/usr/bin/env bash
# Cookbook MI350X High-Throughput cell (DSpark off) plus measurement-only flags.
set -euo pipefail
source $E2E_CAMPAIGN/bench/env.sh
port=${1:?port}
shift
exec python -m sglang.launch_server \
  --model-path $MODEL_ROOT/DeepSeek-V4.1-Flash \
  --served-model-name deepseek-ai/DeepSeek-V4.1-Flash --trust-remote-code \
  --tp 4 --ep-size "${DSV41_EP_SIZE:-4}" --disable-radix-cache \
  --cuda-graph-backend-prefill breakable --cuda-graph-max-bs-prefill 4096 \
  --reasoning-parser auto --tool-call-parser auto \
  --enable-metrics --random-seed 42 --host 127.0.0.1 --port "$port" "$@"
