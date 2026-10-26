#!/usr/bin/env bash
# glm5.3-fp8-mi355x-sglang-agentic-mtp-1019d64e6136.yaml (override_tp8_c2); delta: arg monokernel-backend: None -> 'amd'
exec env \
  PYTHONNOUSERSITE=1 \
  SGLANG_OPT_USE_TOPK_V2=false \
  SGLANG_SIMULATE_ACC_LEN=2.99 \
  SGLANG_SIMULATE_ACC_METHOD=match-expected \
  SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token \
  SGLANG_TIMEOUT_KEEP_ALIVE=900 \
  python3 -m sglang.launch_server \
  --model-path zai-org/GLM-5.3 \
  --host 127.0.0.1 \
  --port 31000 \
  --chunked-prefill-size 32768 \
  --cuda-graph-max-bs-decode 4 \
  --dsa-decode-backend tilelang \
  --dsa-prefill-backend tilelang \
  --enable-metrics \
  --expert-parallel-size 1 \
  --kv-cache-dtype fp8_e4m3 \
  --max-running-requests 4 \
  --mem-fraction-static 0.85 \
  --monokernel-backend amd \
  --reasoning-parser glm45 \
  --served-model-name zai-org/GLM-5.3 \
  --speculative-algorithm EAGLE \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --speculative-num-steps 3 \
  --tensor-parallel-size 8 \
  --tool-call-parser glm47 \
  --trust-remote-code \
  --watchdog-timeout 1800
