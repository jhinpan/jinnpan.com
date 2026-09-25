# Sourced by the dev-cu13 drivers: image, cookbook GB300 cells, shared measurement flags.
img=lmsysorg/sglang@sha256:b8257f5c5f8f7c5aeeba7e6f230c8cb1a6af8b010bf1d906d3d05f72128c1d7c
port=30000
bbuf_bench="python3 bbuf/benchmark.py bench --prompt bbuf/prompt.json --max-tokens 1024 --out {out}/bench --repeat 6 --url http://127.0.0.1:$port"

common="--model-path /models/DeepSeek-V4.1-Flash --served-model-name deepseek-ai/DeepSeek-V4.1-Flash --trust-remote-code --tp 4 --ep-size 4"
measure="--disable-radix-cache --reasoning-parser auto --tool-call-parser auto --enable-metrics --random-seed 42 --host 127.0.0.1 --port $port"
low_latency="--mem-fraction-static 0.8 --speculative-algorithm DSPARK --speculative-dspark-block-size 5 --cuda-graph-max-bs-decode 64"
high_throughput="--max-running-requests 256"
sim_env=(--env SGLANG_RAGGED_VERIFY_MODE=static --env SGLANG_SIMULATE_ACC_LEN=5.5 --env SGLANG_SIMULATE_ACC_METHOD=match-expected)
