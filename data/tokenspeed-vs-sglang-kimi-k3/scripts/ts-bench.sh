#!/bin/bash
# Serve Kimi-K3 with TokenSpeed on 8x gfx950 and benchmark it on the same
# workloads the SGLang data on this box already covers.
#
# Metric definitions match: both bench harnesses descend from vLLM's
# benchmark_serving.py and emit the same "Output token throughput"/TPOT/TTFT.
# The one incompatibility is --random-range-ratio, whose meaning is inverted
# between the two: SGLang's 1 == TokenSpeed's 0 == exact lengths.
set -uo pipefail

TS_ROOT=/sgl-workspace/tokenspeed
VENV=$TS_ROOT/.venv
OUT=${OUT:-/sgl-workspace/workspace/ts_results/$(date +%Y%m%d_%H%M%S)}
PORT=${PORT:-8100}
MODEL=${MODEL:-moonshotai/Kimi-K3}
TOKENIZER=${TOKENIZER:-/sgl-workspace/models/Kimi-K3-flat}
MAXLEN=${MAXLEN:-8192}
MAXSEQS=${MAXSEQS:-32}
GPUFRAC=${GPUFRAC:-0.92}
# isl:osl:conc:num_prompts:warmups
WORKLOADS=${WORKLOADS:-"4096:1024:1:3:1 1024:1024:1:4:1 1024:1024:8:16:2 1024:1024:32:64:4"}
BOOT_TIMEOUT=${BOOT_TIMEOUT:-3600}

# Same mutex every other harness on this box takes.
exec 9>/tmp/k3-grid.lock
flock -n 9 || { echo "another run holds /tmp/k3-grid.lock; refusing to start"; exit 1; }
echo "$$" >&9

mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
echo "=== $(date -Is) TokenSpeed K3 run -> $OUT ==="
echo "model=$MODEL maxlen=$MAXLEN maxseqs=$MAXSEQS gpufrac=$GPUFRAC port=$PORT"

# shellcheck disable=SC1091
source "$VENV/bin/activate"
export HF_HOME=/sgl-workspace/models
export HF_HUB_OFFLINE=1
export HF_MODULES_CACHE=/sgl-workspace/workspace/.ts_hf_modules
mkdir -p "$HF_MODULES_CACHE"

SERVER_LOG="$OUT/server.log"
cleanup() {
    echo "=== $(date -Is) shutting down ==="
    [[ -n "${SRV_PID:-}" ]] && kill "$SRV_PID" 2>/dev/null
    for _ in $(seq 60); do kill -0 "${SRV_PID:-0}" 2>/dev/null || break; sleep 2; done
    kill -9 "${SRV_PID:-0}" 2>/dev/null
    pkill -9 -f "tokenspeed.*serve|ts serve" 2>/dev/null
    sleep 5
}
trap cleanup EXIT

# Two deviations from docs/recipes/models.md, both deliberate:
#   --disable-prefill-graph     capturing the 40 prefill buckets drains the ~20 GB
#                               left after weights+KV and OOMs; TokenSpeed's own
#                               CI perf config disables it for the same reason.
#   --no-enable-prefix-caching  the SGLang baselines on this box ran with
#                               --disable-radix-cache, so match them.
echo "=== booting server (log: $SERVER_LOG) ==="
tokenspeed serve "$MODEL" \
    --served-model-name kimi-k3 \
    --trust-remote-code \
    --max-model-len "$MAXLEN" \
    --kv-cache-dtype fp8 \
    --tensor-parallel-size 8 \
    --mm-encoder-tp-mode data \
    --enable-expert-parallel \
    --attention-backend mla \
    --moe-backend auto \
    --gpu-memory-utilization "$GPUFRAC" \
    --max-num-seqs "$MAXSEQS" \
    --disable-kvstore \
    --disable-prefill-graph \
    --no-enable-prefix-caching \
    --host 127.0.0.1 \
    --port "$PORT" \
    > "$SERVER_LOG" 2>&1 &
SRV_PID=$!

BASE_URL="http://127.0.0.1:$PORT"
echo "=== waiting for readiness (pid $SRV_PID, up to ${BOOT_TIMEOUT}s) ==="
boot_start=$(date +%s)
ready=0
while (( $(date +%s) - boot_start < BOOT_TIMEOUT )); do
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
        echo "SERVER DIED after $(( $(date +%s) - boot_start ))s"
        tail -60 "$SERVER_LOG"; exit 1
    fi
    if curl -sf "$BASE_URL/readiness" >/dev/null 2>&1; then ready=1; break; fi
    sleep 10
done
(( ready )) || { echo "BOOT TIMEOUT"; tail -60 "$SERVER_LOG"; exit 1; }
echo "=== ready after $(( $(date +%s) - boot_start ))s ==="

echo "=== smoke test ==="
curl -s "$BASE_URL/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "kimi-k3",
  "messages": [{"role": "user", "content": "Explain why the sky appears blue in two sentences."}],
  "temperature": 0, "max_tokens": 128 }' | tee "$OUT/smoke.json" | head -c 1200
echo

for w in $WORKLOADS; do
    IFS=: read -r isl osl conc np warm <<<"$w"
    label="k3-${isl}-${osl}-c${conc}"
    echo "=== $(date -Is) workload $label (np=$np warmups=$warm) ==="
    tokenspeed bench serve \
        --base-url "$BASE_URL" \
        --model kimi-k3 \
        --tokenizer "$TOKENIZER" \
        --dataset-name random \
        --input-len "$isl" \
        --output-len "$osl" \
        --random-range-ratio 0 \
        --num-prompts "$np" \
        --max-concurrency "$conc" \
        --num-warmups "$warm" \
        --ready-check-timeout-sec 0 \
        --ignore-eos \
        --extra-body '{"temperature": 0}' \
        --percentile-metrics ttft,tpot,itl,e2el \
        --metric-percentiles 50,90,99 \
        --label "$label" \
        --save-result --result-dir "$OUT" \
        2>&1 | tee "$OUT/$label.txt"
done

echo "=== $(date -Is) all workloads done ==="
