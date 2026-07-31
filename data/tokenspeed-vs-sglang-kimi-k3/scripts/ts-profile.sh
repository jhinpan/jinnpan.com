#!/bin/bash
# Capture Proton + VizTracer traces of TokenSpeed serving Kimi-K3 on TP8, so the
# per-kernel decode/prefill composition can be compared against the existing
# SGLang kda_prof traces on this box.
#
# Methodology is matched to that SGLang baseline deliberately:
#   - batch 1, ISL 4096, no speculative decoding
#   - graph OFF (--enforce-eager). The SGLang composition traces were captured
#     graph-off too; inside a replayed HIP graph the individual decode kernels
#     are not separately attributable.
#   - prefix caching off
set -uo pipefail

TS_ROOT=/sgl-workspace/tokenspeed
VENV=$TS_ROOT/.venv
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=${OUT:-/sgl-workspace/workspace/ts_prof/$STAMP}
TRACE_DIR="$OUT/traces"
PORT=${PORT:-8100}
MODEL=${MODEL:-moonshotai/Kimi-K3}
TOKENIZER=${TOKENIZER:-/sgl-workspace/models/Kimi-K3-flat}
ISL=${ISL:-4096}
NUM_STEPS=${NUM_STEPS:-16}
PROFILE_ID=${PROFILE_ID:-kimi-k3-tp8}
BOOT_TIMEOUT=${BOOT_TIMEOUT:-3600}

exec 9>/tmp/k3-grid.lock
flock -n 9 || { echo "another run holds /tmp/k3-grid.lock; refusing to start"; exit 1; }
echo "$$" >&9

mkdir -p "$TRACE_DIR"
exec > >(tee -a "$OUT/run.log") 2>&1
echo "=== $(date -Is) TokenSpeed K3 profiling -> $OUT ==="

# shellcheck disable=SC1091
source "$VENV/bin/activate"
export HF_HOME=/sgl-workspace/models
export HF_HUB_OFFLINE=1
export HF_MODULES_CACHE=/sgl-workspace/workspace/.ts_hf_modules

# Proton cannot be used here: on ROCm it calls rocprofiler_force_configure, which
# must run before HIP is initialised, so attaching it to a live server fails with
# error 16. The torch/roctracer profiler attaches fine at runtime and emits the
# same chrome-trace artifact the SGLang kda_prof baseline used, which makes the
# two sides directly comparable.
export TOKENSPEED_PROFILER_DIR="$TRACE_DIR"

SERVER_LOG="$OUT/server.log"
SRV_PID=""
cleanup() {
    echo "=== $(date -Is) shutting down ==="
    [[ -n "$SRV_PID" ]] && kill "$SRV_PID" 2>/dev/null
    for _ in $(seq 60); do kill -0 "${SRV_PID:-0}" 2>/dev/null || break; sleep 2; done
    kill -9 "${SRV_PID:-0}" 2>/dev/null
    sleep 5
}
trap cleanup EXIT

echo "=== booting (eager, prefix-cache off) ==="
tokenspeed serve "$MODEL" \
    --served-model-name kimi-k3 \
    --trust-remote-code \
    --max-model-len 8192 \
    --kv-cache-dtype fp8 \
    --tensor-parallel-size 8 \
    --mm-encoder-tp-mode data \
    --enable-expert-parallel \
    --attention-backend mla \
    --moe-backend auto \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 32 \
    --disable-kvstore \
    --disable-prefill-graph \
    --no-enable-prefix-caching \
    --enforce-eager \
    --host 127.0.0.1 --port "$PORT" \
    > "$SERVER_LOG" 2>&1 &
SRV_PID=$!

BASE_URL="http://127.0.0.1:$PORT"
boot_start=$(date +%s); ready=0
while (( $(date +%s) - boot_start < BOOT_TIMEOUT )); do
    kill -0 "$SRV_PID" 2>/dev/null || { echo "SERVER DIED"; tail -40 "$SERVER_LOG"; exit 1; }
    curl -sf "$BASE_URL/readiness" >/dev/null 2>&1 && { ready=1; break; }
    sleep 10
done
(( ready )) || { echo "BOOT TIMEOUT"; tail -40 "$SERVER_LOG"; exit 1; }
echo "=== ready after $(( $(date +%s) - boot_start ))s ==="

echo "=== warmup (untraced), so JIT/autotune does not land in the trace ==="
tokenspeed bench serve --base-url "$BASE_URL" --model kimi-k3 --tokenizer "$TOKENIZER" \
    --dataset-name random --input-len "$ISL" --output-len 8 --random-range-ratio 0 \
    --num-prompts 2 --max-concurrency 1 --ready-check-timeout-sec 0 --ignore-eos \
    --extra-body '{"temperature": 0}' > "$OUT/warmup.txt" 2>&1
echo "warmup rc=$?"

echo "=== arming profiler (${NUM_STEPS} steps, stage-separated, GPU activities) ==="
curl -sS -X POST "$BASE_URL/start_profile" -H 'Content-Type: application/json' -d "{
    \"num_steps\": $NUM_STEPS,
    \"activities\": [\"GPU\"],
    \"profile_by_stage\": true,
    \"profile_id\": \"$PROFILE_ID\"
  }" | tee "$OUT/start_profile.json"
echo

echo "=== traced request: ISL $ISL, batch 1, enough output to capture decode ==="
tokenspeed bench serve --base-url "$BASE_URL" --model kimi-k3 --tokenizer "$TOKENIZER" \
    --dataset-name random --input-len "$ISL" --output-len 24 --random-range-ratio 0 \
    --num-prompts 1 --max-concurrency 1 --ready-check-timeout-sec 0 --ignore-eos \
    --extra-body '{"temperature": 0}' 2>&1 | tee "$OUT/traced-request.txt"

echo "=== waiting for trace files to land ==="
for _ in $(seq 90); do
    n=$(ls "$TRACE_DIR"/*.trace.json.gz 2>/dev/null | wc -l)
    echo "  $n trace files in $TRACE_DIR"
    [[ "$n" -ge 16 ]] && break
    sleep 5
done
ls -la "$TRACE_DIR" | head -40

echo "=== $(date -Is) profiling done ==="
ls -la "$OUT"
