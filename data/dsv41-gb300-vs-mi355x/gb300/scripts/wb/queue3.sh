#!/usr/bin/env bash
# Session 3 on the GB300 workbench (current main ffac53d779, NUMA-bound via CAP_SYS_NICE):
#   stream concurrency A/B  - base (default), opt0 (SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=0),
#                             serial (opt0 + shared experts on the forward stream)
#   serial nsys attribution - p3-serial-real (100 verify cycles), p3-serial-off (200 steps)
#   real-text BS=1 contract - the MI355X contract's four prompts, 2 launches
#   graph floor             - the MI355X graph_floor.py microbenchmark on GPU0
# The idle workbench server is stopped first and restored with its own command at the end.
set -uo pipefail
cd /work
W=/work/scripts/wb
LOG() { echo "[$(date -u +%FT%TZ)] $*"; }

common="--model-path /models/DeepSeek-V4.1-Flash --served-model-name deepseek-ai/DeepSeek-V4.1-Flash --trust-remote-code --tp 4 --ep-size 4"
measure="--disable-radix-cache --reasoning-parser auto --tool-call-parser auto --enable-metrics --random-seed 42 --host 127.0.0.1 --port 30000"
low_latency="--mem-fraction-static 0.8 --speculative-algorithm DSPARK --speculative-dspark-block-size 5 --cuda-graph-max-bs-decode 64"
high_throughput="--max-running-requests 256"
LL="$common $low_latency $measure"
HT="$common $high_throughput $measure"
bbuf="python3 bbuf/benchmark.py bench --prompt bbuf/prompt.json --max-tokens 1024 --out {out}/bench --repeat 6 --url http://127.0.0.1:30000"
realtext="python3 $W/bs1_realtext.py --inputs $W/realtext-inputs.json --out {out}/realtext"
sim=(--env SGLANG_RAGGED_VERIFY_MODE=static --env SGLANG_SIMULATE_ACC_LEN=5.5 --env SGLANG_SIMULATE_ACC_METHOD=match-expected)
opt0=(--env SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=0)
serial=(--env SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=0 --pythonpath $W/serial_moe)

arm() {  # arm NAME VARIANT MODE
  local name=$1 variant=$2 mode=$3 v=() m=() args
  case $variant in base) ;; opt0) v=("${opt0[@]}") ;; serial) v=("${serial[@]}") ;; esac
  case $mode in sim) m=("${sim[@]}"); args=$LL ;; real) args=$LL ;; off) args=$HT ;; esac
  LOG "start $name"
  python3 $W/wb_arm.py --name "$name" --server-args "$args" "${v[@]}" "${m[@]}" --record-placement \
    --bench "$bbuf"
  LOG "end $name rc=$?"
}

# 0. Stop the idle workbench server (one smoke request at 03:36 UTC, nothing since).
wb=$(pgrep -f '^/opt/sglang/bin/python3 /opt/sglang/bin/sglang serve .*--port 30000' | head -1)
if [ -n "$wb" ]; then
  tr '\0' ' ' < /proc/$wb/cmdline > /work/logs/workbench-server.cmdline
  for _ in $(seq 1 30); do
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv,noheader
    sleep 1
  done > /work/logs/workbench-idle-window.csv 2>&1
  if awk -F', ' '$3+0 > 0 {busy=1} END {exit busy}' /work/logs/workbench-idle-window.csv; then
    LOG "stopping idle workbench server pid=$wb"
    kill -TERM "$wb"
    for _ in $(seq 1 60); do pgrep -f '^sglang::' > /dev/null || break; sleep 3; done
    pgrep -f '^sglang::' > /dev/null && { LOG "workbench server did not stop; aborting"; exit 3; }
  else
    LOG "workbench server busy during the idle window; aborting"; exit 3
  fi
fi

# 1. Graph floor microbenchmark and the serial patch preflight (no model).
LOG "start graph floor"
mkdir -p /work/results/mb-cuda
CUDA_VISIBLE_DEVICES=0 python3 $W/graph_floor.py --output /work/results/mb-cuda/default.json --label default \
  > /work/results/mb-cuda/default.log 2>&1
LOG "end graph floor rc=$?"
PYTHONPATH=$W/serial_moe python3 -c "import sglang.srt.models.deepseek_v2 as m; print(m.DeepseekV2MoE.__init__.__qualname__)" \
  > /work/results/mb-cuda/serial-preflight.log 2>&1
LOG "serial preflight rc=$? $(tail -2 /work/results/mb-cuda/serial-preflight.log | tr '\n' ' ')"

# 2. Stream concurrency A/B, mirrored in time.
for mode in sim off; do
  arm w3-base-$mode-a base $mode
  arm w3-serial-$mode-a serial $mode
  arm w3-opt0-$mode-a opt0 $mode
  arm w3-opt0-$mode-b opt0 $mode
  arm w3-serial-$mode-b serial $mode
  arm w3-base-$mode-b base $mode
done

# 3. Serialized-stream attribution (same windows as p2-real / p2-off).
LOG "start p3-serial-real"
python3 $W/wb_arm.py --name p3-serial-real --server-args "$LL" "${serial[@]}" --profile-steps 100
LOG "end p3-serial-real rc=$?"
LOG "start p3-serial-off"
python3 $W/wb_arm.py --name p3-serial-off --server-args "$HT" "${serial[@]}" --profile-steps 200
LOG "end p3-serial-off rc=$?"

# 4. Real-text BS=1 contract at the default (production) stream layout.
for x in a b; do
  LOG "start w3-rt-$x"
  python3 $W/wb_arm.py --name w3-rt-$x --server-args "$LL" --record-placement --bench "$realtext"
  LOG "end w3-rt-$x rc=$?"
done

# 5. Restore the workbench server with its own command line.
if [ -s /work/logs/workbench-server.cmdline ]; then
  LOG "restoring workbench server"
  setsid nohup bash -c "exec $(cat /work/logs/workbench-server.cmdline)" \
    > /work/logs/workbench-server-restored.log 2>&1 < /dev/null &
  for _ in $(seq 1 120); do curl -sf -m 2 http://127.0.0.1:30000/health > /dev/null && break; sleep 5; done
  LOG "workbench server health rc=$(curl -sf -m 2 -o /dev/null -w '%{http_code}' http://127.0.0.1:30000/health)"
fi
LOG QUEUE3_DONE
