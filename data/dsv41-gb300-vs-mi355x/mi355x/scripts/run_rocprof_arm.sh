#!/usr/bin/env bash
# run_rocprof_arm.sh LABEL MODE
#   One MI355X server (TP4/EP4 on HIP 4-7, SGLang's split affinity) under rocprofv3 kernel
#   tracing, BBuf's client with one warm-up and one timed request, then a graceful stop.
#   MODE: real = Low-Latency cell, natural acceptance; off = High-Throughput cell.
#   rocprofv3 writes a process's trace from its own signal handler, and in a process with
#   children that handler waits for the children instead of running SGLang's shutdown. So the
#   schedulers and detokenizer (no children) get SIGTERM first and flush their traces, then the
#   HTTP server; the process group is killed only if that has not finished within 180 s.
set -uo pipefail
C=$CAMPAIGN
E=$E2E_CAMPAIGN
PY=$CAMPAIGN_CACHE/env/bin/python
TRACE_ROOT=$ROCPROF_TRACES
label=${1:?label} mode=${2:?mode}
port=30251
out=$C/results/$label
trace=$TRACE_ROOT/$label
mkdir "$out" || { echo "refusing to overwrite $out" >&2; exit 2; }
mkdir -p "$trace"
case $mode in
  off) launcher=$E/bench/launch_ht.sh ;;
  real) launcher=$E/bench/launch_ll.sh ;;
  *) echo "bad mode $mode" >&2; exit 2 ;;
esac
utc() { date -u +%FT%T.%3NZ; }
cat > "$out/manifest.json" <<EOF
{"label": "$label", "mode": "$mode", "affinity_mode": "split", "launcher": "$launcher",
 "launcher_sha256": "$(sha256sum "$launcher" | cut -d' ' -f1)", "sglang_tree": "$E/sources/rt-base",
 "rocprofv3": "$(/opt/rocm/bin/rocprofv3 --version 2>&1 | awk '/version:/ {print $2}')",
 "trace_dir": "$trace", "gpu_lock_record": "${DSV41_GPU_LOCK_RECORD:-}", "start_utc": "$(utc)"}
EOF
export BS1_SGLANG_ROOT=$E/sources/rt-base
setsid /opt/rocm/bin/rocprofv3 --kernel-trace --output-format csv -d "$trace" -- \
  bash "$launcher" "$port" > "$out/server.log" 2>&1 &
sid=$!
ready=0
for _ in $(seq 1 600); do
  kill -0 "$sid" 2>/dev/null || break
  # A queue abort leaves the schedulers hung rather than exited.
  grep -q "aborting with error" "$out/server.log" && break
  if curl -sf -m 2 "http://127.0.0.1:$port/health" > /dev/null; then ready=1; break; fi
  sleep 2
done
t_ready=$(utc)
crc=99
if [ "$ready" = 1 ]; then
  $PY $C/bbuf/benchmark.py bench --url "http://127.0.0.1:$port" --prompt $C/bbuf/prompt.json \
    --max-tokens 1024 --out "$out/bench" --repeat 1 > "$out/client.log" 2>&1
  crc=$?
fi
t_bench=$(utc)
workers=$(pgrep -s "$sid" -f '^sglang::')
[ -n "$workers" ] && kill -TERM $workers 2>/dev/null
for _ in $(seq 1 60); do
  pgrep -s "$sid" -f '^sglang::' > /dev/null || break
  sleep 2
done
http=$(pgrep -f -s "$sid" 'sglang.launch_server' | head -1)
[ -n "$http" ] && kill -TERM "$http" 2>/dev/null
graceful=0
for _ in $(seq 1 30); do
  pgrep -s "$sid" > /dev/null || { graceful=1; break; }
  sleep 2
done
kill -KILL -- -"$sid" 2>/dev/null
pkill -KILL -s "$sid" 2>/dev/null
$PY $C/bench/wait_free.py 300 > "$out/teardown.log" 2>&1
frc=$?
traces=$(find "$trace" -name '*kernel_trace.csv' | wc -l)
ok=false
[ "$ready" = 1 ] && [ "$crc" = 0 ] && [ "$frc" = 0 ] && [ "$traces" -ge 4 ] && ok=true
cat > "$out/status.json" <<EOF
{"label": "$label", "ok": $ok, "server_ready": $ready, "client_rc": $crc, "graceful_exit": $graceful,
 "gpus_released_rc": $frc, "kernel_trace_files": $traces, "ready_utc": "$t_ready",
 "bench_end_utc": "$t_bench", "end_utc": "$(utc)"}
EOF
echo "$label ok=$ok ready=$ready client_rc=$crc graceful=$graceful traces=$traces"
[ "$ok" = true ]
