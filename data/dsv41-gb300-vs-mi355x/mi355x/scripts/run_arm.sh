#!/usr/bin/env bash
# run_arm.sh LABEL MODE BIND
#   One fresh MI355X server (TP4/EP4 on HIP 4-7), BBuf's BS=1 client
#   (1 warm-up + 6 timed, 4096 in / 1024 out, cache flushed per run), clean stop.
#   MODE: off = High-Throughput cell (DSpark off); real = Low-Latency cell;
#         sim  = Low-Latency cell with simulated acceptance 5.5 (BBuf's contract).
#   BIND: 0 = unbound; 1 = whole server pinned to NUMA node 1 (the node of HIP 4-7).
set -uo pipefail
C=$CAMPAIGN
E=$E2E_CAMPAIGN
PY=$CAMPAIGN_CACHE/env/bin/python
NODE1_CPUS=64-127,192-255
label=${1:?label} mode=${2:?mode} bind=${3:?bind}
port=30251
out=$C/results/$label
mkdir "$out" || { echo "refusing to overwrite $out" >&2; exit 2; }
case $mode in
  off) launcher=$E/bench/launch_ht.sh ;;
  real) launcher=$E/bench/launch_ll.sh ;;
  sim) launcher=$C/bench/launch_ll_sim.sh ;;
  *) echo "bad mode $mode" >&2; exit 2 ;;
esac
prefix=()
[ "$bind" = 1 ] && prefix=(taskset -c "$NODE1_CPUS")
utc() { date -u +%FT%T.%3NZ; }
t_start=$(utc)
cat > "$out/manifest.json" <<EOF
{"label": "$label", "mode": "$mode", "bind": $bind, "node1_cpus": "$NODE1_CPUS",
 "launcher": "$launcher", "launcher_sha256": "$(sha256sum "$launcher" | cut -d' ' -f1)",
 "env_sh_sha256": "$(sha256sum $E/bench/env.sh | cut -d' ' -f1)",
 "sglang_tree": "$E/sources/rt-base", "sglang_tree_manifest": $(cat $E/sources/rt-base.json),
 "aiter_tree": "$E/sources/aiter-acf8fdf9-e2e",
 "client_sha256": "$(sha256sum $C/bbuf/benchmark.py | cut -d' ' -f1)",
 "prompt_sha256": "$(sha256sum $C/bbuf/prompt.json | cut -d' ' -f1)",
 "gpu_lock_record": "${DSV41_GPU_LOCK_RECORD:-}", "start_utc": "$t_start"}
EOF

export BS1_SGLANG_ROOT=$E/sources/rt-base
setsid "${prefix[@]}" bash "$launcher" "$port" > "$out/server.log" 2>&1 &
sid=$!
ready=0
for _ in $(seq 1 450); do
  kill -0 "$sid" 2>/dev/null || break
  if curl -sf -m 2 "http://127.0.0.1:$port/health" > /dev/null; then ready=1; break; fi
  sleep 2
done
t_ready=$(utc)
crc=99
if [ "$ready" = 1 ]; then
  $PY $C/bench/snapshot_procs.py "$sid" "$out/processes.json" > /dev/null 2>&1
  $PY $C/bbuf/benchmark.py bench --url "http://127.0.0.1:$port" --prompt $C/bbuf/prompt.json \
    --max-tokens 1024 --out "$out/bench" --repeat 6 > "$out/client.log" 2>&1
  crc=$?
fi
t_bench=$(utc)
kill -TERM -- -"$sid" 2>/dev/null
for _ in $(seq 1 45); do kill -0 -- -"$sid" 2>/dev/null || break; sleep 2; done
kill -KILL -- -"$sid" 2>/dev/null
pkill -KILL -s "$sid" 2>/dev/null
$PY $C/bench/wait_free.py 300 > "$out/teardown.log" 2>&1
frc=$?
ok=false
[ "$ready" = 1 ] && [ "$crc" = 0 ] && [ -f "$out/bench/summary.json" ] && [ "$frc" = 0 ] && ok=true
cat > "$out/status.json" <<EOF
{"label": "$label", "ok": $ok, "server_ready": $ready, "client_rc": $crc, "gpus_released_rc": $frc,
 "start_utc": "$t_start", "ready_utc": "$t_ready", "bench_end_utc": "$t_bench", "end_utc": "$(utc)"}
EOF
echo "$label ok=$ok ready=$ready client_rc=$crc"
[ "$ok" = true ]
