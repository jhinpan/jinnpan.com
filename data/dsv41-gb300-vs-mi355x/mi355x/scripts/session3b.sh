#!/usr/bin/env bash
# Session 3b (MI355X, exclusive lease of HIP 4-7). rocprofv3 1.1.0 kernel tracing aborts the
# first replay of a packet-captured HIP graph (HSA_STATUS_ERROR_INVALID_PACKET_FORMAT), so every
# run here disables graph packet capture. Kernel durations are what the ledger needs; the
# larger dispatch gaps this causes are not used. The server arms run only if the traced
# microbenchmark completes.
set -uo pipefail
cd $CAMPAIGN
PY=$CAMPAIGN_CACHE/env/bin/python
GF=$BS1_FLOOR/bench/graph_floor.py
T=$ROCPROF_TRACES
LOG() { echo "[$(date -u +%FT%TZ)] $*"; }
export DEBUG_CLR_GRAPH_PACKET_CAPTURE=0

HIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4 $PY $GF --quick --output results/mb-hip/quick-nopc.json \
  --label quick-nopc > results/mb-hip/quick-nopc.log 2>&1
LOG "end quick no-packet-capture rc=$?"
HIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4 /opt/rocm/bin/rocprofv3 --kernel-trace --output-format csv \
  -d "$T/cal-nopc" -- $PY $GF --quick --output results/mb-hip/quick-nopc-traced.json --label quick-nopc-traced \
  > results/mb-hip/quick-nopc-traced.log 2>&1
rc=$?
LOG "end quick no-packet-capture traced rc=$rc"
[ "$rc" = 0 ] || { LOG "tracing still fails; skipping server arms"; exit 0; }

for spec in "rp-real-b real" "rp-off-b off"; do
  set -- $spec
  LOG "start $1"
  bash bench/run_rocprof_arm.sh "$1" "$2"
  LOG "end $1 rc=$?"
done
LOG SESSION3B_DONE
