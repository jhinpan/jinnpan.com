#!/usr/bin/env bash
# Session 3 (MI355X, inside an exclusive lease of HIP 4-7):
#   graph floor  - graph_floor.py on HIP 4, the same script GB300 runs on GPU0
#   calibration  - graph_floor.py --quick with and without rocprofv3 kernel tracing, to test
#                  whether traced small-kernel durations can be trusted (Kineto's cannot)
#   ledger       - kernel-traced Low-Latency (natural acceptance) and High-Throughput servers
set -uo pipefail
cd $CAMPAIGN
PY=$CAMPAIGN_CACHE/env/bin/python
GF=$BS1_FLOOR/bench/graph_floor.py
T=$ROCPROF_TRACES
LOG() { echo "[$(date -u +%FT%TZ)] $*"; }
mkdir -p results/mb-hip "$T"

LOG "start graph floor"
HIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4 $PY $GF --output results/mb-hip/default.json --label default \
  > results/mb-hip/default.log 2>&1
LOG "end graph floor rc=$?"
HIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4 $PY $GF --quick --output results/mb-hip/quick.json --label quick \
  > results/mb-hip/quick.log 2>&1
LOG "end quick untraced rc=$?"
HIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4 /opt/rocm/bin/rocprofv3 --kernel-trace --output-format csv \
  -d "$T/cal" -- $PY $GF --quick --output results/mb-hip/quick-traced.json --label quick-traced \
  > results/mb-hip/quick-traced.log 2>&1
LOG "end quick traced rc=$?"

for spec in "rp-real-a real" "rp-off-a off"; do
  set -- $spec
  LOG "start $1"
  bash bench/run_rocprof_arm.sh "$1" "$2"
  LOG "end $1 rc=$?"
done
LOG SESSION3_DONE
