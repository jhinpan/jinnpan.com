#!/usr/bin/env bash
# Step 1A: BBuf's random 4K prompt and client on current main (dev-cu13 = ffac53d779),
# cookbook GB300 cells (TP4/EP4) plus the shared measurement flags.
#   off  - High-Throughput cell, DSpark off
#   sim  - Low-Latency cell, DSpark with simulated acceptance 5.5 (match-expected)
#   real - Low-Latency cell, DSpark with real acceptance
# ARM_EXTRA (array-like string) is appended to every arm.py call.
set -euo pipefail
cd $GB300_WORK
source scripts/cells_cu13.sh
read -r -a arm_extra <<< "${ARM_EXTRA:-}"

run() {
  local name=$1 cell=$2
  shift 2
  python3 scripts/arm.py --name "$name" --image "$img" --cache-key cu13-ffac53d7 --port $port \
    "${arm_extra[@]}" "$@" --server-cmd "sglang serve $common $cell $measure" --bench "$bbuf_bench"
}

for arm in "$@"; do
  case $arm in
    *-off-*)  run "$arm" "$high_throughput" ;;
    *-sim-*)  run "$arm" "$low_latency" "${sim_env[@]}" ;;
    *-real-*) run "$arm" "$low_latency" ;;
    *) echo "unknown arm $arm"; exit 2 ;;
  esac
done
echo STEP1A_DONE
