#!/usr/bin/env bash
# Step 2: nsys captures on current main with SGLang's NUMA binding enabled.
#   p2-real - Low-Latency cell, DSpark real acceptance, 100 verify cycles
#   p2-off  - High-Throughput cell, DSpark off, 200 decode steps
set -euo pipefail
cd $GB300_WORK
source scripts/cells_cu13.sh

prof() {
  python3 scripts/profile_arm.py --image "$img" --cache-key cu13-ffac53d7 --port $port \
    --docker-arg=--cap-add=SYS_NICE "$@"
}
prof --name p2-real --num-steps 100 --server-cmd "sglang serve $common $low_latency $measure"
prof --name p2-off --num-steps 200 --server-cmd "sglang serve $common $high_throughput $measure"
echo STEP2_DONE
