#!/usr/bin/env bash
# Step 0: reproduce BBuf's GB300 numbers with his code (835c3909), image (dev-dsv41),
# launchers and client. Order alternates configs like his TP->EP->TP->EP run.
# ARM_EXTRA (space-separated) is appended to every arm.py call.
set -euo pipefail
cd $GB300_WORK
read -r -a arm_extra <<< "${ARM_EXTRA:-}"

img=lmsysorg/sglang@sha256:4a5d132a06a77c8331e15845f2e925adc788b00105097ad55409afa3f4fa4860
bench="python3 bbuf/benchmark.py bench --prompt bbuf/prompt.json --max-tokens 1024 --out {out}/bench --repeat 6 --url http://127.0.0.1:30021"

run() {
  local name=$1 launcher=$2
  shift 2
  python3 scripts/arm.py --name "$name" --image "$img" --cache-key dsv41-835c3909 \
    --workdir /work/src/bbuf-sglang-835c3909 --port 30021 \
    --env MODEL_PATH=/models/DeepSeek-V4.1-Flash "${arm_extra[@]}" "$@" \
    --server-cmd "bash $launcher" --bench "$bench"
}

for arm in "$@"; do
  case $arm in
    *-tp4-*)  run "$arm" /work/bbuf/launch-tp4.sh ;;
    *-ep4-*)  run "$arm" /work/scripts/bbuf-launch-ep4.sh ;;
    *-nods-*) run "$arm" /work/bbuf/launch.sh --env DSPARK=0 ;;
    *) echo "unknown arm $arm"; exit 2 ;;
  esac
done
echo STEP0_DONE
