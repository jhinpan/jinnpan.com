#!/usr/bin/env bash
# Decode-only attribution sweep: one profiler session per context point.
#
#   ./sweep2.sh [tag-prefix]
set -uo pipefail

D=/sgl-workspace/workspace/kda_prof
PORT="${PORT:-30100}"
PREFIX="${1:-ctx}"
STEPS="${STEPS:-24}"
NAME_MAP="${NAME_MAP:-${D}/results/name_map_4k.json}"
CONTEXTS="${CONTEXTS:-4096 32768 65536 524288 1047552}"

mkdir -p "${D}/results"
LOG="${D}/results/sweep2-${PREFIX}.log"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "${LOG}"; }

label() {
  case "$1" in
    4096) echo 4k ;; 32768) echo 32k ;; 65536) echo 64k ;;
    524288) echo 512k ;; 1047552) echo 1m ;; *) echo "$1" ;;
  esac
}

log "=== sweep2 ${PREFIX}: contexts=${CONTEXTS} steps=${STEPS} ==="
log "warmup (unprofiled; first decode pays triton autotune for unseen shapes)"
python3 - "$PORT" <<'PY' 2>&1 | tail -1 | tee -a "${LOG}"
import json, random, sys, urllib.request
rng = random.Random(7)
req = urllib.request.Request(
    f"http://127.0.0.1:{sys.argv[1]}/generate",
    data=json.dumps({
        "input_ids": [rng.randint(1000, 100000) for _ in range(4096)],
        "sampling_params": {"max_new_tokens": 12, "temperature": 0.0, "ignore_eos": True},
    }).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=3600) as r:
    print("warmup ok:", json.loads(r.read().decode())["meta_info"]["completion_tokens"], "tokens")
PY

for ISL in ${CONTEXTS}; do
  TAG="${PREFIX}$(label "${ISL}")"
  log "--- point ${TAG} (isl=${ISL}) ---"
  python3 "${D}/profile_decode.py" --isl "${ISL}" --tag "${TAG}" \
      --num-steps "${STEPS}" 2>&1 | tee -a "${LOG}"
  python3 "${D}/chrome_bucketize.py" "${D}/traces/${TAG}" --name-map "${NAME_MAP}" \
      --out "${D}/results/${TAG}.json" 2>&1 | tee -a "${LOG}"
  log ""
done

log "=== sweep2 done ==="
