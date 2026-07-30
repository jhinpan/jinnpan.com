#!/usr/bin/env bash
# Prefill attribution sweep. One profiler session per point; points are independent
# so a mid-sweep server loss only costs the remaining ones (libkineto has been seen
# to segfault on teardown once several sessions accumulate in a process).
#
#   ./sweep-prefill.sh [tag-prefix]
set -uo pipefail

D=/sgl-workspace/workspace/kda_prof
PORT="${PORT:-30100}"
PREFIX="${1:-pf}"
CONTEXTS="${CONTEXTS:-1024 4096 8192 32768}"

mkdir -p "${D}/results"
LOG="${D}/results/sweep-prefill-${PREFIX}.log"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "${LOG}"; }

label() {
  case "$1" in
    1024) echo 1k ;; 4096) echo 4k ;; 8192) echo 8k ;; 32768) echo 32k ;; *) echo "$1" ;;
  esac
}

alive() { curl -s -m 20 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; }

log "=== prefill sweep ${PREFIX}: contexts=${CONTEXTS} ==="
log "warmup (unprofiled)"
python3 - "$PORT" <<'PY' 2>&1 | tail -1 | tee -a "${LOG}"
import json, random, sys, urllib.request
rng = random.Random(7)
req = urllib.request.Request(
    f"http://127.0.0.1:{sys.argv[1]}/generate",
    data=json.dumps({
        "input_ids": [rng.randint(1000, 100000) for _ in range(8192)],
        "sampling_params": {"max_new_tokens": 1, "temperature": 0.0, "ignore_eos": True},
    }).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=3600) as r:
    print("warmup ok:", json.loads(r.read().decode())["meta_info"]["prompt_tokens"], "prompt tokens")
PY

for ISL in ${CONTEXTS}; do
  TAG="${PREFIX}$(label "${ISL}")"
  if ! alive; then log "SERVER DEAD before ${TAG}; stopping"; break; fi
  log "--- point ${TAG} (isl=${ISL}) ---"
  python3 "${D}/profile_prefill.py" --isl "${ISL}" --tag "${TAG}" 2>&1 | tee -a "${LOG}"
  python3 "${D}/prefill_bucketize.py" "${D}/traces/${TAG}" \
      --out "${D}/results/${TAG}.json" 2>&1 | tee -a "${LOG}"
  log ""
done

log "=== prefill sweep done ==="
