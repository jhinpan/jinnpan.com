#!/usr/bin/env bash
# bsz=1 (concurrency 1) ISL/OSL characterisation against a running Kimi-K3 server.
#
#   ./sweep-bsz1.sh [tag]
#
# Two orthogonal sweeps rather than a cross product. At concurrency 1 there is no
# batch to form, so prefill cost lands entirely in TTFT and decode cost entirely
# in TPOT; the two are separable and the cross product would cost ~10x for almost
# no extra information.
#   A: ISL scaling at OSL 1024
#   B: OSL scaling at ISL 1024
# (ISL 1024 / OSL 1024 is shared, and is the point that cross-checks against the
# grid's `p3-lat-win` row: out_tps 111.58, tpot 8.78 ms, accept_len 2.757.)
set -uo pipefail

W=/sgl-workspace/workspace
SGL=/sgl-workspace/sglang
TOOLS="${W}/gridtools.py"
PORT="${PORT:-30100}"
MODEL=moonshotai/Kimi-K3
BENCH_FILTER='Calling super\(\)\.encode|^\s*$|it/s\]|aiter\]|Namespace\(|benchmark_args='

TAG="${1:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${W}/bsz1_results/${TAG}"
mkdir -p "${RUN_DIR}/logs"
CSV="${RUN_DIR}/results.csv"
RUNLOG="${RUN_DIR}/sweep.log"

NP="${NP:-4}"
WARMUP="${WARMUP:-1}"

COLS="ts,sweep,isl,osl,conc,np,status,out_tps,total_tps,req_tps,accept_len,mean_ttft_ms,median_ttft_ms,mean_tpot_ms,median_tpot_ms,mean_itl_ms,mean_e2e_ms,median_e2e_ms,gen_tok,retok_div_pct,duration_s,bench_s"
[[ -s "${CSV}" ]] || echo "${COLS}" > "${CSV}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${RUNLOG}"; }

# The grid harness tears down whatever server is on the box between configs, so
# hold its lock: without it a grid run started mid-sweep would kill the server
# under us and every remaining point would record a spurious crash.
exec 9>/tmp/k3-grid.lock
if ! flock -n 9; then
  log "another grid/sweep run holds /tmp/k3-grid.lock; refusing to start"
  exit 1
fi
echo "$$" >&9

alive() { curl -s -m 10 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; }

point() {
  local sweep="$1" isl="$2" osl="$3"
  local conc=1
  local tag="${sweep}_isl${isl}_osl${osl}"
  local bjson="${RUN_DIR}/logs/${tag}.jsonl"
  local btext="${RUN_DIR}/logs/${tag}.txt"

  if ! alive; then
    log "  SERVER DEAD before ${tag}; aborting sweep"
    return 1
  fi

  log "point ${tag} (np=${NP} warmup=${WARMUP})"
  local t0; t0=$(date +%s)
  ( cd "${SGL}" && timeout "${POINT_TIMEOUT:-3600}" python3 -m sglang.benchmark.serving \
      --backend sglang-oai-chat --host 127.0.0.1 --port "${PORT}" --model "${MODEL}" \
      --dataset-name random \
      --random-input-len "${isl}" --random-output-len "${osl}" --random-range-ratio 1 \
      --num-prompts "${NP}" --max-concurrency "${conc}" \
      --warmup-requests "${WARMUP}" --flush-cache \
      --output-file "${bjson}" --tag "${tag}" ) 2>&1 \
    | rg -v "${BENCH_FILTER}" > "${btext}" || true
  local bench_s=$(( $(date +%s) - t0 ))

  local status=NO_RESULT
  local out_tps=NA total_tps=NA req_tps=NA conc_ach=NA accept_len=NA
  local mean_ttft_ms=NA median_ttft_ms=NA p99_ttft_ms=NA mean_tpot_ms=NA median_tpot_ms=NA
  local mean_itl_ms=NA mean_e2e_ms=NA median_e2e_ms=NA
  local gen_tok=NA retok_tok=NA retok_div_pct=NA cache_hit_pct=NA
  local completed=NA duration_s=NA
  eval "$(python3 "${TOOLS}" parse-bench --jsonl "${bjson}" --text "${btext}" 2>/dev/null)" || true

  if ! alive; then
    status=CRASH
    log "  SERVER DIED during ${tag}"
  fi

  printf '%s\n' "$(date -Is),${sweep},${isl},${osl},${conc},${NP},${status},${out_tps},${total_tps},${req_tps},${accept_len},${mean_ttft_ms},${median_ttft_ms},${mean_tpot_ms},${median_tpot_ms},${mean_itl_ms},${mean_e2e_ms},${median_e2e_ms},${gen_tok},${retok_div_pct},${duration_s},${bench_s}" >> "${CSV}"

  log "  -> ${status} out_tps=${out_tps} total_tps=${total_tps} ttft_med=${median_ttft_ms}ms tpot_med=${median_tpot_ms}ms accept=${accept_len} (${bench_s}s)"
  [[ "${status}" == "CRASH" ]] && return 1
  return 0
}

log "run dir   ${RUN_DIR}"
log "port      ${PORT}"
log "sglang    $(git -C "${SGL}" rev-parse --short HEAD 2>/dev/null || echo unknown)"

log "=== sweep A: ISL scaling at OSL 1024, bsz=1 ==="
for ISL in ${ISL_LIST:-128 1024 4096 8192 16384 32768 65536}; do
  point A "${ISL}" 1024 || break
done

log "=== sweep B: OSL scaling at ISL 1024, bsz=1 ==="
for OSL in ${OSL_LIST:-128 512 2048 4096}; do
  point B 1024 "${OSL}" || break
done

log "=== done, csv: ${CSV} ==="
