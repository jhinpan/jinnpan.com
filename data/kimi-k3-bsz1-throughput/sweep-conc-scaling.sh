#!/usr/bin/env bash
# Concurrency scaling curve at ISL/OSL 1024 against a running Kimi-K3 server, so
# the bsz=1 numbers from sweep-bsz1.sh sit in context and the throughput knee is
# visible.
#
#   ./sweep-conc-scaling.sh [tag]
#
# Protocol is the grid's `w1` workload (random, fixed lengths, --flush-cache,
# --warmup-requests 2) so rows drop straight into the same comparison space as
# results.csv. One deliberate change: num-prompts has a floor of 8 instead of
# being exactly conc*2. The bsz=1 sweep showed out_tps tracks accept_len almost
# 1:1, and accept_len on random-token data is noisy, so the low-conc points need
# more samples than conc*2 would give them (conc*2 is 2 requests at conc=1).
# At conc>=4 the floor is inactive and num-prompts equals the grid's conc*2.
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

ISL="${ISL:-1024}"
OSL="${OSL:-1024}"
NP_FLOOR="${NP_FLOOR:-8}"
WARMUP="${WARMUP:-2}"

COLS="ts,sweep,isl,osl,conc,np,status,out_tps,total_tps,req_tps,conc_ach,accept_len,mean_ttft_ms,median_ttft_ms,p99_ttft_ms,mean_tpot_ms,median_tpot_ms,mean_itl_ms,mean_e2e_ms,median_e2e_ms,gen_tok,retok_div_pct,duration_s,bench_s"
[[ -s "${CSV}" ]] || echo "${COLS}" > "${CSV}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${RUNLOG}"; }

exec 9>/tmp/k3-grid.lock
if ! flock -n 9; then
  log "another grid/sweep run holds /tmp/k3-grid.lock; refusing to start"
  exit 1
fi
echo "$$" >&9

alive() { curl -s -m 10 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; }

point() {
  local conc="$1"
  local np=$(( conc * 2 ))
  (( np < NP_FLOOR )) && np="${NP_FLOOR}"
  local tag="C_conc${conc}"
  local bjson="${RUN_DIR}/logs/${tag}.jsonl"
  local btext="${RUN_DIR}/logs/${tag}.txt"

  if ! alive; then
    log "  SERVER DEAD before ${tag}; aborting sweep"
    return 1
  fi

  log "point ${tag} (isl=${ISL} osl=${OSL} np=${np} warmup=${WARMUP})"
  local t0; t0=$(date +%s)
  ( cd "${SGL}" && timeout "${POINT_TIMEOUT:-3600}" python3 -m sglang.benchmark.serving \
      --backend sglang-oai-chat --host 127.0.0.1 --port "${PORT}" --model "${MODEL}" \
      --dataset-name random \
      --random-input-len "${ISL}" --random-output-len "${OSL}" --random-range-ratio 1 \
      --num-prompts "${np}" --max-concurrency "${conc}" \
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

  printf '%s\n' "$(date -Is),C,${ISL},${OSL},${conc},${np},${status},${out_tps},${total_tps},${req_tps},${conc_ach},${accept_len},${mean_ttft_ms},${median_ttft_ms},${p99_ttft_ms},${mean_tpot_ms},${median_tpot_ms},${mean_itl_ms},${mean_e2e_ms},${median_e2e_ms},${gen_tok},${retok_div_pct},${duration_s},${bench_s}" >> "${CSV}"

  log "  -> ${status} out_tps=${out_tps} total_tps=${total_tps} conc_ach=${conc_ach} ttft_med=${median_ttft_ms}ms tpot_med=${median_tpot_ms}ms accept=${accept_len} (${bench_s}s)"
  [[ "${status}" == "CRASH" ]] && return 1
  return 0
}

log "run dir   ${RUN_DIR}"
log "port      ${PORT}"
log "sglang    $(git -C "${SGL}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
log "=== sweep C: concurrency scaling at ISL ${ISL} / OSL ${OSL} ==="
# 48 is max_running_requests for this dspark config, so it is the saturation
# ceiling; anything above it would only measure client-side queueing.
for C in ${CONC_LIST:-1 2 4 8 16 32 48}; do
  point "${C}" || break
done
log "=== done, csv: ${CSV} ==="
