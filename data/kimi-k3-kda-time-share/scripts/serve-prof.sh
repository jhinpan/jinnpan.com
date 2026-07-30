#!/usr/bin/env bash
# Kimi-K3 server for the KDA/attention/MoE decode attribution study.
#
# Same recipe as the deployed serve-k3-ext.sh (TP8, triton attention, bf16
# activations over MXFP4 experts, radix cache off) with two deliberate changes:
#
#   * no speculative decoding -- one decode step is then exactly one target-model
#     forward, so a kernel's cost is attributable to a layer type without having
#     to separate draft-model work from target work, and there is no accept-length
#     confound in the per-step time.
#   * GRAPH=off (default) disables CUDA graphs, because the torch profiler cannot
#     see kernels replayed from inside a HIP graph. Kernel durations do not depend
#     on whether the launch came from a graph, so the *composition* of device time
#     is faithful; only the launch gaps (wall time) are inflated. GRAPH=on gives
#     the un-inflated per-step latency for the same server config.
#
#   GRAPH=off ./serve-prof.sh      # composition run, profiler ranges armed
#   GRAPH=on  ./serve-prof.sh      # latency anchor run
set -uo pipefail

GRAPH="${GRAPH:-off}"
PORT="${PORT:-30100}"
LOG="${LOG:-/sgl-workspace/workspace/kda_prof/k3-prof-graph${GRAPH}.log}"

export HF_HUB_OFFLINE=1
export SGLANG_USE_AITER=1
export SGLANG_AITER_K3_OPT=1
export AITER_FLYDSL_FORCE=1
export AITER_SITUV2_A8W4=1

# Host-side block ranges. They force CPU activity in the profiler, which is what
# grew the kineto trace buffers until teardown segfaulted, so they stay off by
# default. Prefill is the case that wants them on: it never runs through a CUDA
# graph, so ranges attribute its kernels directly and no name map is needed.
if [[ "${RANGES:-0}" == "1" ]]; then
  export SGLANG_K3_PROF_RANGES=1
fi

ARGS=(
  --model-path moonshotai/Kimi-K3
  --trust-remote-code
  --tp 8
  --attention-backend triton
  --dtype bfloat16
  --mem-fraction-static "${MEM_FRAC:-0.92}"
  --host 127.0.0.1
  --port "${PORT}"
  --reasoning-parser kimi_k3
  --tool-call-parser kimi_k3
  --disable-radix-cache
  --max-running-requests "${MAX_RUNNING:-48}"
)

if [[ "${GRAPH}" == "off" ]]; then
  ARGS+=(--disable-cuda-graph)
else
  ARGS+=(--cuda-graph-max-bs-decode 256)
fi

echo "=== $(date -Is) launching graph=${GRAPH} port=${PORT} ranges=${SGLANG_K3_PROF_RANGES:-0} ===" | tee -a "${LOG}"
exec sglang serve "${ARGS[@]}" >> "${LOG}" 2>&1
