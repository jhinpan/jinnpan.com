#!/usr/bin/env bash
# Inside the workbench: start current main with a cookbook GB300 cell plus the shared
# measurement flags, i.e. the same server command the measured arms use.
#   serve_cu13.sh ll       Low-Latency cell, DSpark with real acceptance
#   serve_cu13.sh ll-sim   Low-Latency cell, DSpark with simulated acceptance 5.5
#   serve_cu13.sh ht       High-Throughput cell, DSpark off
# Extra arguments are appended to `sglang serve`.
set -euo pipefail
source /work/scripts/cells_cu13.sh
mode=${1:-}
shift || true
case $mode in
  ll) cell=$low_latency ;;
  ll-sim)
    cell=$low_latency
    export SGLANG_RAGGED_VERIFY_MODE=static SGLANG_SIMULATE_ACC_LEN=5.5 SGLANG_SIMULATE_ACC_METHOD=match-expected
    ;;
  ht) cell=$high_throughput ;;
  *) echo "usage: $0 ll|ll-sim|ht [extra sglang serve args]" >&2; exit 2 ;;
esac
# shellcheck disable=SC2086
exec sglang serve $common $cell $measure "$@"
