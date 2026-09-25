#!/usr/bin/env bash
# Session 3c (MI355X, exclusive lease of HIP 4-7): the kernel-traced servers of session 3b with
# the corrected stop order (workers first, so each flushes its rocprofv3 trace). Graph packet
# capture stays off for the reason given in session3b.sh.
set -uo pipefail
cd $CAMPAIGN
LOG() { echo "[$(date -u +%FT%TZ)] $*"; }
export DEBUG_CLR_GRAPH_PACKET_CAPTURE=0
for spec in "rp-real-c real" "rp-off-c off"; do
  set -- $spec
  LOG "start $1"
  bash bench/run_rocprof_arm.sh "$1" "$2"
  LOG "end $1 rc=$?"
done
LOG SESSION3C_DONE
