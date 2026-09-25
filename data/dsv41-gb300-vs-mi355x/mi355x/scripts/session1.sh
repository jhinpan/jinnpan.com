#!/usr/bin/env bash
# Session 1: MI355X under BBuf's BS=1 protocol, unbound (u) vs NUMA-node-1-bound (n).
# Each mode is mirrored in time (u n n u) so drift cannot masquerade as a binding effect;
# sim gets a third pair because it is the arm that varied most on GB300.
set -uo pipefail
cd $CAMPAIGN
R=bench/run_arm.sh
for spec in \
  "u-off-a off 0" "n-off-a off 1" \
  "u-real-a real 0" "n-real-a real 1" \
  "u-sim-a sim 0" "n-sim-a sim 1" "n-sim-b sim 1" "u-sim-b sim 0" \
  "n-real-b real 1" "u-real-b real 0" \
  "n-off-b off 1" "u-off-b off 0" \
  "n-sim-c sim 1" "u-sim-c sim 0"; do
  set -- $spec
  echo "[$(date -u +%FT%TZ)] start $1"
  bash $R "$1" "$2" "$3"
  echo "[$(date -u +%FT%TZ)] end $1 rc=$?"
done
