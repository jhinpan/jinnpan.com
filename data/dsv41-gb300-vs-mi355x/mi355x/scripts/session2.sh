#!/usr/bin/env bash
# Session 2: where the four schedulers run. split = SGLang's inherited per-rank affinity
# (TP0/TP1 on node 0, remote from HIP 4-7); node1 = all on node 1; none = no pinning.
# DSpark modes are mirrored in time (split node1 none none node1 split) to separate the
# placement effect from drift; off (plain decode) is the host-insensitive control.
set -uo pipefail
cd $CAMPAIGN
R=bench/run_arm2.sh
for spec in \
  "s2-split-sim-a sim split" "s2-node1-sim-a sim node1" "s2-none-sim-a sim none" \
  "s2-none-sim-b sim none" "s2-node1-sim-b sim node1" "s2-split-sim-b sim split" \
  "s2-node1-real-a real node1" "s2-none-real-a real none" "s2-split-real-a real split" \
  "s2-split-real-b real split" "s2-none-real-b real none" "s2-node1-real-b real node1" \
  "s2-node1-off-a off node1" "s2-none-off-a off none"; do
  set -- $spec
  echo "[$(date -u +%FT%TZ)] start $1"
  bash $R "$1" "$2" "$3"
  echo "[$(date -u +%FT%TZ)] end $1 rc=$?"
done
