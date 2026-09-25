#!/usr/bin/env bash
# Diagnostic: does SGLang's own NUMA binding (blocked in Docker without CAP_SYS_NICE)
# remove the startup-to-startup variance of the simulated-acceptance DSpark arm?
set -euo pipefail
cd $GB300_WORK
diag="--record-placement --sample-gpu"
for arm in "$@"; do
  case $arm in
    *-nice-*) ARM_EXTRA="$diag --docker-arg=--cap-add=SYS_NICE" bash scripts/step1a_current_main.sh "$arm" ;;
    *-base-*) ARM_EXTRA="$diag" bash scripts/step1a_current_main.sh "$arm" ;;
    *) echo "unknown arm $arm"; exit 2 ;;
  esac
done
echo EXP_NUMA_DONE
