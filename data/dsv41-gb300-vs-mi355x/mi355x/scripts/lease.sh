#!/usr/bin/env bash
# lease.sh NAME -- CMD...: exclusive-node GPU lease (GPUs 4-7, all eight reserved,
# 30-sample idle admission, HBM canaries). A lease the idle check rejects before
# CMD starts is retried; once CMD has run (exit.json written) its result is final.
set -uo pipefail
cd $CAMPAIGN
name=${1:?name}
shift
[ "${1:-}" = "--" ] && shift
for attempt in 1 2 3 4 5 6; do
  record=results/gpu-claim-$name
  [ "$attempt" -gt 1 ] && record=$record-a$attempt
  /opt/venv/bin/python3 $PA_PR/bench/with_gpus.py \
    --smi 4,5,6,7 --samples 30 --exclusive-node --canary --record "$record" -- "$@"
  rc=$?
  [ -f "$record/exit.json" ] && exit "$rc"
  echo "lease attempt $attempt rejected before the command started; retrying" >&2
  sleep 60
done
exit 1
