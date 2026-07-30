#!/usr/bin/env bash
# Quack RMSNorm fwd/bwd sweep on one idle Hopper GPU.
#
# Runs three providers in a single pass so they share one roofline probe and
# one thermal state: quack_tuned (@autotune search), quack (analytical
# heuristic), and torch (same-device reference).
#
#   bash sweep-rmsnorm.sh <repo-root> <gpu-index> <output-dir>
#
# Example:
#   bash sweep-rmsnorm.sh /root/quack-FlyDSL-h200-test 3 /root/xbench-h200-tuned
#
# Requirements: a CUDA PyTorch build, the Quack checkout under test, and the
# quack_tuned provider in benchmarks/benchmark_rmsnorm_flydsl.py (see
# quack_tuned_provider.patch in this directory).
set -euo pipefail

REPO="${1:?usage: sweep-rmsnorm.sh <repo-root> <gpu-index> <output-dir>}"
GPU="${2:?missing gpu index}"
OUT="${3:?missing output dir}"

# Refuse to measure on a busy GPU. Percent-of-peak is normalised against a
# bandwidth probe taken at the start of the run; a co-tenant invalidates it.
read -r used util < <(
  nvidia-smi --query-gpu=memory.used,utilization.gpu \
             --format=csv,noheader,nounits -i "$GPU" | tr -d ','
)
if [ "$used" -gt 64 ] || [ "$util" -gt 5 ]; then
  echo "GPU $GPU is not idle (${used} MiB used, ${util}% util). Pick another." >&2
  exit 1
fi

# PYTHONPATH is load bearing. Running the harness by path puts benchmarks/ at
# the front of sys.path, so a bare `import quack` silently resolves to the pip
# wheel in site-packages instead of the checkout under test.
export PYTHONPATH="$REPO"
export CUDA_VISIBLE_DEVICES="$GPU"

cd "$REPO"
python -c "import importlib; m = importlib.import_module('quack.rmsnorm'); print('quack.rmsnorm ->', m.__file__)"

exec python benchmarks/benchmark_rmsnorm_flydsl.py \
  --providers quack_tuned quack torch \
  --warmup-rounds 3 --sample-rounds 12 \
  --copy-mib 512 --copy-samples 30 \
  --min-rotation-buffers 2 --max-rotation-buffers 4 \
  --expected-arch sm_90 \
  --output-dir "$OUT"
