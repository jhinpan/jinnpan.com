#!/bin/bash
# Build TokenSpeed for ROCm/gfx950 into an isolated venv.
#
# Isolation is the point: TokenSpeed pins torch==2.11.0 and transformers==5.12.0,
# both of which would break the SGLang install that shares this container.
set -euo pipefail

TS_ROOT=${TS_ROOT:-/sgl-workspace/tokenspeed}
VENV=${VENV:-$TS_ROOT/.venv}
ROCM_INDEX="https://download.pytorch.org/whl/rocm7.2"
export MAX_JOBS=${MAX_JOBS:-32}

echo "=== Step 0: apt deps ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends \
    openmpi-bin libopenmpi-dev libssl-dev libnuma1 pkg-config

echo "=== Step 1: venv at $VENV ==="
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -V
python -m pip install --upgrade pip "setuptools<82" wheel cmake ninja

echo "=== Step 2: torch 2.11.0+rocm7.2 ==="
python -m pip install --index-url "$ROCM_INDEX" \
    torch==2.11.0+rocm7.2 torchvision torchaudio

echo "=== Step 3: tokenspeed-kernel-amd (in-tree, must precede tokenspeed-kernel) ==="
cd "$TS_ROOT"
python -m pip install --force-reinstall --no-deps \
    "$TS_ROOT/tokenspeed-kernel-amd" --no-build-isolation

echo "=== Step 4: tokenspeed-kernel (rocm backend) ==="
export PIP_EXTRA_INDEX_URL="$ROCM_INDEX"
TOKENSPEED_KERNEL_BACKEND=rocm \
python -m pip install "$TS_ROOT/tokenspeed-kernel/python/" --no-build-isolation

echo "=== Step 5: tokenspeed-scheduler (FlatKV ON — K3 is FlatKV-only) ==="
python -m pip install "$TS_ROOT/tokenspeed-scheduler/" \
    --config-settings=cmake.define.TOKENSPEED_FLAT_KVCACHE=ON

echo "=== Step 6: tokenspeed runtime ==="
python -m pip install -e "$TS_ROOT/python" --no-build-isolation \
    --extra-index-url "$ROCM_INDEX"

echo "=== Step 7: verify ==="
python -c "
import torch, tokenspeed_scheduler
print('torch      :', torch.__version__)
print('hip        :', torch.version.hip)
print('gpus       :', torch.cuda.device_count())
print('FLAT_KVCACHE:', tokenspeed_scheduler.FLAT_KVCACHE)
assert tokenspeed_scheduler.FLAT_KVCACHE, 'FlatKV scheduler NOT enabled'
"
tokenspeed env || true

echo "=========================================="
echo "BUILD OK"
echo "=========================================="
