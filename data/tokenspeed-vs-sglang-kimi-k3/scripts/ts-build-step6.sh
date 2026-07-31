#!/bin/bash
# Finish the TokenSpeed install: the runtime package plus every dependency
# except tokenspeed-mooncake, which repo main pins at a version that was never
# published to PyPI. Its only importers are the prefill/decode disaggregation
# and kvstore-storage paths, both lazy and both unused by an aggregated
# single-node server launched with --disable-kvstore.
set -euo pipefail

TS_ROOT=/sgl-workspace/tokenspeed
VENV=$TS_ROOT/.venv
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/rocm7.2"

echo "=== deps from pyproject, minus mooncake and minus torch (already ROCm) ==="
python - <<'PY' > /tmp/ts-deps.txt
import pathlib, re
text = pathlib.Path("/sgl-workspace/tokenspeed/python/pyproject.toml").read_text()
block = re.search(r"^dependencies\s*=\s*\[(.*?)^\]", text, re.S | re.M).group(1)
skip = ("tokenspeed-mooncake", "torch")
for dep in re.findall(r'"([^"]+)"', block):
    name = re.split(r"[><=\[!~;]", dep)[0].strip()
    print(f"# SKIPPED {dep}" if name in skip else dep)
PY
grep -v '^#' /tmp/ts-deps.txt > /tmp/ts-deps-clean.txt
echo "--- skipping:"; grep '^#' /tmp/ts-deps.txt
echo "--- installing $(wc -l < /tmp/ts-deps-clean.txt) deps"

python -m pip install -r /tmp/ts-deps-clean.txt

echo "=== runtime package itself (no deps; satisfied above) ==="
python -m pip install -e "$TS_ROOT/python" --no-build-isolation --no-deps

echo "=== verify ==="
python -c "
import torch, tokenspeed_scheduler, tokenspeed_kernel
print('torch       :', torch.__version__)
print('hip         :', torch.version.hip)
print('FLAT_KVCACHE:', tokenspeed_scheduler.FLAT_KVCACHE)
assert tokenspeed_scheduler.FLAT_KVCACHE
"
python -c "import torch; print('gpus:', torch.cuda.device_count())"
tokenspeed --help > /dev/null && echo "CLI OK"
echo "=========================================="
echo "BUILD OK"
echo "=========================================="
