#!/usr/bin/env bash
# Stage DeepSeek-V4.1-Flash@dba1be0a from the shared HF cache to local NVMe and
# verify every LFS file's content hash against its blob name (= HF LFS sha256).
set -euo pipefail

rev=dba1be0a40aa45a94ad051997016db3960a90277
src=/SFS-aGqda6ct/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/$rev
dst=$GB300_MODEL_DIR
work=$GB300_WORK
mkdir -p "$dst"

copy_one() {
  local name=$1 src=$2 dst=$3
  local blob expected actual
  blob=$(readlink -f "$src/$name")
  expected=$(basename "$blob")
  actual=$(tee "$dst/$name.partial" < "$blob" | sha256sum | cut -d' ' -f1)
  if [[ ${#expected} == 64 && "$actual" != "$expected" ]]; then
    echo "HASH_MISMATCH $name expected=$expected actual=$actual"
    return 1
  fi
  mv "$dst/$name.partial" "$dst/$name"
  echo "OK $name $actual"
}
export -f copy_one

ls "$src" | xargs -P 8 -I{} bash -c 'copy_one "$@"' _ {} "$src" "$dst"

base=https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/$rev
for f in README.md LICENSE encoding/README.md encoding/encoding.py encoding/test_encoding.py \
         inference/README.md inference/config.json inference/engram.py inference/model.py; do
  mkdir -p "$dst/$(dirname "$f")"
  curl -sfSL -m 120 -o "$dst/$f" "$base/$f"
  echo "FETCHED $f $(sha256sum "$dst/$f" | cut -d' ' -f1)"
done

python3 - "$work/records/hf-tree-dba1be0a.json" "$dst" <<'EOF'
import json, os, sys
tree, dst = json.load(open(sys.argv[1])), sys.argv[2]
bad = []
for e in tree:
    if e['type'] != 'file':
        continue
    p = os.path.join(dst, e['path'])
    if os.path.exists(p) and os.path.getsize(p) != e['size']:
        bad.append(e['path'])
print('SIZE_CHECK', 'FAIL' if bad else 'PASS', bad)
EOF
echo STAGE_DONE
