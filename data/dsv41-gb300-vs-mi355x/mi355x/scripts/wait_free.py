#!/usr/bin/env python3
"""Block until no KFD process holds GPUs 4-7 (HIP 4-7 on this host), so the next
server starts on released devices. Exit 1 on timeout."""
import sys
import time
from pathlib import Path

sys.path.insert(0, "$PA_PR/bench")
from gpu_state import kfd_process_state  # noqa: E402

GPUS = [dict(smi_idx=4, bdf="0000:85:00.0"), dict(smi_idx=5, bdf="0000:95:00.0"),
        dict(smi_idx=6, bdf="0000:e5:00.0"), dict(smi_idx=7, bdf="0000:f5:00.0")]
deadline = time.monotonic() + float(sys.argv[1] if len(sys.argv) > 1 else 300)
while True:
    state = kfd_process_state(GPUS)
    busy = [d for d in state if d["processes"]]
    if not busy:
        print("gpus 4-7 free")
        sys.exit(0)
    if time.monotonic() > deadline:
        print(f"timeout: {busy}")
        sys.exit(1)
    time.sleep(3)
