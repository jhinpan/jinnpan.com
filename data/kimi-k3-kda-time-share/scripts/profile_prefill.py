#!/usr/bin/env python3
"""Profile the prefill of one request at a fixed input length.

Prefill never runs through a CUDA graph in this configuration (the prefill graph
backend is disabled), so unlike decode the torch profiler sees every kernel and the
`K3/*` record_function ranges attribute them directly -- no name-map transfer.

Exactly one profiler session per point. The profiler is armed with a forward-count
budget equal to the number of chunked-prefill chunks and starts immediately; the
server is idle at that moment, so the profiled window is precisely those chunks. A
single output token keeps decode out of the window.

  ./profile_prefill.py --isl 8192 --tag pf8k
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import urllib.error
import urllib.request

PORT = int(os.environ.get("PORT", "30100"))
BASE = f"http://127.0.0.1:{PORT}"
CHUNK = 16384


def post(path: str, payload: dict | None = None, timeout: float = 7200.0):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload or {}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return body


def generate(input_ids: list[int]) -> dict:
    t0 = time.time()
    r = post(
        "/generate",
        {
            "input_ids": input_ids,
            "sampling_params": {"max_new_tokens": 1, "temperature": 0.0,
                                "ignore_eos": True},
        },
    )
    return {"wall_s": time.time() - t0, "meta_info": r.get("meta_info", {})}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--isl", type=int, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--activities", default="CPU,GPU")
    ap.add_argument("--out-root", default="/sgl-workspace/workspace/kda_prof/traces")
    args = ap.parse_args()

    out_dir = os.path.join(args.out_root, args.tag)
    os.makedirs(out_dir, exist_ok=True)

    rng = random.Random(1234)
    input_ids = [rng.randint(1000, 100000) for _ in range(args.isl)]
    chunks = max(1, math.ceil(args.isl / CHUNK))

    try:
        post("/flush_cache")
    except urllib.error.HTTPError as e:
        print(f"  flush_cache: {e}", flush=True)
    time.sleep(2)

    # Unprofiled reference first: profiling perturbs wall time, and TTFT is the
    # number we want to compare the summed kernel time against.
    ref = generate(input_ids)
    ref_ttft = ref["meta_info"].get("e2e_latency", ref["wall_s"])
    print(f"[{args.tag}] isl={args.isl} chunks={chunks} "
          f"reference TTFT {ref_ttft:.3f}s "
          f"({args.isl/ref_ttft:,.0f} tok/s)", flush=True)

    try:
        post("/flush_cache")
    except urllib.error.HTTPError:
        pass
    time.sleep(2)

    r = post(
        "/start_profile",
        {
            "output_dir": out_dir,
            "num_steps": chunks,
            "activities": args.activities.split(","),
            "with_stack": False,
            "record_shapes": False,
            "profile_id": args.tag,
        },
    )
    print(f"  start_profile: {r}", flush=True)

    prof = generate(input_ids)
    prof_ttft = prof["meta_info"].get("e2e_latency", prof["wall_s"])
    print(f"[{args.tag}] profiled TTFT {prof_ttft:.3f}s", flush=True)

    try:
        print(f"  stop_profile: {post('/stop_profile')}", flush=True)
    except urllib.error.HTTPError as e:
        print(f"  stop_profile: already auto-stopped ({e.code})", flush=True)

    time.sleep(10)
    files = sorted(f for f in os.listdir(out_dir) if f.endswith(".gz"))
    print(f"[{args.tag}] {len(files)} traces", flush=True)

    with open(os.path.join(out_dir, "run.json"), "w") as fh:
        json.dump(
            {"tag": args.tag, "isl": args.isl, "chunks": chunks,
             "ref_ttft_s": ref_ttft, "profiled_ttft_s": prof_ttft,
             "ref_prefill_tok_per_s": args.isl / ref_ttft,
             "meta_info": ref["meta_info"]},
            fh, indent=1,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
