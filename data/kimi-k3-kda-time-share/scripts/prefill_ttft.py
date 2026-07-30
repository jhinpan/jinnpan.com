#!/usr/bin/env python3
"""Clean prefill TTFT, measured with no profiler ranges compiled in.

The record_function ranges are cheap against a saturated GPU but not against an
idle one: with ranges enabled a 4K prefill took 692 ms versus 340 ms without,
while 32K was unchanged. That gap is the point -- short prefill is limited by
host-side launch work, not by the GPU -- so the wall-clock reference has to come
from a server built without the annotations.

  ./prefill_ttft.py --out results/prefill_ttft.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.request

CONTEXTS = [("1k", 1024), ("4k", 4096), ("8k", 8192), ("32k", 32768)]


def post(base: str, path: str, payload: dict, timeout: float = 3600.0):
    req = urllib.request.Request(
        base + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        # /flush_cache answers in plain text.
        return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30100)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    rng = random.Random(1234)

    post(base, "/generate", {
        "input_ids": [rng.randint(1000, 100000) for _ in range(8192)],
        "sampling_params": {"max_new_tokens": 1, "temperature": 0.0, "ignore_eos": True},
    })

    rows = []
    for label, isl in CONTEXTS:
        ids = [rng.randint(1000, 100000) for _ in range(isl)]
        samples = []
        for _ in range(args.reps):
            post(base, "/flush_cache", {})
            time.sleep(1.5)
            t0 = time.time()
            r = post(base, "/generate", {
                "input_ids": ids,
                "sampling_params": {"max_new_tokens": 1, "temperature": 0.0,
                                    "ignore_eos": True},
            })
            samples.append(r["meta_info"].get("e2e_latency", time.time() - t0))
        best = min(samples)
        rows.append({"label": label, "isl": isl, "ttft_s_min": best,
                     "ttft_s_all": samples, "tok_per_s": isl / best})
        print(f"[{label}] isl={isl} TTFT min {best*1000:.0f} ms "
              f"({isl/best:,.0f} tok/s)  samples={[round(s*1000) for s in samples]}",
              flush=True)

    with open(args.out, "w") as fh:
        json.dump({"rows": rows}, fh, indent=1)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
