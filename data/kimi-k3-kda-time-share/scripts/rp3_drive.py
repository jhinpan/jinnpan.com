#!/usr/bin/env python3
"""Drive the context-length points against a graphs-on server, one request each.

Separated by long idle gaps so the offline parser can cut the single continuous
rocprofv3 kernel trace back into per-context islands without having to correlate
two different clocks. Streams the response so inter-token latency is recorded even
though rocprofv3's per-dispatch interception inflates it.

  ./rp3_drive.py --out rp3/graphon/drive.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.request

CONTEXTS = [
    ("4k", 4096),
    ("32k", 32768),
    ("64k", 65536),
    ("512k", 524288),
    ("1m", 1047552),
]


def flush(base: str) -> None:
    try:
        urllib.request.urlopen(
            urllib.request.Request(base + "/flush_cache", data=b"{}",
                                   headers={"Content-Type": "application/json"}),
            timeout=120,
        ).read()
    except Exception as e:
        print(f"  flush_cache: {e}", flush=True)


def stream_generate(base: str, input_ids: list[int], max_new: int) -> dict:
    payload = {
        "input_ids": input_ids,
        "sampling_params": {"max_new_tokens": max_new, "temperature": 0.0,
                            "ignore_eos": True},
        "stream": True,
    }
    req = urllib.request.Request(
        base + "/generate", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    stamps: list[float] = []
    meta = {}
    with urllib.request.urlopen(req, timeout=7200) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            stamps.append(time.time())
            try:
                meta = json.loads(body).get("meta_info", meta)
            except json.JSONDecodeError:
                pass
    ttft = (stamps[0] - t0) if stamps else float("nan")
    # First inter-token gap can absorb scheduler slack; use the median of the
    # steady-state gaps as the per-decode-step time (no spec decode, so one
    # streamed token is exactly one forward pass).
    gaps = [1000.0 * (b - a) for a, b in zip(stamps, stamps[1:])]
    gaps_sorted = sorted(gaps[1:]) if len(gaps) > 2 else sorted(gaps)
    median = gaps_sorted[len(gaps_sorted) // 2] if gaps_sorted else float("nan")
    return {
        "ttft_s": ttft,
        "n_streamed": len(stamps),
        "itl_median_ms": median,
        "itl_mean_ms": sum(gaps) / len(gaps) if gaps else float("nan"),
        "itl_min_ms": min(gaps) if gaps else float("nan"),
        "wall_s": time.time() - t0,
        "meta_info": meta,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30100)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--gap-s", type=float, default=25.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="", help="comma separated labels subset")
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    wanted = set(args.only.split(",")) if args.only else None
    rng = random.Random(1234)

    results = []
    print("warmup", flush=True)
    stream_generate(base, [rng.randint(1000, 100000) for _ in range(4096)], 8)
    time.sleep(args.gap_s)

    for label, isl in CONTEXTS:
        if wanted and label not in wanted:
            continue
        ids = [rng.randint(1000, 100000) for _ in range(isl)]
        flush(base)
        # The idle gap is the island delimiter the parser keys on.
        print(f"[{label}] idle {args.gap_s:.0f}s, then isl={isl}", flush=True)
        time.sleep(args.gap_s)
        t_start = time.time()
        r = stream_generate(base, ids, args.max_new)
        r.update({"label": label, "isl": isl, "t_start": t_start,
                  "t_end": time.time()})
        results.append(r)
        print(
            f"[{label}] ttft={r['ttft_s']:.2f}s tokens={r['n_streamed']} "
            f"itl_median={r['itl_median_ms']:.2f}ms itl_min={r['itl_min_ms']:.2f}ms",
            flush=True,
        )
        with open(args.out, "w") as fh:
            json.dump({"contexts": results, "max_new": args.max_new,
                       "gap_s": args.gap_s}, fh, indent=1)

    time.sleep(args.gap_s)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
