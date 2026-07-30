#!/usr/bin/env python3
"""Attribute prefill kernel time to K3 block types, straight from the ranges.

Prefill differs from decode in two ways that make this the easier measurement.
It does not run through a CUDA graph, so the profiler sees every kernel; and its
kernels are millisecond-scale, so per-rank launch jitter is negligible against
them and the TP8 collectives report their real cost instead of spin-wait. Both
mean the numbers here can be read directly, collectives included.

  ./prefill_bucketize.py traces/pf8k --out results/pf8k.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

from bucketize import analyse

N_KDA_LAYERS = 69
N_FULL_ATTN = 24
BUCKET_ORDER = ["K3/kda", "K3/full_attn", "K3/moe", "K3/dense_mlp", "other"]
PRETTY = {
    "K3/kda": "KDA",
    "K3/full_attn": "full attention (MLA)",
    "K3/moe": "MoE FFN",
    "K3/dense_mlp": "dense MLP (layer 0)",
    "other": "attn-residual / norms / other",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dir")
    ap.add_argument("--tp", type=int, default=0)
    ap.add_argument("--out")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.trace_dir, f"*TP-{args.tp}*.trace.json.gz")))
    if not paths:
        print(f"no TP-{args.tp} trace in {args.trace_dir}", file=sys.stderr)
        return 1

    r = analyse(paths[0])

    run = {}
    rj = os.path.join(args.trace_dir, "run.json")
    if os.path.exists(rj):
        with open(rj) as fh:
            run = json.load(fh)
    isl = run.get("isl")
    chunks_expected = run.get("chunks")

    chunks = r["kda_range_count"] / N_KDA_LAYERS if r["kda_range_count"] else 0
    total_ms = r["total_gpu_us"] / 1000.0

    print(f"=== {r['file']} (isl={isl}) ===")
    print(f"chunks profiled = {chunks:.2f} (expected {chunks_expected}); "
          f"full-attn ranges = {r['full_attn_range_count']} "
          f"(expected {N_FULL_ATTN * (chunks_expected or 1)})")
    print(f"kernel events = {r['kernel_events']}")
    print(f"GPU time summed over profiled chunks: {total_ms:.2f} ms")
    if isl:
        print(f"  per input token: {r['total_gpu_us']/isl:.2f} us/token")
    if run.get("ref_ttft_s"):
        ttft_ms = run["ref_ttft_s"] * 1000
        print(f"  unprofiled TTFT: {ttft_ms:.1f} ms  "
              f"(GPU busy / TTFT = {total_ms/ttft_ms:.2f})")
        print(f"  prefill throughput: {run['ref_prefill_tok_per_s']:,.0f} tok/s")
    print("composition (collectives included -- see module docstring):")
    for b in BUCKET_ORDER:
        if b in r["buckets"]:
            bb = r["buckets"][b]
            print(f"  {bb['pct']:6.2f}%  {bb['us']/1000:9.3f} ms  {PRETTY[b]}")
            print(f"           {bb['by_class']}")

    payload = {
        "isl": isl,
        "chunks": chunks,
        "total_gpu_ms": total_ms,
        "us_per_input_token": (r["total_gpu_us"] / isl) if isl else None,
        "ref_ttft_s": run.get("ref_ttft_s"),
        "ref_prefill_tok_per_s": run.get("ref_prefill_tok_per_s"),
        "buckets": {
            b: {
                "ms": r["buckets"][b]["us"] / 1000.0,
                "pct": r["buckets"][b]["pct"],
                "by_class_ms": {c: v / 1000.0 for c, v in r["buckets"][b]["by_class"].items()},
                "top_kernels": r["buckets"][b]["top_kernels"],
            }
            for b in r["buckets"]
        },
        "unattributed_top": r["unattributed_top"][:10],
    }
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=1)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
