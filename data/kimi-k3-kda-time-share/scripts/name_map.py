#!/usr/bin/env python3
"""How well does a kernel name alone identify the K3 block that launched it?

The eager instrumented run gives ground-truth labels (via record_function ranges)
but distorted collective durations. A CUDA-graph run has honest durations but no
ranges. Bridging the two requires knowing whether a name is unique to one block --
this script measures that, reporting per-name bucket distributions and the share of
compute time carried by unambiguous names.

  ./name_map.py traces/val4k --out name_map.json
"""
from __future__ import annotations

import argparse
import bisect
import collections
import glob
import gzip
import json
import os
import sys

from bucketize import classify_kernel


def load(path: str) -> dict:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        return json.load(fh)


def labelled_kernels(path: str):
    """Yield (kernel_name, duration_us, bucket, launch_ts) with bucket from the
    innermost enclosing K3/* range."""
    trace = load(path)
    runtime_by_corr = {}
    ranges_by_tid = collections.defaultdict(list)
    kernels = []
    for e in trace["traceEvents"]:
        cat = e.get("cat")
        if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            kernels.append(e)
        elif cat in ("cuda_runtime", "hip_runtime"):
            corr = (e.get("args") or {}).get("correlation")
            if corr is not None:
                runtime_by_corr[corr] = e
        elif cat == "user_annotation" and e.get("name", "").startswith("K3/"):
            ranges_by_tid[e.get("tid")].append(e)

    prepared = {}
    for tid, rs in ranges_by_tid.items():
        rs.sort(key=lambda r: r["ts"])
        prepared[tid] = ([r["ts"] for r in rs], rs)

    def range_for(tid, ts):
        entry = prepared.get(tid)
        if entry is None:
            return None
        starts, rs = entry
        i = bisect.bisect_right(starts, ts)
        best, best_dur = None, None
        for j in range(i - 1, max(-1, i - 400), -1):
            r = rs[j]
            if r["ts"] + r["dur"] >= ts:
                if best_dur is None or r["dur"] < best_dur:
                    best, best_dur = r["name"], r["dur"]
        return best

    for k in kernels:
        corr = (k.get("args") or {}).get("correlation")
        rt = runtime_by_corr.get(corr) if corr is not None else None
        bucket = range_for(rt.get("tid"), rt["ts"]) if rt is not None else None
        yield k["name"], float(k.get("dur", 0.0)), bucket or "other", k["ts"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dir")
    ap.add_argument("--out")
    args = ap.parse_args()

    paths = glob.glob(os.path.join(args.trace_dir, "*TP-0*-DECODE.trace.json.gz"))
    if not paths:
        print("no TP-0 decode trace", file=sys.stderr)
        return 1

    # name -> bucket -> [count, us]
    by_name = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0.0]))
    for name, dur, bucket, _ in labelled_kernels(paths[0]):
        rec = by_name[name][bucket]
        rec[0] += 1
        rec[1] += dur

    rows = []
    for name, buckets in by_name.items():
        tot_n = sum(v[0] for v in buckets.values())
        tot_us = sum(v[1] for v in buckets.values())
        # Dominance by *count*, not time: counts are exact, and eager collective
        # durations are inflated by rank skew.
        dom_bucket, dom = max(buckets.items(), key=lambda kv: kv[1][0])
        rows.append(
            {
                "name": name,
                "cls": classify_kernel(name),
                "n": tot_n,
                "us": round(tot_us, 1),
                "dominant": dom_bucket,
                "purity": dom[0] / tot_n,
                "split": {b: v[0] for b, v in sorted(buckets.items(), key=lambda kv: -kv[1][0])},
            }
        )
    rows.sort(key=lambda r: -r["n"])

    compute = [r for r in rows if r["cls"] != "collective"]
    comp_n = sum(r["n"] for r in compute)
    pure_n = sum(r["n"] for r in compute if r["purity"] > 0.999)
    print(f"{len(rows)} distinct kernel names; {len(compute)} non-collective")
    print(f"non-collective dispatches: {comp_n}, of which "
          f"{pure_n} ({100*pure_n/comp_n:.1f}%) have a name unique to one block")
    print()
    print(f"{'n':>7} {'purity':>7} {'dominant':<14} {'cls':<11} name")
    for r in rows[:45]:
        print(f"{r['n']:7d} {r['purity']:7.3f} {r['dominant']:<14} {r['cls']:<11} {r['name'][:78]}")
    print("\n--- ambiguous names (purity < 0.999), by dispatch count ---")
    for r in rows:
        if r["purity"] <= 0.999:
            print(f"{r['n']:7d} {r['cls']:<11} {r['split']}  {r['name'][:70]}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(rows, fh, indent=1)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
