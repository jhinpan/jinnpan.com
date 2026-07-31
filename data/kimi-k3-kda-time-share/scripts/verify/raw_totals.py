#!/usr/bin/env python3
"""Third path: read the chrome JSON directly, with no importer in between.

trace_processor sees 47 `_fwd_kernel` dispatches where bucketize.py sees 48, and
the structural count says 48 (24 MLA layers x 2 chunks). One of the two is dropping
an event. This settles which, and prints the launch geometry the profiler recorded
for each dispatch so chunk 1 and chunk 2 can be compared directly.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import sys


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        return json.load(fh)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--kernel", default="_fwd_kernel")
    args = ap.parse_args()

    ev = load(args.trace)["traceEvents"]

    by_cat = collections.Counter()
    dur_by_cat = collections.Counter()
    for e in ev:
        c = e.get("cat")
        by_cat[c] += 1
        if e.get("dur"):
            dur_by_cat[c] += e["dur"]

    print("=== raw event counts by cat ===")
    for c, n in by_cat.most_common(12):
        print(f"  {str(c):<24} n={n:<8} sum_dur={dur_by_cat[c]/1000:12,.2f} ms")

    gpu = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    total = sum(e["dur"] for e in gpu)
    iv = sorted((e["ts"], e["ts"] + e["dur"]) for e in gpu)
    union, cs, ce = 0, *iv[0]
    for s, e in iv[1:]:
        if s > ce:
            union += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    union += ce - cs
    span = max(e["ts"] + e["dur"] for e in gpu) - min(e["ts"] for e in gpu)
    print(f"\n  GPU dispatches (kernel+memcpy+memset) = {len(gpu)}")
    print(f"  sum of dur                            = {total/1000:,.2f} ms")
    print(f"  union of intervals                    = {union/1000:,.2f} ms "
          f"(sum overstates by {100*(total-union)/total:.3f}%)")
    print(f"  first -> last dispatch                = {span/1000:,.2f} ms "
          f"(GPU busy {union/span:.3f})")

    # duplicate (tid, ts) pairs are what a nesting-based importer has to drop
    seen = collections.Counter((e.get("tid"), e.get("ts")) for e in gpu)
    dups = [k for k, v in seen.items() if v > 1]
    print(f"  dispatches sharing an exact (tid, ts) = {len(dups)}")

    k = [e for e in gpu if e["name"] == args.kernel]
    k.sort(key=lambda e: e["ts"])
    print(f"\n=== {args.kernel}: {len(k)} dispatches, "
          f"{sum(e['dur'] for e in k)/1000:,.2f} ms total ===")
    print("   #   dur_ms   grid                block            regs  stream")
    for i, e in enumerate(k):
        a = e.get("args") or {}
        print(f"  {i:3d} {e['dur']/1000:8.3f}   {str(a.get('grid')):<18} "
              f"{str(a.get('block')):<16} {str(a.get('registers per thread')):<5} "
              f"{a.get('stream')}")

    # Does any pair of _fwd_kernel dispatches overlap in device time?
    ov = 0
    for i in range(len(k) - 1):
        if k[i]["ts"] + k[i]["dur"] > k[i + 1]["ts"]:
            ov += 1
    print(f"\n  overlapping consecutive pairs: {ov}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
