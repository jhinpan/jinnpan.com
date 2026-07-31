#!/usr/bin/env python3
"""Diff the two chunked-prefill passes of one request, kernel by kernel.

At 32K the second chunk's attention costs 12.8x the first chunk's while doing only
3x the attention work. Either it runs kernels the first chunk does not, or it runs
the same kernels less efficiently. The trace can distinguish those without another
server run: split the kernel stream at the forward-pass boundary and compare.

  ./chunk_diff.py traces/pf32k/pf32k-TP-0.trace.json.gz
"""
from __future__ import annotations

import collections
import gzip
import json
import sys

BOUNDARY = "_vocab_parallel_embedding_kernel"


def main(path: str) -> int:
    tr = json.load(gzip.open(path, "rt"))
    ks = sorted((e for e in tr["traceEvents"]
                 if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")),
                key=lambda e: float(e["ts"]))

    marks = [i for i, e in enumerate(ks) if e["name"] == BOUNDARY]
    print(f"{path}\n  {len(ks)} kernels, {len(marks)} forward passes "
          f"(by {BOUNDARY})")
    if len(marks) < 2:
        print("  need >= 2 passes to diff", file=sys.stderr)
        return 1

    bounds = marks + [len(ks)]
    passes = []
    for a, b in zip(bounds, bounds[1:]):
        agg = collections.defaultdict(lambda: [0, 0.0])
        for e in ks[a:b]:
            r = agg[e["name"]]
            r[0] += 1
            r[1] += float(e.get("dur", 0.0))
        passes.append(agg)

    for i, agg in enumerate(passes):
        tot = sum(v[1] for v in agg.values())
        print(f"  pass {i+1}: {sum(v[0] for v in agg.values())} kernels, "
              f"{tot/1000:.1f} ms GPU")

    if len(passes) != 2:
        print("  (more than two passes; comparing the first two)")
    p1, p2 = passes[0], passes[1]
    names = set(p1) | set(p2)

    only2 = [n for n in names if n not in p1]
    only1 = [n for n in names if n not in p2]
    print(f"\n  kernels only in pass 2: {len(only2)}")
    for n in sorted(only2, key=lambda n: -p2[n][1])[:10]:
        print(f"      {p2[n][1]/1000:8.2f} ms  n={p2[n][0]:4d}  {n[:60]}")
    print(f"  kernels only in pass 1: {len(only1)}")
    for n in sorted(only1, key=lambda n: -p1[n][1])[:10]:
        print(f"      {p1[n][1]/1000:8.2f} ms  n={p1[n][0]:4d}  {n[:60]}")

    print(f"\n  shared kernels, biggest pass2 - pass1 deltas:")
    print(f"  {'p1 ms':>9} {'p2 ms':>9} {'ratio':>7} {'n1':>5} {'n2':>5}  name")
    shared = [n for n in names if n in p1 and n in p2]
    for n in sorted(shared, key=lambda n: -(p2[n][1] - p1[n][1]))[:12]:
        r = p2[n][1] / p1[n][1] if p1[n][1] else float("inf")
        print(f"  {p1[n][1]/1000:9.2f} {p2[n][1]/1000:9.2f} {r:7.2f} "
              f"{p1[n][0]:5d} {p2[n][0]:5d}  {n[:56]}")
    return 0


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
