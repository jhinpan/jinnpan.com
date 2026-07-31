#!/usr/bin/env python3
"""Split a chunked prefill trace by chunk and show why the chunks differ.

A 32K prefill is two 16384-token chunks, and they are not alike: chunk 1 has no
prefix, chunk 2 attends to all of chunk 1. Averaging them hides that; looking only
at the start of the trace hides the opposite thing. This prints, per chunk, the
block composition, the kernel inventory inside the full-attention block, and where
in wall-clock time the attention actually sits.

  ./verify_chunks.py ../traces/pf32k/pf32k-TP-0.trace.json.gz
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import sys

BIG_KERNEL_US = 10_000


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        return json.load(fh)


def short(n, w=56):
    return n if len(n) <= w else n[: w - 3] + "..."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--kernel", default="_fwd_kernel")
    args = ap.parse_args()
    ev = load(args.trace)["traceEvents"]

    gpu = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    t0 = min(e["ts"] for e in gpu)
    t1 = max(e["ts"] + e["dur"] for e in gpu)

    def ann(name):
        return sorted((e for e in ev
                       if e.get("cat") == "gpu_user_annotation" and e.get("name") == name),
                      key=lambda e: e["ts"])

    runs = ann("scheduler.run_batch")
    blocks = collections.defaultdict(list)
    for e in ev:
        if e.get("cat") == "gpu_user_annotation" and e.get("name", "").startswith("K3/"):
            blocks[e["name"]].append(e)

    print(f"GPU timeline 0 .. {(t1-t0)/1e6:.3f} s, {len(gpu):,} dispatches, "
          f"{sum(e['dur'] for e in gpu)/1000:,.2f} ms of kernel time")

    for ci, rb in enumerate(runs):
        lo, hi = rb["ts"], rb["ts"] + rb["dur"]
        mine = [k for k in gpu if lo <= k["ts"] < hi]
        tot = sum(k["dur"] for k in mine)
        print(f"\n########## chunk {ci+1}: {(lo-t0)/1e6:.3f} .. {(hi-t0)/1e6:.3f} s, "
              f"{len(mine):,} kernels, {tot/1000:,.2f} ms ##########")
        for bname, bs in sorted(blocks.items()):
            inside = [b for b in bs if lo <= b["ts"] < hi]
            ms = sum(
                k["dur"] for b in inside for k in gpu
                if b["ts"] <= k["ts"] < b["ts"] + b["dur"]
            )
            if inside:
                print(f"  {bname:<16} n={len(inside):<4} {ms/1000:9,.2f} ms   "
                      f"{100*ms/tot:5.1f}% of the chunk")

        inv = collections.defaultdict(lambda: [0, 0.0])
        for b in [b for b in blocks.get("K3/full_attn", []) if lo <= b["ts"] < hi]:
            for k in gpu:
                if b["ts"] <= k["ts"] < b["ts"] + b["dur"]:
                    inv[k["name"]][0] += 1
                    inv[k["name"]][1] += k["dur"]
        if inv:
            print("  inside K3/full_attn:")
            for n, v in sorted(inv.items(), key=lambda kv: -kv[1][1])[:10]:
                print(f"    n={v[0]:<4} {v[1]/1000:9,.2f} ms  {short(n)}")

    k = sorted((e for e in gpu if e["name"] == args.kernel), key=lambda e: e["ts"])
    if not k:
        return 0
    print(f"\n=== {args.kernel}: {len(k)} dispatches, "
          f"{sum(e['dur'] for e in k)/1000:,.2f} ms, by launch geometry ===")
    geo = collections.defaultdict(lambda: [0, 0.0])
    for e in k:
        a = e.get("args") or {}
        geo[(tuple(a.get("grid") or []), tuple(a.get("block") or []))][0] += 1
        geo[(tuple(a.get("grid") or []), tuple(a.get("block") or []))][1] += e["dur"]
    for (grid, block), (n, us) in sorted(geo.items(), key=lambda kv: kv[1][1]):
        warps = block[0] // 64 if block else 0
        print(f"  grid={str(list(grid)):<16} block={str(list(block)):<14} "
              f"{warps} warps   n={n:<4} {us/1000:9,.2f} ms  "
              f"{us/1000/n:8.3f} ms each")

    small = [e for e in k if e["dur"] <= BIG_KERNEL_US]
    big = [e for e in k if e["dur"] > BIG_KERNEL_US]
    for label, group in ((f"<={BIG_KERNEL_US/1000:.0f} ms population", small),
                         (f">{BIG_KERNEL_US/1000:.0f} ms population", big)):
        if not group:
            continue
        a = (group[0]["ts"] - t0) / 1e6
        b = (group[-1]["ts"] + group[-1]["dur"] - t0) / 1e6
        ms = sum(e["dur"] for e in group) / 1000
        print(f"  {label}: n={len(group)}, {ms:,.1f} ms, over t={a:.3f}..{b:.3f} s "
              f"-> {100*ms/((b-a)*1000):.1f}% of that window")
    return 0


if __name__ == "__main__":
    sys.exit(main())
