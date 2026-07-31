#!/usr/bin/env python3
"""Rebuild prefill-composition.csv from the traces, two independent ways.

`bucketize.py` attributes a kernel by walking its correlation id back to the
host-side `record_function` range that was open when it was launched. If that walk
is wrong, re-running it cannot reveal the error. So this recomputes the same table
a second way, from the `gpu_user_annotation` ranges the PyTorch profiler itself
projects onto the device timeline, and prints both against the published CSV.

It also runs the two sanity checks the whole method rests on:

  * the GPU bands must not overlap, or containment double counts;
  * the two methods must agree kernel by kernel, not just in aggregate.

  ./verify_composition.py ../traces --csv .../results/prefill-composition.csv
"""
from __future__ import annotations

import argparse
import bisect
import collections
import csv
import glob
import gzip
import json
import os
import re
import sys

COLLECTIVE_PAT = re.compile(
    r"cross_device_reduce|allgather|all_gather|allreduce|all_reduce|reduce_scatter"
    r"|nccl|rccl|quick_reduce|quickreduce|one_shot|two_shot", re.I)
N_KDA, N_FULL = 69, 24
POINTS = (("pf1k", "1k"), ("pf4k", "4k"), ("pf8k", "8k"), ("pf32k", "32k"))


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        return json.load(fh)


def split_events(ev):
    kernels, rt_by_corr = [], {}
    host = collections.defaultdict(list)
    bands = []
    for e in ev:
        cat = e.get("cat")
        if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            kernels.append(e)
        elif cat in ("cuda_runtime", "hip_runtime"):
            c = (e.get("args") or {}).get("correlation")
            if c is not None:
                rt_by_corr[c] = e
        elif cat == "user_annotation" and e.get("name", "").startswith("K3/"):
            host[e.get("tid")].append(e)
        elif cat == "gpu_user_annotation" and e.get("name", "").startswith("K3/"):
            bands.append(e)
    return kernels, rt_by_corr, host, bands


def attribute_corr(kernels, rt_by_corr, host):
    """Method A: kernel -> launch -> innermost open host range."""
    prepared = {}
    for tid, rs in host.items():
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
            if r["ts"] + r["dur"] >= ts and (best_dur is None or r["dur"] < best_dur):
                best, best_dur = r["name"], r["dur"]
        return best

    out = []
    for k in kernels:
        c = (k.get("args") or {}).get("correlation")
        rt = rt_by_corr.get(c) if c is not None else None
        out.append(range_for(rt.get("tid"), rt["ts"]) if rt is not None else None)
    return out


def attribute_band(kernels, bands):
    """Method B: kernel -> the profiler's own GPU-projected range covering it."""
    bands = sorted(bands, key=lambda b: b["ts"])
    starts = [b["ts"] for b in bands]
    out = []
    for k in kernels:
        i = bisect.bisect_right(starts, k["ts"])
        b = None
        for j in range(i - 1, max(-1, i - 50), -1):
            if bands[j]["ts"] + bands[j]["dur"] > k["ts"]:
                b = bands[j]["name"]
                break
        out.append(b)
    return out


def compose(kernels, labels):
    """ms by block with collectives stripped, plus the collective total."""
    net = collections.defaultdict(float)
    coll = 0.0
    for k, lab in zip(kernels, labels):
        if COLLECTIVE_PAT.search(k["name"]):
            coll += k["dur"]
            continue
        net[lab or "other"] += k["dur"]
    net["other"] += net.pop("K3/dense_mlp", 0.0)
    return {k: v / 1000 for k, v in net.items()}, coll / 1000


def union_ms(intervals):
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    total = 0
    cs, ce = intervals[0]
    for s, e in intervals[1:]:
        if s > ce:
            total += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return (total + ce - cs) / 1000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_root")
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()

    pub = {}
    with open(args.csv) as fh:
        for row in csv.DictReader(fh):
            pub[row["context"]] = row

    hdr = (f"{'pt':>6} {'method':>12} {'coll ms':>9} "
           f"{'KDA%':>8} {'MLA%':>8} {'MoE%':>8} {'other%':>8} "
           f"{'MLA ms':>9} {'KDA ms':>9} {'MLA/layer':>10} {'MLA/KDA':>8}")
    print("### composition, rebuilt from the traces vs the published CSV\n")
    print(hdr)
    print("-" * len(hdr))

    checks = []
    for tag, ctx in POINTS:
        tr = glob.glob(os.path.join(args.trace_root, tag, "*TP-0*.trace.json.gz"))
        if not tr:
            continue
        ev = load(tr[0])["traceEvents"]
        kernels, rt, host, bands = split_events(ev)
        labs = {"correlation": attribute_corr(kernels, rt, host),
                "gpu band": attribute_band(kernels, bands)}

        for label in ("correlation", "gpu band"):
            net, coll = compose(kernels, labs[label])
            base = sum(net.values())
            kda = net.get("K3/kda", 0.0)
            mla = net.get("K3/full_attn", 0.0)
            moe = net.get("K3/moe", 0.0)
            oth = net.get("other", 0.0)
            print(f"{tag:>6} {label:>12} {coll:>9,.1f} "
                  f"{100*kda/base:>7.2f}% {100*mla/base:>7.2f}% "
                  f"{100*moe/base:>7.2f}% {100*oth/base:>7.2f}% "
                  f"{mla:>9,.1f} {kda:>9,.1f} {mla/N_FULL:>10.4f} "
                  f"{(mla/N_FULL)/(kda/N_KDA):>8.3f}")

        p = pub.get(ctx)
        if p:
            print(f"{'':>6} {'PUBLISHED':>12} {float(p['collective_ms']):>9,.1f} "
                  f"{float(p['kda_pct']):>7.2f}% {float(p['full_attn_pct']):>7.2f}% "
                  f"{float(p['moe_pct']):>7.2f}% {float(p['other_pct']):>7.2f}% "
                  f"{float(p['full_attn_ms']):>9,.1f} {float(p['kda_ms']):>9,.1f} "
                  f"{float(p['full_attn_ms_per_layer']):>10.4f} "
                  f"{float(p['full_attn_over_kda']):>8.3f}")
        print()
        checks.append((tag, kernels, bands, labs))

    print("\n### sanity: do the GPU bands overlap, and do the two methods agree?\n")
    print(f"{'pt':>6} {'gpu ms':>9} {'bands ms':>10} {'band union':>11} "
          f"{'overlap':>9} {'kernels in >1 band':>19} {'method disagreement':>21}")
    print("-" * 92)
    for tag, kernels, bands, labs in checks:
        gpu_ms = sum(k["dur"] for k in kernels) / 1000
        bands_ms = sum(b["dur"] for b in bands) / 1000
        u = union_ms([(b["ts"], b["ts"] + b["dur"]) for b in bands])
        starts = sorted(b["ts"] for b in bands)
        multi = 0
        for k in kernels:
            i = bisect.bisect_right(starts, k["ts"])
            c = sum(1 for b in bands
                    if b["ts"] <= k["ts"] < b["ts"] + b["dur"])
            if c > 1:
                multi += 1
        dis = sum(k["dur"] for k, a, b in
                  zip(kernels, labs["correlation"], labs["gpu band"])
                  if (a or "other") != (b or "other")) / 1000
        print(f"{tag:>6} {gpu_ms:>9,.1f} {bands_ms:>10,.1f} {u:>11,.1f} "
              f"{bands_ms-u:>9,.2f} {multi:>19} "
              f"{dis:>13,.2f} ms ({100*dis/gpu_ms:.2f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
