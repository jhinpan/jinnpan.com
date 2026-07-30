#!/usr/bin/env python3
"""Attribute GPU kernel time in an SGLang decode trace to K3 block types.

Every GPU kernel event in a chrome trace carries a correlation id shared with the
host-side launch (`cuda_runtime`) event that issued it. The launch event sits on a
host thread inside whatever `record_function` ranges were open at that moment, so
walking the innermost enclosing `K3/*` range gives each kernel a block type --
KDA, full attention (MLA), MoE -- that no kernel-name heuristic can recover, since
Tensile GEMM names encode tile shapes rather than which projection they served.

Kernels are additionally classified by *what they do* (collective, GEMM, attention
core, elementwise/other) so the block breakdown can be cross-tabulated against
communication cost.

  ./bucketize.py traces/ctx64k --out ctx64k.json
"""
from __future__ import annotations

import argparse
import bisect
import collections
import glob
import gzip
import json
import os
import re
import sys

# Innermost-wins ordering is handled by nesting depth, not this list; it only
# names the ranges we care to keep as buckets.
K3_RANGES = ("K3/kda", "K3/full_attn", "K3/moe", "K3/dense_mlp")

COLLECTIVE_PAT = re.compile(
    r"cross_device_reduce|allgather|all_gather|allreduce|all_reduce|reduce_scatter"
    r"|nccl|rccl|ncclDevKernel|quick_reduce|one_shot|two_shot",
    re.I,
)
# Tensile (`Cijk_*`) and the hipBLASLt/CK entry points are the dense GEMMs; the
# aiter MoE kernels are matched before these so they are not swallowed here.
GEMM_PAT = re.compile(r"^Cijk_|Cijk_|gemm|Gemm|GEMM|hipblaslt|_ck_|wvSplitK|splitk", re.I)
MOE_KERNEL_PAT = re.compile(r"moe|expert|topk|Topk|TopK", re.I)
ATTN_CORE_PAT = re.compile(
    r"_fwd_kernel|_fwd_grouped_kernel|fwd_kernel_stage|attn_fwd|flash|mla_decode"
    r"|paged_attention|_attn_",
    re.I,
)
# KDA's own state-space maths: the gated delta-rule recurrence, the depthwise
# short conv over the 4-token window, and the fused gated output norm.
KDA_CORE_PAT = re.compile(
    r"sigmoid_gating|delta_rule|fused_recurrent|chunk_kda|kda_|causal_conv1d"
    r"|conv1d_update|fused_norm_gate|rmsnorm_gated|mamba",
    re.I,
)


def classify_kernel(name: str) -> str:
    if COLLECTIVE_PAT.search(name):
        return "collective"
    if KDA_CORE_PAT.search(name):
        return "kda_core"
    if ATTN_CORE_PAT.search(name):
        return "attn_core"
    if MOE_KERNEL_PAT.search(name):
        return "moe_kernel"
    if GEMM_PAT.search(name):
        return "gemm"
    return "other"


def load(path: str) -> dict:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        return json.load(fh)


def analyse(path: str) -> dict:
    trace = load(path)
    events = trace["traceEvents"]

    kernels = []
    runtime_by_corr: dict[int, dict] = {}
    # Ranges are grouped per host thread: a launch is inside a range only if it is
    # on the same thread.
    ranges_by_tid: dict[int, list[dict]] = collections.defaultdict(list)

    for e in events:
        cat = e.get("cat")
        if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            kernels.append(e)
        elif cat == "cuda_runtime" or cat == "hip_runtime":
            corr = (e.get("args") or {}).get("correlation")
            if corr is not None:
                runtime_by_corr[corr] = e
        elif cat == "user_annotation":
            name = e.get("name", "")
            if name.startswith("K3/"):
                ranges_by_tid[e.get("tid")].append(e)

    # Sort each thread's ranges by start so a launch timestamp can be located by
    # binary search; among the ranges that contain it, the shortest is innermost.
    prepared: dict[int, tuple[list[float], list[dict]]] = {}
    for tid, rs in ranges_by_tid.items():
        rs.sort(key=lambda r: r["ts"])
        prepared[tid] = ([r["ts"] for r in rs], rs)

    def range_for(tid: int, ts: float) -> str | None:
        entry = prepared.get(tid)
        if entry is None:
            return None
        starts, rs = entry
        i = bisect.bisect_right(starts, ts)
        best = None
        best_dur = None
        # Ranges here nest at most a couple deep, so a short backward scan finds
        # every candidate that can still be open at ts.
        for j in range(i - 1, max(-1, i - 400), -1):
            r = rs[j]
            if r["ts"] + r["dur"] >= ts:
                if best_dur is None or r["dur"] < best_dur:
                    best, best_dur = r["name"], r["dur"]
        return best

    per_bucket: dict[str, dict] = collections.defaultdict(
        lambda: {"us": 0.0, "n": 0, "by_class": collections.defaultdict(float),
                 "by_kernel": collections.defaultdict(lambda: [0, 0.0])}
    )
    unattributed_names: dict[str, list] = collections.defaultdict(lambda: [0, 0.0])

    total_us = 0.0
    for k in kernels:
        dur = float(k.get("dur", 0.0))
        total_us += dur
        corr = (k.get("args") or {}).get("correlation")
        rt = runtime_by_corr.get(corr) if corr is not None else None
        bucket = None
        if rt is not None:
            bucket = range_for(rt.get("tid"), rt["ts"])
        if bucket is None:
            bucket = "other"
            rec = unattributed_names[k["name"]]
            rec[0] += 1
            rec[1] += dur
        b = per_bucket[bucket]
        b["us"] += dur
        b["n"] += 1
        cls = classify_kernel(k["name"])
        b["by_class"][cls] += dur
        kr = b["by_kernel"][k["name"]]
        kr[0] += 1
        kr[1] += dur

    # Number of decode steps in the window, inferred from how many times the KDA
    # range appeared: 69 KDA layers per forward pass.
    kda_range_count = sum(
        1 for rs in ranges_by_tid.values() for r in rs if r["name"] == "K3/kda"
    )
    full_range_count = sum(
        1 for rs in ranges_by_tid.values() for r in rs if r["name"] == "K3/full_attn"
    )

    out = {
        "file": os.path.basename(path),
        "total_gpu_us": total_us,
        "kernel_events": len(kernels),
        "kda_range_count": kda_range_count,
        "full_attn_range_count": full_range_count,
        "buckets": {},
        "unattributed_top": sorted(
            ({"name": n, "n": v[0], "us": v[1]} for n, v in unattributed_names.items()),
            key=lambda d: -d["us"],
        )[:25],
    }
    for name, b in sorted(per_bucket.items(), key=lambda kv: -kv[1]["us"]):
        out["buckets"][name] = {
            "us": b["us"],
            "pct": 100.0 * b["us"] / total_us if total_us else 0.0,
            "n": b["n"],
            "by_class": {c: round(v, 1) for c, v in sorted(b["by_class"].items(), key=lambda kv: -kv[1])},
            "top_kernels": [
                {"name": n, "n": v[0], "us": round(v[1], 1)}
                for n, v in sorted(b["by_kernel"].items(), key=lambda kv: -kv[1][1])[:12]
            ],
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dir")
    ap.add_argument("--stage", default="DECODE", choices=["DECODE", "EXTEND"])
    ap.add_argument("--tp", type=int, default=0, help="-1 for every rank")
    ap.add_argument("--out")
    args = ap.parse_args()

    pat = (
        f"*TP-{args.tp}*-{args.stage}.trace.json.gz"
        if args.tp >= 0
        else f"*-{args.stage}.trace.json.gz"
    )
    paths = sorted(glob.glob(os.path.join(args.trace_dir, pat)))
    if not paths:
        print(f"no traces matching {pat} in {args.trace_dir}", file=sys.stderr)
        return 1

    results = [analyse(p) for p in paths]
    payload = {"stage": args.stage, "ranks": results}

    for r in results:
        print(f"\n=== {r['file']} ===")
        print(
            f"total GPU {r['total_gpu_us']/1000:.2f} ms over {r['kernel_events']} kernels; "
            f"KDA ranges={r['kda_range_count']} full_attn ranges={r['full_attn_range_count']}"
        )
        steps = r["kda_range_count"] / 69 if r["kda_range_count"] else 0
        if steps:
            print(f"inferred decode steps = {steps:.2f}  "
                  f"-> GPU busy {r['total_gpu_us']/1000/steps:.3f} ms/step")
        for name, b in r["buckets"].items():
            print(f"  {b['pct']:6.2f}%  {b['us']/1000:9.3f} ms  n={b['n']:7d}  {name}")
            print(f"           by class: {b['by_class']}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=1)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
