#!/usr/bin/env python3
"""Consolidate the per-context decode traces into one table + a canvas-ready JSON.

Refines the four coarse buckets into the cost centres a reader actually asks about,
and splits each block's time into the part that is specific to the mechanism (the
KDA recurrence, the MLA KV scan, the expert GEMMs) versus the ordinary projections
around it -- which is where most of a "KDA layer" turns out to go.

  ./summarize.py --out results/summary.json
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

from bucketize import classify_kernel
from chrome_bucketize import kernel_totals, load_name_map

N_KDA_LAYERS = 69
N_FULL_ATTN = 24
N_MOE = 92
N_LAYERS = 93

KDA_RECURRENT = "fused_recurrent_kda_packed_decode_kernel"

# Kernels the ground-truth run put outside every block range. They are real work,
# so name them instead of leaving them in a shrug bucket: _score_/_combine_ are
# K3's attention-residual bank (attn_res_block_size=12, two per layer), and the
# 187-per-step add_rmsnorm is the pair of block layernorms.
ATTN_RESIDUAL_KERNELS = ("_score_kernel", "_combine_kernel")


def refine(per_name: dict[str, list], name_map: dict, steps: int) -> dict:
    """Distribute per-name device time over the reported cost centres."""
    groups = collections.defaultdict(float)
    detail = collections.defaultdict(lambda: collections.defaultdict(float))

    for name, (n, us) in per_name.items():
        cls = classify_kernel(name)
        if cls == "collective":
            groups["collective"] += us
            continue
        if name in ATTN_RESIDUAL_KERNELS:
            groups["attn_residual"] += us
            detail["attn_residual"][name] += us
            continue

        weights = name_map.get(name, {"other": 1.0})
        for bucket, w in weights.items():
            share = us * w
            if bucket == "K3/kda":
                groups["kda"] += share
                key = ("recurrent_state_update" if name == KDA_RECURRENT
                       else "conv_short" if "conv1d" in name
                       else "gated_out_norm" if "layer_norm_gated" in name
                       else "projections" if cls == "gemm"
                       else "misc")
                detail["kda"][key] += share
            elif bucket == "K3/full_attn":
                groups["full_attn"] += share
                key = ("kv_scan_stage1" if "stage1" in name
                       else "split_reduce_stage2" if "stage2" in name
                       else "projections" if cls == "gemm"
                       else "misc")
                detail["full_attn"][key] += share
            elif bucket in ("K3/moe", "K3/dense_mlp"):
                groups["moe"] += share
                key = ("expert_gemms" if name.startswith("mfma_moe")
                       else "route_sort" if cls == "moe_kernel"
                       else "shared_expert_gemms" if cls == "gemm"
                       else "misc")
                detail["moe"][key] += share
            else:
                groups["other"] += share
                detail["other"][name[:60]] += share

    return {
        "groups_us_per_step": {k: v / steps for k, v in groups.items()},
        "detail_us_per_step": {
            g: {k: v / steps for k, v in sorted(d.items(), key=lambda kv: -kv[1])}
            for g, d in detail.items()
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-root", default="/sgl-workspace/workspace/kda_prof/traces")
    ap.add_argument("--name-map",
                    default="/sgl-workspace/workspace/kda_prof/results/name_map_4k.json")
    ap.add_argument("--prefix", default="ctx")
    ap.add_argument("--out")
    args = ap.parse_args()

    name_map = load_name_map(args.name_map)
    order = ["4k", "32k", "64k", "512k", "1m"]
    rows = []

    for label in order:
        d = os.path.join(args.trace_root, f"{args.prefix}{label}")
        traces = [p for p in glob.glob(os.path.join(d, "*TP-0*.trace.json.gz"))
                  if "EXTEND" not in p]
        if not traces:
            continue
        per_name, total_us = kernel_totals(traces[0])
        n_rec = per_name.get(KDA_RECURRENT, [0, 0.0])[0]
        if not n_rec or n_rec % N_KDA_LAYERS:
            print(f"{label}: bad KDA dispatch count {n_rec}", file=sys.stderr)
            continue
        steps = n_rec // N_KDA_LAYERS

        r = refine(per_name, name_map, steps)
        g = r["groups_us_per_step"]
        compute = sum(v for k, v in g.items() if k != "collective")

        run = {}
        rj = os.path.join(d, "run.json")
        if os.path.exists(rj):
            with open(rj) as fh:
                run = json.load(fh)

        rows.append({
            "label": label,
            "isl": run.get("isl"),
            "decode_steps": steps,
            "ttft_s": run.get("ttft_s"),
            "eager_itl_ms": run.get("itl_median_ms"),
            "gpu_busy_us_per_step": total_us / steps,
            "collective_us_per_step": g.get("collective", 0.0),
            "compute_us_per_step": compute,
            "groups_us_per_step": g,
            "groups_pct_of_compute": {k: 100.0 * v / compute
                                      for k, v in g.items() if k != "collective"},
            "detail_us_per_step": r["detail_us_per_step"],
        })

    hdr = (f"{'ctx':>5} {'steps':>5} {'compute':>9} {'KDA':>15} {'FullAttn':>15} "
           f"{'MoE':>15} {'AttnRes':>13} {'Other':>13}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        g, p = r["groups_us_per_step"], r["groups_pct_of_compute"]
        def cell(k):
            return f"{g.get(k,0)/1000:6.2f}ms {p.get(k,0):5.1f}%"
        print(f"{r['label']:>5} {r['decode_steps']:5d} "
              f"{r['compute_us_per_step']/1000:7.2f}ms "
              f"{cell('kda'):>15} {cell('full_attn'):>15} {cell('moe'):>15} "
              f"{cell('attn_residual'):>13} {cell('other'):>13}")

    print("\nmechanism-specific kernels (us/step):")
    print(f"{'ctx':>5} {'KDA recurrence':>15} {'per layer':>10} "
          f"{'MLA kv-scan':>12} {'per layer':>10} {'MLA reduce':>11} {'expert GEMMs':>13}")
    for r in rows:
        kd = r["detail_us_per_step"].get("kda", {})
        fa = r["detail_us_per_step"].get("full_attn", {})
        mo = r["detail_us_per_step"].get("moe", {})
        rec = kd.get("recurrent_state_update", 0)
        s1 = fa.get("kv_scan_stage1", 0)
        s2 = fa.get("split_reduce_stage2", 0)
        print(f"{r['label']:>5} {rec:15.1f} {rec/N_KDA_LAYERS:10.2f} "
              f"{s1:12.1f} {s1/N_FULL_ATTN:10.2f} {s2:11.1f} "
              f"{mo.get('expert_gemms',0):13.1f}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"rows": rows}, fh, indent=1)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
