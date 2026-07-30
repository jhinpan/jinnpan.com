#!/usr/bin/env python3
"""Consolidate the prefill points into one table + a canvas-ready JSON.

Reports per-input-token cost, which is the quantity that separates the linear
cost centres (KDA, MoE -- their per-token cost flattens out) from the quadratic
one (full attention -- its per-token cost keeps climbing).

Collectives are broken out rather than dropped. In prefill they carry real
payloads (a 16384-token chunk all-reduces 235 MB per layer) and the compute
kernels around them are millisecond-scale, so launch skew is a small correction
-- unlike decode, where the payload is 14 KB and the measurement was pure skew.
The correction is not zero at 1K, though, which is why both views are printed.

  ./summarize_prefill.py --out results/prefill_summary.json
"""
from __future__ import annotations

import argparse
import json
import os

N_KDA = 69
N_FULL = 24
N_MOE = 92

LABELS = [("1k", 1024), ("4k", 4096), ("8k", 8192), ("32k", 32768)]
BUCKETS = ["K3/kda", "K3/full_attn", "K3/moe", "K3/dense_mlp", "other"]
SHORT = {"K3/kda": "kda", "K3/full_attn": "full_attn", "K3/moe": "moe",
         "K3/dense_mlp": "dense_mlp", "other": "other"}

# The single kernel that carries MLA prefill attention, and the KDA chunked
# state-passing kernel that is its linear-attention counterpart.
MLA_PREFILL_KERNEL = "_fwd_kernel"
KDA_CHUNK_KERNELS = (
    "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
    "chunk_kda_fwd_kernel_intra_token_parallel",
    "chunk_gla_fwd_kernel_o",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/sgl-workspace/workspace/kda_prof/results")
    ap.add_argument("--prefix", default="pf")
    ap.add_argument("--out")
    args = ap.parse_args()

    rows = []
    for label, isl in LABELS:
        p = os.path.join(args.results, f"{args.prefix}{label}.json")
        if not os.path.exists(p):
            continue
        with open(p) as fh:
            d = json.load(fh)

        groups, coll = {}, 0.0
        kernel_ms = {}
        for b in BUCKETS:
            bb = d["buckets"].get(b)
            if not bb:
                continue
            c = bb["by_class_ms"].get("collective", 0.0)
            coll += c
            groups[SHORT[b]] = bb["ms"] - c
            for k in bb["top_kernels"]:
                kernel_ms[k["name"]] = kernel_ms.get(k["name"], 0.0) + k["us"] / 1000.0

        compute = sum(groups.values())
        total = compute + coll
        rows.append({
            "label": label,
            "isl": isl,
            "chunks": d["chunks"],
            "total_gpu_ms": total,
            "collective_ms": coll,
            "compute_ms": compute,
            "ttft_ms": (d["ref_ttft_s"] or 0) * 1000,
            "tok_per_s": d["ref_prefill_tok_per_s"],
            "gpu_busy_over_ttft": total / ((d["ref_ttft_s"] or 1) * 1000),
            "groups_ms": groups,
            "groups_pct_of_compute": {k: 100.0 * v / compute for k, v in groups.items()},
            "us_per_token": {k: 1000.0 * v / isl for k, v in groups.items()},
            "total_us_per_token": 1000.0 * total / isl,
            "kda_us_per_layer_per_ktok": 1e6 * groups["kda"] / N_KDA / isl,
            "full_attn_us_per_layer_per_ktok": 1e6 * groups["full_attn"] / N_FULL / isl,
            "mla_prefill_kernel_ms": kernel_ms.get(MLA_PREFILL_KERNEL, 0.0),
            "kda_chunk_kernels_ms": sum(kernel_ms.get(k, 0.0) for k in KDA_CHUNK_KERNELS),
        })

    hdr = (f"{'ctx':>4} {'TTFT':>8} {'tok/s':>9} {'busy/TTFT':>10} {'GPU ms':>9} "
           f"{'coll':>8} | {'KDA':>14} {'FullAttn':>14} {'MoE':>14} {'other':>13}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        g, p = r["groups_ms"], r["groups_pct_of_compute"]
        def cell(k):
            return f"{g.get(k,0):7.1f}ms {p.get(k,0):5.1f}%"
        other = g.get("other", 0) + g.get("dense_mlp", 0)
        otherp = p.get("other", 0) + p.get("dense_mlp", 0)
        print(f"{r['label']:>4} {r['ttft_ms']:7.0f}ms {r['tok_per_s']:9,.0f} "
              f"{r['gpu_busy_over_ttft']:10.2f} {r['total_gpu_ms']:8.1f} "
              f"{r['collective_ms']:7.1f} | {cell('kda'):>14} {cell('full_attn'):>14} "
              f"{cell('moe'):>14} {f'{other:7.1f}ms {otherp:5.1f}%':>13}")

    print("\nper input token (us/token, collectives excluded):")
    print(f"{'ctx':>4} {'KDA':>8} {'FullAttn':>9} {'MoE':>8} {'other':>8} {'total':>9} "
          f"| {'MLA attn kernel':>16} {'KDA chunk kernels':>18}")
    for r in rows:
        u = r["us_per_token"]
        print(f"{r['label']:>4} {u.get('kda',0):8.1f} {u.get('full_attn',0):9.1f} "
              f"{u.get('moe',0):8.1f} {u.get('other',0)+u.get('dense_mlp',0):8.1f} "
              f"{r['total_us_per_token']:9.1f} | "
              f"{r['mla_prefill_kernel_ms']:13.1f}ms {r['kda_chunk_kernels_ms']:15.1f}ms")

    print("\nper layer (ms per layer over the whole prefill, collectives excluded):")
    print(f"{'ctx':>4} {'one KDA layer':>15} {'one full-attn layer':>21} {'ratio FA:KDA':>13}")
    for r in rows:
        kda_l = r["groups_ms"]["kda"] / N_KDA
        fa_l = r["groups_ms"]["full_attn"] / N_FULL
        print(f"{r['label']:>4} {kda_l:13.3f}ms {fa_l:19.3f}ms {fa_l/kda_l:12.2f}x")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"rows": rows}, fh, indent=1)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
