#!/usr/bin/env python3
"""Attribute a GPU-only decode chrome trace to K3 block types by kernel name.

Uses the name -> block weights that the instrumented (record_function) run
established. Two invariants are checked rather than assumed, because they are what
makes the transfer valid: every mapped name must appear a whole number of times per
decode step, and the structural counts must come out at 69 KDA / 24 full-attention /
92 MoE layers per pass. A violation means the map does not describe this trace.

Collective kernels are reported separately and excluded from the composition
percentages: in eager mode the TP8 all-reduce busy-waits on rank skew, so its
duration measures host launch jitter, not communication cost.

  ./chrome_bucketize.py traces/ctx64k --name-map results/name_map_4k.json
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import os
import sys

from bucketize import classify_kernel

KDA_RECURRENT = "fused_recurrent_kda_packed_decode_kernel"
N_KDA_LAYERS = 69
N_FULL_ATTN = 24
N_MOE = 92

BUCKET_ORDER = ["K3/kda", "K3/full_attn", "K3/moe", "K3/dense_mlp", "other"]


def load_name_map(path: str) -> dict[str, dict[str, float]]:
    with open(path) as fh:
        rows = json.load(fh)
    out = {}
    for r in rows:
        total = sum(r["split"].values())
        if total:
            out[r["name"]] = {b: n / total for b, n in r["split"].items()}
    return out


def kernel_totals(path: str) -> tuple[dict[str, list], float]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        trace = json.load(fh)
    per_name: dict[str, list] = collections.defaultdict(lambda: [0, 0.0])
    total_us = 0.0
    for e in trace["traceEvents"]:
        if e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        d = float(e.get("dur", 0.0))
        total_us += d
        rec = per_name[e["name"]]
        rec[0] += 1
        rec[1] += d
    return per_name, total_us


def analyse(path: str, name_map: dict, isl: int | None) -> dict:
    per_name, total_us = kernel_totals(path)

    n_recurrent = per_name.get(KDA_RECURRENT, [0, 0.0])[0]
    if n_recurrent == 0 or n_recurrent % N_KDA_LAYERS:
        raise SystemExit(
            f"{os.path.basename(path)}: {n_recurrent} {KDA_RECURRENT} dispatches is "
            f"not a multiple of {N_KDA_LAYERS} -- cannot infer decode steps"
        )
    steps = n_recurrent // N_KDA_LAYERS

    buckets = collections.defaultdict(float)
    by_class = collections.defaultdict(float)
    bucket_class = collections.defaultdict(lambda: collections.defaultdict(float))
    unmapped, nonintegral = [], []

    for name, (n, us) in per_name.items():
        cls = classify_kernel(name)
        by_class[cls] += us
        weights = name_map.get(name)
        if weights is None:
            unmapped.append({"name": name, "n": n, "us": us})
            weights = {"other": 1.0}
        elif n % steps:
            nonintegral.append({"name": name, "n": n, "per_step": n / steps})
        for b, w in weights.items():
            buckets[b] += us * w
            bucket_class[b][cls] += us * w

    collective_us = by_class.get("collective", 0.0)
    compute_us = total_us - collective_us
    compute_buckets = {
        b: buckets[b] - bucket_class[b].get("collective", 0.0) for b in buckets
    }
    compute_total = sum(compute_buckets.values())

    return {
        "file": os.path.basename(path),
        "isl": isl,
        "decode_steps": steps,
        "n_distinct_kernels": len(per_name),
        "dispatches_per_step": sum(v[0] for v in per_name.values()) / steps,
        "structural_check": {
            "kda_layers": n_recurrent / steps,
            "full_attn_layers": per_name.get("_fwd_grouped_kernel_stage1", [0])[0] / steps,
            "moe_layers": per_name.get(
                "mfma_moe2_afp8_wfp4_bf16_cshuffle_t32x256x128_vscale_fix3_fp4opt_v1_pm1_acc0",
                [0],
            )[0] / steps,
        },
        "total_gpu_us_per_step": total_us / steps,
        "collective_us_per_step": collective_us / steps,
        "compute_us_per_step": compute_us / steps,
        "compute_buckets_us_per_step": {
            b: compute_buckets[b] / steps for b in compute_buckets
        },
        "compute_buckets_pct": {
            b: 100.0 * compute_buckets[b] / compute_total for b in compute_buckets
        },
        "class_us_per_step": {c: by_class[c] / steps for c in by_class},
        "top_kernels": [
            {"name": n, "per_step": v[0] / steps, "us_per_step": v[1] / steps,
             "cls": classify_kernel(n), "bucket": max(
                 name_map.get(n, {"other": 1.0}).items(), key=lambda kv: kv[1])[0]}
            for n, v in sorted(per_name.items(), key=lambda kv: -kv[1][1])[:25]
        ],
        "unmapped": sorted(unmapped, key=lambda d: -d["us"])[:12],
        "nonintegral": nonintegral[:12],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dir")
    ap.add_argument("--name-map", required=True)
    ap.add_argument("--tp", type=int, default=0)
    ap.add_argument("--out")
    args = ap.parse_args()

    name_map = load_name_map(args.name_map)
    paths = sorted(glob.glob(os.path.join(args.trace_dir, f"*TP-{args.tp}*.trace.json.gz")))
    paths = [p for p in paths if "EXTEND" not in p]
    if not paths:
        print(f"no TP-{args.tp} trace in {args.trace_dir}", file=sys.stderr)
        return 1

    isl = None
    run_json = os.path.join(args.trace_dir, "run.json")
    itl = None
    if os.path.exists(run_json):
        with open(run_json) as fh:
            rj = json.load(fh)
        isl, itl = rj.get("isl"), rj.get("itl_median_ms")

    r = analyse(paths[0], name_map, isl)
    r["itl_median_ms_eager"] = itl

    print(f"=== {r['file']} (isl={isl}) ===")
    print(f"decode steps={r['decode_steps']}  dispatches/step={r['dispatches_per_step']:.0f}"
          f"  distinct kernels={r['n_distinct_kernels']}")
    sc = r["structural_check"]
    print(f"structural check: KDA layers/step={sc['kda_layers']:.2f} (expect {N_KDA_LAYERS}), "
          f"full-attn={sc['full_attn_layers']:.2f} (expect {N_FULL_ATTN}), "
          f"MoE={sc['moe_layers']:.2f} (expect {N_MOE})")
    print(f"GPU busy/step   {r['total_gpu_us_per_step']/1000:8.3f} ms")
    print(f"  collectives   {r['collective_us_per_step']/1000:8.3f} ms  "
          f"(eager skew-inflated; excluded below)")
    print(f"  compute       {r['compute_us_per_step']/1000:8.3f} ms")
    if itl:
        print(f"eager ITL      {itl:8.3f} ms/token")
    print("compute composition:")
    for b in BUCKET_ORDER:
        if b in r["compute_buckets_pct"]:
            print(f"  {r['compute_buckets_pct'][b]:6.2f}%  "
                  f"{r['compute_buckets_us_per_step'][b]/1000:8.3f} ms/step  {b}")
    if r["unmapped"]:
        print(f"unmapped ({len(r['unmapped'])}): "
              + ", ".join(f"{u['name'][:44]}[{u['n']}]" for u in r["unmapped"][:4]))
    if r["nonintegral"]:
        print(f"non-integral per-step counts: "
              + ", ".join(f"{u['name'][:34]}={u['per_step']:.2f}" for u in r["nonintegral"][:4]))

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(r, fh, indent=1)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
