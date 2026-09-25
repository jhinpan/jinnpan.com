#!/usr/bin/env python3
"""Put the GB300 nsys attribution (per verify cycle / decode step, TP0) and the MI355X
BS=1 cycle ledger into one category scheme.

GB300 reports two views per category: wall_us (the category's share of the cycle, with
overlapping intervals split evenly) and kernel_us (summed kernel durations, which exceed the
cycle when streams overlap). The MI355X graph runs its kernels back to back, so its ledger
(unprofiled phase totals, profiled per-family shares) is both views at once.

GB300 fuses the MoE finalize into the all-reduce kernel; MI355X fuses the mHC post-mix into
it instead and runs the finalize separately. Both are therefore reported in one
"all-reduce + MoE finalize" bucket.
"""
import csv
import json
from pathlib import Path

CAMP = Path(__file__).resolve().parents[1]
FLOOR = Path("$BS1_FLOOR/analysis")

MI_FAMILY = {
    "moe": ["MoE G1", "MoE G2", "MoE sorting", "router gate", "SiLU/clamp", "MoE quant/sort", "router GEMV"],
    "dense_gemm": ["mxfp8 GEMV", "hipBLASLt/Tensile GEMM", "WO-A partial", "WO-A reduce", "RMSNorm+fake quant"],
    "attention": ["PA main", "PA split reduce + inverse RoPE", "KV norm/RoPE/store", "indexer logits",
                  "metadata / attention glue (Triton)"],
    "mhc": ["mHC RMSNorm+Sinkhorn", "mHC boundary partial", "mHC combine"],
    "all_reduce": ["fused AR + mHC post", "RCCL collective", "other custom collective",
                   "MoE topk reduce + shared add"],
    "engram": ["engram"],
    "other": ["PyTorch elementwise/index/copy", "other", "runtime copyBuffer", "embedding"],
}
ORDER = ["moe", "dense_gemm", "attention", "mhc", "all_reduce", "engram", "draft", "other", "host_gap"]
LABEL = {"moe": "MoE experts + routing", "dense_gemm": "dense GEMM + act. quant",
         "attention": "attention / indexer / KV", "mhc": "mHC", "all_reduce": "all-reduce + MoE finalize",
         "engram": "Engram", "draft": "DSpark draft", "other": "other glue", "host_gap": "GPU idle"}
# From budget.md: the GPU-idle segments of the unprofiled cycle before kevin-mii/sglang#12.
MI_IDLE_US = 647.0 + 40.0


def gb300(name):
    d = json.loads((CAMP / f"gb300/results/{name}/attribution.json").read_text())
    cats = {c["category"]: c for c in d["categories"]}
    out = {}
    for k in ORDER[:-1]:
        c = cats.get(k, {"wall_us": 0.0, "kernel_us": 0.0, "kernels": 0.0})
        out[k] = dict(wall_us=c["wall_us"], kernel_us=c["kernel_us"], kernels=c["kernels"])
    mem = cats.get("memcpy", {"wall_us": 0.0, "kernel_us": 0.0, "kernels": 0.0})
    out["other"] = {k: out["other"][k] + mem[k] for k in ("wall_us", "kernel_us", "kernels")}
    out["host_gap"] = dict(wall_us=d["host_gap_us"], kernel_us=0.0, kernels=0.0)
    kernel_sum = sum(v["kernel_us"] for v in out.values())
    busy = d["cycle_us_median"] - d["host_gap_us"]
    return dict(cycle_us_median=d["cycle_us_median"], cycle_us_mean=d["cycle_us_mean"], cycles=d["cycles"],
                categories=out, kernel_sum_us=kernel_sum, busy_us=busy,
                kernels=sum(v["kernels"] for v in out.values()), overlap=kernel_sum / busy)


def mi355x():
    rows = list(csv.DictReader((FLOOR / "kernel-ledger.csv").open()))
    fam_cat = {f: c for c, fams in MI_FAMILY.items() for f in fams}
    out = {k: dict(wall_us=0.0, kernel_us=0.0, kernels=0.0) for k in ORDER}
    unmapped = []
    for r in rows:
        us, calls = float(r["est_unprofiled_us"]), float(r["calls_per_cycle"])
        cat = "draft" if r["phase"] == "DRAFT" else fam_cat.get(r["family"])
        if cat is None:
            unmapped.append(r["family"])
            cat = "other"
        out[cat]["wall_us"] += us
        out[cat]["kernel_us"] += us
        out[cat]["kernels"] += calls
    out["host_gap"] = dict(wall_us=MI_IDLE_US, kernel_us=0.0, kernels=0.0)
    total = sum(v["wall_us"] for v in out.values())
    kernel_sum = sum(v["kernel_us"] for v in out.values())
    return dict(cycle_us=total, categories=out, kernel_sum_us=kernel_sum, busy_us=kernel_sum,
                kernels=sum(v["kernels"] for v in out.values()), overlap=1.0, unmapped=unmapped,
                source="dsv41-bs1-floor-20260922 kernel ledger (prior base, genuine acceptance, "
                       "unprofiled phase totals with profiled per-family shares)")


def main():
    res = dict(order=ORDER, labels=LABEL, gb300_verify=gb300("p2-real"), gb300_decode=gb300("p2-off"),
               mi355x_verify=mi355x(), mi_family_map=MI_FAMILY)
    (CAMP / "analysis/attribution_compare.json").write_text(json.dumps(res, indent=2) + "\n")
    g, m = res["gb300_verify"], res["mi355x_verify"]
    print(f"{'category':26} {'MI355X':>8} {'GB300 wall':>10} {'GB300 kern':>10} {'MI k':>6} {'GB k':>6}")
    for k in ORDER:
        a, b = m["categories"][k], g["categories"][k]
        print(f"{LABEL[k]:26} {a['wall_us']:8.0f} {b['wall_us']:10.0f} {b['kernel_us']:10.0f} "
              f"{a['kernels']:6.0f} {b['kernels']:6.0f}")
    print(f"{'total':26} {m['cycle_us']:8.0f} {g['cycle_us_median']:10.0f} {g['kernel_sum_us']:10.0f} "
          f"{m['kernels']:6.0f} {g['kernels']:6.0f}")
    print(f"GB300 overlap = kernel sum / busy = {g['overlap']:.2f}; decode step overlap = "
          f"{res['gb300_decode']['overlap']:.2f}; unmapped MI families: {m['unmapped']}")


if __name__ == "__main__":
    main()
