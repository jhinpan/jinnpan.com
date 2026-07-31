#!/usr/bin/env python3
"""Bucket TokenSpeed decode/prefill chrome traces into per-block kernel time.

Mirrors the methodology of the SGLang kda_prof baseline on this box so the two
are directly comparable: sum device-kernel time per rank, divide by the number
of profiled steps, and attribute each kernel to a K3 block (KDA linear
attention / full MLA attention / MoE / attention residual / other) by name.

Attribution is by kernel name because TokenSpeed's Proton semantic scopes are
unavailable on ROCm (rocprofiler_force_configure cannot attach to a live
server). The SGLang side used a name map for the same reason on its
graph-off traces.
"""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import re
from pathlib import Path

# Ordered: first match wins. Patterns are deliberately explicit so an
# unrecognised kernel shows up as "unclassified" rather than being silently
# folded into "other".
BLOCK_PATTERNS: list[tuple[str, str]] = [
    # --- KDA / linear attention (gated delta rule, short conv, recurrent state)
    ("kda", r"kda|gated_delta|delta_rule|chunk_gla|recurrent|causal_conv1d|l2norm|gate_chunk_cumsum"),
    # --- full attention / MLA
    ("full_attn", r"mla|_fwd_kernel|_fwd_grouped|stage1|stage2|kv_scan|split_reduce|flash|attn_fwd|paged_attention"),
    # --- MoE (routed + shared experts, routing, sorting, quant, situ activation)
    ("moe", r"moe|expert|situ|topk|route|sort|mxfp4|a8w4|afp8|wfp4|swiglu|silu_mul|gluon"),
    # --- attention residual / score+combine epilogue
    ("attn_residual", r"_score_kernel|_combine_kernel|attn_res"),
    # --- collectives
    ("collective", r"all_?reduce|all_?gather|all_?to_?all|nccl|rccl|iris|reduce_scatter|quickreduce|cross_device"),
    # --- generic dense GEMM (projections, lm_head) and norms/elementwise
    ("gemm", r"^Cijk_|hgemm|gemm|matmul|^mm_|cutlass|hipblas|rocblas|tensile"),
    ("norm_elem", r"rmsnorm|layernorm|norm|elementwise|vectorized|cast|copy|memcpy|memset|index_|embedding|sample|argmax|softmax|cat_|concat|transpose|permute|fill"),
]


def classify(name: str) -> str:
    for block, pattern in BLOCK_PATTERNS:
        if re.search(pattern, name, re.IGNORECASE):
            return block
    return "unclassified"


def load_events(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        trace = json.load(fh)
    events = trace.get("traceEvents", trace if isinstance(trace, list) else [])
    return [
        e
        for e in events
        if e.get("ph") == "X"
        and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
        and "dur" in e
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+", help="per-rank *.trace.json.gz files")
    ap.add_argument("--steps", type=int, required=True, help="profiled steps in this trace")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--label", default="tokenspeed")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    per_rank: dict[int, dict[str, tuple[int, float]]] = {}
    for i, raw in enumerate(args.traces):
        path = Path(raw)
        m = re.search(r"TP(\d+)", path.name)
        rank = int(m.group(1)) if m else i
        agg: dict[str, list] = collections.defaultdict(lambda: [0, 0.0])
        for e in load_events(path):
            entry = agg[e["name"]]
            entry[0] += 1
            entry[1] += float(e["dur"])
        per_rank[rank] = {k: (v[0], v[1]) for k, v in agg.items()}
        print(f"rank {rank}: {len(agg)} distinct kernels, "
              f"{sum(v[1] for v in agg.values())/1000:.1f} ms device time")

    # Per-rank kernel table
    rows = []
    for rank, agg in sorted(per_rank.items()):
        for name, (count, total) in agg.items():
            rows.append((rank, classify(name), count, count / args.steps,
                         f"{total:.1f}", f"{total/args.steps:.1f}", name))
    rows.sort(key=lambda r: (r[0], -float(r[5])))
    with (out / "kernels-by-rank.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(("rank", "block", "count", "count_per_step", "total_us",
                    "us_per_step", "kernel"))
        w.writerows(rows)

    # Block rollup, per rank then averaged across ranks
    block_rank: dict[int, dict[str, float]] = collections.defaultdict(
        lambda: collections.defaultdict(float))
    for rank, agg in per_rank.items():
        for name, (_count, total) in agg.items():
            block_rank[rank][classify(name)] += total

    ranks = sorted(block_rank)
    blocks = sorted({b for r in ranks for b in block_rank[r]})
    with (out / "block-composition.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(("block", "mean_us_per_step", "mean_ms_per_step", "pct_of_device",
                    "min_rank_us_per_step", "max_rank_us_per_step"))
        totals = {
            b: [block_rank[r][b] / args.steps for r in ranks] for b in blocks
        }
        grand = sum(sum(v) / len(ranks) for v in totals.values())
        for b in sorted(blocks, key=lambda b: -sum(totals[b]) / len(ranks)):
            mean = sum(totals[b]) / len(ranks)
            w.writerow((b, f"{mean:.1f}", f"{mean/1000:.4f}",
                        f"{100*mean/grand:.2f}" if grand else "0",
                        f"{min(totals[b]):.1f}", f"{max(totals[b]):.1f}"))
        w.writerow(("TOTAL", f"{grand:.1f}", f"{grand/1000:.4f}", "100.00", "", ""))

    # Cross-rank kernel rollup (mean over ranks), the headline table
    names = {n for agg in per_rank.values() for n in agg}
    cross = []
    for name in names:
        totals = [per_rank[r].get(name, (0, 0.0))[1] / args.steps for r in ranks]
        counts = [per_rank[r].get(name, (0, 0.0))[0] / args.steps for r in ranks]
        cross.append((classify(name), sum(totals) / len(ranks),
                      sum(counts) / len(ranks), max(totals), name))
    cross.sort(key=lambda r: -r[1])
    with (out / "kernels-cross-rank.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(("block", "mean_us_per_step", "dispatches_per_step",
                    "slowest_rank_us_per_step", "kernel"))
        for block, mean, cnt, mx, name in cross:
            w.writerow((block, f"{mean:.1f}", f"{cnt:.1f}", f"{mx:.1f}", name))

    print(f"\nwrote {out}/block-composition.csv, kernels-cross-rank.csv, "
          f"kernels-by-rank.csv")
    print(f"\n=== {args.label}: block composition (mean over {len(ranks)} ranks, "
          f"{args.steps} steps) ===")
    print((out / "block-composition.csv").read_text())


if __name__ == "__main__":
    main()
