"""Summarize evaluator receipts + ISA stats for every fetched arm. Writes results/summary.json and a CSV.

Speedup convention (FlyBench): ratio = stock_time / candidate_time, so < 1 means the arm is SLOWER than
the stock kernel. For ablation arms "candidate" = stock with one author decision removed.
"""

import csv
import json
from pathlib import Path

import isa_compare

HERE = Path(__file__).resolve().parent


def main():
    exps = json.loads((HERE / "experiments.json").read_text())
    ledger = json.loads((HERE / "jobs.json").read_text())
    rows, out = [], {}
    for e in exps:
        label = e["label"]
        base = HERE / "results" / label / "flybench-eval" / label
        receipt = base / "receipt.json"
        if not receipt.exists():
            continue
        r = json.loads(receipt.read_text())
        sp = r.get("speedup_vs_flydsl") or {}
        per = r.get("per_shape") or []
        shapes = []
        for s in per:
            fs = s.get("flydsl") or {}
            t = s.get("t_us") or {}
            shapes.append({"shape": s.get("shape"), "ratio": fs.get("ratio"), "ci95": fs.get("ci95"),
                           "stock_us": t.get("flydsl"), "cand_us": t.get("candidate"), "drawn": s.get("drawn", False)})
        isa = {}
        tar = base / "zz-isa.tar.gz"
        if tar.exists():
            stats, rcs = isa_compare.load_arm(tar)
            for (side, shape), kernels in stats.items():
                for kernel, st in kernels.items():
                    isa.setdefault(shape, {}).setdefault(kernel, {})[side] = st
        ratios = [s["ratio"] for s in shapes if isinstance(s["ratio"], (int, float))]
        doc = {"label": label, "record": e["record"], "arm": e["arm"], "job": ledger.get(label, {}).get("job_id"),
               "verdict": r.get("verdict"), "correct": r.get("correct"), "improved": r.get("improved"),
               "ratio": sp.get("ratio"), "ci95": sp.get("ci95"), "pairs": sp.get("pairs"),
               "noise_floor": r.get("noise_floor"), "regressions": r.get("regressions"),
               "worst_shape": r.get("worst_shape"), "min_shape_ratio": min(ratios) if ratios else None,
               "max_shape_ratio": max(ratios) if ratios else None, "per_shape": shapes, "isa": isa,
               "reason": r.get("reason")}
        out[label] = doc
        rows.append([label, e["record"], e["arm"], doc["verdict"], doc["correct"],
                     f"{doc['ratio']:.4f}" if doc["ratio"] else "", 
                     f"{doc['ci95'][0]:.4f}" if doc["ci95"] else "", f"{doc['ci95'][1]:.4f}" if doc["ci95"] else "",
                     doc["pairs"], doc["noise_floor"], len(doc["regressions"] or []),
                     f"{doc['min_shape_ratio']:.3f}" if doc["min_shape_ratio"] else "",
                     f"{doc['max_shape_ratio']:.3f}" if doc["max_shape_ratio"] else ""])
    (HERE / "results" / "summary.json").write_text(json.dumps(out, indent=1))
    with open(HERE / "results" / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "record", "arm", "verdict", "correct", "speedup", "ci95_lo", "ci95_hi", "pairs",
                    "noise_floor", "regressions", "min_shape_speedup", "max_shape_speedup"])
        w.writerows(rows)
    for row in rows:
        print("  ".join(str(x) for x in row))


if __name__ == "__main__":
    main()
