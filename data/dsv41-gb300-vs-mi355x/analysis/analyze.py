#!/usr/bin/env python3
"""Recompute every BS=1 arm (GB300 and MI355X) from bench/measurements.json.

Per timed round (warm-up excluded):
  tps       = (completion_tokens - first_event_tokens) / (last_event - first_event)   [BBuf metric]
  cycle_ms  = (elapsed_s - ttft_s) / spec_verify_ct                                   [DSpark arms]
  step_ms   = (elapsed_s - ttft_s) / (output_tokens - first_event_tokens)             [plain decode]
cycle_ms is independent of the acceptance length, so it compares arms whose AL differs.
"""
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    # platform, group -> (description, arm-name prefix)
    ("gb300", "s0-tp4"): "BBuf 835c3909, MoE TP4 (EP1), sim 5.5, unbound",
    ("gb300", "s0-ep4"): "BBuf 835c3909, MoE EP4, sim 5.5, unbound",
    ("gb300", "s0-nods"): "BBuf 835c3909, EP4, DSpark off, unbound",
    ("gb300", "n0-tp4"): "BBuf 835c3909, MoE TP4 (EP1), sim 5.5, NUMA-bound",
    ("gb300", "n0-ep4"): "BBuf 835c3909, MoE EP4, sim 5.5, NUMA-bound",
    ("gb300", "n0-nods"): "BBuf 835c3909, EP4, DSpark off, NUMA-bound",
    ("gb300", "s1a-off"): "main ffac53d779, HT cell (DSpark off), unbound",
    ("gb300", "s1a-sim"): "main ffac53d779, LL cell, sim 5.5, unbound",
    ("gb300", "s1a-real"): "main ffac53d779, LL cell, real acceptance, unbound",
    ("gb300", "n1a-off"): "main ffac53d779, HT cell (DSpark off), NUMA-bound",
    ("gb300", "n1a-sim"): "main ffac53d779, LL cell, sim 5.5, NUMA-bound",
    ("gb300", "n1a-real"): "main ffac53d779, LL cell, real acceptance, NUMA-bound",
    ("gb300", "e1-base-sim"): "main ffac53d779, LL cell, sim 5.5, unbound (NUMA control)",
    ("gb300", "e1-nice-sim"): "main ffac53d779, LL cell, sim 5.5, NUMA-bound (NUMA control)",
    ("gb300", "w3-base-sim"): "main ffac53d779, LL cell, sim 5.5, NUMA-bound, default streams (session 3)",
    ("gb300", "w3-opt0-sim"): "main ffac53d779, LL cell, sim 5.5, NUMA-bound, SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=0",
    ("gb300", "w3-serial-sim"): "main ffac53d779, LL cell, sim 5.5, NUMA-bound, every side stream off",
    ("gb300", "w3-base-off"): "main ffac53d779, HT cell (DSpark off), NUMA-bound, default streams (session 3)",
    ("gb300", "w3-opt0-off"): "main ffac53d779, HT cell (DSpark off), NUMA-bound, SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=0",
    ("gb300", "w3-serial-off"): "main ffac53d779, HT cell (DSpark off), NUMA-bound, every side stream off",
    ("mi355x", "u-off"): "e2e824dc58+#8, HT cell, SGLang split affinity",
    ("mi355x", "u-real"): "e2e824dc58+#8, LL cell, real, SGLang split affinity",
    ("mi355x", "u-sim"): "e2e824dc58+#8, LL cell, sim 5.5, SGLang split affinity",
    ("mi355x", "n-off"): "e2e824dc58+#8, HT cell, split affinity + main on node 1",
    ("mi355x", "n-real"): "e2e824dc58+#8, LL cell, real, split affinity + main on node 1",
    ("mi355x", "n-sim"): "e2e824dc58+#8, LL cell, sim 5.5, split affinity + main on node 1",
    ("mi355x", "s2-split-sim"): "session 2, sim 5.5, SGLang split affinity",
    ("mi355x", "s2-node1-sim"): "session 2, sim 5.5, all schedulers on node 1",
    ("mi355x", "s2-none-sim"): "session 2, sim 5.5, no pinning",
    ("mi355x", "s2-split-real"): "session 2, real, SGLang split affinity",
    ("mi355x", "s2-node1-real"): "session 2, real, all schedulers on node 1",
    ("mi355x", "s2-none-real"): "session 2, real, no pinning",
    ("mi355x", "s2-node1-off"): "session 2, DSpark off, all schedulers on node 1",
    ("mi355x", "s2-none-off"): "session 2, DSpark off, no pinning",
}


def rounds(arm_dir):
    m = json.loads((arm_dir / "bench" / "measurements.json").read_text())
    out = []
    for r in m:
        if r.get("warmup"):
            continue
        span = r["elapsed_s"] - r["ttft_s"]
        row = dict(tps=r["output_tps"], ttft_ms=r["ttft_s"] * 1e3, al=r.get("accept_length"),
                   verify=r.get("verify_steps"), repeat16=r.get("max_repeated_16gram"),
                   out=r["output_tokens"])
        if row["verify"]:
            row["cycle_ms"] = span / row["verify"] * 1e3
        else:
            row["step_ms"] = span / (r["output_tokens"] - r["first_event_tokens"]) * 1e3
        out.append(row)
    return out


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def main():
    table = []
    for (platform, group), desc in GROUPS.items():
        base = ROOT / ("gb300/results" if platform == "gb300" else "results")
        arms = sorted(p for p in base.glob(f"{group}-*") if (p / "bench" / "measurements.json").exists()
                      and p.name[len(group) + 1:].isalpha() and len(p.name) == len(group) + 2)
        if not arms:
            continue
        per_launch, pooled = [], []
        for a in arms:
            rs = rounds(a)
            pooled += rs
            per_launch.append(dict(launch=a.name[-1], tps=med(r["tps"] for r in rs),
                                   al=med(r["al"] for r in rs),
                                   cycle_ms=med(r.get("cycle_ms") for r in rs),
                                   step_ms=med(r.get("step_ms") for r in rs)))
        table.append(dict(platform=platform, group=group, desc=desc, launches=len(arms), rounds=len(pooled),
                          tps=med(r["tps"] for r in pooled), tps_min=min(r["tps"] for r in pooled),
                          tps_max=max(r["tps"] for r in pooled), al=med(r["al"] for r in pooled),
                          al_min=min((r["al"] for r in pooled if r["al"]), default=None),
                          al_max=max((r["al"] for r in pooled if r["al"]), default=None),
                          cycle_ms=med(r.get("cycle_ms") for r in pooled),
                          step_ms=med(r.get("step_ms") for r in pooled),
                          ttft_ms=med(r["ttft_ms"] for r in pooled),
                          repeat16=med(r["repeat16"] for r in pooled),
                          per_launch=per_launch))
    out = ROOT / "analysis" / "arms.json"
    out.write_text(json.dumps(table, indent=2) + "\n")
    fmt = lambda x, n=1: "-" if x is None else f"{x:.{n}f}"
    print(f"{'platform':7} {'group':15} {'L':>2} {'n':>3} {'tps':>8} {'min':>7} {'max':>7} {'AL':>5} "
          f"{'cycle':>6} {'step':>5} {'ttft':>6}  per-launch tps")
    for t in table:
        print(f"{t['platform']:7} {t['group']:15} {t['launches']:>2} {t['rounds']:>3} {fmt(t['tps']):>8} "
              f"{fmt(t['tps_min']):>7} {fmt(t['tps_max']):>7} {fmt(t['al'], 2):>5} {fmt(t['cycle_ms'], 2):>6} "
              f"{fmt(t['step_ms'], 2):>5} {fmt(t['ttft_ms'], 0):>6}  "
              + " ".join(f"{p['launch']}:{fmt(p['tps'])}" for p in t["per_launch"]))


if __name__ == "__main__":
    sys.exit(main())
