#!/usr/bin/env python3
"""Structure of the current-base MI355X BS=1 cycle, from a rocprofv3 kernel trace of TP0.

rocprofv3 1.1.0 aborts the first replay of a packet-captured HIP graph, so the traced server
runs with DEBUG_CLR_GRAPH_PACKET_CAPTURE=0. Under that trace small graph kernels come out near
5 us and collectives absorb the rank skew that tracing adds, so durations here are not a
ledger; kernel counts and kernel identities per cycle are what the trace settles.
A cycle runs from one verify step's first attention main pass (partial_gluon, 40 per step)
to the next step's.
"""
import argparse
import collections
import csv
import json
import statistics as st
from pathlib import Path

TP0_LOCATION = 0x8500  # PCI bus 0x85 = HIP 4, the first GPU of the TP4 group
ANCHOR = "partial_gluon"
WATCH = {"attention main (partial_gluon)": "partial_gluon", "attention combine (_combine)": "_combine",
         "fused all-reduce + mHC post": "dsv41_allreduce_mhc_post", "MoE G1": "mfma_moe1_", "MoE G2": "mfma_moe2_",
         "MoE sorting": "moe_sorting_entry", "router gate": "_router_gate_kernel",
         "RMSNorm + fake quant": "_rmsnorm_fake_quant"}


def tp0_trace(trace_dir):
    for info in sorted(trace_dir.rglob("*_agent_info.csv")):
        agents = {r["Node_Id"]: int(r["Location_Id"]) for r in csv.DictReader(info.open()) if r["Agent_Type"] == "GPU"}
        kt = info.with_name(info.name.replace("_agent_info.csv", "_kernel_trace.csv"))
        if not kt.exists():
            continue
        with kt.open() as f:
            first = next(csv.DictReader(f), None)
        if first and agents.get(first["Agent_Id"].removeprefix("Agent ")) == TP0_LOCATION:
            return kt
    raise SystemExit(f"no TP0 kernel trace under {trace_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("trace_dir", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cycles", type=int, default=100, help="steady-state cycles at the end of the last request")
    a = p.parse_args()
    kt = tp0_trace(a.trace_dir)
    k = sorted((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"]) for r in csv.DictReader(kt.open()))
    anchors = [i for i, (_, _, n) in enumerate(k) if n == ANCHOR]
    steps, cur = [], [anchors[0]]
    for i in anchors[1:]:
        if i - cur[-1] > 80:
            steps.append(cur)
            cur = [i]
        else:
            cur.append(i)
    steps.append(cur)
    starts = [s[0] for s in steps if len(s) == 40][-(a.cycles + 1):]
    per_cycle = [hi - lo for lo, hi in zip(starts, starts[1:])]
    window = k[starts[0]:starts[-1]]
    n = len(starts) - 1
    counts = collections.Counter(name for _, _, name in window)
    watched = {}
    for label, pat in WATCH.items():
        d = [(e - s) / 1e3 for s, e, name in window if pat in name]
        watched[label] = dict(calls_per_cycle=len(d) / n, traced_median_us=st.median(d) if d else None)
    small = [(e - s) / 1e3 for s, e, name in window]
    out = dict(trace=kt.name, cycles=n, kernels_per_cycle=st.median(per_cycle),
               distinct_kernels=len(counts), traced_duration_median_us=st.median(small),
               traced_share_between_4_and_6_us=sum(4 <= x <= 6 for x in small) / len(small), watched=watched,
               top_by_calls=[(name[:100], c / n) for name, c in counts.most_common(25)])
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({key: out[key] for key in ("cycles", "kernels_per_cycle", "distinct_kernels",
                                                 "traced_duration_median_us", "traced_share_between_4_and_6_us")}))
    for label, v in watched.items():
        print(f"  {label:32s} {v['calls_per_cycle']:6.1f}/cycle  traced median {v['traced_median_us']} us")


if __name__ == "__main__":
    main()
