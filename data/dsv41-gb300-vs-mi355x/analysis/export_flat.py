#!/usr/bin/env python3
"""Flatten the published Experiment 005 data (data/dsv41-gb300-vs-mi355x/) into analysis-ready CSVs.

Reads only the sanitized package, so everything it writes is already public. Run from the package:
    python3 analysis/export_flat.py --out /tmp/dsv41-flat
The output is what the gist holds: tidy CSVs, a README with the conclusions and a column guide, and
scoreboard.py, which recomputes the page's headline numbers from the CSVs alone.
"""
import argparse
import csv
import json
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = "https://jinnpan.com/sources/dsv41-gb300-vs-mi355x.html"
DATA = "https://github.com/jhinpan/jinnpan.com/tree/main/data/dsv41-gb300-vs-mi355x"


def write(path, header, rows):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return len(rows)


def num(x, n=4):
    return "" if x is None else round(x, n)


def rounds_rows(groups):
    out = []
    for plat in ("gb300", "mi355x"):
        for arm in sorted((ROOT / plat / "arms").iterdir()):
            m = re.fullmatch(r"(.+)-([a-z])", arm.name)
            if not m or (plat, m.group(1)) not in groups or not (arm / "rounds.csv").exists():
                continue
            for r in csv.DictReader((arm / "rounds.csv").open()):
                ttft, elapsed = float(r["ttft_s"]), float(r["elapsed_s"])
                verify = int(r["verify_steps"]) if r["verify_steps"] not in ("", "None") else None
                out_tok, first = int(r["output_tokens"]), int(r["first_event_tokens"])
                cycle = (elapsed - ttft) / verify * 1e3 if verify else None
                step = None if verify else (elapsed - ttft) / (out_tok - first) * 1e3
                al = float(r["accept_length"]) if r["accept_length"] not in ("", "None") else None
                out.append([plat, m.group(1), arm.name, m.group(2), int(r["repeat"]), r["warmup"] == "True",
                            num(float(r["output_tps"]), 3), num(al), verify or "", num(ttft * 1e3, 2),
                            num(elapsed, 5), num(cycle), num(step), out_tok, first, r["max_repeated_16gram"],
                            r["output_text_sha256"]])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    arms = json.loads((ROOT / "analysis/arms.json").read_text())
    att = json.loads((ROOT / "analysis/attribution_compare.json").read_text())
    rt = json.loads((ROOT / "analysis/realtext.json").read_text())
    place = json.loads((ROOT / "analysis/gb300_placement.json").read_text())
    trace = json.loads((ROOT / "analysis/mi355x_trace_real.json").read_text())
    mb = {"gb300": json.loads((ROOT / "gb300/arms/mb-cuda/default.json").read_text()),
          "mi355x": json.loads((ROOT / "mi355x/graph-floor/default.json").read_text())}
    groups = {(x["platform"], x["group"]) for x in arms}
    counts = {}

    counts["rounds.csv"] = write(a.out / "rounds.csv", [
        "platform", "group", "arm", "launch", "repeat", "warmup", "output_tps", "accept_length", "verify_steps",
        "ttft_ms", "elapsed_s", "cycle_ms", "step_ms", "output_tokens", "first_event_tokens",
        "max_repeated_16gram", "output_text_sha256"], rounds_rows(groups))

    counts["arms.csv"] = write(a.out / "arms.csv", [
        "platform", "group", "description", "launches", "rounds", "tps_median", "tps_min", "tps_max",
        "accept_length_median", "cycle_ms_median", "step_ms_median", "ttft_ms_median", "max_repeated_16gram_median",
        "per_launch_tps"], [[x["platform"], x["group"], x["desc"], x["launches"], x["rounds"], num(x["tps"], 2),
                             num(x["tps_min"], 2), num(x["tps_max"], 2), num(x["al"]), num(x["cycle_ms"]),
                             num(x["step_ms"]), num(x["ttft_ms"], 1), num(x["repeat16"], 1),
                             "|".join(f"{q['launch']}:{q['tps']:.1f}" for q in x["per_launch"])] for x in arms])

    mv, gv, gs = att["mi355x_verify"], att["gb300_verify"], att["gb300_verify_serial"]
    rows = []
    for k in att["order"]:
        m, d, s = mv["categories"][k], gv["categories"][k], gs["categories"][k]
        rows.append([k, att["labels"][k], num(m["wall_us"], 1), num(m["kernels"], 1), num(d["wall_us"], 1),
                     num(d["kernel_us"], 1), num(d["kernels"], 1), num(s["wall_us"], 1), num(s["kernels"], 1),
                     num(m["wall_us"] - s["wall_us"], 1)])
    counts["verify_cycle_categories.csv"] = write(a.out / "verify_cycle_categories.csv", [
        "category", "label", "mi355x_us", "mi355x_kernels", "gb300_default_wall_us", "gb300_default_kernel_us",
        "gb300_default_kernels", "gb300_serial_us", "gb300_serial_kernels", "gap_mi355x_minus_gb300_serial_us"], rows)

    gd, gds = att["gb300_decode"], att["gb300_decode_serial"]
    counts["decode_step_categories.csv"] = write(a.out / "decode_step_categories.csv", [
        "category", "label", "gb300_default_wall_us", "gb300_default_kernel_us", "gb300_serial_us", "gb300_kernels"],
        [[k, att["labels"][k], num(gd["categories"][k]["wall_us"], 1), num(gd["categories"][k]["kernel_us"], 1),
          num(gds["categories"][k]["wall_us"], 1), num(gds["categories"][k]["kernels"], 1)]
         for k in att["order"] if k != "draft"])

    rows = []
    for cap in ("p2-real", "p2-off", "p3-serial-real", "p3-serial-off"):
        d = json.loads((ROOT / f"gb300/arms/{cap}/attribution.json").read_text())
        rows += [[cap, x["category"], x["sub"], num(x["kernel_us"], 2), num(x["wall_us"], 2), num(x["kernels"], 2)]
                 for x in d["subcategories"]]
    counts["gb300_subcategories.csv"] = write(a.out / "gb300_subcategories.csv", [
        "capture", "category", "sub", "kernel_us_per_cycle", "wall_us_per_cycle", "kernels_per_cycle"], rows)

    rows = []
    for plat, d in mb.items():
        for case, v in d["cases"].items():
            for n, r in v["rows"].items():
                rows.append([plat, d["device"], case, n, num(r["device_median_us"], 3), num(r["host_median_us"], 3),
                             num(v["fit"]["per_kernel_us"], 4) if v.get("fit") else "",
                             num(v["fit"]["intercept_us"], 3) if v.get("fit") else ""])
    counts["graph_floor.csv"] = write(a.out / "graph_floor.csv", [
        "platform", "device", "case", "size", "device_median_us_per_replay", "host_median_us_per_replay",
        "fit_per_kernel_us", "fit_intercept_us"], rows)

    rows = [[plat, r["launch"], r["index"], num(r["accepted_length"]), num(r["wall_per_verify_proxy_ms"]),
             num(r["stream_decode_tps"], 2), num(r["request_output_tps"], 2), num(r["server_ttft_ms"], 2),
             r["output_sha256"]] for plat in ("gb300", "mi355x") for r in rt["rows"][plat]]
    counts["realtext_requests.csv"] = write(a.out / "realtext_requests.csv", [
        "platform", "launch", "prompt_index", "accepted_length", "wall_per_verify_proxy_ms", "stream_decode_tps",
        "request_output_tps", "server_ttft_ms", "output_sha256"], rows)

    rows = []
    for key, ranks in place.items():
        arm, snap = key.split("/")
        rows += [["gb300", arm, snap, r["tp"], r["cpus"], r.get("policy", ""), r["local_thread_frac"],
                  r["local_page_frac"]] for r in ranks]
    for arm in sorted((ROOT / "mi355x/arms").glob("*/processes.json")):
        for pr in json.loads(arm.read_text()):
            m = re.search(r"scheduler_TP(\d)", pr["cmdline"])
            if m:
                pages = pr["numa_pages"]
                rows.append(["mi355x", arm.parent.name, "after_ready", int(m.group(1)), pr["cpus_allowed"], "", "",
                             round(pages.get("N1", 0) / max(1, sum(pages.values())), 3)])
    counts["placement.csv"] = write(a.out / "placement.csv", [
        "platform", "arm", "snapshot", "tp", "cpus_allowed", "policy", "local_thread_frac", "local_page_frac"], rows)

    # Headline numbers for the README, from the same inputs.
    A = {(x["platform"], x["group"]): x for x in arms}
    gb_c, mi_c = A[("gb300", "n1a-sim")]["cycle_ms"], A[("mi355x", "u-sim")]["cycle_ms"]
    gb_s, mi_s = A[("gb300", "n1a-off")]["step_ms"], A[("mi355x", "u-off")]["step_ms"]
    w = {v: A[("gb300", f"w3-{v}-sim")]["cycle_ms"] for v in ("base", "opt0", "serial")}
    ws = {v: A[("gb300", f"w3-{v}-off")]["step_ms"] for v in ("base", "opt0", "serial")}
    gap = mi_c - gb_c
    conc = w["serial"] - w["base"]
    gaps = {k: mv["categories"][k]["wall_us"] - gs["categories"][k]["wall_us"] for k in att["order"]}
    floor = {p: mb[p]["cases"]["triton_chain_p1"]["fit"]["per_kernel_us"] for p in mb}
    ladder = {p: mb[p]["cases"]["fork_join_ladder_p1"]["rows"]["128"]["device_median_us"] / 384 for p in mb}
    rg, rm = rt["gb300"], rt["mi355x"]
    ranked = sorted((k for k in att["order"] if k not in ("host_gap", "engram")), key=gaps.get, reverse=True)
    files = "\n".join(f"| `{f}` | {n} |" for f, n in counts.items())
    readme = f"""# Experiment 005 · DeepSeek-V4.1 Flash at BS=1, GB300 vs MI355X: flat results

Analysis-ready CSVs behind [the write-up]({PAGE}). Every file is derived from the sanitized raw data in
[`data/dsv41-gb300-vs-mi355x/`]({DATA}) by `analysis/export_flat.py`; `scoreboard.py` below recomputes the
headline numbers from these CSVs alone.

## Conclusions

1. **MI355X is about 2x slower per step at BS=1, and faster at prefill.** Plain decode {mi_s:.2f} vs {gb_s:.2f} ms
   per step ({mi_s / gb_s:.2f}x); DSpark verify cycle {mi_c:.2f} vs {gb_c:.2f} ms ({mi_c / gb_c:.2f}x). The 4,096-token
   prefill is {A[('gb300', 'n1a-off')]['ttft_ms'] / A[('mi355x', 'u-off')]['ttft_ms']:.2f}x faster on MI355X
   ({A[('mi355x', 'u-off')]['ttft_ms']:.0f} vs {A[('gb300', 'n1a-off')]['ttft_ms']:.0f} ms). On real text both machines
   accept the same number of tokens per verify ({rg['accepted_length_median']:.2f} vs {rm['accepted_length_median']:.2f}),
   so the gap is the cost per verify.
2. **The biggest gap is stream concurrency, {conc:.2f} ms of the {gap:.2f} ms per verify ({conc / gap:.0%}).** Turning
   off GB300's side streams raises its verify cycle {w['base']:.2f} -> {w['opt0']:.2f} ms
   (`SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=0`: attention preparation, mHC statistics, routed quantization, draft)
   -> {w['serial']:.2f} ms (shared experts also on the forward stream). Plain decode: {ws['base']:.2f} -> {ws['opt0']:.2f}
   -> {ws['serial']:.2f} ms. This is what the overlapped layout is worth as SGLang implements it, a little more than
   overlap alone: without side streams SGLang also launches {gs['kernels'] - gv['kernels']:.0f} more kernels per verify.
   MI355X cannot take this win today: in a HIP graph a kernel behind a fork/join costs
   {ladder['mi355x']:.1f} us vs {ladder['gb300']:.2f} us in a CUDA graph (`graph_floor.csv`, same script on both).
3. **With both machines serial, {mi_c - w['serial']:.2f} ms per verify remains**, mostly kernel time: """ + ", ".join(
        f"{att['labels'][k]} {gaps[k]:+,.0f} us" for k in ranked) + f"""; GPU idle {gaps['host_gap']:+,.0f} us.
   MI355X is ahead on mHC. Behind many rows: a dependent kernel costs {floor['mi355x']:.2f} us in a HIP graph vs
   {floor['gb300']:.2f} us in a CUDA graph, and a verify cycle with its draft runs {trace['kernels_per_cycle']:,.0f}
   kernels on MI355X today vs {gv["kernels"]:,.0f} on GB300.
4. **Where to start on MI355X:** (a) concurrency: ROCm graph branches as cheap as CUDA's, and until then shrink the
   side-stream work (fusion; A/B MoE TP4 so the shared expert can fuse); (b) kernel count, worth
   {floor['mi355x'] / floor['gb300']:.1f}x more per kernel than on GB300; (c) MoE and attention kernel time;
   (d) host idle (kevin-mii/sglang#12: 9.15 -> 8.63 ms).
5. **Measurement traps, not machine properties:** the published GB300 target was two weeks stale; a 30% GB300
   launch-to-launch swing was CPU placement (Docker without SYS_NICE); natural acceptance on a random prompt
   measures repetition loops, not the model.

Caveats: the MI355X per-category ledger is from the previous base ({mv['cycle_us'] / 1e3:.2f} ms per cycle vs
{mi_c:.2f} ms today; kineto shares scaled to unprofiled phase totals). A current-base rocprofv3 trace was
attempted but its small-kernel durations sit on a ~5 us floor, so only its kernel counts are used.

## Files

| file | rows |
|---|---:|
{files}

- `rounds.csv`: one row per client round (warm-ups flagged), every arm of every measured group.
  `cycle_ms = (elapsed_s - ttft_s) / verify_steps` for DSpark arms (independent of acceptance length);
  `step_ms = (elapsed_s - ttft_s) / (output_tokens - first_event_tokens)` for plain decode. `output_tps` is
  BBuf's metric. Groups: `s0-*`/`n0-*` author's commit 835c3909 (unbound / NUMA-bound); `s1a-*`/`n1a-*` main
  ffac53d779; `e1-*` NUMA control; `w3-*` session 3 stream A/B (`base`, `opt0`, `serial`); MI355X `u-*`/`n-*`/`s2-*`
  e2e824dc58+#8 placement modes. Modes: `off` plain decode, `sim` simulated acceptance 5.5, `real` natural.
- `arms.csv`: one row per group, medians over pooled rounds, per-launch medians in `per_launch_tps`.
- `verify_cycle_categories.csv`: us per DSpark verify cycle by category. MI355X: previous-base ledger.
  GB300 default layout: `wall` splits overlapping kernels evenly, `kernel` sums durations. GB300 serial:
  every side stream off (standalone cost). `gap_*` = MI355X minus GB300 serial.
- `decode_step_categories.csv`: the same for GB300's plain decode step (default and serial).
- `gb300_subcategories.csv`: nsys attribution sub-buckets per capture (TP0; p2 = default, p3 = serial).
- `graph_floor.csv`: `graph_floor.py` on one GPU per machine; device us per graph replay, and fitted us per
  kernel for the chain cases. `fork_join_ladder_p1` has 3 kernels per rung.
- `realtext_requests.csv`: the real-text BS=1 contract, one row per scored request (four 4,096-token prompts,
  24 samples each per server; prompts not published, outputs as sha256).
- `placement.csv`: where each TP scheduler ran (GB300 from /proc thread CPUs and numa_maps; MI355X allowed CPUs
  and the share of resident pages on node 1, the GPUs' node).

## Quick start

```python
import pandas as pd
rounds = pd.read_csv("rounds.csv")
timed = rounds[~rounds.warmup]
print(timed.groupby(["platform", "group"])[["cycle_ms", "step_ms", "output_tps"]].median())
```

`python3 scoreboard.py` prints the headline numbers from the CSVs.
"""
    (a.out / "README.md").write_text(readme)
    (a.out / "scoreboard.py").write_text(SCOREBOARD)
    print(json.dumps(counts))


SCOREBOARD = '''#!/usr/bin/env python3
"""Recompute Experiment 005's headline numbers from the flat CSVs (standard library only)."""
import csv
import statistics as st
from collections import defaultdict


def load(name):
    with open(name, newline="") as f:
        return list(csv.DictReader(f))


cyc, step = defaultdict(list), defaultdict(list)
for r in load("rounds.csv"):
    if r["warmup"] == "True":
        continue
    key = (r["platform"], r["group"])
    if r["cycle_ms"]:
        cyc[key].append(float(r["cycle_ms"]))
    if r["step_ms"]:
        step[key].append(float(r["step_ms"]))
m = lambda d, p, g: st.median(d[(p, g)])
gb_c, mi_c = m(cyc, "gb300", "n1a-sim"), m(cyc, "mi355x", "u-sim")
gb_s, mi_s = m(step, "gb300", "n1a-off"), m(step, "mi355x", "u-off")
w = {v: m(cyc, "gb300", f"w3-{v}-sim") for v in ("base", "opt0", "serial")}
print(f"verify cycle   GB300 {gb_c:.2f} ms  MI355X {mi_c:.2f} ms  ratio {mi_c / gb_c:.2f}x")
print(f"decode step    GB300 {gb_s:.2f} ms  MI355X {mi_s:.2f} ms  ratio {mi_s / gb_s:.2f}x")
print(f"GB300 streams  default {w['base']:.2f}  opt0 {w['opt0']:.2f}  serial {w['serial']:.2f} ms per verify")
print(f"concurrency    {w['serial'] - w['base']:.2f} ms of the {mi_c - gb_c:.2f} ms gap; "
      f"{mi_c - w['serial']:.2f} ms remains with both serial")
for r in load("verify_cycle_categories.csv"):
    print(f"  {r['label']:28s} MI355X {float(r['mi355x_us']):7.0f}  GB300 serial {float(r['gb300_serial_us']):7.0f}"
          f"  gap {float(r['gap_mi355x_minus_gb300_serial_us']):+7.0f} us")
al = defaultdict(list)
for r in load("realtext_requests.csv"):
    al[r["platform"]].append(float(r["accepted_length"]))
print("real-text accepted length", {p: round(st.median(v), 3) for p, v in al.items()})
fl = {}
for r in load("graph_floor.csv"):
    if r["case"] == "triton_chain_p1" and r["fit_per_kernel_us"]:
        fl[r["platform"]] = float(r["fit_per_kernel_us"])
print("graph floor per dependent kernel (us)", {p: round(v, 2) for p, v in fl.items()})
'''


if __name__ == "__main__":
    main()
