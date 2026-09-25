#!/usr/bin/env python3
"""Build jinnpan.com/public/sources/dsv41-gb300-vs-mi355x.html (Experiment 005).

Every number on the page is read from the analysis outputs or the raw rounds; nothing is
typed by hand. The SVG plates are static markup whose coordinates are computed here.
Run after analyze.py, compare_attribution.py and package_data.py.
"""
import csv
import html
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import rounds  # noqa: E402

CAMP = Path(__file__).resolve().parents[1]
SITE = Path("$WORKSPACE/jinnpan.com")
OUT = SITE / "public/sources/dsv41-gb300-vs-mi355x.html"
SLUG = "dsv41-gb300-vs-mi355x"
GH = "https://github.com/sgl-project/sglang"
DATA = f"https://github.com/jhinpan/jinnpan.com/tree/main/data/{SLUG}"

ARMS = {(a["platform"], a["group"]): a for a in json.loads((CAMP / "analysis/arms.json").read_text())}
ATT = json.loads((CAMP / "analysis/attribution_compare.json").read_text())
PLACE = json.loads((CAMP / "analysis/gb300_placement.json").read_text())
BBUF = json.loads((CAMP / "gb300/bbuf/results.json").read_text())["summaries"]


def A(p, g):
    return ARMS[(p, g)]


def arm_rounds(platform, prefix):
    base = CAMP / ("gb300/results" if platform == "gb300" else "results")
    out = []
    for d in sorted(base.glob(f"{prefix}-*")):
        if len(d.name) == len(prefix) + 2 and (d / "bench/measurements.json").exists():
            for r in rounds(d):
                r["launch"] = d.name[-1]
                r["src"] = prefix
                out.append(r)
    return out


def f0(x):
    return f"{x:,.0f}"


def f1(x):
    return f"{x:,.1f}"


def f2(x):
    return f"{x:.2f}"


def pct(new, old):
    return (new / old - 1) * 100


def up1(x):
    return f"{math.ceil(x * 10) / 10:.1f}"


def sgn(x, n=1):
    if round(x, n) == 0:
        return f"{0:.{n}f}"
    return f"{x:+,.{n}f}".replace("-", "\u2212")


def gpu_power(arm):
    rows = list(csv.reader((CAMP / f"gb300/results/{arm}/gpu_samples.csv").open()))
    sm = [float(r[2].split()[0]) for r in rows if len(r) > 4 and r[1].strip() == "0" and r[2].strip()[:1].isdigit()]
    pw = [float(r[4].split()[0]) for r in rows if len(r) > 4 and r[1].strip() == "0" and r[4].strip()[:1].isdigit()]
    return st.median(sm), st.median(pw)


def locality(key):
    rows = PLACE[key]
    return {r["tp"]: (r["local_thread_frac"], r["local_page_frac"]) for r in rows}


# ---------------------------------------------------------------- numbers
BB_TP4, BB_EP4, BB_OFF = (BBUF[k]["output_tps_median"] for k in ("tp4_padding", "ep4_control", "no-dspark"))
s0 = {k: A("gb300", f"s0-{k}") for k in ("tp4", "ep4", "nods")}
n0 = {k: A("gb300", f"n0-{k}") for k in ("tp4", "ep4", "nods")}
s1 = {k: A("gb300", f"s1a-{k}") for k in ("off", "sim", "real")}
n1 = {k: A("gb300", f"n1a-{k}") for k in ("off", "sim", "real")}
e1b, e1n = A("gb300", "e1-base-sim"), A("gb300", "e1-nice-sim")
mu = {k: A("mi355x", f"u-{k}") for k in ("off", "sim", "real")}
mn = {k: A("mi355x", f"n-{k}") for k in ("off", "sim", "real")}
s2 = {(aff, mode): ARMS.get(("mi355x", f"s2-{aff}-{mode}")) for aff in ("split", "node1", "none")
      for mode in ("sim", "real", "off")}
GV, GD, MV = ATT["gb300_verify"], ATT["gb300_decode"], ATT["mi355x_verify"]
ORDER, LAB = ATT["order"], ATT["labels"]

gb_step, mi_step = n1["off"]["step_ms"], mu["off"]["step_ms"]
gb_cyc, mi_cyc = n1["sim"]["cycle_ms"], mu["sim"]["cycle_ms"]
old_step, old_cyc = n0["nods"]["step_ms"], n0["ep4"]["cycle_ms"]
pw_base = [gpu_power(f"e1-base-sim-{x}") for x in "abc"]
pw_nice = [gpu_power(f"e1-nice-sim-{x}") for x in "abc"]
loc_base = [locality(f"e1-base-sim-{x}/placement_post") for x in "abc"]
loc_nice = [locality(f"e1-nice-sim-{x}/placement_post") for x in "abc"]
tp23_thr = [l[tp][0] for l in loc_base for tp in (2, 3)]
tp23_pg = [l[tp][1] for l in loc_base for tp in (2, 3)]
nice_pg = [l[tp][1] for l in loc_nice for tp in range(4)]
gb_real_r = arm_rounds("gb300", "s1a-real") + arm_rounds("gb300", "n1a-real")
mi_real_r = arm_rounds("mi355x", "u-real") + arm_rounds("mi355x", "n-real")
for aff in ("split", "node1", "none"):
    mi_real_r += arm_rounds("mi355x", f"s2-{aff}-real")
gb_al = [r["al"] for r in gb_real_r]
mi_al = [r["al"] for r in mi_real_r]
gb_rep = [r["repeat16"] for r in gb_real_r]
mi_rep = [r["repeat16"] for r in mi_real_r]
MI_REAL_CYC = st.median(r["cycle_ms"] for r in mi_real_r)
GB_REAL_CYC = st.median(r["cycle_ms"] for r in arm_rounds("gb300", "n1a-real"))
kern = {k: (MV["categories"][k]["wall_us"], GV["categories"][k]["wall_us"], GV["categories"][k]["kernel_us"],
            MV["categories"][k]["kernels"], GV["categories"][k]["kernels"]) for k in ORDER}
gb_wall_total = sum(v[1] for v in kern.values())
slow = {k: kern[k][0] - kern[k][2] for k in ("attention", "moe", "other", "draft")}
parity = {k: pct(kern[k][0], kern[k][2]) for k in ("dense_gemm", "all_reduce", "mhc")}
# GB300 kernels are timed while other streams run, so a category's standalone cost lies roughly
# between its wall share (lower) and its kernel time (upper); MI355X runs serially, so its cost is
# known. The per-category gap is therefore an interval.
GAP = {k: (kern[k][0] - kern[k][2], kern[k][0] - kern[k][1]) for k in ORDER if k != "host_gap"}


def gap_txt(k):
    lo, hi = GAP[k]
    return sgn(lo, 0) if round(lo) == round(hi) else f"{sgn(lo, 0)} … {sgn(hi, 0)}"


LAYERS = 40
LEDGER = {(r["phase"], r["segment"], r["family"]): r for r in
          csv.DictReader(Path("$BS1_FLOOR/analysis/kernel-ledger.csv").open())}
mi_attn_parts = {f: float(LEDGER[("VERIFY", "layers", f)]["est_unprofiled_us"]) / LAYERS
                 for f in ("PA main", "PA split reduce + inverse RoPE", "KV norm/RoPE/store")}
GB_SUB = {(x["category"], x["sub"]): x for x in
          json.loads((CAMP / "gb300/results/p2-real/attribution.json").read_text())["subcategories"]}
gb_attn_layer = GB_SUB[("attention", "attention")]["kernel_us"] / LAYERS
gb_attn_kernels_layer = GB_SUB[("attention", "attention")]["kernels"] / LAYERS
mi_moe_layer, gb_moe_layer = kern["moe"][3] / LAYERS, kern["moe"][4] / LAYERS
cal_unb = max(abs(pct(s0[k]["tps"], b)) for k, b in (("tp4", BB_TP4), ("ep4", BB_EP4), ("nods", BB_OFF)))
cal_bnd = max(abs(pct(n0[k]["tps"], b)) for k, b in (("tp4", BB_TP4), ("ep4", BB_EP4), ("nods", BB_OFF)))
two_launch_spread = pct(max(p["tps"] for p in s1["sim"]["per_launch"]), min(p["tps"] for p in s1["sim"]["per_launch"]))
gap_ms = mi_cyc - gb_cyc
GAP_US = gap_ms * 1e3
GV_HIDDEN = GV["kernel_sum_us"] - GV["busy_us"]
_rt = [json.loads(l) for l in Path("$E2E_CAMPAIGN/results/s2-bs1/1-base/"
                                   "requests.jsonl").open()]
MI_REALTEXT_AL = st.median(r["derived"]["accepted_length"] for r in _rt if r.get("scored"))
KM = "https://github.com/kevin-mii/sglang/pull"

# MI355X placement verdict (session 2). Cycle ms per affinity mode, all launches pooled.
def s2cyc(aff, mode):
    rs = arm_rounds("mi355x", f"s2-{aff}-{mode}")
    key = "step_ms" if mode == "off" else "cycle_ms"
    return st.median(r[key] for r in rs) if rs else None


S2 = {(aff, mode): s2cyc(aff, mode) for aff in ("split", "node1", "none") for mode in ("sim", "real", "off")}
HAVE_S2 = all(S2[(a, m)] is not None for a in ("split", "node1", "none") for m in ("sim", "real"))
S1_SPLIT = {m: st.median(r["cycle_ms"] for p in (f"u-{m}", f"n-{m}") for r in arm_rounds("mi355x", p))
            for m in ("sim", "real")}


def canary_ms(lease, gpu, when):
    return json.loads((CAMP / f"results/{lease}/canary-{when}-{gpu}.json").read_text())["median_ms"]


CAN = {g: (canary_ms("gpu-claim-session2", g, "before"), canary_ms("gpu-claim-session2", g, "after"))
       for g in (4, 5, 6, 7)}
CAN_OTHER = max(abs(pct(a, b)) for g, (b, a) in CAN.items() if g != 4)


# ---------------------------------------------------------------- bilingual helpers
_ZH_PUNCT = re.compile(r"([。，：])(?=[^\s<）」』】])|([。，：])(?=<(?!/))")


def zh(text):
    """Site rule (CLAUDE.md): a half-width space after 。，： when content follows."""
    return _ZH_PUNCT.sub(lambda m: (m.group(1) or m.group(2)) + " ", text)


def P(en, zh_, cls=""):
    c = f' class="{cls}"' if cls else ""
    return f'<p lang="en"{c}>{en}</p>\n<p lang="zh"{c}>{zh(zh_)}</p>\n'


def H(level, en, zh_, id_=None):
    i = f' id="{id_}"' if id_ else ""
    return f'<h{level}{i}><span lang="en">{en}</span><span lang="zh">{zh(zh_)}</span></h{level}>\n'


def S(en, zh_):
    return f'<span lang="en">{en}</span><span lang="zh">{zh(zh_)}</span>'


def UL(items):
    out = ['<ul lang="en">'] + [f"<li>{e}</li>" for e, _ in items] + ["</ul>", '<ul lang="zh">']
    out += [f"<li>{zh(z)}</li>" for _, z in items] + ["</ul>"]
    return "\n".join(out) + "\n"


def callout(title_en, title_zh, en, zh, kind=""):
    return (f'<aside class="callout {kind}"><div class="callout-title">{S(title_en, title_zh)}</div>'
            f'{P(en, zh)}</aside>\n')


def pr(n):
    return f'<a href="{GH}/pull/{n}">#{n}</a>'


def plate(num, title_en, title_zh, svg, cap_en, cap_zh, wide=True):
    w = " wide" if wide else ""
    return (f'<figure class="plate{w}"><div class="plate-meta"><span class="plate-num">Plate {num}</span>'
            f'<span class="plate-title">{S(title_en, title_zh)}</span></div>{svg}'
            f'<figcaption class="caption">{S(cap_en, cap_zh)}</figcaption></figure>\n')


# ---------------------------------------------------------------- SVG
TEAL, MAG, LIME, AMBER, INK, MUTED, GRID = "#35d0c0", "#ff5fa2", "#c9f25c", "#ffb454", "#e4ece7", "#8aa19c", "#23312f"
CAT_COLOR = {"moe": "#ff8a5b", "dense_gemm": "#5bc0ff", "attention": "#c9f25c", "mhc": "#b48cff",
             "all_reduce": "#ffd166", "engram": "#7de2d1", "draft": "#f78fb3", "other": "#6f8783",
             "host_gap": "none"}


def svg_open(w, h, label):
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(label)}" '
            f'xmlns="http://www.w3.org/2000/svg">')


def t(x, y, s, cls="lbl", anchor="start", extra=""):
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}"{extra}>{s}</text>'


def plate_calibration():
    W, H0 = 960, 300
    x0, x1, vmax = 250, 900, 1000.0
    X = lambda v: x0 + (x1 - x0) * v / vmax
    rows = [("MoE TP4 · DSpark sim 5.5", BB_TP4, s0["tp4"]["tps"], n0["tp4"]["tps"]),
            ("MoE EP4 · DSpark sim 5.5", BB_EP4, s0["ep4"]["tps"], n0["ep4"]["tps"]),
            ("EP4 · DSpark off", BB_OFF, s0["nods"]["tps"], n0["nods"]["tps"])]
    s = [svg_open(W, H0, "Calibration against BBuf's published GB300 numbers")]
    for i, (fill, tag) in enumerate((("none", "published (README)"), (TEAL + "80", "ours · unbound"),
                                     (TEAL, "ours · NUMA-bound"))):
        lx = x0 + i * 200
        stroke = f' stroke="{MUTED}" stroke-dasharray="3 3"' if fill == "none" else ""
        s.append(f'<rect x="{lx}" y="12" width="22" height="11" fill="{fill}"{stroke}/>')
        s.append(t(lx + 30, 22, tag, "legend"))
    for v in range(0, 1001, 200):
        s.append(f'<line x1="{X(v)}" y1="40" x2="{X(v)}" y2="{H0 - 40}" class="grid"/>')
        s.append(t(X(v), H0 - 22, f"{v}", "tick", "middle"))
    s.append(t(x1, H0 - 6, "tokens / s (BS=1, 4096 in / 1024 out)", "axis", "end"))
    y = 50
    for name, pub, unb, bnd in rows:
        s.append(t(x0 - 14, y + 20, name, "lbl", "end"))
        for i, (v, fill) in enumerate(((pub, "none"), (unb, TEAL + "80"), (bnd, TEAL))):
            yy = y + i * 15
            stroke = f' stroke="{MUTED}" stroke-dasharray="3 3"' if fill == "none" else ""
            s.append(f'<rect x="{x0}" y="{yy}" width="{X(v) - x0:.1f}" height="11" fill="{fill}"{stroke}/>')
            s.append(t(X(v) + 6, yy + 9.5, f"{v:.1f}", "val"))
        y += 74
    s.append("</svg>")
    return "".join(s)


def plate_versions():
    W, H0 = 960, 290
    x0, x1, vmax = 300, 900, 10.0
    X = lambda v: x0 + (x1 - x0) * v / vmax
    rows = [("plain decode step", [("GB300 · 835c3909", old_step, TEAL + "70"), ("GB300 · main", gb_step, TEAL),
                                   ("MI355X · e2e824dc58", mi_step, MAG)]),
            ("DSpark verify cycle", [("GB300 · 835c3909 (EP4)", old_cyc, TEAL + "70"),
                                     ("GB300 · main", gb_cyc, TEAL), ("MI355X · e2e824dc58", mi_cyc, MAG)])]
    s = [svg_open(W, H0, "Per-step cost across versions and platforms")]
    for v in range(0, 11, 2):
        s.append(f'<line x1="{X(v)}" y1="26" x2="{X(v)}" y2="{H0 - 38}" class="grid"/>')
        s.append(t(X(v), H0 - 20, f"{v}", "tick", "middle"))
    s.append(t(x1, H0 - 4, "milliseconds per step (lower is faster; NUMA-bound where available)", "axis", "end"))
    y = 34
    for head, bars in rows:
        s.append(t(24, y + 10, head.upper(), "head"))
        for name, v, c in bars:
            y += 24
            s.append(t(x0 - 14, y + 9, name, "lbl", "end"))
            s.append(f'<rect x="{x0}" y="{y}" width="{X(v) - x0:.1f}" height="13" fill="{c}"/>')
            s.append(t(X(v) + 6, y + 11, f"{v:.2f} ms", "val"))
        y += 34
    s.append("</svg>")
    return "".join(s)


def plate_numa_strip():
    W, H0 = 960, 262
    x0, x1, lo, hi = 190, 640, 750.0, 1250.0
    X = lambda v: x0 + (x1 - x0) * (v - lo) / (hi - lo)
    lanes = [("unbound", arm_rounds("gb300", "s1a-sim") + arm_rounds("gb300", "e1-base-sim"), TEAL + "90"),
             ("NUMA-bound", arm_rounds("gb300", "n1a-sim") + arm_rounds("gb300", "e1-nice-sim"), LIME)]
    s = [svg_open(W, H0, "GB300 simulated-acceptance rounds, unbound versus NUMA-bound")]
    for v in range(800, 1251, 100):
        s.append(f'<line x1="{X(v):.1f}" y1="30" x2="{X(v):.1f}" y2="215" class="grid"/>')
        s.append(t(X(v), 232, f"{v}", "tick", "middle"))
    s.append(t(x1, 250, "tokens / s per timed round (sim 5.5, main, 5 launches per lane)", "axis", "end"))
    for i, (name, rs, c) in enumerate(lanes):
        yc = 75 + i * 90
        s.append(t(x0 - 14, yc + 4, name, "lbl", "end"))
        launches = sorted({(r["src"], r["launch"]) for r in rs})
        for r in rs:
            j = launches.index((r["src"], r["launch"]))
            s.append(f'<circle cx="{X(r["tps"]):.1f}" cy="{yc - 24 + j * 12:.1f}" r="4.2" fill="{c}"/>')
        m = st.median(r["tps"] for r in rs)
        s.append(f'<line x1="{X(m):.1f}" y1="{yc - 34}" x2="{X(m):.1f}" y2="{yc + 32}" stroke="{INK}" stroke-width="1.5"/>')
        s.append(t(X(m) + 5, yc - 36, f"median {m:.0f}", "val"))
    # TTFT mini-chart
    bx0, bx1, tmax = 700, 930, 450.0
    BX = lambda v: bx0 + (bx1 - bx0) * v / tmax
    s.append(t(bx0, 34, "TTFT, 4096-token prefill (ms)", "head"))
    pairs = [("off", s1["off"]["ttft_ms"], n1["off"]["ttft_ms"]), ("real", s1["real"]["ttft_ms"], n1["real"]["ttft_ms"]),
             ("sim", e1b["ttft_ms"], e1n["ttft_ms"])]
    for i, (name, u, b) in enumerate(pairs):
        y = 56 + i * 52
        s.append(t(bx0, y + 2, name, "lbl"))
        s.append(f'<rect x="{bx0}" y="{y + 8}" width="{BX(u) - bx0:.1f}" height="11" fill="{TEAL}90"/>')
        s.append(t(BX(u) + 4, y + 17.5, f"{u:.0f}", "val"))
        s.append(f'<rect x="{bx0}" y="{y + 22}" width="{BX(b) - bx0:.1f}" height="11" fill="{LIME}"/>')
        s.append(t(BX(b) + 4, y + 31.5, f"{b:.0f}", "val"))
    s.append(t(bx0, 232, "unbound", "legend", extra=f' fill="{TEAL}"'))
    s.append(t(bx0 + 70, 232, "bound", "legend", extra=f' fill="{LIME}"'))
    s.append("</svg>")
    return "".join(s)


def plate_topology():
    W, H0 = 960, 380
    s = [svg_open(W, H0, "Where the four TP schedulers run on each machine")]
    # GB300 tray
    s.append(t(20, 28, "GB300 TRAY · 2 × Grace (72 cores each) · NVLink-C2C", "head"))
    for i, (cpus, gpus) in enumerate((("CPU 0–71 · node 0", ("GPU0 · TP0", "GPU1 · TP1")),
                                       ("CPU 72–143 · node 1", ("GPU2 · TP2", "GPU3 · TP3")))):
        x = 20 + i * 230
        s.append(f'<rect x="{x}" y="48" width="210" height="46" class="box"/>')
        s.append(t(x + 105, 76, cpus, "lbl", "middle"))
        for j, g in enumerate(gpus):
            gx = x + j * 108
            s.append(f'<rect x="{gx}" y="150" width="102" height="40" class="box gpu"/>')
            s.append(t(gx + 51, 175, g, "lbl", "middle"))
            s.append(f'<line x1="{gx + 51}" y1="94" x2="{gx + 51}" y2="150" class="edge"/>')
    s.append(t(20, 222, f"unbound: TP2/TP3 threads on their GPU's node {min(tp23_thr):.0%}–{max(tp23_thr):.0%}, "
                        f"their pages {min(tp23_pg):.0%}–{max(tp23_pg):.0%}", "note", extra=f' fill="{AMBER}"'))
    s.append(t(20, 240, f"bound (--cap-add SYS_NICE): every rank 100% local threads, "
                        f"{min(nice_pg):.0%}–{max(nice_pg):.0%} local pages", "note", extra=f' fill="{LIME}"'))
    # MI355X node
    ox = 500
    s.append(t(ox, 28, "MI355X NODE · 2 sockets · GPUs 4–7 (our TP4) on node 1", "head"))
    s.append(f'<rect x="{ox}" y="48" width="210" height="46" class="box"/>')
    s.append(t(ox + 105, 68, "node 0 · CPU 0–63, 128–191", "lbl", "middle"))
    s.append(t(ox + 105, 84, "GPUs 0–3 (unused)", "note", "middle"))
    s.append(f'<rect x="{ox + 230}" y="48" width="210" height="46" class="box"/>')
    s.append(t(ox + 335, 68, "node 1 · CPU 64–127, 192–255", "lbl", "middle"))
    s.append(t(ox + 335, 84, "GPUs 4–7", "note", "middle"))
    ranks = [("TP0", "CPU 0–31", False), ("TP1", "CPU 32–63", False), ("TP2", "CPU 64–95", True),
             ("TP3", "CPU 96–127", True)]
    for j, (tp, cpus, local) in enumerate(ranks):
        gx = ox + j * 110
        c = LIME if local else AMBER
        s.append(f'<rect x="{gx}" y="150" width="100" height="40" class="box gpu"/>')
        s.append(t(gx + 50, 168, f"GPU{4 + j} · {tp}", "lbl", "middle"))
        s.append(t(gx + 50, 183, cpus, "note", "middle"))
        target = ox + 105 if j < 2 else ox + 335
        s.append(f'<line x1="{gx + 50}" y1="150" x2="{target}" y2="94" stroke="{c}" stroke-width="2"/>')
    s.append(t(ox, 222, "SGLANG_SET_CPU_AFFINITY=1: rank r gets physical cores [32r, 32r+32)", "note",
               extra=f' fill="{AMBER}"'))
    s.append(t(ox, 240, "→ TP0/TP1 are pinned to node 0, remote from GPUs 4–7", "note", extra=f' fill="{AMBER}"'))
    s.append(t(ox, 258, "SGLang's own NUMA binding is CUDA/XPU-only; numactl is not installed", "note"))
    # legend
    s.append(f'<line x1="20" y1="300" x2="44" y2="300" stroke="{LIME}" stroke-width="2"/>')
    s.append(t(50, 304, "local (same NUMA node as the GPU)", "legend"))
    s.append(f'<line x1="300" y1="300" x2="324" y2="300" stroke="{AMBER}" stroke-width="2"/>')
    s.append(t(330, 304, "remote", "legend"))
    s.append(t(20, 340, "Every layer ends in a TP all-reduce, so a host-launched step runs at the pace of the "
                        "slowest rank's CPU.", "note"))
    s.append("</svg>")
    return "".join(s)


def plate_acceptance():
    W, H0 = 960, 252
    x0, x1, lo, hi = 190, 640, 2.0, 6.2
    X = lambda v: x0 + (x1 - x0) * (v - lo) / (hi - lo)
    s = [svg_open(W, H0, "Natural acceptance length per round on the same random prompt")]
    for v in (2, 3, 4, 5, 6):
        s.append(f'<line x1="{X(v):.1f}" y1="30" x2="{X(v):.1f}" y2="205" class="grid"/>')
        s.append(t(X(v), 222, f"{v}", "tick", "middle"))
    s.append(f'<line x1="{X(5.5):.1f}" y1="30" x2="{X(5.5):.1f}" y2="205" stroke="{MUTED}" stroke-dasharray="4 4"/>')
    s.append(t(X(5.5), 24, "sim target 5.5", "note", "middle"))
    s.append(t(x1, 240, "accepted tokens per verify (real acceptance, one dot per timed round)", "axis", "end"))
    for i, (name, als, c) in enumerate((("GB300", gb_al, TEAL), ("MI355X", mi_al, MAG))):
        yc = 80 + i * 80
        s.append(t(x0 - 14, yc + 4, name, "lbl", "end"))
        for k, v in enumerate(sorted(als)):
            s.append(f'<circle cx="{X(v):.1f}" cy="{yc - 16 + (k % 5) * 8:.1f}" r="3.8" fill="{c}" fill-opacity="0.85"/>')
    # cycle bars
    bx0, bx1, cmax = 700, 930, 10.0
    BX = lambda v: bx0 + (bx1 - bx0) * v / cmax
    s.append(t(bx0, 34, "cost per verify cycle (ms)", "head"))
    bars = [("GB300 real", GB_REAL_CYC, TEAL), ("GB300 sim", gb_cyc, TEAL + "90"),
            ("MI355X real", MI_REAL_CYC, MAG), ("MI355X sim", mi_cyc, MAG + "90")]
    for i, (name, v, c) in enumerate(bars):
        y = 50 + i * 38
        s.append(t(bx0, y + 2, name, "lbl"))
        s.append(f'<rect x="{bx0}" y="{y + 8}" width="{BX(v) - bx0:.1f}" height="12" fill="{c}"/>')
        s.append(t(BX(v) + 4, y + 18, f"{v:.2f}", "val"))
    s.append("</svg>")
    return "".join(s)


def plate_anatomy():
    W, H0 = 960, 360
    x0, x1, vmax = 190, 930, 9000.0
    X = lambda v: x0 + (x1 - x0) * v / vmax
    rows = [("MI355X · serial", MV["categories"], "wall_us", "8,921 µs, prior base"),
            ("GB300 · wall time", GV["categories"], "wall_us", f"{gb_wall_total:,.0f} µs mean"),
            ("GB300 · kernel time", GV["categories"], "kernel_us", f"{GV['kernel_sum_us']:,.0f} µs summed")]
    s = [svg_open(W, H0, "Anatomy of one DSpark verify cycle at BS=1")]
    for v in range(0, 9001, 1000):
        s.append(f'<line x1="{X(v):.1f}" y1="30" x2="{X(v):.1f}" y2="200" class="grid"/>')
        s.append(t(X(v), 216, f"{v // 1000}", "tick", "middle"))
    s.append(t(x1, 234, "milliseconds per verify cycle (TP0)", "axis", "end"))
    for i, (name, cats, key, note) in enumerate(rows):
        y = 44 + i * 52
        s.append(t(x0 - 14, y + 16, name, "lbl", "end"))
        s.append(t(x0 - 14, y + 31, note, "note", "end"))
        x = x0
        for k in ORDER:
            v = cats[k][key]
            if v <= 0:
                continue
            w = X(v) - x0
            if k == "host_gap":
                s.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="24" fill="url(#hatch)" stroke="{MUTED}"/>')
            else:
                dash = ' stroke-dasharray="3 2" stroke="#0a0e0f"' if key == "kernel_us" else ""
                s.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="24" fill="{CAT_COLOR[k]}"{dash}/>')
            x += w
    lx, ly = 30, 262
    for i, k in enumerate(ORDER):
        cx = lx + (i % 5) * 186
        cy = ly + (i // 5) * 24
        fill = "url(#hatch)" if k == "host_gap" else CAT_COLOR[k]
        s.append(f'<rect x="{cx}" y="{cy - 10}" width="14" height="12" fill="{fill}" stroke="{MUTED}" stroke-width="0.5"/>')
        s.append(t(cx + 20, cy, LAB[k], "legend"))
    s.append(t(30, 330, "Wall shares split overlapping kernels evenly, so the GB300 wall bar sums to the mean cycle; "
                        "the kernel bar is total kernel duration and exceeds it.", "note"))
    s.append("</svg>")
    return "".join(s)


# ---------------------------------------------------------------- sections
def anatomy_table():
    head = ("<tr><th>" + S("category", "类别") + "</th><th>MI355X µs</th><th>" + S("GB300 wall µs", "GB300 墙钟 µs")
            + "</th><th>" + S("GB300 kernel µs", "GB300 kernel µs") + "</th><th>" + S("gap µs", "差距 µs") + "</th><th>"
            + S("kernels MI / GB", "kernel 数 MI / GB") + "</th></tr>")
    body = []
    zh = {"moe": "MoE 专家与路由", "dense_gemm": "dense GEMM 与激活量化", "attention": "attention / indexer / KV",
          "mhc": "mHC", "all_reduce": "all-reduce 与 MoE finalize", "engram": "Engram", "draft": "DSpark draft",
          "other": "其他胶水 kernel", "host_gap": "GPU 空闲"}
    for k in ORDER:
        mi, gw, gk, mk, gkc = kern[k]
        kc = "–" if k == "host_gap" else f"{mk:.0f} / {gkc:.0f}"
        gks = "–" if k == "host_gap" else f0(gk)
        gap = sgn(mi - gw, 0) if k == "host_gap" else gap_txt(k)
        body.append(f"<tr><td>{S(LAB[k], zh[k])}</td><td>{f0(mi)}</td><td>{f0(gw)}</td><td>{gks}</td><td>{gap}</td>"
                    f"<td>{kc}</td></tr>")
    total = (f"<tr class='total'><td>{S('total', '合计')}</td><td>{f0(MV['cycle_us'])}</td><td>{f0(gb_wall_total)}</td>"
             f"<td>{f0(GV['kernel_sum_us'])}</td><td>{sgn(MV['cycle_us'] - gb_wall_total, 0)}</td>"
             f"<td>{MV['kernels']:.0f} / {GV['kernels']:.0f}</td></tr>")
    return f'<div class="tablewrap"><table class="data">{head}{"".join(body)}{total}</table></div>\n'


DRAFT = "--draft" in sys.argv


DESC_ZH = {
    "s0-tp4": "作者 commit 835c3909，MoE TP4（EP1），sim 5.5，不绑定",
    "s0-ep4": "作者 commit 835c3909，MoE EP4，sim 5.5，不绑定",
    "s0-nods": "作者 commit 835c3909，EP4，关闭 DSpark，不绑定",
    "n0-tp4": "作者 commit 835c3909，MoE TP4（EP1），sim 5.5，NUMA 绑定",
    "n0-ep4": "作者 commit 835c3909，MoE EP4，sim 5.5，NUMA 绑定",
    "n0-nods": "作者 commit 835c3909，EP4，关闭 DSpark，NUMA 绑定",
    "s1a-off": "main ffac53d779，HT cell（关闭 DSpark），不绑定",
    "s1a-sim": "main ffac53d779，LL cell，sim 5.5，不绑定",
    "s1a-real": "main ffac53d779，LL cell，真实接受，不绑定",
    "n1a-off": "main ffac53d779，HT cell（关闭 DSpark），NUMA 绑定",
    "n1a-sim": "main ffac53d779，LL cell，sim 5.5，NUMA 绑定",
    "n1a-real": "main ffac53d779，LL cell，真实接受，NUMA 绑定",
    "e1-base-sim": "main ffac53d779，LL cell，sim 5.5，不绑定（NUMA 对照）",
    "e1-nice-sim": "main ffac53d779，LL cell，sim 5.5，NUMA 绑定（NUMA 对照）",
    "u-off": "e2e824dc58+#8，HT cell，SGLang 切分亲和性",
    "u-real": "e2e824dc58+#8，LL cell，真实接受，SGLang 切分亲和性",
    "u-sim": "e2e824dc58+#8，LL cell，sim 5.5，SGLang 切分亲和性",
    "n-off": "e2e824dc58+#8，HT cell，切分亲和性 + 主进程在 node 1",
    "n-real": "e2e824dc58+#8，LL cell，真实接受，切分亲和性 + 主进程在 node 1",
    "n-sim": "e2e824dc58+#8，LL cell，sim 5.5，切分亲和性 + 主进程在 node 1",
    "s2-split-sim": "session 2，sim 5.5，SGLang 切分亲和性",
    "s2-node1-sim": "session 2，sim 5.5，所有 scheduler 在 node 1",
    "s2-none-sim": "session 2，sim 5.5，不固定",
    "s2-split-real": "session 2，真实接受，SGLang 切分亲和性",
    "s2-node1-real": "session 2，真实接受，所有 scheduler 在 node 1",
    "s2-none-real": "session 2，真实接受，不固定",
    "s2-node1-off": "session 2，关闭 DSpark，所有 scheduler 在 node 1",
    "s2-none-off": "session 2，关闭 DSpark，不固定",
}


def arms_table():
    head = ("<tr><th>" + S("arm", "arm") + "</th><th>" + S("configuration", "配置") + "</th><th>"
            + S("launches", "启动") + "</th><th>" + S("tok/s median (range)", "tok/s 中位数（范围）") + "</th><th>AL</th><th>"
            + S("ms / cycle or step", "ms / 周期或步") + "</th><th>TTFT ms</th><th>" + S("per-launch tok/s", "各次启动 tok/s")
            + "</th></tr>")
    rows, last = [], None
    for a in json.loads((CAMP / "analysis/arms.json").read_text()):
        if a["platform"] != last:
            last = a["platform"]
            title = "4× GB300 · CUDA 13.2" if last == "gb300" else "4× MI355X · ROCm 7.2"
            rows.append(f'<tr class="group"><td colspan="8">{title}</td></tr>')
        ms = a["cycle_ms"] if a["cycle_ms"] is not None else a["step_ms"]
        al = f2(a["al"]) if a["al"] else "–"
        per = " · ".join(f"{p['tps']:.0f}" for p in a["per_launch"])
        rows.append(f"<tr><td><code>{a['group']}</code></td>"
                    f"<td>{S(html.escape(a['desc']), DESC_ZH[a['group']])}</td><td>{a['launches']}</td>"
                    f"<td>{f1(a['tps'])} ({a['tps_min']:,.0f}–{a['tps_max']:,.0f})</td><td>{al}</td>"
                    f"<td>{f2(ms)}</td><td>{f0(a['ttft_ms'])}</td><td>{per}</td></tr>")
    return f'<div class="tablewrap"><table class="data wide-table">{head}{"".join(rows)}</table></div>\n'


def placement_table():
    def mi_rank(prefix):
        cells = {}
        for d in sorted((CAMP / "results").glob(f"{prefix}-*")):
            status = d / "status.json"
            if not (d / "processes.json").exists() or not status.exists() or not json.loads(status.read_text())["ok"]:
                continue
            for p in json.loads((d / "processes.json").read_text()):
                m = re.search(r"scheduler_TP(\d)", p["cmdline"])
                if m:
                    pages = p["numa_pages"]
                    frac = pages.get("N1", 0) / max(1, sum(pages.values()))
                    cells.setdefault(int(m.group(1)), []).append((p["cpus_allowed"], frac))
        return cells

    def gb_rank(prefix):
        cells = {}
        for x in "abc":
            for tp, (thr, pg) in locality(f"{prefix}-{x}/placement_post").items():
                cells.setdefault(tp, []).append((thr, pg))
        return cells

    def rng(v):
        lo, hi = round(min(v) * 100), round(max(v) * 100)
        return f"{lo}%" if lo == hi else f"{lo}–{hi}%"

    head = ("<tr><th>" + S("machine · placement", "机器 · 放置") + "</th>"
            + "".join(f"<th>TP{i}</th>" for i in range(4)) + "</tr>")
    rows = []
    for label_en, label_zh, cells in (("GB300 · unbound", "GB300 · 不绑定", gb_rank("e1-base-sim")),
                                      ("GB300 · NUMA-bound", "GB300 · NUMA 绑定", gb_rank("e1-nice-sim"))):
        tds = "".join(f"<td>{S('threads', '线程')} {rng([c[0] for c in cells[i]])}<br>"
                      f"{S('pages', '内存页')} {rng([c[1] for c in cells[i]])}</td>" for i in range(4))
        rows.append(f"<tr><td>{S(label_en, label_zh)}</td>{tds}</tr>")
    for aff, label_en, label_zh in (("split", "MI355X · split (default)", "MI355X · 切分（默认）"),
                                    ("node1", "MI355X · node 1", "MI355X · node 1"),
                                    ("none", "MI355X · unpinned", "MI355X · 不固定")):
        cells = mi_rank(f"s2-{aff}")
        tds = "".join(f"<td>CPU {sorted({c[0] for c in cells[i]})[0].split(',')[0]}<br>"
                      f"{S('pages', '内存页')} {rng([c[1] for c in cells[i]])}</td>" for i in range(4))
        rows.append(f"<tr><td>{S(label_en, label_zh)}</td>{tds}</tr>")
    note = P("Local = on the NUMA node of the rank's GPU (GB300: node 0 for TP0/TP1, node 1 for TP2/TP3; MI355X: node 1 "
             "for all four). GB300 threads are counted by the CPU each thread last ran on; page shares come from "
             "numa_maps and include the memory-mapped model files.",
             "本地指位于该 rank 所用 GPU 的 NUMA 节点（GB300：TP0/TP1 为 node 0，TP2/TP3 为 node 1；MI355X：四个 rank 都是 node 1）。"
             "GB300 的线程按每个线程最近一次运行的 CPU 统计；内存页比例来自 numa_maps，包含 mmap 进来的模型文件。", "tnote")
    return f'<div class="tablewrap"><table class="data">{head}{"".join(rows)}</table></div>\n' + note


def decode_table():
    zhl = {"moe": "MoE 专家与路由", "dense_gemm": "dense GEMM 与激活量化", "attention": "attention / indexer / KV",
           "mhc": "mHC", "all_reduce": "all-reduce 与 MoE finalize", "engram": "Engram", "draft": "DSpark draft",
           "other": "其他胶水 kernel", "host_gap": "GPU 空闲"}
    head = ("<tr><th>" + S("category (GB300 plain decode step)", "类别（GB300 plain decode 每步）") + "</th><th>"
            + S("wall µs", "墙钟 µs") + "</th><th>" + S("kernel µs", "kernel µs") + "</th><th>"
            + S("kernels", "kernel 数") + "</th></tr>")
    rows = []
    for k in ORDER:
        c = GD["categories"][k]
        if k == "draft":
            continue
        kk = "–" if k == "host_gap" else f0(c["kernel_us"])
        kc = "–" if k == "host_gap" else f"{c['kernels']:.0f}"
        rows.append(f"<tr><td>{S(LAB[k], zhl[k])}</td><td>{f0(c['wall_us'])}</td><td>{kk}</td><td>{kc}</td></tr>")
    wall = sum(GD["categories"][k]["wall_us"] for k in ORDER)
    rows.append(f"<tr class='total'><td>{S('total', '合计')}</td><td>{f0(wall)}</td><td>{f0(GD['kernel_sum_us'])}</td>"
                f"<td>{GD['kernels']:.0f}</td></tr>")
    return f'<div class="tablewrap"><table class="data">{head}{"".join(rows)}</table></div>\n'


def placement_mi355x_paragraph():
    if not HAVE_S2:
        if DRAFT:
            return "[session 2 pending]", "[session 2 待补]", 0.0, 0.0
        raise SystemExit("session 2 (MI355X placement A/B) results are missing; run it before building")
    sp, nd, no = S2[("split", "sim")], S2[("node1", "sim")], S2[("none", "sim")]
    rs, rn, ro = S2[("split", "real")], S2[("node1", "real")], S2[("none", "real")]
    d_sim, d_real = pct(nd, sp), pct(rn, rs)
    en = (f"We then ran the placement A/B on MI355X in one session, mirrored in time: SGLang's split affinity, "
          f"all four schedulers on node 1, and no pinning. Cost per verify cycle with simulated acceptance: "
          f"{sp:.2f} ms split, {nd:.2f} ms node 1, {no:.2f} ms unpinned (node 1 {sgn(d_sim)}%, unpinned "
          f"{sgn(pct(no, sp))}% against split); with real acceptance {rs:.2f}, {rn:.2f} and {ro:.2f} ms "
          f"({sgn(d_real)}% and {sgn(pct(ro, rs))}%).")
    zh = (f"随后我们在 MI355X 上用同一个 session、按时间对称的顺序做了放置对照：SGLang 的切分亲和性、四个 scheduler "
          f"全部放在 node 1、完全不绑定。模拟接受时每个 verify 周期分别是 {sp:.2f}、{nd:.2f}、{no:.2f} ms（相对切分，node 1 "
          f"{sgn(d_sim)}%，不绑定 {sgn(pct(no, sp))}%）；真实接受时分别是 {rs:.2f}、{rn:.2f}、{ro:.2f} ms（{sgn(d_real)}% 和 "
          f"{sgn(pct(ro, rs))}%）。")
    off_split = [r["step_ms"] for p in ("u-off", "n-off") for r in arm_rounds("mi355x", p)]
    on, oo = S2[("node1", "off")], S2[("none", "off")]
    if on and oo and off_split:
        so = st.median(off_split)
        en += (f" Plain decode, against the split runs of the first session: {so:.2f} ms per step split, {on:.2f} "
               f"node 1, {oo:.2f} unpinned.")
        zh += (f"plain decode 以第一个 session 的切分结果为基准：切分每步 {so:.2f} ms，node 1 为 {on:.2f} ms，"
               f"不绑定为 {oo:.2f} ms。")
    elif not DRAFT:
        raise SystemExit("session 2 plain-decode arms are missing; run them before building")
    return en, zh, d_sim, d_real


def build():
    pe, pz, d_sim, d_real = placement_mi355x_paragraph()
    big = HAVE_S2 and max(abs(pct(S2[(a, m)], S2[("split", m)])) for a in ("node1", "none")
                          for m in ("sim", "real")) >= 1.0
    if big:
        verdict_en = ("On MI355X the remote ranks do cost time, so the affinity arithmetic is a real bug on this "
                      "node, not a cosmetic one.")
        verdict_zh = "在 MI355X 上，远端 rank 确实会拖慢周期，所以这段亲和性算法在这台机器上是真问题，不只是不美观。"
    else:
        maxd = (max(abs(pct(S2[(a, m)], S2[("split", m)])) for a in ("node1", "none") for m in ("sim", "real"))
                if HAVE_S2 else 0.0)
        verdict_en = (f"Placement moves the MI355X BS=1 cycle by at most {maxd:.1f}%. The host is still on the critical "
                      "path here (the verify-to-draft wait that kevin-mii/sglang#12 removes), but that share of the cycle "
                      "does not depend on which socket runs it, and sim costs the same as real acceptance on this machine. "
                      "The affinity arithmetic is still wrong and should be fixed, but the 2x does not come from it.")
        verdict_zh = (f"放置方式对 MI355X BS=1 周期的影响最多只有 {maxd:.1f}%。host 在这里仍处在关键路径上（就是 "
                      "kevin-mii/sglang#12 消掉的那段 verify 到 draft 的等待），但这部分耗时和它跑在哪个 socket 上无关；在这台机器上"
                      "sim 和真实接受的每周期耗时也相同。这段亲和性算法仍然是错的，应该修，但 2 倍差距不来自它。")

    short_en = "and there it does cost cycle time" if big else "but it does not move the BS=1 cycle"
    short_zh = "它确实拖慢了周期" if big else "它不影响 BS=1 的周期"
    gain_nice = pct(e1n["tps"], e1b["tps"])
    gain_n1 = pct(n1["sim"]["tps"], s1["sim"]["tps"])
    base_launch = " / ".join(f1(p["tps"]) for p in e1b["per_launch"])
    nice_launch = " / ".join(f1(p["tps"]) for p in e1n["per_launch"])
    pw_b = f"{min(p for _, p in pw_base):.0f}–{max(p for _, p in pw_base):.0f} W"
    pw_n = f"{min(p for _, p in pw_nice):.0f}–{max(p for _, p in pw_nice):.0f} W"
    clk = {round(c) for c, _ in pw_base + pw_nice}
    ratio_step, ratio_cyc = mi_step / gb_step, mi_cyc / gb_cyc
    over = GV["overlap"]
    hidden = GV["kernel_sum_us"] - GV["busy_us"]

    sec = []
    # Prologue
    sec.append('<section id="prologue">' + H(2, "Prologue", "序", None))
    sec.append(P(
        "SGLang's DeepSeek-V4.1 Flash numbers were produced on 4&times;GB300. Before borrowing anything from that work "
        "we wanted three answers: how far MI355X is behind when the request, the client and the metric are identical; "
        "which part of the difference is the machine and which part is software we can port; and which parts of the "
        "published numbers are measurement artifacts. We reran the published GB300 protocol on a GB300 tray, profiled "
        "it with nsys, and replayed the same prompt with the same client on four MI355X GPUs.",
        "SGLang 的 DeepSeek-V4.1 Flash 数据来自 4&times;GB300。在借鉴这些工作之前，我们想先回答三个问题：请求、客户端和"
        "指标完全相同时，MI355X 落后多少；差距里哪部分来自机器，哪部分是可以移植的软件；公开数字里哪些是测量方式造成的。"
        "为此我们在一个 GB300 托盘上重跑了公开的测试流程并用 nsys 做了 profile，再用同一个客户端、同一个 prompt 在四张 "
        "MI355X 上重放。"))
    sec.append(UL([
        (f"Per step GB300 is {ratio_step:.2f}&times; faster in plain decode ({f2(gb_step)} vs {f2(mi_step)} ms) and "
         f"{ratio_cyc:.2f}&times; per DSpark verify cycle ({f2(gb_cyc)} vs {f2(mi_cyc)} ms).",
         f"按每一步算，plain decode 下 GB300 快 {ratio_step:.2f} 倍（{f2(gb_step)} 对 {f2(mi_step)} ms），"
         f"DSpark 每个 verify 周期快 {ratio_cyc:.2f} 倍（{f2(gb_cyc)} 对 {f2(mi_cyc)} ms）。"),
        (f"Both machines launch about {MV['kernels']:,.0f}–{GV['kernels']:,.0f} kernels per verify cycle. GB300's kernel "
         f"durations add up to {GV['kernel_sum_us'] / 1e3:.1f} ms yet occupy {GV['busy_us'] / 1e3:.1f} ms of GPU time because "
         f"streams overlap ({over:.2f}&times;); the MI355X graph runs its {MV['kernel_sum_us'] / 1e3:.1f} ms back to back.",
         f"两台机器每个 verify 周期都要发射约 {MV['kernels']:,.0f}–{GV['kernels']:,.0f} 个 kernel。GB300 的 kernel 时长加起来有 "
         f"{GV['kernel_sum_us'] / 1e3:.1f} ms，但因为多个 stream 重叠，只占用了 {GV['busy_us'] / 1e3:.1f} ms 的 GPU 时间（{over:.2f} 倍）；"
         f"MI355X 的 graph 则把 {MV['kernel_sum_us'] / 1e3:.1f} ms 串行执行完。"),
        (f"Per category the gap is an interval, because GB300's kernels are timed while sharing the GPU. Attention "
         f"(+{GAP['attention'][0] / 1e3:.2f} to +{GAP['attention'][1] / 1e3:.2f} ms) and MoE (+{GAP['moe'][0] / 1e3:.2f} to "
         f"+{GAP['moe'][1] / 1e3:.2f} ms) are gaps under either bound, glue kernels and the draft follow, all-reduce is within "
         f"{GAP['all_reduce'][1] / 1e3:.2f} ms, and dense GEMM (+{GAP['dense_gemm'][0] / 1e3:.2f} to "
         f"+{GAP['dense_gemm'][1] / 1e3:.2f} ms) and mHC ({sgn(GAP['mhc'][0] / 1e3, 2)} to {sgn(GAP['mhc'][1] / 1e3, 2)} ms) "
         "cannot be settled from this profile.",
         f"按类别看，差距是一个区间，因为 GB300 的 kernel 是在共享 GPU 的情况下计时的。attention（多 {GAP['attention'][0] / 1e3:.2f} 到 "
         f"{GAP['attention'][1] / 1e3:.2f} ms）和 MoE（多 {GAP['moe'][0] / 1e3:.2f} 到 {GAP['moe'][1] / 1e3:.2f} ms）无论取哪个边界都是差距；"
         f"其次是胶水 kernel 和 draft；all-reduce 相差在 {GAP['all_reduce'][1] / 1e3:.2f} ms 以内；dense GEMM（多 "
         f"{GAP['dense_gemm'][0] / 1e3:.2f} 到 {GAP['dense_gemm'][1] / 1e3:.2f} ms）和 mHC（{sgn(GAP['mhc'][0] / 1e3, 2)} 到 "
         f"{sgn(GAP['mhc'][1] / 1e3, 2)} ms）靠这份 profile 定不下来。"),
        (f"The largest variance on GB300, {gain_nice:.0f}% between launches of one configuration, was CPU placement. "
         f"MI355X has its own placement bug (half of the ranks are pinned to the far socket), {short_en}.",
         f"GB300 上最大的波动，即同一配置不同启动之间差 {gain_nice:.0f}%，来自 CPU 放置。MI355X 有它自己的放置问题（一半 "
         f"rank 被钉在远端 socket 上），{'而且' if big else '但'}{short_zh}。"),
        ("Two measurement traps: simulated acceptance adds host work that real acceptance does not, and natural "
         "acceptance on a random prompt depends on which repetition loop each machine's numerics fall into.",
         "两个测量陷阱：模拟接受会引入真实接受没有的 host 工作；而随机 prompt 上的真实接受长度，取决于各机器的数值误差"
         "把生成带进了哪种重复循环。"),
    ]))
    sec.append("</section>")

    # Contract
    sec.append('<section id="contract">' + H(2, "1 · What was held fixed", "1 · 固定了什么", None))
    sec.append(P(
        "A cross-platform number means something only when everything that is not the platform is identical, and the "
        "metric does not depend on something the platform changes by accident. Both machines therefore ran the same "
        "checkpoint, the same 4,096 token ids, the same client code and the same timing formula:",
        "跨平台的数字要有意义，前提是平台以外的一切完全相同，而且指标不依赖平台会顺带改变的东西。所以两台机器用的是同一个 "
        "checkpoint、同样的 4,096 个 token id、同一份客户端代码和同一个计时公式："))
    sec.append(UL([
        ("checkpoint <code>deepseek-ai/DeepSeek-V4.1-Flash@dba1be0a</code>;",
         "checkpoint <code>deepseek-ai/DeepSeek-V4.1-Flash@dba1be0a</code>；"),
        ("BBuf's random prompt: 4,096 ids drawn with seed 42, no chat template (<code>prompt.json</code> "
         "sha256 <code>2331eb91…</code>);", "BBuf 的随机 prompt：用 seed 42 抽取的 4,096 个 id，不套 chat 模板（<code>prompt.json</code> "
         "sha256 <code>2331eb91…</code>）；"),
        ("BBuf's client: flush the cache before every request, freeze GC once, temperature 0, "
         "<code>ignore_eos</code>, 1,024 output tokens, one discarded warm-up and six timed rounds per server;",
         "BBuf 的客户端：每个请求前清缓存，只 freeze 一次 GC，temperature 0，<code>ignore_eos</code>，输出 1,024 个 token，"
         "每个 server 丢弃一次热身、计时六轮；"),
        ("a fresh server for every launch and at least two launches per configuration;",
         "每次启动都是全新的 server，每个配置至少启动两次；"),
        ("TP4 / EP4 with each platform's cookbook cell: Low-Latency (DSpark block 5, decode graph cap 64, memory "
         "fraction 0.8) or High-Throughput (DSpark off).",
         "TP4 / EP4，用各平台自己的 cookbook cell：Low-Latency（DSpark block 5，decode graph 上限 64，显存比例 0.8）或 "
         "High-Throughput（关闭 DSpark）。"),
    ]))
    sec.append(P(
        "Three modes per platform: <b>off</b> (plain decode), <b>sim</b> (static verify, acceptance forced to 5 or 6 "
        "tokens with mean 5.5, the published protocol) and <b>real</b> (natural acceptance).",
        "每个平台三种模式：<b>off</b>（plain decode）、<b>sim</b>（static verify，每步强制接受 5 或 6 个 token，均值 5.5，即公开"
        "的测试流程）和 <b>real</b>（真实接受）。"))
    sec.append(callout(
        "The unit of comparison", "用什么单位比较",
        "DSpark throughput is accepted tokens per verify divided by the cost of one verify. The first factor belongs to "
        "the model's trajectory (section 5 shows it differs between the machines on this prompt); the second belongs to "
        "the machine. So every DSpark comparison here uses <b>cost per verify cycle</b> = (last event &minus; first event) "
        "&divide; verify count, and plain decode uses cost per step = 1 &divide; tokens/s.",
        "DSpark 的吞吐等于每次 verify 接受的 token 数除以一次 verify 的耗时。前者由模型的生成轨迹决定（第 5 节会看到，"
        "在这个 prompt 上两台机器不同）；后者才由机器决定。所以本文所有 DSpark 对比都用<b>每个 verify 周期的耗时</b>，即"
        "（最后一个事件 &minus; 首个事件）&divide; verify 次数；plain decode 用每步耗时，即 1 &divide; tokens/s。"))
    sec.append(P(
        "Not removed, and recorded instead: the hardware (four GB300 in one NVL72 tray with two Grace CPUs and NVLink-C2C, "
        "SM clock 2,070 MHz, 1,400 W limit; four MI355X on one socket of an eight-GPU node, 1,400 W), the stacks (CUDA 13.2 "
        f"with SGLang main <code>ffac53d779</code> in the <code>dev-cu13</code> image; ROCm 7.2 with "
        f"<code>dsv41-amd-main@e2e824dc58</code>, the branch of {pr(39857)}, AITER <code>acf8fdf9</code> and the int64 "
        f"store-offset fix from <a href=\"{KM}/8\">kevin-mii/sglang#8</a>), the "
        "auto-selected backends, and prefill: current main disables the prefill CUDA graph for DeepSeek-V4 on CUDA, while "
        "the MI355X cell captures breakable prefill graphs.",
        "没有消除、改为记录下来的差异：硬件（四张 GB300 在一个 NVL72 托盘里，带两颗 Grace CPU 和 NVLink-C2C，SM 频率 2,070 MHz，"
        "功耗上限 1,400 W；四张 MI355X 在一台八卡机器的同一个 socket 上，1,400 W）、软件栈（CUDA 13.2，SGLang main "
        f"<code>ffac53d779</code>，镜像 <code>dev-cu13</code>；ROCm 7.2，<code>dsv41-amd-main@e2e824dc58</code>（即 {pr(39857)} 的"
        f"分支），AITER <code>acf8fdf9</code>，外加 <a href=\"{KM}/8\">kevin-mii/sglang#8</a> 的 int64 store 偏移修复）、自动选择的 "
        "backend，以及 prefill：当前 main 在 CUDA "
        "上为 DeepSeek-V4 关掉了 prefill CUDA graph，而 MI355X 的 cell 会捕获 breakable prefill graph。"))
    sec.append("</section>")

    # Calibration
    sec.append('<section id="calibration">' + H(2, "2 · Calibration: reproduce the published GB300 numbers first",
                                               "2 · 校准：先复现公开的 GB300 数字", None))
    sec.append(P(
        "A gap measured against a misconfigured baseline is fiction, so the first GPU hours went to reproducing the "
        "published series with the author's own commit (<code>835c3909</code>), image (<code>dev-dsv41</code>), launchers "
        "and client.",
        "和配置错误的 baseline 比出来的差距没有意义，所以最先的 GPU 时间都花在复现公开数据上：用作者自己的 commit"
        "（<code>835c3909</code>）、镜像（<code>dev-dsv41</code>）、启动脚本和客户端。"))
    sec.append(plate("I", "Calibration against the published GB300 series", "与公开 GB300 数据的校准",
                     plate_calibration(),
                     f"Two launches per bar, six rounds each. Unbound: {sgn(pct(s0['tp4']['tps'], BB_TP4))}%, "
                     f"{sgn(pct(s0['ep4']['tps'], BB_EP4))}%, {sgn(pct(s0['nods']['tps'], BB_OFF))}% against the published "
                     f"medians; NUMA-bound: {sgn(pct(n0['tp4']['tps'], BB_TP4))}%, {sgn(pct(n0['ep4']['tps'], BB_EP4))}%, "
                     f"{sgn(pct(n0['nods']['tps'], BB_OFF))}%.",
                     f"每根柱子两次启动、每次六轮。不绑定时相对公开中位数为 {sgn(pct(s0['tp4']['tps'], BB_TP4))}%、"
                     f"{sgn(pct(s0['ep4']['tps'], BB_EP4))}%、{sgn(pct(s0['nods']['tps'], BB_OFF))}%；NUMA 绑定时为 "
                     f"{sgn(pct(n0['tp4']['tps'], BB_TP4))}%、{sgn(pct(n0['ep4']['tps'], BB_EP4))}%、{sgn(pct(n0['nods']['tps'], BB_OFF))}%。"))
    sec.append(P(
        f"Every configuration lands within {cal_unb:.1f}% of its published median, and within {cal_bnd:.1f}% with NUMA "
        "binding. The harness is sound and the tray is not an outlier, which is what licenses the comparisons below.",
        f"每个配置都落在公开中位数的 {cal_unb:.1f}% 以内，绑定 NUMA 后在 {cal_bnd:.1f}% 以内。说明测试环境可靠、这个托盘也"
        "不是异常机器，后面的对比才站得住。"))
    sec.append("</section>")

    # Main moved
    sec.append('<section id="main-moved">' + H(2, "3 · The published target is two weeks stale", "3 · 公开数据已经过时两周", None))
    sec.append(P(
        f"On the same tray, current main (<code>ffac53d779</code>, 2026-09-23) against the author's 2026-09-11 commit: the "
        f"plain decode step fell from {f2(old_step)} to {f2(gb_step)} ms ({sgn(pct(gb_step, old_step), 0)}%), and the DSpark "
        f"verify cycle with the same EP4 layout and simulated acceptance from {f2(old_cyc)} to {f2(gb_cyc)} ms "
        f"({sgn(pct(gb_cyc, old_cyc), 0)}%). The blog's {BB_TP4:.0f} tokens/s is no longer the number to beat: main runs "
        f"{f1(n1['sim']['tps'])} tokens/s under the identical protocol.",
        f"在同一个托盘上，当前 main（<code>ffac53d779</code>，2026-09-23）对比作者 2026-09-11 的 commit：plain decode 每步从 "
        f"{f2(old_step)} ms 降到 {f2(gb_step)} ms（{sgn(pct(gb_step, old_step), 0)}%），相同 EP4 布局、模拟接受下的 DSpark verify 周期从 "
        f"{f2(old_cyc)} ms 降到 {f2(gb_cyc)} ms（{sgn(pct(gb_cyc, old_cyc), 0)}%）。博客里的 {BB_TP4:.0f} tokens/s 已经不是要追的目标：在完全"
        f"相同的流程下，main 已经跑到 {f1(n1['sim']['tps'])} tokens/s。"))
    sec.append(plate("II", "Cost per step across versions and platforms", "不同版本与平台的每步耗时",
                     plate_versions(),
                     "NUMA-bound GB300 arms; MI355X at SGLang's default placement. Old-commit rows use the author's "
                     "launchers; main rows use the cookbook cells.",
                     "GB300 取 NUMA 绑定的数据；MI355X 为 SGLang 默认放置。旧 commit 用作者的启动脚本，main 用 cookbook cell。"))
    sec.append(P(
        f"The V4.1 work that reached main in between includes the integration series {pr(39646)}, {pr(39648)}, "
        f"{pr(39653)} and {pr(38798)} (which carried the BS=1 decode, verify and communication kernels developed on the "
        f"side branch), {pr(39704)} (mHC, metadata and router overhead), "
        f"{pr(39957)} (inverse-RoPE + WO-A + MXFP8 fusion) and {pr(40431)} (FP4 indexer tile skipping). This page does "
        "not attribute the gain to individual PRs; the point is methodological: compare against current main measured "
        "on hardware you control, not against a blog figure.",
        f"这期间进入 main 的 V4.1 工作包括集成系列 {pr(39646)}、{pr(39648)}、{pr(39653)} 和 {pr(38798)}（带进了在侧分支上开发的 "
        f"BS=1 decode、verify 和通信 kernel）、"
        f"{pr(39704)}（mHC、元数据和 router 开销）、{pr(39957)}（inverse-RoPE + WO-A + MXFP8 融合）和 {pr(40431)}"
        "（FP4 indexer 跳过不可见 tile）。本文不把收益拆到单个 PR 上；要说明的是方法：要和自己能控制的硬件上测出的当前 "
        "main 比，而不是和博客里的数字比。"))
    sec.append("</section>")

    # Placement
    sec.append('<section id="placement">' + H(2, "4 · A 30% swing that was not the GPU", "4 · 与 GPU 无关的 30% 波动", None))
    sec.append(P(
        f"The first GB300 pass had one configuration that would not repeat: the simulated-acceptance arm read "
        f"{f1(s1['sim']['per_launch'][0]['tps'])} and then {f1(s1['sim']['per_launch'][1]['tps'])} tokens/s on two launches "
        f"of an identical container. A control with three launches per side confirmed it: {base_launch} tokens/s unbound "
        f"against {nice_launch} with SGLang's NUMA binding ({sgn(gain_nice)}%). The same switch moved real acceptance by "
        f"{sgn(pct(n1['real']['tps'], s1['real']['tps']))}% and plain decode by {sgn(pct(n1['off']['tps'], s1['off']['tps']))}%.",
        f"GB300 第一轮测试里有一个配置无法复现：模拟接受的 arm 在两次启动完全相同的容器时，分别得到 "
        f"{f1(s1['sim']['per_launch'][0]['tps'])} 和 {f1(s1['sim']['per_launch'][1]['tps'])} tokens/s。每边各启动三次的对照证实了"
        f"这一点：不绑定时是 {base_launch} tokens/s，打开 SGLang 的 NUMA 绑定后是 {nice_launch}（{sgn(gain_nice)}%）。同一个开关"
        f"对真实接受只改变了 {sgn(pct(n1['real']['tps'], s1['real']['tps']))}%，对 plain decode 只改变了 "
        f"{sgn(pct(n1['off']['tps'], s1['off']['tps']))}%。"))
    sec.append(plate("III", "GB300 sim rounds, unbound vs NUMA-bound; TTFT", "GB300 sim 各轮（不绑定对比 NUMA 绑定）与 TTFT",
                     plate_numa_strip(),
                     f"Each row of dots is one launch. GPU0 SM clock stayed at {', '.join(str(c) for c in sorted(clk))} MHz "
                     f"in all six control launches; its median power was {pw_b} unbound and {pw_n} bound, i.e. the GPU was "
                     "waiting, not throttling.",
                     f"每一行点是一次启动。六次对照启动里 GPU0 的 SM 频率都是 {', '.join(str(c) for c in sorted(clk))} MHz；"
                     f"功耗中位数不绑定时为 {pw_b}，绑定后为 {pw_n}，说明 GPU 是在等待，而不是降频。"))
    sec.append(P(
        "Why would CPU placement move a GPU benchmark? The four TP ranks meet at an all-reduce in every layer, so "
        "whenever work launched from the host is on the critical path, the cycle runs at the pace of the slowest "
        f"rank's CPU. Unbound, the Linux scheduler put most of TP2 and TP3 on the Grace that is not attached to their "
        f"GPUs ({min(tp23_thr):.0%}–{max(tp23_thr):.0%} of their threads local, {min(tp23_pg):.0%}–{max(tp23_pg):.0%} of "
        f"their pages local). Bound, every rank ran 100% local with {min(nice_pg):.0%}–{max(nice_pg):.0%} local pages. "
        "SGLang binds each scheduler to its GPU's node only if the process may call <code>set_mempolicy</code>; in Docker "
        "that needs <code>--cap-add SYS_NICE</code>, and without it SGLang logs a warning and runs unbound.",
        "CPU 放置为什么会影响 GPU 测试？四个 TP rank 每一层都要在 all-reduce 处会合，所以只要由 host 发起的工作处在关键路径"
        f"上，整个周期就会按最慢那个 rank 的 CPU 节奏走。不绑定时，Linux 调度器把 TP2 和 TP3 的大部分线程放到了没有连接它们 "
        f"GPU 的那颗 Grace 上（只有 {min(tp23_thr):.0%}–{max(tp23_thr):.0%} 的线程、{min(tp23_pg):.0%}–{max(tp23_pg):.0%} 的内存页在本地）。"
        f"绑定后，每个 rank 的线程 100% 在本地，内存页 {min(nice_pg):.0%}–{max(nice_pg):.0%} 在本地。SGLang 只有在进程有权限调用 "
        "<code>set_mempolicy</code> 时才会把每个 scheduler 绑到它 GPU 所在的节点；在 Docker 里这需要 <code>--cap-add SYS_NICE</code>，"
        "否则 SGLang 只打一条警告，然后不绑定运行。"))
    sec.append(P(
        "Why only sim, and why prefill? Real acceptance is resolved inside the captured verify graph. Simulated acceptance "
        "overrides it afterwards: <code>apply_dflash_simulated_acceptance</code> draws the forced length on the CPU, then "
        "launches <code>fill_</code> and <code>copy_</code> kernels and recomputes the sequence lengths eagerly, so every "
        "cycle gains a host-launched tail. Prefill is host-launched too, because main turns the breakable prefill graph off "
        f"for DeepSeek-V4 on CUDA; its 4,096-token TTFT dropped from {f0(s1['off']['ttft_ms'])} to {f0(n1['off']['ttft_ms'])} "
        "ms with binding.",
        "为什么只有 sim 受影响，为什么 prefill 也受影响？真实接受在已捕获的 verify graph 内部完成。模拟接受则在之后覆盖结果："
        "<code>apply_dflash_simulated_acceptance</code> 先在 CPU 上抽出强制接受的长度，再发射 <code>fill_</code> 和 <code>copy_</code> "
        "kernel，并以 eager 方式重新计算序列长度，所以每个周期都多出一段由 host 发起的尾巴。prefill 同样由 host 逐个发射 kernel，"
        f"因为 main 在 CUDA 上为 DeepSeek-V4 关掉了 breakable prefill graph；绑定后 4,096 token 的 TTFT 从 "
        f"{f0(s1['off']['ttft_ms'])} ms 降到 {f0(n1['off']['ttft_ms'])} ms。"))
    sec.append(plate("IV", "Where the four schedulers run", "四个 scheduler 在哪里运行", plate_topology(),
                     "Left: GB300 tray, placement measured from /proc (threads' last CPU and numa_maps). Right: the MI355X "
                     "node as the container configures it, measured the same way.",
                     "左：GB300 托盘，放置情况从 /proc 读取（线程最近运行的 CPU 和 numa_maps）。右：容器默认配置下的 MI355X 节点，"
                     "用同样的方法测得。"))
    sec.append(placement_table())
    sec.append(P(
        "The same check on MI355X found two problems. SGLang's NUMA binding is CUDA/XPU-only (it returns early on ROCm), "
        "and the container sets <code>SGLANG_SET_CPU_AFFINITY=1</code>, whose arithmetic gives rank <i>r</i> physical cores "
        "32<i>r</i> to 32<i>r</i>+31 regardless of which socket its GPU is on. Our TP4 runs on HIP devices 4–7, all on node "
        "1, so TP0 and TP1 were pinned to node 0 with their memory there too; every number in our recent MI355X PRs was "
        "measured with two remote ranks.",
        "在 MI355X 上做同样的检查，发现了两个问题。SGLang 的 NUMA 绑定只支持 CUDA/XPU（在 ROCm 上直接返回）；而容器设置了 "
        "<code>SGLANG_SET_CPU_AFFINITY=1</code>，它的算法不管 GPU 挂在哪个 socket 上，都把物理核 32<i>r</i> 到 32<i>r</i>+31 分给第 "
        "<i>r</i> 个 rank。我们的 TP4 跑在 HIP 设备 4–7 上，全部位于 node 1，于是 TP0 和 TP1 被钉在 node 0，内存也在那里；我们"
        "近期 MI355X PR 里的每个数字，都是在两个 rank 位于远端的状态下测的。"))
    sec.append(P(pe + " " + verdict_en, pz + verdict_zh))
    sec.append("</section>")

    # Acceptance
    sec.append('<section id="acceptance">' + H(2, "5 · Natural acceptance measures the prompt's attractor",
                                              "5 · 真实接受长度测的是 prompt 的吸引子", None))
    sec.append(P(
        f"With real acceptance GB300 posts {f1(n1['real']['tps'])} tokens/s, above its own simulated run, which looks like "
        f"a free win for real acceptance until you divide by the acceptance length. GB300 accepts a median "
        f"{st.median(gb_al):.2f} tokens per verify (range {min(gb_al):.2f}–{max(gb_al):.2f}) because most of its greedy "
        f"continuations fall into loops (one 16-gram repeats up to {max(gb_rep):.0f} times); MI355X accepts a median "
        f"{st.median(mi_al):.2f} (range {min(mi_al):.2f}–{max(mi_al):.2f}) with far less repetition (at most "
        f"{max(mi_rep):.0f}). Neither machine is run-to-run deterministic at temperature 0 on this prompt: identical "
        "requests return different text.",
        f"真实接受下 GB300 跑到 {f1(n1['real']['tps'])} tokens/s，比它自己的模拟接受还高，看起来像是真实接受白送的收益，但除以"
        f"接受长度就不是了。GB300 每次 verify 接受 token 数的中位数是 {st.median(gb_al):.2f}（范围 {min(gb_al):.2f}–"
        f"{max(gb_al):.2f}），因为它的 greedy 续写大多掉进了循环（某个 16-gram 最多重复 {max(gb_rep):.0f} 次）；MI355X 的中位数只有 "
        f"{st.median(mi_al):.2f}（范围 {min(mi_al):.2f}–{max(mi_al):.2f}），重复少得多（最多 {max(mi_rep):.0f} 次）。在这个 prompt 上，"
        "两台机器在 temperature 0 下都不是逐轮确定的：相同的请求会返回不同的文本。"))
    sec.append(plate("V", "Acceptance per round, and the cost of a verify", "每轮接受长度与每次 verify 的耗时",
                     plate_acceptance(),
                     "Real-acceptance rounds from every launch on each machine. The verify cost is the same with real or "
                     "simulated acceptance on both machines; only the accepted length differs.",
                     "每台机器所有启动的真实接受轮次。两台机器上，真实接受和模拟接受的每次 verify 耗时相同，不同的只是接受长度。"))
    sec.append(P(
        f"The cost of one verify does not care: GB300 spends {f2(GB_REAL_CYC)} ms per cycle with real acceptance and "
        f"{f2(gb_cyc)} with simulated; MI355X {f2(MI_REAL_CYC)} and {f2(mi_cyc)}. A random prompt has no meaning, so "
        "tiny numeric differences decide which degenerate continuation wins. Comparing tokens/s under natural acceptance "
        "on it would credit one machine for its rounding. Acceptance is worth comparing only on real text, where it "
        f"reflects the model (our real-text BS=1 contract measures {MI_REALTEXT_AL:.2f} on MI355X).",
        f"每次 verify 的耗时并不受影响：GB300 真实接受时每周期 {f2(GB_REAL_CYC)} ms，模拟接受时 {f2(gb_cyc)} ms；MI355X 分别是 "
        f"{f2(MI_REAL_CYC)} 和 {f2(mi_cyc)} ms。随机 prompt 本身没有意义，微小的数值差异就能决定模型走向哪种退化的续写。在它上面"
        "比较真实接受的 tokens/s，等于把某台机器的舍入方式算成了它的功劳。接受长度只有在真实文本上才值得比较，那时它反映的是"
        f"模型本身（我们在 MI355X 上的真实文本 BS=1 测试是 {MI_REALTEXT_AL:.2f}）。"))
    sec.append("</section>")

    # Anatomy
    sec.append('<section id="anatomy">' + H(2, "6 · Where 4.6 ms and 9 ms go", "6 · 4.6 ms 和 9 ms 各花在哪里", None))
    sec.append(P(
        "The GB300 profile captured 100 verify cycles on TP0 under nsys with CUDA-graph node tracing, then split every "
        "cycle into the categories of our MI355X ledger. Two views are needed because streams overlap on GB300: the "
        "<b>wall</b> share of each category (concurrent kernels split an interval evenly) and its <b>kernel</b> time "
        "(summed durations). The MI355X graph executes its kernels back to back, so its ledger is both views at once; it "
        f"comes from our earlier BS=1 campaign on the previous base ({MV['cycle_us'] / 1e3:.2f} ms per cycle against "
        f"{f2(mi_cyc)} ms today, genuine acceptance, profiled family shares scaled to unprofiled phase totals).",
        "GB300 的 profile 在 nsys 下以 CUDA graph 节点粒度采集了 TP0 上的 100 个 verify 周期，再把每个周期按我们 MI355X 账本"
        "的类别拆开。因为 GB300 上多个 stream 会并发，需要两种视角：每个类别的<b>墙钟</b>占比（并发的 kernel 平分同一段时间）"
        "和它的 <b>kernel</b> 时长（直接求和）。MI355X 的 graph 串行执行 kernel，所以它的账本同时就是这两种视角；这份账本来自"
        f"我们在上一个 base 上做的 BS=1 实验（每周期 {MV['cycle_us'] / 1e3:.2f} ms，今天是 {f2(mi_cyc)} ms；真实接受；按 profile "
        "得到的各族占比，缩放到未开 profiler 时实测的阶段总时长）。"))
    sec.append(plate("VI", "Anatomy of one DSpark verify cycle", "一个 DSpark verify 周期的构成", plate_anatomy(),
                     "Hatched: GPU idle. The GB300 kernel bar sums durations measured while streams overlap, so it "
                     "overstates what each category would cost alone; its excess over the wall bar is the most overlap "
                     "can be hiding.",
                     "斜线：GPU 空闲。GB300 的 kernel 条是在多 stream 重叠时测得的时长之和，会高估每一类单独运行时的开销；"
                     "它比墙钟条多出的部分，是重叠最多可能藏掉的时间。"))
    sec.append(anatomy_table())
    sec.append(P(
        f"Read the table in two passes. Along the kernel column the machines are closer than the cycle suggests: MI355X "
        f"spends {MV['kernel_sum_us'] / 1e3:.1f} ms of kernel time where GB300 spends {GV['kernel_sum_us'] / 1e3:.1f}. The cycle "
        f"differs by far more because GB300 overlaps {over:.2f}&times; and hides about {hidden / 1e3:.1f} ms. That figure is an "
        "upper bound: concurrent kernels share SMs, so each runs longer than it would alone. Plain decode shows the same "
        f"shape ({GD['overlap']:.2f}&times; overlap in a {GD['cycle_us_median'] / 1e3:.2f} ms step).",
        f"这张表要分两遍读。先看 kernel 那一列，两台机器比周期时间显示的更接近：MI355X 的 kernel 时长之和是 "
        f"{MV['kernel_sum_us'] / 1e3:.1f} ms，GB300 是 {GV['kernel_sum_us'] / 1e3:.1f} ms。周期时间差得多，是因为 GB300 有 {over:.2f} 倍的"
        f"并发，藏掉了约 {hidden / 1e3:.1f} ms。这个数是上限：并发的 kernel 共享 SM，每个都会比单独运行时更慢。plain decode 也是"
        f"同样的形状（每步 {GD['cycle_us_median'] / 1e3:.2f} ms，并发 {GD['overlap']:.2f} 倍）。"))
    sec.append(P(
        "Then the per-category gaps, which come as intervals. A kernel that shares the GPU with another stream runs "
        "longer than it would alone, so GB300's kernel column overstates each category's standalone cost, while its wall "
        "column (overlap split evenly) understates the cost of whatever was overlapped. The standalone cost lies roughly "
        "between the two, and the gap column brackets MI355X against both. Attention "
        f"({gap_txt('attention')} µs) and MoE ({gap_txt('moe')} µs) "
        f"are gaps whichever bound holds; glue kernels ({gap_txt('other')}) and the draft "
        f"({gap_txt('draft')}) follow; all-reduce plus MoE finalize is close "
        f"({gap_txt('all_reduce')}). Dense GEMM ({gap_txt('dense_gemm')}) and mHC ({gap_txt('mhc')}) overlap the most "
        "on GB300, so this profile cannot say whether MI355X is behind on them; a GB300 capture with the side streams "
        "serialized would. The host-idle row is what our follow-up to "
        f"{pr(39857)}, <a href=\"{KM}/12\">kevin-mii/sglang#12</a>, removes by keeping speculative overlap scheduling on "
        f"the forward stream (9.15 to 8.63 ms per cycle on this base); <a href=\"{KM}/13\">#13</a> additionally builds the "
        "draft-block metadata inside the draft graph.",
        "再看各类别的差距，它们是区间。和其他 stream 共享 GPU 的 kernel 会比单独运行时更慢，所以 GB300 的 kernel 那一列高估了"
        "各类别单独运行的开销；而墙钟那一列（并发时间平分）又低估了被重叠部分的开销。单独运行的真实开销大致在两者之间，差距那一列"
        f"就是 MI355X 分别对这两个边界的差。attention（{gap_txt('attention')} µs）和 MoE"
        f"（{gap_txt('moe')} µs）不论哪个边界成立都是差距；其次是胶水 kernel（{gap_txt('other')}）"
        f"和 draft（{gap_txt('draft')}）；all-reduce 加 MoE finalize 很接近"
        f"（{gap_txt('all_reduce')}）。dense GEMM（{gap_txt('dense_gemm')}）和 mHC（{gap_txt('mhc')}）在 GB300 上重叠得最多，"
        "这份 profile 判断不了 MI355X 在这两项上是否落后；把侧 stream 串行化后再采一次 GB300 就能回答。GPU 空闲那一行，正是我们在 "
        f"{pr(39857)} 之上的后续改动 <a href=\"{KM}/12\">kevin-mii/sglang#12</a> 要消掉的：它把 speculative overlap 调度留在 "
        f"forward stream 上（在这个 base 上每周期从 9.15 ms 降到 8.63 ms）；<a href=\"{KM}/13\">#13</a> 则进一步把 draft-block "
        "metadata 放进 draft graph 里构建。"))
    sec.append(P(
        "For the next MI355X ledger, the GB300 plain-decode step in the same categories (200 steps, TP0):",
        "作为下一份 MI355X 账本的对照，GB300 plain decode 每一步按同样类别的拆分（200 步，TP0）："))
    sec.append(decode_table())
    sec.append("</section>")

    # Lessons
    sec.append('<section id="lessons">' + H(2, "7 · What transfers to MI355X", "7 · 哪些可以用到 MI355X 上", None))
    sec.append(H(3, "7.1 · Concurrency is the largest single gap", "7.1 · 并发是最大的一项差距"))
    exc = {k: (GV["categories"][k]["kernel_us"] - GV["categories"][k]["wall_us"]) / 1e3 for k in ("mhc", "dense_gemm", "moe")}
    sec.append(P(
        f"Up to {hidden / 1e3:.1f} ms of the {mi_cyc - gb_cyc:.1f} ms difference per verify is kernel work GB300 runs "
        f"concurrently. The excess of kernel time over wall share says where: mHC {exc['mhc']:.2f} ms, dense GEMM "
        f"{exc['dense_gemm']:.2f} ms and MoE {exc['moe']:.2f} ms, consistent with the mHC statistics running on a side "
        "stream under the GEMMs, plus compressor and indexer work overlapped with attention preparation. "
        "On MI355X we measured why the same trick loses: a HIP graph with parallel branches leaves the packet-capture path "
        "and costs about four times as much per kernel, and a wait pending in another hardware queue slows every dispatch "
        "of the running graph by about 1.3 µs. Two ways forward, not mutually exclusive: ask the ROCm runtime for "
        "multi-stream capture that keeps packet capture (this page sizes the prize), and meanwhile turn overlap into "
        "fusion, folding side-stream work into the kernels that consume it.",
        f"每次 verify 的 {mi_cyc - gb_cyc:.1f} ms 差距里，最多有 {hidden / 1e3:.1f} ms 是 GB300 并发完成的 kernel 工作。kernel 时长比墙钟"
        f"占用多出来的部分说明了并发发生在哪里：mHC {exc['mhc']:.2f} ms，dense GEMM {exc['dense_gemm']:.2f} ms，MoE "
        f"{exc['moe']:.2f} ms，这与 mHC 统计在侧 stream 上和 GEMM 同时运行一致，另外还有和 attention 准备工作重叠执行的 compressor "
        "与 indexer。我们在 MI355X 上测过同样的做法为什么反而更慢："
        "带并行分支的 HIP graph 会退出 packet capture 路径，每个 kernel 的开销约变成四倍；另一个硬件队列里挂着的等待，会让"
        "正在运行的 graph 每次派发慢约 1.3 µs。有两条路，可以同时走：请 ROCm runtime 支持保留 packet capture 的多 stream "
        "捕获（本文量化了能拿到多少）；在那之前，把并发改成融合，把侧 stream 上的工作并进消费它的 kernel。"))
    sec.append(H(3, "7.2 · Kernel gaps, ranked by the lower bound", "7.2 · 按下界排序的 kernel 差距"))
    pa, sr, kv = (mi_attn_parts[k] for k in ("PA main", "PA split reduce + inverse RoPE", "KV norm/RoPE/store"))
    sec.append(P(
        f"Attention (at least +{slow['attention']:.0f} µs) is first. Per layer, MI355X spends {pa + sr + kv:.0f} µs in its sparse "
        f"attention chain: {pa:.1f} µs in the main pass, {sr:.1f} µs in the split reduce with inverse RoPE and {kv:.1f} µs in "
        f"the KV norm/RoPE/store. GB300's attention kernels, about {gb_attn_kernels_layer:.0f} per layer, total "
        f"{gb_attn_layer:.1f} µs. The kernel count is similar, so the difference is time per kernel, starting with the main "
        f"pass. MoE (at least +{slow['moe']:.0f} µs) is next, with about {mi_moe_layer:.0f} kernels per layer on MI355X (two GEMMs "
        f"plus routing, sorting, quantization and activation) against {gb_moe_layer:.0f} on GB300. Glue kernels "
        f"(at least +{slow['other']:.0f} µs) and the draft (at least +{slow['draft']:.0f} µs) follow. All-reduce is not where BS=1 time goes; "
        "dense GEMM waits for the serialized GB300 profile.",
        f"排第一的是 attention（至少多 {slow['attention']:.0f} µs）。MI355X 每层在稀疏 attention 链路上花 {pa + sr + kv:.0f} µs：主计算 "
        f"{pa:.1f} µs，带 inverse RoPE 的 split reduce {sr:.1f} µs，KV norm/RoPE/store {kv:.1f} µs。GB300 每层约 "
        f"{gb_attn_kernels_layer:.0f} 个 attention kernel，合计 {gb_attn_layer:.1f} µs。两边 kernel 数量差不多，差距在每个 kernel 的"
        f"耗时，首先是主计算那一步。其次是 MoE（至少多 {slow['moe']:.0f} µs），MI355X 每层约 {mi_moe_layer:.0f} 个 kernel（两个 GEMM，"
        f"加上路由、排序、量化和激活），GB300 是 {gb_moe_layer:.0f} 个。再后面是胶水 kernel（至少多 {slow['other']:.0f} µs）和 draft"
        f"（至少多 {slow['draft']:.0f} µs）。all-reduce 不是 BS=1 时间的去处；dense GEMM 要等串行化的 GB300 profile 出来再判断。"))
    sec.append(H(3, "7.3 · Keep what MI355X already does better", "7.3 · 保留 MI355X 已经做得更好的部分"))
    sec.append(P(
        f"MI355X answers the 4,096-token prefill in {f0(mu['off']['ttft_ms'])} ms against GB300's {f0(n1['off']['ttft_ms'])} ms "
        f"bound and {f0(s1['off']['ttft_ms'])} ms unbound, because the MI350X cell captures breakable prefill graphs while "
        "CUDA main disables them for this model. That advantage should survive any port of NVIDIA-side changes: a "
        "change that needs eager prefill to work is a regression on this machine even if its decode numbers improve.",
        f"MI355X 完成 4,096 token prefill 只要 {f0(mu['off']['ttft_ms'])} ms，GB300 绑定时要 {f0(n1['off']['ttft_ms'])} ms，不绑定时 "
        f"{f0(s1['off']['ttft_ms'])} ms，原因是 MI350X 的 cell 捕获了 breakable prefill graph，而 CUDA 版 main 对这个模型关掉了它。"
        "移植 NVIDIA 侧改动时要保住这个优势：如果某个改动必须靠 eager prefill 才能工作，那么即使 decode 数字变好，在这台机器上也是"
        "一次回退。"))
    sec.append(H(3, "7.4 · Placement and measurement rules", "7.4 · 放置与测量规则"))
    sec.append(UL([
        ("Record where every scheduler runs (<code>Cpus_allowed_list</code>, per-thread CPU, <code>numa_maps</code>) in "
         "every arm; bind to the GPU's NUMA node, and on MI355X fix <code>SGLANG_SET_CPU_AFFINITY</code> to read the "
         "node from the PCI device instead of multiplying the local rank.",
         "每个 arm 都记录每个 scheduler 在哪里运行（<code>Cpus_allowed_list</code>、每个线程的 CPU、<code>numa_maps</code>）；"
         "绑到 GPU 所在的 NUMA 节点；在 MI355X 上把 <code>SGLANG_SET_CPU_AFFINITY</code> 改成从 PCI 设备读取节点，而不是用本地 "
         "rank 去乘。"),
        ("Use simulated acceptance only as a controlled experiment and bind NUMA when you do: it adds host-launched work "
         "that real acceptance does not have.",
         "模拟接受只在受控实验里用，用的时候一定要绑 NUMA：它会引入真实接受所没有的、由 host 发起的工作。"),
        ("Compare cost per verify across platforms, and acceptance only on real text.",
         "跨平台比较每次 verify 的耗时；接受长度只在真实文本上比较。"),
        (f"At least three fresh launches per configuration: the first two launches of the sim arm differed by "
         f"{two_launch_spread:.0f}% and their pooled median still looked like a plausible number.",
         f"每个配置至少启动三次新的 server：sim arm 的前两次启动相差 {two_launch_spread:.0f}%，合并后的中位数看起来却仍像一个"
         "合理的数字。"),
        ("Calibrate against the published number with its own commit, then move to current main.",
         "先用公开数字自己的 commit 校准，再换到当前 main。"),
    ]))
    sec.append("</section>")

    # Next
    sec.append('<section id="next">' + H(2, "8 · Next experiments", "8 · 下一步实验", None))
    sec.append(UL([
        ("Rebuild the MI355X ledger on the current base with a kernel trace (not a torch profile, which inflates graph "
         "kernels to about 4 µs each), in the categories used here, for both the verify cycle and the plain decode step.",
         "在当前 base 上用 kernel trace 重建 MI355X 的账本（不用 torch profiler，它会把 graph 里的每个 kernel 拉长到约 4 µs），"
         "按本文的类别拆分 verify 周期和 plain decode 的每一步。"),
        ("Capture GB300 again with the side streams serialized, to replace the dense GEMM and mHC intervals by numbers.",
         "在侧 stream 串行化的条件下再采一次 GB300，把 dense GEMM 和 mHC 的区间变成确定的数字。"),
        ("Microbenchmark multi-stream HIP graph capture against CUDA on the same kernel mix, to turn &ldquo;up to "
         f"{hidden / 1e3:.1f} ms&rdquo; into a runtime request with numbers.",
         f"在相同的 kernel 组合上，对比 HIP 和 CUDA 的多 stream graph 捕获，把&ldquo;最多 {hidden / 1e3:.1f} ms&rdquo;变成一个带数据的 "
         "runtime 需求。"),
        ("Fuse the attention chain and the MoE routing/sort/quant kernels on MI355X, one change at a time, each with "
         "the BS=1 A/B used here.",
         "在 MI355X 上融合 attention 链路和 MoE 的路由、排序、量化 kernel，每次只改一处，每次都用本文的 BS=1 A/B 验证。"),
        ("Run the real-text BS=1 contract on GB300, to compare acceptance where it is a model property.",
         "在 GB300 上跑真实文本的 BS=1 测试，在接受长度能代表模型本身的场景下比较它。"),
    ]))
    sec.append("</section>")

    # Appendix
    sec.append('<section id="appendix">' + H(2, "A · Every arm", "A · 所有 arm", None))
    sec.append(P(
        "Every group recomputed from its per-round measurements (warm-up excluded, launches pooled). Cycle cost for "
        "DSpark arms, step cost for plain decode.",
        "每一组都从逐轮测量重新计算（去掉热身，合并各次启动）。DSpark arm 给出每周期耗时，plain decode 给出每步耗时。"))
    sec.append(arms_table())
    sec.append("</section>")

    # Reproduce
    sec.append('<section id="reproduce">' + H(2, "9 · Reproduce", "9 · 复现", None))
    sec.append(P(
        f"All inputs, per-round CSVs, placements, profiles, scripts and the analysis are in <a href=\"{DATA}\">"
        f"<code>data/{SLUG}/</code></a>, with hostnames and machine paths replaced by placeholders. The GB300 arms were "
        "driven by <code>gb300/scripts/arm.py</code> (one container per launch, idle check, nvidia-smi before and after); "
        "the MI355X arms by <code>mi355x/scripts/run_arm*.sh</code> under an exclusive eight-GPU lease with HBM canaries.",
        f"所有输入、逐轮 CSV、进程放置记录、profile、脚本和分析都在 <a href=\"{DATA}\"><code>data/{SLUG}/</code></a>，主机名和机器"
        "路径已替换成占位符。GB300 的 arm 由 <code>gb300/scripts/arm.py</code> 驱动（每次启动一个容器，先做空闲检查，前后各记录"
        "一次 nvidia-smi）；MI355X 的 arm 由 <code>mi355x/scripts/run_arm*.sh</code> 在独占八卡的租约下运行，前后做 HBM canary。"))
    b4, a4 = CAN[4]
    s12 = max(abs(pct(S2[("split", m)], S1_SPLIT[m])) for m in ("sim", "real")) if HAVE_S2 else 0.0
    sec.append(P(
        f"One lease needs a note. After session 2 the canary on GPU 4 read {a4:.2f} ms per copy against {b4:.2f} ms "
        f"before it, {pct(b4, a4):.1f}% faster, which crosses the lease tool's 10% threshold, so the tool flagged the "
        "session. We kept it: every process the monitor saw on the node's GPUs appeared and disappeared inside one of "
        f"our server launches, the other three canaries moved by under {up1(CAN_OTHER)}%, a busier device can only "
        f"lengthen a copy, and session 2's split-affinity cycles match session 1's within {up1(s12)}%. The lease "
        "records are in <code>mi355x/leases/</code>.",
        f"有一次租约需要说明。session 2 结束后，GPU 4 上的 canary 每次拷贝耗时 {a4:.2f} ms，开始前是 {b4:.2f} ms，快了 "
        f"{pct(b4, a4):.1f}%，超过了租约工具 10% 的阈值，所以工具把这个 session 标记为可疑。我们保留了它：监控看到的每个"
        f"使用这台机器 GPU 的进程，都在我们某一次 server 启动的时间窗内出现和消失；另外三张卡的 canary 变化都在 "
        f"{up1(CAN_OTHER)}% 以内；设备被占用只会让拷贝变慢；session 2 切分亲和性的周期和 session 1 相差不到 "
        f"{up1(s12)}%。租约记录在 <code>mi355x/leases/</code>。"))
    sec.append('<pre><code># GB300, one launch of the Low-Latency cell with simulated acceptance, NUMA binding on\n'
               'docker run -d --init --gpus all --ipc=host --shm-size 32g --network host --cap-add=SYS_NICE \\\n'
               '  -e SGLANG_RAGGED_VERIFY_MODE=static -e SGLANG_SIMULATE_ACC_LEN=5.5 -e SGLANG_SIMULATE_ACC_METHOD=match-expected \\\n'
               '  -v $MODEL_DIR:/models/DeepSeek-V4.1-Flash:ro lmsysorg/sglang@sha256:b8257f5c5f8f7c5a... \\\n'
               '  sglang serve --model-path /models/DeepSeek-V4.1-Flash --trust-remote-code --tp 4 --ep-size 4 \\\n'
               '    --mem-fraction-static 0.8 --speculative-algorithm DSPARK --speculative-dspark-block-size 5 \\\n'
               '    --cuda-graph-max-bs-decode 64 --disable-radix-cache --random-seed 42 --port 30000\n'
               'python3 benchmark.py bench --prompt prompt.json --max-tokens 1024 --repeat 6 --out result \\\n'
               '  --url http://127.0.0.1:30000\n\n'
               '# MI355X, the same client against the MI350X cell (HIP 4-7), placement mode split|node1|none\n'
               'bash lease.sh session2 -- bash session2.sh      # calls run_arm2.sh LABEL MODE AFF per arm\n\n'
               '# analysis\n'
               'python3 analysis/analyze.py && python3 analysis/compare_attribution.py\n'
               '</code></pre>\n')
    sec.append("</section>")

    # Epilogue
    sec.append('<section id="epilogue">' + H(2, "Epilogue", "后记", None))
    sec.append(P(
        "The GB300 run was supposed to give us a target. It gave us three things instead: a factor of two per step, "
        "split between stream concurrency and four kernel families; a reminder that a GPU benchmark can be a CPU "
        "benchmark in disguise; and a placement bug on our own machine that no GPU counter would have shown.",
        "这次 GB300 实验本来是想拿到一个目标值，结果得到了三样东西：一个每步 2 倍的差距，它分布在多 stream 并发和四类 kernel 上；"
        "一个提醒，GPU 测试有时其实是在测 CPU；以及我们自己机器上的一个放置 bug，任何 GPU 计数器都看不出来。"))
    sec.append("</section>")
    return "\n".join(sec)


CSS = r"""
:root{--bg:#0a0e0f;--panel:#10171a;--grid:#1c2729;--ink:#e4ece7;--muted:#8aa19c;--teal:#35d0c0;--mag:#ff5fa2;
--lime:#c9f25c;--amber:#ffb454;--display:'Unbounded','Noto Sans SC',sans-serif;--body:'Onest','Noto Sans SC',sans-serif;
--mono:'Chivo Mono','Noto Sans SC',monospace}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--body);font-size:17px;line-height:1.68;
background-image:linear-gradient(var(--grid) 1px,transparent 1px),linear-gradient(90deg,var(--grid) 1px,transparent 1px);
background-size:48px 48px;background-attachment:fixed}
a{color:var(--teal);text-decoration:none;border-bottom:1px solid #35d0c055}a:hover{border-bottom-color:var(--teal)}
code{font-family:var(--mono);font-size:.86em;background:#16211f;padding:.08em .35em;border-radius:3px;color:#d7f7ee}
pre{background:#0d1416;border:1px solid var(--grid);padding:16px 18px;overflow-x:auto;border-radius:4px;line-height:1.5}
pre code{background:none;padding:0;font-size:13px;color:#cfe3dd}
.layout{display:grid;grid-template-columns:230px minmax(0,1fr);gap:40px;max-width:1240px;margin:0 auto;padding:0 28px}
.layout>*{min-width:0}p code,li code{overflow-wrap:anywhere}
nav.rail{position:sticky;top:0;align-self:start;height:100vh;overflow-y:auto;padding:34px 0 24px;font-family:var(--mono);
font-size:12.5px;letter-spacing:.02em}
nav.rail .back{display:block;margin-bottom:26px;color:var(--lime);border:none}
nav.rail ol{list-style:none;margin:0;padding:0;border-left:1px solid var(--grid)}
nav.rail li a{display:block;padding:6px 0 6px 14px;color:var(--muted);border:none}
nav.rail li a:hover{color:var(--ink)}
header.masthead{padding:56px 0 30px;max-width:980px}
.kicker{font-family:var(--mono);font-size:13px;letter-spacing:.18em;text-transform:uppercase;color:var(--lime)}
h1{font-family:var(--display);font-weight:800;font-size:clamp(30px,4.4vw,52px);line-height:1.08;margin:14px 0 18px;
letter-spacing:-.01em}
h1 .em{color:var(--teal)}h1 .em2{color:var(--mag)}
.subtitle{font-size:19px;color:#c4d3ce;max-width:760px}
.spec{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1px;margin-top:30px;background:var(--grid);
border:1px solid var(--grid)}
.spec div{background:var(--panel);padding:12px 14px;font-size:13.5px}
.spec b{display:block;font-family:var(--mono);font-weight:500;font-size:11px;letter-spacing:.12em;text-transform:uppercase;
color:var(--muted);margin-bottom:4px}
main.article section{max-width:720px;padding:8px 0 26px}
h2{font-family:var(--display);font-weight:600;font-size:25px;line-height:1.25;margin:46px 0 14px;color:#f4faf6}
h3{font-family:var(--display);font-weight:400;font-size:18px;margin:30px 0 8px;color:var(--lime)}
p{margin:0 0 16px}ul{padding-left:20px;margin:0 0 18px}li{margin:0 0 9px}
.callout{border-left:3px solid var(--lime);background:#121b17;padding:14px 18px 4px;margin:22px 0}
.callout-title{font-family:var(--mono);font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--lime);
margin-bottom:8px}
figure.plate{margin:30px 0 34px;background:var(--panel);border:1px solid var(--grid);padding:16px 18px 12px}
figure.plate.wide{width:min(1040px,calc(100vw - 330px));max-width:none}
.plate-meta{display:flex;gap:14px;align-items:baseline;font-family:var(--mono);font-size:12px;letter-spacing:.1em;
text-transform:uppercase;margin-bottom:10px}
.plate-num{color:var(--amber)}.plate-title{color:var(--ink)}
figure svg{width:100%;height:auto;display:block}
.caption{font-style:italic;color:var(--muted);font-size:14px;margin-top:8px}
svg .grid{stroke:#1f2c2e;stroke-width:1}svg .lbl{fill:#dbe7e2;font:13px var(--body)}
svg .tick{fill:var(--muted);font:11px var(--mono)}svg .axis{fill:var(--muted);font:11.5px var(--mono)}
svg .val{fill:#f1f7f3;font:11.5px var(--mono)}svg .legend{fill:#c9d8d3;font:12px var(--body)}
svg .head{fill:var(--amber);font:600 11.5px var(--mono);letter-spacing:.08em}
svg .note{fill:#9fb4af;font:12px var(--body)}
svg .box{fill:#131d1f;stroke:#35514d;stroke-width:1}svg .box.gpu{fill:#172422}
svg .edge{stroke:#4b6a65;stroke-width:1.5}
.tablewrap{overflow-x:auto;margin:10px 0 22px}
table.data{border-collapse:collapse;width:100%;font-size:14px}
table.data th,table.data td{border-bottom:1px solid var(--grid);padding:7px 10px;text-align:right;font-family:var(--mono)}
table.data th:first-child,table.data td:first-child{text-align:left;font-family:var(--body)}
table.data th{color:var(--muted);font-weight:500;font-size:12px;letter-spacing:.06em}
table.data tr.total td{color:var(--lime);border-top:1px solid #35514d}
table.data td{white-space:nowrap}table.data td:first-child{white-space:normal}
table.wide-table{font-size:12px}table.wide-table td:nth-child(2){text-align:left;font-family:var(--body);
white-space:normal;min-width:200px}table.wide-table td:first-child{white-space:nowrap}
table.wide-table td:last-child{white-space:normal;min-width:90px}
table.data tr.group td{text-align:left;font-family:var(--mono);color:var(--amber);font-size:11.5px;letter-spacing:.1em;
padding-top:16px}p.tnote{font-size:13.5px;color:var(--muted)}
main.article section#appendix{max-width:1040px}
footer.colophon{max-width:980px;margin:40px 0 70px;padding-top:20px;border-top:1px solid var(--grid);font-size:13px;
color:var(--muted);display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px}
footer.colophon b{display:block;font-family:var(--mono);font-size:11px;letter-spacing:.12em;text-transform:uppercase;
color:#b9cbc6}
.lang-toggle{position:fixed;top:16px;right:18px;z-index:10;display:flex;border:1px solid #35514d;background:#0d1416;
border-radius:999px;overflow:hidden}
.lang-toggle button{font:12px var(--mono);letter-spacing:.08em;color:var(--muted);background:none;border:none;
padding:7px 13px;cursor:pointer}
body[data-lang="en"] .lang-toggle button[data-set="en"],body[data-lang="zh"] .lang-toggle button[data-set="zh"]{
background:var(--lime);color:#0a0e0f}
@media (max-width:980px){.layout{grid-template-columns:1fr;padding:0 18px}nav.rail{position:static;height:auto;
padding:70px 0 0}nav.rail ol{display:flex;flex-wrap:wrap;gap:4px 14px;border:none}nav.rail li a{padding:2px 0}
figure.plate.wide{width:auto}}
body[data-lang="en"] [lang="zh"]:not(html){display:none !important}
body[data-lang="zh"] [lang="en"]:not(html){display:none !important}
"""

TOC = [("prologue", "Prologue", "序"), ("contract", "1 · Held fixed", "1 · 固定条件"),
       ("calibration", "2 · Calibration", "2 · 校准"), ("main-moved", "3 · Main moved", "3 · main 的变化"),
       ("placement", "4 · Placement", "4 · 放置"), ("acceptance", "5 · Acceptance", "5 · 接受长度"),
       ("anatomy", "6 · Anatomy", "6 · 周期构成"), ("lessons", "7 · Lessons", "7 · 可借鉴的"),
       ("next", "8 · Next", "8 · 下一步"), ("appendix", "A · Every arm", "A · 所有 arm"),
       ("reproduce", "9 · Reproduce", "9 · 复现"), ("epilogue", "Epilogue", "后记")]


def page():
    body = build()
    rail = ('<nav class="rail"><a class="back" href="/sources/">' + S("← The Library", "← 文库") + "</a><ol>"
            + "".join(f'<li><a href="#{i}">{S(e, z)}</a></li>' for i, e, z in TOC) + "</ol></nav>")
    spec = [("model", "DeepSeek-V4.1 Flash @ dba1be0a", "DeepSeek-V4.1 Flash @ dba1be0a"),
            ("request", "BS=1 · 4,096 random ids · 1,024 out", "BS=1 · 4,096 个随机 id · 输出 1,024"),
            ("parallelism", "TP4 / EP4 · cookbook cells", "TP4 / EP4 · cookbook cell"),
            ("GB300", "4× GB300 (1 NVL72 tray) · CUDA 13.2 · main ffac53d779", "4× GB300（1 个 NVL72 托盘）· CUDA 13.2 · main ffac53d779"),
            ("MI355X", "4× MI355X · ROCm 7.2 · dsv41-amd-main e2e824dc58", "4× MI355X · ROCm 7.2 · dsv41-amd-main e2e824dc58"),
            ("runs", f"{sum(a['launches'] for a in ARMS.values())} launches · 2 nsys profiles · 2026-09-24/25",
             f"{sum(a['launches'] for a in ARMS.values())} 次启动 · 2 个 nsys profile · 2026-09-24/25")]
    spec_html = "".join(f"<div><b>{k}</b>{S(e, z)}</div>" for k, e, z in spec)
    mast = ('<header class="masthead"><div class="kicker">Experiment 005 · '
            + S("DeepSeek-V4.1 Flash on SGLang", "SGLang 上的 DeepSeek-V4.1 Flash") + "</div>"
            '<h1><span class="em">GB300</span> vs <span class="em2">MI355X</span>, one token at a time</h1>'
            + P(f"We reproduced SGLang's GB300 numbers, profiled where each {f2(gb_cyc)} ms verify cycle goes, and replayed the "
                f"same prompt on MI355X ({f2(mi_cyc)} ms). Up to {GV_HIDDEN / GAP_US:.0%} of the gap is concurrency the ROCm "
                "graph path cannot exploit today, most of the rest is four kernel families, and a "
                f"{pct(e1n['tps'], e1b['tps']):.0f}% swing on GB300 turned out to be CPU placement.",
                f"我们复现了 SGLang 的 GB300 数据，拆解了每个 {f2(gb_cyc)} ms 的 verify 周期花在哪里，并在 MI355X 上重放了同一个 "
                f"prompt（{f2(mi_cyc)} ms）。差距里最多 {GV_HIDDEN / GAP_US:.0%} 是 ROCm 的 graph 路径目前用不上的并发，其余大部分来自"
                f"四类 kernel；而 GB300 上 {pct(e1n['tps'], e1b['tps']):.0f}% 的波动，最后查明是 CPU 放置造成的。", "subtitle")
            + f'<div class="spec">{spec_html}</div></header>')
    colophon = ('<footer class="colophon">'
                f"<div><b>{S('data', '数据')}</b><a href=\"{DATA}\">data/{SLUG}/</a></div>"
                f"<div><b>{S('protocol', '测试流程')}</b>{S('BBuf random-dspark (b001f347)', 'BBuf random-dspark（b001f347）')}</div>"
                f"<div><b>{S('type', '字体')}</b>Unbounded · Onest · Chivo Mono · Noto Sans SC</div>"
                f"<div><b>{S('palette', '配色')}</b>{S('phosphor black, teal (GB300), magenta (MI355X), lime (placement)', '荧光黑，青色（GB300），洋红（MI355X），柠檬绿（放置）')}</div>"
                "</footer>")
    script = """<script>
(function(){var KEY='dsv41-gb300-vs-mi355x-source-lang',body=document.body,stored=null;
try{stored=localStorage.getItem(KEY);}catch(e){}
if(stored==='en'||stored==='zh'){body.setAttribute('data-lang',stored);}
else{var nav=(navigator.language||'en').toLowerCase();body.setAttribute('data-lang',nav.indexOf('zh')===0?'zh':'en');}
document.querySelectorAll('.lang-toggle button[data-set]').forEach(function(btn){btn.addEventListener('click',function(){
var v=btn.getAttribute('data-set');body.setAttribute('data-lang',v);try{localStorage.setItem(KEY,v);}catch(e){}});});})();
</script>"""
    defs = ('<svg width="0" height="0" style="position:absolute"><defs><pattern id="hatch" width="6" height="6" '
            'patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="6" '
            'stroke="#8aa19c" stroke-width="1.4"/></pattern></defs></svg>')
    return ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" "
            "content=\"width=device-width,initial-scale=1\"><title>Experiment 005 · DeepSeek-V4.1 Flash, GB300 vs MI355X"
            "</title><meta name=\"description\" content=\"BS=1 DeepSeek-V4.1 Flash on SGLang: GB300 calibration, NUMA "
            "placement, acceptance traps and a per-category verify-cycle comparison with MI355X.\">"
            "<link rel=\"preconnect\" href=\"https://fonts.googleapis.com\"><link rel=\"preconnect\" "
            "href=\"https://fonts.gstatic.com\" crossorigin><link href=\"https://fonts.googleapis.com/css2?family=Unbounded:"
            "wght@400;600;800&family=Onest:wght@400;500;600&family=Chivo+Mono:wght@400;500&family=Noto+Sans+SC:wght@400;500;"
            "700&display=swap\" rel=\"stylesheet\"><style>" + CSS + "</style></head>\n"
            "<body data-lang=\"en\"><div class=\"lang-toggle\" role=\"group\" aria-label=\"Language\"><button type=\"button\" "
            "data-set=\"en\" aria-label=\"English\">EN</button><button type=\"button\" data-set=\"zh\" aria-label=\"中文\">"
            "中文</button></div>" + defs + "<div class=\"layout\">" + rail + "<div>" + mast + "<main class=\"article\">"
            + body + "</main>" + colophon + "</div></div>" + script + "</body></html>\n")


def main():
    doc = page()
    OUT.write_text(doc)
    print(f"wrote {OUT} ({len(doc) / 1e3:.0f} KB, {doc.count(chr(10))} lines)")


if __name__ == "__main__":
    main()
