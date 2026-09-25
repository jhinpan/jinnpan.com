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


LAYERS = 40
LEDGER = {(r["phase"], r["segment"], r["family"]): r for r in
          csv.DictReader(Path("$BS1_FLOOR/analysis/kernel-ledger.csv").open())}
mi_attn_parts = {f: float(LEDGER[("VERIFY", "layers", f)]["est_unprofiled_us"]) / LAYERS
                 for f in ("PA main", "PA split reduce + inverse RoPE", "KV norm/RoPE/store")}
mi_moe_layer = kern["moe"][3] / LAYERS
cal_unb = max(abs(pct(s0[k]["tps"], b)) for k, b in (("tp4", BB_TP4), ("ep4", BB_EP4), ("nods", BB_OFF)))
cal_bnd = max(abs(pct(n0[k]["tps"], b)) for k, b in (("tp4", BB_TP4), ("ep4", BB_EP4), ("nods", BB_OFF)))
two_launch_spread = pct(max(p["tps"] for p in s1["sim"]["per_launch"]), min(p["tps"] for p in s1["sim"]["per_launch"]))
gap_ms = mi_cyc - gb_cyc
GAP_US = gap_ms * 1e3
GV_HIDDEN = GV["kernel_sum_us"] - GV["busy_us"]
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

# Session 3: GB300 stream A/B (same session, mirrored order), serialized attribution,
# real-text contract on GB300, and the graph-floor microbenchmark on both machines.
W3 = {(v, m): A("gb300", f"w3-{v}-{m}") for v in ("base", "opt0", "serial") for m in ("sim", "off")}
w3_cyc = {v: W3[(v, "sim")]["cycle_ms"] for v in ("base", "opt0", "serial")}
w3_step = {v: W3[(v, "off")]["step_ms"] for v in ("base", "opt0", "serial")}
CONC_MS = w3_cyc["serial"] - w3_cyc["base"]
CONC_OPT_MS = w3_cyc["opt0"] - w3_cyc["base"]
CONC_MOE_MS = w3_cyc["serial"] - w3_cyc["opt0"]
SERIAL_GAP_MS = mi_cyc - w3_cyc["serial"]
GS, GDS = ATT["gb300_verify_serial"], ATT["gb300_decode_serial"]
GS_SUB = {(x["category"], x["sub"]): x for x in
          json.loads((CAMP / "gb300/results/p3-serial-real/attribution.json").read_text())["subcategories"]}
gs_attn_layer = GS_SUB[("attention", "attention")]["kernel_us"] / LAYERS
gs_moe_layer = GS["categories"]["moe"]["kernels"] / LAYERS
KT = json.loads((CAMP / "gb300/results/p3-serial-real/kernel_table.json").read_text())["kernels"]


def kt(*prefixes):
    """GB300 single-stream us per layer and calls per layer for kernels whose names start with a prefix."""
    sel = [x for x in KT if x["kernel"].startswith(prefixes)]
    return sum(x["us_per_cycle"] for x in sel) / LAYERS, sum(x["calls_per_cycle"] for x in sel) / LAYERS


def led(*families):
    """MI355X previous-base us per layer and calls per layer (verify layers segment of the ledger)."""
    rows = [LEDGER[("VERIFY", "layers", f)] for f in families]
    return (sum(float(r["est_unprofiled_us"]) for r in rows) / LAYERS,
            sum(float(r["calls_per_cycle"]) for r in rows) / LAYERS)
RT = json.loads((CAMP / "analysis/realtext.json").read_text())
MB = {"cuda": json.loads((CAMP / "gb300/results/mb-cuda/default.json").read_text()),
      "hip": json.loads((CAMP / "results/mb-hip/default.json").read_text())}


def mb_fit(p, case):
    return MB[p]["cases"][case]["fit"]["per_kernel_us"]


def mb_row(p, case, n):
    return MB[p]["cases"][case]["rows"][str(n)]["device_median_us"]


TRACE = json.loads((CAMP / "analysis/mi355x_trace_real.json").read_text())
MI_KERNELS_NOW = TRACE["kernels_per_cycle"]
FLOOR = {p: mb_fit(p, "triton_chain_p1") for p in MB}
LADDER = {p: mb_row(p, "fork_join_ladder_p1", 128) / (3 * 128) for p in MB}
BRANCH = {p: mb_row(p, "two_branch_p1", 128) / 128 for p in MB}
GEMV = {p: mb_row(p, "gemv_6x5120x1152_bf16", 40) / 40 for p in MB}


# ---------------------------------------------------------------- bilingual helpers
_ZH_PUNCT = re.compile(r"([。，：])(?=[^\s<）」』】])|([。，：])(?=<(?!/))")


def zh(text):
    """Site rule (CLAUDE.md): a half-width space after 。，： when content follows."""
    return _ZH_PUNCT.sub(lambda m: (m.group(1) or m.group(2)) + " ", text)


def P(en, zh_, cls=""):
    c = f' class="{cls}"' if cls else ""
    return f'<p lang="en"{c}>{en}</p>\n<p lang="zh"{c}>{zh(zh_)}</p>\n'


_SECNO = re.compile(r"^([0-9]+(?:\.[0-9]+)?|[A-Z]) · (.+)$", re.S)


def _secno(text):
    m = _SECNO.match(text)
    return f'<span class="no">{m.group(1)}</span>{m.group(2)}' if m else text


def H(level, en, zh_, id_=None):
    i = f' id="{id_}"' if id_ else ""
    return f'<h{level}{i}><span lang="en">{_secno(en)}</span><span lang="zh">{_secno(zh(zh_))}</span></h{level}>\n'


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
# Platform hues are fixed page-wide (spruce = GB300, persimmon = MI355X); placement uses fern
# for local and ochre for remote, always paired with solid/dashed or filled/hollow marks.
GB, MI, MI_INK = "#2b6b64", "#c8562a", "#a8431d"
LOCAL, LOCAL_INK, REMOTE, REMOTE_INK = "#5e8d3e", "#4a7430", "#c4861c", "#8f5f0e"
PAPER, INK, MUTED = "#fffcf7", "#2e2822", "#7a6d5e"
CAT_COLOR = {"moe": "#e0b45c", "dense_gemm": "#6f93ba", "attention": "#93b36b", "mhc": "#a68bc2",
             "all_reduce": "#dc8f7c", "engram": "#8cc5b7", "draft": "#c48aa8", "other": "#c2b8a8",
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
    for i, (fill, tag) in enumerate((("none", "published (README)"), (GB + "80", "ours · unbound"),
                                     (GB, "ours · NUMA-bound"))):
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
        for i, (v, fill) in enumerate(((pub, "none"), (unb, GB + "80"), (bnd, GB))):
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
    rows = [("Plain decode step", [("GB300 · 835c3909", old_step, GB + "70"), ("GB300 · main", gb_step, GB),
                                   ("MI355X · e2e824dc58", mi_step, MI)]),
            ("DSpark verify cycle", [("GB300 · 835c3909 (EP4)", old_cyc, GB + "70"),
                                     ("GB300 · main", gb_cyc, GB), ("MI355X · e2e824dc58", mi_cyc, MI)])]
    s = [svg_open(W, H0, "Per-step cost across versions and platforms")]
    for v in range(0, 11, 2):
        s.append(f'<line x1="{X(v)}" y1="26" x2="{X(v)}" y2="{H0 - 38}" class="grid"/>')
        s.append(t(X(v), H0 - 20, f"{v}", "tick", "middle"))
    s.append(t(x1, H0 - 4, "milliseconds per step (lower is faster; NUMA-bound where available)", "axis", "end"))
    y = 34
    for head, bars in rows:
        s.append(t(24, y + 10, head, "head"))
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
    lanes = [("unbound", arm_rounds("gb300", "s1a-sim") + arm_rounds("gb300", "e1-base-sim"), None),
             ("NUMA-bound", arm_rounds("gb300", "n1a-sim") + arm_rounds("gb300", "e1-nice-sim"), LOCAL)]
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
            mark = (f'fill="{c}"' if c else f'fill="{PAPER}" stroke="{REMOTE}" stroke-width="1.6"')
            s.append(f'<circle cx="{X(r["tps"]):.1f}" cy="{yc - 24 + j * 12:.1f}" r="4.2" {mark}/>')
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
        s.append(f'<rect x="{bx0}" y="{y + 8}" width="{BX(u) - bx0:.1f}" height="11" fill="{REMOTE}"/>')
        s.append(t(BX(u) + 4, y + 17.5, f"{u:.0f}", "val"))
        s.append(f'<rect x="{bx0}" y="{y + 22}" width="{BX(b) - bx0:.1f}" height="11" fill="{LOCAL}"/>')
        s.append(t(BX(b) + 4, y + 31.5, f"{b:.0f}", "val"))
    s.append(t(bx0, 232, "unbound", "legend", extra=f' fill="{REMOTE_INK}"'))
    s.append(t(bx0 + 70, 232, "bound", "legend", extra=f' fill="{LOCAL_INK}"'))
    s.append("</svg>")
    return "".join(s)


def plate_topology():
    W, H0 = 960, 380
    s = [svg_open(W, H0, "Where the four TP schedulers run on each machine")]
    # GB300 tray
    s.append(t(20, 28, "GB300 tray · 2 × Grace (72 cores each) · NVLink-C2C", "head"))
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
                        f"their pages {min(tp23_pg):.0%}–{max(tp23_pg):.0%}", "note", extra=f' fill="{REMOTE_INK}"'))
    s.append(t(20, 240, f"bound (--cap-add SYS_NICE): every rank 100% local threads, "
                        f"{min(nice_pg):.0%}–{max(nice_pg):.0%} local pages", "note", extra=f' fill="{LOCAL_INK}"'))
    # MI355X node
    ox = 500
    s.append(t(ox, 28, "MI355X node · 2 sockets · GPUs 4–7 (our TP4) on node 1", "head"))
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
        link = (f'stroke="{LOCAL}" stroke-width="2"' if local
                else f'stroke="{REMOTE}" stroke-width="2" stroke-dasharray="6 4"')
        s.append(f'<rect x="{gx}" y="150" width="100" height="40" class="box gpu"/>')
        s.append(t(gx + 50, 168, f"GPU{4 + j} · {tp}", "lbl", "middle"))
        s.append(t(gx + 50, 183, cpus, "note", "middle"))
        target = ox + 105 if j < 2 else ox + 335
        s.append(f'<line x1="{gx + 50}" y1="150" x2="{target}" y2="94" {link}/>')
    s.append(t(ox, 222, "SGLANG_SET_CPU_AFFINITY=1: rank r gets physical cores [32r, 32r+32)", "note",
               extra=f' fill="{REMOTE_INK}"'))
    s.append(t(ox, 240, "→ TP0/TP1 are pinned to node 0, remote from GPUs 4–7", "note", extra=f' fill="{REMOTE_INK}"'))
    s.append(t(ox, 258, "SGLang's own NUMA binding is CUDA/XPU-only; numactl is not installed", "note"))
    # legend
    s.append(f'<line x1="20" y1="300" x2="44" y2="300" stroke="{LOCAL}" stroke-width="2"/>')
    s.append(t(50, 304, "local (same NUMA node as the GPU)", "legend"))
    s.append(f'<line x1="300" y1="300" x2="324" y2="300" stroke="{REMOTE}" stroke-width="2" stroke-dasharray="6 4"/>')
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
    for i, (name, als, c) in enumerate((("GB300", gb_al, GB), ("MI355X", mi_al, MI))):
        yc = 80 + i * 80
        s.append(t(x0 - 14, yc + 4, name, "lbl", "end"))
        for k, v in enumerate(sorted(als)):
            s.append(f'<circle cx="{X(v):.1f}" cy="{yc - 16 + (k % 5) * 8:.1f}" r="3.8" fill="{c}" fill-opacity="0.85"/>')
    # cycle bars
    bx0, bx1, cmax = 700, 930, 10.0
    BX = lambda v: bx0 + (bx1 - bx0) * v / cmax
    s.append(t(bx0, 34, "cost per verify cycle (ms)", "head"))
    bars = [("GB300 real", GB_REAL_CYC, GB), ("GB300 sim", gb_cyc, GB + "99"),
            ("MI355X real", MI_REAL_CYC, MI), ("MI355X sim", mi_cyc, MI + "99")]
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
    gs_total = sum(GS["categories"][k]["wall_us"] for k in ORDER)
    rows = [("MI355X · serial", MV["categories"], "wall_us", f"{MV['cycle_us']:,.0f} µs, prior base"),
            ("GB300 · one stream", GS["categories"], "wall_us", f"{gs_total:,.0f} µs mean"),
            ("GB300 · default streams", GV["categories"], "wall_us", f"{gb_wall_total:,.0f} µs, wall share")]
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
                s.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="24" fill="{CAT_COLOR[k]}" '
                         f'stroke="{PAPER}" stroke-width="1"/>')
            x += w
    lx, ly = 30, 262
    for i, k in enumerate(ORDER):
        cx = lx + (i % 5) * 186
        cy = ly + (i // 5) * 24
        fill = "url(#hatch)" if k == "host_gap" else CAT_COLOR[k]
        s.append(f'<rect x="{cx}" y="{cy - 10}" width="14" height="12" fill="{fill}" stroke="{MUTED}" stroke-width="0.5"/>')
        s.append(t(cx + 20, cy, LAB[k], "legend"))
    s.append(t(30, 330, "Each bar sums to its mean cycle. With one stream a category's share is its standalone kernel "
                        "time; in the default layout overlapping kernels split their interval evenly.", "note"))
    s.append("</svg>")
    return "".join(s)


def plate_streams():
    W, H0 = 960, 330
    x0, x1, vmax = 330, 900, 10.0
    X = lambda v: x0 + (x1 - x0) * v / vmax
    groups = [("DSpark verify cycle (sim 5.5)", w3_cyc, mi_cyc), ("Plain decode step", w3_step, mi_step)]
    labels = [("base", "GB300 · default streams", GB), ("opt0", "GB300 · OPT_USE_MULTI_STREAM_OVERLAP=0", GB + "99"),
              ("serial", "GB300 · every side stream off", GB + "55")]
    s = [svg_open(W, H0, "GB300 cycle and step cost as side streams are turned off")]
    for v in range(0, 11, 2):
        s.append(f'<line x1="{X(v)}" y1="24" x2="{X(v)}" y2="{H0 - 38}" class="grid"/>')
        s.append(t(X(v), H0 - 20, f"{v}", "tick", "middle"))
    s.append(t(x1, H0 - 4, "milliseconds (lower is faster; GB300 session 3, NUMA-bound)", "axis", "end"))
    y = 30
    for head, vals, mi in groups:
        s.append(t(24, y + 10, head, "head"))
        for key, name, fill in labels:
            y += 24
            s.append(t(x0 - 14, y + 10, name, "lbl", "end"))
            s.append(f'<rect x="{x0}" y="{y}" width="{X(vals[key]) - x0:.1f}" height="14" fill="{fill}" '
                     f'stroke="{GB}" stroke-width="1"/>')
            s.append(t(X(vals[key]) + 6, y + 11.5, f"{vals[key]:.2f} ms", "val"))
        y += 24
        s.append(t(x0 - 14, y + 10, "MI355X · e2e824dc58", "lbl", "end"))
        s.append(f'<rect x="{x0}" y="{y}" width="{X(mi) - x0:.1f}" height="14" fill="{MI}"/>')
        s.append(t(X(mi) + 6, y + 11.5, f"{mi:.2f} ms", "val"))
        y += 40
    s.append("</svg>")
    return "".join(s)


def plate_graph_floor():
    W, H0 = 960, 362
    x0, x1, vmax = 330, 890, 9.0
    X = lambda v: x0 + (x1 - x0) * v / vmax
    rows = [("dependent tiny kernel (1 program)", FLOOR),
            ("PyTorch elementwise chain", {p: mb_fit(p, "torch_add_chain") for p in MB}),
            ("16 KiB producer → consumer", {p: mb_fit(p, "producer_consumer_16k") for p in MB}),
            ("BF16 GEMV 6×5120×1152, per call", GEMV),
            ("two parallel branches, per kernel", BRANCH),
            ("fork/join ladder, per kernel", LADDER)]
    s = [svg_open(W, H0, "Graph cost per kernel, CUDA on GB300 against HIP on MI355X")]
    for i, (fill, tag) in enumerate(((GB, "GB300 · CUDA 13.0 graph"), (MI, "MI355X · HIP 7.2 graph"))):
        s.append(f'<rect x="{x0 + i * 230}" y="12" width="14" height="11" fill="{fill}"/>')
        s.append(t(x0 + i * 230 + 20, 22, tag, "legend"))
    for v in range(0, 10):
        s.append(f'<line x1="{X(v):.1f}" y1="36" x2="{X(v):.1f}" y2="{H0 - 38}" class="grid"/>')
        s.append(t(X(v), H0 - 20, f"{v}", "tick", "middle"))
    s.append(t(x1, H0 - 4, "microseconds per kernel inside a replayed graph (median of 31 rounds)", "axis", "end"))
    y = 46
    for name, vals in rows:
        s.append(t(x0 - 14, y + 17, name, "lbl", "end"))
        for j, (p, fill) in enumerate((("cuda", GB), ("hip", MI))):
            yy = y + j * 14
            s.append(f'<rect x="{x0}" y="{yy}" width="{X(vals[p]) - x0:.1f}" height="11" fill="{fill}"/>')
            s.append(t(X(vals[p]) + 6, yy + 9.5, f"{vals[p]:.2f}", "val"))
        y += 46
    s.append("</svg>")
    return "".join(s)


def hero_cycle():
    W, H0 = 880, 146
    x0, x1, vmax = 150, 856, 10.0
    X = lambda v: x0 + (x1 - x0) * v / vmax
    rows = [("GB300", gb_cyc, GB, GB, ""), ("GB300, one stream", w3_cyc["serial"], GB + "55", GB, f' stroke="{GB}"'),
            ("MI355X", mi_cyc, MI, MI_INK, "")]
    s = [svg_open(W, H0, "One DSpark verify cycle on each machine, drawn to scale")]
    for v in range(0, 11):
        s.append(f'<line x1="{X(v):.1f}" y1="8" x2="{X(v):.1f}" y2="114" class="grid"/>')
        s.append(t(X(v), 130, f"{v}", "tick", "middle"))
    s.append(t(x1, 144, "ms per DSpark verify cycle, BS=1", "axis", "end"))
    for i, (name, v, fill, ink, stroke) in enumerate(rows):
        y = 14 + i * 34
        s.append(t(x0 - 14, y + 16, name, "head", "end", extra=f' fill="{ink}"'))
        s.append(f'<rect x="{x0}" y="{y}" width="{X(v) - x0:.1f}" height="22" rx="3" fill="{fill}"{stroke}/>')
        s.append(t(X(v) + 8, y + 16, f"{v:.2f} ms", "val"))
    s.append("</svg>")
    return "".join(s)


# ---------------------------------------------------------------- sections
def anatomy_table():
    head = ("<tr><th>" + S("category", "类别") + "</th><th>MI355X µs</th><th>" + S("GB300 one stream µs", "GB300 单 stream µs")
            + "</th><th>" + S("gap µs", "差距 µs") + "</th><th>" + S("GB300 default, wall µs", "GB300 默认布局，墙钟 µs")
            + "</th><th>" + S("kernels MI / GB", "kernel 数 MI / GB") + "</th></tr>")
    body = []
    for k in ORDER:
        mi, gw, mk = kern[k][0], kern[k][1], kern[k][3]
        gs = GS["categories"][k]
        kc = "–" if k == "host_gap" else f"{mk:.0f} / {gs['kernels']:.0f}"
        body.append(f"<tr><td>{S(LAB[k], ZH_CAT[k])}</td><td>{f0(mi)}</td><td>{f0(gs['wall_us'])}</td>"
                    f"<td>{sgn(serial_gap(k), 0)}</td><td>{f0(gw)}</td><td>{kc}</td></tr>")
    gs_total = sum(GS["categories"][k]["wall_us"] for k in ORDER)
    total = (f"<tr class='total'><td>{S('total', '合计')}</td><td>{f0(MV['cycle_us'])}</td><td>{f0(gs_total)}</td>"
             f"<td>{sgn(MV['cycle_us'] - gs_total, 0)}</td><td>{f0(gb_wall_total)}</td>"
             f"<td>{MV['kernels']:.0f} / {GS['kernels']:.0f}</td></tr>")
    return f'<div class="tablewrap"><table class="data">{head}{"".join(body)}{total}</table></div>\n'


ZH_CAT = {"moe": "MoE 专家与路由", "dense_gemm": "dense GEMM 与激活量化", "attention": "attention / indexer / KV",
          "mhc": "mHC", "all_reduce": "all-reduce 与 MoE finalize", "engram": "Engram", "draft": "DSpark draft",
          "other": "其他胶水 kernel", "host_gap": "GPU 空闲"}


def serial_gap(k):
    """MI355X (prior-base ledger) minus GB300 with every side stream off, per verify cycle."""
    return kern[k][0] - GS["categories"][k]["wall_us"]


def kernel_table():
    tw = TRACE["watched"]
    rows = [
        (S("MoE expert GEMM 1 (gate/up)", "MoE expert GEMM 1（gate/up）"), kt("bmm_MxE4m3"), led("MoE G1"),
         tw["MoE G1"]["traced_median_us"]),
        (S("MoE expert GEMM 2 (down)", "MoE expert GEMM 2（down）"), kt("bmm_Bfloat16"), led("MoE G2"),
         tw["MoE G2"]["traced_median_us"]),
        (S("MoE routing, sorting, quantization", "MoE 路由、排序、量化"),
         kt("_router_triton", "tiny_n_gemm", "routingIndices"),
         led("router GEMV", "router gate", "MoE sorting", "MoE quant/sort"), None),
        (S("MoE activation", "MoE 激活"), kt("silu_mul_clamp"), led("SiLU/clamp"), None),
        (S("attention main pass", "attention 主计算"), kt("partial_gluon"), led("PA main"),
         tw["attention main (partial_gluon)"]["traced_median_us"]),
        (S("attention combine / split reduce", "attention combine / split reduce"), kt("_combine"),
         led("PA split reduce + inverse RoPE"), tw["attention combine (_combine)"]["traced_median_us"]),
    ]
    head = (f"<tr><th>{S('per layer', '每层')}</th><th>{S('GB300 one stream µs', 'GB300 单 stream µs')}</th>"
            f"<th>{S('MI355X previous base µs', 'MI355X 上一个 base µs')}</th>"
            f"<th>{S('MI355X current base, traced µs', 'MI355X 当前 base，trace µs')}</th>"
            f"<th>{S('kernels GB / MI', 'kernel 数 GB / MI')}</th></tr>")
    body = "".join(f"<tr><td>{name}</td><td>{gb[0]:.1f}</td><td>{mi[0]:.1f}</td>"
                   f"<td>{'–' if tr is None else f'≤ {tr:.1f}'}</td><td>{gb[1]:.1f} / {mi[1]:.1f}</td></tr>"
                   for name, gb, mi, tr in rows)
    note = P("GB300: nsys kernel time with every side stream off (kernels that use programmatic dependent launch can "
             "include a wait on their predecessor). MI355X previous base: the scaled Kineto ledger, whose small-kernel "
             "rows are upper bounds. Current base: rocprofv3 medians with packet capture off, upper bounds by the "
             "calibration above.",
             "GB300：关闭所有侧 stream 时 nsys 测得的 kernel 时长（使用 programmatic dependent launch 的 kernel 可能包含等待"
             "前一个 kernel 的时间）。MI355X 上一个 base：缩放后的 Kineto 账本，小 kernel 那几行是上界。当前 base：关闭 packet "
             "capture 时 rocprofv3 的中位数，按上面的校准也是上界。", "tnote")
    return f'<div class="tablewrap"><table class="data">{head}{body}</table></div>\n' + note


def conclusions_table():
    rt_gb, rt_mi = RT["gb300"]["accepted_length_median"], RT["mi355x"]["accepted_length_median"]
    rows = [
        (S("Plain decode step", "plain decode 每步"), f"{f2(gb_step)} ms", f"{f2(mi_step)} ms",
         f"GB300 {mi_step / gb_step:.2f}×"),
        (S("DSpark verify cycle", "DSpark verify 周期"), f"{f2(gb_cyc)} ms", f"{f2(mi_cyc)} ms",
         f"GB300 {mi_cyc / gb_cyc:.2f}×"),
        (S("Output tokens/s, published protocol (sim 5.5)", "输出 tokens/s，公开流程（sim 5.5）"),
         f1(n1["sim"]["tps"]), f1(mu["sim"]["tps"]), f"GB300 {n1['sim']['tps'] / mu['sim']['tps']:.2f}×"),
        (S("Verify cycle, GB300 with every side stream off", "verify 周期，GB300 关闭所有侧 stream"),
         f"{f2(w3_cyc['serial'])} ms", f"{f2(mi_cyc)} ms", f"GB300 {mi_cyc / w3_cyc['serial']:.2f}×"),
        (S("Plain decode step, GB300 with every side stream off", "plain decode 每步，GB300 关闭所有侧 stream"),
         f"{f2(w3_step['serial'])} ms", f"{f2(mi_step)} ms", f"GB300 {mi_step / w3_step['serial']:.2f}×"),
        (S("Accepted tokens per verify, real text", "真实文本上每次 verify 接受的 token 数"), f2(rt_gb), f2(rt_mi), "–"),
        (S("4,096-token prefill (TTFT)", "4,096 token prefill（TTFT）"), f"{f0(n1['off']['ttft_ms'])} ms",
         f"{f0(mu['off']['ttft_ms'])} ms", f"MI355X {n1['off']['ttft_ms'] / mu['off']['ttft_ms']:.2f}×"),
        (S("Graph cost of one dependent tiny kernel", "graph 里一个相互依赖的微小 kernel 的开销"),
         f"{FLOOR['cuda']:.2f} µs", f"{FLOOR['hip']:.2f} µs", f"GB300 {FLOOR['hip'] / FLOOR['cuda']:.2f}×"),
        (S("Graph cost per kernel with fork/join branches", "带 fork/join 分支时每个 kernel 的 graph 开销"),
         f"{LADDER['cuda']:.2f} µs", f"{LADDER['hip']:.2f} µs", f"GB300 {LADDER['hip'] / LADDER['cuda']:.1f}×"),
    ]
    head = (f"<tr><th>{S('metric (BS=1, 4,096 in / 1,024 out)', '指标（BS=1，输入 4,096 / 输出 1,024）')}</th>"
            f"<th>GB300</th><th>MI355X</th><th>{S('faster', '更快的一方')}</th></tr>")
    body = "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td><td>{d}</td></tr>" for a, b, c, d in rows)
    return f'<div class="tablewrap"><table class="data">{head}{body}</table></div>\n'


def sec_conclusions(maxd):
    rt_gb, rt_mi = RT["gb300"]["accepted_length_median"], RT["mi355x"]["accepted_length_median"]
    share = CONC_MS * 1e3 / GAP_US
    step_gap = mi_step - gb_step
    step_conc = w3_step["serial"] - w3_step["base"]
    ranked = sorted((k for k in ORDER if k not in ("host_gap", "engram")), key=serial_gap, reverse=True)
    top = [k for k in ranked if serial_gap(k) > 50]
    parity = [k for k in ranked if abs(serial_gap(k)) <= 50]
    ahead = [k for k in ranked if serial_gap(k) < -50]
    idle_gap = serial_gap("host_gap")
    lst_en = ", ".join(f"{LAB[k]} {sgn(serial_gap(k), 0)} µs" for k in top)
    lst_zh = "，".join(f"{ZH_CAT[k]} {sgn(serial_gap(k), 0)} µs" for k in top)
    par_en = " and ".join(LAB[k] for k in parity) if parity else ""
    par_zh = "和".join(ZH_CAT[k] for k in parity) if parity else ""
    ahead_en = "; MI355X is ahead on " + ", ".join(f"{LAB[k]} ({sgn(serial_gap(k), 0)} µs)" for k in ahead) if ahead else ""
    ahead_zh = "；MI355X 领先的是" + "、".join(f"{ZH_CAT[k]}（{sgn(serial_gap(k), 0)} µs）" for k in ahead) if ahead else ""
    s = ['<section id="conclusions">' + H(2, "Conclusions", "结论", None)]
    s.append(P("The three questions above, answered by the measurements in sections 1 to 8.",
               "上面三个问题的答案，依据是第 1 到第 8 节的测量。"))
    s.append(conclusions_table())
    s.append(H(3, "How far behind is MI355X?", "MI355X 落后多少？"))
    s.append(P(
        f"About 2× per step at BS=1, and ahead at prefill. With the same request, client and metric, MI355X needs "
        f"{f2(mi_step)} ms per plain decode step against {f2(gb_step)} ms and {f2(mi_cyc)} ms per DSpark verify cycle "
        f"against {f2(gb_cyc)} ms. The verify cost does not depend on the acceptance mode and moves by at most "
        f"{maxd:.1f}% with CPU placement on MI355X, so the factor belongs to the machines and their software, not to the "
        f"protocol. On real text the two machines accept {f2(rt_gb)} and {f2(rt_mi)} tokens per verify, so acceptance "
        "does not change the picture. MI355X answers the 4,096-token prefill "
        f"{n1['off']['ttft_ms'] / mu['off']['ttft_ms']:.2f}× faster because its cell captures breakable prefill graphs.",
        f"BS=1 下每步大约慢 2 倍，但 prefill 更快。请求、客户端和指标相同时，MI355X 的 plain decode 每步 {f2(mi_step)} ms，GB300 "
        f"{f2(gb_step)} ms；DSpark 每个 verify 周期 {f2(mi_cyc)} ms 对 {f2(gb_cyc)} ms。verify 的耗时与接受方式无关，MI355X 上 CPU 放置"
        f"对它的影响最多 {maxd:.1f}%，所以这个倍数属于机器和它的软件，而不是测试流程。真实文本上两台机器每次 verify 分别接受 "
        f"{f2(rt_gb)} 和 {f2(rt_mi)} 个 token，接受长度不改变结论。MI355X 的 4,096 token prefill 快 "
        f"{n1['off']['ttft_ms'] / mu['off']['ttft_ms']:.2f} 倍，因为它的 cell 捕获了 breakable prefill graph。"))
    s.append(H(3, "Where is the gap?", "差距在哪里？"))
    s.append(P(
        f"The largest part is stream concurrency. With every side stream turned off, GB300's verify cycle rises from "
        f"{f2(w3_cyc['base'])} to {f2(w3_cyc['serial'])} ms, so {CONC_MS:.2f} ms of the {GAP_US / 1e3:.2f} ms gap "
        f"({share:.0%}) is what GB300's overlapped layout buys (a little more than overlap alone, because without side "
        f"streams SGLang also launches {GS['kernels'] - GV['kernels']:.0f} more kernels per verify). "
        f"{CONC_OPT_MS:.2f} ms of it comes from the attention-preparation, mHC-statistics, routed-quantization and "
        f"draft streams and {CONC_MOE_MS:.2f} ms from running the shared experts beside the routed experts. Plain "
        f"decode has the same shape: {step_conc:.2f} of its {step_gap:.2f} ms gap. MI355X cannot take this win today, "
        f"because a HIP graph kernel behind a fork/join costs {LADDER['hip']:.1f} µs against {LADDER['cuda']:.2f} µs "
        f"in a CUDA graph (section 7).",
        f"最大的一块是 stream 并发。关闭所有侧 stream 后，GB300 的 verify 周期从 {f2(w3_cyc['base'])} ms 升到 "
        f"{f2(w3_cyc['serial'])} ms，所以 {GAP_US / 1e3:.2f} ms 差距里有 {CONC_MS:.2f} ms（{share:.0%}）是 GB300 的重叠布局带来的"
        f"（比纯粹的重叠收益略多，因为关闭侧 stream 后 SGLang 每个 verify 还会多发射 {GS['kernels'] - GV['kernels']:.0f} 个 kernel）。"
        f"其中 {CONC_OPT_MS:.2f} ms 来自 attention 准备、mHC 统计、routed 量化和 draft 这几条 "
        f"stream，{CONC_MOE_MS:.2f} ms 来自让 shared expert 与 routed expert 并行。plain decode 的形状一样：{step_gap:.2f} ms "
        f"差距里有 {step_conc:.2f} ms。MI355X 今天拿不到这部分收益，因为 HIP graph 里跟在 fork/join 后面的 kernel 每个要 "
        f"{LADDER['hip']:.1f} µs，CUDA graph 里只要 {LADDER['cuda']:.2f} µs（第 7 节）。"))
    s.append(P(
        f"With both machines serial, {SERIAL_GAP_MS:.2f} ms per verify remains. By category (MI355X ledger against the "
        f"serialized GB300 profile): {lst_en}; GPU idle {idle_gap:+,.0f} µs"
        + (f"; {par_en} are within 50 µs" if par_en else "") + ahead_en + ". "
        f"Behind many of these rows is one cost: a dependent kernel takes at least {FLOOR['hip']:.2f} µs in a HIP graph "
        f"against {FLOOR['cuda']:.2f} µs in a CUDA graph, and a verify cycle with its draft runs {MI_KERNELS_NOW:,.0f} "
        f"kernels on MI355X today against {GV['kernels']:,.0f} on GB300.",
        f"两台机器都串行时，每个 verify 周期还差 {SERIAL_GAP_MS:.2f} ms。按类别（MI355X 账本对比串行化后的 GB300 profile）："
        f"{lst_zh}；GPU 空闲 {idle_gap:+,.0f} µs"
        + (f"；{par_zh}相差在 50 µs 以内" if par_zh else "") + ahead_zh + "。"
        f"很多行背后是同一个开销：一个相互依赖的 kernel 在 HIP graph 里至少要 {FLOOR['hip']:.2f} µs，在 CUDA graph 里只要 "
        f"{FLOOR['cuda']:.2f} µs，而一个 verify 周期连同 draft，今天在 MI355X 上要执行 {MI_KERNELS_NOW:,.0f} 个 kernel，GB300 上是 "
        f"{GV['kernels']:,.0f} 个。"))
    s.append(H(3, "What was measurement rather than machine?", "哪些是测量问题，不是机器问题？"))
    s.append(P(
        f"Three things, none of them a GPU property. The published target was two weeks stale: current main is "
        f"{-pct(gb_cyc, old_cyc):.0f}% faster per verify than the blog's commit ({f1(n1['sim']['tps'])} against "
        f"{BB_TP4:.0f} tokens/s). A {pct(e1n['tps'], e1b['tps']):.0f}% launch-to-launch swing on GB300 was CPU placement "
        "(Docker without SYS_NICE drops SGLang's NUMA binding, and simulated acceptance adds host work). And natural "
        "acceptance on a random prompt measures which repetition loop each machine's numerics fall into "
        f"({st.median(gb_al):.2f} against {st.median(mi_al):.2f} tokens per verify), while on real text the machines "
        "agree; compare cost per verify across machines.",
        f"三件事，都不是 GPU 的属性。公开的目标已经过时两周：当前 main 每个 verify 比博客所用的 commit 快 "
        f"{-pct(gb_cyc, old_cyc):.0f}%（{f1(n1['sim']['tps'])} 对 {BB_TP4:.0f} tokens/s）。GB300 上同一配置不同启动之间 "
        f"{pct(e1n['tps'], e1b['tps']):.0f}% 的波动来自 CPU 放置（Docker 没有 SYS_NICE 时 SGLang 的 NUMA 绑定失效，而模拟接受又"
        "增加了 host 工作）。随机 prompt 上的真实接受长度，测的是各机器的数值误差把生成带进了哪种重复循环"
        f"（每次 verify {st.median(gb_al):.2f} 对 {st.median(mi_al):.2f} 个 token），在真实文本上两台机器一致；跨机器应当比较"
        "每次 verify 的耗时。"))
    s.append(H(3, "Where to start", "从哪里开始"))
    s.append(UL([
        (f"<b>Concurrency, {CONC_MS:.1f} ms per verify.</b> Ask the ROCm runtime for graph branches that cost what "
         f"they cost on CUDA ({LADDER['hip']:.1f} against {LADDER['cuda']:.2f} µs per kernel in section 7's "
         "microbenchmark). Until then, make the side work cheap enough that overlap no longer matters: fold the "
         f"attention preparation and mHC statistics into neighbouring kernels ({CONC_OPT_MS:.2f} ms of GB300's "
         f"overlap, with the draft streams), and remove the separate shared-expert pass ({CONC_MOE_MS:.2f} ms). "
         "Both machines log that SGLang disables shared-expert fusion under EP4 because a rank holds only a slice of "
         "the routed experts; MoE TP4 lifts that restriction and deserves an A/B.",
         f"<b>并发，每个 verify {CONC_MS:.1f} ms。</b>请 ROCm runtime 让 graph 分支的开销与 CUDA 持平（第 7 节 microbenchmark 里"
         f"每个 kernel {LADDER['hip']:.1f} µs 对 {LADDER['cuda']:.2f} µs）。在那之前，把侧 stream 上的工作做得足够便宜，让重叠"
         f"变得无关紧要：把 attention 准备和 mHC 统计并进相邻的 kernel（连同 draft stream，占 GB300 重叠收益的 "
         f"{CONC_OPT_MS:.2f} ms），并去掉单独的 shared expert 一趟（{CONC_MOE_MS:.2f} ms）。两台机器的日志都显示，EP4 下每个 "
         "rank 只持有部分 routed expert，SGLang 因此关闭了 shared expert 融合；MoE TP4 没有这个限制，值得做一次 A/B。"),
        (f"<b>Kernel count.</b> Each kernel removed from the verify graph saves at least {FLOOR['hip']:.2f} µs on MI355X, "
         f"{FLOOR['hip'] / FLOOR['cuda']:.1f}× what it saves on GB300, so fusion pays more here than it did for the "
         f"CUDA work being ported. At {MI_KERNELS_NOW:,.0f} kernels per cycle the floor alone is "
         f"{FLOOR['hip'] * MI_KERNELS_NOW / 1e3:.1f} ms on MI355X against {FLOOR['cuda'] * GV['kernels'] / 1e3:.1f} ms on GB300.",
         f"<b>kernel 数量。</b>从 verify graph 里每去掉一个 kernel，MI355X 上至少省 {FLOOR['hip']:.2f} µs，是 GB300 上的 "
         f"{FLOOR['hip'] / FLOOR['cuda']:.1f} 倍，所以融合在这里的回报比在被移植的 CUDA 工作里更高。每个周期 "
         f"{MI_KERNELS_NOW:,.0f} 个 kernel，仅这个下限在 MI355X 上就是 {FLOOR['hip'] * MI_KERNELS_NOW / 1e3:.1f} ms，GB300 上是 "
         f"{FLOOR['cuda'] * GV['kernels'] / 1e3:.1f} ms。"),
        (f"<b>Kernel time: MoE ({sgn(serial_gap('moe'), 0)} µs) and attention ({sgn(serial_gap('attention'), 0)} µs) "
         "first.</b> Every other category is within " + f"{max(serial_gap(k) for k in top if k not in ('moe', 'attention')):.0f}"
         f" µs, and MI355X is already faster on mHC ({sgn(serial_gap('mhc'), 0)} µs): GB300's mHC is slower on its own "
         "and hidden behind a side stream, so mHC is not what MI355X needs to borrow.",
         f"<b>kernel 耗时：先看 MoE（{sgn(serial_gap('moe'), 0)} µs）和 attention（{sgn(serial_gap('attention'), 0)} µs）。</b>"
         "其他每一类都在 " + f"{max(serial_gap(k) for k in top if k not in ('moe', 'attention')):.0f}"
         f" µs 以内，而 mHC 上 MI355X 已经更快（{sgn(serial_gap('mhc'), 0)} µs）：GB300 的 mHC 单独运行更慢，只是被藏在了侧 "
         "stream 后面，所以 mHC 不是 MI355X 需要向 GB300 借鉴的地方。"),
        (f"<b>Host idle, {idle_gap:,.0f} µs.</b> <a href=\"{KM}/12\">kevin-mii/sglang#12</a> keeps speculative overlap "
         "scheduling on the forward stream and takes the cycle from 9.15 to 8.63 ms on this base.",
         f"<b>host 空闲，{idle_gap:,.0f} µs。</b><a href=\"{KM}/12\">kevin-mii/sglang#12</a> 把 speculative overlap 调度留在 "
         "forward stream 上，在这个 base 上把周期从 9.15 ms 降到 8.63 ms。"),
        ("<b>Keep</b> the prefill-graph advantage, and fix <code>SGLANG_SET_CPU_AFFINITY</code> for correctness; it "
         "does not move BS=1 speed.",
         "<b>保留</b> prefill graph 的优势；为正确性修好 <code>SGLANG_SET_CPU_AFFINITY</code>，它不影响 BS=1 的速度。"),
    ]))
    s.append("</section>")
    return "\n".join(s)


def sec_concurrency():
    step_conc = w3_step["serial"] - w3_step["base"]
    chain = {p: mb_row(p, "triton_chain_p1", 128) for p in MB}
    two = {p: mb_row(p, "two_branch_p1", 128) for p in MB}
    s = ['<section id="concurrency">' + H(2, "7 · What concurrency is worth, measured", "7 · 并发值多少：实测", None)]
    s.append(P(
        f"A profile of the default layout can only bound the concurrency prize: GB300's kernels add up to "
        f"{GV['kernel_sum_us'] / 1e3:.1f} ms per verify but occupy {GV['busy_us'] / 1e3:.1f} ms, so overlap hides at most "
        f"{GV_HIDDEN / 1e3:.1f} ms. To measure it, we turned GB300's side streams off and timed the same cells again, in one "
        "session, two launches per arm, with the order mirrored in time. <code>SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=0</code> "
        "removes the attention-preparation (KV, compressor, indexer), mHC-statistics, routed-quantization and "
        "DSpark-draft streams. The shared experts still run beside the routed experts on another stream that no switch "
        "controls, so the fully serial arm also loads a patch that clears it "
        "(<code>gb300/scripts/wb/serial_moe</code>). The default arm of this session reproduced the earlier NUMA-bound "
        f"numbers ({f2(w3_cyc['base'])} ms per verify, {f2(w3_step['base'])} ms per step).",
        f"默认布局下的 profile 只能给并发收益一个上限：GB300 的 kernel 时长加起来每个 verify 有 {GV['kernel_sum_us'] / 1e3:.1f} ms，"
        f"却只占 {GV['busy_us'] / 1e3:.1f} ms，所以重叠最多藏掉 {GV_HIDDEN / 1e3:.1f} ms。为了实测，我们在 GB300 上关掉侧 stream，"
        "在同一个 session 里重跑相同的 cell，每个 arm 启动两次，并按时间对称排序。<code>SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=0</code> "
        "会去掉 attention 准备（KV、compressor、indexer）、mHC 统计、routed 量化和 DSpark draft 这几条 stream。shared expert "
        "仍然在另一条 stream 上与 routed expert 并行，没有开关能关掉它，所以完全串行的 arm 另外加载了一个补丁把这条 stream "
        "清掉（<code>gb300/scripts/wb/serial_moe</code>）。这个 session 的默认 arm 复现了之前 NUMA 绑定的数字（每个 verify "
        f"{f2(w3_cyc['base'])} ms，每步 {f2(w3_step['base'])} ms）。"))
    s.append(plate("VII", "GB300 with its side streams turned off", "关闭侧 stream 后的 GB300", plate_streams(),
                   "Two launches per GB300 bar, six rounds each; the verify cycle uses simulated acceptance 5.5. MI355X "
                   "rows are the default-placement arms of sessions 1 and 2.",
                   "GB300 每根柱子两次启动、每次六轮；verify 周期用模拟接受 5.5。MI355X 为 session 1 和 2 的默认放置 arm。"))
    s.append(P(
        f"The verify cycle grows from {f2(w3_cyc['base'])} to {f2(w3_cyc['opt0'])} to {f2(w3_cyc['serial'])} ms and the "
        f"plain decode step from {f2(w3_step['base'])} to {f2(w3_step['opt0'])} to {f2(w3_step['serial'])} ms. Concurrency "
        f"is worth {CONC_MS:.2f} ms per verify on GB300, {CONC_MS * 1e3 / GV_HIDDEN:.0%} of the bound, and {step_conc:.2f} ms "
        "per decode step. The overlapped kernels hardly slow each other: at BS=1 they are small and leave most of the GPU "
        "idle, so a side stream fills it almost for free. One qualification: with its side streams off SGLang also takes "
        f"a different path that launches {GS['kernels'] - GV['kernels']:.0f} more kernels per verify (mHC statistics and "
        f"glue that the overlapped path fuses), so {CONC_MS:.2f} ms is what the overlapped layout is worth to GB300 as "
        "SGLang implements it, a little more than overlap alone. Against GB300's serial "
        f"{f2(w3_cyc['serial'])} ms, MI355X's {f2(mi_cyc)} ms leaves {SERIAL_GAP_MS:.2f} ms per verify, and "
        f"{mi_step - w3_step['serial']:.2f} ms per decode step.",
        f"verify 周期从 {f2(w3_cyc['base'])} ms 升到 {f2(w3_cyc['opt0'])} ms，再到 {f2(w3_cyc['serial'])} ms；plain decode 每步从 "
        f"{f2(w3_step['base'])} ms 升到 {f2(w3_step['opt0'])} ms，再到 {f2(w3_step['serial'])} ms。所以在 GB300 上，并发每个 verify "
        f"值 {CONC_MS:.2f} ms，是上限的 {CONC_MS * 1e3 / GV_HIDDEN:.0%}，每个 decode 步值 {step_conc:.2f} ms。重叠执行的 kernel "
        "几乎不互相拖慢：BS=1 时它们都很小，GPU 大部分时间空着，侧 stream 几乎是白白把它填满。需要说明一点：关闭侧 stream 后，"
        f"SGLang 还会走另一条代码路径，每个 verify 多发射 {GS['kernels'] - GV['kernels']:.0f} 个 kernel（重叠路径里融合掉的 mHC "
        f"统计和胶水 kernel），所以 {CONC_MS:.2f} ms 是按 SGLang 现有实现、重叠布局对 GB300 的价值，比纯粹的重叠收益略多一点。以 GB300 串行的 "
        f"{f2(w3_cyc['serial'])} ms 为基准，MI355X 的 {f2(mi_cyc)} ms 每个 verify 还差 {SERIAL_GAP_MS:.2f} ms，每个 decode 步差 "
        f"{mi_step - w3_step['serial']:.2f} ms。"))
    s.append(P(
        "Why not overlap the same way on MI355X? We ran the graph microbenchmark of our earlier MI355X BS=1 campaign "
        "unchanged on GB300 (<code>graph_floor.py</code>: Triton and PyTorch kernels captured into one graph and replayed; "
        f"device time per kernel, median of 31 rounds). A dependent tiny kernel costs {FLOOR['cuda']:.2f} µs in a CUDA "
        f"graph and {FLOOR['hip']:.2f} µs in a HIP graph. With fork/join branches the CUDA cost stays at "
        f"{LADDER['cuda']:.2f} µs per kernel, while HIP rises to {LADDER['hip']:.1f} µs, because a HIP graph with parallel "
        f"branches leaves the packet-capture path. Two independent branches of 64 kernels finish in {two['cuda']:.0f} µs on "
        f"CUDA, sooner than one chain of 128 ({chain['cuda']:.0f} µs), and in {two['hip']:.0f} µs on HIP, later than one "
        f"chain ({chain['hip']:.0f} µs). Larger kernels narrow the ratio: a BF16 GEMV costs {GEMV['cuda']:.1f} against "
        f"{GEMV['hip']:.1f} µs per call.",
        "为什么不在 MI355X 上同样重叠？我们把之前 MI355X BS=1 实验里的 graph microbenchmark 原样放到 GB300 上跑"
        "（<code>graph_floor.py</code>：把 Triton 和 PyTorch kernel 捕获进一个 graph 再重放，统计每个 kernel 的设备时间，取 31 轮"
        f"中位数）。一个相互依赖的微小 kernel 在 CUDA graph 里要 {FLOOR['cuda']:.2f} µs，在 HIP graph 里要 {FLOOR['hip']:.2f} µs。"
        f"加上 fork/join 分支后，CUDA 每个 kernel 仍是 {LADDER['cuda']:.2f} µs，HIP 却涨到 {LADDER['hip']:.1f} µs，因为带并行分支的 "
        f"HIP graph 会退出 packet capture 路径。两条各 64 个 kernel 的独立分支，CUDA 上 {two['cuda']:.0f} µs 就完成，比一条 128 个 "
        f"kernel 的链（{chain['cuda']:.0f} µs）还快；HIP 上要 {two['hip']:.0f} µs，比一条链（{chain['hip']:.0f} µs）还慢。kernel "
        f"越大差距越小：一个 BF16 GEMV 每次调用 {GEMV['cuda']:.1f} µs 对 {GEMV['hip']:.1f} µs。"))
    s.append(plate("VIII", "Graph cost per kernel, CUDA on GB300 and HIP on MI355X",
                   "graph 中每个 kernel 的开销：GB300 上的 CUDA 与 MI355X 上的 HIP", plate_graph_floor(),
                   "One GPU per machine, the same script and kernels. Tiny kernels are one program of 64 elements; the "
                   "ladder puts one kernel on the capture stream and two on side streams per rung, 128 rungs.",
                   "每台机器一张 GPU，脚本和 kernel 相同。微小 kernel 是一个 64 元素的 program；阶梯结构每一级在捕获 stream 上放一个 "
                   "kernel、在侧 stream 上放两个，共 128 级。"))
    s.append(P(
        f"So the stream structure that saves GB300 {CONC_MS:.1f} ms would cost MI355X time today, which is why the ROCm path "
        "keeps one stream. The same numbers make a runtime request concrete: branches as cheap as on CUDA, and a lower "
        "floor per dependent kernel. Until that lands, the lever on MI355X is to shrink the side work, not to overlap it.",
        f"所以，同样的 stream 结构在 GB300 上省下 {CONC_MS:.1f} ms，在今天的 MI355X 上反而会增加耗时，这也是 ROCm 路径保持单 "
        "stream 的原因。这些数字让 runtime 需求变得具体：分支要和 CUDA 一样便宜，每个依赖 kernel 的下限要更低。在那之前，"
        "MI355X 上能用的办法是把侧 stream 上的工作做小，而不是去重叠它。"))
    s.append("</section>")
    return "\n".join(s)


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
    "w3-base-sim": "main ffac53d779，LL cell，sim 5.5，NUMA 绑定，默认 stream 布局（session 3）",
    "w3-opt0-sim": "main ffac53d779，LL cell，sim 5.5，NUMA 绑定，SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=0",
    "w3-serial-sim": "main ffac53d779，LL cell，sim 5.5，NUMA 绑定，关闭所有侧 stream",
    "w3-base-off": "main ffac53d779，HT cell（关闭 DSpark），NUMA 绑定，默认 stream 布局（session 3）",
    "w3-opt0-off": "main ffac53d779，HT cell（关闭 DSpark），NUMA 绑定，SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=0",
    "w3-serial-off": "main ffac53d779，HT cell（关闭 DSpark），NUMA 绑定，关闭所有侧 stream",
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
            rows.append(f'<tr class="group {last}"><td colspan="8">{title}</td></tr>')
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
    head = ("<tr><th>" + S("category (GB300 plain decode step)", "类别（GB300 plain decode 每步）") + "</th><th>"
            + S("one stream µs", "单 stream µs") + "</th><th>" + S("default, wall µs", "默认布局，墙钟 µs") + "</th><th>"
            + S("default, kernel µs", "默认布局，kernel µs") + "</th><th>" + S("kernels", "kernel 数") + "</th></tr>")
    rows = []
    for k in ORDER:
        c, cs = GD["categories"][k], GDS["categories"][k]
        if k == "draft":
            continue
        kk = "–" if k == "host_gap" else f0(c["kernel_us"])
        kc = "–" if k == "host_gap" else f"{cs['kernels']:.0f}"
        rows.append(f"<tr><td>{S(LAB[k], ZH_CAT[k])}</td><td>{f0(cs['wall_us'])}</td><td>{f0(c['wall_us'])}</td>"
                    f"<td>{kk}</td><td>{kc}</td></tr>")
    wall = sum(GD["categories"][k]["wall_us"] for k in ORDER)
    wall_s = sum(GDS["categories"][k]["wall_us"] for k in ORDER)
    rows.append(f"<tr class='total'><td>{S('total', '合计')}</td><td>{f0(wall_s)}</td><td>{f0(wall)}</td>"
                f"<td>{f0(GD['kernel_sum_us'])}</td><td>{GDS['kernels']:.0f}</td></tr>")
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
    maxd_all = max(abs(pct(S2[(a, m)], S2[("split", m)])) for a in ("node1", "none") for m in ("sim", "real"))
    big = maxd_all >= 1.0
    if big:
        verdict_en = ("On MI355X the remote ranks do cost time, so the affinity arithmetic is a real bug on this "
                      "node, not a cosmetic one.")
        verdict_zh = "在 MI355X 上，远端 rank 确实会拖慢周期，所以这段亲和性算法在这台机器上是真问题，不只是不美观。"
    else:
        maxd = maxd_all
        verdict_en = (f"Placement moves the MI355X BS=1 cycle by at most {maxd:.1f}%. The host is still on the critical "
                      "path here (the verify-to-draft wait that kevin-mii/sglang#12 removes), but that share of the cycle "
                      "does not depend on which socket runs it, and sim costs the same as real acceptance on this machine. "
                      "The affinity arithmetic is still wrong and should be fixed, but the 2x does not come from it.")
        verdict_zh = (f"放置方式对 MI355X BS=1 周期的影响最多只有 {maxd:.1f}%。host 在这里仍处在关键路径上（就是 "
                      "kevin-mii/sglang#12 消掉的那段 verify 到 draft 的等待），但这部分耗时和它跑在哪个 socket 上无关；在这台机器上"
                      "sim 和真实接受的每周期耗时也相同。这段亲和性算法仍然是错的，应该修，但 2 倍差距不来自它。")

    gain_nice = pct(e1n["tps"], e1b["tps"])
    base_launch = " / ".join(f1(p["tps"]) for p in e1b["per_launch"])
    nice_launch = " / ".join(f1(p["tps"]) for p in e1n["per_launch"])
    pw_b = f"{min(p for _, p in pw_base):.0f}–{max(p for _, p in pw_base):.0f} W"
    pw_n = f"{min(p for _, p in pw_nice):.0f}–{max(p for _, p in pw_nice):.0f} W"
    clk = {round(c) for c, _ in pw_base + pw_nice}
    sec = []
    # Prologue
    sec.append('<section id="prologue">' + H(2, "Prologue", "序", None))
    sec.append(P(
        "SGLang's DeepSeek-V4.1 Flash numbers were produced on 4&times;GB300. Before borrowing anything from that work "
        "we wanted three answers: how far MI355X is behind when the request, the client and the metric are identical; "
        "which part of the difference is the machine and which part is software we can port; and which parts of the "
        "published numbers are measurement artifacts. We reran the published GB300 protocol on a GB300 tray, profiled "
        "it with nsys, replayed the same prompt with the same client on four MI355X GPUs, and then went back to the "
        "GB300 to measure what its stream concurrency is worth.",
        "SGLang 的 DeepSeek-V4.1 Flash 数据来自 4&times;GB300。在借鉴这些工作之前，我们想先回答三个问题：请求、客户端和"
        "指标完全相同时，MI355X 落后多少；差距里哪部分来自机器，哪部分是可以移植的软件；公开数字里哪些是测量方式造成的。"
        "为此我们在一个 GB300 托盘上重跑了公开的测试流程并用 nsys 做了 profile，再用同一个客户端、同一个 prompt 在四张 "
        "MI355X 上重放，最后回到 GB300 上实测它的 stream 并发到底值多少。"))
    sec.append("</section>")
    sec.append(sec_conclusions(maxd_all))

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
        "reflects the model.",
        f"每次 verify 的耗时并不受影响：GB300 真实接受时每周期 {f2(GB_REAL_CYC)} ms，模拟接受时 {f2(gb_cyc)} ms；MI355X 分别是 "
        f"{f2(MI_REAL_CYC)} 和 {f2(mi_cyc)} ms。随机 prompt 本身没有意义，微小的数值差异就能决定模型走向哪种退化的续写。在它上面"
        "比较真实接受的 tokens/s，等于把某台机器的舍入方式算成了它的功劳。接受长度只有在真实文本上才值得比较，那时它反映的是"
        "模型本身。"))
    rg, rm = RT["gb300"], RT["mi355x"]
    al_diff = pct(rg["accepted_length_median"], rm["accepted_length_median"])
    by_en = ", ".join(f"{rg['accepted_length_by_prompt'][str(i)]:.2f}/{rm['accepted_length_by_prompt'][str(i)]:.2f}"
                      for i in range(4))
    sec.append(P(
        "So we replayed our MI355X real-text contract on GB300: the same four chat-encoded 4,096-token prompts, two "
        "unscored warm-ups and then 24 samples of each, the same request body and client code, two launches at the "
        f"default stream layout. GB300 accepts a median {rg['accepted_length_median']:.2f} tokens per verify, MI355X "
        f"{rm['accepted_length_median']:.2f} over its two base-arm servers ({sgn(al_diff)}%; per prompt, GB300/MI355X: "
        f"{by_en}). On real text the accepted length is a property of the model and the two machines "
        f"{'agree' if abs(al_diff) <= 5 else 'differ'}, so tokens/s on real text follows the cost per verify. With host "
        f"and scheduler time included, a verify on this text takes {rg['wall_per_verify_proxy_ms_median']:.2f} ms on GB300 "
        f"and {rm['wall_per_verify_proxy_ms_median']:.2f} ms on MI355X, and decode runs at "
        f"{rg['stream_decode_tps_median']:.0f} against {rm['stream_decode_tps_median']:.0f} tokens/s.",
        "所以我们把 MI355X 的真实文本契约拿到 GB300 上重放：同样四个 chat 编码的 4,096 token prompt，先两次不计分的热身，再每个 "
        "prompt 采样 24 次，请求体和客户端代码相同，默认 stream 布局下启动两次。GB300 每次 verify 接受 token 数的中位数是 "
        f"{rg['accepted_length_median']:.2f}，MI355X 两个 base arm server 合计是 {rm['accepted_length_median']:.2f}（{sgn(al_diff)}%；"
        f"按 prompt，GB300/MI355X：{by_en}）。在真实文本上，接受长度是模型的属性，两台机器"
        f"{'一致' if abs(al_diff) <= 5 else '不一致'}，所以真实文本上的 tokens/s 取决于每次 verify 的耗时。算上 host 和 scheduler "
        f"的时间，这段文本上每次 verify GB300 要 {rg['wall_per_verify_proxy_ms_median']:.2f} ms，MI355X 要 "
        f"{rm['wall_per_verify_proxy_ms_median']:.2f} ms；decode 速度分别是 {rg['stream_decode_tps_median']:.0f} 和 "
        f"{rm['stream_decode_tps_median']:.0f} tokens/s。"))
    sec.append("</section>")

    # Anatomy
    sec.append('<section id="anatomy">' + H(2, "6 · Where 4.6 ms and 9 ms go", "6 · 4.6 ms 和 9 ms 各花在哪里", None))
    sec.append(P(
        "The GB300 profiles capture TP0 under nsys with CUDA-graph node tracing and split every cycle into the "
        "categories of our MI355X ledger. In the default layout streams overlap, so a category's kernel time there is "
        "not what it costs alone. We therefore also captured GB300 with every side stream off (section 7), where kernel "
        f"time is standalone cost ({GS['cycles']} verify cycles, overlap {GS['overlap']:.2f}). The MI355X graph executes "
        "its kernels back to back; its ledger comes from our earlier BS=1 campaign on the previous base "
        f"({MV['cycle_us'] / 1e3:.2f} ms per cycle against {f2(mi_cyc)} ms today, genuine acceptance, profiled family shares "
        "scaled to unprofiled phase totals).",
        "GB300 的 profile 在 nsys 下以 CUDA graph 节点粒度采集 TP0，再把每个周期按我们 MI355X 账本的类别拆开。默认布局下多个 "
        "stream 会并发，所以那时某一类的 kernel 时长并不是它单独运行的开销。因此我们又在关闭所有侧 stream 的条件下采了一次 "
        f"GB300（第 7 节），这时 kernel 时长就是单独运行的开销（{GS['cycles']} 个 verify 周期，重叠系数 {GS['overlap']:.2f}）。"
        "MI355X 的 graph 串行执行 kernel；它的账本来自我们在上一个 base 上做的 BS=1 实验（每周期 "
        f"{MV['cycle_us'] / 1e3:.2f} ms，今天是 {f2(mi_cyc)} ms；真实接受；按 profile 得到的各族占比，缩放到未开 profiler 时实测的"
        "阶段总时长）。"))
    sec.append(plate("VI", "Anatomy of one DSpark verify cycle", "一个 DSpark verify 周期的构成", plate_anatomy(),
                     "Hatched: GPU idle. The one-stream bar is GB300 with every side stream off, so each segment is a "
                     "standalone cost; the default bar splits the time of overlapping kernels evenly.",
                     "斜线：GPU 空闲。单 stream 那一条是关闭所有侧 stream 的 GB300，每一段都是单独运行的开销；默认布局那一条"
                     "把重叠 kernel 的时间平均分摊。"))
    sec.append(anatomy_table())
    g = {k: sgn(serial_gap(k), 0) for k in ORDER}
    gs_total = sum(GS["categories"][k]["wall_us"] for k in ORDER)
    close_us = abs((MV["cycle_us"] - gs_total) - SERIAL_GAP_MS * 1e3)
    sec.append(P(
        f"With GB300 serial, the gap column is a number per category. MoE is the largest ({g['moe']} µs), then attention "
        f"({g['attention']} µs); all-reduce with MoE finalize ({g['all_reduce']}), the draft ({g['draft']}) and dense GEMM "
        f"({g['dense_gemm']}) follow, and glue kernels are close ({g['other']}). MI355X is faster on mHC ({g['mhc']} µs): "
        "GB300's mHC statistics are cheap only because a side stream hides them. The GPU-idle row "
        f"({g['host_gap']} µs against serial GB300, {sgn(kern['host_gap'][0] - kern['host_gap'][1], 0)} against the default "
        f"layout) is the host wait that our follow-up to {pr(39857)}, <a href=\"{KM}/12\">kevin-mii/sglang#12</a>, removes "
        "by keeping speculative overlap scheduling on the forward stream (9.15 to 8.63 ms per cycle on this base); "
        f"<a href=\"{KM}/13\">#13</a> additionally builds the draft-block metadata inside the draft graph. The table's total "
        f"gap is within {close_us:.0f} µs of the timed one ({SERIAL_GAP_MS:.2f} ms): the MI355X ledger is "
        f"{MV['cycle_us'] / 1e3:.2f} ms from the previous base against {f2(mi_cyc)} ms today, and the serial capture runs "
        f"{GS['cycle_us_median'] / 1e3:.2f} ms under nsys against {f2(w3_cyc['serial'])} ms untraced.",
        f"GB300 串行之后，差距那一列每一类都是一个确定的数。MoE 最大（{g['moe']} µs），其次是 attention（{g['attention']} µs）；"
        f"all-reduce 加 MoE finalize（{g['all_reduce']}）、draft（{g['draft']}）和 dense GEMM（{g['dense_gemm']}）随后，胶水 kernel "
        f"很接近（{g['other']}）。mHC 上 MI355X 更快（{g['mhc']} µs）：GB300 的 mHC 统计之所以便宜，只是因为被侧 stream 藏起来了。"
        f"GPU 空闲那一行（对比串行 GB300 为 {g['host_gap']} µs，对比默认布局为 "
        f"{sgn(kern['host_gap'][0] - kern['host_gap'][1], 0)}），正是我们在 {pr(39857)} 之上的后续改动 "
        f"<a href=\"{KM}/12\">kevin-mii/sglang#12</a> 要消掉的 host 等待：它把 speculative overlap 调度留在 forward stream 上（在这个 "
        f"base 上每周期从 9.15 ms 降到 8.63 ms）；<a href=\"{KM}/13\">#13</a> 则进一步把 draft-block metadata 放进 draft graph 里构建。"
        f"表中的总差距和计时得到的差距（{SERIAL_GAP_MS:.2f} ms）相差 {close_us:.0f} µs 以内：MI355X 账本来自上一个 base，"
        f"{MV['cycle_us'] / 1e3:.2f} ms，今天是 {f2(mi_cyc)} ms；串行 capture 在 nsys 下是 {GS['cycle_us_median'] / 1e3:.2f} ms，不开 "
        f"profiler 时是 {f2(w3_cyc['serial'])} ms。"))
    wt = TRACE["watched"]
    sec.append(P(
        "We also tried to rebuild the MI355X ledger on the current base with a kernel trace instead of a torch profile. "
        "rocprofv3 1.1.0 aborts the first replay of a packet-captured HIP graph "
        "(<code>HSA_STATUS_ERROR_INVALID_PACKET_FORMAT</code>). With packet capture disabled the server runs, but "
        f"{TRACE['traced_share_between_4_and_6_us']:.0%} of traced kernels land between 4 and 6 µs whatever they do "
        f"(median {TRACE['traced_duration_median_us']:.1f} µs), the kind of instrumentation floor that ruled out the torch "
        f"profile, and the fused all-reduce reads {wt['fused all-reduce + mHC post']['traced_median_us']:.0f} µs per call "
        "because it absorbs the rank skew that tracing adds. What the trace does settle is structure: the current base "
        f"runs {MI_KERNELS_NOW:,.0f} kernels per verify with its draft, against {MV['kernels']:,.0f} on the previous base and "
        f"{GV['kernels']:,.0f} on GB300, and its attention main pass is now <code>partial_gluon</code> with "
        "<code>_combine</code>, the kernels GB300 runs. The ledger above therefore stays on the previous base.",
        "我们也尝试用 kernel trace 代替 torch profiler，在当前 base 上重建 MI355X 账本。rocprofv3 1.1.0 在 packet capture 的 HIP "
        "graph 第一次重放时就会中止（<code>HSA_STATUS_ERROR_INVALID_PACKET_FORMAT</code>）。关掉 packet capture 后 server 能跑，"
        f"但 {TRACE['traced_share_between_4_and_6_us']:.0%} 的 kernel 不论做什么，trace 出来都落在 4 到 6 µs 之间（中位数 "
        f"{TRACE['traced_duration_median_us']:.1f} µs），这正是当初否定 torch profiler 的那种计时下限；融合的 all-reduce 每次调用读出 "
        f"{wt['fused all-reduce + mHC post']['traced_median_us']:.0f} µs，因为它吸收了 trace 带来的 rank 间等待。trace 能确定的是结构："
        f"当前 base 每个 verify 连同 draft 要执行 {MI_KERNELS_NOW:,.0f} 个 kernel，上一个 base 是 {MV['kernels']:,.0f} 个，GB300 是 "
        f"{GV['kernels']:,.0f} 个；它的 attention 主计算已经换成 <code>partial_gluon</code> 加 <code>_combine</code>，和 GB300 用的是同一组 "
        "kernel。所以上面的账本仍然用上一个 base 的。"))
    sec.append(P(
        "The GB300 plain-decode step in the same categories, with side streams on and off (200 steps, TP0):",
        "GB300 plain decode 每一步按同样类别的拆分，分别是打开和关闭侧 stream（200 步，TP0）："))
    sec.append(decode_table())
    sec.append("</section>")

    sec.append(sec_concurrency())

    # Lessons
    share = CONC_MS * 1e3 / GAP_US
    sec.append('<section id="lessons">' + H(2, "8 · What transfers to MI355X", "8 · 哪些可以用到 MI355X 上", None))
    sec.append(H(3, "8.1 · Concurrency is the largest single gap", "8.1 · 并发是最大的一项差距"))
    sec.append(P(
        f"Stream concurrency is worth {CONC_MS:.2f} ms of GB300's verify cycle, {share:.0%} of the gap. The route GB300 "
        f"takes is closed on MI355X today: a HIP graph kernel behind a fork/join costs {LADDER['hip']:.1f} µs against "
        f"{LADDER['cuda']:.2f} µs on CUDA, and in our earlier measurement a wait pending in another hardware queue slowed "
        "every dispatch of the running graph by about 1.3 µs. Two ways forward, not mutually exclusive: a ROCm runtime "
        "that keeps packet capture for branched graphs (section 7 sizes the prize), and meanwhile shrinking the side "
        f"work until overlap no longer matters. The A/B says where that work sits: {CONC_OPT_MS:.2f} ms in the streams "
        "<code>SGLANG_OPT_USE_MULTI_STREAM_OVERLAP</code> controls (attention preparation, mHC statistics, routed "
        f"quantization, draft) and {CONC_MOE_MS:.2f} ms in the shared experts.",
        f"stream 并发值 GB300 verify 周期的 {CONC_MS:.2f} ms，占差距的 {share:.0%}。GB300 走的这条路今天在 MI355X 上走不通："
        f"HIP graph 里跟在 fork/join 后面的 kernel 每个要 {LADDER['hip']:.1f} µs，CUDA 上是 {LADDER['cuda']:.2f} µs；我们之前还测到，"
        "另一个硬件队列里挂着的等待，会让正在运行的 graph 每次派发慢约 1.3 µs。有两条路，可以同时走：让 ROCm runtime 在带分支的 "
        "graph 上保留 packet capture（第 7 节量化了能拿到多少）；在那之前，把侧 stream 上的工作做小，小到重叠不再重要。A/B 说明了"
        f"这些工作在哪里：{CONC_OPT_MS:.2f} ms 在 <code>SGLANG_OPT_USE_MULTI_STREAM_OVERLAP</code> 控制的几条 stream 上（attention 准备、"
        f"mHC 统计、routed 量化、draft），{CONC_MOE_MS:.2f} ms 在 shared expert 上。"))
    sec.append(H(3, "8.2 · Kernel count costs more on MI355X", "8.2 · kernel 数量在 MI355X 上更贵"))
    extra = MI_KERNELS_NOW - GV["kernels"]
    sec.append(P(
        f"A dependent kernel costs {FLOOR['hip']:.2f} µs in a HIP graph and {FLOOR['cuda']:.2f} µs in a CUDA graph, so every "
        f"kernel removed from the verify graph saves {FLOOR['hip'] / FLOOR['cuda']:.1f}× more on MI355X, and a fusion that "
        f"is marginal on GB300 can pay here. The current base runs {MI_KERNELS_NOW:,.0f} kernels per cycle, {extra:,.0f} more "
        f"than GB300; at the floor alone that difference is worth {extra * FLOOR['hip'] / 1e3:.2f} ms per verify.",
        f"一个相互依赖的 kernel 在 HIP graph 里要 {FLOOR['hip']:.2f} µs，在 CUDA graph 里要 {FLOOR['cuda']:.2f} µs，所以从 verify graph "
        f"里每去掉一个 kernel，MI355X 上省下的是 GB300 上的 {FLOOR['hip'] / FLOOR['cuda']:.1f} 倍；在 GB300 上可有可无的融合，在这里"
        f"可能值得做。当前 base 每个周期执行 {MI_KERNELS_NOW:,.0f} 个 kernel，比 GB300 多 {extra:,.0f} 个；仅按下限算，这个差值每个 "
        f"verify 就值 {extra * FLOOR['hip'] / 1e3:.2f} ms。"))
    sec.append(H(3, "8.3 · Kernel time, ranked by the serialized gap", "8.3 · 按串行化差距排序的 kernel 耗时"))
    pa, sr, kv = (mi_attn_parts[k] for k in ("PA main", "PA split reduce + inverse RoPE", "KV norm/RoPE/store"))
    sec.append(P(
        f"MoE ({sgn(serial_gap('moe'), 0)} µs) is first, with about {mi_moe_layer:.0f} kernels per layer on MI355X (two GEMMs "
        f"plus routing, sorting, quantization and activation) against {gs_moe_layer:.0f} on GB300. Attention "
        f"({sgn(serial_gap('attention'), 0)} µs) is second. On the previous base MI355X spent {pa + sr + kv:.0f} µs per layer "
        f"in its sparse attention chain ({pa:.1f} µs main pass, {sr:.1f} µs split reduce with inverse RoPE, {kv:.1f} µs KV "
        f"norm/RoPE/store); GB300's attention kernels take {gs_attn_layer:.1f} µs per layer on their own. The current base "
        "already runs GB300's attention kernels, so what remains there is time per kernel, not a missing kernel. "
        f"All-reduce with MoE finalize ({sgn(serial_gap('all_reduce'), 0)} µs), the draft ({sgn(serial_gap('draft'), 0)} µs) "
        f"and dense GEMM ({sgn(serial_gap('dense_gemm'), 0)} µs) are each about 0.2 ms; together they weigh as much as "
        "attention, but no single kernel dominates them.",
        f"排第一的是 MoE（{sgn(serial_gap('moe'), 0)} µs），MI355X 每层约 {mi_moe_layer:.0f} 个 kernel（两个 GEMM，加上路由、排序、"
        f"量化和激活），GB300 是 {gs_moe_layer:.0f} 个。其次是 attention（{sgn(serial_gap('attention'), 0)} µs）。在上一个 base 上，"
        f"MI355X 每层在稀疏 attention 链路上花 {pa + sr + kv:.0f} µs（主计算 {pa:.1f} µs，带 inverse RoPE 的 split reduce {sr:.1f} µs，"
        f"KV norm/RoPE/store {kv:.1f} µs）；GB300 的 attention kernel 单独运行每层 {gs_attn_layer:.1f} µs。当前 base 已经在用 GB300 "
        "的 attention kernel，所以这里剩下的是每个 kernel 的耗时，而不是缺了哪个 kernel。all-reduce 加 MoE finalize"
        f"（{sgn(serial_gap('all_reduce'), 0)} µs）、draft（{sgn(serial_gap('draft'), 0)} µs）和 dense GEMM"
        f"（{sgn(serial_gap('dense_gemm'), 0)} µs）各约 0.2 ms；加起来和 attention 一样重，但没有哪个 kernel 占主导。"))
    sec.append(kernel_table())
    g1, g2 = kt("bmm_MxE4m3"), kt("bmm_Bfloat16")
    m1, m2 = led("MoE G1"), led("MoE G2")
    gr, mr = kt("_router_triton", "tiny_n_gemm", "routingIndices"), led("router GEMV", "router gate", "MoE sorting",
                                                                         "MoE quant/sort")
    ga, ma = kt("silu_mul_clamp"), led("SiLU/clamp")
    gp, mp = kt("partial_gluon"), led("PA main")
    tr_pa = TRACE["watched"]["attention main (partial_gluon)"]["traced_median_us"]
    sec.append(P(
        f"Kernel by kernel, the largest single gap is the first expert GEMM ({m1[0]:.1f} µs per layer on the previous base "
        f"against {g1[0]:.1f} µs), while the second is at parity ({m2[0]:.1f} against {g2[0]:.1f}). Around the GEMMs MI355X "
        f"ran {mr[1]:.1f} routing, sorting and quantization kernels per layer against {gr[1]:.0f}, for {mr[0]:.1f} against "
        f"{gr[0]:.1f} µs, and its activation cost {ma[0]:.1f} against {ga[0]:.1f} µs. On these numbers most of the MoE gap "
        "sits in the small kernels rather than the GEMMs (the small-kernel rows are upper bounds), which is where kernel "
        "count and fusion (8.2) apply. The attention main pass was "
        f"{mp[0]:.1f} µs per layer on the previous base against GB300's {gp[0]:.1f}; the current base runs GB300's "
        f"<code>partial_gluon</code>, at most {tr_pa:.1f} µs traced, so that row has probably shrunk, and the rebuilt ledger "
        "will say by how much.",
        f"逐个 kernel 看，差距最大的是第一个 expert GEMM（上一个 base 每层 {m1[0]:.1f} µs，GB300 {g1[0]:.1f} µs），第二个基本持平"
        f"（{m2[0]:.1f} 对 {g2[0]:.1f}）。GEMM 周围，MI355X 每层要跑 {mr[1]:.1f} 个路由、排序和量化 kernel，GB300 是 {gr[1]:.0f} 个，"
        f"耗时 {mr[0]:.1f} 对 {gr[0]:.1f} µs；激活 {ma[0]:.1f} 对 {ga[0]:.1f} µs。按这些数字，MoE 的差距大半在小 kernel 上，而不在 "
        f"GEMM（小 kernel 那几行是上界），这正是 kernel 数量和融合（8.2）起作用的地方。attention 主计算在上一个 base 上每层 "
        f"{mp[0]:.1f} µs，GB300 是 "
        f"{gp[0]:.1f} µs；当前 base 已经换成 GB300 的 <code>partial_gluon</code>，trace 读数最多 {tr_pa:.1f} µs，所以这一行很可能已经"
        "缩小，重建后的账本会给出具体数字。"))
    sec.append(H(3, "8.4 · Keep what MI355X already does better", "8.4 · 保留 MI355X 已经做得更好的部分"))
    sec.append(P(
        f"MI355X answers the 4,096-token prefill in {f0(mu['off']['ttft_ms'])} ms against GB300's {f0(n1['off']['ttft_ms'])} ms "
        f"bound and {f0(s1['off']['ttft_ms'])} ms unbound, because the MI350X cell captures breakable prefill graphs while "
        "CUDA main disables them for this model. That advantage should survive any port of NVIDIA-side changes: a "
        "change that needs eager prefill to work is a regression on this machine even if its decode numbers improve. "
        f"The same holds for mHC, which MI355X runs {-serial_gap('mhc'):.0f} µs per verify faster than GB300 does on its own.",
        f"MI355X 完成 4,096 token prefill 只要 {f0(mu['off']['ttft_ms'])} ms，GB300 绑定时要 {f0(n1['off']['ttft_ms'])} ms，不绑定时 "
        f"{f0(s1['off']['ttft_ms'])} ms，原因是 MI350X 的 cell 捕获了 breakable prefill graph，而 CUDA 版 main 对这个模型关掉了它。"
        "移植 NVIDIA 侧改动时要保住这个优势：如果某个改动必须靠 eager prefill 才能工作，那么即使 decode 数字变好，在这台机器上也是"
        f"一次回退。mHC 也一样：MI355X 每个 verify 在 mHC 上比 GB300 单独运行快 {-serial_gap('mhc'):.0f} µs。"))
    sec.append(H(3, "8.5 · Placement and measurement rules", "8.5 · 放置与测量规则"))
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
        (f"Measure what overlap is worth by turning it off. The kernel-sum bound ({GV_HIDDEN / 1e3:.1f} ms) happened to be "
         f"close here ({CONC_MS:.2f} ms measured), but it is only a bound.",
         f"要知道重叠值多少，就把它关掉来测。kernel 时长之和给出的上限（{GV_HIDDEN / 1e3:.1f} ms）这次恰好接近实测"
         f"（{CONC_MS:.2f} ms），但它只是上限。"),
        ("Calibrate a kernel trace against an untraced microbenchmark before trusting small-kernel durations; on ROCm 7.2 "
         "rocprofv3 needs graph packet capture disabled and still reports a floor near 5 µs.",
         "在相信小 kernel 的时长之前，先用不开 profiler 的 microbenchmark 校准 kernel trace；在 ROCm 7.2 上，rocprofv3 需要关掉 "
         "graph packet capture，而且仍然有接近 5 µs 的计时下限。"),
    ]))
    sec.append("</section>")

    # Next
    tp4_gain = pct(n0["tp4"]["tps"], n0["ep4"]["tps"])
    sec.append('<section id="next">' + H(2, "9 · Next experiments", "9 · 下一步实验", None))
    sec.append(UL([
        ("Fold side-stream work into its neighbours on MI355X, one change at a time, each measured with the BS=1 A/B "
         "used here: the mHC statistics, the attention preparation (KV norm/RoPE/store with the compressor and indexer "
         "projections), then the MoE routing, sorting and quantization chain.",
         "在 MI355X 上把侧 stream 的工作并进相邻的 kernel，每次只改一处，每次都用本文的 BS=1 A/B 测量：先是 mHC 统计，再是 "
         "attention 准备（KV norm/RoPE/store，连同 compressor 和 indexer 的投影），然后是 MoE 的路由、排序和量化链路。"),
        ("A/B MoE TP4 (EP1) against EP4 on MI355X. SGLang fuses the shared expert only without expert parallelism, and on "
         f"GB300's older commit MoE TP4 ran {tp4_gain:.1f}% faster than EP4 under the published protocol.",
         "在 MI355X 上对比 MoE TP4（EP1）和 EP4。SGLang 只有在不用 expert parallelism 时才会融合 shared expert；在 GB300 的旧 "
         f"commit 上，按公开流程 MoE TP4 比 EP4 快 {tp4_gain:.1f}%。"),
        (f"File the ROCm requests with this page's numbers: fork/join graph branches ({LADDER['hip']:.1f} against "
         f"{LADDER['cuda']:.2f} µs per kernel), the per-kernel graph floor ({FLOOR['hip']:.2f} against {FLOOR['cuda']:.2f} µs), "
         "and rocprofv3 aborting packet-captured graph replays.",
         f"把本文的数字附上，向 ROCm 提需求：fork/join graph 分支（每个 kernel {LADDER['hip']:.1f} µs 对 {LADDER['cuda']:.2f} µs）、"
         f"每个 kernel 的 graph 下限（{FLOOR['hip']:.2f} µs 对 {FLOOR['cuda']:.2f} µs），以及 rocprofv3 在 packet capture 的 graph 重放时"
         "中止。"),
        ("Rebuild the MI355X ledger on the current base once a trace runs without that floor, or with event-timed "
         "phases as the previous ledger did.",
         "等 trace 没有这个计时下限了，或者像上一份账本那样按阶段用 event 计时，再在当前 base 上重建 MI355X 账本。"),
        ("Land kevin-mii/sglang#12 and #13, then measure the GPU-idle row again.",
         "合入 kevin-mii/sglang#12 和 #13，然后重新测 GPU 空闲那一行。"),
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
    sec.append('<section id="reproduce">' + H(2, "10 · Reproduce", "10 · 复现", None))
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
    sec.append(P(
        "Session 3 ran inside the GB300's persistent workbench container, which has no Docker CLI: "
        "<code>gb300/scripts/wb/wb_arm.py</code> starts each server as a fresh process there with the same cells, client, "
        "JIT caches and records as <code>arm.py</code> (and with <code>CAP_SYS_NICE</code>, so every arm is NUMA-bound); "
        "<code>queue3.sh</code> is the whole session, <code>serial_moe/</code> the shared-expert patch and "
        "<code>bs1_realtext.py</code> the real-text client. On MI355X, <code>graph_floor.py</code> ran under a clean "
        "exclusive lease (session 3). The rocprofv3 calibration and traced server ran in sessions 3b and 3c, whose "
        "canaries moved by up to 11% in both directions after traced servers were killed; nothing from them is used as a "
        "timing, only the trace's kernel counts.",
        "session 3 在 GB300 常驻的 workbench 容器里运行，那里没有 Docker CLI：<code>gb300/scripts/wb/wb_arm.py</code> 在容器里把"
        "每个 server 作为全新进程启动，cell、客户端、JIT cache 和记录方式都与 <code>arm.py</code> 相同（容器有 "
        "<code>CAP_SYS_NICE</code>，所以每个 arm 都绑定了 NUMA）；<code>queue3.sh</code> 是整个 session，<code>serial_moe/</code> "
        "是 shared expert 的补丁，<code>bs1_realtext.py</code> 是真实文本客户端。MI355X 上，<code>graph_floor.py</code> 在一个干净的"
        "独占租约里运行（session 3）。rocprofv3 的校准和带 trace 的 server 在 session 3b 和 3c 里运行，强杀带 trace 的 server 之后，"
        "这两个租约的 canary 双向变化最多 11%；它们的结果都没有被当作计时使用，只用了 trace 里的 kernel 数量。"))
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
               '# GB300 session 3 (inside the workbench): stream A/B, serial nsys, real text, graph floor\n'
               'bash gb300/scripts/wb/queue3.sh\n\n'
               '# analysis\n'
               'python3 analysis/analyze.py && python3 analysis/compare_attribution.py && python3 analysis/realtext.py\n'
               '</code></pre>\n')
    sec.append("</section>")

    # Epilogue
    sec.append('<section id="epilogue">' + H(2, "Epilogue", "后记", None))
    sec.append(P(
        "The GB300 run was supposed to give us a target. It gave us a decomposition instead: of MI355X's "
        f"{GAP_US / 1e3:.1f} ms per verify, {CONC_MS:.1f} ms is concurrency that CUDA graphs give GB300 almost for free and "
        f"HIP graphs charge for, {serial_gap('moe') / 1e3 + serial_gap('attention') / 1e3:.1f} ms is MoE and attention "
        "kernel time, and the rest is spread thin. Along the way it also gave us a reminder that a GPU benchmark can "
        "be a CPU benchmark in disguise, and a placement bug on our own machine that no GPU counter would have shown.",
        f"这次 GB300 实验本来是想拿到一个目标值，结果得到的是一份拆解：MI355X 每个 verify 多出的 {GAP_US / 1e3:.1f} ms 里，"
        f"{CONC_MS:.1f} ms 是并发，CUDA graph 让 GB300 几乎白得，HIP graph 却要为它付出代价；"
        f"{serial_gap('moe') / 1e3 + serial_gap('attention') / 1e3:.1f} ms 是 MoE 和 attention 的 kernel 耗时；其余分散在各处。"
        "顺带还得到一个提醒，GPU 测试有时其实是在测 CPU；以及我们自己机器上的一个放置 bug，任何 GPU 计数器都看不出来。"))
    sec.append("</section>")
    return "\n".join(sec)


CSS = r"""
:root{--paper:#fffcf7;--panel:#faf5ec;--wash:#f3ecdf;--rule:#e8ddcb;--rule-2:#d6c8b1;--ink:#2e2822;
--ink-2:#5c5147;--muted:#7a6d5e;--gb:#2b6b64;--mi:#c8562a;--mi-ink:#a8431d;--local:#5e8d3e;--remote:#c4861c;
--display:'Vollkorn','Noto Serif SC',Georgia,serif;--body:'Radio Canada','Noto Sans SC',sans-serif;
--mono:'Sono','Noto Sans SC',monospace}
*{box-sizing:border-box}
@media (prefers-reduced-motion:no-preference){html{scroll-behavior:smooth}}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--body);font-size:17.5px;line-height:1.7;
-webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
a{color:var(--gb);text-decoration:underline;text-decoration-color:#2b6b6455;text-decoration-thickness:1px;
text-underline-offset:3px}a:hover{text-decoration-color:var(--gb)}
:focus-visible{outline:2px solid var(--gb);outline-offset:3px;border-radius:2px}
code{font-family:var(--mono);font-size:.84em;background:var(--wash);padding:.1em .38em;border-radius:4px;color:#3b3129}
pre{background:var(--panel);border:1px solid var(--rule);padding:16px 18px;overflow-x:auto;border-radius:8px;line-height:1.55}
pre code{background:none;padding:0;font-size:13px;color:var(--ink)}
.layout{display:grid;grid-template-columns:210px minmax(0,1fr);gap:52px;max-width:1240px;margin:0 auto;padding:0 32px}
.layout>*{min-width:0}p code,li code{overflow-wrap:anywhere}
nav.rail{position:sticky;top:0;align-self:start;height:100vh;overflow-y:auto;padding:42px 0 24px;font-size:14px;line-height:1.45}
nav.rail .back{display:inline-block;margin-bottom:28px;color:var(--ink-2);text-decoration:none}
nav.rail .back:hover{color:var(--gb)}
nav.rail ol{list-style:none;margin:0;padding:0;border-left:1px solid var(--rule)}
nav.rail li a{display:block;padding:5px 0 5px 16px;margin-left:-1px;border-left:2px solid transparent;color:var(--muted);
text-decoration:none}
nav.rail li a:hover{color:var(--ink);border-left-color:var(--rule-2)}
header.masthead{padding:64px 0 26px;max-width:880px}
.kicker{font-family:var(--display);font-style:italic;font-size:18px;color:var(--ink-2)}
h1{font-family:var(--display);font-weight:700;font-size:clamp(34px,4.6vw,56px);line-height:1.06;margin:10px 0 18px;
letter-spacing:-.012em;font-variant-numeric:lining-nums}
.subtitle{font-size:19.5px;line-height:1.6;color:var(--ink-2);max-width:740px}
figure.hero{margin:26px 0 6px;max-width:880px}
figure.hero figcaption{font-size:13.5px;color:var(--ink-2);margin-top:4px}
.spec{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:0 32px;margin-top:26px;
border-top:1px solid var(--rule)}
.spec div{padding:10px 0 11px;border-bottom:1px solid var(--rule);font-size:14.5px;line-height:1.5}
.spec b{display:block;font-weight:600;font-size:13px;color:var(--muted);margin-bottom:2px}
.spec .sw{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px}
main.article section{max-width:720px;padding:6px 0 22px}
h2{font-family:var(--display);font-weight:600;font-size:30px;line-height:1.2;margin:56px 0 14px;letter-spacing:-.005em}
h3{font-family:var(--display);font-weight:600;font-size:21px;line-height:1.3;margin:34px 0 8px}
h2 .no,h3 .no{color:var(--muted);font-weight:500;margin-right:.4em}
h2,h3,.plate-title,.callout-title,table.data tr.group td{font-variant-numeric:lining-nums}
p{margin:0 0 16px}ul{padding-left:22px;margin:0 0 18px}li{margin:0 0 9px}li::marker{color:var(--muted)}
.callout{background:var(--panel);border:1px solid var(--rule);border-left:3px solid var(--remote);border-radius:6px;
padding:14px 20px 4px;margin:24px 0}
.callout-title{font-family:var(--display);font-weight:600;font-size:17.5px;margin-bottom:6px}
figure.plate{margin:32px 0 36px;background:var(--panel);border:1px solid var(--rule);border-radius:10px;padding:18px 20px 14px}
figure.plate.wide{width:min(1040px,calc(100vw - 346px));max-width:none}
.plate-meta{display:flex;gap:12px;align-items:baseline;margin-bottom:10px}
.plate-num{font-family:var(--display);font-style:italic;font-size:16px;color:var(--muted)}
.plate-title{font-family:var(--display);font-weight:600;font-size:18.5px}
figure svg{width:100%;height:auto;display:block}
.caption{color:var(--ink-2);font-size:14px;line-height:1.55;margin-top:10px}
svg .grid{stroke:#ece2d2;stroke-width:1}svg .lbl{fill:var(--ink);font:13px var(--body)}
svg .tick{fill:var(--muted);font:11.5px var(--body);font-variant-numeric:tabular-nums}
svg .axis{fill:var(--muted);font:12px var(--body)}
svg .val{fill:var(--ink);font:500 12px var(--body);font-variant-numeric:tabular-nums}
svg .legend{fill:var(--ink-2);font:12.5px var(--body)}
svg .head{fill:var(--ink);font:600 13px var(--body)}
svg .note{fill:var(--ink-2);font:12px var(--body)}
svg .box{fill:#fff;stroke:var(--rule-2);stroke-width:1}svg .box.gpu{fill:var(--wash)}
svg .edge{stroke:var(--muted);stroke-width:1.5}
.tablewrap{overflow-x:auto;margin:12px 0 24px}
table.data{border-collapse:collapse;width:100%;font-size:14.5px;font-variant-numeric:tabular-nums}
table.data th,table.data td{border-bottom:1px solid var(--rule);padding:8px 10px;text-align:right}
table.data th:first-child,table.data td:first-child{text-align:left}
table.data th{color:var(--muted);font-weight:600;font-size:13px;vertical-align:bottom;border-bottom-color:var(--rule-2)}
table.data tr.total td{font-weight:600;border-top:1px solid var(--rule-2);border-bottom:none}
table.data td{white-space:nowrap}table.data td:first-child{white-space:normal}
table.wide-table{font-size:13px}table.wide-table th:nth-child(2){text-align:left}
table.wide-table td:nth-child(2){text-align:left;white-space:normal;min-width:200px}
table.wide-table td:first-child{white-space:nowrap}table.wide-table td:last-child{white-space:normal;min-width:90px}
table.data tr.group td{text-align:left;font-family:var(--display);font-weight:600;font-size:15.5px;padding-top:20px}
table.data tr.group.gb300 td{color:var(--gb)}table.data tr.group.mi355x td{color:var(--mi-ink)}
p.tnote{font-size:13.5px;color:var(--ink-2)}
main.article section#appendix{max-width:1040px}
footer.colophon{max-width:880px;margin:44px 0 72px;padding-top:20px;border-top:1px solid var(--rule);font-size:13.5px;
color:var(--ink-2);display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px 28px}
footer.colophon b{display:block;font-weight:600;font-size:13px;color:var(--muted);margin-bottom:2px}
.lang-toggle{position:fixed;top:16px;right:18px;z-index:10;display:flex;border:1px solid var(--rule-2);
background:var(--paper);border-radius:999px;overflow:hidden}
.lang-toggle button{font:13px var(--body);color:var(--ink-2);background:none;border:none;padding:6px 13px;cursor:pointer}
body[data-lang="en"] .lang-toggle button[data-set="en"],body[data-lang="zh"] .lang-toggle button[data-set="zh"]{
background:var(--ink);color:var(--paper)}
@media (max-width:980px){.layout{grid-template-columns:1fr;padding:0 18px}nav.rail{position:static;height:auto;
padding:68px 0 0}nav.rail ol{display:flex;flex-wrap:wrap;gap:4px 16px;border:none}nav.rail li a{padding:2px 0;border:none}
figure.plate.wide{width:auto}h2{font-size:26px}}
body[data-lang="en"] [lang="zh"]:not(html){display:none !important}
body[data-lang="zh"] [lang="en"]:not(html){display:none !important}
"""

TOC = [("prologue", "Prologue", "序"), ("conclusions", "Conclusions", "结论"),
       ("contract", "1 · Held fixed", "1 · 固定条件"), ("calibration", "2 · Calibration", "2 · 校准"),
       ("main-moved", "3 · Main moved", "3 · main 的变化"), ("placement", "4 · Placement", "4 · 放置"),
       ("acceptance", "5 · Acceptance", "5 · 接受长度"), ("anatomy", "6 · Anatomy", "6 · 周期构成"),
       ("concurrency", "7 · Concurrency", "7 · 并发"), ("lessons", "8 · Lessons", "8 · 可借鉴的"),
       ("next", "9 · Next", "9 · 下一步"), ("appendix", "A · Every arm", "A · 所有 arm"),
       ("reproduce", "10 · Reproduce", "10 · 复现"), ("epilogue", "Epilogue", "后记")]


def page():
    body = build()
    rail = ('<nav class="rail"><a class="back" href="/sources/">' + S("← The Library", "← 文库") + "</a><ol>"
            + "".join(f'<li><a href="#{i}">{S(e, z)}</a></li>' for i, e, z in TOC) + "</ol></nav>")
    spec = [("model", "DeepSeek-V4.1 Flash @ dba1be0a", "DeepSeek-V4.1 Flash @ dba1be0a"),
            ("request", "BS=1 · 4,096 random ids · 1,024 out", "BS=1 · 4,096 个随机 id · 输出 1,024"),
            ("parallelism", "TP4 / EP4 · cookbook cells", "TP4 / EP4 · cookbook cell"),
            (f'<span class="sw" style="background:{GB}"></span>GB300',
             "4× GB300 (1 NVL72 tray) · CUDA 13.2 · main ffac53d779", "4× GB300（1 个 NVL72 托盘）· CUDA 13.2 · main ffac53d779"),
            (f'<span class="sw" style="background:{MI}"></span>MI355X',
             "4× MI355X · ROCm 7.2 · dsv41-amd-main e2e824dc58", "4× MI355X · ROCm 7.2 · dsv41-amd-main e2e824dc58"),
            ("runs", f"{sum(a['launches'] for a in ARMS.values())} timed launches · 2 real-text launches · 4 nsys "
                     "profiles · 2026-09-24/25",
             f"{sum(a['launches'] for a in ARMS.values())} 次计时启动 · 2 次真实文本启动 · 4 个 nsys profile · 2026-09-24/25")]
    spec_html = "".join(f"<div><b>{k}</b>{S(e, z)}</div>" for k, e, z in spec)
    mast = ('<header class="masthead"><div class="kicker">Experiment 005 · '
            + S("DeepSeek-V4.1 Flash on SGLang", "SGLang 上的 DeepSeek-V4.1 Flash") + "</div>"
            '<h1>GB300 vs MI355X, one token at a time</h1>'
            + P(f"We reproduced SGLang's GB300 numbers and measured where MI355X's 2× per step comes from. Turning off GB300's "
                f"side streams takes its {f2(gb_cyc)} ms verify cycle to {f2(w3_cyc['serial'])} ms, against MI355X's "
                f"{f2(mi_cyc)} ms: {CONC_MS * 1e3 / GAP_US:.0%} of the gap is stream concurrency, which HIP graphs make "
                "expensive, and most of the rest is MoE and attention kernel time. A "
                f"{pct(e1n['tps'], e1b['tps']):.0f}% swing on GB300 turned out to be CPU placement.",
                f"我们复现了 SGLang 的 GB300 数据，并实测了 MI355X 每步慢 2 倍的原因。关掉 GB300 的侧 stream 后，它 {f2(gb_cyc)} ms 的 "
                f"verify 周期变成 {f2(w3_cyc['serial'])} ms，MI355X 是 {f2(mi_cyc)} ms：差距里 {CONC_MS * 1e3 / GAP_US:.0%} 是 stream "
                "并发，而 HIP graph 让并发变得很贵；其余大部分是 MoE 和 attention 的 kernel 耗时。GB300 上 "
                f"{pct(e1n['tps'], e1b['tps']):.0f}% 的波动，最后查明是 CPU 放置造成的。", "subtitle")
            + '<figure class="hero">' + hero_cycle() + '<figcaption>'
            + S("Same checkpoint, prompt and client on both machines. GB300 is NUMA-bound, at its default stream layout "
                "and with every side stream off; MI355X runs at SGLang's default placement. Plain decode: "
                f"{f2(gb_step)}, {f2(w3_step['serial'])} and {f2(mi_step)} ms per step.",
                "两台机器用同一个 checkpoint、同一个 prompt 和同一个客户端。GB300 绑定 NUMA，分别为默认 stream 布局和关闭所有侧 "
                f"stream；MI355X 为 SGLang 默认放置。plain decode 每步分别是 {f2(gb_step)}、{f2(w3_step['serial'])} 和 "
                f"{f2(mi_step)} ms。")
            + '</figcaption></figure>'
            + f'<div class="spec">{spec_html}</div></header>')
    colophon = ('<footer class="colophon">'
                f"<div><b>{S('data', '数据')}</b><a href=\"{DATA}\">data/{SLUG}/</a></div>"
                f"<div><b>{S('protocol', '测试流程')}</b>{S('BBuf random-dspark (b001f347)', 'BBuf random-dspark（b001f347）')}</div>"
                f"<div><b>{S('type', '字体')}</b>Vollkorn · Radio Canada · Sono · Noto Serif SC · Noto Sans SC</div>"
                f"<div><b>{S('palette', '配色')}</b>{S('linen and walnut ink; spruce for GB300, persimmon for MI355X, fern and ochre for local and remote placement', '亚麻白底、胡桃墨字；云杉绿代表 GB300，柿子橙代表 MI355X，蕨绿和赭黄代表本地与远端放置')}</div>"
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
            f'stroke="{MUTED}" stroke-width="1.2"/></pattern></defs></svg>')
    return ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" "
            "content=\"width=device-width,initial-scale=1\"><title>Experiment 005 · DeepSeek-V4.1 Flash, GB300 vs MI355X"
            "</title><meta name=\"description\" content=\"BS=1 DeepSeek-V4.1 Flash on SGLang: GB300 calibration, NUMA "
            "placement, acceptance traps and a per-category verify-cycle comparison with MI355X.\">"
            "<link rel=\"preconnect\" href=\"https://fonts.googleapis.com\"><link rel=\"preconnect\" "
            "href=\"https://fonts.gstatic.com\" crossorigin><link href=\"https://fonts.googleapis.com/css2?family=Vollkorn:"
            "ital,wght@0,400..700;1,400..600&family=Radio+Canada:ital,wght@0,400..700;1,400..500&family=Sono:wght@400..600"
            "&family=Noto+Serif+SC:wght@500;600;700&family=Noto+Sans+SC:wght@400;500;700&display=swap\" rel=\"stylesheet\">"
            "<style>" + CSS + "</style></head>\n"
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
