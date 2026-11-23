"""Render public/sources/flydsl-compiler-redesign.html from the experiment outputs.

Static inline SVG only: chart coordinates are computed here, at authoring time, from the files in
data/flydsl-compiler-redesign/; the page runs no rendering code. Entry point: make_page.py.
"""

import csv
import json
import math
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1]
SUMMARY = json.loads((DATA_DIR / "summary.json").read_text())
MINING = json.loads((DATA_DIR / "agent_mining_summary.json").read_text())
COMPILE = list(csv.DictReader(open(DATA_DIR / "compile_breakdown.csv")))
FAILURES = json.loads((DATA_DIR / "failures.json").read_text())
OUT = DATA_DIR.parents[1] / "public" / "sources" / "flydsl-compiler-redesign.html"

V042 = "https://github.com/ROCm/FlyDSL/blob/cb770e9a96b3454fa9cbbbcdc16dceeda1d8e4e7"
MAIN = "https://github.com/ROCm/FlyDSL/blob/fe81c40773431335c9f77ccd196a7adc0e3126d7"
LLVM = "https://github.com/llvm/llvm-project/blob/941a04e69ee8fe4c7a162b2f1e215aa8df867534"
DATA = "https://github.com/jhinpan/jinnpan.com/tree/main/data/flydsl-compiler-redesign"
DATAF = "https://github.com/jhinpan/jinnpan.com/blob/main/data/flydsl-compiler-redesign"


def arm(label):
    return SUMMARY.get(label)


def ratio(label):
    d = arm(label)
    return d["ratio"] if d and d["verdict"] == "scored" else None


def x(r):
    return f"{r:.2f}×" if r is not None else "—"


def x3(r):
    return f"{r:.3f}×" if r is not None else "—"


def ci(label):
    d = arm(label)
    return f"[{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]" if d and d.get("ci95") else "—"


def more_time(r):
    return f"{(1 / r - 1) * 100:+.0f}%" if r else "—"


def shape(label, sid):
    d = arm(label)
    for s in d["per_shape"]:
        if s["shape"] == sid:
            return s
    return None


def public_shapes(label):
    return {s["shape"]: s for s in arm(label)["per_shape"] if not s["drawn"] and s["ratio"]}


def tests(label):
    """'1 of 880' style count of FlyDSL tests the arm broke, read from the receipt reason."""
    import re
    reason = arm(label)["reason"] or ""
    f = re.search(r"'failed': (\d+)", reason)
    p = re.search(r"'passed': (\d+)", reason)
    if f and p:
        failed, passed = int(f.group(1)), int(p.group(1))
        return failed, failed + passed
    return None, None


# ---------------------------------------------------------------------------------------------
# charts
# ---------------------------------------------------------------------------------------------

def bar_chart(rows, lo, hi, width=860, label_w=330, title_id="c"):
    """Horizontal speed-vs-stock bars from 1.0, CI whiskers; rows: dict(en, zh, label|value, cls, note_en, note_zh)."""
    row_h, top = 30, 34
    h = top + row_h * len(rows) + 34
    x0, x1 = label_w + 10, width - 90

    def X(v):
        return x0 + (min(max(v, lo), hi) - lo) / (hi - lo) * (x1 - x0)

    out = [f'<svg viewBox="0 0 {width} {h}" role="img" aria-labelledby="{title_id}-t"><title id="{title_id}-t">speed relative to the stock kernel</title>']
    # grid
    t = lo
    step = 0.05
    while t <= hi + 1e-9:
        xx = X(t)
        cls = "ax-one" if abs(t - 1.0) < 1e-9 else "ax-grid"
        out.append(f'<line class="{cls}" x1="{xx:.1f}" y1="{top - 8}" x2="{xx:.1f}" y2="{h - 26}"/>')
        out.append(f'<text class="ax-lab" x="{xx:.1f}" y="{h - 10}" text-anchor="middle">{t:.2f}×</text>')
        t = round(t + step, 4)
    out.append(f'<text class="ax-lab" x="{x0}" y="16" lang="en">← slower than stock</text>')
    out.append(f'<text class="ax-lab" x="{x0}" y="16" lang="zh">← 比 stock 慢</text>')
    out.append(f'<text class="ax-lab" x="{x1}" y="16" text-anchor="end" lang="en">faster than stock →</text>')
    out.append(f'<text class="ax-lab" x="{x1}" y="16" text-anchor="end" lang="zh">比 stock 快 →</text>')
    for i, r in enumerate(rows):
        y = top + i * row_h
        cy = y + row_h / 2
        out.append(f'<text class="row-lab" x="{label_w}" y="{cy + 4:.1f}" text-anchor="end" lang="en">{r["en"]}</text>')
        out.append(f'<text class="row-lab" x="{label_w}" y="{cy + 4:.1f}" text-anchor="end" lang="zh">{r["zh"]}</text>')
        v = r.get("value")
        if v is None:
            out.append(f'<text class="row-bad" x="{X(1.0) - 6:.1f}" y="{cy + 4:.1f}" text-anchor="end" lang="en">✕ {r["note_en"]}</text>')
            out.append(f'<text class="row-bad" x="{X(1.0) - 6:.1f}" y="{cy + 4:.1f}" text-anchor="end" lang="zh">✕ {r["note_zh"]}</text>')
            continue
        a, b = sorted((X(1.0), X(v)))
        out.append(f'<rect class="bar {r["cls"]}" x="{a:.1f}" y="{y + 8}" width="{max(b - a, 1.5):.1f}" height="{row_h - 16}"/>')
        if r.get("ci"):
            c0, c1 = X(r["ci"][0]), X(r["ci"][1])
            out.append(f'<line class="whisk" x1="{c0:.1f}" y1="{cy:.1f}" x2="{c1:.1f}" y2="{cy:.1f}"/>')
        tx = X(v) + (6 if v >= 1 else -6)
        anchor = "start" if v >= 1 else "end"
        if v < 1:
            tx = max(tx, x0 + 40)
        out.append(f'<text class="val" x="{x1 + 8}" y="{cy + 4:.1f}">{v:.3f}×</text>')
    out.append("</svg>")
    return "\n".join(out)


def per_shape_plot(series, width=860, height=330, title_id="ps"):
    """Dots per public shape (sorted by stock time), one color per arm; y = speed vs stock."""
    base = public_shapes(series[0][0])
    shapes = sorted(base, key=lambda s: base[s]["stock_us"])
    lo, hi = 0.75, 1.15
    left, right, top, bottom = 56, width - 16, 20, height - 46

    def Y(v):
        return bottom - (min(max(v, lo), hi) - lo) / (hi - lo) * (bottom - top)

    n = len(shapes)

    def Xi(i):
        return left + (i + 0.5) * (right - left) / n

    out = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-labelledby="{title_id}-t"><title id="{title_id}-t">per-shape speed of each arm on preshuffle-a8</title>']
    v = lo
    while v <= hi + 1e-9:
        cls = "ax-one" if abs(v - 1.0) < 1e-9 else "ax-grid"
        out.append(f'<line class="{cls}" x1="{left}" y1="{Y(v):.1f}" x2="{right}" y2="{Y(v):.1f}"/>')
        out.append(f'<text class="ax-lab" x="{left - 6}" y="{Y(v) + 4:.1f}" text-anchor="end">{v:.2f}×</text>')
        v = round(v + 0.05, 4)
    for i, s in enumerate(shapes):
        if i % 4 == 0:
            out.append(f'<text class="ax-lab" x="{Xi(i):.1f}" y="{bottom + 16}" text-anchor="middle">{base[s]["stock_us"]:.0f}</text>')
    out.append(f'<text class="ax-lab" x="{(left + right) / 2:.0f}" y="{height - 6}" text-anchor="middle" lang="en">public shapes, sorted by stock time (µs)</text>')
    out.append(f'<text class="ax-lab" x="{(left + right) / 2:.0f}" y="{height - 6}" text-anchor="middle" lang="zh">公开 shape，按 stock 耗时排序（µs）</text>')
    for k, (label, cls, dx) in enumerate(series):
        pts = public_shapes(label)
        for i, s in enumerate(shapes):
            if s in pts:
                out.append(f'<circle class="dot {cls}" cx="{Xi(i) + dx:.1f}" cy="{Y(pts[s]["ratio"]):.1f}" r="3.2"/>')
    out.append("</svg>")
    return "\n".join(out), shapes


def oracle(labels):
    """Per public shape, the best of stock (1.0) and the arms; geomean of that over the shapes.

    `strict` credits an arm only where its 95% interval clears 1.0, and then only its lower bound.
    """
    sets = [public_shapes(l) for l in labels]
    shapes = sorted(set(sets[0]).intersection(*sets[1:]))
    point = strict = 0.0
    for s in shapes:
        point += math.log(max([1.0] + [d[s]["ratio"] for d in sets]))
        strict += math.log(max([1.0] + [d[s]["ci95"][0] for d in sets if d[s]["ci95"] and d[s]["ci95"][0] > 1.0]))
    return math.exp(point / len(shapes)), math.exp(strict / len(shapes)), len(shapes)


def hbar_pairs(items, width=860, label_w=300, title_id="hb", fmt="{:.0%}", maxv=None):
    row_h, top = 26, 10
    h = top + row_h * len(items) + 12
    x0, x1 = label_w + 12, width - 70
    mv = maxv or max(v for _en, _zh, v, _c in items)
    out = [f'<svg viewBox="0 0 {width} {h}" role="img" aria-labelledby="{title_id}-t"><title id="{title_id}-t">bar chart</title>']
    for i, (en, zh, v, cls) in enumerate(items):
        y = top + i * row_h
        out.append(f'<text class="row-lab" x="{label_w}" y="{y + 17}" text-anchor="end" lang="en">{en}</text>')
        out.append(f'<text class="row-lab" x="{label_w}" y="{y + 17}" text-anchor="end" lang="zh">{zh}</text>')
        w = (x1 - x0) * v / mv
        out.append(f'<rect class="bar {cls}" x="{x0}" y="{y + 5}" width="{max(w, 1.5):.1f}" height="{row_h - 10}"/>')
        out.append(f'<text class="val" x="{x0 + w + 6:.1f}" y="{y + 17}">{fmt.format(v)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def compile_chart(width=860, title_id="cc"):
    rows = [r for r in COMPILE]
    names = {"vector_add": "vectorAdd", "preshuffle_fp8": "preshuffle FP8 GEMM",
             "fp8_gemm_4wave": "fp8_gemm_4wave 256×256", "flash_attn_gfx950": "flash_attn_gfx950 D128"}
    row_h, top, label_w = 46, 34, 210
    h = top + row_h * len(rows) + 40
    x0, x1 = label_w + 12, width - 24
    mx = 3.0

    def X(v):
        return x0 + v / mx * (x1 - x0)

    out = [f'<svg viewBox="0 0 {width} {h}" role="img" aria-labelledby="{title_id}-t"><title id="{title_id}-t">cold compile time by stage</title>']
    for t in [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        out.append(f'<line class="ax-grid" x1="{X(t):.1f}" y1="{top - 6}" x2="{X(t):.1f}" y2="{h - 30}"/>')
        out.append(f'<text class="ax-lab" x="{X(t):.1f}" y="{h - 14}" text-anchor="middle">{t:.1f} s</text>')
    legend = [("seg-trace", "trace + print", "追踪 + 打印"), ("seg-mlir", "MLIR passes", "MLIR pass"),
              ("seg-llvm", "LLVM backend", "LLVM 后端"), ("seg-warm", "a cached call", "命中缓存的一次调用")]
    lx = x0
    for cls, en, zh in legend:
        out.append(f'<rect class="{cls}" x="{lx}" y="8" width="12" height="12"/>')
        out.append(f'<text class="ax-lab" x="{lx + 16}" y="18" lang="en">{en}</text>')
        out.append(f'<text class="ax-lab" x="{lx + 16}" y="18" lang="zh">{zh}</text>')
        lx += 150
    for i, r in enumerate(rows):
        y = top + i * row_h
        trace = float(r["trace_s"]) + float(r["verify_print_s"])
        mlir = float(r["mlir_passes_1_14_s"])
        llvm = float(r["llvm_backend_s"])
        warm = float(r["warm_inmem_hit_call_s"])
        out.append(f'<text class="row-lab" x="{label_w}" y="{y + 18}" text-anchor="end">{names[r["kernel"]]}</text>')
        xs = X(0)
        for cls, v in (("seg-trace", trace), ("seg-mlir", mlir), ("seg-llvm", llvm)):
            out.append(f'<rect class="{cls}" x="{xs:.1f}" y="{y + 6}" width="{max(X(v) - X(0), 1):.1f}" height="16"/>')
            xs += X(v) - X(0)
        out.append(f'<text class="val" x="{xs + 6:.1f}" y="{y + 19}">{float(r["cold_compile_s"]):.2f} s</text>')
        out.append(f'<rect class="seg-warm" x="{X(0):.1f}" y="{y + 26}" width="{max(X(warm) - X(0), 1):.1f}" height="8"/>')
        out.append(f'<text class="val small" x="{X(warm) + 6:.1f}" y="{y + 34}">{warm * 1000:.0f} ms</text>')
    out.append("</svg>")
    return "\n".join(out)
