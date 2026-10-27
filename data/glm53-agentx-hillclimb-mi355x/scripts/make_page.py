#!/usr/bin/env python3
"""Build public/sources/glm53-agentx-hillclimb-mi355x.html from this data directory alone.

    python3 data/glm53-agentx-hillclimb-mi355x/scripts/make_page.py
"""
import csv
import html
import json
import math
import re
from pathlib import Path

D = Path(__file__).resolve().parents[1]
REPO = D.parents[1]
SLUG = D.name
OUT = REPO / "public" / "sources" / f"{SLUG}.html"
GH = "https://github.com"
DATA_URL = f"{GH}/jhinpan/jinnpan.com/tree/main/data/{SLUG}"
BLOB = f"{GH}/jhinpan/jinnpan.com/blob/main/data/{SLUG}"
C1, C2, C3 = "cohort1-bf16kv-900s", "cohort2-fp8kv-900s", "cohort3-fp8kv-3600s"
CONCS = (1, 2, 4, 10)


def J(rel):
    return json.loads((D / rel).read_text())


def E(s):
    return html.escape(str(s), quote=False)


def a(url, text):
    return f'<a href="{url}">{text}</a>'


def pr(n, repo="sgl-project/sgl-amd-mono"):
    return a(f"{GH}/{repo}/pull/{n}", f"{repo.split('/')[1]}#{n}")


def sg(x, nd=1):
    """Signed number with a real minus sign."""
    s = f"{abs(x):.{nd}f}"
    if float(s) == 0:
        return s
    return ("+" if x > 0 else "−") + s


def mcnemar_p(b, c):
    n, k = b + c, min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


ZH_FIX = re.compile(r"([。，：])(?=[^\s<）」』”》】、。，；：！？)]|<(?!/))")


def zh(s):
    return ZH_FIX.sub(r"\1 ", s)


def bi(tag, en, zh_text, attrs=""):
    at = f" {attrs}" if attrs else ""
    return f'<{tag}{at} lang="en">{en}</{tag}><{tag}{at} lang="zh">{zh(zh_text)}</{tag}>'


def sp(en, zh_text):
    return bi("span", en, zh_text)


def P(en, zh_text, cls=""):
    return bi("p", en, zh_text, f'class="{cls}"' if cls else "")


def UL(items, cls=""):
    c = f' class="{cls}"' if cls else ""
    return f"<ul{c}>" + "".join(bi("li", en, z) for en, z in items) + "</ul>"


def OL(items):
    return "<ol>" + "".join(bi("li", en, z) for en, z in items) + "</ol>"


def callout(en, zh_text):
    return f'<aside class="callout">{P(en, zh_text)}</aside>'


def table(head, rows, cls=""):
    """head: list of (en, zh) or str; rows: list of lists of str or (en, zh)."""
    cell = lambda x: sp(*x) if isinstance(x, tuple) else x  # noqa: E731
    th = "".join(f"<th>{cell(h)}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{cell(x)}</td>" for x in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def plate(num, title, svg, caption):
    return (f'<figure class="plate"><div class="plate-meta"><span class="pn">Plate {num}</span>'
            f"{sp(*title)}</div>{svg}<figcaption class=\"caption\">{sp(*caption)}</figcaption></figure>")


def section(sid, num, title, body):
    return f'<section id="{sid}"><h2><span class="num">{num}</span>{sp(*title)}</h2>{body}</section>'


def svg_text(x, y, en, zh_text, cls="lbl", anchor="start"):
    return (f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}" lang="en">{E(en)}</text>'
            f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}" lang="zh">{E(zh(zh_text))}</text>')


def nt(x, y, s, cls="lbl", anchor="start"):
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{E(s)}</text>'


# ----------------------------------------------------------------------------- data
c3 = J(f"{C3}/analysis/compare_round1_base_cand8.json")
pts = {int(p["point"].split("c")[-1]): p for p in c3["points"]}
noise3 = c3["noise_floor"]
noise1 = J(f"{C1}/analysis/compare_round1_base_cand.json")["noise_floor"]
noise2 = J(f"{C2}/analysis/compare_round1_base_cand8.json")["noise_floor"]
pe = {c: J(f"{c}/analysis/pair_effects.json") for c in (C1, C2, C3)}
eff = {c: {(e["conc"], e["run"]): e for e in pe[c]["effects"]} for c in pe}
tail = {c: {(t["conc"], t["run"], t["subset"]): t for t in pe[c]["tail"]} for c in pe}
gsm3 = J(f"{C3}/analysis/gsm8k_pairs.json")
gsm1, gsm2 = J(f"{C1}/analysis/gsm8k_paired.json"), J(f"{C2}/analysis/gsm8k_paired.json")
par = {c["key"].split("/")[2]: c for c in J(f"{C3}/analysis/pareto_coverage.json")}
refs3 = J(f"{C3}/analysis/refs_compare.json")
prof = J("ttft-profile/summary.json")
pb, pz = prof["base_fp8_e4m3_kv"], prof["b0_bf16_kv"]
e2e = J("fp8kv-validation/e2e_checks.json")
idx = J("index.json")
osl_pub = J("references/osl_published.json")["rows"]


def run3(rid):
    return J(f"{C3}/runs/{rid}/run.json")


cov = {c: idx[C3][f"agentx-glm53-l2ov-c{c}-cand8-r1"]["coverage"] for c in CONCS}
share = {c: cov[c]["share_running_le_2"] for c in CONCS}
hist_share = {c: {k: v / cov[c]["decode_batch_lines"] for k, v in cov[c]["running_req_histogram"].items()} for c in CONCS}
gcov = {k: idx[C3][k]["coverage"]["share_running_le_2"] for k in ("gsm8k-glm53-l2ov-c2-cand8q", "gsm8k-glm53-l2ov-c10-cand8q", "gsm8k-glm53-l2ov-c10-cand8q-r2")}
g = lambda c, m: 100 * pts[c]["gain"][m]  # noqa: E731
B = lambda c, m: pts[c]["baseline"][m]  # noqa: E731
K = lambda c, m: pts[c]["candidate"][m]  # noqa: E731


def rng(vals, nd=0):
    lo, hi = min(vals), max(vals)
    if nd == 0 and max(abs(lo), abs(hi)) < 5:
        nd = 1
    a_, b_ = sorted((lo, hi), key=abs)
    return f"{sg(a_, nd)}% to {sg(b_, nd)}%" if round(lo, nd) != round(hi, nd) else f"{sg(lo, nd)}%"


def zrng(vals, nd=0):
    return rng(vals, nd).replace(" to ", " 到 ")


cand_runs = {C1: ("cand-r1", "cand-r2"), C2: ("cand8-r1", "cand8-r2")}
pv = {c: {m: [eff[c][(k, r)][m] for k in (1, 2, 4) for r in cand_runs[c]] for m in ("time_to_first_token", "inter_token_latency", "request_latency")} for c in (C1, C2)}
tv = {c: [tail[c][(k, r, "slowest 10%")]["e2el_per_token"] for k in (1, 2, 4) for r in cand_runs[c]] for c in (C1, C2)}
mean2 = lambda c, r, sub, m: sum(tail[c][(1, x, sub)][m] for x in r) / 2  # noqa: E731

strict3 = [r for r in gsm3 if r["filter"] == "strict-match"]
pairs3 = [r for r in strict3 if r["kind"] == "base vs cand8"]
ob, oc = sum(r["only_a"] for r in pairs3), sum(r["only_b"] for r in pairs3)
rerun_c = next(r for r in strict3 if r["kind"] == "rerun of cand8")


def gsm_rows(rid):
    with (D / f"{C3}/gsm8k/{rid}.csv").open() as f:
        return {int(r["doc_id"]): r for r in csv.DictReader(f)}


gb, gc = gsm_rows("gsm8k-glm53-l2ov-c10-baseq"), gsm_rows("gsm8k-glm53-l2ov-c10-cand8q")
lost = [k for k in gb if gb[k]["strict_match"] == "1" and gc[k]["strict_match"] == "0"]
runaway = sorted(int(gc[k]["response_chars"]) for k in lost if int(gc[k]["response_chars"]) > 40000)

ref = lambda curve, c: next((r for r in refs3 if r["curve"].startswith(curve) and r["conc"] == c), None)  # noqa: E731
tile, b300 = ref("GLM-5.3 MI355X TileRT", 1), {c: ref("GLM-5.3 B300", c) for c in (1, 4, 10)}
rel = lambda x, y: 100 * (x / y - 1)  # noqa: E731
osl_local = {c: run3(f"agentx-glm53-l2ov-c{c}-cand8-r1")["agentx_summary"]["request_metrics"]["tokens"]["output_actual"] for c in (1, 4, 10)}
osl_b300 = {r["conc"]: r for r in osl_pub if r["model"] == "glm5.3" and r["hardware"] == "b300"}
osl_ratio = [osl_local[c]["p90"] / osl_b300[c]["p90_output_tokens_actual"] for c in (1, 4, 10) if c in osl_b300]
e2ec = lambda rid, m="layer.kv.cosine": e2e[rid]["metrics_first_test"].get(m)  # noqa: E731

gain_lo = lambda m, cs=(1, 2, 4): round(min(g(c, m) for c in cs))  # noqa: E731
gain_hi = lambda m, cs=(1, 2, 4): round(max(g(c, m) for c in cs))  # noqa: E731
itl3 = {c: eff[C3][(c, "cand8-r1")]["inter_token_latency"] for c in CONCS}
ttft3 = {c: eff[C3][(c, "cand8-r1")]["time_to_first_token"] for c in CONCS}
e2el3 = {c: eff[C3][(c, "cand8-r1")]["request_latency"] for c in CONCS}
itl1 = [abs(x) for x in pv[C1]["inter_token_latency"]]
itl_lo, itl_hi = round(min(itl1)), round(max(itl1))
ttft_up = round(100 * (pz["ttft_median_ms"] / pb["ttft_median_ms"] - 1))
k_ratio = pz["grid2080_per_layer_us_median"] / pb["grid2080_per_layer_us_median"]
other_delta = abs((pz["kernel_total_ms"] - pz["tilelang_main_kernel_ms"]) - (pb["kernel_total_ms"] - pb["tilelang_main_kernel_ms"]))

# ----------------------------------------------------------------------------- cover
def cover_svg():
    """Ridgelines of per-request ITL: one curve per 20-minute window of each 3600-s run."""
    runs = [(c, grp, f"agentx-glm53-l2ov-c{c}-{arm}-r1") for c in CONCS for grp, arm in (("base", "base"), ("mono", "cand8"))]
    ridges = []
    for c, grp, rid in runs:
        with (D / f"{C3}/requests/{rid}.csv").open() as f:
            rows = [r for r in csv.DictReader(f) if r["phase"] == "profiling" and r["cancelled"] == "0" and r["itl_ms"] and float(r["itl_ms"]) > 0]
        t0 = min(float(r["start_s"]) for r in rows)
        for w in range(3):
            xs = [math.log10(float(r["itl_ms"])) for r in rows if w * 1200 <= float(r["start_s"]) - t0 < (w + 1) * 1200]
            if len(xs) > 5:
                ridges.append((grp, xs))
    lo, hi = math.log10(2.6), math.log10(16.0)
    W, H, top, step, amp, n = 1600, 300, 62, 9.4, 50, 120
    grid = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    px = lambda v: 160 + (v - lo) / (hi - lo) * (W - 320)  # noqa: E731
    out = [f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid slice" aria-hidden="true">']
    for i, (grp, xs) in enumerate(ridges):
        m = sum(xs) / len(xs)
        sd = (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5
        h = max(0.02, 1.06 * sd * len(xs) ** -0.2)
        dens = [sum(math.exp(-0.5 * ((gx - x) / h) ** 2) for x in xs) for gx in grid]
        peak = max(dens)
        keep = [k for k, d in enumerate(dens) if d / peak > 0.004]
        base_y = top + i * step
        pts = [(px(grid[k]), base_y - amp * dens[k] / peak) for k in range(keep[0], keep[-1] + 1)]
        d_attr = (f"M{pts[0][0] - 60:.0f},{base_y:.1f} " + " ".join(f"L{x:.0f},{y:.1f}" for x, y in pts)
                  + f" L{pts[-1][0] + 60:.0f},{base_y:.1f}")
        out.append(f'<path d="{d_attr}" class="ridge {grp}"/>')
    out.append("</svg>")
    return "".join(out)


# ----------------------------------------------------------------------------- plates
ARROW = '<defs><marker id="ah" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 z" class="ahd"/></marker></defs>'


def flow(steps, title_y=None):
    w, gap, x0 = 186, 20, 10
    out = [f'<svg viewBox="0 0 1040 176" role="img" aria-label="steps">{ARROW}']
    for i, (state, t1e, t1z, t2e, t2z, ce, cz) in enumerate(steps):
        x = x0 + i * (w + gap)
        out.append(f'<rect x="{x}" y="20" width="{w}" height="112" rx="10" class="box st-{state}"/>')
        out.append(svg_text(x + w / 2, 52, t1e, t1z, "lbl b", "middle"))
        out.append(svg_text(x + w / 2, 76, t2e, t2z, "lbl s", "middle"))
        out.append(svg_text(x + w / 2, 112, ce, cz, f"chip c-{state}", "middle"))
        if i:
            out.append(f'<line x1="{x - gap + 2}" y1="76" x2="{x - 3}" y2="76" class="edge" marker-end="url(#ah)"/>')
    out.append("</svg>")
    return "".join(out)


def plate_steps():
    return flow([
        ("bad", "Test 1 · 900 s", "测试 1 · 900 秒", "BF16 KV + monokernel", "BF16 KV + monokernel", "no-go", "no-go"),
        ("plain", "Profile", "profile", "prefill: BF16 vs FP8 KV", "prefill：BF16 对 FP8 KV", "cause found", "找到原因"),
        ("done", "sgl-amd-mono#60", "sgl-amd-mono#60", "monokernel uses FP8 KV", "monokernel 使用 FP8 KV", "merged", "已合并"),
        ("open", "Test 2 · 900 s", "测试 2 · 900 秒", "FP8 KV + monokernel", "FP8 KV + monokernel", "inside noise", "在噪声内"),
        ("done", "Test 3 · 3600 s", "测试 3 · 3600 秒", "FP8 KV + monokernel", "FP8 KV + monokernel", "pass", "通过"),
    ])


def plate_anatomy():
    ttft_all = mean2(C1, cand_runs[C1], "all", "ttft_share")
    ttft_t10 = mean2(C1, cand_runs[C1], "slowest 10%", "ttft_share")
    rows = (("All requests", "全部请求", "all"), ("Slowest 10%", "最慢的 10%", "slowest 10%"))
    W, x0 = 300, 150
    out = ['<svg viewBox="0 0 1040 250" role="img" aria-label="TTFT share and change">',
           svg_text(20, 26, "TTFT share of E2EL (base group)", "TTFT 占 E2EL 的比例（base 组）", "lbl b"),
           svg_text(560, 26, "Change of E2EL per output token", "每个输出 token 的 E2EL 变化", "lbl b")]
    for i, ((en, zh_, _), s_) in enumerate(zip(rows, (ttft_all, ttft_t10))):
        y = 52 + i * 70
        out.append(svg_text(20, y + 20, en, zh_, "lbl"))
        out.append(f'<rect x="{x0}" y="{y}" width="{W * s_:.1f}" height="30" class="seg ttft"/>')
        out.append(f'<rect x="{x0 + W * s_:.1f}" y="{y}" width="{W * (1 - s_):.1f}" height="30" class="seg dec"/>')
        out.append(nt(x0 + W + 10, y + 20, f"{100 * s_:.0f}%", "lbl b"))
    zero, scale = 900, 5.0
    out.append(f'<line x1="{zero}" y1="44" x2="{zero}" y2="186" class="axis"/>')
    for j, (en, zh_, sub) in enumerate(rows):
        y = 52 + j * 70
        out.append(svg_text(560, y + 20, en, zh_, "lbl"))
        for i, (coh, cls) in enumerate(((C1, "bf16"), (C2, "mono"))):
            v = mean2(coh, cand_runs[coh], sub, "e2el_per_token")
            yy = y + i * 17
            w = abs(v) * scale
            x = zero - w if v < 0 else zero
            out.append(f'<rect x="{x:.1f}" y="{yy}" width="{w:.1f}" height="13" class="bar {cls}"/>')
            out.append(nt(x - 6 if v < 0 else x + w + 6, yy + 11, f"{sg(v)}%", "lbl s", "end" if v < 0 else "start"))
    ly = 222
    out.append(f'<rect x="20" y="{ly - 10}" width="12" height="12" class="seg ttft"/>' + nt(38, ly, "TTFT", "lbl s"))
    out.append(f'<rect x="90" y="{ly - 10}" width="12" height="12" class="seg dec"/>' + svg_text(108, ly, "decode", "decode", "lbl s"))
    out.append(f'<rect x="560" y="{ly - 10}" width="12" height="12" class="bar bf16"/>' + svg_text(578, ly, "Test 1: BF16 KV + monokernel", "测试 1：BF16 KV + monokernel", "lbl s"))
    out.append(f'<rect x="800" y="{ly - 10}" width="12" height="12" class="bar mono"/>' + svg_text(818, ly, "Test 2: FP8 KV + monokernel", "测试 2：FP8 KV + monokernel", "lbl s"))
    out.append("</svg>")
    return "".join(out)


def plate_rootcause():
    W, x0 = 560 / 280, 210
    out = ['<svg viewBox="0 0 1040 290" role="img" aria-label="prefill kernel time">',
           svg_text(20, 26, "GPU kernel time of one prefill, rank 0 (ms)", "一次 prefill 的 GPU kernel 时间，rank 0（毫秒）", "lbl b")]
    for i, (en, zh_, p_, cls) in enumerate((("FP8 E4M3 KV", "FP8 E4M3 KV", pb, "base"), ("BF16 KV", "BF16 KV", pz, "bf16"))):
        y = 48 + i * 62
        other, tk = p_["kernel_total_ms"] - p_["tilelang_main_kernel_ms"], p_["tilelang_main_kernel_ms"]
        out.append(svg_text(20, y + 22, en, zh_, "lbl"))
        out.append(f'<rect x="{x0}" y="{y}" width="{other * W:.1f}" height="32" class="seg dec"/>')
        out.append(f'<rect x="{x0 + other * W:.1f}" y="{y}" width="{tk * W:.1f}" height="32" class="seg {cls}"/>')
        out.append(svg_text(x0 + 8, y + 21, f"other kernels {other:.1f}", f"其他 kernel {other:.1f}", "lbl"))
        out.append(nt(x0 + other * W + tk * W / 2, y + 21, f"TileLang {tk:.1f}", "lbl w", "middle"))
        tx = x0 + p_["ttft_median_ms"] * W
        out.append(f'<line x1="{tx:.1f}" y1="{y - 6}" x2="{tx:.1f}" y2="{y + 38}" class="tick"/>')
        out.append(nt(tx + 6, y + 21, f"TTFT {p_['ttft_median_ms']:.0f}", "lbl b"))
    out.append(svg_text(20, 196, "Bytes per KV row", "每行 KV 的字节数", "lbl b"))
    for i, (lab, n, cls) in enumerate((("FP8: (512 + 64) × 1 B", 576, "base"), ("BF16: (512 + 64) × 2 B", 1152, "bf16"))):
        y = 210 + i * 34
        out.append(nt(20, y + 16, lab, "lbl"))
        out.append(f'<rect x="{x0}" y="{y}" width="{n * 0.42:.1f}" height="22" class="seg {cls}"/>')
        out.append(nt(x0 + n * 0.42 + 8, y + 16, f"{n} B", "lbl b"))
    out.append("</svg>")
    return "".join(out)


def plate_datapath():
    box = lambda x, y, w, h, cls="box": f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" class="{cls}"/>'  # noqa: E731
    return "".join([
        f'<svg viewBox="0 0 1040 320" role="img" aria-label="FP8 KV data path">{ARROW}',
        box(380, 116, 280, 92, "box pool"),
        nt(520, 142, "SGLang MLA KV pool", "lbl b", "middle"),
        nt(520, 164, "uint8 [slots, 576], read as float8_e4m3fn", "lbl s", "middle"),
        nt(520, 184, "512 latent + 64 rope values, no block scales", "lbl s", "middle"),
        box(20, 26, 300, 74), nt(170, 54, "SGLang prefill and decode", "lbl b", "middle"),
        svg_text(170, 76, "read the rows", "读取这些行", "lbl s", "middle"),
        box(20, 226, 300, 74), nt(170, 254, "SGLang KV write", "lbl b", "middle"),
        svg_text(170, 276, "writes E4M3 rows", "写入 E4M3 行", "lbl s", "middle"),
        box(720, 26, 300, 74, "box mono"), nt(870, 54, "monokernel, FP8 build", "lbl b", "middle"),
        svg_text(870, 76, "reads E4M3, converts to BF16", "读取 E4M3，转成 BF16", "lbl s", "middle"),
        box(720, 226, 300, 74, "box mono"), nt(870, 254, "monokernel KV write", "lbl b", "middle"),
        svg_text(870, 276, "rounds to BF16, writes E4M3", "先舍入到 BF16，再写入 E4M3", "lbl s", "middle"),
        '<line x1="320" y1="63" x2="378" y2="126" class="edge" marker-end="url(#ah)"/>',
        '<line x1="320" y1="263" x2="378" y2="198" class="edge" marker-end="url(#ah)"/>',
        '<line x1="718" y1="63" x2="662" y2="126" class="edge mono" marker-end="url(#ah)"/>',
        '<line x1="718" y1="263" x2="662" y2="198" class="edge mono" marker-end="url(#ah)"/>',
        "</svg>"])


def plate_score():
    X0, X1, Y0, Y1 = 90, 1010, 380, 30
    xmin, xmax, ymin, ymax = 48, 124, 0, 10000
    sx = lambda v: X0 + (v - xmin) / (xmax - xmin) * (X1 - X0)  # noqa: E731
    sy = lambda v: Y0 - (v - ymin) / (ymax - ymin) * (Y0 - Y1)  # noqa: E731
    out = [f'<svg viewBox="0 0 1040 470" role="img" aria-label="score map">{ARROW}']
    for v in range(50, 124, 10):
        out.append(f'<line x1="{sx(v):.1f}" y1="{Y1}" x2="{sx(v):.1f}" y2="{Y0}" class="grid"/>' + nt(sx(v), Y0 + 18, str(v), "lbl s", "middle"))
    for v in range(0, 10001, 2000):
        out.append(f'<line x1="{X0}" y1="{sy(v):.1f}" x2="{X1}" y2="{sy(v):.1f}" class="grid"/>' + nt(X0 - 8, sy(v) + 4, f"{v:,}", "lbl s", "end"))
    out.append(svg_text((X0 + X1) / 2, Y0 + 40, "Interactivity (tok/s/user) →", "交互性（tok/s/user）→", "lbl", "middle"))
    out.append(svg_text(X0, Y1 - 10, "Throughput (tok/s/GPU) ↑", "吞吐（tok/s/GPU）↑", "lbl"))
    groups = {}
    for r in refs3:
        groups.setdefault(r["curve"], []).append(r)
    labels = [("GLM-5.2 MI355X SGLang FP8", 10, 10, 4, "start", "GLM-5.2 MI355X SGLang FP8"),
              ("GLM-5.2 MI355X SGLang FP4", 4, 10, -6, "start", "GLM-5.2 MI355X SGLang FP4"),
              ("GLM-5.2 MI355X ATOM FP4", 1, -10, 4, "end", "GLM-5.2 MI355X ATOM FP4"),
              ("GLM-5.3 MI355X TileRT", 1, 0, 20, "middle", "GLM-5.3 MI355X TileRT (16 GPUs)"),
              ("GLM-5.3 B300", 10, -10, 4, "end", "GLM-5.3 B300 SGLang FP4")]
    for name, rows in groups.items():
        if "this work" in name:
            continue
        rows = sorted(rows, key=lambda r: r["conc"])
        if len(rows) > 1:
            out.append('<polyline points="' + " ".join(f"{sx(r['int90']):.1f},{sy(r['tput']):.1f}" for r in rows) + '" class="refline"/>')
        for r in rows:
            out.append(f'<circle cx="{sx(r["int90"]):.1f}" cy="{sy(r["tput"]):.1f}" r="5" class="refpt"/>')
        for prefix, conc, dx, dy, anchor, text in labels:
            if name.startswith(prefix):
                r = next(r for r in rows if r["conc"] == conc)
                out.append(nt(sx(r["int90"]) + dx, sy(r["tput"]) + dy, text, "lbl s", anchor))
    for c in CONCS:
        bx, by, kx, ky = sx(B(c, "e2e_int_p90")), sy(B(c, "tput_gpu")), sx(K(c, "e2e_int_p90")), sy(K(c, "tput_gpu"))
        out.append(f'<line x1="{bx:.1f}" y1="{by:.1f}" x2="{kx:.1f}" y2="{ky:.1f}" class="move" marker-end="url(#ah)"/>')
        out.append(f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="6.5" class="pt base"/><circle cx="{kx:.1f}" cy="{ky:.1f}" r="6.5" class="pt mono"/>')
        out.append(nt(kx + 10, ky - 8, f"c{c}", "lbl b"))
    ly = 448
    out.append(f'<circle cx="100" cy="{ly}" r="6" class="pt base"/>' + svg_text(112, ly + 4, "base group", "base 组", "lbl s"))
    out.append(f'<circle cx="250" cy="{ly}" r="6" class="pt mono"/>' + svg_text(262, ly + 4, "monokernel group", "monokernel 组", "lbl s"))
    out.append(f'<circle cx="440" cy="{ly}" r="5" class="refpt"/>' + svg_text(452, ly + 4, "published InferenceX result (context only)", "InferenceX 已发布结果（只作背景）", "lbl s"))
    out.append("</svg>")
    return "".join(out)


def plate_gain():
    out = ['<svg viewBox="0 0 1040 250" role="img" aria-label="coverage and gain">',
           svg_text(20, 24, "Decode steps by running requests (monokernel group)", "按同时运行的请求数划分 decode 步（monokernel 组）", "lbl b"),
           svg_text(620, 24, "Change on the same requests", "同一批请求的变化", "lbl b")]
    W = 380
    for i, c in enumerate(CONCS):
        y = 44 + i * 40
        h = hist_share[c]
        one, two = h.get("1", 0), h.get("2", 0)
        out.append(nt(20, y + 19, f"c{c}", "lbl b"))
        x = 60
        for v, cls in ((one, "base"), (two, "mono"), (1 - one - two, "dec")):
            out.append(f'<rect x="{x:.1f}" y="{y}" width="{W * v:.1f}" height="26" class="seg {cls}"/>')
            x += W * v
        out.append(nt(60 + W + 14, y + 18, f"{100 * share[c]:.0f}%", "lbl s"))
    ly = 222
    for k, (cls, en, zh_) in enumerate((("base", "1 request", "1 个请求"), ("mono", "2 requests", "2 个请求"), ("dec", "3 or more: normal forward", "3 个以上：普通 forward"))):
        out.append(f'<rect x="{60 + k * 130}" y="{ly - 10}" width="12" height="12" class="seg {cls}"/>' + svg_text(78 + k * 130, ly, en, zh_, "lbl s"))
    zero, scale = 1010, 9.0
    out.append(f'<line x1="{zero}" y1="40" x2="{zero}" y2="200" class="axis"/>')
    for i, c in enumerate(CONCS):
        y = 44 + i * 40
        out.append(nt(620, y + 18, f"c{c}", "lbl b"))
        for j, (m, cls) in enumerate(((itl3[c], "mono"), (e2el3[c], "base"))):
            w = abs(m) * scale
            out.append(f'<rect x="{zero - w:.1f}" y="{y + j * 13}" width="{w:.1f}" height="11" class="bar {cls}"/>')
            out.append(nt(zero - w - 6, y + j * 13 + 10, f"{sg(m)}%", "lbl s", "end"))
    out.append(f'<rect x="620" y="{ly - 10}" width="12" height="12" class="bar mono"/>' + nt(638, ly, "ITL", "lbl s"))
    out.append(f'<rect x="690" y="{ly - 10}" width="12" height="12" class="bar base"/>' + nt(708, ly, "E2EL", "lbl s"))
    out.append("</svg>")
    return "".join(out)


def plate_upstream():
    return flow([
        ("done", "sgl-amd-mono#60", "sgl-amd-mono#60", "merged as 0a107f1", "已合并为 0a107f1", "done", "完成"),
        ("todo", "SGLang pin update", "SGLang 更新 pin", "main pins 05e0bb02", "main 仍是 05e0bb02", "not started", "未开始"),
        ("todo", "MI35x nightly image", "MI35x nightly 镜像", "includes the new pin", "包含新的 pin", "not started", "未开始"),
        ("open", "cookbook #46662", "cookbook #46662", "draft, waits for the pin", "draft，等 pin 更新", "open", "进行中"),
        ("todo", "InferenceX PR", "InferenceX PR", "sweep, evals, sign-off", "sweep、eval、签核", "not started", "未开始"),
    ])


# ----------------------------------------------------------------------------- retrospective data
abl = J("ablation/ablation.json")
tl = J("prs/timelines.json")
LADDER_STEPS = {"bf16": ("#15", "#24", "#42"), "fp8": ("F1", "F2", "F3")}


def ladder_eff(lad, step, c, m):
    """Mean over the runs of one step at one concurrency (F3 has two runs)."""
    vals = [e[m] for e in abl[lad]["effects"] if e["step"] == step and e["conc"] == c]
    return sum(vals) / len(vals) if vals else None


def ladder_runs(lad, step, c):
    return [e for e in abl[lad]["effects"] if e["step"] == step and e["conc"] == c]


def ladder_head(lad, step, c, m):
    """Headline gain (%) of one step at one concurrency; F3 averages its two runs."""
    vals = []
    for key, h in abl[lad]["headline"].items():
        if key.split()[0] == step and "points" in h:
            vals += [100 * p["gain"][m] for p in h["points"] if p["point"].endswith(f"c{c}")]
    return sum(vals) / len(vals) if vals else None


drift = next((e for e in abl["fp8"]["effects"] if e["step"] == "base rerun"), None)
fp8_ready = all(ladder_eff("fp8", s, c, "inter_token_latency") is not None for s in LADDER_STEPS["fp8"] for c in (1, 2, 4))
ISL_BINS = (("below 64k", "64k 以下"), ("64k to 128k", "64k 到 128k"), ("128k to 192k", "128k 到 192k"), ("192k or more", "192k 及以上"))
isl_v = lambda lad, s: [r["itl"] for r in abl[lad]["by_isl"][s]]  # noqa: E731
isl_n = lambda lad, s: [r["n"] for r in abl[lad]["by_isl"][s]]  # noqa: E731


def isl_share(lad, step, c):
    rows = ladder_runs(lad, step, c)[0]["by_isl"]
    return [100 * r["n"] / sum(q["n"] for q in rows) for r in rows]


def plate_arch():
    box = lambda x, y, w, h, cls="box": f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" class="{cls}"/>'  # noqa: E731
    out = [f'<svg viewBox="0 0 1040 330" role="img" aria-label="monokernel step">{ARROW}',
           box(10, 40, 210, 96), svg_text(115, 72, "SGLang", "SGLang", "lbl b", "middle"),
           svg_text(115, 94, "embedding", "embedding", "lbl s", "middle"),
           svg_text(115, 112, "dense layers 0 to 2", "dense 层 0 到 2", "lbl s", "middle"),
           box(270, 24, 500, 128, "box mono"),
           svg_text(520, 54, "75 monokernel launches: layers 3 to 77", "75 次 monokernel launch：第 3 到 77 层", "lbl b", "middle"),
           svg_text(520, 80, "one launch per MoE decoder layer", "每个 MoE decoder 层一次 launch", "lbl s", "middle"),
           svg_text(520, 102, "18 launches run the indexer", "18 次 launch 运行 indexer", "lbl s", "middle"),
           svg_text(520, 124, "57 launches reuse the top-k of the layer before", "57 次 launch 复用上一层的 top-k", "lbl s", "middle"),
           box(820, 40, 210, 96), svg_text(925, 72, "SGLang", "SGLang", "lbl b", "middle"),
           svg_text(925, 94, "final norm, logits", "final norm、logits", "lbl s", "middle"),
           svg_text(925, 112, "sampling, scheduling", "采样、调度", "lbl s", "middle"),
           '<line x1="220" y1="88" x2="268" y2="88" class="edge" marker-end="url(#ah)"/>',
           '<line x1="770" y1="88" x2="818" y2="88" class="edge" marker-end="url(#ah)"/>',
           svg_text(20, 196, "Stages of one launch, in order:", "一次 launch 的各个阶段，按顺序：", "lbl b")]
    stages = [(("q / kv projections", "RoPE"), ("q / kv 投影", "RoPE")), (("KV norm", "MLA cache write"), ("KV norm", "写 MLA cache")),
              (("indexer scores", "top-2048"), ("indexer 打分", "top-2048")), (("sparse MLA", "o_proj"), ("稀疏 MLA", "o_proj")),
              (("TP all-reduce", ""), ("TP all-reduce", "")), (("RMSNorm", "router (f32)"), ("RMSNorm", "router（f32）")),
              (("experts", "1 shared + 8 routed"), ("专家", "1 共享 + 8 路由")), (("TP all-reduce", ""), ("TP all-reduce", ""))]
    w, gap = 120, 8
    for i, (lines_en, lines_zh) in enumerate(stages):
        x = 10 + i * (w + gap)
        cls = "box st-open" if i == 2 else "box"
        out.append(box(x, 212, w, 70, cls))
        out.append(nt(x + 12, 236, str(i + 1), "lbl b"))
        for j, (we, wz) in enumerate(zip(lines_en, lines_zh)):
            if we or wz:
                out.append(svg_text(x + w / 2, 256 + j * 16, we, wz, "lbl s", "middle"))
    out.append(svg_text(20, 312, "Stage 3 runs only on the 18 indexer layers. It is the stage that grows with the context.",
                        "阶段 3 只在 18 个 indexer 层运行。它是随上下文变长而变慢的阶段。", "lbl s"))
    out.append("</svg>")
    return "".join(out)


def plate_timeline():
    days = {f"10-{d:02d}": d for d in range(16, 27)}
    X0, X1 = 70, 1010
    sx = lambda d: X0 + (d - 16) / 10 * (X1 - X0)  # noqa: E731
    out = ['<svg viewBox="0 0 1040 280" role="img" aria-label="PR timeline">',
           f'<line x1="{X0}" y1="150" x2="{X1}" y2="150" class="axis"/>']
    for lab, d in days.items():
        out.append(f'<line x1="{sx(d):.0f}" y1="146" x2="{sx(d):.0f}" y2="154" class="axis"/>' + nt(sx(d), 172, lab, "lbl s", "middle"))
    items = [(18, "#7", "decode monokernel", "decode monokernel", "mono", 1),
             (22, "#15", "MTP verify steps", "MTP verify 步", "mono", 2),
             (25, "#24", "parallel top-k", "并行 top-k", "mono", 1),
             (25.25, "#42", "per-wave scoring", "按 wave 打分", "mono", 2),
             (26, "#60", "FP8 KV cache", "FP8 KV cache", "mono", 3)]
    for d, num, en, zh_, cls, level in items:
        x, y = sx(d), 150 - 34 * level
        tx, anchor = (x - 8, "end") if x > 850 else (x + 8, "start")
        out.append(f'<line x1="{x:.0f}" y1="150" x2="{x:.0f}" y2="{y - 10}" class="edge {cls}"/>')
        out.append(f'<circle cx="{x:.0f}" cy="150" r="6" class="pt {cls}"/>')
        out.append(nt(tx, y, num, "lbl b", anchor))
        out.append(svg_text(tx, y + 15, en, zh_, "lbl s", anchor))
    side = [(18.5, "#11 #12", "common refactors", "公共层重构", 1, "start"), (22.1, "#25", "SGLang interface", "SGLang 接口", 1, "start"),
            (21.1, "#19", "open: guarded waits", "未合并：guarded wait", 2, "start"), (20.9, "#18", "closed: A4W4, TP4", "已关闭：A4W4、TP4", 3, "end")]
    for d, num, en, zh_, level, anchor in side:
        x, y = sx(d), 150 + 6 + 30 * level
        tx = x + 8 if anchor == "start" else x - 8
        out.append(f'<line x1="{x:.0f}" y1="150" x2="{x:.0f}" y2="{y - 12}" class="edge ref"/>')
        out.append(f'<circle cx="{x:.0f}" cy="150" r="4" class="refpt"/>')
        out.append(nt(tx, y, num, "lbl b", anchor))
        out.append(svg_text(tx, y + 15, en, zh_, "lbl s", anchor))
    out.append("</svg>")
    return "".join(out)


def plate_longctx():
    keys = ("8k", "32k", "64k", "128k", "256k")
    out = ['<svg viewBox="0 0 1040 330" role="img" aria-label="estimated step time">']
    builds = (("#15", "ref", "pr24", "step_before_ms"), ("#24", "base", "pr24", "step_after_ms"), ("#42", "mono", "pr42", "step_after_ms"))
    for panel, (rows, x0) in enumerate(((1, 70), (4, 590))):
        W, Y0, Y1 = 400, 270, 50
        vals = [r[f] for _, _, pr_, f in builds for r in tl[pr_]["rows"] if r["rows"] == rows and r["keys"] in keys]
        ymax = math.ceil(max(vals) / 2) * 2
        sx = lambda i: x0 + i * W / (len(keys) - 1)  # noqa: E731
        sy = lambda v: Y0 - v / ymax * (Y0 - Y1)  # noqa: E731
        out.append(svg_text(x0, 28, f"{rows} row{'s' if rows > 1 else ''} in the step: estimated step (ms)", f"一步 {rows} 行：预估单步时间（毫秒）", "lbl b"))
        for v in range(0, ymax + 1, 2):
            out.append(f'<line x1="{x0}" y1="{sy(v):.1f}" x2="{x0 + W}" y2="{sy(v):.1f}" class="grid"/>' + nt(x0 - 8, sy(v) + 4, str(v), "lbl s", "end"))
        for i, k in enumerate(keys):
            out.append(nt(sx(i), Y0 + 18, k, "lbl s", "middle"))
        out.append(svg_text(x0 + W / 2, Y0 + 40, "keys per row", "每行的 key 数", "lbl s", "middle"))
        for name, cls, pr_, f in builds:
            pts_ = [(sx(i), sy(next(r[f] for r in tl[pr_]["rows"] if r["rows"] == rows and r["keys"] == k))) for i, k in enumerate(keys)]
            out.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts_) + f'" class="line {cls}"/>')
            for x, y in pts_:
                out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" class="pt {cls}"/>')
            out.append(nt(pts_[-1][0] + 8, pts_[-1][1] + 4, name, "lbl b"))
    out.append("</svg>")
    return "".join(out)


def plate_ladder(lad):
    steps = LADDER_STEPS[lad]
    cls = {"#15": "ref", "#24": "base", "#42": "mono", "F1": "ref", "F2": "base", "F3": "mono"}
    lab = {"#15": ("#15", "#15"), "#24": ("+ #24", "+ #24"), "#42": ("+ #42", "+ #42"),
           "F1": ("F1: #7 #15", "F1：#7 #15"), "F2": ("F2: + #24", "F2：+ #24"), "F3": ("F3: + #42", "F3：+ #42")}
    zero, scale = 560, 12.0
    out = [f'<svg viewBox="0 0 1040 300" role="img" aria-label="ladder">',
           svg_text(20, 24, "ITL on the same requests, against the base (cumulative)", "同一批请求的 ITL，相对 base（累积）", "lbl b"),
           f'<line x1="{zero}" y1="36" x2="{zero}" y2="252" class="axis"/>']
    for i, c in enumerate((1, 2, 4)):
        y = 44 + i * 70
        out.append(nt(20, y + 30, f"c{c}", "lbl b"))
        for j, s in enumerate(steps):
            v = ladder_eff(lad, s, c, "inter_token_latency")
            yy = y + j * 19
            if v is None:
                out.append(svg_text(zero + 8, yy + 13, f"{lab[s][0]}: pending", f"{lab[s][1]}：待测", "lbl s"))
                continue
            w = abs(v) * scale
            x = zero - w if v < 0 else zero
            out.append(f'<rect x="{x:.1f}" y="{yy}" width="{w:.1f}" height="15" class="bar {cls[s]}"/>')
            out.append(nt(x - 6 if v < 0 else x + w + 6, yy + 12, f"{sg(v)}%", "lbl s", "end" if v < 0 else "start"))
            out.append(svg_text(zero + 8 if v < 0 else zero - 8, yy + 12, lab[s][0], lab[s][1], "lbl s", "start" if v < 0 else "end"))
    out.append(nt(zero, 270, "0", "lbl s", "middle"))
    out.append(svg_text(zero - 170, 290, "← faster", "← 更快", "lbl s", "middle"))
    out.append("</svg>")
    return "".join(out)


def plate_isl():
    names = {"#15": "#15", "#24": "+ #24", "#42": "+ #42", "F1": "F1", "F2": "F2", "F3": "F3"}
    ticks = ("< 64k", "64k–128k", "128k–192k", "≥ 192k")
    YMAX, YMIN, Y1, Y0, W = 20, -35, 50, 280, 380
    sy = lambda v: Y1 + (YMAX - v) / (YMAX - YMIN) * (Y0 - Y1)  # noqa: E731
    out = ['<svg viewBox="0 0 1040 330" role="img" aria-label="ITL change by input length">']
    for lad, x0, en, zh_ in (("bf16", 70, "BF16 ladder, 3600 s: ITL change (%)", "BF16 阶梯，3600 秒：ITL 变化（%）"),
                             ("fp8", 590, "FP8 ladder, 900 s: ITL change (%)", "FP8 阶梯，900 秒：ITL 变化（%）")):
        sx = lambda i: x0 + 30 + i * (W - 60) / 3  # noqa: E731
        out.append(svg_text(x0, 28, en, zh_, "lbl b"))
        for v in range(-30, YMAX + 1, 10):
            out.append(f'<line x1="{x0}" y1="{sy(v):.1f}" x2="{x0 + W}" y2="{sy(v):.1f}" class="{"axis" if v == 0 else "grid"}"/>'
                       + nt(x0 - 8, sy(v) + 4, sg(v, 0) if v else "0", "lbl s", "end"))
        for i, t in enumerate(ticks):
            out.append(nt(sx(i), Y0 + 20, t, "lbl s", "middle"))
        out.append(svg_text(x0 + W / 2, Y0 + 42, "input length of the request", "请求的输入长度", "lbl s", "middle"))
        for s, cls in zip(LADDER_STEPS[lad], ("ref", "base", "mono")):
            runs_per_conc = len(ladder_runs(lad, s, 1))
            pts_ = [(sx(i), sy(v), n // runs_per_conc) for i, (v, n) in enumerate(zip(isl_v(lad, s), isl_n(lad, s))) if v is not None]
            out.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in pts_) + f'" class="line {cls}"/>')
            for x, y, n in pts_:
                out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" class="pt {cls}{" hollow" if n < 30 else ""}"/>')
            out.append(nt(pts_[-1][0] + 12, pts_[-1][1] + 4, names[s], "lbl b"))
    out.append("</svg>")
    return "".join(out)


# ----------------------------------------------------------------------------- sections
T24 = {(r["keys"], r["rows"]): r for r in tl["pr24"]["rows"]}
T42 = {(r["keys"], r["rows"]): r for r in tl["pr42"]["rows"]}
b0c = {c: eff[C1][(c, "b0-r1")]["time_to_first_token"] for c in (1, 2)}


def prng(vals, zh_text=False):
    lo, hi = round(min(vals)), round(max(vals))
    return f"{lo}% {'到' if zh_text else 'to'} {hi}%" if lo != hi else f"{lo}%"


def trio(lad, step, m="inter_token_latency"):
    vals = [ladder_eff(lad, step, c, m) for c in (1, 2, 4)]
    return " / ".join("…" if v is None else f"{sg(v)}%" for v in vals)


def trio_head(lad, step, m):
    vals = [ladder_head(lad, step, c, m) for c in (1, 2, 4)]
    return " / ".join("…" if v is None else f"{sg(v)}%" for v in vals)


cell = lambda c, m, nd, unit="": f"{B(c, m):.{nd}f} → <b>{K(c, m):.{nd}f}</b>{unit}<br><span class=\"{'up' if g(c, m) > 0 else 'flat'}\">{sg(g(c, m))}%</span>"  # noqa: E731
score_rows = [[f"c{c}", cell(c, "tput_gpu", 0), cell(c, "e2e_int_p90", 1), cell(c, "e2el_p90", 1, " s"), cell(c, "ttft_p90", 2, " s")] for c in CONCS]
score_head = [("Concurrency", "并发"), ("Throughput (tok/s/GPU)", "吞吐（tok/s/GPU）"), ("Interactivity", "交互性"), ("P90 E2EL", "P90 E2EL"), ("P90 TTFT", "P90 TTFT")]
pend = ("pending", "待测")

contrib_rows = [
    [pr(7), sp("decode monokernel: one launch per MoE layer", "decode monokernel：每个 MoE 层一次 launch"), "-", "-",
     sp("Base of all later PRs. It cannot serve the MTP recipe alone.", "后续所有 PR 的基础。它本身不能用于 MTP 配方。")],
    [pr(15), sp("MTP verify steps on the monokernel", "monokernel 运行 MTP 的 verify 步"), trio("bf16", "#15"), trio("fp8", "F1"),
     sp("Needed for the recipe. Faster below 192k tokens, slower above (section 09).", "配方需要它。192k token 以下更快，以上更慢（第 09 节）。")],
    [pr(24), sp("top-k of a long row on up to 16 CTAs", "长行的 top-k 分给最多 16 个 CTA"), trio("bf16", "#24"), trio("fp8", "F2"),
     sp("Removes most of the loss above 192k.", "去掉了 192k 以上的大部分损失。")],
    [pr(42), sp("score 16 keys per wave, reads in flight", "每个 wave 给 16 个 key 打分，读取提前发出"), trio("bf16", "#42"), trio("fp8", "F3"),
     sp("Largest gain, almost equal at all context lengths.", "收益最大，各个上下文长度下几乎相同。")],
    [pr(60), sp("use the FP8 E4M3 KV cache of the recipe", "使用配方的 FP8 E4M3 KV cache"), "-", "-",
     sp("Removes the TTFT cost of BF16 KV. The ITL gain becomes a score gain.", "去掉 BF16 KV 的 TTFT 代价，ITL 收益才变成分数收益。")],
]

s_result = section("result", "01", ("Result", "结果"), "".join([
    P("This record reviews the glm53 monokernel work for GLM-5.3 on MI355X. Five PRs built the monokernel. For each PR, we measured what it added on InferenceX AgentX.",
      "这份记录复盘 GLM-5.3 在 MI355X 上的 glm53 monokernel 工作。monokernel 由五个 PR 构成。对每个 PR，我们都测量了它在 InferenceX AgentX 上带来的收益。"),
    P("The table gives the ITL change on the same requests against the base, at concurrency 1, 2, and 4. Each row includes the PRs above it. The BF16 ladder is the PR campaign (3600 s, BF16 KV in both groups). The FP8 ladder uses the real recipe (900 s, FP8 KV).",
      "下表是同一批请求相对 base 的 ITL 变化，并发为 1、2、4。每一行都包含它上面的所有 PR。BF16 阶梯是 PR 开发时的测试（3600 秒，两组都用 BF16 KV）。FP8 阶梯使用正式配方（900 秒，FP8 KV）。"),
    table([("PR", "PR"), ("Change", "改动"), ("ITL, BF16 ladder", "ITL，BF16 阶梯"), ("ITL, FP8 ladder", "ITL，FP8 阶梯"), ("Verdict", "结论")], contrib_rows, "contrib"),
    P(f"We measured the final result in the 3600-s window of InferenceX. With all five PRs, the recipe beats the recipe without the monokernel on all three headline metrics. Each gain is larger than the A/A noise at concurrency 2: {100 * noise3['tput_gpu']:.1f}% for throughput, {100 * noise3['e2e_int_p90']:.1f}% for interactivity, {100 * noise3['e2el_p90']:.1f}% for P90 E2EL.",
      f"最终结果是在 InferenceX 的 3600 秒窗口下测得的。加上全部五个 PR 后，三个头条指标都好于不开 monokernel 的配方。每个增益都大于 c2 的 A/A 噪声：吞吐 {100 * noise3['tput_gpu']:.1f}%，交互性 {100 * noise3['e2e_int_p90']:.1f}%，P90 E2EL {100 * noise3['e2el_p90']:.1f}%。"),
    table(score_head, score_rows, "score"),
    P("Each cell shows base → monokernel and the gain. For a latency, the gain is base / monokernel − 1. A positive gain is always better.",
      "每格显示 base → monokernel 和增益。延迟的增益等于 base / monokernel − 1。增益为正，总是表示更好。", "small"),
]))

s_mono = section("monokernel", "02", ("What the monokernel does", "monokernel 做了什么"), "".join([
    P("On the normal path, SGLang runs a GLM-5.3 decode step layer by layer. For one request at TP8, a step launches 1,821 GPU kernels and takes 11.70 ms of GPU time.",
      "在 normal path 上，SGLang 逐层运行 GLM-5.3 的一个 decode 步。TP8、单请求时，一步要发起 1,821 个 GPU kernel，GPU 时间是 11.70 ms。"),
    P(f"The monokernel replaces each of the 75 MoE decoder layers (layers 3 to 77) with one persistent launch. Then the same step has 212 kernels and takes 5.59 ms ({pr(7)}, no MTP).",
      f"monokernel 把 75 个 MoE decoder 层（第 3 到 77 层）各换成一次常驻 launch。这样同一步只有 212 个 kernel，用时 5.59 ms（{pr(7)}，不开 MTP）。"),
    plate("I", ("One decode or verify step", "一个 decode 或 verify 步"), plate_arch(),
          ("SGLang keeps the embedding, the dense layers, the logits, sampling, and scheduling. The monokernel runs the MoE layers.",
           "SGLang 保留 embedding、dense 层、logits、采样和调度。monokernel 运行 MoE 层。")),
    UL([
        ("Coverage: at TP8 the monokernel runs steps of 8 rows or fewer. A step with fewer rows runs at the next width (2, 4, or 8) with pad rows. Larger steps use the normal path.",
         "覆盖范围：TP8 时，monokernel 运行 8 行以内的步。行数不够时，按下一档宽度（2、4 或 8）运行，多出的是填充行。更大的步走 normal path。"),
        ("With MTP (3 draft steps), each request has 4 verify rows. Thus the monokernel serves verify steps of one or two requests.",
         "开 MTP（3 个 draft 步）时，每个请求有 4 个 verify 行。所以 monokernel 负责 1 到 2 个请求的 verify 步。"),
        ("GLM-5.3 uses DeepSeek Sparse Attention. On an indexer layer, each row scores all keys of its context and keeps the top 2048. The other layers reuse that selection.",
         "GLM-5.3 使用 DeepSeek 稀疏注意力。在 indexer 层，每一行给上下文中的所有 key 打分，保留前 2048 个。其他层复用这个选择结果。"),
        ("The indexer stage grows with the context. AgentX contexts are long: the median input of a request is 89k to 285k tokens.",
         "indexer 阶段随上下文变长而变慢。AgentX 的上下文很长：请求输入的中位数是 89k 到 285k token。"),
    ]),
]))

s_metric = section("metric", "03", ("How InferenceX scores AgentX", "InferenceX 怎样给 AgentX 打分"), "".join([
    P("AgentX replays recorded multi-turn agent sessions. Each turn adds text to the prompt and reuses the cached prefix. Concurrency is the number of live sessions.",
      "AgentX 回放录制的多轮 agent 会话。每一轮都在 prompt 后面追加内容，并复用已缓存的前缀。并发数是同时在跑的会话数。"),
    UL([("Throughput: total tokens per second per GPU.", "吞吐：每张 GPU 每秒处理的总 token 数。"),
        ("Interactivity: 1 / P90 of (E2EL / OSL), in tokens per second per user. This page uses \"interactivity\" for this metric.",
         "交互性：1 / (E2EL / OSL) 的 P90，单位是每用户每秒 token。本页用“交互性”指这个指标。"),
        ("P90 E2EL: the 90th percentile of the end-to-end latency.", "P90 E2EL：端到端延迟的 90 分位。")]),
    P("Interactivity is sensitive to TTFT. E2EL = TTFT + (OSL − 1) × ITL. For a short reply, most of E2EL is TTFT, and the P90 of E2EL / OSL comes from the short replies. Thus a faster decode helps interactivity only a little, and a longer TTFT makes it worse.",
      "交互性对 TTFT 很敏感。E2EL = TTFT + (OSL − 1) × ITL。短回复的 E2EL 主要是 TTFT，而 E2EL / OSL 的 P90 由短回复决定。所以 decode 变快，交互性只改善一点；TTFT 变长，交互性就变差。"),
    plate("II", ("TTFT share and change per output token", "TTFT 的占比和每个输出 token 的变化"), plate_anatomy(),
          ("Test 1 and Test 2 at concurrency 1, mean of two runs. With BF16 KV, the slowest 10% did not get faster.",
           "测试 1 和测试 2，c1，两次运行的平均。使用 BF16 KV 时，最慢的 10% 没有变快。")),
]))

pr_rows = [
    [pr(7), "10-18", sp("adds the decode monokernel", "加入 decode monokernel"),
     sp("TPOT 11.24 → 5.60 ms, TP8, 1 request, 8192 in / 256 out, no MTP", "TPOT 11.24 → 5.60 ms，TP8，单请求，8192 输入 / 256 输出，不开 MTP")],
    [pr(15), "10-22", sp("serves MTP verify steps; fixes 3 defects", "运行 MTP verify 步；修复 3 个缺陷"),
     sp("verify step 13.56 → 8.35 ms, 1 request, 4096-token prompts", "verify 步 13.56 → 8.35 ms，单请求，4096 token 的 prompt")],
    [pr(24), "10-25", sp("selects a long row's top-k on up to 16 CTAs", "长行的 top-k 分给最多 16 个 CTA 选"),
     sp(f"selection at 256k keys, 1 row: {T24[('256k', 1)]['selection_us_before']:.0f} → {T24[('256k', 1)]['selection_us_after']:.0f} µs",
        f"256k key、1 行时，选择阶段 {T24[('256k', 1)]['selection_us_before']:.0f} → {T24[('256k', 1)]['selection_us_after']:.0f} µs")],
    [pr(42), "10-25", sp("scores 16 keys per wave over 32 heads", "每个 wave 为 32 个 head 给 16 个 key 打分"),
     sp(f"score stage at 256k keys, 1 row: {T42[('256k', 1)]['score_us_before']:.0f} → {T42[('256k', 1)]['score_us_after']:.0f} µs",
        f"256k key、1 行时，打分阶段 {T42[('256k', 1)]['score_us_before']:.0f} → {T42[('256k', 1)]['score_us_after']:.0f} µs")],
    [pr(60), "10-26", sp("reads and writes SGLang's FP8 E4M3 MLA cache", "读写 SGLang 的 FP8 E4M3 MLA cache"),
     sp("TTFT stays at the recipe's value (section 07)", "TTFT 保持在配方的水平（第 07 节）")],
    [f"{pr(11)} {pr(12)} {pr(25)}", "10-18 – 10-22", sp("common code and the SGLang interface", "公共代码和 SGLang 接口"), sp("no GLM-5.3 speed change measured", "没有测到 GLM-5.3 的速度变化")],
    [pr(55), "10-25", sp("test settings only", "只改测试设置"), "-"],
]
s_prs = section("prs", "04", ("The PR series", "PR 系列"), "".join([
    P("Five PRs changed the GLM-5.3 path. Each PR measured its own change in its description. Those tests used short prompts or kernel timelines, not the AgentX workload.",
      "五个 PR 改动了 GLM-5.3 的路径。每个 PR 都在正文里测量了自己的改动。那些测试用的是短 prompt 或 kernel timeline，不是 AgentX 负载。"),
    plate("III", ("Merged PRs and other work", "已合并的 PR 和其他工作"), plate_timeline(),
          ("Raspberry: GLM-5.3 PRs. Gray: common code, an open PR (#19), and a closed PR (#18).",
           "覆盆子红：GLM-5.3 的 PR。灰色：公共代码、未合并的 PR（#19）和已关闭的 PR（#18）。")),
    table([("PR", "PR"), ("Merged", "合并"), ("Change", "改动"), ("Measured in the PR", "PR 自己的测量")], pr_rows),
]))

s_long = section("longctx", "05", ("Why long contexts were slow", "长上下文为什么慢"), "".join([
    P(f"The PRs measured each stage with a debug build. At 256k keys and one row, the exact top-2048 selection took {T24[('256k', 1)]['selection_us_before']:.0f} µs on one CTA. The other CTAs of the launch waited for it. A launch without the indexer took about 54 µs at all contexts.",
      f"这些 PR 用 debug 版本测量了每个阶段。256k key、1 行时，精确的 top-2048 选择在一个 CTA 上要 {T24[('256k', 1)]['selection_us_before']:.0f} µs。这次 launch 的其他 CTA 都在等它。不带 indexer 的 launch 在任何上下文下都约 54 µs。"),
    UL([
        ("#24 splits the selection of a row longer than 32768 keys over up to 16 CTAs. Each CTA builds a histogram of its chunks. One CTA finds the threshold. Each CTA then gathers its candidates. The result is still the exact top-2048.",
         "#24 把超过 32768 个 key 的行分给最多 16 个 CTA 来选。每个 CTA 给自己的分块建直方图。一个 CTA 找出阈值。然后每个 CTA 收集自己的候选。结果仍然是精确的 top-2048。"),
        ("#42 changes the score stage. Before, two waves read each key and added their results through LDS with two barriers. Each tile also waited for two memory round trips. Now one wave scores a group of 16 keys over all 32 heads, and the reads of the next groups are in flight.",
         "#42 改了打分阶段。原来每个 key 被两个 wave 读取，两者再经过 LDS 和两次 barrier 相加。每个 tile 还要等两次内存往返。现在一个 wave 为全部 32 个 head 给一组 16 个 key 打分，后面几组的读取提前发出。"),
    ]),
    plate("IV", ("Estimated step time by context", "不同上下文下的预估单步时间"), plate_longctx(),
          ("From the kernel timelines of #24 and #42 (debug builds, decode steps). At 8k keys the builds are equal.",
           "数据来自 #24 和 #42 的 kernel timeline（debug 版本，decode 步）。8k key 时，三个版本一样快。")),
    P(f"At 256k keys and one row, the estimated step falls from {T24[('256k', 1)]['step_before_ms']:.2f} ms (#15) to {T24[('256k', 1)]['step_after_ms']:.2f} ms (#24) and {T42[('256k', 1)]['step_after_ms']:.2f} ms (#42). At four rows, it falls from {T24[('256k', 4)]['step_before_ms']:.2f} ms to {T24[('256k', 4)]['step_after_ms']:.2f} ms and {T42[('256k', 4)]['step_after_ms']:.2f} ms.",
      f"256k key、1 行时，预估单步时间从 {T24[('256k', 1)]['step_before_ms']:.2f} ms（#15）降到 {T24[('256k', 1)]['step_after_ms']:.2f} ms（#24），再降到 {T42[('256k', 1)]['step_after_ms']:.2f} ms（#42）。4 行时，从 {T24[('256k', 4)]['step_before_ms']:.2f} ms 降到 {T24[('256k', 4)]['step_after_ms']:.2f} ms，再降到 {T42[('256k', 4)]['step_after_ms']:.2f} ms。"),
]))

bf16_rows = []
for s in LADDER_STEPS["bf16"]:
    for c in (1, 2, 4):
        bf16_rows.append([s, f"c{c}", f"{sg(ladder_eff('bf16', s, c, 'time_to_first_token'))}%", f"{sg(ladder_eff('bf16', s, c, 'inter_token_latency'))}%",
                          f"{sg(ladder_eff('bf16', s, c, 'request_latency'))}%", f"{sg(ladder_head('bf16', s, c, 'tput_gpu'))}%", f"{sg(ladder_head('bf16', s, c, 'e2e_int_p90'))}%"])
s_bf16 = section("ablation-bf16", "06", ("Ablation 1: the PR ladder with BF16 KV", "Ablation 1：BF16 KV 下的 PR 阶梯"), "".join([
    P("The PR campaign ran each PR on AgentX for 3600 s. Both groups used a BF16 KV cache, because the monokernel accepted only BF16 KV before #60. Setup: image 20261019, SGLang e45cbce599, simulated accept length 2.99.",
      "PR 开发期间，每个 PR 都在 AgentX 上跑了 3600 秒。两组都用 BF16 KV cache，因为 #60 之前 monokernel 只接受 BF16 KV。环境：镜像 20261019，SGLang e45cbce599，模拟接受长度 2.99。"),
    plate("V", ("ITL with BF16 KV, cumulative", "BF16 KV 下的 ITL，累积"), plate_ladder("bf16"),
          ("Each bar includes the PRs above it. The base is the normal path with MTP and BF16 KV.", "每根柱子都包含它上面的 PR。base 是开 MTP、BF16 KV 的 normal path。")),
    table([("Up to", "累积到"), ("Concurrency", "并发"), ("TTFT", "TTFT"), ("ITL", "ITL"), ("E2EL", "E2EL"), ("Throughput", "吞吐"), ("Interactivity", "交互性")], bf16_rows),
    UL([
        (f"#15 alone does not reduce ITL at concurrency 1 ({sg(ladder_eff('bf16', '#15', 1, 'inter_token_latency'))}%). At concurrency 4, ITL is worse ({sg(ladder_eff('bf16', '#15', 4, 'inter_token_latency'))}%). Throughput is 1% to 3% lower.",
         f"只有 #15 时，c1 的 ITL 没有降低（{sg(ladder_eff('bf16', '#15', 1, 'inter_token_latency'))}%）。c4 的 ITL 反而变差（{sg(ladder_eff('bf16', '#15', 4, 'inter_token_latency'))}%）。吞吐低 1% 到 3%。"),
        (f"With #24, ITL decreases by {prng([abs(ladder_eff('bf16', '#24', c, 'inter_token_latency')) for c in (1, 2, 4)])}.",
         f"加上 #24 后，ITL 降低 {prng([abs(ladder_eff('bf16', '#24', c, 'inter_token_latency')) for c in (1, 2, 4)], True)}。"),
        (f"With #42, ITL decreases by {prng([abs(ladder_eff('bf16', '#42', c, 'inter_token_latency')) for c in (1, 2, 4)])}, about twice the effect of #24 alone.",
         f"再加上 #42 后，ITL 降低 {prng([abs(ladder_eff('bf16', '#42', c, 'inter_token_latency')) for c in (1, 2, 4)], True)}，大约是只有 #24 时的两倍。"),
        ("TTFT does not change in this ladder, because both groups use BF16 KV.", "这个阶梯里 TTFT 不变，因为两组都用 BF16 KV。"),
    ]),
    callout("The short-prompt test of #15 showed a 38.5% faster verify step. On AgentX at concurrency 1, #15 alone gave no net gain. A kernel test must use the context lengths of the target workload.",
            "#15 在短 prompt 测试中，verify 步快了 38.5%。可是在 AgentX 的 c1 上，只有 #15 时没有净收益。kernel 测试必须使用目标负载的上下文长度。"),
]))

e2e_rows = []
for rid, en, zh_ in (("e2e-glm53-fp8kv-verify-check", "F3 (main): FP8 verify check", "F3（main）：FP8 verify check"),
                     ("e2e-glm53-abl-f1-fp8-verify", "F1: FP8 verify check", "F1：FP8 verify check"),
                     ("e2e-glm53-abl-f2-fp8-verify", "F2: FP8 verify check", "F2：FP8 verify check"),
                     ("e2e-glm53-bf16-verify-check", "F3: BF16 verify check (regression)", "F3：BF16 verify check（回归）"),
                     ("e2e-glm53-fp8kv-neg-verify-check", "Negative control: latent bytes swapped", "负对照：交换 latent 的字节")):
    v = e2e[rid]
    kv, stv = e2ec(rid), e2ec(rid, "layer.streams.cosine")
    state = sp("fails, as expected", "按预期失败") if v["tests_failed"] else sp("passes", "通过")
    e2e_rows.append([sp(en, zh_), state, str(kv["checks"]), f"{kv['min']:.5f}", "-" if not stv else f"{stv['min']:.5f}", str(v["violation_lines"])])
s_gap = section("baseline", "07", ("The baseline gap and #60", "对照组的差距与 #60"), "".join([
    P("The InferenceX recipe uses an FP8 E4M3 KV cache. The PR campaign compared against a BF16 baseline. Thus its gains did not show what the monokernel adds to the real recipe.",
      "InferenceX 的配方使用 FP8 E4M3 KV cache。PR 开发期间的对照组用的是 BF16。所以那些收益不能说明 monokernel 给正式配方带来了多少。"),
    plate("VI", ("Tests on the real recipe", "在正式配方上的测试"), plate_steps(),
          ("Each box shows the change and the result. Green: passed or merged. Amber: inside noise. Red: no-go.",
           "每个方框写明改动和结果。绿色：通过或已合并。琥珀色：在噪声内。红色：no-go。")),
    P(f"Test 1 compared the monokernel with all PRs up to #42 (BF16 KV) against the recipe (FP8 KV), at 900 s. On the same requests, ITL changed by {rng(pv[C1]['inter_token_latency'])}, but TTFT changed by {rng(pv[C1]['time_to_first_token'])}. Interactivity did not improve. The decision rule gave no-go.",
      f"测试 1 用包含 #42 及以前所有 PR 的 monokernel（BF16 KV），和配方（FP8 KV）比较，窗口 900 秒。在同一批请求上，ITL 变化 {zrng(pv[C1]['inter_token_latency'])}，但 TTFT 变化 {zrng(pv[C1]['time_to_first_token'])}。交互性没有改善，判定规则给出 no-go。"),
    P(f"BF16 KV alone causes the TTFT cost. Without the monokernel, BF16 KV changed TTFT by {sg(b0c[1])}% at concurrency 1 and by {sg(b0c[2])}% at concurrency 2.",
      f"TTFT 的代价来自 BF16 KV 本身。不开 monokernel 时，BF16 KV 让 c1 的 TTFT 变化 {sg(b0c[1])}%，c2 变化 {sg(b0c[2])}%。"),
    P(f"A prefill profile found the cause. With BF16 KV, one TileLang kernel of the DSA prefill takes ×{k_ratio:.2f} its FP8 time, and TTFT increases by {ttft_up}%. By its grid, this kernel is the sparse attention, which reads 1152 bytes per KV row in BF16 and 576 in FP8. We did not confirm the kernel identity in the source.",
      f"一次 prefill profile 找到了原因。使用 BF16 KV 时，DSA prefill 里的一个 TileLang kernel 耗时是 FP8 下的 ×{k_ratio:.2f}，TTFT 增加 {ttft_up}%。按 grid 判断，这个 kernel 是稀疏 attention：BF16 下每行 KV 读 1152 字节，FP8 下读 576 字节。我们没有在源码中确认这个 kernel 的身份。"),
    plate("VII", ("Prefill time with FP8 and BF16 KV", "FP8 和 BF16 KV 下的 prefill 时间"), plate_rootcause(),
          ("With BF16 KV, the TileLang kernel takes about 2× longer. The other kernels do not change.",
           "使用 BF16 KV 时，TileLang kernel 的时间约为 2 倍。其他 kernel 不变。")),
    P(f"{pr(60)} let the monokernel use the FP8 cache of the recipe. The monokernel already had an FP8 path in the same format as SGLang: 576-byte E4M3 rows without block scales. #60 changed only the integration: the spec, the build, and the binding of the KV rows.",
      f"{pr(60)} 让 monokernel 使用配方的 FP8 cache。monokernel 本来就有 FP8 路径，格式和 SGLang 相同：每行 576 字节的 E4M3，没有分块 scale。#60 只改了集成部分：spec、构建，以及 KV 行的绑定。"),
    plate("VIII", ("One KV cache, two users", "同一个 KV cache，两个使用方"), plate_datapath(),
          ("SGLang and the monokernel read and write the same rows in the same format.", "SGLang 和 monokernel 用同样的格式读写同样的行。")),
    P("The e2e checks run SGLang and the monokernel on the same steps and compare each layer by cosine similarity. The negative control shows that the checks can fail. F1 and F2 are the ablation builds of section 08.",
      "e2e check 让 SGLang 和 monokernel 跑同样的步，逐层比较 cosine 相似度。负对照证明这些检查能够失败。F1 和 F2 是第 08 节的 ablation 版本。"),
    table([("Check", "检查"), ("Result", "结果"), ("Checks", "次数"), ("Min KV cosine", "KV cosine 最小值"), ("Min stream cosine", "stream cosine 最小值"), ("Violations", "违例")], e2e_rows),
    P(f"Test 2 repeated the 900-s tests with #60. ITL changed by {rng(pv[C2]['inter_token_latency'])}, and TTFT by {rng(pv[C2]['time_to_first_token'])}. But the A/A noise of interactivity in 900 s was {100 * noise2['e2e_int_p90']:.0f}%. Test 3 used the 3600-s window, where the A/A noise was {100 * noise3['e2e_int_p90']:.1f}% (section 10).",
      f"测试 2 用 #60 重做 900 秒测试。ITL 变化 {zrng(pv[C2]['inter_token_latency'])}，TTFT 变化 {zrng(pv[C2]['time_to_first_token'])}。但 900 秒内交互性的 A/A 噪声有 {100 * noise2['e2e_int_p90']:.0f}%。测试 3 改用 3600 秒窗口，A/A 噪声降到 {100 * noise3['e2e_int_p90']:.1f}%（第 10 节）。"),
]))

fp8_rows = []
for s in LADDER_STEPS["fp8"]:
    for c in (1, 2, 4):
        runs_ = ladder_runs("fp8", s, c)
        if not runs_:
            fp8_rows.append([s, f"c{c}", *[sp(*pend)] * 4])
            continue
        fp8_rows.append([s, f"c{c}", f"{sg(ladder_eff('fp8', s, c, 'time_to_first_token'))}%", f"{sg(ladder_eff('fp8', s, c, 'inter_token_latency'))}%",
                         f"{sg(ladder_eff('fp8', s, c, 'request_latency'))}%", f"{100 * sum(e['coverage'] for e in runs_) / len(runs_):.0f}%"])
fp8_find = []
if fp8_ready:
    f1 = [ladder_eff("fp8", "F1", c, "inter_token_latency") for c in (1, 2, 4)]
    f2 = [ladder_eff("fp8", "F2", c, "inter_token_latency") for c in (1, 2, 4)]
    f3 = [ladder_eff("fp8", "F3", c, "inter_token_latency") for c in (1, 2, 4)]
    ttft_fp8 = math.ceil(max(abs(e["time_to_first_token"]) for e in abl["fp8"]["effects"] if e["step"] in LADDER_STEPS["fp8"]))
    fp8_find = [
        (f"F1 (#7, #15) changes ITL by {' / '.join(f'{sg(v)}%' for v in f1)} at concurrency 1, 2, and 4.",
         f"F1（#7、#15）让 c1、c2、c4 的 ITL 分别变化 {' / '.join(f'{sg(v)}%' for v in f1)}。"),
        (f"F2 (+ #24) changes ITL by {' / '.join(f'{sg(v)}%' for v in f2)}.", f"F2（+ #24）让 ITL 分别变化 {' / '.join(f'{sg(v)}%' for v in f2)}。"),
        (f"F3 (+ #42) changes ITL by {' / '.join(f'{sg(v)}%' for v in f3)}.", f"F3（+ #42）让 ITL 分别变化 {' / '.join(f'{sg(v)}%' for v in f3)}。"),
        (f"TTFT stays within {ttft_fp8}% in every build, because all builds use the FP8 cache of the recipe.",
         f"每个版本的 TTFT 变化都在 {ttft_fp8}% 以内，因为所有版本都使用配方的 FP8 cache。"),
        (f"At concurrency 1, F1 and #15 of the BF16 ladder differ: {sg(ladder_eff('fp8', 'F1', 1, 'inter_token_latency'))}% against {sg(ladder_eff('bf16', '#15', 1, 'inter_token_latency'))}%. Section 09 explains this difference.",
         f"在 c1，F1 和 BF16 阶梯的 #15 结果不同：{sg(ladder_eff('fp8', 'F1', 1, 'inter_token_latency'))}% 对 {sg(ladder_eff('bf16', '#15', 1, 'inter_token_latency'))}%。第 09 节解释了这个差别。"),
    ]
    if drift:
        fp8_find.append((f"Drift check: a new base run at concurrency 2 differs from the old base runs by {sg(drift['inter_token_latency'])}% in ITL and {sg(drift['time_to_first_token'])}% in TTFT.",
                         f"漂移检查：c2 新跑的 base 和以前的 base 相比，ITL 差 {sg(drift['inter_token_latency'])}%，TTFT 差 {sg(drift['time_to_first_token'])}%。"))
s_fp8 = section("ablation-fp8", "08", ("Ablation 2: the PR ladder with FP8 KV", "Ablation 2：FP8 KV 下的 PR 阶梯"), "".join([
    P("To measure each PR on the real recipe, we cherry-picked #60 onto two older commits. F1 is 05e0bb02 (#7, #15) plus #60. F2 is 240d587 (#7, #15, #24) plus #60. F3 is main (all five PRs). Each build passed the FP8 layer check with 0 violations (section 07).",
      "为了在正式配方上测量每个 PR，我们把 #60 cherry-pick 到两个更早的提交上。F1 是 05e0bb02（#7、#15）加 #60。F2 是 240d587（#7、#15、#24）加 #60。F3 是 main（全部五个 PR）。每个版本都通过了 FP8 逐层检查，违例为 0（第 07 节）。"),
    P("Each point ran for 900 s on a new server, at concurrency 1, 2, and 4. We compare the same requests against the two base runs of Test 2. For F3 we use the two monokernel runs of Test 2.",
      "每个点用新的服务器跑 900 秒，并发为 1、2、4。我们把同一批请求和测试 2 的两次 base 运行比较。F3 使用测试 2 里 monokernel 组的两次运行。"),
    plate("IX", ("ITL with FP8 KV, cumulative", "FP8 KV 下的 ITL，累积"), plate_ladder("fp8"),
          ("Each bar includes the PRs above it. The base is the recipe without the monokernel.", "每根柱子都包含它上面的 PR。base 是不开 monokernel 的配方。")),
    table([("Build", "版本"), ("Concurrency", "并发"), ("TTFT", "TTFT"), ("ITL", "ITL"), ("E2EL", "E2EL"), ("Coverage", "覆盖率")], fp8_rows),
    UL(fp8_find) if fp8_find else P("The 900-s runs are in progress.", "900 秒的运行还在进行中。"),
    P(f"We use the paired metrics here. In 900-s windows, the A/A noise of the headline interactivity is {100 * noise2['e2e_int_p90']:.0f}%, too high for this comparison.",
      f"这里用的是配对指标。900 秒窗口下，头条交互性的 A/A 噪声有 {100 * noise2['e2e_int_p90']:.0f}%，太大，不适合这种比较。", "small"),
]))

pct = lambda vals: [f"{sg(v)}%" if v is not None else "-" for v in vals]  # noqa: E731
ctx_rows = []
for b_s, f_s in zip(LADDER_STEPS["bf16"], LADDER_STEPS["fp8"]):
    ctx_rows.append([sp(f"{'' if b_s == '#15' else '+ '}{b_s}, BF16", f"{'' if b_s == '#15' else '+ '}{b_s}，BF16"), *pct(isl_v("bf16", b_s))])
    ctx_rows.append([sp(f"{f_s}, FP8", f"{f_s}，FP8"), *pct(isl_v("fp8", f_s))])
ctx_rows.append([sp("Requests, BF16", "请求数，BF16"), *map(str, isl_n("bf16", "#15"))])
ctx_rows.append([sp("Requests, FP8", "请求数，FP8"), *map(str, isl_n("fp8", "F1"))])
big = [i for i, n in enumerate(isl_n("fp8", "F1")) if n >= 30]
gap_pts = [isl_v("fp8", f)[i] - isl_v("bf16", b)[i] for b, f in zip(LADDER_STEPS["bf16"], LADDER_STEPS["fp8"]) for i in big]
assert all(d > 0 for d in gap_pts), "the FP8 ladder is no longer less favorable in every large bin"
cov_min = min(e["coverage"] for lad in ("bf16", "fp8") for e in abl[lad]["effects"] if e["conc"] in (1, 2) and e["coverage"] is not None)
below = lambda lad, s: [abs(v) for v in isl_v(lad, s)[:3]]  # noqa: E731
b15c1, f1c1 = ladder_eff("bf16", "#15", 1, "inter_token_latency"), ladder_eff("fp8", "F1", 1, "inter_token_latency")
mix = {lad: (s, isl_share(lad, s, 1), [r["itl"] for r in ladder_runs(lad, s, 1)[0]["by_isl"]]) for lad, s in (("bf16", "#15"), ("fp8", "F1"))}
mix_rows = []
for lad, en, zh_ in (("bf16", "3600 s, #15 (BF16)", "3600 秒，#15（BF16）"), ("fp8", "900 s, F1 (FP8)", "900 秒，F1（FP8）")):
    s, shares, itls = mix[lad]
    mix_rows.append([sp(f"{en}: share of requests", f"{zh_}：请求占比"), *[f"{v:.0f}%" for v in shares], "-"])
    mix_rows.append([sp(f"{en}: ITL change", f"{zh_}：ITL 变化"), *pct(itls), f"<b>{sg(ladder_eff(lad, s, 1, 'inter_token_latency'))}%</b>"])
bin_head = [("Build", "版本"), *ISL_BINS]
s_ctx = section("context", "09", ("Gain by context length", "按上下文长度看收益"), "".join([
    P(f"The first steps of the two ladders disagree. At concurrency 1, #15 changed ITL by {sg(b15c1)}%, but F1 changed it by {sg(f1c1)}%.",
      f"两个阶梯的第一步结果不一致。在 c1，#15 让 ITL 变化 {sg(b15c1)}%，F1 却让它变化 {sg(f1c1)}%。"),
    P(f"To find the cause, we put each pair of requests into a bin by its input length. We pooled concurrency 1 and 2, where the coverage is {math.floor(100 * cov_min)}% or more. Then we compared the two ladders in the same bin.",
      f"为了找出原因，我们按输入长度把每一对请求分段。我们合并了 c1 和 c2，这两个并发下覆盖率都在 {math.floor(100 * cov_min)}% 以上。然后在同一段内比较两个阶梯。"),
    plate("X", ("ITL change by input length", "按输入长度划分的 ITL 变化"), plate_isl(),
          ("Median change of ITL on the same requests, concurrency 1 and 2 pooled. Up is slower. Hollow point: fewer than 30 requests.",
           "同一批请求上 ITL 变化的中位数，合并 c1 和 c2。向上表示更慢。空心点：请求数少于 30。")),
    table(bin_head, ctx_rows, "bins"),
    P("F3 pools its two runs at each concurrency. The other steps of a ladder have up to 3 more or fewer pairs than the first step.",
      "F3 合并了每个并发下的两次运行。同一阶梯的其他步，配对数和第一步最多相差 3 个。", "small"),
    UL([
        (f"In each bin with 30 or more requests, the two ladders agree in direction. The FP8 results are {round(min(gap_pts))} to {round(max(gap_pts))} percentage points worse.",
         f"在请求数不少于 30 的各段中，两个阶梯的方向一致。FP8 阶梯的结果差 {round(min(gap_pts))} 到 {round(max(gap_pts))} 个百分点。"),
        (f"#15 alone makes ITL {prng(below('bf16', '#15'))} lower below 192k, but changes it by {sg(isl_v('bf16', '#15')[3])}% at 192k or more. F1 shows the same pattern: {prng(below('fp8', 'F1'))} lower, and {sg(isl_v('fp8', 'F1')[3])}% at 192k or more.",
         f"只有 #15 时，192k 以下 ITL 降低 {prng(below('bf16', '#15'), True)}，192k 及以上却变化 {sg(isl_v('bf16', '#15')[3])}%。F1 的规律相同：192k 以下降低 {prng(below('fp8', 'F1'), True)}，192k 及以上变化 {sg(isl_v('fp8', 'F1')[3])}%。"),
        ("This agrees with section 05. Before #24, one CTA selected the top-2048 of a long row, and the other CTAs waited for it.",
         "这和第 05 节一致：#24 之前，长行的 top-2048 由一个 CTA 选出，其他 CTA 都在等它。"),
        (f"#24 removes most of this loss. At 192k or more, ITL changes by {sg(isl_v('bf16', '#24')[3])}% (BF16) and {sg(isl_v('fp8', 'F2')[3])}% (FP8).",
         f"#24 去掉了大部分损失。192k 及以上，ITL 变化 {sg(isl_v('bf16', '#24')[3])}%（BF16）和 {sg(isl_v('fp8', 'F2')[3])}%（FP8）。"),
        (f"#42 makes the gain almost equal at all lengths. ITL is {prng([abs(v) for v in isl_v('bf16', '#42')])} lower (BF16) and {prng([abs(v) for v in isl_v('fp8', 'F3')])} lower (FP8).",
         f"#42 让各段的收益几乎相同。ITL 降低 {prng([abs(v) for v in isl_v('bf16', '#42')], True)}（BF16）和 {prng([abs(v) for v in isl_v('fp8', 'F3')], True)}（FP8）。"),
    ]),
    P("The aggregate values differ because the two windows contain different requests. The aggregate is the median change over all requests of a run.",
      "汇总值不同，是因为两个窗口包含的请求不同。汇总值是一次运行中所有请求变化的中位数。"),
    table([("Window and build, c1", "窗口和版本，c1"), *ISL_BINS, ("All requests", "全部请求")], mix_rows, "bins"),
    P(f"At concurrency 1, {mix['bf16'][1][3]:.0f}% of the 3600-s requests are 192k or more, against {mix['fp8'][1][3]:.0f}% of the 900-s requests. These long requests are slower with #15 alone. Thus the median in the 3600-s window is near zero.",
      f"在 c1，3600 秒窗口里有 {mix['bf16'][1][3]:.0f}% 的请求在 192k 及以上，900 秒窗口里只有 {mix['fp8'][1][3]:.0f}%。只有 #15 时，这些长请求变慢。所以 3600 秒窗口的中位数接近 0。"),
    P("The two ladders also differ in image, SGLang commit, KV type, and window. Thus we cannot assign the remaining difference in each bin to one cause.",
      "两个阶梯的镜像、SGLang 提交、KV 类型和窗口都不同。所以每一段里剩下的差别，无法归于某一个原因。"),
    callout("Compare two ablations in the same context bin. Two windows of the same workload can have different context mixes.",
            "比较两组 ablation 时，要在同一个上下文长度段内比较。同一负载的两个窗口，上下文长度的分布可以不同。"),
]))

gain_rows = [[f"c{c}", str(eff[C3][(c, 'cand8-r1')]["requests"]), f"{sg(ttft3[c])}%", f"{sg(itl3[c])}%", f"{sg(e2el3[c])}%", f"{100 * share[c]:.0f}%"] for c in CONCS]
s_final = section("final", "10", ("Result at 3600 s", "3600 秒的结果"), "".join([
    P(f"Test 3 used the 3600-s window of InferenceX at concurrency 1, 2, 4, and 10. Each point used a new server. All nine runs were valid, and the measured accept length was 2.990 (target 2.99).",
      "测试 3 使用 InferenceX 的 3600 秒窗口，并发为 1、2、4、10。每个点都用新的服务器。九次运行全部有效，实测接受长度是 2.990（目标值 2.99）。"),
    plate("XI", ("Throughput and interactivity at 3600 s", "3600 秒的吞吐和交互性"), plate_score(),
          ("Each arrow goes from the base group to the monokernel group. Hollow points are published InferenceX results.",
           "每个箭头从 base 组指向 monokernel 组。空心点是 InferenceX 已发布的结果。")),
    table([("Concurrency", "并发"), ("Pairs", "配对数"), ("TTFT", "TTFT"), ("ITL", "ITL"), ("E2EL", "E2EL"), ("Coverage", "覆盖率")], gain_rows),
    plate("XII", ("Coverage and gain", "覆盖率和收益"), plate_gain(),
          ("Left: decode steps by the number of running requests. Right: change of ITL and E2EL on the same requests.",
           "左：按同时运行的请求数划分 decode 步。右：同一批请求上 ITL 和 E2EL 的变化。")),
    P(f"Coverage is the share of decode steps with one or two running requests. At concurrency 10 it is {100 * share[10]:.0f}%, so the gain is smaller there.",
      f"覆盖率是同时运行 1 到 2 个请求的 decode 步所占的比例。c10 的覆盖率是 {100 * share[10]:.0f}%，所以那里的收益更小。"),
]))

nogain_rows = [
    [sp(f"{pr(15)} alone on AgentX", f"只有 {pr(15)} 时的 AgentX"),
     sp(f"ITL {trio('bf16', '#15')} (BF16 ladder); {sg(isl_v('bf16', '#15')[3])}% and {sg(isl_v('fp8', 'F1')[3])}% at 192k or more",
        f"ITL {trio('bf16', '#15')}（BF16 阶梯）；192k 及以上 {sg(isl_v('bf16', '#15')[3])}% 和 {sg(isl_v('fp8', 'F1')[3])}%"),
     sp("needed, but slower at long contexts until #24", "必需，但在 #24 之前长上下文更慢")],
    [sp(f"{pr(19)}: cheaper guarded waits (open)", f"{pr(19)}：更便宜的 guarded wait（未合并）"), sp("TPOT c1 −2.5%, c4 −0.6%, c8 +2.3% (no MTP, fixed length)", "TPOT c1 −2.5%、c4 −0.6%、c8 +2.3%（不开 MTP，固定长度）"),
     sp("small, and slower at c8", "收益小，c8 还变慢")],
    [sp("histogram in the score stage (local attempt)", "在打分阶段建直方图（本地尝试）"), sp("correct, but global atomics slowed scoring: 0.87× to 1.17×", "结果正确，但全局 atomic 拖慢了打分：0.87× 到 1.17×"),
     sp("dropped", "已放弃")],
    [sp("route to the normal path by context length", "按上下文长度切回 normal path"), sp("not possible: the route is fixed for each captured decode graph", "做不到：每个 decode graph 捕获时路由就固定了"),
     sp("dropped", "已放弃")],
    [sp("f32 against bf16 expert routing (SGLang normal path)", "专家路由用 f32 还是 bf16（SGLang normal path）"),
     sp("accept length unchanged; c1 TPOT 5.36 → 5.52 ms with f32 (+3.0%); GSM8K 97.73% vs 97.57% (p = 0.77)", "接受长度不变；c1 TPOT 用 f32 时 5.36 → 5.52 ms（+3.0%）；GSM8K 97.73% 对 97.57%（p = 0.77）"),
     sp("no accuracy gain, small speed cost", "没有精度收益，速度略有损失")],
    [sp("monokernel with BF16 KV (Test 1)", "monokernel 配 BF16 KV（测试 1）"), sp(f"ITL {rng(pv[C1]['inter_token_latency'])}, TTFT {rng(pv[C1]['time_to_first_token'])}", f"ITL {zrng(pv[C1]['inter_token_latency'])}，TTFT {zrng(pv[C1]['time_to_first_token'])}"),
     sp("no-go against the FP8 recipe", "和 FP8 配方相比 no-go")],
    [sp("900-s windows for P90 decisions", "用 900 秒窗口判定 P90"), sp(f"A/A noise of interactivity {100 * noise2['e2e_int_p90']:.0f}%", f"交互性的 A/A 噪声 {100 * noise2['e2e_int_p90']:.0f}%"),
     sp("too noisy; use 3600 s", "噪声太大；改用 3600 秒")],
    [sp(f"{pr(18)}: A4W4, TP4 16-token scheduling, indexer CP (closed)", f"{pr(18)}：A4W4、TP4 16 token 调度、indexer CP（已关闭）"), sp("not measured here", "这里没有测"), "-"],
]
s_nogain = section("nogain", "11", ("What did not help", "没有帮助的尝试"), "".join([
    P("These changes or methods gave a small gain, no gain, or a loss. Some of them were needed anyway.",
      "这些改动或方法的收益很小、没有收益，或者带来了损失。其中有一些仍然是必需的。"),
    table([("Item", "项目"), ("Measurement", "测量结果"), ("Verdict", "结论")], nogain_rows),
]))

q_rows = [
    [pr(7), sp("GSM8K 1319, thinking", "GSM8K 1319，thinking"), "97.35% → 97.27%", sp("McNemar p = 1.00", "McNemar p = 1.00")],
    [pr(7), sp("AIME26, 30 × 8", "AIME26，30 × 8"), "82.92% → 86.67%", sp("inside noise", "在噪声内")],
    [pr(15), sp("GSM8K 1319, 5-shot, MTP", "GSM8K 1319，5-shot，MTP"), "92.65% → 92.34%", sp("p = 0.70; accept length 3.253 → 3.288", "p = 0.70；接受长度 3.253 → 3.288")],
    [pr(24), sp("exact top-k against an oracle", "精确 top-k 对照 oracle"), sp("312 rows, 0 differ", "312 行，0 行不同"), sp("8k to 256k keys", "8k 到 256k key")],
    [pr(42), sp("scores against #24 in one launch", "同一次 launch 里和 #24 的打分比较"), sp("59.6M keys, all bitwise equal", "5960 万个 key，全部逐位相等"), sp("decode and MTP verify", "decode 和 MTP verify")],
    [pr(60), sp("GSM8K 1319, three pairs, strict-match", "GSM8K 1319，三组，strict-match"), f"{ob} / {oc}", sp(f"McNemar p = {mcnemar_p(ob, oc):.2f}", f"McNemar p = {mcnemar_p(ob, oc):.2f}")],
]
s_quality = section("accuracy", "12", ("Accuracy", "精度"), "".join([
    P("Each PR tested accuracy in its own way. No PR showed a measurable loss. The last row compares base and monokernel answers on the real recipe at 3600-s settings.",
      "每个 PR 用自己的方法测试了精度。没有一个 PR 显示出可测量的损失。最后一行是在正式配方上，比较 base 和 monokernel 的答案。"),
    table([("PR", "PR"), ("Test", "测试"), ("Result", "结果"), ("Note", "说明")], q_rows),
    P(f"For #60, two runs of the monokernel group differ by {rerun_c['only_a']} to {rerun_c['only_b']} questions, more than the base-vs-monokernel pairs. Thus the differences are run-to-run variation.",
      f"对 #60，monokernel 组自己跑两次就差 {rerun_c['only_a']} 对 {rerun_c['only_b']} 题，比 base 对 monokernel 的差别还大。所以这些差别是两次运行之间的波动。"),
]))

ref_rows = [[f"c{r['conc']}", E(r["curve"]), f"{r['tput']:.0f}", f"{r['int90']:.1f}", f"{r['e2el90']:.1f}", a(r["url"], E(r["source"])) if r.get("url") else E(r["source"])]
            for r in refs3 if "this work" not in r["curve"]]
s_refs = section("refs", "13", ("Comparison with published results", "和已发布结果比较"), "".join([
    P(f"The InferenceX API has no vLLM AgentX result for GLM-5.x. For GLM-5.3 on MI355X, it has one TileRT point at concurrency 1 on 16 GPUs. Against it, the monokernel group has {sg(rel(K(1, 'tput_gpu'), tile['tput']), 0)}% throughput per GPU and {sg(rel(K(1, 'e2e_int_p90'), tile['int90']), 0)}% interactivity.",
      f"InferenceX 的 API 里没有 GLM-5.x 的 vLLM AgentX 结果。MI355X 上的 GLM-5.3 只有一个 TileRT 的点：c1，16 张 GPU。和它相比，monokernel 组的每卡吞吐是 {sg(rel(K(1, 'tput_gpu'), tile['tput']), 0)}%，交互性是 {sg(rel(K(1, 'e2e_int_p90'), tile['int90']), 0)}%。"),
    P(f"GLM-5.3 on B300 with NVFP4 is better. At concurrency 10, our throughput is {abs(rel(K(10, 'tput_gpu'), b300[10]['tput'])):.0f}% lower and our interactivity is {abs(rel(K(10, 'e2e_int_p90'), b300[10]['int90'])):.0f}% lower. Our P90 output length is {min(osl_ratio):.1f} to {max(osl_ratio):.1f} times the B300 value, so do not compare E2EL directly.",
      f"B300 上的 GLM-5.3 NVFP4 更好。在 c10，我们的吞吐低 {abs(rel(K(10, 'tput_gpu'), b300[10]['tput'])):.0f}%，交互性低 {abs(rel(K(10, 'e2e_int_p90'), b300[10]['int90'])):.0f}%。我们的 P90 输出长度是 B300 的 {min(osl_ratio):.1f} 到 {max(osl_ratio):.1f} 倍，所以不要直接比较 E2EL。"),
    table([("Concurrency", "并发"), ("Published configuration", "已发布配置"), ("Throughput", "吞吐"), ("Interactivity", "交互性"), ("P90 E2EL (s)", "P90 E2EL（秒）"), ("Source", "来源")], ref_rows),
]))

s_up = section("upstream", "14", ("Steps to an InferenceX submission", "提交 InferenceX 还需要的步骤"), "".join([
    P("The kernel work is complete. These steps remain, in this order:", "kernel 的工作已经完成。还剩这些步骤，按顺序进行："),
    plate("XIII", ("Remaining steps", "剩下的步骤"), plate_upstream(),
          ("State on 2026-10-26. Each step needs the step before it.", "状态截至 2026-10-26。每一步都依赖前一步。")),
    OL([
        (f"SGLang pin update. SGLang main pins sgl-amd-mono at <code>05e0bb02</code>, which accepts only BF16 KV. The open PR {pr(45045, 'sgl-project/sglang')} pins <code>e952b221</code>, which is also older than #60.",
         f"SGLang 更新 pin。SGLang main 把 sgl-amd-mono pin 在 <code>05e0bb02</code>，它只接受 BF16 KV。现有的 PR {pr(45045, 'sgl-project/sglang')} pin 的是 <code>e952b221</code>，也比 #60 早。"),
        ("Image. Use the first MI35x nightly image that includes the new pin.", "镜像。使用第一个包含新 pin 的 MI35x nightly 镜像。"),
        (f"Cookbook. {pr(46662, 'sgl-project/sglang')} (draft) adds the recipe to the GLM-5.3 page.", f"cookbook。{pr(46662, 'sgl-project/sglang')}（draft）把配方加到 GLM-5.3 页面。"),
        ("Confirmation run on that image without the overlay, with five or more frontier points.", "在那个镜像上不加 overlay 做确认运行，至少要有 5 个前沿点。"),
        ("InferenceX PR, sweep, evals, CODEOWNER sign-off, and an authorized <code>/use</code>.", "InferenceX PR、sweep、eval、CODEOWNER 签核，以及有权限的人发出 <code>/use</code>。"),
    ]),
]))

s_method = section("method", "15", ("Method and limits", "方法和限制"), UL([
    ("We used one local node, not the InferenceX cluster. The monokernel commits were installed over the published image.",
     "我们用的是一台本地节点，不是 InferenceX 的集群。monokernel 的提交装在已发布的镜像上。"),
    ("The BF16 ladder comes from an older image and SGLang (20261019, e45cbce599). Compare the two ladders only in the same context bin (section 09).",
     "BF16 阶梯来自更早的镜像和 SGLang（20261019，e45cbce599）。两个阶梯只能在同一个上下文长度段内比较（第 09 节）。"),
    (f"The context bins pool concurrency 1 and 2. The FP8 bin below 64k has only {isl_n('fp8', 'F1')[0]} requests.",
     f"上下文长度分段合并了 c1 和 c2。FP8 阶梯 64k 以下的分段只有 {isl_n('fp8', 'F1')[0]} 个请求。"),
    ("The BF16 ladder ran one run per point, in PR order, from 2026-10-20 to 10-22. Thus day-to-day drift is not separated from the PR effects.",
     "BF16 阶梯每个点只跑一次，按 PR 的先后顺序，在 2026-10-20 到 10-22 跑完。所以不同日子之间的漂移没有和 PR 的效果分开。"),
    ("The FP8 ladder ran F1 and F2 one time each at 900 s. F3 and the base have two runs each. The paired A/A noise of ITL is below 0.5%.",
     "FP8 阶梯里，F1 和 F2 各跑一次 900 秒。F3 和 base 各有两次运行。配对 ITL 的 A/A 噪声低于 0.5%。"),
    ("The kernel timelines come from the PR descriptions (debug builds, decode steps, random prompts).", "kernel timeline 来自 PR 正文（debug 版本，decode 步，随机 prompt）。"),
    ("The AgentX runs use the synthetic acceptance of InferenceX (accept length 2.99). The GSM8K runs use real acceptance.",
     "AgentX 运行使用 InferenceX 的模拟接受（接受长度 2.99）。GSM8K 运行使用真实接受。"),
    ("Before each run, we checked that all 8 GPUs were idle and locked them. Each point used a new server.",
     "每次运行前，我们检查 8 张 GPU 都空闲，然后锁定。每个点都用新的服务器。"),
]))

files = [
    ("ablation/ablation.json", "both ladders: paired effects, accept length, coverage, effects by input length", "两个阶梯：配对效果、接受长度、覆盖率、按输入长度分段的效果"),
    ("prs/timelines.json", "the kernel timeline tables of #24 and #42", "#24 和 #42 的 kernel timeline 表"),
    ("ladder-bf16-3600s", "the BF16 ladder runs", "BF16 阶梯的运行"),
    ("ladder-fp8-900s", "the FP8 ladder runs (F1, F2, base rerun)", "FP8 阶梯的运行（F1、F2、base 重跑）"),
    ("cohort3-fp8kv-3600s/analysis/compare_round1_base_cand8.json", "the 3600-s headline metrics", "3600 秒的头条指标"),
    ("cohort3-fp8kv-3600s/requests", "one CSV row per request of each 3600-s run", "每个 3600 秒运行的逐请求 CSV"),
    ("fp8kv-validation/e2e_checks.json", "the e2e checks, including F1 and F2", "e2e 检查，包括 F1 和 F2"),
    ("scripts/make_page.py", "the script that builds this page", "生成本页的脚本"),
]
link = lambda p: a(f"{BLOB}/{p}" if "." in p.split("/")[-1] else f"{DATA_URL}/{p}", f"<code>{p}</code>")  # noqa: E731
s_data = section("data", "16", ("Data and files", "数据和文件"), "".join([
    P(f"The raw data, the decision rules, and the analysis results are in {a(DATA_URL, f'<code>data/{SLUG}/</code>')}. We replaced host names and local paths.",
      f"原始数据、判定规则和分析结果都在 {a(DATA_URL, f'<code>data/{SLUG}/</code>')}。我们替换了主机名和本机路径。"),
    UL([(f"{link(p)}: {en}", f"{link(p)}：{z}") for p, en, z in files]),
]))

s_lessons = section("lessons", "17", ("Lessons", "经验"), UL([
    ("Test a kernel PR at the context lengths of the target workload. #15 made a verify step 38.5% faster with 4096-token prompts. On AgentX, it made requests above 192k tokens slower.",
     "要用目标负载的上下文长度测试 kernel PR。#15 在 4096 token 的 prompt 上让 verify 步快了 38.5%。在 AgentX 上，它却让 192k token 以上的请求变慢。"),
    ("Compare ablation results in the same context bin. In the same workload, a 900-s window and a 3600-s window had different shares of long requests.",
     "要在同一个上下文长度段内比较 ablation 结果。同一个负载，900 秒窗口和 3600 秒窗口里长请求的占比不同。"),
    ("Find the stage that grows with the workload. At long contexts, the indexer selection and score stages set the step time. #24 and #42 gave most of the gain.",
     "找出随负载增长的那个阶段。长上下文下，indexer 的选择和打分阶段决定了单步时间。#24 和 #42 带来了大部分收益。"),
    ("Compare against the published recipe. The BF16 baseline of the PR campaign hid the TTFT cost of BF16 KV. #60 removed that cost.",
     "要和已发布的配方比较。PR 开发期间的 BF16 对照组掩盖了 BF16 KV 的 TTFT 代价。#60 去掉了这个代价。"),
    (f"Use the window of the score for P90 decisions. In 900 s, two identical runs differed by {100 * noise2['e2e_int_p90']:.0f}%. In 3600 s, they differed by {100 * noise3['e2e_int_p90']:.1f}%.",
     f"判定 P90 时，要用分数所用的窗口。900 秒时，两次相同的运行相差 {100 * noise2['e2e_int_p90']:.0f}%。3600 秒时，只相差 {100 * noise3['e2e_int_p90']:.1f}%。"),
]))

SECTIONS = [("result", "01", "Result", "结果"), ("monokernel", "02", "The monokernel", "monokernel"), ("metric", "03", "How the score works", "打分方式"),
            ("prs", "04", "The PR series", "PR 系列"), ("longctx", "05", "Long contexts", "长上下文"), ("ablation-bf16", "06", "Ablation: BF16 KV", "Ablation：BF16 KV"),
            ("baseline", "07", "Baseline gap and #60", "对照差距与 #60"), ("ablation-fp8", "08", "Ablation: FP8 KV", "Ablation：FP8 KV"), ("context", "09", "Context length", "上下文长度"),
            ("final", "10", "Result at 3600 s", "3600 秒结果"), ("nogain", "11", "What did not help", "没有帮助的尝试"), ("accuracy", "12", "Accuracy", "精度"),
            ("refs", "13", "Published results", "已发布结果"), ("upstream", "14", "Steps to InferenceX", "提交步骤"),
            ("method", "15", "Method and limits", "方法和限制"), ("data", "16", "Data and files", "数据和文件"), ("lessons", "17", "Lessons", "经验")]
rail = ('<nav class="rail" aria-label="Contents"><a class="back" href="/sources/">' + sp("← The Library", "← 文库") + "</a><ol>"
        + "".join(f'<li><a href="#{sid}"><span class="rn">{num}</span>{sp(en, z)}</a></li>' for sid, num, en, z in SECTIONS) + "</ol></nav>")

bf16_42 = [abs(ladder_eff("bf16", "#42", c, "inter_token_latency")) for c in (1, 2, 4)]
summary = [
    ("The glm53 monokernel runs most of a GLM-5.3 decode or verify step in 75 launches, one per MoE layer. Five PRs built it: #7, #15, #24, #42, and #60.",
     "glm53 monokernel 用 75 次 launch（每个 MoE 层一次）完成 GLM-5.3 一个 decode 或 verify 步的大部分计算。它由五个 PR 构成：#7、#15、#24、#42 和 #60。"),
    ("#15 let the monokernel serve MTP verify steps. Alone, it made requests below 192k tokens faster and longer requests slower, because the indexer was slow at long contexts.",
     "#15 让 monokernel 能运行 MTP 的 verify 步。只有它时，192k token 以下的请求变快，更长的请求变慢，因为长上下文下 indexer 很慢。"),
    (f"#24 and #42 made the indexer fast at long contexts. Together they reduced ITL by {round(min(bf16_42))}–{round(max(bf16_42))}% on the same requests.",
     f"#24 和 #42 让 indexer 在长上下文下变快。两者合计让同一批请求的 ITL 降低 {round(min(bf16_42))}–{round(max(bf16_42))}%。"),
    ("These gains used BF16 KV in both groups. Against the FP8 recipe, the score improved only after #60 let the monokernel use the FP8 cache.",
     "这些收益是在两组都用 BF16 KV 时测的。和 FP8 配方相比，直到 #60 让 monokernel 使用 FP8 cache，分数才提高。"),
    (f"Result at 3600 s: throughput +{gain_lo('tput_gpu')}–{gain_hi('tput_gpu')}% and interactivity +{gain_lo('e2e_int_p90')}–{gain_hi('e2e_int_p90')}% at concurrency 1–4. GSM8K did not change.",
     f"3600 秒的结果：c1–c4 的吞吐提高 {gain_lo('tput_gpu')}–{gain_hi('tput_gpu')}%，交互性提高 {gain_lo('e2e_int_p90')}–{gain_hi('e2e_int_p90')}%。GSM8K 不变。"),
]
spec = [
    (("Hardware", "硬件"), ("8 × AMD Instinct MI355X (gfx950), one node, TP8 EP1", "8 × AMD Instinct MI355X（gfx950），单节点，TP8 EP1")),
    (("Model", "模型"), (f"{a('https://huggingface.co/zai-org/GLM-5.3', 'zai-org/GLM-5.3')} at <code>ba32f8c1</code> (FP8), EAGLE MTP 3-1-4", f"{a('https://huggingface.co/zai-org/GLM-5.3', 'zai-org/GLM-5.3')}，<code>ba32f8c1</code>（FP8），EAGLE MTP 3-1-4")),
    (("Monokernel", "monokernel"), (f"sgl-amd-mono glm53: {pr(7)}, {pr(15)}, {pr(24)}, {pr(42)}, {pr(60)}", f"sgl-amd-mono glm53：{pr(7)}、{pr(15)}、{pr(24)}、{pr(42)}、{pr(60)}")),
    (("Engine", "引擎"), ("SGLang <code>fe1bd108</code> in <code>lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261024</code>", "SGLang <code>fe1bd108</code>，镜像 <code>lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261024</code>")),
    (("Client", "客户端"), ("InferenceX AgentX (AIPerf), accept length 2.99", "InferenceX AgentX（AIPerf），接受长度 2.99")),
    (("Measured", "测量时间"), ("2026-10-20 to 2026-10-27", "2026-10-20 到 2026-10-27")),
    (("Raw data", "原始数据"), (a(DATA_URL, f"<code>data/{SLUG}/</code>"), a(DATA_URL, f"<code>data/{SLUG}/</code>"))),
]
TITLE = "GLM-5.3 on MI355X: a retrospective of the glm53 monokernel hill-climb"
TITLE_ZH = "GLM-5.3 在 MI355X 上的 glm53 monokernel hill-climbing 复盘"
cover_note = P("Cover: ITL of each request in the 3600-s runs. Slate: base. Raspberry: monokernel. Left is faster.",
               "封面：3600 秒测试中每个请求的 ITL。蓝灰色是 base 组，覆盆子红是 monokernel 组。越靠左越快。", "cover-note")
masthead = (f'<div class="cover">{cover_svg()}</div>'
            + '<header class="masthead"><div class="mast-in">' + cover_note
            + bi("div", "Experiment 006 · Model hill-climbing", "实验 006 · 模型 hill-climbing 记录", 'class="kicker"')
            + bi("h1", TITLE, TITLE_ZH)
            + '<ul class="summary">' + "".join(bi("li", en, z) for en, z in summary) + "</ul>"
            + '<dl class="spec">' + "".join(f"<dt>{sp(*k)}</dt><dd>{sp(*v)}</dd>" for k, v in spec) + "</dl></div></header>")
BODY = [s_result, s_mono, s_metric, s_prs, s_long, s_bf16, s_gap, s_fp8, s_ctx, s_final, s_nogain, s_quality, s_refs, s_up, s_method, s_data, s_lessons]

CSS = """
:root{--paper:#f3f5f2;--card:#ffffff;--sunk:#e8ede9;--rule:#d3dbd5;--rule2:#a5b5aa;--ink:#13211d;--ink2:#34463f;--ink3:#56685f;
--base:#4a6787;--mono:#c2366b;--bf16:#a77a1f;--ref:#8a958f;--good:#2e7d4f;--warn:#b5651d;--bad:#b3392f;
--serif:"Instrument Serif",Georgia,"Noto Serif SC",serif;--sans:"Public Sans","Noto Sans SC","PingFang SC","Microsoft YaHei",sans-serif;--mono-f:"Reddit Mono",ui-monospace,Menlo,Consolas,monospace}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:var(--paper);color:var(--ink);font:17px/1.7 var(--sans);-webkit-font-smoothing:antialiased}
a{color:var(--base);text-decoration:underline;text-decoration-color:var(--rule2);text-underline-offset:2px}a:hover{color:var(--mono)}
code{font:0.85em var(--mono-f);background:var(--sunk);padding:1px 5px;border-radius:4px}
.cover{position:relative;height:clamp(150px,20vw,250px);overflow:hidden;background:linear-gradient(90deg,#e3eae5,#eef2ee 55%,#e9eee9);border-bottom:1px solid var(--rule)}
.cover svg{position:absolute;inset:0;width:100%;height:100%;display:block}
.cover .ridge{stroke-width:1.2;vector-effect:non-scaling-stroke;stroke-linejoin:round}
.cover .ridge.base{fill:#e6ecef;stroke:rgba(74,103,135,.42)}.cover .ridge.mono{fill:#f5eaee;stroke:rgba(194,54,107,.42)}
.masthead{background:var(--card);border-bottom:1px solid var(--rule)}
.mast-in{max-width:1000px;margin:0 auto;padding:14px 28px 34px}
.cover-note{margin:0 0 26px;text-align:right;font-size:12.5px;line-height:1.5;color:var(--ink3)}
.kicker{font:600 13px var(--mono-f);letter-spacing:.12em;text-transform:uppercase;color:var(--mono)}
h1{font:400 clamp(30px,4.2vw,46px)/1.12 var(--serif);margin:12px 0 22px;max-width:880px}
.summary{margin:0 0 26px;padding:0;list-style:none;max-width:820px;border-left:3px solid var(--mono)}
.summary li{margin:0;padding:6px 0 6px 16px;color:var(--ink2)}
.spec{display:grid;grid-template-columns:140px 1fr;gap:6px 18px;margin:0;max-width:880px;font-size:15px;border-top:1px solid var(--rule);padding-top:16px}
.spec dt{font:600 12px var(--mono-f);text-transform:uppercase;letter-spacing:.08em;color:var(--ink3);padding-top:4px}.spec dd{margin:0}
.layout{display:grid;grid-template-columns:210px minmax(0,1fr);gap:44px;max-width:1240px;margin:0 auto;padding:36px 28px 80px}
.rail{position:sticky;top:20px;align-self:start;font-size:14.5px}.rail .back{display:inline-block;margin-bottom:16px;font-weight:700;text-decoration:none;color:var(--ink)}
.rail ol{list-style:none;margin:0;padding:0;border-left:2px solid var(--rule)}.rail li a{display:flex;gap:10px;padding:5px 12px;color:var(--ink2);text-decoration:none}
.rail li a:hover{color:var(--mono)}.rn{font:600 12px var(--mono-f);color:var(--ink3);padding-top:3px}
.article{max-width:740px}.article section{margin:0 0 56px}
h2{font:400 32px/1.2 var(--serif);margin:0 0 16px;display:flex;gap:14px;align-items:baseline}
h2 .num{font:600 14px var(--mono-f);color:var(--mono);letter-spacing:.08em}
p{margin:0 0 14px}.small{font-size:14.5px;color:var(--ink3)}ul,ol{padding-left:22px;margin:0 0 16px}li{margin:0 0 7px}
.tw{overflow-x:auto;margin:6px 0 18px}table{border-collapse:collapse;width:100%;font-size:14.5px;font-variant-numeric:tabular-nums;background:var(--card)}
th,td{border-bottom:1px solid var(--rule);padding:8px 10px;text-align:left;vertical-align:top}th{font:600 12px var(--mono-f);text-transform:uppercase;letter-spacing:.05em;color:var(--ink3);background:var(--sunk)}
table.score td b{color:var(--mono)}.up{color:var(--good);font-weight:700}.flat{color:var(--ink3);font-weight:700}
.callout{border-left:4px solid var(--base);background:var(--card);padding:12px 16px;margin:18px 0;border-radius:0 8px 8px 0}.callout p{margin:0}
.plate{margin:24px 0 26px;background:var(--card);border:1px solid var(--rule);border-radius:12px;padding:14px 16px 12px;width:min(1040px,calc(100vw - 330px))}
.plate-meta{display:flex;gap:12px;align-items:baseline;margin-bottom:8px;font-weight:700}.pn{font:600 12px var(--mono-f);color:var(--mono);letter-spacing:.1em;text-transform:uppercase}
.plate svg{width:100%;height:auto;display:block}.caption{color:var(--ink2);font-size:14.5px;margin-top:8px}
svg .lbl{font:14px var(--sans);fill:var(--ink)}svg .lbl.s{font-size:12.5px;fill:var(--ink2)}svg .lbl.b{font-weight:700}svg .lbl.w{fill:#fff;font-weight:700}
svg .ahd{fill:var(--ink2)}svg .seg.ttft,svg .seg.bf16{fill:var(--bf16)}svg .seg.dec{fill:var(--rule2)}svg .seg.base{fill:var(--base)}svg .seg.mono{fill:var(--mono)}
svg .bar.bf16{fill:var(--bf16)}svg .bar.mono{fill:var(--mono)}svg .bar.base{fill:var(--base)}svg .axis{stroke:var(--ink2);stroke-width:1.2}svg .tick{stroke:var(--ink);stroke-width:2;stroke-dasharray:3 3}
svg .box{fill:var(--paper);stroke:var(--rule2);stroke-width:1.4}svg .box.pool{fill:#e9efea;stroke:var(--ink2)}svg .box.mono{stroke:var(--mono)}
svg .edge{stroke:var(--ink2);stroke-width:1.6;fill:none}svg .edge.mono{stroke:var(--mono)}
svg .grid{stroke:var(--rule);stroke-width:1}svg .pt{stroke:var(--card);stroke-width:2}svg .pt.base{fill:var(--base)}svg .pt.mono{fill:var(--mono)}
svg .move{stroke:var(--ink2);stroke-width:1.5}svg .refpt{fill:var(--card);stroke:var(--ref);stroke-width:2}svg .refline{fill:none;stroke:var(--ref);stroke-width:1.2;stroke-dasharray:4 4}
svg .box.st-done{stroke:var(--good);stroke-width:2}svg .box.st-open{stroke:var(--warn);stroke-width:2}svg .box.st-bad{stroke:var(--bad);stroke-width:2}svg .box.st-plain{stroke:var(--ink2);stroke-width:1.6}svg .box.st-todo{stroke:var(--rule2);stroke-dasharray:5 4}
svg .chip{font:600 12px var(--mono-f);text-transform:uppercase;letter-spacing:.08em}svg .c-done{fill:var(--good)}svg .c-open{fill:var(--warn)}svg .c-bad{fill:var(--bad)}svg .c-plain{fill:var(--ink2)}svg .c-todo{fill:var(--ink3)}
.lang-toggle{position:fixed;top:14px;right:16px;z-index:10;display:flex;border:1px solid var(--rule2);border-radius:999px;overflow:hidden;background:var(--card)}
.lang-toggle button{border:0;background:transparent;padding:6px 12px;font:600 12px var(--mono-f);color:var(--ink2);cursor:pointer}
body[data-lang="en"] .lang-toggle button[data-set="en"],body[data-lang="zh"] .lang-toggle button[data-set="zh"]{background:var(--ink);color:var(--card)}
.colophon{max-width:1000px;margin:0 auto;padding:22px 28px 60px;border-top:1px solid var(--rule);font-size:13.5px;color:var(--ink3)}
@media (max-width:980px){.layout{grid-template-columns:1fr}.rail{position:static}.rail ol{columns:2}.plate{width:100%}.spec{grid-template-columns:1fr}}
svg .line{fill:none;stroke-width:2.2}svg .line.ref{stroke:var(--ref)}svg .line.base{stroke:var(--base)}svg .line.mono{stroke:var(--mono)}
svg .pt.ref{fill:var(--ref)}svg .bar.ref{fill:var(--ref)}svg .edge.ref{stroke:var(--ref)}
svg .pt.hollow{fill:var(--card)}svg .pt.hollow.ref{stroke:var(--ref)}svg .pt.hollow.base{stroke:var(--base)}svg .pt.hollow.mono{stroke:var(--mono)}
table.contrib td:first-child{white-space:nowrap}table.bins th{text-transform:none;letter-spacing:0}
body[data-lang="en"] [lang="zh"]:not(html){display:none !important}
body[data-lang="zh"] [lang="en"]:not(html){display:none !important}
"""

JS = """<script>
(function(){var KEY='%s-source-lang',body=document.body,stored=null;
try{stored=localStorage.getItem(KEY);}catch(e){}
if(stored==='en'||stored==='zh'){body.setAttribute('data-lang',stored);}
else{var nav=(navigator.language||'en').toLowerCase();body.setAttribute('data-lang',nav.indexOf('zh')===0?'zh':'en');}
document.querySelectorAll('.lang-toggle button[data-set]').forEach(function(btn){btn.addEventListener('click',function(){
var v=btn.getAttribute('data-set');body.setAttribute('data-lang',v);try{localStorage.setItem(KEY,v);}catch(e){}});});})();
</script>""" % SLUG

FONTS = ("https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Public+Sans:ital,wght@0,400;0,700;1,400"
         "&family=Reddit+Mono:wght@400;600&family=Noto+Sans+SC:wght@400;700&display=swap")

colophon = ('<footer class="colophon">' + P(
    f"Experiment 006 on jinnpan.com. {a(f'{BLOB}/scripts/make_page.py', '<code>scripts/make_page.py</code>')} builds this page from {a(DATA_URL, 'the archived data')}. Fonts: Instrument Serif, Public Sans, Reddit Mono. Colors: slate blue for the base group, raspberry for the monokernel group.",
    f"jinnpan.com 实验 006。{a(f'{BLOB}/scripts/make_page.py', '<code>scripts/make_page.py</code>')} 从{a(DATA_URL, '归档数据')}生成本页。字体：Instrument Serif、Public Sans、Reddit Mono。颜色：蓝灰色表示 base 组，覆盆子红表示 monokernel 组。") + "</footer>")

page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{TITLE} · Experiment 006</title>
<meta name="description" content="A retrospective of the glm53 monokernel for GLM-5.3 on MI355X: what each sgl-amd-mono PR added on InferenceX AgentX, with an ablation on BF16 and FP8 KV. Final: +15-19% throughput at concurrency 1-4 over 3600 s.">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="{FONTS}">
<style>{CSS}</style>
</head>
<body data-lang="en">
<div class="lang-toggle" role="group" aria-label="Language"><button type="button" data-set="en" aria-label="English">EN</button><button type="button" data-set="zh" aria-label="中文">中文</button></div>
{masthead}
<div class="layout">
{rail}
<main class="article">
{"".join(BODY)}
</main>
</div>
{colophon}
{JS}
</body>
</html>
"""
OUT.write_text(page)
print(f"wrote {OUT} ({len(page) / 1e3:.0f} KB)")
