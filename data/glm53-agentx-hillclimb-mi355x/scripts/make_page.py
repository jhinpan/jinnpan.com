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
    return ("+" if x > 0 else "−" if x < 0 else "±") + s


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


# ----------------------------------------------------------------------------- sections
cell = lambda c, m, nd, unit="": f"{B(c, m):.{nd}f} → <b>{K(c, m):.{nd}f}</b>{unit}<br><span class=\"up\">{sg(g(c, m))}%</span>"  # noqa: E731
score_rows = [[f"c{c}", cell(c, "tput_gpu", 0), cell(c, "e2e_int_p90", 1), cell(c, "e2el_p90", 1, " s"), cell(c, "ttft_p90", 2, " s")] for c in CONCS]
score_head = [("Concurrency", "并发"), ("Throughput (tok/s/GPU)", "吞吐（tok/s/GPU）"), ("Interactivity", "交互性"), ("P90 E2EL", "P90 E2EL"), ("P90 TTFT", "P90 TTFT")]

s_result = section("result", "01", ("Result", "结果"), "".join([
    P("We ran the InferenceX recipe in two groups: without the monokernel (base) and with it (monokernel). Each run used the 3600-s window of InferenceX.",
      "我们用 InferenceX 的配方跑了两组：不开 monokernel（base 组）和开 monokernel（monokernel 组）。每次都用 InferenceX 的 3600 秒窗口。"),
    P(f"The monokernel group is better on all three headline metrics at each concurrency. Each gain is larger than the A/A noise at concurrency 2. The A/A noise is {100 * noise3['tput_gpu']:.1f}% for throughput, {100 * noise3['e2e_int_p90']:.1f}% for interactivity, and {100 * noise3['e2el_p90']:.1f}% for P90 E2EL.",
      f"在每个并发点，monokernel 组的三个头条指标都更好。每个增益都大于 c2 的 A/A 噪声。A/A 噪声是：吞吐 {100 * noise3['tput_gpu']:.1f}%，交互性 {100 * noise3['e2e_int_p90']:.1f}%，P90 E2EL {100 * noise3['e2el_p90']:.1f}%。"),
    table(score_head, score_rows, "score"),
    P("Each cell shows base → monokernel and the gain. For a latency, the gain is base / monokernel − 1. A positive gain is always better.",
      "每格显示 base → monokernel 和增益。延迟的增益等于 base / monokernel − 1。增益为正，总是表示更好。", "small"),
    UL([
        ("The monokernel code is the same in Test 1 and Test 3. Only the KV cache type changed (sections 04 to 06).",
         "测试 1 和测试 3 的 monokernel 代码相同。只有 KV cache 的类型变了（第 04 到 06 节）。"),
        (f"Accuracy did not change. On 1319 GSM8K questions, the two groups differ by {ob} to {oc} questions (McNemar p = {mcnemar_p(ob, oc):.2f}). Two runs of the monokernel group differ by {rerun_c['only_a']} to {rerun_c['only_b']} (section 09).",
         f"精度没有变化。在 1319 道 GSM8K 题上，两组只有一方答对的题是 {ob} 对 {oc}（McNemar p = {mcnemar_p(ob, oc):.2f}）。monokernel 组自己跑两次，也差 {rerun_c['only_a']} 对 {rerun_c['only_b']}（第 09 节）。"),
        ("The InferenceX submission waits for three steps: an SGLang pin update, a new image, and the cookbook PR (section 11).",
         "提交 InferenceX 之前还有三步：SGLang 更新 pin、发布新镜像、合并 cookbook PR（第 11 节）。"),
    ]),
]))

s_steps = section("steps", "02", ("Test sequence", "测试过程"), "".join([
    P("The work had five steps. We wrote the decision rule of each test before the test started.",
      "整个工作分五步。每次测试的判定规则都在测试开始前写好。"),
    plate("I", ("Steps and results", "步骤和结果"), plate_steps(),
          ("Each box shows the change and the result. Green: passed or merged. Amber: inside noise. Red: no-go.",
           "每个方框写明改动和结果。绿色：通过或已合并。琥珀色：在噪声内。红色：no-go。")),
]))

s_metric = section("metric", "03", ("How InferenceX scores AgentX", "InferenceX 怎样给 AgentX 打分"), "".join([
    P("AgentX replays recorded multi-turn agent sessions. The prompts are long. Each turn adds text to the prompt and reuses the cached prefix. Concurrency is the number of live sessions.",
      "AgentX 回放录制的多轮 agent 会话。prompt 很长。每一轮都在 prompt 后面追加内容，并复用已缓存的前缀。并发数是同时在跑的会话数。"),
    P("InferenceX reports three headline metrics at each concurrency:", "InferenceX 在每个并发点报告三个头条指标："),
    UL([("Throughput: total tokens per second per GPU.", "吞吐：每张 GPU 每秒处理的总 token 数。"),
        ("Interactivity: 1 / P90 of (E2EL / OSL), in tokens per second per user. This page uses \"interactivity\" for this metric.",
         "交互性：1 / (E2EL / OSL) 的 P90，单位是每用户每秒 token。本页用“交互性”指这个指标。"),
        ("P90 E2EL: the 90th percentile of the end-to-end latency.", "P90 E2EL：端到端延迟的 90 分位。")]),
    P("Interactivity is sensitive to TTFT, for these reasons:", "交互性对 TTFT 很敏感，原因如下："),
    OL([("E2EL = TTFT + (OSL − 1) × ITL.", "E2EL = TTFT + (OSL − 1) × ITL。"),
        ("For a long reply, most of E2EL is ITL. For a short reply, most of E2EL is TTFT.", "长回复的 E2EL 主要是 ITL。短回复的 E2EL 主要是 TTFT。"),
        ("The P90 of E2EL / OSL comes from the short replies.", "E2EL / OSL 的 P90 由短回复决定。"),
        ("Thus a faster decode changes interactivity only a little. A longer TTFT makes interactivity worse.",
         "所以 decode 变快，交互性只改善一点。TTFT 变长，交互性就变差。")]),
    plate("II", ("TTFT share and change per output token", "TTFT 的占比和每个输出 token 的变化"), plate_anatomy(),
          ("Concurrency 1, mean of two runs. In Test 1 the monokernel helps all requests but not the slowest 10%. In Test 2 it helps both.",
           "c1，两次运行的平均。测试 1 中，monokernel 让全部请求变快，但最慢的 10% 没有变快。测试 2 中，两者都变快。")),
    callout("We judge by the P90 metric, not the median, because InferenceX publishes the P90. In Test 1 the median improved, but the P90 did not.",
            "我们按 P90 指标判断，不看中位数，因为 InferenceX 发布的是 P90。测试 1 中，中位数变好了，但 P90 没有。"),
]))

b0 = {c: eff[C1].get((c, "b0-r1")) for c in (1, 2)}
b1 = {c: eff[C1].get((c, "b1-r1")) for c in (1, 2)}
diag_rows = [[f"c{c}", sp(*name), f"{sg(d['time_to_first_token'])}%", f"{sg(d['inter_token_latency'])}%", f"{sg(d['request_latency'])}%"]
             for c in (1, 2) for name, d in ((("b0: BF16 KV, no monokernel", "b0：BF16 KV，不开 monokernel"), b0[c]),
                                             (("b1: BF16 KV, monokernel 05e0bb02", "b1：BF16 KV，monokernel 05e0bb02"), b1[c])) if d]
s_attempt = section("attempt", "04", ("Test 1: BF16 KV cache", "测试 1：BF16 KV cache"), "".join([
    P(f"{pr(7)} added the glm53 monokernel. The monokernel runs the verify step of GLM-5.3 when one or two requests are running. In #7, the monokernel accepted only a BF16 KV cache. The SGLang cookbook cells for GLM-5.3 on MI355X also use BF16 KV. #7 listed FP8 KV as out of scope.",
      f"{pr(7)} 加入了 glm53 monokernel。同时运行 1 到 2 个请求时，monokernel 负责 GLM-5.3 的 verify 步。在 #7 中，monokernel 只接受 BF16 KV cache。SGLang cookbook 里 GLM-5.3 在 MI355X 上的配置也用 BF16 KV。#7 把 FP8 KV 列为不在范围内。"),
    P("The InferenceX recipe uses an FP8 E4M3 KV cache. Thus Test 1 changed the recipe to BF16 KV and added the monokernel.",
      "InferenceX 的配方使用 FP8 E4M3 KV cache。所以测试 1 把配方改成 BF16 KV，再开启 monokernel。"),
    P(f"Test 1 used 900-s windows. On the same requests, the monokernel changed ITL by {rng(pv[C1]['inter_token_latency'])} and E2EL by {rng(pv[C1]['request_latency'])}. But TTFT changed by {rng(pv[C1]['time_to_first_token'])}.",
      f"测试 1 使用 900 秒窗口。在同一批请求上，monokernel 让 ITL 变化 {zrng(pv[C1]['inter_token_latency'])}，E2EL 变化 {zrng(pv[C1]['request_latency'])}。但 TTFT 变化了 {zrng(pv[C1]['time_to_first_token'])}。"),
    P(f"For the slowest 10% of requests, E2EL per output token changed by {rng(tv[C1], 1)}. Interactivity did not improve. The decision rule gave no-go.",
      f"对最慢的 10% 请求，每个输出 token 的 E2EL 变化 {zrng(tv[C1], 1)}。交互性没有改善。判定规则给出 no-go。"),
    P("Two extra groups isolated the cause:", "另外两组用来找出原因："),
    UL([("b0 uses BF16 KV without the monokernel. TTFT also increased. Thus BF16 KV causes the TTFT increase.",
         "b0 使用 BF16 KV，不开 monokernel。TTFT 同样变长。所以 TTFT 变长是 BF16 KV 造成的。"),
        (f"b1 uses BF16 KV with the monokernel version in the image (05e0bb02). The gain was small. That version does not include {pr(24)} and {pr(42)}.",
         f"b1 使用 BF16 KV 和镜像自带的 monokernel 版本（05e0bb02）。收益很小。这个版本不包含 {pr(24)} 和 {pr(42)}。")]),
    table([("Concurrency", "并发"), ("Group", "组"), ("TTFT", "TTFT"), ("ITL", "ITL"), ("E2EL", "E2EL")], diag_rows),
]))

prof_rows = [
    [sp("TTFT, median of 8", "TTFT，8 次的中位数"), f"{pb['ttft_median_ms']:.0f} ms", f"{pz['ttft_median_ms']:.0f} ms", f"{sg(ttft_up, 0)}%"],
    [sp(f"All GPU kernels ({pb['kernel_launches']} launches)", f"全部 GPU kernel（{pb['kernel_launches']} 次 launch）"), f"{pb['kernel_total_ms']:.1f} ms", f"{pz['kernel_total_ms']:.1f} ms", f"{sg(pz['kernel_total_ms'] - pb['kernel_total_ms'])} ms"],
    [sp("TileLang <code>main_kernel</code>, 2 per layer", "TileLang <code>main_kernel</code>，每层 2 次"), f"{pb['tilelang_main_kernel_ms']:.1f} ms", f"{pz['tilelang_main_kernel_ms']:.1f} ms", f"{sg(pz['tilelang_main_kernel_ms'] - pb['tilelang_main_kernel_ms'])} ms"],
    [sp("Its grid-2080 launch, per layer", "其中 grid 2080 的 launch，每层"), f"{pb['grid2080_per_layer_us_median']:.0f} µs", f"{pz['grid2080_per_layer_us_median']:.0f} µs", f"×{k_ratio:.2f}"],
    [sp("Its grid-4160 launch, per layer", "其中 grid 4160 的 launch，每层"), f"{pb['grid4160_per_layer_us_median']:.1f} µs", f"{pz['grid4160_per_layer_us_median']:.1f} µs", sp("no change", "不变")],
]
s_cause = section("cause", "05", ("Root cause", "根因"), "".join([
    P("We profiled one prefill that is similar to an AgentX request. It has about 100k cached tokens and 2048 new tokens. We used TP8 and the GPU trace of rank 0. The base group uses FP8 E4M3 KV. The b0 group uses BF16 KV. Neither group uses the monokernel.",
      "我们 profile 了一次和 AgentX 请求相似的 prefill。它有约 10 万个已缓存 token 和 2048 个新 token。我们用 TP8，取 rank 0 的 GPU trace。base 组使用 FP8 E4M3 KV，b0 组使用 BF16 KV。两组都不开 monokernel。"),
    table([("", ""), ("FP8 E4M3 KV", "FP8 E4M3 KV"), ("BF16 KV", "BF16 KV"), ("Change", "变化")], prof_rows),
    P(f"One kernel causes the increase. The other kernels change by {other_delta:.1f} ms in total.",
      f"变慢来自一个 kernel。其他 kernel 合计只变化 {other_delta:.1f} ms。"),
    P(f"The grid of this kernel (2080) matches the DSA sparse attention of prefill. We did not confirm this in the source code. For each new token, this kernel reads 2048 selected KV rows. A KV row has 576 bytes in FP8 and 1152 bytes in BF16. If the reads limit the kernel, BF16 doubles its time. The trace shows ×{k_ratio:.2f}.",
      f"这个 kernel 的 grid（2080）和 prefill 的 DSA 稀疏 attention 相符。我们没有在源码中确认这一点。对每个新 token，这个 kernel 读取 2048 行选中的 KV。一行 KV 在 FP8 下是 576 字节，在 BF16 下是 1152 字节。如果读取是瓶颈，BF16 会让它的时间翻倍。trace 显示的是 ×{k_ratio:.2f}。"),
    plate("III", ("Prefill time with FP8 and BF16 KV", "FP8 和 BF16 KV 下的 prefill 时间"), plate_rootcause(),
          ("With BF16 KV, the TileLang kernel takes about 2× longer. The other kernels do not change.",
           "使用 BF16 KV 时，TileLang kernel 的时间约为 2 倍。其他 kernel 不变。")),
    callout("We did not keep BF16 KV and optimize this kernel, for three reasons. The kernel is part of the SGLang DSA backend, not the monokernel. The recipe already uses FP8 KV. FP8 also halves the bytes of all other DSA reads.",
            "我们没有保留 BF16 KV 去优化这个 kernel，原因有三点。这个 kernel 属于 SGLang 的 DSA 后端，不属于 monokernel。配方本来就使用 FP8 KV。FP8 还让其他所有 DSA 读取的字节数减半。"),
]))

e2e_rows = []
for rid, en, zh_ in (("e2e-glm53-fp8kv-verify-check", "FP8 verify check (new test)", "FP8 verify check（新测试）"),
                     ("e2e-glm53-fp8kv-decode-check-local", "FP8 decode check", "FP8 decode check"),
                     ("e2e-glm53-bf16-verify-check", "BF16 verify check (regression)", "BF16 verify check（回归）"),
                     ("e2e-glm53-fp8kv-neg-verify-check", "Negative control: latent bytes swapped", "负对照：交换 latent 的字节")):
    v = e2e[rid]
    kv, stv = e2ec(rid), e2ec(rid, "layer.streams.cosine")
    state = sp("fails, as expected", "按预期失败") if v["tests_failed"] else sp("passes", "通过")
    e2e_rows.append([sp(en, zh_), state, str(kv["checks"]), f"{kv['min']:.5f}", "-" if not stv else f"{stv['min']:.5f}", str(v["violation_lines"])])
s_fix = section("fix", "06", ("Fix: use the FP8 KV cache", "修复：使用 FP8 KV cache"), "".join([
    P("The monokernel already had an FP8 path. It uses the ATOM layout, which stores FP8 KV as E4M3 values without block scales. The SGLang DSA backend on HIP uses the same format.",
      "monokernel 本来就有 FP8 路径。它使用 ATOM 布局，用不带分块 scale 的 E4M3 值存储 FP8 KV。SGLang 在 HIP 上的 DSA 后端使用同样的格式。"),
    P("Each row has 576 bytes: 512 latent values and 64 rope values. SGLang stores the rows as <code>uint8</code> and reads them as <code>float8_e4m3fn</code>. Both SGLang and the monokernel round a value to BF16 first and then convert it to E4M3.",
      "每行 576 字节：512 个 latent 值和 64 个 rope 值。SGLang 用 <code>uint8</code> 存储这些行，用 <code>float8_e4m3fn</code> 读取。SGLang 和 monokernel 都先把值舍入到 BF16，再转成 E4M3。"),
    P(f"Thus {pr(60)} changed only the integration code:", f"所以 {pr(60)} 只改了集成代码："),
    UL([("The spec accepts <code>fp8_e4m3</code>.", "spec 接受 <code>fp8_e4m3</code>。"),
        ("The build selects the FP8 version of the monokernel.", "构建时选择 FP8 版本的 monokernel。"),
        ("The runtime checks the dtype and the 576-byte row stride. Then it binds the SGLang KV rows.", "runtime 检查 dtype 和 576 字节的行步长，然后绑定 SGLang 的 KV 行。")]),
    P(f"#60 merged as {a(f'{GH}/sgl-project/sgl-amd-mono/commit/0a107f15362714c45be34ca37e4dec286fa3d303', '<code>0a107f1</code>')} on 2026-10-26. The kernel code did not change.",
      f"#60 于 2026-10-26 合并，提交是 {a(f'{GH}/sgl-project/sgl-amd-mono/commit/0a107f15362714c45be34ca37e4dec286fa3d303', '<code>0a107f1</code>')}。kernel 代码没有改动。"),
    plate("IV", ("One KV cache, two users", "同一个 KV cache，两个使用方"), plate_datapath(),
          ("SGLang and the monokernel read and write the same rows in the same format. Thus each step can use either path.",
           "SGLang 和 monokernel 用同样的格式读写同样的行。所以每一步都可以走任意一条路径。")),
    P("The e2e checks run SGLang and the monokernel on the same steps. They compare the KV output and the stream output of each layer by cosine similarity.",
      "e2e check 让 SGLang 和 monokernel 跑同样的步。它们逐层比较 KV 输出和 stream 输出的 cosine 相似度。"),
    P("A negative control shows that the checks can fail. We swap the E4M3 byte pairs of the latent. Then the minimum KV cosine is 0.095, and the check reports 1189 violations.",
      "负对照证明这些检查能够失败。我们交换 latent 中两两相邻的 E4M3 字节。这时 KV cosine 的最小值是 0.095，检查报告 1189 次违例。"),
    table([("Check", "检查"), ("Result", "结果"), ("Checks", "次数"), ("Min KV cosine", "KV cosine 最小值"), ("Min stream cosine", "stream cosine 最小值"), ("Violations", "违例")], e2e_rows),
    callout("The monokernel does not quantize the cache with its own block scales. Reason: SGLang owns the cache. The monokernel must use the same rows as the SGLang prefill and decode. Otherwise the two paths give different results.",
            "monokernel 不用自己的分块 scale 量化 cache。原因：cache 归 SGLang 管理。monokernel 必须和 SGLang 的 prefill、decode 使用同样的行。否则两条路径的结果会不同。"),
]))

s_score = section("score", "07", ("Tests 2 and 3: FP8 KV cache", "测试 2 和 3：FP8 KV cache"), "".join([
    P(f"Test 2 repeated the 900-s tests with #60. On the same requests, ITL changed by {rng(pv[C2]['inter_token_latency'])} and E2EL by {rng(pv[C2]['request_latency'])}. TTFT changed by {rng(pv[C2]['time_to_first_token'])}. For the slowest 10% of requests, E2EL per output token changed by {rng(tv[C2], 1)}.",
      f"测试 2 用 #60 重做 900 秒测试。在同一批请求上，ITL 变化 {zrng(pv[C2]['inter_token_latency'])}，E2EL 变化 {zrng(pv[C2]['request_latency'])}。TTFT 变化 {zrng(pv[C2]['time_to_first_token'])}。对最慢的 10% 请求，每个输出 token 的 E2EL 变化 {zrng(tv[C2], 1)}。"),
    P(f"But in a 900-s window, the A/A noise of interactivity was {100 * noise2['e2e_int_p90']:.0f}% ({100 * noise1['e2e_int_p90']:.0f}% in Test 1). The second run at concurrency 2 was inside this noise. Thus the decision rule gave no-go again. The 900-s window was too short to measure a P90 reliably.",
      f"但在 900 秒窗口中，交互性的 A/A 噪声是 {100 * noise2['e2e_int_p90']:.0f}%（测试 1 中是 {100 * noise1['e2e_int_p90']:.0f}%）。c2 的第二次运行落在噪声内。所以判定规则又给出 no-go。900 秒窗口太短，不能可靠地测量 P90。"),
    P(f"Test 3 used the 3600-s window of InferenceX at concurrency 1, 2, 4, and 10. Each point used a new server. An A/A pair at concurrency 2 measured the noise: {100 * noise3['e2e_int_p90']:.1f}% for interactivity.",
      f"测试 3 使用 InferenceX 的 3600 秒窗口，并发为 1、2、4、10。每个点都用新的服务器。c2 的一组 A/A 测量噪声：交互性的噪声是 {100 * noise3['e2e_int_p90']:.1f}%。"),
    P("All nine runs were valid:", "九次运行全部有效："),
    UL([("Each run completed the full 3600-s window.", "每次运行都跑满 3600 秒。"),
        ("Failed and cancelled requests were far below 10%.", "失败和取消的请求远低于 10%。"),
        ("The InferenceX validator passed.", "InferenceX 的 validator 通过。"),
        ("The measured accept length was 2.990. The InferenceX target is 2.99.", "实测的接受长度是 2.990。InferenceX 的目标值是 2.99。")]),
    plate("V", ("Throughput and interactivity at 3600 s", "3600 秒的吞吐和交互性"), plate_score(),
          ("Each arrow goes from the base group to the monokernel group at one concurrency. Hollow points are published InferenceX results.",
           "每个箭头从某个并发点的 base 组指向 monokernel 组。空心点是 InferenceX 已发布的结果。")),
    P(f"InferenceX Check 14 recommends at least five Pareto-frontier points for each curve (throughput against P90 E2EL). Our four points give {par['baseline']['frontierPoints']} frontier points without the monokernel and {par['candidate']['frontierPoints']} with it.",
      f"InferenceX 的 Check 14 建议每条曲线至少有 5 个 Pareto 前沿点（吞吐对 P90 E2EL）。我们的 4 个点里，不开 monokernel 有 {par['baseline']['frontierPoints']} 个前沿点，开了有 {par['candidate']['frontierPoints']} 个。"),
    P("The reason: AgentX P90 E2EL does not increase with concurrency. At concurrency 4, P90 E2EL is lower than at 1 and 2, and throughput is higher. Thus concurrency 4 dominates concurrency 1 and 2.",
      "原因：AgentX 的 P90 E2EL 不随并发单调增加。c4 的 P90 E2EL 比 c1、c2 低，吞吐又更高。所以 c4 支配 c1 和 c2。"),
]))

gain_rows = [[f"c{c}", str(eff[C3][(c, 'cand8-r1')]["requests"]), f"{sg(ttft3[c])}%", f"{sg(itl3[c])}%", f"{sg(e2el3[c])}%",
              f"{sg(tail[C3][(c, 'cand8-r1', 'slowest 10%')]['e2el_per_token'])}%", f"{100 * share[c]:.0f}%"] for c in CONCS]
s_gain = section("gain", "08", ("Source of the gain", "收益来源"), "".join([
    P("We compared the same requests across runs. A pair has the same conversation and turn, and the input lengths differ by 8 tokens or less. For each metric, we took the median log change against the base group.",
      "我们在不同运行之间比较同一批请求。一对请求属于同一个会话和同一轮，输入长度相差不超过 8 个 token。对每个指标，我们取相对 base 组的对数变化中位数。"),
    P("Coverage is the share of decode steps with one or two running requests. We read it from the server log. The monokernel runs only these steps.",
      "覆盖率是同时运行 1 到 2 个请求的 decode 步所占的比例。我们从服务器日志中读取。monokernel 只运行这些步。"),
    table([("Concurrency", "并发"), ("Pairs", "配对数"), ("TTFT", "TTFT"), ("ITL", "ITL"), ("E2EL", "E2EL"), ("Slowest 10%, E2EL per token", "最慢 10%，每 token E2EL"), ("Coverage", "覆盖率")], gain_rows),
    plate("VI", ("Coverage and gain", "覆盖率和收益"), plate_gain(),
          ("Left: decode steps by the number of running requests. Right: change of ITL and E2EL on the same requests.",
           "左：按同时运行的请求数划分 decode 步。右：同一批请求上 ITL 和 E2EL 的变化。")),
    UL([
        (f"At concurrency 1, {100 * hist_share[1].get('1', 0):.0f}% of decode steps have one request. At concurrency 4, {100 * hist_share[4].get('1', 0):.0f}% have one request and {100 * hist_share[4].get('2', 0):.0f}% have two.",
         f"c1 时，{100 * hist_share[1].get('1', 0):.0f}% 的 decode 步只有 1 个请求。c4 时，{100 * hist_share[4].get('1', 0):.0f}% 有 1 个请求，{100 * hist_share[4].get('2', 0):.0f}% 有 2 个请求。"),
        (f"Coverage is above 95% at both points. But ITL decreases by {abs(itl3[1]):.1f}% at concurrency 1 and by only {abs(itl3[4]):.1f}% at concurrency 4.",
         f"两个点的覆盖率都高于 95%。但 ITL 在 c1 降低 {abs(itl3[1]):.1f}%，在 c4 只降低 {abs(itl3[4]):.1f}%。"),
        ("A possible reason: a step with two requests gains less than a step with one request. We did not measure this. The reason is not verified.",
         "一个可能的原因：2 个请求的步比 1 个请求的步收益小。我们没有测量这一点。这个原因未经验证。"),
        (f"At concurrency 10, coverage is {100 * share[10]:.0f}%, so the gain is smaller.", f"c10 的覆盖率是 {100 * share[10]:.0f}%，所以收益更小。"),
        ("TTFT changes by less than 1.5% at each point. The FP8 KV cache keeps the prefill time unchanged.", "每个点的 TTFT 变化都小于 1.5%。FP8 KV cache 让 prefill 时间保持不变。"),
    ]),
]))

q_rows = [[sp("Test 1: FP8 KV vs BF16 KV + monokernel", "测试 1：FP8 KV 对 BF16 KV + monokernel"), "c2", str(gsm1["filters"]["strict-match"]["n"]),
           f"{gsm1['filters']['strict-match']['base']:.3f}", f"{gsm1['filters']['strict-match']['cand']:.3f}",
           f"{gsm1['filters']['strict-match']['only_base_right']} / {gsm1['filters']['strict-match']['only_cand_right']}", f"{gsm1['filters']['strict-match']['mcnemar_p']:.2f}"],
          [sp("Test 2: base vs monokernel", "测试 2：base 对 monokernel"), "c2", str(gsm2["filters"]["strict-match"]["n"]),
           f"{gsm2['filters']['strict-match']['base']:.3f}", f"{gsm2['filters']['strict-match']['cand']:.3f}",
           f"{gsm2['filters']['strict-match']['only_base_right']} / {gsm2['filters']['strict-match']['only_cand_right']}", f"{gsm2['filters']['strict-match']['mcnemar_p']:.2f}"]]
for r in strict3:
    lab = {"base vs cand8": ("Test 3: base vs monokernel", "测试 3：base 对 monokernel"), "rerun of base": ("Test 3: base, two runs", "测试 3：base 的两次运行"),
           "rerun of cand8": ("Test 3: monokernel, two runs", "测试 3：monokernel 的两次运行")}[r["kind"]]
    q_rows.append([sp(f"{lab[0]} ({r['a']} / {r['b']})", f"{lab[1]}（{r['a']} / {r['b']}）"), r["a"].split()[0], str(r["n"]),
                   f"{r['acc_a']:.3f}", f"{r['acc_b']:.3f}", f"{r['only_a']} / {r['only_b']}", f"{r['mcnemar_p']:.2f}"])
s_quality = section("accuracy", "09", ("Accuracy", "精度"), "".join([
    P("We measured GSM8K with real MTP acceptance and the strict-match filter. We used the InferenceX eval (lm-evaluation-harness). Tests 1 and 2 used 500 questions. Test 3 used all 1319 questions. At concurrency 10, we ran each group two times to measure the run-to-run variation.",
      "我们用真实的 MTP 接受和 strict-match 过滤测量 GSM8K。评测用的是 InferenceX 的 eval（lm-evaluation-harness）。测试 1 和 2 用 500 题。测试 3 用全部 1319 题。在 c10，每组跑两次，用来测量两次运行之间的波动。"),
    table([("Pair", "对比"), ("Concurrency", "并发"), ("Questions", "题数"), ("A", "A"), ("B", "B"), ("Only A right / only B right", "只有 A 对 / 只有 B 对"), ("McNemar p", "McNemar p")], q_rows),
    UL([
        (f"All three base-vs-monokernel pairs together: {ob} to {oc} (p = {mcnemar_p(ob, oc):.2f}).", f"三组 base 对 monokernel 合计：{ob} 对 {oc}（p = {mcnemar_p(ob, oc):.2f}）。"),
        (f"The first pair alone gives {pairs3[0]['only_a']} to {pairs3[0]['only_b']}. But two runs of the monokernel group give {rerun_c['only_a']} to {rerun_c['only_b']}. Thus the difference in the first pair is run-to-run variation.",
         f"只看第一组是 {pairs3[0]['only_a']} 对 {pairs3[0]['only_b']}。但 monokernel 组的两次运行就差 {rerun_c['only_a']} 对 {rerun_c['only_b']}。所以第一组的差别是两次运行之间的波动。"),
        (f"In the first pair, the monokernel group answered {len(lost)} questions wrong. In {len(runaway)} of them, the model did not stop reasoning and gave no answer ({runaway[0] // 1000}k and {runaway[-1] // 1000}k characters). The other {len(lost) - len(runaway)} are normal solution errors. None shows corrupted numbers.",
         f"第一组中，monokernel 组答错 {len(lost)} 题。其中 {len(runaway)} 题，模型一直推理、没有给出答案（{runaway[0] // 1000}k 和 {runaway[-1] // 1000}k 字符）。其余 {len(lost) - len(runaway)} 题是普通的解题错误。没有一题显示数值被破坏。"),
        (f"Coverage in the GSM8K runs: {100 * gcov['gsm8k-glm53-l2ov-c2-cand8q']:.0f}% at concurrency 2, and {100 * gcov['gsm8k-glm53-l2ov-c10-cand8q']:.0f}% and {100 * gcov['gsm8k-glm53-l2ov-c10-cand8q-r2']:.0f}% at concurrency 10.",
         f"GSM8K 运行的覆盖率：c2 是 {100 * gcov['gsm8k-glm53-l2ov-c2-cand8q']:.0f}%，c10 是 {100 * gcov['gsm8k-glm53-l2ov-c10-cand8q']:.0f}% 和 {100 * gcov['gsm8k-glm53-l2ov-c10-cand8q-r2']:.0f}%。"),
        ("InferenceX runs its eval at the highest concurrency of a configuration. There the monokernel runs only about a quarter of the steps. Thus the concurrency-2 result is the accuracy evidence for the monokernel path.",
         "InferenceX 在一个配置的最高并发点跑 eval。在那里，monokernel 只运行约四分之一的步。所以 c2 的结果才是 monokernel 路径的精度证据。"),
    ]),
]))

ref_rows = [[f"c{r['conc']}", E(r["curve"]), f"{r['tput']:.0f}", f"{r['int90']:.1f}", f"{r['e2el90']:.1f}", a(r["url"], E(r["source"])) if r.get("url") else E(r["source"])]
            for r in refs3 if "this work" not in r["curve"]]
osl_rows = []
for c in (1, 4, 10):
    o = osl_local[c]
    osl_rows.append([f"c{c}", sp("monokernel group (3600 s)", "monokernel 组（3600 秒）"), f"{o['p50']:.0f}", f"{o['p90']:.0f}"])
    if c in osl_b300:
        r = osl_b300[c]
        osl_rows.append([f"c{c}", sp("GLM-5.3 B300 SGLang FP4 (published)", "GLM-5.3 B300 SGLang FP4（已发布）"), f"{r['median_output_tokens_actual']:.0f}", f"{r['p90_output_tokens_actual']:.0f}"])
s_refs = section("refs", "10", ("Comparison with published results", "和已发布结果比较"), "".join([
    P("The InferenceX API has no vLLM AgentX result for GLM-5.x models.", "InferenceX 的 API 里没有 GLM-5.x 模型的 vLLM AgentX 结果。"),
    P(f"For GLM-5.3 on MI355X, InferenceX published one point: TileRT at concurrency 1 on 16 GPUs. At concurrency 1, the monokernel group has {sg(rel(K(1, 'tput_gpu'), tile['tput']), 0)}% throughput per GPU and {sg(rel(K(1, 'e2e_int_p90'), tile['int90']), 0)}% interactivity against it.",
      f"对 MI355X 上的 GLM-5.3，InferenceX 只发布了一个点：TileRT，c1，16 张 GPU。在 c1，monokernel 组相对它的每卡吞吐是 {sg(rel(K(1, 'tput_gpu'), tile['tput']), 0)}%，交互性是 {sg(rel(K(1, 'e2e_int_p90'), tile['int90']), 0)}%。"),
    P(f"GLM-5.3 on B300 with NVFP4 is better. At concurrency 10, our throughput is {abs(rel(K(10, 'tput_gpu'), b300[10]['tput'])):.0f}% lower, and our interactivity is {abs(rel(K(10, 'e2e_int_p90'), b300[10]['int90'])):.0f}% lower.",
      f"B300 上的 GLM-5.3 NVFP4 更好。在 c10，我们的吞吐低 {abs(rel(K(10, 'tput_gpu'), b300[10]['tput'])):.0f}%，交互性低 {abs(rel(K(10, 'e2e_int_p90'), b300[10]['int90'])):.0f}%。"),
    table([("Concurrency", "并发"), ("Published configuration", "已发布配置"), ("Throughput", "吞吐"), ("Interactivity", "交互性"), ("P90 E2EL (s)", "P90 E2EL（秒）"), ("Source", "来源")], ref_rows),
    P(f"Do not compare E2EL directly between these configurations, because the reply lengths are different. At the same concurrency, our P90 output length is {min(osl_ratio):.1f} to {max(osl_ratio):.1f} times the B300 value. The median lengths are similar.",
      f"不要直接比较这些配置的 E2EL，因为回复长度不同。在同样的并发下，我们的 P90 输出长度是 B300 的 {min(osl_ratio):.1f} 到 {max(osl_ratio):.1f} 倍。中位数长度相近。"),
    P("An older image gives the same lengths. Thus the image and the chat template are not the cause. The weights are a possible cause (FP8 here, NVFP4 on B300). We did not verify this. Interactivity divides by the output length. Thus it is the best metric for this comparison.",
      "旧镜像上的长度也一样。所以原因不是镜像，也不是 chat template。权重可能是原因（这里是 FP8，B300 上是 NVFP4）。我们没有验证这一点。交互性除以了输出长度，所以它最适合这种比较。"),
    table([("Concurrency", "并发"), ("Run", "运行"), ("Median OSL", "OSL 中位数"), ("P90 OSL", "OSL P90")], osl_rows),
]))

s_up = section("upstream", "11", ("Steps to an InferenceX submission", "提交 InferenceX 还需要的步骤"), "".join([
    P("The kernel work is complete. These steps remain, in this order:", "kernel 的工作已经完成。还剩这些步骤，按顺序进行："),
    plate("VII", ("Remaining steps", "剩下的步骤"), plate_upstream(),
          ("State on 2026-10-26. Each step needs the step before it. Until the pin update, the cookbook command fails at server start.",
           "状态截至 2026-10-26。每一步都依赖前一步。pin 更新之前，cookbook 里的命令会在服务器启动时失败。")),
    OL([
        (f"SGLang pin update. SGLang main pins sgl-amd-mono at <code>05e0bb02</code>. Its glm53 spec accepts only BF16 KV. The open PR {pr(45045, 'sgl-project/sglang')} pins <code>e952b221</code>, which is also older than #60.",
         f"SGLang 更新 pin。SGLang main 把 sgl-amd-mono pin 在 <code>05e0bb02</code>。它的 glm53 spec 只接受 BF16 KV。现有的 PR {pr(45045, 'sgl-project/sglang')} pin 的是 <code>e952b221</code>，也比 #60 早。"),
        ("Image. Use the first MI35x nightly image that includes the new pin. Record its digest.", "镜像。使用第一个包含新 pin 的 MI35x nightly 镜像，并记录它的 digest。"),
        (f"Cookbook. {pr(46662, 'sgl-project/sglang')} (draft) adds the recipe as section 3.6 of the GLM-5.3 page. InferenceX requires all production-relevant flags on a merged cookbook page.",
         f"cookbook。{pr(46662, 'sgl-project/sglang')}（draft）把配方加为 GLM-5.3 页面的第 3.6 节。InferenceX 要求所有影响生产效果的参数都写在已合并的 cookbook 页面上。"),
        ("Confirmation run. Run the 3600-s tests again on that image, without the overlay. Add points, for example concurrency 6 and 8, to get five frontier points.",
         "确认运行。在那个镜像上、不加 overlay，重跑 3600 秒测试。再加几个点，例如 c6 和 c8，凑够 5 个前沿点。"),
        ("InferenceX PR. Add the config, the recipe, one perf-changelog entry, and a bilingual description. The account jhinpan has read access only. Thus the PR must come from a fork or from a collaborator.",
         "InferenceX PR。加入配置、配方、一条 perf-changelog 和中英双语说明。jhinpan 账号只有读权限，所以 PR 要从 fork 提交，或者由有权限的同事提交。"),
        ("Sweep, evals, CODEOWNER sign-off, and an authorized <code>/use</code> of the sweep run.", "sweep、eval、CODEOWNER 签核，以及有权限的人对 sweep run 发出 <code>/use</code>。"),
    ]),
]))

s_method = section("method", "12", ("Method and limits", "方法和限制"), UL([
    ("We used one local node, not the InferenceX cluster.", "我们用的是一台本地节点，不是 InferenceX 的集群。"),
    ("We installed the monokernel commit over the published image. Thus these runs do not replace the confirmation run on a published image.",
     "我们把 monokernel 的提交装在已发布的镜像上。所以这些运行不能代替在已发布镜像上的确认运行。"),
    ("We ran each point one time for 3600 s. We measured the A/A noise only at concurrency 2. We assume the same noise at the other points.",
     "每个点只跑一次 3600 秒。A/A 噪声只在 c2 测量。我们假设其他点的噪声相同。"),
    ("The AgentX runs use the synthetic acceptance of InferenceX (accept length 2.99) in both groups. The GSM8K runs use real acceptance.",
     "AgentX 运行在两组中都使用 InferenceX 的模拟接受（接受长度 2.99）。GSM8K 运行使用真实接受。"),
    ("Before each run, we checked that all 8 GPUs were idle (30 samples) and locked them. During each run, we sampled amd-smi every 10 s and checked for other processes.",
     "每次运行前，我们检查 8 张 GPU 都空闲（采样 30 次），然后锁定它们。运行时每 10 秒用 amd-smi 采样一次，并检查有没有其他进程。"),
    ("Each point used a new server. No request was sent before the measured phase.", "每个点都用新的服务器。测量阶段之前不发送任何请求。"),
    ("We wrote the decision rule of each test before the test. We changed one rule one time, before any result. We replaced a span check that does not apply to AgentX with a phase-based validity check.",
     "每次测试的判定规则都在测试前写好。我们只改过一次规则，而且是在出结果之前。我们把不适用于 AgentX 的跨度检查，换成了按阶段的有效性检查。"),
    ("The published results are for context only. The hardware, precision, deployment, and output lengths are different.", "已发布的结果只作背景。硬件、精度、部署方式和输出长度都不同。"),
]))

files = [
    ("cohort3-fp8kv-3600s/analysis/compare_round1_base_cand8.json", "headline metrics, gains, and A/A noise", "头条指标、增益和 A/A 噪声"),
    ("cohort3-fp8kv-3600s/analysis/pair_effects.json", "changes on the same requests", "同一批请求的变化"),
    ("cohort3-fp8kv-3600s/requests", "one CSV row per request for each 3600-s run", "每个 3600 秒运行的逐请求 CSV"),
    ("cohort3-fp8kv-3600s/gsm8k", "GSM8K result for each question", "GSM8K 每道题的结果"),
    ("ttft-profile/summary.json", "the prefill kernel profile", "prefill 的 kernel profile"),
    ("fp8kv-validation/e2e_checks.json", "the e2e checks of #60", "#60 的 e2e 检查"),
    ("inferencex-config", "the local InferenceX config and recipe", "本地的 InferenceX 配置和配方"),
    ("scripts/make_page.py", "the script that builds this page", "生成本页的脚本"),
]
link = lambda p: a(f"{BLOB}/{p}" if "." in p.split("/")[-1] else f"{DATA_URL}/{p}", f"<code>{p}</code>")  # noqa: E731
s_data = section("data", "13", ("Data and files", "数据和文件"), "".join([
    P(f"The raw data, the decision rules, and the analysis results are in {a(DATA_URL, f'<code>data/{SLUG}/</code>')}. We replaced host names and local paths. We reduced the request records, the GSM8K outputs, and the profiler traces to compact CSV and JSON files.",
      f"原始数据、判定规则和分析结果都在 {a(DATA_URL, f'<code>data/{SLUG}/</code>')}。我们替换了主机名和本机路径。逐请求记录、GSM8K 输出和 profiler trace 都缩减成了紧凑的 CSV 和 JSON 文件。"),
    UL([(f"{link(p)}: {en}", f"{link(p)}：{z}") for p, en, z in files]),
]))

s_lessons = section("lessons", "14", ("Lessons", "经验"), UL([
    ("A kernel can make its own step faster and still not improve the score. Then examine the conditions that the kernel requires. Here the BF16 KV condition made prefill slower, and the P90 metric depends mostly on TTFT.",
     "kernel 可以让自己负责的那一步变快，但分数仍然不提高。这时要检查 kernel 要求的条件。这里，BF16 KV 这个条件让 prefill 变慢，而 P90 指标主要取决于 TTFT。"),
    ("The fix did not change the kernel. It let the kernel use the KV format that the recipe already uses.",
     "修复没有改 kernel，而是让 kernel 使用配方本来就用的 KV 格式。"),
    (f"Measure a P90 metric with the window that the score uses. In 900-s windows, two identical runs differed by {100 * noise2['e2e_int_p90']:.0f}%. In 3600-s windows, they differed by {100 * noise3['e2e_int_p90']:.1f}%.",
     f"测量 P90 指标时，要用分数所用的窗口。900 秒窗口中，两次相同的运行相差 {100 * noise2['e2e_int_p90']:.0f}%。3600 秒窗口中，只相差 {100 * noise3['e2e_int_p90']:.1f}%。"),
]))

SECTIONS = [("result", "01", "Result", "结果"), ("steps", "02", "Test sequence", "测试过程"), ("metric", "03", "How the score works", "打分方式"),
            ("attempt", "04", "Test 1: BF16 KV", "测试 1：BF16 KV"), ("cause", "05", "Root cause", "根因"), ("fix", "06", "Fix: FP8 KV", "修复：FP8 KV"),
            ("score", "07", "Tests 2 and 3", "测试 2 和 3"), ("gain", "08", "Source of the gain", "收益来源"), ("accuracy", "09", "Accuracy", "精度"),
            ("refs", "10", "Published results", "已发布结果"), ("upstream", "11", "Steps to InferenceX", "提交步骤"),
            ("method", "12", "Method and limits", "方法和限制"), ("data", "13", "Data and files", "数据和文件"), ("lessons", "14", "Lessons", "经验")]
rail = ('<nav class="rail" aria-label="Contents"><a class="back" href="/sources/">' + sp("← The Library", "← 文库") + "</a><ol>"
        + "".join(f'<li><a href="#{sid}"><span class="rn">{num}</span>{sp(en, z)}</a></li>' for sid, num, en, z in SECTIONS) + "</ol></nav>")

summary = [
    (f"The glm53 decode monokernel reduced inter-token latency (ITL) by {itl_lo}–{itl_hi}% on MI355X. But the InferenceX AgentX score did not improve.",
     f"glm53 decode monokernel 让 MI355X 上的 inter-token latency（ITL）降低 {itl_lo}–{itl_hi}%。但 InferenceX AgentX 的分数没有提高。"),
    (f"Cause: the monokernel accepted only a BF16 KV cache. With BF16 KV, one DSA prefill kernel took {k_ratio:.1f}× longer, and TTFT increased by {ttft_up}%.",
     f"原因：monokernel 当时只接受 BF16 KV cache。使用 BF16 KV 时，DSA prefill 里的一个 kernel 耗时变为 {k_ratio:.1f} 倍，TTFT 增加 {ttft_up}%。"),
    (f"Fix: {pr(60)} lets the monokernel use the FP8 KV cache of the recipe. The kernel code did not change.",
     f"修复：{pr(60)} 让 monokernel 直接使用配方的 FP8 KV cache。kernel 代码没有改动。"),
    (f"Result at 3600 s: throughput +{gain_lo('tput_gpu')}–{gain_hi('tput_gpu')}% and interactivity +{gain_lo('e2e_int_p90')}–{gain_hi('e2e_int_p90')}% at concurrency 1–4. GSM8K accuracy did not change.",
     f"3600 秒窗口的结果：c1–c4 的吞吐提高 {gain_lo('tput_gpu')}–{gain_hi('tput_gpu')}%，交互性提高 {gain_lo('e2e_int_p90')}–{gain_hi('e2e_int_p90')}%。GSM8K 精度没有变化。"),
    ("Status: the InferenceX submission waits for an SGLang pin update, a new image, and the cookbook PR.",
     "状态：提交 InferenceX 之前，还要等 SGLang 更新 pin、发布新镜像，并合并 cookbook PR。"),
]
spec = [
    (("Hardware", "硬件"), ("8 × AMD Instinct MI355X (gfx950), one node, TP8 EP1", "8 × AMD Instinct MI355X（gfx950），单节点，TP8 EP1")),
    (("Model", "模型"), (f"{a('https://huggingface.co/zai-org/GLM-5.3', 'zai-org/GLM-5.3')} at <code>ba32f8c1</code> (FP8), EAGLE MTP 3-1-4", f"{a('https://huggingface.co/zai-org/GLM-5.3', 'zai-org/GLM-5.3')}，<code>ba32f8c1</code>（FP8），EAGLE MTP 3-1-4")),
    (("Engine", "引擎"), (f"SGLang {a(f'{GH}/sgl-project/sglang/commit/fe1bd108372c9ba2ab959fcf99cf196433d155d5', '<code>fe1bd108</code>')} in <code>lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261024</code>, AITER <code>285579b6</code>",
                          f"SGLang {a(f'{GH}/sgl-project/sglang/commit/fe1bd108372c9ba2ab959fcf99cf196433d155d5', '<code>fe1bd108</code>')}，镜像 <code>lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261024</code>，AITER <code>285579b6</code>")),
    (("Monokernel", "monokernel"), (f"sgl-amd-mono glm53 at <code>0a107f1</code> ({pr(60)}), installed over the image's <code>05e0bb02</code>", f"sgl-amd-mono glm53，<code>0a107f1</code>（{pr(60)}），装在镜像自带的 <code>05e0bb02</code> 之上")),
    (("Client", "客户端"), ("InferenceX AgentX (AIPerf), 3600-s windows, accept length 2.99", "InferenceX AgentX（AIPerf），3600 秒窗口，接受长度 2.99")),
    (("Measured", "测量时间"), ("2026-10-24 to 2026-10-26", "2026-10-24 到 2026-10-26")),
    (("Raw data", "原始数据"), (a(DATA_URL, f"<code>data/{SLUG}/</code>"), a(DATA_URL, f"<code>data/{SLUG}/</code>"))),
]
TITLE = "GLM-5.3 on MI355X: the decode monokernel needs an FP8 KV cache to raise the AgentX score"
TITLE_ZH = "GLM-5.3 在 MI355X 上的 decode monokernel 要配合 FP8 KV cache 才能提高 AgentX 分数"
masthead = ('<header class="masthead"><div class="mast-in">'
            + bi("div", "Experiment 006 · Model hill-climbing", "实验 006 · 模型 hill-climbing 记录", 'class="kicker"')
            + bi("h1", TITLE, TITLE_ZH)
            + '<ul class="summary">' + "".join(bi("li", en, z) for en, z in summary) + "</ul>"
            + '<dl class="spec">' + "".join(f"<dt>{sp(*k)}</dt><dd>{sp(*v)}</dd>" for k, v in spec) + "</dl></div></header>")

CSS = """
:root{--paper:#f3f5f2;--card:#ffffff;--sunk:#e8ede9;--rule:#d3dbd5;--rule2:#a5b5aa;--ink:#13211d;--ink2:#34463f;--ink3:#56685f;
--base:#4a6787;--mono:#c2366b;--bf16:#a77a1f;--ref:#8a958f;--good:#2e7d4f;--warn:#b5651d;--bad:#b3392f;
--serif:"Instrument Serif",Georgia,"Noto Serif SC",serif;--sans:"Public Sans","Noto Sans SC","PingFang SC","Microsoft YaHei",sans-serif;--mono-f:"Reddit Mono",ui-monospace,Menlo,Consolas,monospace}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:var(--paper);color:var(--ink);font:17px/1.7 var(--sans);-webkit-font-smoothing:antialiased}
a{color:var(--base);text-decoration:underline;text-decoration-color:var(--rule2);text-underline-offset:2px}a:hover{color:var(--mono)}
code{font:0.85em var(--mono-f);background:var(--sunk);padding:1px 5px;border-radius:4px}
.masthead{background:var(--card);border-bottom:1px solid var(--rule)}
.mast-in{max-width:1000px;margin:0 auto;padding:52px 28px 34px}
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
table.score td b{color:var(--mono)}.up{color:var(--good);font-weight:700}
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
<meta name="description" content="GLM-5.3 on MI355X: the glm53 decode monokernel reduced ITL by 28-30%, but the AgentX score improved only after the monokernel used the FP8 KV cache: +15-19% throughput at concurrency 1-4 over 3600 s.">
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
{s_result}{s_steps}{s_metric}{s_attempt}{s_cause}{s_fix}{s_score}{s_gain}{s_quality}{s_refs}{s_up}{s_method}{s_data}{s_lessons}
</main>
</div>
{colophon}
{JS}
</body>
</html>
"""
OUT.write_text(page)
print(f"wrote {OUT} ({len(page) / 1e3:.0f} KB)")
