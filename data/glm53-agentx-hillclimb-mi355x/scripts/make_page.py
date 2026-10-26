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


def callout(en, zh_text, kind="note"):
    return f'<aside class="callout {kind}">{P(en, zh_text)}</aside>'


def table(head, rows, cls=""):
    """head: list of (en, zh) or str; rows: list of lists of str or (en, zh)."""
    cell = lambda x: sp(*x) if isinstance(x, tuple) else x  # noqa: E731
    th = "".join(f"<th>{cell(h)}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{cell(x)}</td>" for x in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def plate(num, title, svg, caption, wide=True):
    return (f'<figure class="plate{" wide" if wide else ""}"><div class="plate-meta"><span class="pn">Plate {num}</span>'
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
checks3 = J(f"{C3}/analysis/run_checks.json")
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


def rng(vals, nd=0, pct=True):
    lo, hi = min(vals), max(vals)
    if nd == 0 and max(abs(lo), abs(hi)) < 5:
        nd = 1
    u = "%" if pct else ""
    a_, b_ = sorted((lo, hi), key=abs)
    return f"{sg(a_, nd)}{u} … {sg(b_, nd)}{u}" if round(lo, nd) != round(hi, nd) else f"{sg(lo, nd)}{u}"


cand_runs = {C1: ("cand-r1", "cand-r2"), C2: ("cand8-r1", "cand8-r2")}
pair_rng = {c: {m: rng([eff[c][(k, r)][m] for k in (1, 2, 4) for r in cand_runs[c]]) for m in ("time_to_first_token", "inter_token_latency", "request_latency")} for c in (C1, C2)}
tail_rng = {c: rng([tail[c][(k, r, "slowest 10%")]["e2el_per_token"] for k in (1, 2, 4) for r in cand_runs[c]], 1) for c in (C1, C2)}
mean2 = lambda c, r, sub, m: sum(tail[c][(1, x, sub)][m] for x in r) / 2  # noqa: E731

strict3 = [r for r in gsm3 if r["filter"] == "strict-match"]
pairs3 = [r for r in strict3 if r["kind"] == "base vs cand8"]
ob, oc = sum(r["only_a"] for r in pairs3), sum(r["only_b"] for r in pairs3)
rerun_c = next(r for r in strict3 if r["kind"] == "rerun of cand8")
rerun_b = next(r for r in strict3 if r["kind"] == "rerun of base")


def gsm_rows(rid):
    with (D / f"{C3}/gsm8k/{rid}.csv").open() as f:
        return {int(r["doc_id"]): r for r in csv.DictReader(f)}


gb, gc = gsm_rows("gsm8k-glm53-l2ov-c10-baseq"), gsm_rows("gsm8k-glm53-l2ov-c10-cand8q")
lost = [k for k in gb if gb[k]["strict_match"] == "1" and gc[k]["strict_match"] == "0"]
runaway = [k for k in lost if int(gc[k]["response_chars"]) > 40000]
run_chars = sorted(int(gc[k]["response_chars"]) for k in runaway)

ref = lambda curve, c: next((r for r in refs3 if r["curve"].startswith(curve) and r["conc"] == c), None)  # noqa: E731
tile, b300 = ref("GLM-5.3 MI355X TileRT", 1), {c: ref("GLM-5.3 B300", c) for c in (1, 4, 10)}
rel = lambda x, y: 100 * (x / y - 1)  # noqa: E731
osl_local = {c: run3(f"agentx-glm53-l2ov-c{c}-cand8-r1")["agentx_summary"]["request_metrics"]["tokens"]["output_actual"] for c in (1, 4, 10)}
osl_b300 = {r["conc"]: r for r in osl_pub if r["model"] == "glm5.3" and r["hardware"] == "b300"}
osl_ratio = [osl_local[c]["p90"] / osl_b300[c]["p90_output_tokens_actual"] for c in (1, 4, 10) if c in osl_b300]
e2ec = lambda rid, m="layer.kv.cosine": e2e[rid]["metrics_first_test"].get(m)  # noqa: E731
valid3 = [r for r in checks3 if str(r.get("run", r.get("id", ""))).startswith("agentx")] if isinstance(checks3, list) else []

gain_lo = lambda m, cs=(1, 2, 4): round(min(g(c, m) for c in cs))  # noqa: E731
gain_hi = lambda m, cs=(1, 2, 4): round(max(g(c, m) for c in cs))  # noqa: E731
itl3 = {c: eff[C3][(c, "cand8-r1")]["inter_token_latency"] for c in CONCS}
ttft3 = {c: eff[C3][(c, "cand8-r1")]["time_to_first_token"] for c in CONCS}
e2el3 = {c: eff[C3][(c, "cand8-r1")]["request_latency"] for c in CONCS}
itl1 = [eff[C1][(k, r)]["inter_token_latency"] for k in (1, 2, 4) for r in cand_runs[C1]]

# ----------------------------------------------------------------------------- plates
ARROW = '<defs><marker id="ah" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 z" class="ahd"/></marker></defs>'


def plate_route():
    camps = [
        (70, 222, "base", "Recipe, no monokernel", "配方，不开 monokernel", "FP8 KV · EAGLE 3-1-4", "FP8 KV · EAGLE 3-1-4"),
        (235, 248, "bf16", "Cohort 1 · 900 s", "cohort 1 · 900 秒", "BF16 KV + kernel: no-go", "BF16 KV + kernel：no-go"),
        (400, 206, "plain", "Profile", "profile", "one prefill kernel ×2", "一个 prefill kernel ×2"),
        (560, 156, "mono", "sgl-amd-mono#60", "sgl-amd-mono#60", "kernel reads FP8 KV", "kernel 读写 FP8 KV"),
        (712, 138, "plain", "Cohort 2 · 900 s", "cohort 2 · 900 秒", "ahead, inside noise", "领先，但在噪声内"),
        (858, 76, "mono", "Cohort 3 · 3600 s", "cohort 3 · 3600 秒", "ahead at every point", "每个点都领先"),
    ]
    contours = "".join(
        f'<path d="M0,{y} C220,{y - 34} 430,{y + 26} 640,{y - 18} S920,{y - 52} 1040,{y - 30}" class="contour"/>'
        for y in (300, 266, 232, 198, 164, 130, 96, 62))
    trail = " ".join(f"{x},{y}" for x, y, *_ in camps)
    out = [f'<svg viewBox="0 0 1040 310" role="img" aria-label="route">{ARROW}{contours}',
           f'<polyline points="{trail}" class="trail"/>',
           f'<path d="M{camps[-1][0]},{camps[-1][1]} C915,58 950,44 1000,34" class="trail dashed" marker-end="url(#ah)"/>',
           svg_text(972, 22, "InferenceX dashboard", "InferenceX 榜单", "lbl b", "end"),
           svg_text(972, 38, "pin bump → image → cookbook → PR", "pin 升级 → 镜像 → cookbook → PR", "lbl s", "end")]
    for i, (x, y, cls, t1e, t1z, t2e, t2z) in enumerate(camps):
        out.append(f'<circle cx="{x}" cy="{y}" r="7" class="camp {cls}"/>')
        if i == len(camps) - 1:
            out.append(svg_text(x + 14, y + 26, t1e, t1z, "lbl b"))
            out.append(svg_text(x + 14, y + 42, t2e, t2z, "lbl s"))
            continue
        ty = y + 26 if i in (0, 1, 2) else y - 40
        out.append(svg_text(x, ty, t1e, t1z, "lbl b", "middle"))
        out.append(svg_text(x, ty + 16, t2e, t2z, "lbl s", "middle"))
    out.append("</svg>")
    return "".join(out)


def plate_anatomy():
    rows = [
        ("Cohort 1: BF16 KV + kernel", "cohort 1：BF16 KV + kernel", C1, cand_runs[C1], "bf16"),
        ("Cohort 2: FP8 KV + kernel", "cohort 2：FP8 KV + kernel", C2, cand_runs[C2], "mono"),
    ]
    ttft_all = mean2(C1, cand_runs[C1], "all", "ttft_share")
    ttft_t10 = mean2(C1, cand_runs[C1], "slowest 10%", "ttft_share")
    osl_all = tail[C1][(1, "cand-r1", "all")]["osl"]
    osl_t10 = tail[C1][(1, "cand-r1", "slowest 10%")]["osl"]
    W, x0 = 380, 200
    out = [f'<svg viewBox="0 0 1040 330" role="img" aria-label="metric anatomy">',
           svg_text(20, 26, "Where a request's time goes (base arm, concurrency 1)", "一个请求的时间花在哪里（base，c1）", "lbl b"),
           ]
    for i, (en, zh_, share_, osl) in enumerate((("All requests", "全部请求", ttft_all, osl_all),
                                                ("Slowest 10% per token", "每 token 最慢的 10%", ttft_t10, osl_t10))):
        y = 50 + i * 54
        out.append(svg_text(20, y + 21, en, zh_, "lbl"))
        out.append(nt(20, y + 37, f"median OSL {osl:g}", "lbl s"))
        out.append(f'<rect x="{x0}" y="{y}" width="{W * share_:.1f}" height="30" class="seg ttft"/>')
        out.append(f'<rect x="{x0 + W * share_:.1f}" y="{y}" width="{W * (1 - share_):.1f}" height="30" class="seg dec"/>')
        out.append(nt(x0 + 6, y + 20, f"TTFT {100 * share_:.0f}%", "lbl w"))
        out.append(nt(x0 + W - 6, y + 20, f"decode {100 * (1 - share_):.0f}%", "lbl", "end"))
    # right panel: measured change of per-token E2EL
    X0, zero, scale = 640, 830, 6.0
    out.append(svg_text(X0 - 20, 26, "Measured change of E2EL per output token", "实测：每个输出 token 的 E2EL 变化", "lbl b"))
    out.append(f'<line x1="{zero}" y1="44" x2="{zero}" y2="160" class="axis"/>')
    for i, (en, zh_, coh, runs, cls) in enumerate(rows):
        for j, sub in enumerate(("all", "slowest 10%")):
            v = mean2(coh, runs, sub, "e2el_per_token")
            y = 50 + j * 54 + i * 17
            w = abs(v) * scale
            x = zero - w if v < 0 else zero
            out.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="14" class="bar {cls}"/>')
            out.append(nt(x - 6 if v < 0 else x + w + 6, y + 11, f"{sg(v)}%", "lbl s", "end" if v < 0 else "start"))
    out.append(nt(zero, 176, "0", "lbl s", "middle"))
    out.append(f'<rect x="{X0}" y="196" width="12" height="12" class="bar bf16"/>' + svg_text(X0 + 18, 207, rows[0][0], rows[0][1], "lbl s"))
    out.append(f'<rect x="{X0 + 210}" y="196" width="12" height="12" class="bar mono"/>' + svg_text(X0 + 228, 207, rows[1][0], rows[1][1], "lbl s"))
    out.append(svg_text(20, 250, "E2EL = TTFT + (OSL − 1) × ITL. A decode kernel shrinks the ITL term. The P90 of E2EL / OSL",
                        "E2EL = TTFT + (OSL − 1) × ITL。decode kernel 缩短的是 ITL 那一项。E2EL / OSL 的 P90", "lbl"))
    out.append(svg_text(20, 270, "is set by short replies, where TTFT is most of the time, so a TTFT cost flips their sign.",
                        "由短回复决定，它们的时间大部分是 TTFT，所以 TTFT 一变慢，它们的变化就从负翻成正。", "lbl"))
    out.append(svg_text(20, 296, "Bars: mean of the two rounds at concurrency 1; median per-request log change against the base.",
                        "条形：c1 两轮的平均；每个请求相对 base 的对数变化取中位数。", "lbl s"))
    out.append("</svg>")
    return "".join(out)


def plate_rootcause():
    W = 560 / 280
    x0 = 210
    out = [f'<svg viewBox="0 0 1040 300" role="img" aria-label="prefill kernel time">',
           svg_text(20, 26, "GPU kernel time of one prefill (rank 0), ms", "一次 prefill 的 GPU kernel 时间（rank 0），毫秒", "lbl b")]
    for i, (en, zh_, p_, cls) in enumerate((("FP8 E4M3 KV (recipe)", "FP8 E4M3 KV（配方）", pb, "base"),
                                            ("BF16 KV", "BF16 KV", pz, "bf16"))):
        y = 48 + i * 62
        other = p_["kernel_total_ms"] - p_["tilelang_main_kernel_ms"]
        tk = p_["tilelang_main_kernel_ms"]
        out.append(svg_text(20, y + 22, en, zh_, "lbl"))
        out.append(f'<rect x="{x0}" y="{y}" width="{other * W:.1f}" height="32" class="seg dec"/>')
        out.append(f'<rect x="{x0 + other * W:.1f}" y="{y}" width="{tk * W:.1f}" height="32" class="seg {cls}"/>')
        out.append(nt(x0 + 8, y + 21, f"other kernels {other:.1f}", "lbl"))
        out.append(nt(x0 + other * W + tk * W / 2, y + 21, f"TileLang {tk:.1f}", "lbl w", "middle"))
        tx = x0 + p_["ttft_median_ms"] * W
        out.append(f'<line x1="{tx:.1f}" y1="{y - 6}" x2="{tx:.1f}" y2="{y + 38}" class="tick"/>')
        out.append(nt(tx + 6, y + 21, f"TTFT {p_['ttft_median_ms']:.0f}", "lbl b"))
    out.append(svg_text(20, 196, "Bytes of one KV row that the sparse gather reads", "稀疏 gather 每读一行 KV 的字节数", "lbl b"))
    for i, (lab, bytes_, cls) in enumerate((("FP8 E4M3: (512 + 64) × 1 B", 576, "base"), ("BF16: (512 + 64) × 2 B", 1152, "bf16"))):
        y = 210 + i * 34
        out.append(nt(20, y + 17, lab, "lbl"))
        out.append(f'<rect x="{x0}" y="{y}" width="{bytes_ * 0.42:.1f}" height="22" class="seg {cls}"/>')
        out.append(nt(x0 + bytes_ * 0.42 + 8, y + 16, f"{bytes_} B", "lbl b"))
    out.append("</svg>")
    return "".join(out)


def plate_datapath():
    box = lambda x, y, w, h, cls="box": f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" class="{cls}"/>'  # noqa: E731
    out = [f'<svg viewBox="0 0 1040 330" role="img" aria-label="FP8 KV data path">{ARROW}',
           box(380, 120, 280, 92, "box pool"),
           nt(520, 146, "SGLang MLA KV pool", "lbl b", "middle"),
           nt(520, 166, "uint8 [slots, 576]  ·  viewed as float8_e4m3fn", "lbl s", "middle"),
           nt(520, 186, "512 latent + 64 rope, no block scales", "lbl s", "middle"),
           box(20, 30, 300, 74), nt(170, 56, "SGLang prefill (TileLang DSA)", "lbl b", "middle"),
           svg_text(170, 78, "reads the selected rows", "读取被选中的行", "lbl s", "middle"),
           box(20, 230, 300, 74), nt(170, 256, "set_mla_kv_buffer_triton_fp8_quant", "lbl b", "middle"),
           svg_text(170, 278, "SGLang writes new tokens as E4M3", "SGLang 把新 token 写成 E4M3", "lbl s", "middle"),
           box(720, 30, 300, 74, "box mono"), nt(870, 56, "glm53 monokernel, FP8 build", "lbl b", "middle"),
           svg_text(870, 78, "gather: E4M3 → BF16, 8 values per load", "gather：E4M3 → BF16，每次 8 个值", "lbl s", "middle"),
           box(720, 230, 300, 74, "box mono"), nt(870, 256, "verify-step KV write", "lbl b", "middle"),
           svg_text(870, 278, "BF16 round → E4M3, packed dwords", "先舍入到 BF16 → E4M3，按 dword 打包", "lbl s", "middle"),
           '<line x1="320" y1="67" x2="378" y2="130" class="edge" marker-end="url(#ah)"/>',
           '<line x1="320" y1="267" x2="378" y2="200" class="edge" marker-end="url(#ah)"/>',
           '<line x1="718" y1="67" x2="662" y2="130" class="edge mono" marker-end="url(#ah)"/>',
           '<line x1="718" y1="267" x2="662" y2="200" class="edge mono" marker-end="url(#ah)"/>',
           svg_text(520, 250, "#60: spec accepts fp8_e4m3; the runtime checks", "#60：spec 接受 fp8_e4m3；runtime 检查", "lbl s", "middle"),
           svg_text(520, 266, "dtype and the 576-wide row stride, then binds", "dtype 和 576 宽的行步长，然后绑定", "lbl s", "middle"),
           svg_text(520, 300, "No kernel code changed: the FP8 path already existed.", "kernel 代码没有改：FP8 路径本来就有。", "lbl b", "middle"),
           "</svg>"]
    return "".join(out)


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
    out.append(svg_text((X0 + X1) / 2, Y0 + 40, "P90 E2E-normalized interactivity (tok/s/user) →", "P90 E2E 归一化交互性（tok/s/user）→", "lbl", "middle"))
    out.append(svg_text(X0, Y1 - 10, "tok/s/GPU ↑", "tok/s/GPU ↑", "lbl"))
    groups = {}
    for r in refs3:
        groups.setdefault(r["curve"], []).append(r)
    # label anchor: (curve prefix, concurrency of the anchor point, dx, dy, text-anchor)
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
    out.append(f'<circle cx="100" cy="{ly}" r="6" class="pt base"/>' + svg_text(112, ly + 4, "this work, no monokernel", "本工作，不开 monokernel", "lbl s"))
    out.append(f'<circle cx="330" cy="{ly}" r="6" class="pt mono"/>' + svg_text(342, ly + 4, "this work, with the monokernel", "本工作，开 monokernel", "lbl s"))
    out.append(f'<circle cx="590" cy="{ly}" r="5" class="refpt"/>' + svg_text(602, ly + 4, "published on InferenceX (context only)", "InferenceX 已发布（只作背景）", "lbl s"))
    out.append("</svg>")
    return "".join(out)


def plate_gain():
    out = [f'<svg viewBox="0 0 1040 340" role="img" aria-label="gain and coverage">',
           svg_text(20, 24, "Decode-batch log lines by running requests (monokernel arm)", "按同时运行请求数划分的 decode 日志行（monokernel 组）", "lbl b"),
           svg_text(560, 24, "Matched requests: median change against the base", "同一批请求：相对 base 的变化中位数", "lbl b")]
    W = 380
    for i, c in enumerate(CONCS):
        y = 44 + i * 40
        h = hist_share[c]
        one, two = h.get("1", 0), h.get("2", 0)
        rest = 1 - one - two
        out.append(nt(20, y + 19, f"c{c}", "lbl b"))
        x = 60
        for v, cls in ((one, "base"), (two, "mono"), (rest, "dec")):
            out.append(f'<rect x="{x:.1f}" y="{y}" width="{W * v:.1f}" height="26" class="seg {cls}"/>')
            x += W * v
        out.append(nt(60 + W + 8, y + 18, f"{100 * share[c]:.0f}% ≤ 2", "lbl s"))
    ly = 216
    for k, (cls, en, zh_) in enumerate((("base", "1 running", "1 个在跑"), ("mono", "2 running", "2 个在跑"), ("dec", "3 or more (normal forward)", "3 个以上（普通 forward）"))):
        out.append(f'<rect x="{60 + k * 140}" y="{ly}" width="12" height="12" class="seg {cls}"/>' + svg_text(78 + k * 140, ly + 11, en, zh_, "lbl s"))
    zero, scale = 1010, 9.0
    out.append(f'<line x1="{zero}" y1="40" x2="{zero}" y2="200" class="axis"/>')
    for i, c in enumerate(CONCS):
        y = 44 + i * 40
        for j, (m, cls) in enumerate(((itl3[c], "mono"), (e2el3[c], "base"))):
            w = abs(m) * scale
            out.append(f'<rect x="{zero - w:.1f}" y="{y + j * 13}" width="{w:.1f}" height="11" class="bar {cls}"/>')
            out.append(nt(zero - w - 6, y + j * 13 + 10, f"{sg(m)}%", "lbl s", "end"))
    out.append(f'<rect x="700" y="{ly}" width="12" height="12" class="bar mono"/>' + nt(718, ly + 11, "ITL", "lbl s"))
    out.append(f'<rect x="770" y="{ly}" width="12" height="12" class="bar base"/>' + nt(788, ly + 11, "E2EL", "lbl s"))
    out.append(svg_text(20, 262, "The kernel runs a verify step only when one or two requests are running. As concurrency grows,",
                        "kernel 只在同时运行 1 到 2 个请求时运行 verify 步。并发升高后，", "lbl"))
    out.append(svg_text(20, 282, "more steps carry two requests or fall back to the normal forward, and the ITL gain shrinks with them.",
                        "更多步是 2 个请求一起跑，或者退回普通 forward，ITL 的收益随之变小。", "lbl"))
    out.append(svg_text(20, 310, "TTFT stays within ±1.5% at every point: the FP8 cache keeps the prefill as it was.",
                        "每个点的 TTFT 都在 ±1.5% 以内：FP8 cache 让 prefill 保持原样。", "lbl s"))
    out.append("</svg>")
    return "".join(out)


def plate_upstream():
    steps = [
        ("done", "sgl-amd-mono#60", "merged 0a107f1", "已合并 0a107f1"),
        ("todo", "SGLang pin bump", "main pins 05e0bb02", "main 仍 pin 05e0bb02"),
        ("todo", "MI35x nightly image", "contains the new pin", "包含新的 pin"),
        ("open", "cookbook #46662", "draft, needs the pin", "draft，依赖 pin 升级"),
        ("todo", "InferenceX PR", "config + recipe + changelog", "配置 + 配方 + changelog"),
        ("todo", "sweep · evals · sign-off", "then /use and merge", "然后 /use 并合并"),
    ]
    out = [f'<svg viewBox="0 0 1040 200" role="img" aria-label="upstream path">{ARROW}']
    w, gap = 152, 22
    for i, (state, t1, t2e, t2z) in enumerate(steps):
        x = 10 + i * (w + gap)
        out.append(f'<rect x="{x}" y="50" width="{w}" height="96" rx="9" class="box st-{state}"/>')
        out.append(nt(x + w / 2, 80, t1, "lbl b", "middle"))
        out.append(svg_text(x + w / 2, 102, t2e, t2z, "lbl s", "middle"))
        lab = {"done": ("done", "完成"), "open": ("open", "进行中"), "todo": ("not started", "未开始")}[state]
        out.append(svg_text(x + w / 2, 128, lab[0], lab[1], f"chip c-{state}", "middle"))
        if i:
            out.append(f'<line x1="{x - gap + 2}" y1="98" x2="{x - 3}" y2="98" class="edge" marker-end="url(#ah)"/>')
    out.append(svg_text(10, 178, "State on 2026-10-26. Each arrow is a hard dependency: the cookbook command fails at startup until the pin moves.",
                        "状态截至 2026-10-26。每个箭头都是硬依赖：pin 升级之前，cookbook 里的命令会在启动时失败。", "lbl s"))
    out.append("</svg>")
    return "".join(out)


# ----------------------------------------------------------------------------- sections
cell = lambda c, m, nd, unit="": f"{B(c, m):.{nd}f} → <b>{K(c, m):.{nd}f}</b>{unit}<br><span class=\"up\">{sg(g(c, m))}%</span>"  # noqa: E731
score_rows = [[f"c{c}", cell(c, "tput_gpu", 0), cell(c, "e2e_int_p90", 1), cell(c, "e2el_p90", 1, " s"), cell(c, "ttft_p90", 2, " s")] for c in CONCS]
score_head = [("Concurrency", "并发"), ("tok/s/GPU", "tok/s/GPU"), ("P90 interactivity", "P90 交互性"), ("P90 E2EL", "P90 E2EL"), ("P90 TTFT", "P90 TTFT")]

s_result = section("result", "01", ("The result", "结果"), "".join([
    P(f"At InferenceX's 3600-s window, the recipe plus <code>--monokernel-backend amd</code> beats the same recipe without it at every point we ran, on all three headline metrics. Every margin is larger than the A/A noise measured at concurrency 2: throughput {100 * noise3['tput_gpu']:.1f}%, P90 interactivity {100 * noise3['e2e_int_p90']:.1f}%, P90 E2EL {100 * noise3['e2el_p90']:.1f}%.",
      f"在 InferenceX 的 3600 秒窗口下，配方加上 <code>--monokernel-backend amd</code> 之后，我们跑的每个并发点、三个头条指标都好于不加的配方。每个差距都大于 c2 实测的 A/A 噪声：吞吐 {100 * noise3['tput_gpu']:.1f}%，P90 交互性 {100 * noise3['e2e_int_p90']:.1f}%，P90 E2EL {100 * noise3['e2el_p90']:.1f}%。"),
    table(score_head, score_rows, "score"),
    P("Each cell: without the monokernel → with it, and the gain. Latency gains are base / monokernel − 1, so a positive number is always better.",
      "每格：不开 monokernel → 开，以及增益。延迟类增益按 base / monokernel − 1 计算，所以正数总是更好。", "small"),
    UL([
        ("No kernel code changed between the failed first attempt and this result. What changed is which KV cache the kernel reads (sections 04–06).",
         "从第一次失败到这个结果，kernel 代码一行没改。改的是 kernel 读写哪一种 KV cache（第 04–06 节）。"),
        (f"Quality: on all 1319 GSM8K questions, base and monokernel differ by {ob} to {oc} questions (McNemar p = {mcnemar_p(ob, oc):.2f}); two runs of the monokernel arm alone differ by {rerun_c['only_a']} to {rerun_c['only_b']} (section 09).",
         f"质量：完整 GSM8K 1319 题，base 和 monokernel 之间只有一方答对的题是 {ob} 对 {oc}（McNemar p = {mcnemar_p(ob, oc):.2f}）；monokernel 自己跑两次也差 {rerun_c['only_a']} 对 {rerun_c['only_b']}（第 09 节）。"),
        ("Upstream: the recipe reaches InferenceX only after SGLang pins the kernel commit, an image ships it, and the cookbook PR merges (section 11).",
         "上游：要等 SGLang 把 kernel 的提交 pin 进去、镜像带上它、cookbook PR 合并之后，配方才能进 InferenceX（第 11 节）。"),
    ]),
]))

s_route = section("route", "02", ("The route", "路线"), "".join([
    P("Five camps, each with a decision rule written before the run. The first attempt was a no-go. A profile found why. One integration PR removed the cause. A 900-s rerun was too noisy to decide, and the 3600-s rerun decided it.",
      "五个营地，每一站的判定规则都在开跑之前写好。第一次是 no-go；profile 找到了原因；一个集成层的 PR 去掉了原因；900 秒的重跑噪声太大，判不了；3600 秒的重跑给出了结论。"),
    plate("I", ("The climb, camp by camp", "一站一站的爬坡"), plate_route(),
          ("Height is the score, not to scale. The dip at cohort 1 is the no-go: a faster decode with a slower prefill.",
           "高度代表分数，不按比例。cohort 1 的下凹就是 no-go：decode 变快了，prefill 却变慢了。")),
]))

s_metric = section("metric", "03", ("What the score measures", "分数在量什么"), "".join([
    P("AgentX replays recorded multi-turn agent sessions: long prompts that grow every turn and reuse their own prefix. Concurrency counts live sessions, not requests. InferenceX reports three headline numbers per point: total tokens per second per GPU, the P90 of E2E-normalized interactivity (one over the 90th percentile of E2EL / OSL, in tokens per second per user), and the P90 end-to-end latency.",
      "AgentX 回放录下来的多轮 agent 会话：prompt 很长，每一轮都在变长，并且复用自己的前缀。并发数指的是同时在跑的会话，不是请求。InferenceX 每个点报告三个头条数字：每张 GPU 每秒处理的总 token 数；E2E 归一化交互性的 P90，也就是 1 除以 E2EL / OSL 的 90 分位，单位是每用户每秒 token；以及端到端延迟的 P90。"),
    P("The second number decides how a decode kernel scores. A request's latency is its prefill plus one ITL per extra output token. Divided by the output length, a long reply is mostly ITL, but a short reply is mostly TTFT. The 90th percentile of that ratio is set by the short replies, so a kernel that only speeds up decode barely moves it, and any TTFT cost moves it the wrong way.",
      "决定 decode kernel 得分的是第二个数。一个请求的延迟等于 prefill 时间，加上每多一个输出 token 的一次 ITL。除以输出长度之后，长回复的时间主要是 ITL，短回复的时间主要是 TTFT。这个比值的 90 分位由短回复决定。所以只加速 decode 的 kernel 几乎推不动它，而 TTFT 的任何代价都会把它往坏的方向推。"),
    plate("II", ("Why the tail metric listens to TTFT", "为什么尾部指标对 TTFT 敏感"), plate_anatomy(),
          ("Left: base-arm time split. Right: the same kernel helps all requests in both cohorts, but only cohort 2 also helps the slowest tenth.",
           "左：base 的时间构成。右：同一个 kernel 在两个 cohort 里都让全部请求变快，但只有 cohort 2 也让最慢的那 10% 变快。")),
    callout("Why not judge by the median? The dashboard publishes the P90. A change that helps the median can still lose the score, which is exactly what cohort 1 did.",
            "为什么不看中位数？榜单发布的是 P90。一个改动可以让中位数变好、却让分数变差，cohort 1 正是这样。"),
]))

b0 = {c: eff[C1].get((c, "b0-r1")) for c in (1, 2)}
b1 = {c: eff[C1].get((c, "b1-r1")) for c in (1, 2)}
diag_rows = [[f"c{c}", sp(*name), f"{sg(d['time_to_first_token'])}%", f"{sg(d['inter_token_latency'])}%", f"{sg(d['request_latency'])}%"]
             for c in (1, 2) for name, d in ((("b0: BF16 KV only, no kernel", "b0：只改 BF16 KV，不开 kernel"), b0[c]),
                                             (("b1: BF16 KV + the image's kernel (05e0bb02)", "b1：BF16 KV + 镜像自带的 kernel（05e0bb02）"), b1[c])) if d]
s_attempt = section("attempt", "04", ("First attempt: a fast kernel, a flat score", "第一次：kernel 快了，分数没动"), "".join([
    P(f"The glm53 monokernel in {pr(7)} replaces GLM-5.3's whole verify-step forward when one or two requests are running. It accepted only a BF16 KV cache. That matched SGLang's cookbook cells for GLM-5.3 on MI355X, which run BF16 KV, and #7 listed FP8 KV under \"Not in this PR\". The InferenceX recipe runs <code>fp8_e4m3</code> KV, so the only candidate we could run was the recipe switched to BF16 KV, plus the kernel.",
      f"{pr(7)} 里的 glm53 monokernel 在同时运行 1 到 2 个请求时，替换 GLM-5.3 整个 verify 步的 forward。它当时只接受 BF16 KV cache。这和 SGLang cookbook 里 GLM-5.3 在 MI355X 上的配置一致，那些配置用 BF16 KV；#7 也把 FP8 KV 列在 \"Not in this PR\" 里。InferenceX 的配方用 <code>fp8_e4m3</code> KV，所以我们能跑的候选只有一个：把配方改成 BF16 KV，再加上 kernel。"),
    P(f"Cohort 1 ran 900-s windows in two rounds of opposite order. On matched requests, ITL fell {pair_rng[C1]['inter_token_latency']} and per-request E2EL {pair_rng[C1]['request_latency']}, but TTFT rose {pair_rng[C1]['time_to_first_token']}. The slowest tenth per output token moved {tail_rng[C1]}, so P90 interactivity did not improve, and the rule written before the run called it a no-go.",
      f"cohort 1 跑 900 秒窗口，两轮顺序相反。同一批请求上，ITL 变化 {pair_rng[C1]['inter_token_latency']}，单请求 E2EL 变化 {pair_rng[C1]['request_latency']}，但 TTFT 变化 {pair_rng[C1]['time_to_first_token']}。按每个输出 token 最慢的 10% 请求变化 {tail_rng[C1]}，所以 P90 交互性没有提升，事先写好的规则判为 no-go。"),
    P("Two diagnostic arms split the cause. b0 changes only the KV dtype and shows the TTFT cost without any kernel. b1 runs the kernel version that the image bundles and shows almost no gain, because that pin predates the kernel's own fixes.",
      "两个诊断臂把原因拆开。b0 只改 KV dtype，不开 kernel，TTFT 的代价照样出现。b1 用镜像自带的 kernel 版本，几乎没有收益，因为那个 pin 早于 kernel 自己的几次修复。"),
    table([("Point", "并发"), ("Arm", "诊断臂"), ("TTFT", "TTFT"), ("ITL", "ITL"), ("E2EL", "E2EL")], diag_rows),
]))

prof_rows = [
    [sp("TTFT, median of 8", "TTFT，8 次中位数"), f"{pb['ttft_median_ms']:.0f} ms", f"{pz['ttft_median_ms']:.0f} ms", f"{sg(100 * (pz['ttft_median_ms'] / pb['ttft_median_ms'] - 1), 0)}%"],
    [sp(f"All GPU kernels ({pb['kernel_launches']} launches)", f"全部 GPU kernel（{pb['kernel_launches']} 次 launch）"), f"{pb['kernel_total_ms']:.1f} ms", f"{pz['kernel_total_ms']:.1f} ms", f"{sg(pz['kernel_total_ms'] - pb['kernel_total_ms'])} ms"],
    [sp("TileLang <code>main_kernel</code>, 2 per layer", "TileLang <code>main_kernel</code>，每层 2 次"), f"{pb['tilelang_main_kernel_ms']:.1f} ms", f"{pz['tilelang_main_kernel_ms']:.1f} ms", f"{sg(pz['tilelang_main_kernel_ms'] - pb['tilelang_main_kernel_ms'])} ms"],
    [sp("… its grid-2080 launch, per layer", "… 其中 grid 2080 的那次，每层"), f"{pb['grid2080_per_layer_us_median']:.0f} µs", f"{pz['grid2080_per_layer_us_median']:.0f} µs", f"×{pz['grid2080_per_layer_us_median'] / pb['grid2080_per_layer_us_median']:.2f}"],
    [sp("… its grid-4160 launch, per layer", "… 其中 grid 4160 的那次，每层"), f"{pb['grid4160_per_layer_us_median']:.1f} µs", f"{pz['grid4160_per_layer_us_median']:.1f} µs", sp("unchanged", "不变")],
]
other_delta = (pz["kernel_total_ms"] - pz["tilelang_main_kernel_ms"]) - (pb["kernel_total_ms"] - pb["tilelang_main_kernel_ms"])
s_cause = section("cause", "05", ("Root cause: one prefill kernel, not the monokernel", "根因：prefill 里的一个 kernel，不是 monokernel"), "".join([
    P("To isolate prefill, we profiled one AgentX-shaped request: about 100k cached tokens plus 2048 new ones, TP8, the GPU trace of rank 0. Base is the recipe with FP8 E4M3 KV, b0 the same with BF16 KV; neither runs the monokernel.",
      "为了单独看 prefill，我们 profile 了一个 AgentX 形状的请求：约 10 万个已缓存 token，加 2048 个新 token，TP8，取 rank 0 的 GPU trace。base 是 FP8 E4M3 KV 的配方，b0 是同样的配方换成 BF16 KV；两者都不开 monokernel。"),
    table([("", ""), ("FP8 E4M3 KV", "FP8 E4M3 KV"), ("BF16 KV", "BF16 KV"), ("Change", "变化")], prof_rows),
    P(f"The extra time is one kernel. Everything else moves by {abs(other_delta):.1f} ms. By its grid, the grid-2080 launch is the DSA sparse attention of prefill: each new token gathers its 2048 selected KV rows, and a row is 1152 bytes in BF16 against 576 in FP8. We did not check this identification against the kernel source. The byte count alone predicts a twofold slowdown for a gather-bound kernel, and the trace shows ×{pz['grid2080_per_layer_us_median'] / pb['grid2080_per_layer_us_median']:.2f}.",
      f"多出来的时间来自一个 kernel，其余 kernel 合计只变了 {abs(other_delta):.1f} ms。按 grid 判断，grid 2080 的那次 launch 是 prefill 的 DSA 稀疏 attention：每个新 token 收集它选中的 2048 行 KV，一行在 BF16 下是 1152 字节，在 FP8 下是 576 字节。我们没有对照 kernel 源码核实这个判断。仅凭字节数，一个受 gather 限制的 kernel 就应该慢一倍，trace 里是 ×{pz['grid2080_per_layer_us_median'] / pb['grid2080_per_layer_us_median']:.2f}。"),
    plate("III", ("The prefill tax of a BF16 cache", "BF16 cache 给 prefill 加的税"), plate_rootcause(),
          ("The TileLang kernel doubles; all other kernels together stay flat. The second panel is the reason a gather-bound read doubles.",
           "TileLang kernel 翻倍，其余 kernel 合计不变。下半部分说明了一个受 gather 限制的读取为什么会翻倍。")),
    callout("Why not keep BF16 KV and speed up that kernel? It belongs to SGLang's DSA backend, outside the monokernel. The recipe already runs FP8 KV, and FP8 halves the bytes of every DSA read, not only this one.",
            "为什么不保留 BF16 KV、去优化那个 kernel？它属于 SGLang 的 DSA 后端，不在 monokernel 里。配方本来就用 FP8 KV，而且 FP8 让每一次 DSA 读取的字节数都减半，不只是这一个。"),
]))

e2e_rows = []
for rid, en, zh_ in (("e2e-glm53-fp8kv-verify-check", "FP8 verify check (new test: 4 and 8 rows over 2048 keys)", "FP8 verify check（新测试：2048 个 key 上 4 行和 8 行）"),
                     ("e2e-glm53-fp8kv-decode-check-local", "FP8 decode check (widths 2, 4, 8)", "FP8 decode check（宽度 2、4、8）"),
                     ("e2e-glm53-bf16-verify-check", "BF16 verify check (regression)", "BF16 verify check（回归）"),
                     ("e2e-glm53-fp8kv-neg-verify-check", "Negative control: latent E4M3 byte pairs swapped", "负对照：latent 的 E4M3 字节两两对调")):
    v = e2e[rid]
    kv, stv = e2ec(rid), e2ec(rid, "layer.streams.cosine")
    state = sp("fails, as intended", "按预期失败") if v["tests_failed"] else sp("passes", "通过")
    e2e_rows.append([sp(en, zh_), state, str(kv["checks"]), f"{kv['min']:.5f}", "-" if not stv else f"{stv['min']:.5f}", str(v["violation_lines"])])
s_fix = section("fix", "06", ("The fix: read the recipe's FP8 cache", "修复：直接读写配方的 FP8 cache"), "".join([
    P("The kernel already had an FP8 path. glm53 was written against the ATOM layout, whose FP8 KV rows hold E4M3 values without block scales. SGLang's HIP DSA backend stores exactly that: raw 576-byte E4M3 rows of 512 latent and 64 rope values, kept as <code>uint8</code> and viewed as <code>float8_e4m3fn</code>. Both sides write the BF16-rounded value in E4M3, so no conversion or rescaling sits between them.",
      "kernel 本来就有 FP8 路径。glm53 是按 ATOM 的布局写的，它的 FP8 KV 行存的是没有分块 scale 的 E4M3 值。SGLang 在 HIP 上的 DSA 后端存的正是这种格式：每行 576 字节的原始 E4M3，512 个 latent 值加 64 个 rope 值，用 <code>uint8</code> 存储，以 <code>float8_e4m3fn</code> 读取。两边写入的都是先舍入到 BF16、再转成 E4M3 的值，中间不需要任何转换或重新缩放。"),
    P(f"So {pr(60)} changed only the integration: the spec accepts <code>fp8_e4m3</code>, the build picks the FP8 kernel variant, and the runtime binds SGLang's rows after checking their dtype and 576-wide stride. It merged as {a(f'{GH}/sgl-project/sgl-amd-mono/commit/0a107f15362714c45be34ca37e4dec286fa3d303', '<code>0a107f1</code>')} on 2026-10-26.",
      f"所以 {pr(60)} 只改了集成层：spec 接受 <code>fp8_e4m3</code>，构建时选 FP8 版本的 kernel，runtime 检查 dtype 和 576 宽的行步长之后，绑定 SGLang 的 KV 行。它在 2026-10-26 以 {a(f'{GH}/sgl-project/sgl-amd-mono/commit/0a107f15362714c45be34ca37e4dec286fa3d303', '<code>0a107f1</code>')} 合并。"),
    plate("IV", ("One cache, two writers, two readers", "一个 cache，两个写入方，两个读取方"), plate_datapath(),
          ("SGLang and the monokernel read and write the same rows in the same format, so either can take any step.",
           "SGLang 和 monokernel 用同一种格式读写同一批行，所以任何一步交给哪一边都可以。")),
    P("The e2e checks run SGLang and the kernel on the same steps and compare every layer's KV and stream outputs by cosine. A negative control proves the check can fail: swapping the E4M3 byte pairs of the latent drops the KV cosine to 0.095 and trips 1189 violations.",
      "e2e check 让 SGLang 和 kernel 跑同样的步，逐层比较 KV 和 stream 输出的 cosine。负对照证明这个检查确实会失败：把 latent 的 E4M3 字节两两对调之后，KV cosine 降到 0.095，触发 1189 次违例。"),
    table([("Check", "检查"), ("Result", "结果"), ("Checks", "次数"), ("KV cosine min", "KV cosine 最小"), ("Stream cosine min", "stream cosine 最小"), ("Violations", "违例")], e2e_rows),
    callout("Why not quantize inside the kernel with block scales, as other FP8 caches do? The cache belongs to SGLang. The kernel must read and write the exact rows that SGLang's own prefill and decode use, or the two paths drift apart step by step.",
            "为什么不像其他 FP8 cache 那样，在 kernel 里用分块 scale 自己量化？因为 cache 属于 SGLang。kernel 必须读写 SGLang 自己的 prefill 和 decode 用的那些行，否则两条路径会一步一步地分开。"),
]))

s_score = section("score", "07", ("The score at 3600 s", "3600 秒的分数"), "".join([
    P(f"Cohort 2 repeated the 900-s ladder with the fix. On matched requests ITL fell {pair_rng[C2]['inter_token_latency']}, E2EL {pair_rng[C2]['request_latency']}, and TTFT stayed at {pair_rng[C2]['time_to_first_token']}; the slowest tenth per token now moved {tail_rng[C2]}. Yet the A/A noise of P90 interactivity in a 900-s window was {100 * noise2['e2e_int_p90']:.0f}% ({100 * noise1['e2e_int_p90']:.0f}% in cohort 1), and the second round at concurrency 2 fell inside it. The rule still said no-go. The window, not the kernel, was the limit.",
      f"cohort 2 用修复后的 kernel 重跑 900 秒的梯子。同一批请求上 ITL 变化 {pair_rng[C2]['inter_token_latency']}，E2EL 变化 {pair_rng[C2]['request_latency']}，TTFT 保持在 {pair_rng[C2]['time_to_first_token']}；每 token 最慢的 10% 现在变化 {tail_rng[C2]}。可是 900 秒窗口里 P90 交互性的 A/A 噪声有 {100 * noise2['e2e_int_p90']:.0f}%（cohort 1 是 {100 * noise1['e2e_int_p90']:.0f}%），c2 的第二轮落在噪声内，规则仍然判 no-go。限制在窗口，不在 kernel。"),
    P(f"Cohort 3 used InferenceX's own 3600-s window at concurrency 1, 2, 4, and 10, a fresh server per point, and an A/A pair at concurrency 2. The A/A noise of P90 interactivity fell to {100 * noise3['e2e_int_p90']:.1f}%. All nine runs were valid: each profiling phase ran the full window, failures and drain cancellations stayed far below 10%, the validator passed, and the realized accept length was 2.990 against InferenceX's golden 2.99.",
      f"cohort 3 用 InferenceX 自己的 3600 秒窗口，并发 1、2、4、10，每个点都新起服务器，并在 c2 做一组 A/A。P90 交互性的 A/A 噪声降到 {100 * noise3['e2e_int_p90']:.1f}%。九个 run 全部有效：每个 profiling 阶段都跑满窗口，失败和排空取消都远低于 10%，validator 通过，实现的接受长度是 2.990，InferenceX 的 golden 值是 2.99。"),
    plate("V", ("The score map", "分数地图"), plate_score(),
          ("Each arrow goes from the recipe without the monokernel to the recipe with it. Hollow points are InferenceX's published rows (context only).",
           "每个箭头从不开 monokernel 的配方指向开了的配方。空心点是 InferenceX 已发布的结果（只作背景）。")),
    P(f"InferenceX's Check 14 recommends at least five Pareto-frontier points per curve (throughput against P90 E2EL). This four-point ladder has {par['baseline']['frontierPoints']} frontier points without the monokernel and {par['candidate']['frontierPoints']} with it, because AgentX's P90 E2EL is not monotone in concurrency: concurrency 4 has a lower P90 E2EL than 1 and 2 and a higher throughput, so it dominates them.",
      f"InferenceX 的 Check 14 建议每条曲线至少有 5 个 Pareto 前沿点（吞吐对 P90 E2EL）。这个 4 点的梯子里，不开 monokernel 有 {par['baseline']['frontierPoints']} 个前沿点，开了有 {par['candidate']['frontierPoints']} 个。原因是 AgentX 的 P90 E2EL 不随并发单调：c4 的 P90 E2EL 比 c1、c2 低，吞吐又更高，所以 c1、c2 被它支配。"),
]))

gain_rows = [[f"c{c}", str(eff[C3][(c, 'cand8-r1')]["requests"]), f"{sg(ttft3[c])}%", f"{sg(itl3[c])}%", f"{sg(e2el3[c])}%",
              f"{sg(tail[C3][(c, 'cand8-r1', 'slowest 10%')]['e2el_per_token'])}%" if (c, 'cand8-r1', 'slowest 10%') in tail[C3] else "-",
              f"{100 * share[c]:.0f}%"] for c in CONCS]
s_gain = section("gain", "08", ("Where the gain comes from", "收益从哪里来"), "".join([
    P("To separate the kernel from workload noise, we matched requests across runs by conversation and turn, kept pairs whose inputs differ by at most 8 tokens (the cache-busting prefix), and took the median log change of each metric against the base.",
      "为了把 kernel 的作用和负载噪声分开，我们按会话和轮次在不同 run 之间配对请求，只保留输入长度相差不超过 8 个 token 的对（差别来自防缓存的前缀），然后对每个指标取相对 base 的对数变化中位数。"),
    table([("Point", "并发"), ("Matched", "配对数"), ("TTFT", "TTFT"), ("ITL", "ITL"), ("E2EL", "E2EL"), ("Slowest 10% per token", "每 token 最慢 10%"), ("Lines with ≤ 2 running", "≤ 2 个在跑的日志行")], gain_rows),
    plate("VI", ("The gain follows the steps the kernel can take", "收益跟着 kernel 能接的步走"), plate_gain(),
          ("Left: what the server was running when it logged a decode batch. Right: the matched-request change. The two shrink together.",
           "左：服务器记录 decode batch 时在跑几个请求。右：配对请求的变化。两者一起变小。")),
    P(f"At concurrency 1, {100 * hist_share[1].get('1', 0):.0f}% of decode-batch lines carry one request; at concurrency 4, {100 * hist_share[4].get('1', 0):.0f}% carry one and {100 * hist_share[4].get('2', 0):.0f}% carry two. ITL falls {abs(itl3[1]):.1f}% at concurrency 1 but only {abs(itl3[4]):.1f}% at 4, although both stay above 95% coverage. A two-request step may gain less than a one-request step; we did not measure the per-width speedup, so that explanation is unverified.",
      f"c1 时，{100 * hist_share[1].get('1', 0):.0f}% 的 decode 日志行只有 1 个请求；c4 时，1 个请求的占 {100 * hist_share[4].get('1', 0):.0f}%，2 个请求的占 {100 * hist_share[4].get('2', 0):.0f}%。两者的覆盖率都在 95% 以上，但 ITL 在 c1 降 {abs(itl3[1]):.1f}%，在 c4 只降 {abs(itl3[4]):.1f}%。可能是两个请求一起跑的步比一个请求的步收益小；我们没有分宽度测加速比，所以这个解释未经验证。"),
]))

q_rows = [[sp("Cohort 1: FP8 KV vs BF16 KV + kernel", "cohort 1：FP8 KV 对 BF16 KV + kernel"), "c2", str(gsm1["filters"]["strict-match"]["n"]),
           f"{gsm1['filters']['strict-match']['base']:.3f}", f"{gsm1['filters']['strict-match']['cand']:.3f}",
           f"{gsm1['filters']['strict-match']['only_base_right']} / {gsm1['filters']['strict-match']['only_cand_right']}", f"{gsm1['filters']['strict-match']['mcnemar_p']:.2f}"],
          [sp("Cohort 2: FP8 KV, kernel off vs on", "cohort 2：FP8 KV，kernel 关对开"), "c2", str(gsm2["filters"]["strict-match"]["n"]),
           f"{gsm2['filters']['strict-match']['base']:.3f}", f"{gsm2['filters']['strict-match']['cand']:.3f}",
           f"{gsm2['filters']['strict-match']['only_base_right']} / {gsm2['filters']['strict-match']['only_cand_right']}", f"{gsm2['filters']['strict-match']['mcnemar_p']:.2f}"]]
for r in strict3:
    lab = {"base vs cand8": ("Cohort 3: kernel off vs on", "cohort 3：kernel 关对开"), "rerun of base": ("Cohort 3: base vs base rerun", "cohort 3：base 对 base 重跑"),
           "rerun of cand8": ("Cohort 3: kernel vs kernel rerun", "cohort 3：kernel 对 kernel 重跑")}[r["kind"]]
    q_rows.append([sp(f"{lab[0]} ({r['a']} / {r['b']})", f"{lab[1]}（{r['a']} / {r['b']}）"), r["a"].split()[0], str(r["n"]),
                   f"{r['acc_a']:.3f}", f"{r['acc_b']:.3f}", f"{r['only_a']} / {r['only_b']}", f"{r['mcnemar_p']:.2f}"])
s_quality = section("quality", "09", ("Quality", "质量"), "".join([
    P("GSM8K with real MTP acceptance, strict-match, through InferenceX's harness (lm-evaluation-harness). Cohorts 1 and 2 used 500 questions; cohort 3 used all 1319, with a rerun of each arm at concurrency 10 to measure how much a rerun alone moves the answers.",
      "GSM8K 用真实的 MTP 接受，strict-match，通过 InferenceX 的评测工具（lm-evaluation-harness）。cohort 1 和 2 用 500 题；cohort 3 用全部 1319 题，并在 c10 把每一组都重跑一次，用来测量单是重跑会让答案变化多少。"),
    table([("Pair", "对比"), ("Point", "并发"), ("n", "题数"), ("A", "A"), ("B", "B"), ("Only A / only B right", "只有 A 对 / 只有 B 对"), ("McNemar p", "McNemar p")], q_rows),
    UL([
        (f"Pooled over the three kernel-off/on pairs: {ob} to {oc} (p = {mcnemar_p(ob, oc):.2f}). The first pair alone leans to the base ({pairs3[0]['only_a']} to {pairs3[0]['only_b']}), but two runs of the kernel arm differ {rerun_c['only_a']} to {rerun_c['only_b']}, so that lean is rerun variation.",
         f"三组 kernel 关对开合计：{ob} 对 {oc}（p = {mcnemar_p(ob, oc):.2f}）。只看第一组偏向 base（{pairs3[0]['only_a']} 对 {pairs3[0]['only_b']}），但 kernel 组自己跑两次就差 {rerun_c['only_a']} 对 {rerun_c['only_b']}，所以这个偏向是重跑的波动。"),
        (f"Of the {len(lost)} questions the kernel arm lost in that first pair, {len(runaway)} are runaway reasoning with no final answer ({run_chars[0] // 1000}k–{run_chars[-1] // 1000}k characters) and the rest are ordinary solution errors; none looks like corrupted numerics.",
         f"第一组里 kernel 组答错、base 答对的 {len(lost)} 题中，{len(runaway)} 题是推理失控、没有给出最终答案（{run_chars[0] // 1000}k–{run_chars[-1] // 1000}k 字符），其余是普通的解题错误；没有一题像是数值被破坏。"),
        (f"Kernel coverage in the GSM8K runs: {100 * gcov['gsm8k-glm53-l2ov-c2-cand8q']:.0f}% of decode-batch lines at concurrency 2, {100 * gcov['gsm8k-glm53-l2ov-c10-cand8q']:.0f}% and {100 * gcov['gsm8k-glm53-l2ov-c10-cand8q-r2']:.0f}% at concurrency 10. InferenceX runs its own eval at the highest concurrency of a group, where the kernel serves a quarter of the steps, so the concurrency-2 pair is the evidence for the kernel path.",
         f"GSM8K run 里 kernel 的覆盖：c2 是 {100 * gcov['gsm8k-glm53-l2ov-c2-cand8q']:.0f}% 的 decode 日志行，c10 是 {100 * gcov['gsm8k-glm53-l2ov-c10-cand8q']:.0f}% 和 {100 * gcov['gsm8k-glm53-l2ov-c10-cand8q-r2']:.0f}%。InferenceX 在一组的最高并发跑自己的 eval，那里 kernel 只接四分之一的步，所以 kernel 路径的质量证据是 c2 这一对。"),
    ]),
]))

ref_rows = [[f"c{r['conc']}", E(r["curve"]), f"{r['tput']:.0f}", f"{r['int90']:.1f}", f"{r['e2el90']:.1f}", a(r["url"], E(r["source"])) if r.get("url") else E(r["source"])]
            for r in refs3 if "this work" not in r["curve"]]
osl_rows = []
for c in (1, 4, 10):
    o = osl_local[c]
    osl_rows.append([f"c{c}", sp("this work, with the monokernel (3600 s)", "本工作，开 monokernel（3600 秒）"), f"{o['p50']:.0f}", f"{o['p90']:.0f}"])
    if c in osl_b300:
        r = osl_b300[c]
        osl_rows.append([f"c{c}", sp("published GLM-5.3 B300 SGLang FP4", "已发布 GLM-5.3 B300 SGLang FP4"), f"{r['median_output_tokens_actual']:.0f}", f"{r['p90_output_tokens_actual']:.0f}"])
s_refs = section("refs", "10", ("Against published results", "和已发布结果比"), "".join([
    P(f"InferenceX's public API has no vLLM AgentX result for any GLM-5.x model. On MI355X, GLM-5.3 has one published point, TileRT at concurrency 1 on 16 GPUs; at concurrency 1 this recipe is {sg(rel(K(1, 'tput_gpu'), tile['tput']), 0)}% in throughput per GPU and {sg(rel(K(1, 'e2e_int_p90'), tile['int90']), 0)}% in P90 interactivity against it. GLM-5.3 on B300 with NVFP4 still leads: at concurrency 10 this recipe is {sg(rel(K(10, 'tput_gpu'), b300[10]['tput']), 0)}% in throughput and {sg(rel(K(10, 'e2e_int_p90'), b300[10]['int90']), 0)}% in interactivity.",
      f"InferenceX 的公开 API 里，没有任何 GLM-5.x 模型的 vLLM AgentX 结果。MI355X 上的 GLM-5.3 只有一个已发布点：TileRT，c1，16 张卡；在 c1，这个配方相对它每卡吞吐 {sg(rel(K(1, 'tput_gpu'), tile['tput']), 0)}%，P90 交互性 {sg(rel(K(1, 'e2e_int_p90'), tile['int90']), 0)}%。B300 上的 GLM-5.3 NVFP4 仍然领先：在 c10，这个配方的吞吐是 {sg(rel(K(10, 'tput_gpu'), b300[10]['tput']), 0)}%，交互性是 {sg(rel(K(10, 'e2e_int_p90'), b300[10]['int90']), 0)}%。"),
    table([("Point", "并发"), ("Published configuration", "已发布配置"), ("tok/s/GPU", "tok/s/GPU"), ("P90 interactivity", "P90 交互性"), ("P90 E2EL (s)", "P90 E2EL（秒）"), ("Source", "来源")], ref_rows),
    P(f"Raw E2EL does not compare across these configurations because the replies differ in length. Our P90 output length is {min(osl_ratio):.1f}–{max(osl_ratio):.1f}× B300's at the same concurrency, while medians stay close. It is the same on an older image, so it is not the image or the chat template. The most likely cause is the weights (FP8 here, NVFP4 there); that is unverified. E2E-normalized interactivity divides the length out and is the fairest of the three numbers.",
      f"这些配置之间不能直接比 E2EL，因为回复的长度不同。同样并发下，我们的 P90 输出长度是 B300 的 {min(osl_ratio):.1f}–{max(osl_ratio):.1f} 倍，中位数却很接近。旧镜像上也是这样，所以原因不是镜像，也不是 chat template。最可能的原因是权重（这里是 FP8，那里是 NVFP4），未经验证。E2E 归一化交互性除掉了长度，是三个数里最公平的一个。"),
    table([("Point", "并发"), ("Run", "run"), ("Median OSL", "OSL 中位数"), ("P90 OSL", "OSL P90")], osl_rows),
]))

gap_items = [
    (f"Pin bump: SGLang <code>main</code> still pins sgl-amd-mono at <code>05e0bb02</code>, whose glm53 spec accepts only BF16 KV. The open pin bump {pr(45045, 'sgl-project/sglang')} pins <code>e952b221</code>, which also predates #60.",
     f"pin 升级：SGLang <code>main</code> 仍然把 sgl-amd-mono pin 在 <code>05e0bb02</code>，它的 glm53 spec 只接受 BF16 KV。现有的 pin 升级 PR {pr(45045, 'sgl-project/sglang')} pin 的是 <code>e952b221</code>，也早于 #60。"),
    ("Image: the first MI35x nightly that contains the new pin; pin its digest.", "镜像：第一个包含新 pin 的 MI35x nightly；固定它的 digest。"),
    (f"Cookbook: {pr(46662, 'sgl-project/sglang')} (draft) adds the recipe as section 3.6 of the GLM-5.3 page. InferenceX accepts a recipe only after its production-useful flags are on a merged cookbook page.",
     f"cookbook：{pr(46662, 'sgl-project/sglang')}（draft）把配方加为 GLM-5.3 页面的 3.6 节。InferenceX 只接受影响生产效果的参数都已写进已合并 cookbook 页面的配方。"),
    ("Confirmation: rerun the 3600-s ladder on that image without the overlay, and add points (for example concurrency 6 and 8) to reach five frontier points.",
     "确认：在那个镜像上、不挂 overlay，重跑 3600 秒的梯子，并补几个点（例如 c6、c8），凑够 5 个前沿点。"),
    ("InferenceX PR: the config, the recipe, one perf-changelog block, and a bilingual body; jhinpan has pull access only, so it goes through a fork or a collaborator.",
     "InferenceX PR：配置、配方、一段 perf-changelog、中英双语正文；jhinpan 只有 pull 权限，所以要通过 fork 或者有权限的同事。"),
    ("Sweep, evals, CODEOWNER sign-off, and an authorized <code>/use</code> of the sweep run.", "sweep、eval、CODEOWNER 签核，以及有权限的人对 sweep run 发 <code>/use</code>。"),
]
s_up = section("upstream", "11", ("From a local win to InferenceX", "从本地结果到 InferenceX"), "".join([
    P("No kernel work is left. What is left is release plumbing, and each step depends on the one before it.",
      "kernel 的工作已经做完。剩下的是发布流程，每一步都依赖前一步。"),
    plate("VII", ("The remaining path", "剩下的路"), plate_upstream(),
          ("The cookbook PR is open but cannot merge before the pin bump, because older pins refuse FP8 KV under the monokernel at startup.",
           "cookbook PR 已经开了，但在 pin 升级之前不能合并，因为旧的 pin 会在启动时拒绝 monokernel 加 FP8 KV。")),
    UL(gap_items),
]))

s_method = section("method", "12", ("Method and limits", "方法与限制"), UL([
    ("Local node, not InferenceX's cluster. The kernel commit was installed over the published image, so these runs are not the confirmation run on a shipped image.",
     "本地节点，不是 InferenceX 的集群。kernel 的提交是装在已发布镜像之上的，所以这些 run 不算在发布镜像上的确认 run。"),
    ("One 3600-s run per point; the A/A pair is at concurrency 2 only, and the other points assume the same noise.",
     "每个点只跑一次 3600 秒；A/A 只在 c2，其他点按同样的噪声水平假定。"),
    ("AgentX runs use InferenceX's synthetic acceptance at the golden accept length 2.99 in both arms; GSM8K runs use real acceptance.",
     "AgentX run 在两组都用 InferenceX 的模拟接受，接受长度取 golden 值 2.99；GSM8K run 用真实接受。"),
    ("Every run held all 8 GPUs under a lock after a 30-sample idle check, with amd-smi samples every 10 s and a foreign-process check; every point used a fresh server and sent no generation before the timed phase.",
     "每个 run 都先做 30 次采样的空闲检查，再锁住全部 8 张卡；每 10 秒用 amd-smi 采样一次，并检查外来进程；每个点都新起服务器，计时阶段之前不发任何生成请求。"),
    ("Each cohort's decision rule was written before its runs. One amendment replaced a span check that does not fit AgentX with the phase-based validity check, before any verdict.",
     "每个 cohort 的判定规则都在开跑之前写好。唯一的一次修订，是在出任何结论之前，把不适合 AgentX 的跨度检查换成按阶段的有效性检查。"),
    ("Published rows are context only: different hardware, precision, deployment shape, and output-length distribution.",
     "已发布的结果只作背景：硬件、精度、部署方式和输出长度分布都不同。"),
]))

files = [
    ("cohort3-fp8kv-3600s/analysis/compare_round1_base_cand8.json", "headline metrics, gains, and the A/A noise floor", "头条指标、增益和 A/A 噪声"),
    ("cohort3-fp8kv-3600s/analysis/pair_effects.json", "matched-request effects and tails", "配对请求的效果和尾部"),
    ("cohort3-fp8kv-3600s/requests", "per-request CSV of every 3600-s run", "每个 3600 秒 run 的逐请求 CSV"),
    ("cohort3-fp8kv-3600s/gsm8k", "per-question GSM8K correctness", "GSM8K 逐题对错"),
    ("ttft-profile/summary.json", "the prefill kernel profile", "prefill 的 kernel profile"),
    ("fp8kv-validation/e2e_checks.json", "the #60 layer checks", "#60 的逐层检查"),
    ("inferencex-config", "the local InferenceX config and recipe", "本地的 InferenceX 配置和配方"),
    ("scripts/make_page.py", "the generator of this page", "本页的生成脚本"),
]
s_data = section("data", "13", ("Data and files", "数据与文件"), "".join([
    P(f"Raw data, decision rules, and analysis outputs live in {a(DATA_URL, f'<code>data/{SLUG}/</code>')}. Hostnames and local paths are replaced; per-request records, GSM8K outputs, and profiler traces are reduced to compact CSV and JSON.",
      f"原始数据、判定规则和分析结果都在 {a(DATA_URL, f'<code>data/{SLUG}/</code>')}。主机名和本机路径已经替换；逐请求记录、GSM8K 输出和 profiler trace 都压缩成了紧凑的 CSV 和 JSON。"),
    UL([(f"{a(f'{BLOB}/{p}' if '.' in p.split('/')[-1] else f'{DATA_URL}/{p}', f'<code>{p}</code>')}: {en}", f"{a(f'{BLOB}/{p}' if '.' in p.split('/')[-1] else f'{DATA_URL}/{p}', f'<code>{p}</code>')}：{z}") for p, en, z in files]),
]))

epilogue = section("epilogue", "14", ("Epilogue", "结语"), "".join([
    P("A kernel can win the step it was built for and still lose the score. When the headline number does not move, look at what the kernel's preconditions force onto the rest of the pipeline. Here a dtype precondition taxed prefill, and the tail metric weighs prefill most. The fix was not a faster kernel but a precondition the rest of the system already met.",
      "一个 kernel 可以赢下它负责的那一步，却输掉分数。头条数字不动的时候，要看 kernel 的前提条件给流水线的其他部分带来了什么。这里，一个 dtype 前提给 prefill 加了税，而尾部指标最看重 prefill。修复不是更快的 kernel，而是换成系统其他部分本来就满足的前提。"),
    P("The second lesson is about windows. A P90 over 900 s of agent sessions moved 20% between identical runs; over 3600 s it moved 1.3%. Measure at the window the score is published at before calling a tail result.",
      "第二个教训关于窗口。900 秒的 agent 会话上，两次完全相同的 run 之间 P90 能差 20%；3600 秒时只差 1.3%。在对尾部结果下结论之前，先用分数发布时的那个窗口来测。"),
]))

SECTIONS = [("result", "01", "The result", "结果"), ("route", "02", "The route", "路线"), ("metric", "03", "What the score measures", "分数在量什么"),
            ("attempt", "04", "First attempt", "第一次尝试"), ("cause", "05", "Root cause", "根因"), ("fix", "06", "The fix", "修复"),
            ("score", "07", "Score at 3600 s", "3600 秒的分数"), ("gain", "08", "Where the gain comes from", "收益来源"),
            ("quality", "09", "Quality", "质量"), ("refs", "10", "Published results", "已发布结果"), ("upstream", "11", "To InferenceX", "到 InferenceX"),
            ("method", "12", "Method and limits", "方法与限制"), ("data", "13", "Data and files", "数据与文件"), ("epilogue", "14", "Epilogue", "结语")]
rail = ('<nav class="rail" aria-label="Contents"><a class="back" href="/sources/">' + sp("← The Library", "← 文库") + "</a><ol>"
        + "".join(f'<li><a href="#{sid}"><span class="rn">{num}</span>{sp(en, z)}</a></li>' for sid, num, en, z in SECTIONS) + "</ol></nav>")

spec = [
    (("Hardware", "硬件"), ("8 × AMD Instinct MI355X (gfx950), one node, TP8 EP1", "8 × AMD Instinct MI355X（gfx950），单节点，TP8 EP1")),
    (("Model", "模型"), (f"{a('https://huggingface.co/zai-org/GLM-5.3', 'zai-org/GLM-5.3')} at <code>ba32f8c1</code> (FP8), EAGLE MTP 3-1-4", f"{a('https://huggingface.co/zai-org/GLM-5.3', 'zai-org/GLM-5.3')}，<code>ba32f8c1</code>（FP8），EAGLE MTP 3-1-4")),
    (("Engine", "引擎"), (f"SGLang {a(f'{GH}/sgl-project/sglang/commit/fe1bd108372c9ba2ab959fcf99cf196433d155d5', '<code>fe1bd108</code>')} in <code>lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261024</code>, AITER <code>285579b6</code>",
                          f"SGLang {a(f'{GH}/sgl-project/sglang/commit/fe1bd108372c9ba2ab959fcf99cf196433d155d5', '<code>fe1bd108</code>')}，镜像 <code>lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261024</code>，AITER <code>285579b6</code>")),
    (("Kernel", "kernel"), (f"sgl-amd-mono glm53 at <code>0a107f1</code> ({pr(60)}), installed over the image's <code>05e0bb02</code>", f"sgl-amd-mono glm53，<code>0a107f1</code>（{pr(60)}），装在镜像自带的 <code>05e0bb02</code> 之上")),
    (("Client", "客户端"), ("InferenceX AgentX (AIPerf), 3600-s windows, golden accept length 2.99", "InferenceX AgentX（AIPerf），3600 秒窗口，golden 接受长度 2.99")),
    (("Measured", "测量时间"), ("2026-10-24 → 2026-10-26", "2026-10-24 → 2026-10-26")),
    (("Raw data", "原始数据"), (a(DATA_URL, f"<code>data/{SLUG}/</code>"), a(DATA_URL, f"<code>data/{SLUG}/</code>"))),
]
masthead = ('<header class="masthead"><svg class="topo" viewBox="0 0 1200 260" preserveAspectRatio="none" aria-hidden="true">'
            + "".join(f'<path d="M-20,{y} C200,{y - 50} 420,{y + 40} 640,{y - 26} S1000,{y - 70} 1220,{y - 36}"/>' for y in range(40, 300, 22))
            + '</svg><div class="mast-in">'
            + bi("div", "Experiment 006 · Model hill-climb", "实验 006 · 模型爬坡记录", 'class="kicker"')
            + "<h1>GLM-5.3 on MI355X: the kernel was fast, the score was not</h1>"
            + P(f"A decode monokernel cut GLM-5.3's inter-token latency by {round(min(abs(x) for x in itl1))}–{round(max(abs(x) for x in itl1))}% on MI355X, and the InferenceX AgentX score did not move. The cost sat in prefill: the kernel accepted only a BF16 KV cache, and BF16 KV doubled the time of one DSA prefill kernel. Letting the kernel read the recipe's own FP8 cache, an integration change with no kernel code, turned the same kernel into +{gain_lo('tput_gpu')}–{gain_hi('tput_gpu')}% throughput and +{gain_lo('e2e_int_p90')}–{gain_hi('e2e_int_p90')}% interactivity at concurrency 1–4 over InferenceX's 3600-s window, with GSM8K unchanged.",
                f"一个 decode monokernel 让 GLM-5.3 在 MI355X 上的 inter-token latency 降了 {round(min(abs(x) for x in itl1))}–{round(max(abs(x) for x in itl1))}%，InferenceX AgentX 的分数却没有动。代价在 prefill：kernel 当时只接受 BF16 KV cache，而 BF16 KV 让 DSA prefill 里一个 kernel 的时间翻了一倍。让 kernel 直接读写配方本来的 FP8 cache 之后（只改集成层，不改 kernel 代码），同一个 kernel 在 InferenceX 的 3600 秒窗口下，c1–c4 吞吐提升 {gain_lo('tput_gpu')}–{gain_hi('tput_gpu')}%，交互性提升 {gain_lo('e2e_int_p90')}–{gain_hi('e2e_int_p90')}%，GSM8K 不变。", "dek")
            + '<dl class="spec">' + "".join(f"<dt>{sp(*k)}</dt><dd>{sp(*v)}</dd>" for k, v in spec) + "</dl></div></header>")

CSS = """
:root{--paper:#edf1ec;--card:#f7f9f6;--sunk:#e1e8e2;--rule:#c8d3ca;--rule2:#9db0a3;--ink:#10241f;--ink2:#3d564f;--ink3:#6a7f78;
--base:#4a6787;--mono:#c2366b;--bf16:#a77a1f;--ref:#8a958f;--good:#2e7d4f;--warn:#b5651d;--contour:#cfdbd1;
--serif:"Instrument Serif",Georgia,"Noto Serif SC",serif;--sans:"Atkinson Hyperlegible","Noto Sans SC","PingFang SC","Microsoft YaHei",sans-serif;--mono-f:"Reddit Mono",ui-monospace,Menlo,Consolas,monospace}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:var(--paper);color:var(--ink);font:16.5px/1.62 var(--sans);-webkit-font-smoothing:antialiased}
a{color:var(--base);text-decoration:underline;text-decoration-color:var(--rule2);text-underline-offset:2px}a:hover{color:var(--mono)}
code{font:0.86em var(--mono-f);background:var(--sunk);padding:1px 5px;border-radius:4px}
.masthead{position:relative;overflow:hidden;border-bottom:1px solid var(--rule);background:linear-gradient(180deg,#e4ebe4,var(--paper))}
.topo{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.topo path{fill:none;stroke:var(--contour);stroke-width:1.2}
.mast-in{position:relative;max-width:1040px;margin:0 auto;padding:64px 28px 40px}
.kicker{font:600 13px var(--mono-f);letter-spacing:.14em;text-transform:uppercase;color:var(--mono)}
h1{font:400 clamp(34px,5vw,56px)/1.06 var(--serif);margin:14px 0 18px;max-width:900px;letter-spacing:-.01em}
.dek{font-size:18.5px;color:var(--ink2);max-width:780px;margin:0 0 26px}
.spec{display:grid;grid-template-columns:150px 1fr;gap:6px 18px;margin:0;max-width:860px;font-size:14.5px;border-top:1px solid var(--rule);padding-top:16px}
.spec dt{font:600 12px var(--mono-f);text-transform:uppercase;letter-spacing:.08em;color:var(--ink3);padding-top:3px}.spec dd{margin:0}
.layout{display:grid;grid-template-columns:220px minmax(0,1fr);gap:40px;max-width:1240px;margin:0 auto;padding:36px 28px 80px}
.rail{position:sticky;top:20px;align-self:start;font-size:14px}.rail .back{display:inline-block;margin-bottom:16px;font-weight:700;text-decoration:none;color:var(--ink)}
.rail ol{list-style:none;margin:0;padding:0;border-left:2px solid var(--rule)}.rail li a{display:flex;gap:10px;padding:5px 12px;color:var(--ink2);text-decoration:none}
.rail li a:hover{color:var(--mono)}.rn{font:600 12px var(--mono-f);color:var(--ink3);padding-top:2px}
.article{max-width:760px}.article section{margin:0 0 54px}
h2{font:400 34px/1.15 var(--serif);margin:0 0 16px;display:flex;gap:14px;align-items:baseline}
h2 .num{font:600 14px var(--mono-f);color:var(--mono);letter-spacing:.08em}
p{margin:0 0 15px}.small{font-size:14px;color:var(--ink3)}ul{padding-left:20px;margin:0 0 16px}li{margin:0 0 8px}
.tw{overflow-x:auto;margin:6px 0 18px}table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums;background:var(--card)}
th,td{border-bottom:1px solid var(--rule);padding:8px 10px;text-align:left;vertical-align:top}th{font:600 12px var(--mono-f);text-transform:uppercase;letter-spacing:.06em;color:var(--ink3);background:var(--sunk)}
table.score td b{color:var(--mono)}.up{color:var(--good);font-weight:700}
.callout{border-left:4px solid var(--mono);background:var(--card);padding:12px 16px;margin:18px 0;border-radius:0 8px 8px 0}.callout p{margin:0}
.plate{margin:24px 0 26px;background:var(--card);border:1px solid var(--rule);border-radius:12px;padding:14px 16px 12px}
.plate.wide{width:min(1040px,calc(100vw - 340px))}
.plate-meta{display:flex;gap:12px;align-items:baseline;margin-bottom:8px;font-weight:700}.pn{font:600 12px var(--mono-f);color:var(--mono);letter-spacing:.1em;text-transform:uppercase}
.plate svg{width:100%;height:auto;display:block}.caption{font-style:italic;color:var(--ink2);font-size:14px;margin-top:6px}
svg .lbl{font:13px var(--sans);fill:var(--ink)}svg .lbl.s{font-size:11.5px;fill:var(--ink2)}svg .lbl.b{font-weight:700}svg .lbl.w{fill:#fff;font-weight:700}
svg .contour{fill:none;stroke:var(--contour);stroke-width:1.3}svg .trail{fill:none;stroke:var(--ink);stroke-width:2.2}svg .trail.dashed{stroke-dasharray:6 6;stroke:var(--mono)}
svg .camp{stroke:var(--ink);stroke-width:2}svg .camp.base{fill:var(--base)}svg .camp.bf16{fill:var(--bf16)}svg .camp.mono{fill:var(--mono)}svg .camp.plain{fill:var(--card)}
svg .ahd{fill:var(--ink2)}svg .seg.ttft,svg .seg.bf16{fill:var(--bf16)}svg .seg.dec{fill:var(--rule2)}svg .seg.base{fill:var(--base)}svg .seg.mono{fill:var(--mono)}
svg .bar.bf16{fill:var(--bf16)}svg .bar.mono{fill:var(--mono)}svg .bar.base{fill:var(--base)}svg .axis{stroke:var(--ink2);stroke-width:1.2}svg .tick{stroke:var(--ink);stroke-width:2;stroke-dasharray:3 3}
svg .box{fill:var(--paper);stroke:var(--rule2);stroke-width:1.4}svg .box.pool{fill:#e6eee8;stroke:var(--ink2)}svg .box.mono{stroke:var(--mono)}
svg .edge{stroke:var(--ink2);stroke-width:1.6;fill:none}svg .edge.mono{stroke:var(--mono)}
svg .grid{stroke:var(--rule);stroke-width:1}svg .pt{stroke:var(--card);stroke-width:2}svg .pt.base{fill:var(--base)}svg .pt.mono{fill:var(--mono)}
svg .move{stroke:var(--ink2);stroke-width:1.5}svg .refpt{fill:var(--card);stroke:var(--ref);stroke-width:2}svg .refline{fill:none;stroke:var(--ref);stroke-width:1.2;stroke-dasharray:4 4}
svg .box.st-done{stroke:var(--good);stroke-width:2}svg .box.st-open{stroke:var(--warn);stroke-width:2}svg .box.st-todo{stroke:var(--rule2);stroke-dasharray:5 4}
svg .chip{font:600 11px var(--mono-f);text-transform:uppercase;letter-spacing:.08em}svg .c-done{fill:var(--good)}svg .c-open{fill:var(--warn)}svg .c-todo{fill:var(--ink3)}
.lang-toggle{position:fixed;top:14px;right:16px;z-index:10;display:flex;border:1px solid var(--rule2);border-radius:999px;overflow:hidden;background:var(--card)}
.lang-toggle button{border:0;background:transparent;padding:6px 12px;font:600 12px var(--mono-f);color:var(--ink2);cursor:pointer}
body[data-lang="en"] .lang-toggle button[data-set="en"],body[data-lang="zh"] .lang-toggle button[data-set="zh"]{background:var(--ink);color:var(--paper)}
.colophon{max-width:1040px;margin:0 auto;padding:22px 28px 60px;border-top:1px solid var(--rule);font-size:13px;color:var(--ink3)}
@media (max-width:980px){.layout{grid-template-columns:1fr}.rail{position:static}.rail ol{columns:2}.plate.wide{width:100%}.spec{grid-template-columns:1fr}}
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

FONTS = ("https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400"
         "&family=Reddit+Mono:wght@400;600&family=Noto+Sans+SC:wght@400;700&display=swap")

colophon = ('<footer class="colophon">' + P(
    f"Experiment 006 on jinnpan.com. Generated by {a(f'{BLOB}/scripts/make_page.py', '<code>scripts/make_page.py</code>')} from {a(DATA_URL, 'the archived data')}. Type: Instrument Serif, Atkinson Hyperlegible, Reddit Mono. Palette: survey paper, pine ink, slate for the recipe, raspberry for the monokernel.",
    f"jinnpan.com 实验 006。由 {a(f'{BLOB}/scripts/make_page.py', '<code>scripts/make_page.py</code>')} 从{a(DATA_URL, '归档数据')}生成。字体：Instrument Serif、Atkinson Hyperlegible、Reddit Mono。配色：测绘纸色、松绿墨色，蓝灰代表配方，覆盆子红代表 monokernel。") + "</footer>")

page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GLM-5.3 on MI355X: the kernel was fast, the score was not · Experiment 006</title>
<meta name="description" content="A decode monokernel for GLM-5.3 on MI355X: why a 28% faster decode did not move the InferenceX AgentX score, and how reading the recipe's FP8 KV cache turned it into +15-19% throughput at 3600 s.">
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
{s_result}{s_route}{s_metric}{s_attempt}{s_cause}{s_fix}{s_score}{s_gain}{s_quality}{s_refs}{s_up}{s_method}{s_data}{epilogue}
</main>
</div>
{colophon}
{JS}
</body>
</html>
"""
OUT.write_text(page)
print(f"wrote {OUT} ({len(page) / 1e3:.0f} KB)")
