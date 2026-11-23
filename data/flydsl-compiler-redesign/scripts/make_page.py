"""Assemble the Experiment 008 page from computed numbers, charts and static plates."""

import json
import math
import re
from pathlib import Path

import build_blog as B
from page_css import CSS, JS
from page_plates import PLATE_DECISIONS, PLATE_PIPELINES


def isa_pair(label, sid, kernel_hint=""):
    d = B.arm(label)
    ks = d["isa"].get(sid) or {}
    for k, sides in ks.items():
        if kernel_hint in k and sides.get("stock") and sides.get("cand"):
            return sides["stock"], sides["cand"]
    return None, None


def big_pair(label):
    d = B.arm(label)
    best = None
    for ks in d["isa"].values():
        for sides in ks.values():
            a, b = sides.get("stock"), sides.get("cand")
            if a and b and (best is None or a["insts"] > best[0]["insts"]):
                best = (a, b)
    return best


def spill_note(label):
    """'spills 0→12' style ISA delta at the largest dumped shape."""
    d = B.arm(label)
    best = None
    for sid, ks in d["isa"].items():
        for k, sides in ks.items():
            a, b = sides.get("stock"), sides.get("cand")
            if not (a and b):
                continue
            key = a["insts"]
            if best is None or key > best[0]:
                best = (key, sid, a, b)
    if not best:
        return "—", "—"
    _, sid, a, b = best
    va, vb = a["meta"].get("vgpr_count"), b["meta"].get("vgpr_count")
    sa, sb = a["meta"].get("vgpr_spill_count"), b["meta"].get("vgpr_spill_count")
    same = a["order_sha"] == b["order_sha"]
    if same:
        return "ISA identical", "ISA 完全相同"
    en = f"VGPR {va}→{vb}, spills {sa}→{sb}"
    return en, f"VGPR {va}→{vb}，spill {sa}→{sb}"


def render():
    r = {l: B.ratio(l) for l in B.SUMMARY}
    f02, n02 = B.tests("x02-nosched")
    f04, n04 = B.tests("x04-nosched")
    x10f0 = B.shape("x10-agprllvm", "f0")
    x10p1 = B.shape("x10-agprllvm", "p1")
    s_f0, c_f0 = isa_pair("x10-agprllvm", "f0")
    h08 = B.arm("h08-stratmaxilp")
    h08b = B.arm("h08b-stratmaxilp")
    x13 = B.arm("x13-nogroup")
    x08 = B.arm("x08-noprio")
    x09 = B.arm("x09-postmischedon")

    # Per-shape oracle on preshuffle-a8 over the arms that only move D5 decisions.
    pre_arms = ["x01-nosched", "x14-nogroup", "h10-stratmaxilp", "h11-mischedoff"]
    o_point, o_strict, o_n = B.oracle(pre_arms)

    m = MINING
    rv = m["reviews"]
    src = rv["speedup_source_share_legit"]
    cmds = m["agent_commands"]
    cost = m["agent_cost_per_record"]
    n_tr = cmds["traces"]
    isa_share = cmds["traces_that_inspected_ir_or_isa"] / n_tr
    cc_share = cmds["traces_that_cleared_flydsl_cache"] / n_tr
    by = cmds["by_purpose"]
    nc = cmds["commands"]

    fp8 = next(c for c in B.COMPILE if c["kernel"] == "fp8_gemm_4wave")
    fa = next(c for c in B.COMPILE if c["kernel"] == "flash_attn_gfx950")

    def ok(label):
        d = B.arm(label)
        return d is not None and d["verdict"] == "scored"

    # ---------------- Plate III: ablation chart
    def row(label, en, zh, cls):
        d = B.arm(label)
        if d is None:
            return {"en": en, "zh": zh, "value": None, "note_en": "pending", "note_zh": "运行中"}
        if d["verdict"] != "scored":
            f, n = B.tests(label)
            if f:
                return {"en": en, "zh": zh, "value": None, "note_en": f"wrong results: {f} of {n} tests fail",
                        "note_zh": f"结果错误：{n} 个测试里 {f} 个失败"}
            if d["verdict"] == "unscorable":
                return {"en": en, "zh": zh, "value": None, "note_en": "stock itself failed",
                        "note_zh": "stock 自己出错"}
            return {"en": en, "zh": zh, "value": None, "note_en": "one wrong output",
                    "note_zh": "一次输出错误"}
        return {"en": en, "zh": zh, "value": d["ratio"], "ci": d["ci95"], "cls": cls}

    abl_rows = [
        row("x01-nosched", "preshuffle-a8 · fences + groups", "preshuffle-a8 · fence + group", "d5"),
        row("x14-nogroup", "preshuffle-a8 · groups only", "preshuffle-a8 · 只去 group", "d5"),
        row("x03-nosched", "mx-preshuffle-a8w8 · fences + groups", "mx-preshuffle-a8w8 · fence + group", "d5"),
        row("x05-nosched", "swa-gqa · fences + groups", "swa-gqa · fence + group", "d5"),
        row("x02-nosched", "a16w16 · fences", "a16w16 · fence", "d5"),
        row("x02b-nosched", "a16w16 · fences (repeat)", "a16w16 · fence（重复）", "d5"),
        row("x04-nosched", "flash-fwd-dense · fences + groups", "flash-fwd-dense · fence + group", "d5"),
        row("x13-nogroup", "flash-fwd-dense · groups only", "flash-fwd-dense · 只去 group", "d5"),
        row("x06-noprio", "fp8-rowscale-8wave · s_setprio", "fp8-rowscale-8wave · s_setprio", "prio"),
        row("x07-noprio", "mxfp8-8wave · s_setprio", "mxfp8-8wave · s_setprio", "prio"),
        row("x08-noprio", "flash-fwd-dense · s_setprio", "flash-fwd-dense · s_setprio", "prio"),
        row("x09-postmischedon", "flash-fwd-dense · post-RA sched back on", "flash-fwd-dense · 打开 post-RA 调度", "d5"),
        {"en": "fp8-rowscale-4wave · AGPR pin (16384³)", "zh": "fp8-rowscale-4wave · AGPR pin（16384³）",
         "value": x10f0["ratio"], "ci": x10f0["ci95"], "cls": "d4"},
        {"en": "fp8-rowscale-4wave · AGPR pin (8192³)", "zh": "fp8-rowscale-4wave · AGPR pin（8192³）",
         "value": x10p1["ratio"], "ci": x10p1["ci95"], "cls": "d4"},
        row("x11-occdefault", "fp8-rowscale-4wave · occupancy hints", "fp8-rowscale-4wave · occupancy 提示", "d4"),
        row("x12-nosched", "rmsnorm-fwd · null control", "rmsnorm-fwd · 空白对照", "ctrl"),
        row("x15-null", "a16w16 · null control", "a16w16 · 空白对照", "ctrl"),
    ]
    plate3 = B.bar_chart(abl_rows, 0.80, 1.10, title_id="p3")

    head_rows = [
        row("h01-stratmaxmem", "rmsnorm-fwd · max-memory-clause", "rmsnorm-fwd · max-memory-clause", "head"),
        row("h02-stratmaxilp", "rmsnorm-fwd · max-ilp", "rmsnorm-fwd · max-ilp", "head"),
        row("h12-mischedoff", "rmsnorm-fwd · LLVM scheduler off", "rmsnorm-fwd · 关掉 LLVM 调度器", "off"),
        row("h03-stratmaxmem", "layernorm-fwd · max-memory-clause", "layernorm-fwd · max-memory-clause", "head"),
        row("h04-stratmaxilp", "layernorm-fwd · max-ilp", "layernorm-fwd · max-ilp", "head"),
        row("h05-stratmaxmem", "softmax-fwd · max-memory-clause", "softmax-fwd · max-memory-clause", "head"),
        row("h06-stratmaxilp", "softmax-fwd · max-ilp", "softmax-fwd · max-ilp", "head"),
        row("h07-stratmaxmem", "moe.reduce · max-memory-clause", "moe.reduce · max-memory-clause", "head"),
        row("h09-stratmaxmem", "moe gemm1 · max-memory-clause", "moe gemm1 · max-memory-clause", "head"),
        row("h08-stratmaxilp", "moe gemm1 · max-ilp", "moe gemm1 · max-ilp", "head"),
        row("h08b-stratmaxilp", "moe gemm1 · max-ilp (repeat)", "moe gemm1 · max-ilp（重复）", "head"),
        row("h10-stratmaxilp", "preshuffle-a8 · max-ilp", "preshuffle-a8 · max-ilp", "head"),
        row("h11-mischedoff", "preshuffle-a8 · LLVM scheduler off", "preshuffle-a8 · 关掉 LLVM 调度器", "off"),
    ]
    plate4 = B.bar_chart(head_rows, 0.85, 1.05, title_id="p4")

    plate5, ps_shapes = B.per_shape_plot([("x01-nosched", "s1", -4.5), ("x14-nogroup", "s2", -1.5),
                                          ("h10-stratmaxilp", "s3", 1.5), ("h11-mischedoff", "s4", 4.5)])

    src_items = [
        ("vectorize · memory path", "向量化 · 访存路径", src.get("vectorize_or_memory", 0), "d5"),
        ("dispatch · tiling", "dispatch · tile 选择", src.get("dispatch_or_tiling", 0), "d5"),
        ("algorithm rewrite", "算法重写", src.get("algorithm_rewrite", 0), "d5"),
        ("split-K · grid fill", "split-K · 填满 grid", src.get("split_k_or_grid_fill", 0), "d5"),
        ("schedule · sync knob", "调度 · 同步旋钮", src.get("schedule_or_sync_knob", 0), "prio"),
        ("register · occupancy knob", "寄存器 · occupancy 旋钮", src.get("register_or_occupancy_knob", 0), "d4"),
    ]
    plate6a = B.hbar_pairs(src_items, title_id="p6a", maxv=1.0)
    cmd_items = [
        ("edit code", "改代码", by.get("edit_code", 0) / nc, "d5"),
        ("read code", "读代码", by.get("read_code", 0) / nc, "d5"),
        ("own timing · profiling", "自己测时 · profile", by.get("own_timing_or_profile", 0) / nc, "head"),
        ("inspect IR · ISA", "看 IR · ISA", by.get("inspect_ir_or_isa", 0) / nc, "prio"),
        ("run ./evaluate", "跑 ./evaluate", by.get("evaluate", 0) / nc, "head"),
        ("clear the compile cache", "清编译缓存", by.get("cache_clear", 0) / nc, "off"),
        ("run tests", "跑测试", by.get("run_tests", 0) / nc, "head"),
    ]
    plate6b = B.hbar_pairs(cmd_items, title_id="p6b", fmt="{:.1%}", maxv=0.30)
    plate7 = B.compile_chart()

    def res(label):
        d = B.arm(label)
        if d is None:
            return '<td class="v">pending</td>'
        if d["verdict"] != "scored":
            f, n = B.tests(label)
            txt = f"wrong · {f}/{n} tests" if f else ("stock failed" if d["verdict"] == "unscorable" else "wrong output")
            return f'<td class="bad">{txt}</td>'
        cls = "good" if d["ci95"][0] > 1 + (d["noise_floor"] or 0) else ("bad" if d["ci95"][1] < 1 - (d["noise_floor"] or 0) else "v")
        return f'<td class="{cls}">{d["ratio"]:.3f}× {B.ci(label)}</td>'

    def isa_cell(label, lang):
        en, zh = spill_note(label) if B.arm(label) else ("—", "—")
        return en if lang == "en" else zh

    def trow(label, kernel, what_en, what_zh):
        return (f'<tr><td><code>{kernel}</code></td><td><span lang="en">{what_en}</span><span lang="zh">{what_zh}</span></td>'
                f'{res(label)}<td class="v"><span lang="en">{isa_cell(label, "en")}</span><span lang="zh">{isa_cell(label, "zh")}</span></td></tr>')

    abl_table = "\n".join([
        trow("x01-nosched", "gemm.preshuffle-a8", "schedule fences + interleave groups", "调度 fence + 交错 group"),
        trow("x14-nogroup", "gemm.preshuffle-a8", "interleave groups only", "只去交错 group"),
        trow("x03-nosched", "gemm.mx-preshuffle-a8w8", "schedule fences + interleave groups", "调度 fence + 交错 group"),
        trow("x05-nosched", "attention.swa-gqa", "schedule fences + interleave groups", "调度 fence + 交错 group"),
        trow("x02-nosched", "gemm.a16w16", "schedule fences", "调度 fence"),
        trow("x02b-nosched", "gemm.a16w16", "schedule fences (repeat)", "调度 fence（重复）"),
        trow("x04-nosched", "attention.flash-fwd-dense", "schedule fences + interleave groups", "调度 fence + 交错 group"),
        trow("x13-nogroup", "attention.flash-fwd-dense", "interleave groups only", "只去交错 group"),
        trow("x06-noprio", "gemm.fp8-rowscale-8wave", "<code>s_setprio</code>", "<code>s_setprio</code>"),
        trow("x07-noprio", "gemm.mxfp8-8wave", "<code>s_setprio</code>", "<code>s_setprio</code>"),
        trow("x08-noprio", "attention.flash-fwd-dense", "<code>s_setprio</code>", "<code>s_setprio</code>"),
        trow("x09-postmischedon", "attention.flash-fwd-dense", "the disabled post-RA scheduler", "被关掉的 post-RA 调度器"),
        trow("x10-agprllvm", "gemm.fp8-rowscale-4wave", "AGPR accumulator pin (inline asm)", "AGPR 累加器 pin（inline asm）"),
        trow("x11-occdefault", "gemm.fp8-rowscale-4wave", "occupancy hints", "occupancy 提示"),
        trow("x12-nosched", "norm.rmsnorm-fwd", "nothing to remove (null control)", "没有可去的（空白对照）"),
    ])
    head_table = "\n".join([
        trow("h01-stratmaxmem", "norm.rmsnorm-fwd", "<code>max-memory-clause</code>", "<code>max-memory-clause</code>"),
        trow("h02-stratmaxilp", "norm.rmsnorm-fwd", "<code>max-ilp</code>", "<code>max-ilp</code>"),
        trow("h12-mischedoff", "norm.rmsnorm-fwd", "machine schedulers off", "关掉 machine scheduler"),
        trow("h03-stratmaxmem", "norm.layernorm-fwd", "<code>max-memory-clause</code>", "<code>max-memory-clause</code>"),
        trow("h04-stratmaxilp", "norm.layernorm-fwd", "<code>max-ilp</code>", "<code>max-ilp</code>"),
        trow("h05-stratmaxmem", "norm.softmax-fwd", "<code>max-memory-clause</code>", "<code>max-memory-clause</code>"),
        trow("h06-stratmaxilp", "norm.softmax-fwd", "<code>max-ilp</code>", "<code>max-ilp</code>"),
        trow("h07-stratmaxmem", "moe.reduce", "<code>max-memory-clause</code>", "<code>max-memory-clause</code>"),
        trow("h09-stratmaxmem", "moe.2stage-gemm1", "<code>max-memory-clause</code>", "<code>max-memory-clause</code>"),
        trow("h08-stratmaxilp", "moe.2stage-gemm1", "<code>max-ilp</code>", "<code>max-ilp</code>"),
        trow("h08b-stratmaxilp", "moe.2stage-gemm1", "<code>max-ilp</code>, repeated", "<code>max-ilp</code>，重复一次"),
        trow("h10-stratmaxilp", "gemm.preshuffle-a8", "<code>max-ilp</code>", "<code>max-ilp</code>"),
        trow("h11-mischedoff", "gemm.preshuffle-a8", "machine schedulers off", "关掉 machine scheduler"),
    ])

    compile_rows = []
    names = {"vector_add": "examples/01-vectorAdd", "preshuffle_fp8": "preshuffle FP8 GEMM",
             "fp8_gemm_4wave": "fp8_gemm_4wave 256×256", "flash_attn_gfx950": "flash_attn_gfx950 D128"}
    for c in B.COMPILE:
        compile_rows.append(
            f'<tr><td>{names[c["kernel"]]}</td><td class="v">{float(c["cold_compile_s"]):.2f} s</td>'
            f'<td class="v">{float(c["trace_s"]) + float(c["verify_print_s"]):.2f} s</td>'
            f'<td class="v">{float(c["mlir_passes_1_14_s"]):.2f} s</td><td class="v">{float(c["llvm_backend_s"]):.2f} s</td>'
            f'<td class="v">{float(c["warm_inmem_hit_call_s"]) * 1000:.0f} ms</td>'
            f'<td class="v">{int(c["origin_ir_ops"]):,} → {int(c["after_layout_lowering_ops"]):,}</td></tr>')
    compile_table = "\n".join(compile_rows)

    # helpers for prose
    def pct_slower(label):
        v = r.get(label)
        return f"{(1 - v) * 100:.0f}%" if v else "—"

    ablated_scored = [l for l in ("x01-nosched", "x03-nosched", "x05-nosched", "x06-noprio", "x07-noprio") if r.get(l)]
    slow_lo = min(1 - r[l] for l in ablated_scored) * 100
    slow_hi = max(1 - r[l] for l in ablated_scored) * 100
    n_slow = len(ablated_scored) + 1  # the 4-wave GEMM, counted at the two shapes its AGPR arm touches
    assert slow_lo <= (1 - x10p1["ratio"]) * 100 and (1 - x10f0["ratio"]) * 100 <= slow_hi

    pending = [l for l in ("x08-noprio", "x09-postmischedon", "x13-nogroup", "h08b-stratmaxilp") if B.arm(l) is None]

    def flash_sentence(lang):
        parts = []
        if x13 is not None:
            if x13["verdict"] == "scored":
                parts.append((f"Keeping the full fences and dropping only the interleave groups ran at {x13['ratio']:.3f}× {B.ci('x13-nogroup')} and stayed correct, so it is the fences, not the groups, that hold the answer.",
                              f"保留全部 fence、只去掉交错 group 时，速度是 stock 的 {x13['ratio']:.3f}× {B.ci('x13-nogroup')}， 结果也正确。 所以撑住正确性的是 fence， 不是 group。"))
            else:
                f, n = B.tests("x13-nogroup")
                parts.append((f"Dropping only the interleave groups and keeping the fences also broke it ({x13['verdict']}{', ' + str(f) + ' of ' + str(n) + ' tests' if f else ''}).",
                              f"只去交错 group、保留 fence， 也会出错（{x13['verdict']}{'，' + str(n) + ' 个测试里 ' + str(f) + ' 个失败' if f else ''}）。"))
        return " ".join(p[0] if lang == "en" else p[1] for p in parts)

    def h08_sentence(lang):
        if h08b is not None and h08b["verdict"] == "unscorable":
            return ("We repeated the arm, and the repeat caught the <i>stock</i> kernel returning a wrong output (cosine 0.977), so this kernel has an intermittent correctness problem of its own, most likely a race that we did not root-cause, and the knob cannot be blamed for it. It is a lesson of a different kind: a production kernel whose wrong answers the evaluator catches only some of the time is exactly the case a static race check exists for."
                    if lang == "en" else
                    "我们重复了这个 arm， 结果重复实验抓到了 <i>stock</i> kernel 自己返回错误结果（余弦 0.977）。 所以这个 kernel 本身就有间歇性的正确性问题， 很可能是一个我们没有定位的竞争， 不能怪到这个旋钮头上。 这是另一种教训： 一个生产 kernel 的错误结果只能被评测器偶尔抓到， 这正是静态竞争检查要解决的情况。")
        if h08b is None:
            return ("A repeat of this arm is still running." if lang == "en" else "这个 arm 的重复实验还在运行。")
        if h08b["verdict"] == "scored":
            return (f"A repeat of the same arm scored as correct ({h08b['ratio']:.3f}×), so the failure is intermittent." if lang == "en"
                    else f"同一个 arm 重复一次时结果正确（{h08b['ratio']:.3f}×）， 说明失败是间歇性的。")
        return (f"A repeat failed the same way ({h08b['verdict']})." if lang == "en" else f"重复一次也同样失败（{h08b['verdict']}）。")

    def a16_repeat():
        fa, fb = B.FAILURES.get("x02-nosched"), B.FAILURES.get("x02b-nosched")
        if not (fa and fb):
            return "", ""
        same = fa["failed_tests"] == fb["failed_tests"]
        ma, mb = (int(f["mismatched"][0].split(" / ")[0]) for f in (fa, fb))
        total = int(fa["mismatched"][0].split(" / ")[1].split()[0])
        en = (f"A repeat of the A16W16 arm in a new job failed {'the same' if same else 'a different'} test with different wrong elements ({ma:,} and {mb:,} of {total:,}), the signature of a race rather than of rounding.")
        zh = (f"在新的作业里重复 A16W16 这个 arm， 失败的是{'同一个' if same else '另一个'}测试， 但错的元素不一样（{total:,} 个里分别错了 {ma:,} 个和 {mb:,} 个）， 这是竞争的特征， 不是舍入误差。")
        x15 = B.arm("x15-null")
        if x15 is not None:
            f15, n15 = B.tests("x15-null")
            if x15["verdict"] == "scored" and not f15:
                en += f" With the hint shim installed and nothing removed, the same evaluator passed all of its tests (null control, {x15['ratio']:.3f}×)."
                zh += f" 装上提示 shim、什么都不去掉时， 同一个评测器下它的测试全部通过（空白对照， {x15['ratio']:.3f}×）。"
            else:
                en += f" The null control on the same kernel did not pass cleanly ({x15['verdict']}), so this kernel's tests are themselves unstable."
                zh += f" 同一个 kernel 上的空白对照也没有干净通过（{x15['verdict']}）， 说明这个 kernel 的测试本身就不稳定。"
        return en, zh

    def opt(d, fmt_en, fmt_zh):
        if d is None:
            return ("still running when this page was built", "本页生成时仍在运行")
        if d["verdict"] != "scored":
            return (f"wrong results ({d['verdict']})", f"结果错误（{d['verdict']}）")
        return (fmt_en.format(v=d["ratio"], ci=B.ci(d["label"])), fmt_zh.format(v=d["ratio"], ci=B.ci(d["label"])))

    x08_en, x08_zh = opt(x08, "{v:.3f}× {ci}", "{v:.3f}× {ci}")
    a16_repeat_en, a16_repeat_zh = a16_repeat()
    nop_s, nop_c = (p["s_nop"] for p in big_pair("x09-postmischedon"))
    x09_en, x09_zh = opt(x09, "{v:.3f}× {ci}", "{v:.3f}× {ci}")

    n_arms = len([l for l in B.SUMMARY])
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Experiment 008 · One Decision, One Decider: redesigning the FlyDSL compiler for agent-written kernels</title>
<meta name="description" content="Where FlyDSL's compiler repeats itself, a redesign in which every kernel decision has exactly one owner, and {n_arms} overlay experiments on MI355X that test its premises: taking explicit decisions away, untuned LLVM knobs, and what 827 agent runs actually did.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Epilogue:wght@500;700;800&family=IBM+Plex+Serif:ital,wght@0,400;0,500;0,600;1,400&family=Victor+Mono:ital,wght@0,400;0,600;1,400&family=Noto+Serif+SC:wght@400;600&display=swap" rel="stylesheet">
<style>{CSS}</style>
</head>
<body data-lang="en">
<div class="lang-toggle" role="group" aria-label="Language"><button type="button" data-set="en" aria-label="English">EN</button><button type="button" data-set="zh" aria-label="中文">中文</button></div>
<div class="wrap">
<nav class="rail" aria-label="Contents">
  <a class="back" href="/sources/"><span lang="en">← The Library</span><span lang="zh">← 返回资料库</span></a>
  <ol>
    <li><a href="#question"><span class="num">01</span><span lang="en">Question</span><span lang="zh">问题</span></a></li>
    <li><a href="#decisions"><span class="num">02</span><span lang="en">Five decisions</span><span lang="zh">五类决策</span></a></li>
    <li><a href="#redundancy"><span class="num">03</span><span lang="en">Where it repeats itself</span><span lang="zh">冗余在哪里</span></a></li>
    <li><a href="#redesign"><span class="num">04</span><span lang="en">One decider each</span><span lang="zh">一个决策一个决策者</span></a></li>
    <li><a href="#method"><span class="num">05</span><span lang="en">Testing on the overlay</span><span lang="zh">在 overlay 上检验</span></a></li>
    <li><a href="#ablation"><span class="num">06</span><span lang="en">A · Take the decision away</span><span lang="zh">A · 拿走作者的决策</span></a></li>
    <li><a href="#headroom"><span class="num">07</span><span lang="en">B · Knobs nobody tuned</span><span lang="zh">B · 没人调过的旋钮</span></a></li>
    <li><a href="#agents"><span class="num">08</span><span lang="en">C · What agents did</span><span lang="zh">C · agent 做了什么</span></a></li>
    <li><a href="#overhead"><span class="num">09</span><span lang="en">D · The front-end tax</span><span lang="zh">D · 前端开销</span></a></li>
    <li><a href="#verdict"><span class="num">10</span><span lang="en">What it adds up to</span><span lang="zh">结论</span></a></li>
    <li><a href="#next"><span class="num">11</span><span lang="en">What would change our mind</span><span lang="zh">什么会推翻它</span></a></li>
    <li><a href="#reproduce"><span class="num">12</span><span lang="en">Reproduce</span><span lang="zh">复现</span></a></li>
    <li><a href="#epilogue"><span class="num">13</span><span lang="en">Epilogue</span><span lang="zh">结语</span></a></li>
  </ol>
</nav>
<div>
<header class="masthead">
  <div class="kicker">Experiment 008</div>
  <h1>One Decision <span class="slash">/</span> One Decider</h1>
  <p class="subtitle" lang="en">Rethinking the FlyDSL compiler for kernels that agents write: what to keep, what to throw away, and {n_arms} experiments on the overlay's MI355X fleet that could have proved the argument wrong.</p>
  <p class="subtitle" lang="zh">为 agent 写的 kernel 重新思考 FlyDSL 编译器： 该留下什么、 该扔掉什么， 以及在 overlay 的 MI355X 集群上做的 {n_arms} 组实验， 它们本来有机会推翻这个论证。</p>
  <div class="spec">
    <div><b lang="en">Hardware</b><b lang="zh">硬件</b>MI355X · gfx950 · overlay fleet</div>
    <div><b lang="en">FlyDSL under test</b><b lang="zh">被测 FlyDSL</b><a href="{B.V042.replace('/blob/', '/tree/')}">cb770e9a</a> (FlyBench v1 pin)</div>
    <div><b lang="en">Code reading</b><b lang="zh">读代码</b><a href="{B.MAIN.replace('/blob/', '/tree/')}">fe81c407</a> (v0.5.2)</div>
    <div><b lang="en">Timing</b><b lang="zh">计时</b>FlyBench v1 · ABBA · 6 pairs</div>
    <div><b lang="en">Agent corpus</b><b lang="zh">agent 语料</b>{m['runs']} runs · {m['receipts']} receipts · {n_tr} traces</div>
    <div><b lang="en">Data</b><b lang="zh">数据</b><a href="{B.DATA}">data/flydsl-compiler-redesign</a></div>
  </div>
</header>

<main class="article">

<section id="question">
  <h2><span class="n">01</span><span lang="en">The question</span><span lang="zh">问题</span></h2>
  <p lang="en">My mentor keeps asking me to think about the compiler chain. <a href="/sources/flydsl-compiler-in-the-loop.html">The last reading</a> asked where the compiler could join the kernel-optimization loop. This one asks the more basic question: if we were free to drop MLIR, and even LLVM, what should a kernel DSL whose main users are agents keep, and what should it throw away? A design argument costs nothing to write, so this post also runs the experiments that could have shown it to be wrong, on the same overlay fleet and with the same evaluator that score the agents.</p>
  <p lang="zh">mentor 一直让我想清楚 compiler chain。 <a href="/sources/flydsl-compiler-in-the-loop.html">上一篇</a>问的是编译器能从哪里进入 kernel 优化循环。 这一篇问一个更基础的问题： 如果可以扔掉 MLIR， 甚至扔掉 LLVM， 一个主要给 agent 用的 kernel DSL 应该留下什么、扔掉什么？ 设计论证写起来不花成本， 所以这一篇同时做了有机会推翻它的实验： 用的是给 agent 打分的同一个 overlay 集群和同一个评测器。</p>
  <div class="callout">
    <p lang="en"><b>Short answer.</b> The largest redundancy in FlyDSL's compiler is not code size. It is that two parties decide the same things: the author decides tiles, data movement and instruction interleaving in Python, then LLVM's heuristics decide register placement and instruction order again, then the author takes them back with fences, inline asm and LLVM patches. On MI355X, removing one explicit decision and letting LLVM decide cost {slow_lo:.0f}–{slow_hi:.0f}% speed on {n_slow} kernels and about 1% or less on one more, and on two kernels it made the output wrong, because their "scheduling" fences were holding correctness. Agents, meanwhile, found their speedups almost entirely in decisions the author already owns, and {isa_share:.0%} of their runs went digging in IR or ISA. So the redesign gives every decision exactly one owner, expresses synchronization as data dependencies so that every performance knob is safe to search, and makes the compiler report what it did. MLIR can go entirely; LLVM goes in two steps.</p>
    <p lang="zh"><b>简短回答。</b> FlyDSL 编译器最大的冗余不在代码量， 而在于同一件事有两方在决定： 作者在 Python 里决定 tile、数据搬运和指令交错， LLVM 的启发式再把寄存器放置和指令顺序决定一遍， 作者再用 fence、inline asm 和 LLVM 补丁把它们抢回来。 在 MI355X 上， 拿走作者的一个显式决策、交给 LLVM 去决定， 让 {n_slow} 个 kernel 的速度下降了 {slow_lo:.0f}–{slow_hi:.0f}%， 另一个 kernel 下降约 1% 或更少， 还让两个 kernel 算错了结果， 因为它们的"调度" fence 其实在撑着正确性。 与此同时， agent 的提速几乎全部来自作者本来就掌握的决策， 而 {isa_share:.0%} 的 agent 运行都去翻过 IR 或 ISA。 所以重新设计要做三件事： 每个决策只有一个决策者； 把同步表达成数据依赖， 让每个性能旋钮都可以放心搜索； 让编译器报告它做了什么。 MLIR 可以完全去掉， LLVM 分两步去掉。</p>
  </div>
</section>

<section id="decisions">
  <h2><span class="n">02</span><span lang="en">Five decisions, two deciders</span><span lang="zh">五类决策，两个决策者</span></h2>
  <p lang="en">Start from what is true of any AMD GPU kernel, whatever language wrote it. Its speed is set by five kinds of decisions. Three of them shape the work: how the problem is cut into a grid of tiles (D1), which lane moves which bytes through which memory (D2), and which matrix instruction does the arithmetic (D3). Two of them shape the instruction stream: which register file each value lives in, which caps how many waves fit (D4), and in what order instructions issue, where the waits go and which wave gets priority (D5).</p>
  <p lang="zh">先从任何 AMD GPU kernel 都成立的事实出发， 不管它是用什么语言写的。 它的速度由五类决策决定。 其中三类决定"工作怎么分"： 问题怎样切成 tile 组成的 grid（D1）、哪个 lane 通过哪一级存储搬哪些字节（D2）、用哪条矩阵指令做运算（D3）。 另外两类决定"指令流长什么样"： 每个值放在哪个寄存器堆， 这又决定能驻留几个 wave（D4）； 指令按什么顺序发射、 wait 插在哪里、 哪个 wave 优先（D5）。</p>
  <figure class="plate wide">
    <div class="plate-meta">Plate I · <span lang="en">five decisions and who makes them in FlyDSL today</span><span lang="zh">五类决策，以及今天在 FlyDSL 里由谁来做</span></div>
    {PLATE_DECISIONS}
    <p class="caption" lang="en">D1–D3 are the author's outright; the compiler only translates them. D4–D5 go to LLVM's heuristics, and authors who need a particular answer take it back through the red channel.</p>
    <p class="caption" lang="zh">D1–D3 完全由作者决定， 编译器只负责翻译。 D4–D5 交给 LLVM 的启发式， 需要特定结果的作者再通过红色通道把它们抢回来。</p>
  </figure>
  <p lang="en">The red channel is not rare. In the kernel corpus at <a href="{B.MAIN.replace('/blob/', '/tree/')}">fe81c407</a>, 34 of 180 kernel files call <code>sched_barrier</code> (257 times), 44 place waits by hand (186), 13 use inline asm and 12 set wave priority. Kernel comments say why in plain words: inline asm "so LLVM sees no memory op" and inserts no wait (<a href="{B.MAIN}/kernels/gemm/fp4_gemm_4wave.py#L82-L87"><code>fp4_gemm_4wave.py</code></a>), a fence because "the scheduler may hoist them above s_barrier: a race" (<a href="{B.MAIN}/kernels/mega_moe_tp/gemm.py#L948-L951"><code>mega_moe_tp/gemm.py</code></a>). Pinning one accumulator to AGPRs through LLVM took patches to its register allocator and instruction selector (<a href="https://github.com/ROCm/FlyDSL/pull/1368">#1368</a>), and the patch was reverted because it made another team's kernel spill (<a href="https://github.com/ROCm/FlyDSL/pull/1477">#1477</a>).</p>
  <p lang="zh">红色通道并不少见。 在 <a href="{B.MAIN.replace('/blob/', '/tree/')}">fe81c407</a> 的 kernel 语料里， 180 个文件中有 34 个调用 <code>sched_barrier</code>（共 257 次）， 44 个手工放 wait（186 次）， 13 个用 inline asm， 12 个设置 wave 优先级。 kernel 注释把原因写得很直白： 用 inline asm 是为了"让 LLVM 看不到访存"从而不插 wait（<a href="{B.MAIN}/kernels/gemm/fp4_gemm_4wave.py#L82-L87"><code>fp4_gemm_4wave.py</code></a>）； 加 fence 是因为"调度器可能把它们提到 s_barrier 之前： 这是竞争"（<a href="{B.MAIN}/kernels/mega_moe_tp/gemm.py#L948-L951"><code>mega_moe_tp/gemm.py</code></a>）。 想通过 LLVM 把一个累加器固定在 AGPR 上， 需要修改它的寄存器分配器和指令选择（<a href="https://github.com/ROCm/FlyDSL/pull/1368">#1368</a>）， 而这个补丁又因为让另一个团队的 kernel 出现 spill 被回滚（<a href="https://github.com/ROCm/FlyDSL/pull/1477">#1477</a>）。</p>
  <p lang="en">Why this matters more for an agent than for a person: an agent improves a kernel by local edits and measurement, which only works if a local edit has a local effect. A heuristic that re-decides globally breaks that premise. One example from our runs: removing the four <code>s_setprio</code> calls from the FP8 8-wave GEMM, an instruction whose only documented job is wave priority, changed the register allocation and created 6 VGPR spills. The reason is one line in LLVM's AMDGPU backend: <code>s_setprio</code> is a hard scheduling boundary (<a href="{B.LLVM}/llvm/lib/Target/AMDGPU/SIInstrInfo.cpp#L4554-L4559"><code>SIInstrInfo.cpp</code></a>). One knob, two meanings, and nothing in the source says so.</p>
  <p lang="zh">这件事对 agent 比对人更要紧。 agent 靠局部修改加测量来改进 kernel， 这只有在"局部修改只产生局部影响"时才成立。 一个会全局重新决定的启发式打破了这个前提。 我们实验里的一个例子： 从 FP8 8-wave GEMM 里去掉 4 处 <code>s_setprio</code>（文档上它只负责 wave 优先级）， 寄存器分配随之改变， 多出 6 个 VGPR spill。 原因是 LLVM AMDGPU 后端里的一行代码： <code>s_setprio</code> 是一个硬调度边界（<a href="{B.LLVM}/llvm/lib/Target/AMDGPU/SIInstrInfo.cpp#L4554-L4559"><code>SIInstrInfo.cpp</code></a>）。 一个旋钮有两种含义， 而源码里看不出来。</p>
</section>

<section id="redundancy">
  <h2><span class="n">03</span><span lang="en">Where the current compiler repeats itself</span><span lang="zh">现在的编译器在哪里重复自己</span></h2>
  <p lang="en">With the decision table in hand, the rest of the compiler's redundancy is easier to see. Each item below is a place where the same information is represented, or the same decision is made, more than once.</p>
  <p lang="zh">有了这张决策表， 编译器其余的冗余就容易看清了。 下面每一项， 都是同一份信息被表示了不止一次， 或者同一个决定被做了不止一次。</p>
  <h3 lang="en">Two languages, switched by an implicit rule</h3>
  <h3 lang="zh">两种语言，靠隐式规则切换</h3>
  <p lang="en">A FlyDSL kernel is Python that runs at trace time mixed with Python that becomes runtime control flow, and a 1,928-line AST rewriter decides which is which from the type of a value (<a href="{B.MAIN}/python/flydsl/compiler/ast_rewriter.py"><code>ast_rewriter.py</code></a>). The rule leaks: <a href="https://github.com/ROCm/FlyDSL/pull/1436">#1436</a> fixed a case where <code>if flag: return a</code> followed by <code>return b</code> returned <code>b</code> for every flag, without an error, and one production kernel had silently ignored an input since it was written. The repository's agent guide lists four restrictions that exist only because of this rewriting (<a href="{B.MAIN}/CLAUDE.md?plain=1#L192-L197"><code>CLAUDE.md</code></a>).</p>
  <p lang="zh">一个 FlyDSL kernel 里混着两种 Python： 一种在 trace 时执行， 一种会变成运行时控制流。 由一个 1928 行的 AST 改写器根据值的类型来判断哪段是哪种（<a href="{B.MAIN}/python/flydsl/compiler/ast_rewriter.py"><code>ast_rewriter.py</code></a>）。 这个规则会漏： <a href="https://github.com/ROCm/FlyDSL/pull/1436">#1436</a> 修了一个问题， <code>if flag: return a</code> 后面接 <code>return b</code> 时， 不管 flag 是什么都返回 <code>b</code>， 而且不报错； 一个生产 kernel 从写成那天起就悄悄忽略了一个输入。 仓库给 agent 的指南里列了四条限制， 它们只因为这种改写才存在（<a href="{B.MAIN}/CLAUDE.md?plain=1#L192-L197"><code>CLAUDE.md</code></a>）。</p>
  <h3 lang="en">Layout algebra, built as IR and then deleted</h3>
  <h3 lang="zh">layout algebra：先建成 IR，再删掉</h3>
  <p lang="en">Most layouts are fully static, so their algebra is compile-time arithmetic on integer tuples. FlyDSL nevertheless emits every step as an MLIR op whose type carries the layout, then folds it away. For <code>fp8_gemm_4wave</code> the traced module holds {int(fp8['origin_ir_ops']):,} ops; after canonicalization and layout lowering {int(fp8['after_layout_lowering_ops']):,} remain. Building and deleting the other 84% is most of the {float(fp8['trace_s']):.2f} s trace. The pure-Python reference implementation of the same algebra, <a href="https://github.com/NVIDIA/cutlass/tree/main/python/pycute">pycute</a>, is about 760 lines.</p>
  <p lang="zh">大多数 layout 完全是静态的， 所以它们的代数就是整数元组上的编译期算术。 但 FlyDSL 仍然把每一步都生成为一个 MLIR op， layout 记在 op 的类型里， 然后再折叠掉。 对 <code>fp8_gemm_4wave</code>， trace 出来的模块有 {int(fp8['origin_ir_ops']):,} 个 op， 经过 canonicalize 和 layout lowering 后只剩 {int(fp8['after_layout_lowering_ops']):,} 个。 建出来又删掉的这 84% 占了 {float(fp8['trace_s']):.2f} s trace 时间的大部分。 同一套代数的纯 Python 参考实现 <a href="https://github.com/NVIDIA/cutlass/tree/main/python/pycute">pycute</a> 大约 760 行。</p>
  <h3 lang="en">Registers modeled as memory, then promoted back</h3>
  <h3 lang="zh">寄存器先当内存建模，再提升回来</h3>
  <p lang="en">Register fragments are created as memrefs in a register address space, and two passes (about 900 lines) turn them back into vector values before LLVM sees them (<a href="{B.MAIN}/include/flydsl/Dialect/Fly/Transforms/Passes.td#L29-L40"><code>Passes.td</code></a>). The round trip copies C++ CuTe's habit of declaring register arrays; a compiler can represent a register tile as a value from the start.</p>
  <p lang="zh">寄存器 fragment 一开始被建成寄存器地址空间里的 memref， 再由两个 pass（约 900 行）在交给 LLVM 之前变回 vector 值（<a href="{B.MAIN}/include/flydsl/Dialect/Fly/Transforms/Passes.td#L29-L40"><code>Passes.td</code></a>）。 这个往返照搬了 C++ CuTe 声明寄存器数组的习惯； 编译器完全可以一开始就把寄存器 tile 表示成一个值。</p>
  <h3 lang="en">Every call re-traces; every process re-JITs the launcher</h3>
  <h3 lang="zh">每次调用都重新 trace，每个进程都重新 JIT launcher</h3>
  <p lang="en">A <code>@flyc.jit</code> call traces the whole launcher and kernel, prints the module and hashes the text before it can look up the cache (<a href="{B.MAIN}/python/flydsl/compiler/jit_function.py#L1203-L1213"><code>jit_function.py</code></a>). That keeps the cache key honest, but a cache hit on <code>fp8_gemm_4wave</code> still costs {float(fp8['warm_inmem_hit_call_s']):.2f} s. The host launcher itself is lowered through MLIR and JIT-compiled by an LLVM ExecutionEngine at O3 in every process, 22–33 ms per artifact, to do what a kernarg buffer and <code>hipModuleLaunchKernel</code> already do.</p>
  <p lang="zh"><code>@flyc.jit</code> 每次调用都要先 trace 整个 launcher 和 kernel、 打印模块、 对文本做哈希， 然后才能查缓存（<a href="{B.MAIN}/python/flydsl/compiler/jit_function.py#L1203-L1213"><code>jit_function.py</code></a>）。 这样缓存 key 很可靠， 但 <code>fp8_gemm_4wave</code> 命中缓存的一次调用仍要 {float(fp8['warm_inmem_hit_call_s']):.2f} s。 host 端的 launcher 本身也要经过 MLIR 降级， 再由 LLVM ExecutionEngine 在每个进程里以 O3 JIT 一次， 每个 artifact 22–33 ms， 而它要做的事情， 一个 kernarg 缓冲区加 <code>hipModuleLaunchKernel</code> 就能做到。</p>
  <h3 lang="en">Knobs that reach LLVM through four lanes, some of them dead</h3>
  <h3 lang="zh">旋钮通过四条通道到达 LLVM，有的通道是死的</h3>
  <p lang="en">The repository's own LLVM skill documents four lanes from Python to LLVM, one of which is silently discarded on AMD; a compile-hint dictionary that accepts misspelled keys and still changes the cache key; and an optimization-level variable that is validated and never read (<a href="{B.MAIN}/.claude/skills/llvm/SKILL.md?plain=1#L114-L131">llvm skill §3</a>, <a href="{B.MAIN}/.claude/skills/llvm/SKILL.md?plain=1#L250-L263">§8</a>). Our own experiment found another: in <code>fp8_gemm_4wave</code> the explicit occupancy hints produce byte-identical ISA to no hints at all.</p>
  <p lang="zh">仓库自己的 llvm skill 记录了从 Python 到 LLVM 的四条通道， 其中一条在 AMD 上会被悄悄丢弃； 一个接受拼错 key、却仍会改变缓存 key 的 compile-hint 字典； 一个会被校验、却从没被读取的优化级别变量（<a href="{B.MAIN}/.claude/skills/llvm/SKILL.md?plain=1#L114-L131">llvm skill §3</a>、<a href="{B.MAIN}/.claude/skills/llvm/SKILL.md?plain=1#L250-L263">§8</a>）。 我们自己的实验又发现一个： 在 <code>fp8_gemm_4wave</code> 里， 显式的 occupancy 提示和完全不写提示生成的 ISA 逐字节相同。</p>
  <figure class="plate wide">
    <div class="plate-meta">Plate II · <span lang="en">the same intent, four representations today; one owner per decision after the redesign</span><span lang="zh">今天同一个意图要经过四种表示；重新设计后每个决策只有一个主人</span></div>
    {PLATE_PIPELINES}
    <p class="caption" lang="en">Top: today's path, with the measured costs of section 09 and the channel by which authors take D4/D5 back. Bottom: Python only generates; a small hardware-explicit IR goes through passes that each make one decision deterministically and write it into a report.</p>
    <p class="caption" lang="zh">上： 今天的路径， 标出第 09 节测到的开销， 以及作者把 D4/D5 抢回来的通道。 下： Python 只负责生成； 一个小而显式的硬件 IR 经过若干 pass， 每个 pass 确定性地做一个决定， 并写进报告。</p>
  </figure>
</section>

<section id="redesign">
  <h2><span class="n">04</span><span lang="en">One decider per decision</span><span lang="zh">一个决策，一个决策者</span></h2>
  <p lang="en">The redesign follows from the table. For every decision, either the author makes it explicitly and the compiler honors and checks it, or the author delegates it and the compiler makes it deterministically and says what it chose. Never both. Five consequences:</p>
  <p lang="zh">重新设计直接从这张表推出来。 对每个决策， 要么作者显式做出、编译器照做并检查， 要么作者把它交出去、编译器确定性地做出并说明选了什么。 不允许两者同时存在。 由此有五个推论：</p>
  <ol class="list">
    <li lang="en"><b>Python becomes the meta-language only.</b> Python control flow always runs at compile time; runtime loops and branches are explicit constructs (<code>fx.loop</code>, <code>fx.cond</code>), and using a runtime value in a Python <code>if</code> is an error. This is <a href="https://docs.jax.dev/en/latest/control-flow.html">JAX's model</a>, and it deletes the AST rewriter and its four rules.</li>
    <li lang="zh"><b>Python 只做元语言。</b> Python 控制流永远在编译期执行； 运行时的循环和分支用显式构造（<code>fx.loop</code>、<code>fx.cond</code>）， 在 Python <code>if</code> 里用运行时值就报错。 这就是 <a href="https://docs.jax.dev/en/latest/control-flow.html">JAX 的模型</a>， 它让 AST 改写器和那四条规则一起消失。</li>
    <li lang="en"><b>Layout algebra is evaluated where it is static: in Python.</b> Every intermediate layout is a value one can print and test; only dynamic leaves become IR.</li>
    <li lang="zh"><b>layout algebra 在它静态的地方求值： 在 Python 里。</b> 每个中间 layout 都是可以打印、可以测试的值； 只有动态的部分才生成 IR。</li>
    <li lang="en"><b>A small hardware-explicit IR (call it KIR).</b> Values are typed as uniform (SGPR) or per-lane (VGPR); register tiles carry their thread-value layout and register class; asynchronous copies return tokens, and a wait names the token it waits for. Synchronization becomes a data dependency the compiler can see, so reordering can never break it.</li>
    <li lang="zh"><b>一个小而显式的硬件 IR（这里叫 KIR）。</b> 值的类型区分 uniform（SGPR）和 per-lane（VGPR）； 寄存器 tile 带着它的 thread-value layout 和寄存器类； 异步拷贝返回 token， wait 写明它等的是哪个 token。 同步变成编译器看得见的数据依赖， 于是重排永远不会破坏它。</li>
    <li lang="en"><b>A deterministic backend that reports.</b> Instruction selection by table (generated from AMD's <a href="https://gpuopen.com/machine-readable-isa/">machine-readable ISA XML</a>, which already covers CDNA3, CDNA4 and CDNA5); program order in explicit regions and a list scheduler with a published machine model in automatic ones; linear-scan allocation that honors pins; waits derived from tokens; hazards from tables. Each pass writes what it decided into a report the agent reads.</li>
    <li lang="zh"><b>一个确定性的、会报告的后端。</b> 指令选择查表（由 AMD 的<a href="https://gpuopen.com/machine-readable-isa/">机器可读 ISA XML</a> 生成， 它已经覆盖 CDNA3、CDNA4 和 CDNA5）； 显式区域按程序顺序， 自动区域用带公开机器模型的列表调度； 线性扫描分配， 遵守 pin； 由 token 推出 wait； 由表插入 hazard。 每个 pass 都把它的决定写进 agent 读的报告。</li>
    <li lang="en"><b>Only semantics-preserving knobs.</b> Because correctness depends only on data flow and tokens, schedule, pins, stage count and unroll can only change speed. Agents and autotuners can search them without a correctness review per point.</li>
    <li lang="zh"><b>只保留不改变语义的旋钮。</b> 正确性只依赖数据流和 token， 所以调度、pin、stage 数和 unroll 只会改变速度。 agent 和 autotuner 可以放心搜索， 不需要每个点都做一次正确性审查。</li>
  </ol>
  <p lang="en">What this means for the two big dependencies. MLIR can go entirely: its passes take only 11–12% of a large kernel's cold compile (section 09), and the costs it does carry are the op-by-op trace, the build-then-delete IR, and C++ rebuilds for every pass change. LLVM goes in two steps. First drop its heuristic code generation for explicit kernels and keep it only as an assembler, which is exactly how AMD's own hipBLASLt TensileLite builds its GEMMs: Python writes assembly and <code>amdclang++ -x assembler</code> encodes it (<a href="https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipblaslt/tensilelite/tensilelite/Toolchain/Component.py"><code>Toolchain/Component.py</code></a>). Then, once the new backend has passed the kernel corpus, generate an encoder from the ISA XML and drop LLVM for those kernels.</p>
  <p lang="zh">对两个大依赖意味着什么。 MLIR 可以完全去掉： 它的 pass 只占大 kernel 冷编译的 11–12%（第 09 节）， 它真正带来的成本是逐个 op 的 trace、 先建后删的 IR， 以及每改一个 pass 都要重编 C++。 LLVM 分两步去掉。 第一步， 在显式 kernel 上不再用它的启发式代码生成， 只把它当汇编器用。 AMD 自己的 hipBLASLt TensileLite 正是这样做 GEMM 的： Python 写汇编， <code>amdclang++ -x assembler</code> 负责编码（<a href="https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipblaslt/tensilelite/tensilelite/Toolchain/Component.py"><code>Toolchain/Component.py</code></a>）。 第二步， 等新后端通过 kernel 语料之后， 从 ISA XML 生成编码器， 在这些 kernel 上去掉 LLVM。</p>
</section>

<section id="method">
  <h2><span class="n">05</span><span lang="en">How we tested it on the overlay</span><span lang="zh">怎样在 overlay 上检验</span></h2>
  <p lang="en">The new backend does not exist yet, so the experiments test the premises it rests on, each with an outcome that would have counted against it:</p>
  <p lang="zh">新后端还不存在， 所以实验检验的是它所依赖的前提。 每个前提都有一个会对它不利的可能结果：</p>
  <ul>
    <li lang="en"><b>A.</b> The D4/D5 decisions authors make explicitly carry real speed, and LLVM's heuristics do not recover it when left alone. <i>Against it:</i> removing them changes nothing.</li>
    <li lang="zh"><b>A.</b> 作者显式做出的 D4/D5 决策带着实打实的速度， LLVM 的启发式在没人管时拿不回来。 <i>不利结果：</i> 去掉它们什么都没变。</li>
    <li lang="en"><b>B.</b> Where authors made no such decision, untuned but safe knobs leave headroom. <i>Against it:</i> no knob moves the time.</li>
    <li lang="zh"><b>B.</b> 作者没有做这类决策的地方， 那些没人调过、但安全的旋钮还留有空间。 <i>不利结果：</i> 没有哪个旋钮能改变时间。</li>
    <li lang="en"><b>C.</b> Agents' cost is dominated by what a compiler could show or check. <i>Against it:</i> agents rarely look at compiler output, and their wins come from compiler knobs.</li>
    <li lang="zh"><b>C.</b> agent 的成本主要花在编译器本可以展示或检查的事情上。 <i>不利结果：</i> agent 很少看编译器输出， 而且它们的提速来自编译器旋钮。</li>
    <li lang="en"><b>D.</b> The front end's cost is paid on every iteration. <i>Against it:</i> a cache hit is free.</li>
    <li lang="zh"><b>D.</b> 前端的开销在每次迭代都要付。 <i>不利结果：</i> 命中缓存不花时间。</li>
  </ul>
  <p lang="en"><b>Protocol.</b> Every arm in A and B is a patch to FlyDSL <code>cb770e9a</code>, scored by the FlyBench v1 evaluator, the same one that scores agent submissions: candidate against stock in fresh worker processes for each of six ABBA pairs, each cold call after a 512 MB cache-flushing read, plus warm blocks, over the public timed shapes and freshly drawn ones, with FlyDSL's own tests and a fresh-shape correctness oracle as a gate. A speed ratio below 1 means the arm is slower than the stock kernel. Hint arms install a few lines in <a href="{B.V042}/kernels/__init__.py"><code>kernels/__init__.py</code></a> that turn one family of hints into no-ops or edit the LLVM options; the AGPR and occupancy arms edit one line of the kernel. Each arm ran as one model-free job on one MI355X through the overlay queue, exactly as FlyBench evaluators do.</p>
  <p lang="zh"><b>协议。</b> A 和 B 的每个 arm 都是一个针对 FlyDSL <code>cb770e9a</code> 的补丁， 由 FlyBench v1 评测器打分， 也就是给 agent 提交打分的那一个： 候选和 stock 在新的 worker 进程里成对比较， 做 6 组 ABBA， 每次冷调用前先读一遍 512 MB 来清缓存， 另外还有热调用， 覆盖公开的计时 shape 和每次新抽取的 shape； FlyDSL 自己的测试和新 shape 上的正确性 oracle 是门禁。 速度比低于 1 表示这个 arm 比 stock 慢。 提示类 arm 在 <a href="{B.V042}/kernels/__init__.py"><code>kernels/__init__.py</code></a> 里加几行代码， 把某一类提示变成空操作， 或者修改 LLVM 选项； AGPR 和 occupancy 两个 arm 改 kernel 里的一行。 每个 arm 作为一个不调用模型的作业， 通过 overlay 队列在一张 MI355X 上运行， 和 FlyBench 评测作业完全一样。</p>
  <p lang="en"><b>Controls.</b> Each job also compiled the stock kernel and the arm at two shapes, the first timed shape and the largest, and compared the final ISA. Every arm changed the ISA except three: the occupancy arm (section 06) and two null controls, RMSNorm, which has no hints to remove, and the A16W16 GEMM with the shim installed and nothing removed. The null controls measured {r['x12-nosched']:.3f}× and {r['x15-null']:.3f}×; that is the noise of this setup, so effects under about half a percent are not resolved. The AGPR arm changes code only for the 256×256 tiles that two of its 19 timed shapes use, so we report it at those shapes.</p>
  <p lang="zh"><b>对照。</b> 每个作业还在两个 shape 上（第一个计时 shape 和最大的 shape）分别编译 stock kernel 和这个 arm， 并比较最终 ISA。 除了三个 arm， 其他 arm 都改变了 ISA。 这三个是 occupancy arm（见第 06 节）和两个空白对照： 一个是没有提示可去的 RMSNorm， 另一个是装了 shim、什么都没去掉的 A16W16 GEMM。 两个空白对照测得 {r['x12-nosched']:.3f}× 和 {r['x15-null']:.3f}×， 这就是这套测量的噪声， 所以小于约 0.5% 的效果分辨不出来。 AGPR 这个 arm 只会改变 256×256 tile 的代码， 而 19 个计时 shape 里只有 2 个用这种 tile， 所以我们在这两个 shape 上报告它。</p>
  <div class="callout note">
    <p lang="en"><b>Caveat.</b> The overlay places single-GPU jobs beside other jobs on the same node. Pairing inside one job on one GPU removes drift between the two sides, but not noise from neighbours; the intervals below include it. The agent corpus in C comes from earlier FlyBench runs on the same fleet, not from new agent jobs.</p>
    <p lang="zh"><b>注意。</b> overlay 会把单卡作业和别的作业放在同一个节点上。 在同一个作业、同一张卡上做配对， 能消除两侧之间的漂移， 但消除不了邻居带来的噪声； 下面的区间已经包含了它。 C 里的 agent 语料来自同一集群上之前的 FlyBench 运行， 不是新跑的 agent 作业。</p>
  </div>
</section>

<section id="ablation">
  <h2><span class="n">06</span><span lang="en">Experiment A · take the author's decision away</span><span lang="zh">实验 A · 拿走作者的决策</span></h2>
  <p lang="en">Each arm removes one explicit decision and hands it back to LLVM's default heuristics. If those heuristics were as good as the author, every bar below would sit on 1.0.</p>
  <p lang="zh">每个 arm 去掉一个显式决策， 把它交还给 LLVM 的默认启发式。 如果这些启发式和作者一样好， 下面每根条都应该落在 1.0 上。</p>
  <figure class="plate wide">
    <div class="plate-meta">Plate III · <span lang="en">speed relative to the stock kernel when one explicit decision is removed</span><span lang="zh">去掉一个显式决策后，相对 stock kernel 的速度</span></div>
    {plate3}
    <p class="legend"><span><i style="background:var(--ink2)"></i><span lang="en">D5 order</span><span lang="zh">D5 顺序</span></span><span><i style="background:var(--violet)"></i><span lang="en">D5 wave priority</span><span lang="zh">D5 wave 优先级</span></span><span><i style="background:var(--amber)"></i><span lang="en">D4 registers / occupancy</span><span lang="zh">D4 寄存器 / occupancy</span></span><span><i style="background:var(--hair)"></i><span lang="en">null control</span><span lang="zh">空白对照</span></span></p>
    <p class="caption" lang="en">Bars are the evaluator's geometric-mean speed over all timed and drawn shapes, whiskers its 95% interval over six ABBA pairs. The two AGPR rows are the only shapes that arm can affect. A cross means FlyDSL's own tests or the oracle rejected the output.</p>
    <p class="caption" lang="zh">条形是评测器在所有计时 shape 和抽取 shape 上的几何平均速度， 须线是 6 组 ABBA 上的 95% 区间。 两行 AGPR 是这个 arm 唯一能影响的 shape。 叉号表示 FlyDSL 自己的测试或 oracle 拒绝了输出。</p>
  </figure>
  <table>
    <tr><th lang="en">Kernel</th><th lang="zh">Kernel</th><th lang="en">Removed</th><th lang="zh">去掉的</th><th lang="en">Speed vs stock [95% CI]</th><th lang="zh">相对 stock 的速度 [95% CI]</th><th lang="en">ISA, largest dumped shape</th><th lang="zh">ISA（dump 的最大 shape）</th></tr>
    {abl_table}
  </table>
  <h3 lang="en">The decisions are worth 5–16%, and LLVM does not find them</h3>
  <h3 lang="zh">这些决策值 5–16%，而 LLVM 找不到它们</h3>
  <p lang="en">Removing the schedule hints cost the preshuffle GEMM {pct_slower('x01-nosched')} of its speed ({r['x01-nosched']:.3f}×), the MX preshuffle GEMM {pct_slower('x03-nosched')} and the sliding-window attention {pct_slower('x05-nosched')}. For the preshuffle GEMM the interleave groups carry all of it: dropping only them gives {r['x14-nogroup']:.3f}×, the same as dropping everything. Removing <code>s_setprio</code> cost the two 8-wave GEMMs {pct_slower('x06-noprio')} and {pct_slower('x07-noprio')}; the ISA shows why the first one is more than a priority change, with VGPR spills appearing where there were none. Letting LLVM allocate the FP8 4-wave GEMM's accumulators instead of pinning them cost {(1 - x10f0['ratio']) * 100:.0f}% at 16384³ and {(1 - x10p1['ratio']) * 100:.0f}% at 8192³: spills went from {s_f0['meta'].get('vgpr_spill_count')} to {c_f0['meta'].get('vgpr_spill_count')} and accumulator moves from {s_f0['v_accvgpr']:,} to {c_f0['v_accvgpr']:,}. Interesting in the other direction: even the pinned version spills {s_f0['meta'].get('vgpr_spill_count')} VGPRs at that size, which nobody had been told. Removing the same kernel's occupancy hints (<code>waves_per_eu</code>, <code>flat_work_group_size</code>) changed nothing: the ISA is byte-identical at both dumped shapes, so LLVM's default already matched them. A hint that does nothing does no harm, but no tool tells the author or the agent that it does nothing.</p>
  <p lang="zh">去掉调度提示后， preshuffle GEMM 的速度下降 {pct_slower('x01-nosched')}（{r['x01-nosched']:.3f}×）， MX preshuffle GEMM 下降 {pct_slower('x03-nosched')}， 滑窗 attention 下降 {pct_slower('x05-nosched')}。 对 preshuffle GEMM， 全部收益都来自交错 group： 只去掉 group 得到 {r['x14-nogroup']:.3f}×， 和全部去掉一样。 去掉 <code>s_setprio</code> 让两个 8-wave GEMM 分别下降 {pct_slower('x06-noprio')} 和 {pct_slower('x07-noprio')}； ISA 说明了为什么前者不只是优先级变化： 原来没有 spill 的地方出现了 VGPR spill。 让 LLVM 分配 FP8 4-wave GEMM 的累加器、而不是固定它们， 在 16384³ 上慢了 {(1 - x10f0['ratio']) * 100:.0f}%， 在 8192³ 上慢了 {(1 - x10p1['ratio']) * 100:.0f}%： spill 从 {s_f0['meta'].get('vgpr_spill_count')} 个增加到 {c_f0['meta'].get('vgpr_spill_count')} 个， 累加器搬运从 {s_f0['v_accvgpr']:,} 条增加到 {c_f0['v_accvgpr']:,} 条。 反过来也值得一提： 即使是固定后的版本， 在这个尺寸上也有 {s_f0['meta'].get('vgpr_spill_count')} 个 VGPR spill， 之前没有人知道。 去掉同一个 kernel 的 occupancy 提示（<code>waves_per_eu</code>、<code>flat_work_group_size</code>）什么都没改变： 两个 dump 的 shape 上 ISA 逐字节相同， 说明 LLVM 的默认值本来就和它们一致。 不起作用的提示没有害处， 但没有任何工具告诉作者或 agent 它不起作用。</p>
  <h3 lang="en">Two kernels' "scheduling" fences were holding correctness</h3>
  <h3 lang="zh">两个 kernel 的"调度" fence 其实在撑着正确性</h3>
  <p lang="en">Removing the schedule fences made the A16W16 GEMM fail {f02} of {n02} of FlyDSL's tests (a mismatched result) and the dense flash attention fail {f04} of {n04} (log-sum-exp off by up to 0.43, and NaN). {a16_repeat_en} A scheduling hint is supposed to change only speed. Here it changes the answer. We did not trace which instruction moved, but the kernels point to the cause: they order asynchronous copies and shared-memory reuse with hand-placed waits and barriers that LLVM does not tie to the loads they guard, and a FlyDSL comment elsewhere names exactly this hazard: "the scheduler may hoist them above s_barrier: a race" (<a href="{B.MAIN}/kernels/mega_moe_tp/gemm.py#L948-L951"><code>mega_moe_tp/gemm.py</code></a>). {flash_sentence('en')} This is the strongest result of the post, and it is not about speed: in today's stack an autotuner, or an agent, cannot safely toggle a scheduling knob, because some of them are correctness constraints in disguise.</p>
  <p lang="zh">去掉调度 fence 之后， A16W16 GEMM 在 FlyDSL 自己的 {n02} 个测试里失败了 {f02} 个（结果不一致）， dense flash attention 失败了 {n04} 个里的 {f04} 个（log-sum-exp 最多差 0.43， 还有 NaN）。 {a16_repeat_zh} 调度提示本应只影响速度， 这里却改变了答案。 我们没有追查具体是哪条指令被移动了， 但 kernel 本身指向了原因： 它们用手工放置的 wait 和 barrier 来安排异步拷贝和共享内存复用的顺序， 而 LLVM 不会把这些 wait 和 barrier 与它们所保护的 load 联系起来。 FlyDSL 另一处注释写的正是这种风险： "调度器可能把它们提到 s_barrier 之前： 这是竞争"（<a href="{B.MAIN}/kernels/mega_moe_tp/gemm.py#L948-L951"><code>mega_moe_tp/gemm.py</code></a>）。 {flash_sentence('zh')} 这是本文最强的结果， 而且它和速度无关： 在今天的工具链里， autotuner 或 agent 不能放心地切换调度旋钮， 因为其中一些其实是伪装起来的正确性约束。</p>
  <h3 lang="en">LLVM's free schedule is sometimes better, shape by shape</h3>
  <h3 lang="zh">按 shape 看，LLVM 的自由调度有时反而更好</h3>
  <p lang="en">The same preshuffle arm that is {pct_slower('x01-nosched')} slower overall is {(B.shape('x01-nosched', 'f0')['ratio'] - 1) * 100:.1f}% <i>faster</i> at 16384³ and {(B.shape('x01-nosched', 'p1')['ratio'] - 1) * 100:.1f}% faster at 8192³. The author's interleave template is a good fixed policy, not the best one for every shape. Plate V returns to this.</p>
  <p lang="zh">同一个 preshuffle arm， 整体慢了 {pct_slower('x01-nosched')}， 在 16384³ 上却<i>快</i>了 {(B.shape('x01-nosched', 'f0')['ratio'] - 1) * 100:.1f}%， 在 8192³ 上快了 {(B.shape('x01-nosched', 'p1')['ratio'] - 1) * 100:.1f}%。 作者的交错模板是一个不错的固定策略， 但不是每个 shape 上最好的。 Plate V 会回到这一点。</p>
  <p lang="en">The other two flash arms are small, and they point the same way. Removing its <code>s_setprio</code> gave {x08_en} overall but cost up to {(1 - x08['min_shape_ratio']) * 100:.1f}% on single shapes ({len(x08['regressions'])} shapes beyond their noise floor). The kernel's author had switched LLVM's post-RA scheduler off to protect the hand-tuned order (<a href="{B.V042}/kernels/attention/flash_attn_gfx950.py#L1027"><code>flash_attn_gfx950.py</code></a>); turning it back on gave {x09_en}, with no shape slower beyond noise and the best shape {(x09['max_shape_ratio'] - 1) * 100:.1f}% faster; the ISA has {nop_s - nop_c} fewer <code>s_nop</code>s ({nop_s} → {nop_c}). That gain is about the size of the kernel's {x09['noise_floor'] * 100:.1f}% noise floor, so FlyBench would not count it as an improvement. The author could not see whether that flag helps, and FlyDSL's own notes record that three sibling FMHA kernels disagree on it, with no comment (<a href="{B.V042}/.claude/skills/llvm/references/arch-gfx950.md#L147"><code>arch-gfx950.md</code></a>).</p>
  <p lang="zh">flash 的另外两个 arm 影响很小， 方向和上面一致。 去掉 <code>s_setprio</code> 后整体是 {x08_zh}， 但在单个 shape 上最多慢 {(1 - x08['min_shape_ratio']) * 100:.1f}%（{len(x08['regressions'])} 个 shape 超出噪声范围）。 这个 kernel 的作者为了保护手工调好的顺序， 关掉了 LLVM 的 post-RA 调度器（<a href="{B.V042}/kernels/attention/flash_attn_gfx950.py#L1027"><code>flash_attn_gfx950.py</code></a>）； 重新打开后得到 {x09_zh}， 没有 shape 超出噪声地变慢， 最好的 shape 快了 {(x09['max_shape_ratio'] - 1) * 100:.1f}%； ISA 里少了 {nop_s - nop_c} 条 <code>s_nop</code>（{nop_s} → {nop_c}）。 这个收益和这个 kernel {x09['noise_floor'] * 100:.1f}% 的噪声下限差不多大， 所以 FlyBench 不会把它算作提升。 作者看不到这个开关到底有没有用； FlyDSL 自己的笔记也写着， 三个同系列的 FMHA kernel 在这个开关上取值不一致， 而且没有注释（<a href="{B.V042}/.claude/skills/llvm/references/arch-gfx950.md#L147"><code>arch-gfx950.md</code></a>）。</p>
</section>

<section id="headroom">
  <h2><span class="n">07</span><span lang="en">Experiment B · knobs nobody tuned</span><span lang="zh">实验 B · 没人调过的旋钮</span></h2>
  <p lang="en">The normalization, softmax and MoE kernels carry no schedule hints at all: every D4/D5 decision is LLVM's default. Here we try LLVM's other scheduling strategies, and turning its machine schedulers off, which approximates program order.</p>
  <p lang="zh">normalization、softmax 和 MoE 这些 kernel 完全没有调度提示： 每个 D4/D5 决策都是 LLVM 的默认值。 这里我们试 LLVM 的其他调度策略， 以及关掉它的 machine scheduler（近似于按程序顺序）。</p>
  <figure class="plate wide">
    <div class="plate-meta">Plate IV · <span lang="en">speed relative to stock under one untuned LLVM knob</span><span lang="zh">换一个没人调过的 LLVM 旋钮后，相对 stock 的速度</span></div>
    {plate4}
    <p class="caption" lang="en">Teal: a different <code>amdgpu-sched-strategy</code>. Red: machine schedulers off. Crosses: the evaluator rejected an output; in the MoE repeat it was the stock kernel's. Most bars stay inside each kernel's measured noise floor.</p>
    <p class="caption" lang="zh">青色： 换一个 <code>amdgpu-sched-strategy</code>。 红色： 关掉 machine scheduler。 叉号： 评测器拒绝了某次输出； 在 MoE 的重复实验里， 出错的是 stock kernel。 大多数条都落在各自 kernel 测得的噪声范围以内。</p>
  </figure>
  <table>
    <tr><th lang="en">Kernel</th><th lang="zh">Kernel</th><th lang="en">Knob</th><th lang="zh">旋钮</th><th lang="en">Speed vs stock [95% CI]</th><th lang="zh">相对 stock 的速度 [95% CI]</th><th lang="en">ISA, largest dumped shape</th><th lang="zh">ISA（dump 的最大 shape）</th></tr>
    {head_table}
  </table>
  <p lang="en">Three readings. First, for kernels nobody scheduled by hand, LLVM's default is good: no strategy beat it beyond noise on the norms and the MoE reduce, and turning the machine scheduler off cost RMSNorm {pct_slower('h12-mischedoff')}. An automatic mode needs a real scheduler. Second, on the hand-scheduled preshuffle GEMM the opposite holds: with LLVM's machine schedulers off it ran at {r['h11-mischedoff']:.3f}× {B.ci('h11-mischedoff')}, faster than stock, and <code>max-ilp</code> gave {r['h10-stratmaxilp']:.3f}×. Neither is a clean win: {len(B.arm('h11-mischedoff')['regressions'])} and {len(B.arm('h10-stratmaxilp')['regressions'])} shapes got slower beyond noise (worst {B.arm('h11-mischedoff')['min_shape_ratio']:.2f}× and {B.arm('h10-stratmaxilp')['min_shape_ratio']:.2f}×), so FlyBench would reject both as submitted. The heuristic scheduler is costing this kernel time even with the author's hints in place, which is what an explicit, program-order mode predicts. Third, the MoE GEMM failed correctness on one timed call under <code>max-ilp</code> (cosine similarity 0.989 against a 0.99 bound). {h08_sentence('en')}</p>
  <p lang="zh">三点解读。 第一， 对没有手工调度的 kernel， LLVM 的默认值是好的： 在 norm 和 MoE reduce 上， 没有哪个策略超出噪声地赢过它， 关掉 machine scheduler 让 RMSNorm 慢了 {pct_slower('h12-mischedoff')}。 自动模式需要一个真正的调度器。 第二， 在手工调度的 preshuffle GEMM 上情况相反： 关掉 LLVM 的 machine scheduler 后， 速度是 {r['h11-mischedoff']:.3f}× {B.ci('h11-mischedoff')}， 比 stock 快； <code>max-ilp</code> 是 {r['h10-stratmaxilp']:.3f}×。 两者都不是干净的胜利： 分别有 {len(B.arm('h11-mischedoff')['regressions'])} 个和 {len(B.arm('h10-stratmaxilp')['regressions'])} 个 shape 超出噪声地变慢（最差 {B.arm('h11-mischedoff')['min_shape_ratio']:.2f}× 和 {B.arm('h10-stratmaxilp')['min_shape_ratio']:.2f}×）， 所以按提交标准 FlyBench 会拒绝这两个 arm。 即使有作者的提示， 启发式调度器仍在拖慢这个 kernel， 这正是显式、按程序顺序的模式所预言的。 第三， 在 <code>max-ilp</code> 下， MoE GEMM 的一次计时调用没有通过正确性检查（余弦相似度 0.989， 门限是 0.99）。 {h08_sentence('zh')}</p>
  <figure class="plate wide">
    <div class="plate-meta">Plate V · <span lang="en">preshuffle-a8, shape by shape: no single schedule wins everywhere</span><span lang="zh">preshuffle-a8 逐个 shape 看：没有哪种调度处处都赢</span></div>
    {plate5}
    <p class="legend"><span><i style="background:var(--ink2)"></i><span lang="en">no fences or groups</span><span lang="zh">去掉 fence 和 group</span></span><span><i style="background:var(--red)"></i><span lang="en">no groups</span><span lang="zh">去掉 group</span></span><span><i style="background:var(--teal)"></i>max-ilp</span><span><i style="background:var(--amber)"></i><span lang="en">LLVM schedulers off</span><span lang="zh">关掉 LLVM 调度器</span></span></p>
    <p class="caption" lang="en">Each dot is one public shape under one arm; 1.0 is the stock kernel. The spread is the headroom: picking the best schedule per shape from these four arms and stock gives {o_point:.3f}× on the {o_n} public shapes, or {o_strict:.3f}× if an arm is credited only where its interval clears 1.0, and then only its lower bound.</p>
    <p class="caption" lang="zh">每个点是一个公开 shape 在一个 arm 下的速度； 1.0 是 stock kernel。 这种分散就是空间： 在这四个 arm 和 stock 之间为每个 shape 选最好的调度， 在 {o_n} 个公开 shape 上得到 {o_point:.3f}×； 如果只在区间整体高于 1.0 时才计入某个 arm、 并且只计它的区间下限， 得到 {o_strict:.3f}×。</p>
  </figure>
  <p lang="en">That per-shape selection is a lower bound on what a search over safe schedule knobs could find in this one kernel, and it is not available today, for the reason A gave: in kernels whose fences carry correctness, the knob space is not safe to search.</p>
  <p lang="zh">这种按 shape 的选择， 是在这一个 kernel 里搜索安全调度旋钮所能找到的收益的下限。 而它今天用不上， 原因就是 A 说的： 在 fence 撑着正确性的 kernel 里， 这个旋钮空间不能放心搜索。</p>
</section>

<section id="agents">
  <h2><span class="n">08</span><span lang="en">Experiment C · what {cost['n']} agent runs actually did</span><span lang="zh">实验 C · {cost['n']} 次 agent 运行到底做了什么</span></h2>
  <p lang="en">The overlay has already run FlyBench mode A many times: an agent gets one FlyDSL kernel and a time budget and tries to make it faster without changing its results. We read all of it that survives: {m['runs']} runs, {m['receipts']} evaluator receipts, {rv['n']} independent reviews of the best patches, and {n_tr} command traces ({nc:,} commands). Only aggregates are published.</p>
  <p lang="zh">overlay 已经跑过很多次 FlyBench mode A： agent 拿到一个 FlyDSL kernel 和一段时间预算， 在不改变结果的前提下设法让它变快。 我们读了所有留存下来的记录： {m['runs']} 次 run、 {m['receipts']} 张评测回执、 {rv['n']} 份对最好补丁的独立 review， 以及 {n_tr} 份命令轨迹（共 {nc:,} 条命令）。 这里只发布汇总数据。</p>
  <figure class="plate wide">
    <div class="plate-meta">Plate VI · <span lang="en">where the reviewed speedups came from, and what agents spent their commands on</span><span lang="zh">经过 review 的提速来自哪里，以及 agent 把命令花在了什么上</span></div>
    <p class="caption" lang="en">Share of the {rv['verdicts'].get('legit', 0)} reviews judged legitimate whose summary credits each source (a patch can credit several).</p>
    <p class="caption" lang="zh">在 {rv['verdicts'].get('legit', 0)} 份判定为合法的 review 中， 总结里提到某一来源的比例（一个补丁可以有多个来源）。</p>
    {plate6a}
    <p class="caption" lang="en">Share of all {nc:,} traced commands by purpose (the remainder is mostly running scripts).</p>
    <p class="caption" lang="zh">全部 {nc:,} 条命令按用途的比例（其余大多是运行脚本）。</p>
    {plate6b}
  </figure>
  <p lang="en"><b>Where the wins are.</b> The legitimate speedups (median {rv['legit_reproduced_speedup_median']:.2f}× reproduced by the reviewer) come from vectorizing the memory path, choosing tiles per shape, splitting work to fill an under-occupied grid, and rewriting the algorithm: D1–D3, the decisions the author already owns. Schedule and register knobs appear in {src.get('schedule_or_sync_knob', 0):.0%} and {src.get('register_or_occupancy_knob', 0):.0%} of them. So the redesign must keep, not take away, the expressiveness agents use: Python metaprogramming, layout algebra, atoms. Its promise to agents is not a new source of speedups.</p>
  <p lang="zh"><b>提速在哪里。</b> 合法的提速（review 复现的中位数是 {rv['legit_reproduced_speedup_median']:.2f}×）来自访存路径向量化、 按 shape 选 tile、 拆分工作以填满占用不足的 grid， 以及重写算法： 都是 D1–D3， 作者本来就掌握的决策。 调度和寄存器旋钮分别只出现在 {src.get('schedule_or_sync_knob', 0):.0%} 和 {src.get('register_or_occupancy_knob', 0):.0%} 的 review 里。 所以重新设计必须保留、而不是拿走 agent 用到的表达能力： Python 元编程、 layout algebra、 atom。 它对 agent 的承诺不是一个新的提速来源。</p>
  <p lang="en"><b>Where the cost is.</b> A median record took {cost['median_prompt_tokens'] / 1e6:.1f}M prompt tokens, {cost['median_turns']} turns and {cost['median_wall_s'] / 3600:.1f} h. {isa_share:.0%} of runs went digging in IR or ISA at least once, and {cc_share:.0%} cleared the FlyDSL compile cache at least once ({by.get('cache_clear', 0):,} cache clears in total), the habit of agents that cannot tell whether what ran is what they wrote. The reviews' most common failure was a per-shape regression on shapes the agent did not time ({rv['failed_checklist_items'].get('per-shape regressions vs stock', 0)} of {rv['n']}); a few broke correctness in ways a compiler could catch statically: a missing barrier, an out-of-bounds read, a spill tier that cut one shape to 0.44×. Nine reviews mention host-side launcher caches that agents added to get around per-call tracing.</p>
  <p lang="zh"><b>成本在哪里。</b> 一个中位数 record 花了 {cost['median_prompt_tokens'] / 1e6:.1f}M prompt token、{cost['median_turns']} 轮、{cost['median_wall_s'] / 3600:.1f} 小时。 {isa_share:.0%} 的运行至少翻过一次 IR 或 ISA， {cc_share:.0%} 至少清过一次 FlyDSL 编译缓存（总共 {by.get('cache_clear', 0):,} 次）， 这是 agent 无法确认"跑的是不是自己写的那份"时的习惯。 review 里最常见的失败是 agent 没测过的 shape 出现回退（{rv['n']} 份里 {rv['failed_checklist_items'].get('per-shape regressions vs stock', 0)} 份）； 也有少数正确性问题是编译器可以静态发现的： 漏掉的 barrier、 越界读、 一个让某个 shape 掉到 0.44× 的 spill 档位。 有 9 份 review 提到 agent 为了绕开每次调用的 trace 而自己加了 host 端 launcher 缓存。</p>
  <div class="callout">
    <p lang="en"><b>What the agents need from the compiler</b> is mostly what the redesign's report and checks provide: resources, spills and occupancy per shape before a GPU run; the waits it inserted and why; a race and bounds check on shared memory; a cache that is keyed honestly and a launch path that does not re-trace. None of that needs MLIR or LLVM's heuristics; it needs a compiler that owns its decisions.</p>
    <p lang="zh"><b>agent 需要编译器提供的</b>， 大多正是重新设计里的报告和检查： 上 GPU 之前就知道每个 shape 的资源、spill 和 occupancy； 插了哪些 wait、为什么； 共享内存上的竞争和越界检查； 一个 key 可靠的缓存和一条不重新 trace 的 launch 路径。 这些都不需要 MLIR， 也不需要 LLVM 的启发式， 只需要一个对自己的决定负责的编译器。</p>
  </div>
</section>

<section id="overhead">
  <h2><span class="n">09</span><span lang="en">Experiment D · the front-end tax</span><span lang="zh">实验 D · 前端开销</span></h2>
  <p lang="en">Measured on the CPU with FlyDSL 0.5.2, compile only, target gfx950, three runs each within 1%.</p>
  <p lang="zh">在 CPU 上用 FlyDSL 0.5.2 测量， 只编译， 目标 gfx950， 每项跑 3 次， 波动在 1% 以内。</p>
  <figure class="plate wide">
    <div class="plate-meta">Plate VII · <span lang="en">where a cold compile goes, and what a cache hit still costs</span><span lang="zh">冷编译的时间去了哪里，以及命中缓存仍要付多少</span></div>
    {plate7}
  </figure>
  <table>
    <tr><th lang="en">Kernel</th><th lang="zh">Kernel</th><th lang="en">Cold</th><th lang="zh">冷编译</th><th lang="en">Trace + print</th><th lang="zh">Trace + 打印</th><th lang="en">MLIR passes</th><th lang="zh">MLIR pass</th><th lang="en">LLVM backend</th><th lang="zh">LLVM 后端</th><th lang="en">Cached call</th><th lang="zh">命中缓存的一次调用</th><th lang="en">IR ops built → kept</th><th lang="zh">建出的 op → 留下的</th></tr>
    {compile_table}
  </table>
  <p lang="en">LLVM's backend is the largest cold cost (41–60% on the GEMM and the attention kernel). MLIR's passes are not the bottleneck. The trace is the tax that matters for an agent, because it is paid on every call, cached or not: {float(fp8['warm_inmem_hit_call_s']):.2f} s per call for the FP8 GEMM, {float(fa['warm_inmem_hit_call_s']) * 1000:.0f} ms for the attention kernel, plus about 0.7 s of <code>import flydsl</code> in every fresh test process.</p>
  <p lang="zh">LLVM 后端是冷编译里最大的一块（GEMM 和 attention kernel 上占 41–60%）。 MLIR 的 pass 不是瓶颈。 对 agent 真正要紧的是 trace 这笔开销， 因为每次调用都要付， 不管有没有命中缓存： FP8 GEMM 每次调用 {float(fp8['warm_inmem_hit_call_s']):.2f} s， attention kernel {float(fa['warm_inmem_hit_call_s']) * 1000:.0f} ms， 另外每个新的测试进程还要付约 0.7 s 的 <code>import flydsl</code>。</p>
</section>

<section id="verdict">
  <h2><span class="n">10</span><span lang="en">What the evidence adds up to</span><span lang="zh">证据加起来说明了什么</span></h2>
  <p lang="en">Premise A held, more strongly than expected: removing an explicit D4/D5 decision cost 5–16% in most of the arms that changed code, LLVM's heuristics did not recover it, and in two of the five kernels whose fences we removed, the "hints" were load-bearing for correctness. Premise B held in its split form: where nobody scheduled by hand, LLVM's default is hard to beat and needed; where an author did, the heuristic scheduler still costs time, and per-shape choice among correct schedules is worth {(o_strict - 1) * 100:.1f}–{(o_point - 1) * 100:.1f}% on the one kernel we measured it on. Premise C held with a correction: agents look hard at compiler output and distrust its cache, but their speedups come from the decisions they already own. Premise D held: the trace is paid on every iteration.</p>
  <p lang="zh">前提 A 成立， 而且比预想的更强： 在大多数改变了代码的 arm 里， 去掉一个显式 D4/D5 决策要损失 5–16%， LLVM 的启发式拿不回来， 而且在我们去掉 fence 的五个 kernel 里， 有两个的"提示"在承担正确性。 前提 B 以分化的形式成立： 没有手工调度的地方， LLVM 的默认值很难超越， 也是必需的； 有作者调度的地方， 启发式调度器仍在浪费时间， 而在结果正确的调度之间按 shape 选择， 在我们测的这一个 kernel 上值 {(o_strict - 1) * 100:.1f}–{(o_point - 1) * 100:.1f}%。 前提 C 成立但需要修正： agent 花大力气看编译器输出、 不信任它的缓存， 但它们的提速来自本来就掌握的决策。 前提 D 成立： trace 在每次迭代都要付。</p>
  <p lang="en">So, can the redesign bring a large performance gain? Not uniformly, and the experiments say where. The large, measured stakes are in expert kernels whose authors already fight LLVM: there, a compiler that honors explicit order, pins and waits by construction protects the 5–16% those decisions are worth, and it makes a per-shape schedule search safe in kernels where today a fence may be carrying correctness. For simple kernels it is a wash on speed; LLVM's scheduler is worth keeping as the automatic mode until a replacement matches it. For agents, the larger gain is in iterations, not peak speed: fewer runs spent decoding ISA and fighting caches, and checks that turn silent wrong answers into compile errors.</p>
  <p lang="zh">那么， 重新设计能不能带来很大的性能提升？ 不会处处都有， 实验告诉了我们在哪里有。 实测到的大收益在专家 kernel 上， 它们的作者本来就在和 LLVM 对抗： 在那里， 一个从构造上就遵守显式顺序、pin 和 wait 的编译器， 能保住这些决策值的 5–16%， 也让按 shape 的调度搜索变得安全， 即使在今天 fence 可能承担着正确性的 kernel 里。 对简单 kernel， 速度上不会有明显差别； 在找到能与之匹敌的替代品之前， LLVM 的调度器值得留作自动模式。 对 agent， 更大的收益在迭代次数， 而不在峰值速度： 更少的运行花在解读 ISA 和与缓存斗争上， 也有检查把悄悄算错变成编译错误。</p>
</section>

<section id="next">
  <h2><span class="n">11</span><span lang="en">What would change our mind</span><span lang="zh">什么会推翻这个结论</span></h2>
  <ul>
    <li lang="en"><b>A program-order backend that loses.</b> Build KIR and the deterministic backend for one GEMM (the FP8 4-wave one, whose decisions are all explicit) and compare it with LLVM's output at identical source decisions. If program order plus honored pins is slower than LLVM, the case for dropping LLVM's scheduler in explicit mode fails.</li>
    <li lang="zh"><b>按程序顺序的后端输了。</b> 为一个 GEMM（FP8 4-wave， 它的决策全是显式的）实现 KIR 和确定性后端， 在相同的源码决策下和 LLVM 的输出比较。 如果"程序顺序 + 遵守 pin"比 LLVM 慢， 在显式模式下去掉 LLVM 调度器的理由就不成立。</li>
    <li lang="en"><b>Agents that do not benefit from the report.</b> On the overlay, run the same FlyBench records with two arms that differ only in a <code>report</code> command (resources, spills and occupancy per shape, waits, race and bounds checks). If fast-1.2 counts, regressions and tokens per record do not move, the agent half of the argument fails.</li>
    <li lang="zh"><b>agent 没有从报告中获益。</b> 在 overlay 上用同样的 FlyBench record 跑两个 arm， 唯一区别是有没有一个 <code>report</code> 命令（每个 shape 的资源、spill 和 occupancy， wait， 竞争和越界检查）。 如果 fast-1.2 数、回退和每个 record 的 token 都不变， 论证里关于 agent 的一半就不成立。</li>
    <li lang="en"><b>Correctness fences that are rare.</b> We found them in two of the five kernels whose fences we removed. A corpus-wide scan that removes fences one by one under the evaluator would say how common they are; if they are rare, token-based synchronization matters less than this post claims.</li>
    <li lang="zh"><b>撑着正确性的 fence 其实很少。</b> 在我们去掉 fence 的五个 kernel 里发现了两个。 在评测器下对整个语料逐个去掉 fence 扫一遍， 就能知道它们有多常见； 如果很少， 基于 token 的同步就没有本文说的那么重要。</li>
  </ul>
</section>

<section id="reproduce">
  <h2><span class="n">12</span><span lang="en">Reproduce</span><span lang="zh">复现</span></h2>
  <p lang="en">Everything above is a view of files in <a href="{B.DATA}"><code>data/flydsl-compiler-redesign</code></a>: <a href="{B.DATAF}/arms.csv"><code>arms.csv</code></a> (what each arm changes), <a href="{B.DATAF}/results.csv"><code>results.csv</code></a> (verdict, speed, interval), <a href="{B.DATAF}/per_shape.csv"><code>per_shape.csv</code></a>, <a href="{B.DATAF}/isa.csv"><code>isa.csv</code></a> (stock vs arm ISA at the dumped shapes), <a href="{B.DATAF}/compile_breakdown.csv"><code>compile_breakdown.csv</code></a>, <a href="{B.DATAF}/agent_mining_summary.json"><code>agent_mining_summary.json</code></a>, <a href="{B.DATAF}/failures.json"><code>failures.json</code></a> (the failing tests of each wrong arm), <a href="{B.DATAF}/summary.json"><code>summary.json</code></a> (all of it in one file), the {len(B.SUMMARY)} patches under <code>patches/</code>, and under <code>scripts/</code> the scripts that built and summarized them and the builder of this page (<code>make_page.py</code>), with internal paths replaced by placeholders. Each patch applies to FlyDSL <code>cb770e9a</code> and is scored with the FlyBench v1 evaluator on one MI355X.</p>
  <p lang="zh">上面所有内容都来自 <a href="{B.DATA}"><code>data/flydsl-compiler-redesign</code></a> 里的文件： <a href="{B.DATAF}/arms.csv"><code>arms.csv</code></a>（每个 arm 改了什么）、<a href="{B.DATAF}/results.csv"><code>results.csv</code></a>（判定、速度、区间）、<a href="{B.DATAF}/per_shape.csv"><code>per_shape.csv</code></a>、<a href="{B.DATAF}/isa.csv"><code>isa.csv</code></a>（dump 的 shape 上 stock 与 arm 的 ISA）、<a href="{B.DATAF}/compile_breakdown.csv"><code>compile_breakdown.csv</code></a>、<a href="{B.DATAF}/agent_mining_summary.json"><code>agent_mining_summary.json</code></a>、<a href="{B.DATAF}/failures.json"><code>failures.json</code></a>（每个出错 arm 的失败测试）、<a href="{B.DATAF}/summary.json"><code>summary.json</code></a>（以上全部合在一个文件里）， <code>patches/</code> 下的 {len(B.SUMMARY)} 个补丁， 以及 <code>scripts/</code> 下构建和汇总它们的脚本和生成本页的脚本（<code>make_page.py</code>）， 内部路径换成了占位符。 每个补丁都可以打到 FlyDSL <code>cb770e9a</code> 上， 用 FlyBench v1 评测器在一张 MI355X 上打分。</p>
</section>

<section id="epilogue">
  <h2><span class="n">13</span><span lang="en">Epilogue</span><span lang="zh">结语</span></h2>
  <p lang="en">I started out expecting to argue that LLVM was the problem. The experiments say something narrower and more useful. LLVM's heuristics are good at the decisions nobody makes; the trouble starts when someone does make them and the compiler cannot tell a preference from a constraint. A compiler for agents should be built around that distinction: every decision has one owner, every constraint is visible as data, and everything the compiler chose is written down where the agent can read it.</p>
  <p lang="zh">一开始我以为要论证的是"LLVM 是问题所在"。 实验给出的结论更窄， 也更有用。 LLVM 的启发式很擅长做没人去做的决定； 麻烦出在有人做了这些决定， 而编译器分不清哪些是偏好、 哪些是约束。 给 agent 用的编译器应该围绕这个区别来建： 每个决定只有一个主人， 每个约束都以数据的形式可见， 编译器做出的每个选择都写在 agent 读得到的地方。</p>
</section>

</main>
<footer class="colophon">
  <div><span lang="en">Source: FlyDSL <code>cb770e9a</code> (experiments) and <code>fe81c407</code> (code reading); LLVM <code>941a04e69</code>; FlyBench v1 evaluator on the AMD Pilot v2 overlay, 2026-11-22.</span><span lang="zh">来源： FlyDSL <code>cb770e9a</code>（实验）与 <code>fe81c407</code>（读代码）； LLVM <code>941a04e69</code>； AMD Pilot v2 overlay 上的 FlyBench v1 评测器， 2026-11-22。</span></div>
  <div><span lang="en">Typography: Epilogue · IBM Plex Serif · Victor Mono · Noto Serif SC. Palette: drafting paper, Prussian ink, vermilion for conflict, teal for a single owner.</span><span lang="zh">字体： Epilogue · IBM Plex Serif · Victor Mono · Noto Serif SC。 配色： 绘图纸、普鲁士蓝墨、表示冲突的朱红、表示单一主人的青色。</span></div>
  <div><span lang="en">Related: <a href="/sources/flydsl-compiler-in-the-loop.html">FlyDSL compiler in the loop</a> · <a href="/sources/together-compiler.html">Together Compiler</a> · <a href="/sources/flydsl.html">FlyDSL source reading</a>.</span><span lang="zh">相关： <a href="/sources/flydsl-compiler-in-the-loop.html">FlyDSL compiler in the loop</a> · <a href="/sources/together-compiler.html">Together Compiler</a> · <a href="/sources/flydsl.html">FlyDSL 源码阅读</a>。</span></div>
</footer>
</div>
</div>
<script>{JS}</script>
</body>
</html>
"""
    html = re.sub(r"([。，：；、])(?=[^\s<）」』”])", r"\1 ", html)
    html = re.sub(r'(<(\w+)[^>]*\blang="zh"[^>]*>)(.*?)(</\2>)',
                  lambda m: m.group(1) + m.group(3).replace("D4/D5", "D4 / D5") + m.group(4), html, flags=re.S)
    return html, pending


MINING = B.MINING

if __name__ == "__main__":
    html, pending = render()
    B.OUT.write_text(html)
    print(f"wrote {B.OUT} ({len(html.encode()) / 1024:.0f} KB); pending arms: {pending}")
