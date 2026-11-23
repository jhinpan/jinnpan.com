"""Assemble the Experiment 008 page from computed numbers, charts and static plates."""

import re

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
    """'VGPR 190→220, spills 0→12' style ISA delta at the largest dumped shape."""
    a, b = big_pair(label)
    if a["order_sha"] == b["order_sha"]:
        return "ISA identical", "ISA 完全相同"
    va, vb = a["meta"].get("vgpr_count"), b["meta"].get("vgpr_count")
    sa, sb = a["meta"].get("vgpr_spill_count"), b["meta"].get("vgpr_spill_count")
    return f"VGPR {va}→{vb}, spills {sa}→{sb}", f"VGPR {va}→{vb}，spill {sa}→{sb}"


def cosine(label):
    m = re.search(r"cosine similarity ([0-9.]+)", B.arm(label)["reason"] or "")
    return float(m.group(1))


NUM = {2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}


def link(href, text):
    return f'<a href="{href}">{text}</a>'


def pair(en, zh, tag="p", cls=""):
    c = f' class="{cls}"' if cls else ""
    return f'<{tag}{c} lang="en">{en}</{tag}>\n  <{tag}{c} lang="zh">{zh}</{tag}>'


TERMS_GPU = [
    ("kernel", "kernel",
     "A function that runs on the GPU. Many threads run it at the same time.",
     "在 GPU 上运行的函数。 很多线程同时执行它。"),
    ("wave", "wave",
     "A group of 64 threads that run the same instruction together on an AMD GPU.",
     "AMD GPU 上一起执行同一条指令的 64 个线程。"),
    ("register", "寄存器",
     "The fastest storage in the GPU. Each wave can use only a limited number of registers. "
     "VGPRs and AGPRs are two kinds of registers. Matrix instructions on MI355X can keep their results in AGPRs.",
     "GPU 里最快的存储。 每个 wave 能用的寄存器数量有限。 VGPR 和 AGPR 是两种寄存器。 "
     "MI355X 的矩阵指令可以把结果放在 AGPR 里。"),
    ("spill", "spill",
     "When a kernel needs more registers than it has, the compiler moves some values to slow memory. "
     "The kernel then becomes slower.",
     "kernel 需要的寄存器比能用的多时， 编译器把一部分值放到慢速内存里。 kernel 因此变慢。"),
    ("occupancy", "occupancy",
     "The number of waves that can run on one compute unit at the same time. "
     "When each wave uses more registers, fewer waves fit.",
     "一个计算单元上能同时运行的 wave 数。 每个 wave 用的寄存器越多， 能放下的 wave 越少。"),
    ("instruction order", "指令顺序",
     "The order in which the GPU runs the instructions. A good order keeps the math units busy while data loads.",
     "GPU 执行指令的先后顺序。 好的顺序让计算单元在加载数据时也不空闲。"),
    ("wait (<code>s_waitcnt</code>)", "wait（<code>s_waitcnt</code>）",
     "An instruction that stops a wave until its earlier memory operations are complete.",
     "一条指令： 让 wave 停下， 直到它之前发出的访存操作完成。"),
    ("barrier (<code>s_barrier</code>)", "barrier（<code>s_barrier</code>）",
     "An instruction that stops each wave of a workgroup until all of these waves arrive at it.",
     "一条指令： 让一个 workgroup 里的每个 wave 停下， 直到所有 wave 都执行到这里。"),
    ("fence (<code>sched_barrier</code>)", "fence（<code>sched_barrier</code>）",
     "A marker for the compiler, not an instruction for the GPU. The compiler must not move instructions across it.",
     "给编译器的标记， 不是给 GPU 的指令。 编译器不能把指令移过它。"),
    ("interleave group (<code>sched_group_barrier</code>)", "交错组（<code>sched_group_barrier</code>）",
     "A marker that gives the compiler an order pattern, for example \"one matrix instruction, then one load\".",
     "给编译器的顺序模板， 例如 \"一条矩阵指令， 然后一条加载指令\"。"),
    ("wave priority (<code>s_setprio</code>)", "wave 优先级（<code>s_setprio</code>）",
     "An instruction that tells the GPU which waves to run first.",
     "一条指令： 告诉 GPU 先执行哪些 wave。"),
    ("pin", "pin",
     "A hint that forces a value into one kind of register, for example into AGPRs.",
     "强制把某个值放进指定寄存器（例如 AGPR）的提示。"),
    ("hazard", "hazard",
     "A pair of instructions that must have a gap between them. "
     "The compiler fills the gap with <code>s_nop</code>, an instruction that does nothing.",
     "两条之间必须留出间隔的指令。 编译器用 <code>s_nop</code>（什么都不做的指令）填这个间隔。"),
    ("heuristic", "启发式规则",
     "A general rule that a compiler uses to make a choice automatically. "
     "It works well on average, but it does not know what the author wants.",
     "编译器自动做选择时用的通用规则。 平均效果不错， 但它不知道作者想要什么。"),
    ("LLVM", "LLVM",
     "A large, general compiler project. FlyDSL uses its AMD GPU part to make the final machine instructions.",
     "一个大型通用编译器项目。 FlyDSL 用它的 AMD GPU 部分生成最终的机器指令。"),
    ("MLIR", "MLIR",
     "A framework for building compiler steps. FlyDSL uses it between Python and LLVM.",
     "一个用来搭建编译步骤的框架。 FlyDSL 在 Python 和 LLVM 之间使用它。"),
    ("ISA", "ISA",
     "The machine instructions that the GPU runs. We read the ISA to see what the compiler did.",
     "GPU 真正执行的机器指令。 我们读 ISA 来确认编译器做了什么。"),
    ("trace", "trace",
     "FlyDSL runs the Python code of a kernel once and records its operations. It does this before each launch.",
     "FlyDSL 先把 kernel 的 Python 代码执行一遍， 记录其中的操作。 每次启动 kernel 前都要做一次。"),
    ("layout", "layout",
     "A rule that says which thread holds which element of a data tile.",
     "一条规则： 说明数据 tile 的哪个元素由哪个线程持有。"),
    ("race", "竞争",
     "A bug in which the result depends on which wave runs first. The wrong output can change from run to run.",
     "一种 bug： 结果取决于哪个 wave 先执行。 错误的输出每次运行都可能不同。"),
]

TERMS_EXP = [
    ("agent", "agent",
     "An AI model that writes and tests code by itself, in many steps.",
     "能自己分很多步写代码、 测代码的 AI 模型。"),
    ("overlay", "overlay",
     "The AMD Pilot v2 job system. It runs our jobs on MI355X GPUs in the same way as agent jobs.",
     "AMD Pilot v2 的作业系统。 它在 MI355X GPU 上运行我们的作业， 方式和运行 agent 作业相同。"),
    ("FlyBench evaluator", "FlyBench 评测器",
     "The program that scores agent submissions. First it checks the results. "
     "Then it compares the speed of a change with the stock kernel.",
     "给 agent 提交打分的程序。 它先检查结果是否正确， 再把改动后的速度和 stock kernel 比较。"),
    ("stock kernel", "stock kernel",
     "The unchanged kernel from FlyDSL. Each comparison uses it as the baseline.",
     "FlyDSL 原来的、 没有修改过的 kernel。 所有比较都以它为基准。"),
    ("attempt", "尝试",
     "One agent that works on one kernel, within one time limit. One FlyBench run has many attempts.",
     "一个 agent 在一个时间限制内处理一个 kernel。 一次 FlyBench 运行包含很多次尝试。"),
    ("arm", "arm",
     "One variant in an experiment: the stock kernel with one change.",
     "实验里的一个变体： stock kernel 加上一个改动。"),
    ("speed ratio", "速度比",
     "Stock time divided by arm time. Below 1.0 means that the arm is slower. "
     "In this post, \"12% slower\" means a speed ratio of 0.88×.",
     "stock 用时除以 arm 用时。 小于 1.0 表示 arm 更慢。 本文说 \"慢了 12%\"， 指速度比是 0.88×。"),
    ("noise level", "噪声下限",
     "The smallest change that the evaluator can separate from random variation. "
     "FlyBench measures it with the stock kernel against itself.",
     "评测器能和随机波动区分开的最小变化。 FlyBench 用 stock kernel 和它自己比较来测得。"),
    ("null control", "空白对照",
     "An arm that changes nothing that matters. Its result shows the noise of the measurement.",
     "一个没有实质改动的 arm。 它的结果显示测量本身的噪声。"),
]


def terms_table(rows):
    out = ['<table class="terms">',
           '<tr><th lang="en">Term</th><th lang="zh">术语</th><th lang="en">Meaning in this post</th>'
           '<th lang="zh">在本文中的意思</th></tr>']
    for t_en, t_zh, d_en, d_zh in rows:
        out.append(f'<tr><td><span lang="en">{t_en}</span><span lang="zh">{t_zh}</span></td>'
                   f'<td><span lang="en">{d_en}</span><span lang="zh">{d_zh}</span></td></tr>')
    out.append("</table>")
    return "\n".join(out)


def render():
    r = {l: B.ratio(l) for l in B.SUMMARY}
    f02, n02 = B.tests("x02-nosched")
    f04, n04 = B.tests("x04-nosched")
    x10f0 = B.shape("x10-agprllvm", "f0")
    x10p1 = B.shape("x10-agprllvm", "p1")
    s_f0, c_f0 = isa_pair("x10-agprllvm", "f0")
    x08 = B.arm("x08-noprio")
    x09 = B.arm("x09-postmischedon")
    x13 = B.arm("x13-nogroup")
    x15 = B.arm("x15-null")
    h10 = B.arm("h10-stratmaxilp")
    h11 = B.arm("h11-mischedoff")
    nop_s, nop_c = (p["s_nop"] for p in big_pair("x09-postmischedon"))
    assert x13["verdict"] == "scored" and x15["verdict"] == "scored" and B.arm("h08b-stratmaxilp")["verdict"] == "unscorable"

    pre_arms = ["x01-nosched", "x14-nogroup", "h10-stratmaxilp", "h11-mischedoff"]
    o_point, o_strict, o_n = B.oracle(pre_arms)

    m = B.MINING
    rv = m["reviews"]
    src = rv["speedup_source_share_legit"]
    cmds = m["agent_commands"]
    cost = m["agent_cost_per_record"]
    n_tr = cmds["traces"]
    nc = cmds["commands"]
    by = cmds["by_purpose"]
    isa_share = cmds["traces_that_inspected_ir_or_isa"] / n_tr * 100
    cc_share = cmds["traces_that_cleared_flydsl_cache"] / n_tr * 100

    comp = {c["kernel"]: c for c in B.COMPILE}
    fp8, fa = comp["fp8_gemm_4wave"], comp["flash_attn_gfx950"]

    def share(c, key):
        return float(c[key]) / float(c["cold_compile_s"]) * 100

    llvm_lo, llvm_hi = sorted(share(c, "llvm_backend_s") for c in (fp8, fa))
    mlir_lo, mlir_hi = sorted(share(c, "mlir_passes_1_14_s") for c in (fp8, fa))
    llvm_share = f"{llvm_lo:.0f}–{llvm_hi:.0f}%"

    fa_fail, fb_fail = B.FAILURES["x02-nosched"], B.FAILURES["x02b-nosched"]
    assert fa_fail["failed_tests"] == fb_fail["failed_tests"]
    mis_a, mis_b = (int(f["mismatched"][0].split(" / ")[0]) for f in (fa_fail, fb_fail))
    mis_total = int(fa_fail["mismatched"][0].split(" / ")[1].split()[0])

    def pct(label):
        return f"{(1 - r[label]) * 100:.0f}"

    ablated_scored = ["x01-nosched", "x03-nosched", "x05-nosched", "x06-noprio", "x07-noprio"]
    slow_lo = min(1 - r[l] for l in ablated_scored) * 100
    slow_hi = max(1 - r[l] for l in ablated_scored) * 100
    n_slow = len(ablated_scored) + 1  # the 4-wave GEMM, counted at the two shapes its AGPR arm touches
    assert slow_lo <= (1 - x10p1["ratio"]) * 100 and (1 - x10f0["ratio"]) * 100 <= slow_hi

    # ---------------- charts
    def row(label, en, zh, cls):
        d = B.arm(label)
        if d["verdict"] != "scored":
            f, n = B.tests(label)
            if f:
                return {"en": en, "zh": zh, "value": None, "note_en": f"wrong results: {f} of {n} tests fail",
                        "note_zh": f"结果错误：{n} 个测试里 {f} 个失败"}
            if d["verdict"] == "unscorable":
                return {"en": en, "zh": zh, "value": None, "note_en": "stock itself failed", "note_zh": "stock 自己出错"}
            return {"en": en, "zh": zh, "value": None, "note_en": "one wrong output", "note_zh": "一次输出错误"}
        return {"en": en, "zh": zh, "value": d["ratio"], "ci": d["ci95"], "cls": cls}

    plate3 = B.bar_chart([
        row("x01-nosched", "preshuffle-a8 · fences + groups", "preshuffle-a8 · fence + 交错组", "d5"),
        row("x14-nogroup", "preshuffle-a8 · groups only", "preshuffle-a8 · 只去交错组", "d5"),
        row("x03-nosched", "mx-preshuffle-a8w8 · fences + groups", "mx-preshuffle-a8w8 · fence + 交错组", "d5"),
        row("x05-nosched", "swa-gqa · fences + groups", "swa-gqa · fence + 交错组", "d5"),
        row("x02-nosched", "a16w16 · fences", "a16w16 · fence", "d5"),
        row("x02b-nosched", "a16w16 · fences (repeat)", "a16w16 · fence（重复）", "d5"),
        row("x04-nosched", "flash-fwd-dense · fences + groups", "flash-fwd-dense · fence + 交错组", "d5"),
        row("x13-nogroup", "flash-fwd-dense · groups only", "flash-fwd-dense · 只去交错组", "d5"),
        row("x06-noprio", "fp8-rowscale-8wave · s_setprio", "fp8-rowscale-8wave · s_setprio", "prio"),
        row("x07-noprio", "mxfp8-8wave · s_setprio", "mxfp8-8wave · s_setprio", "prio"),
        row("x08-noprio", "flash-fwd-dense · s_setprio", "flash-fwd-dense · s_setprio", "prio"),
        row("x09-postmischedon", "flash-fwd-dense · post-RA scheduler on", "flash-fwd-dense · 打开 post-RA 调度", "d5"),
        {"en": "fp8-rowscale-4wave · AGPR pin (16384³)", "zh": "fp8-rowscale-4wave · AGPR pin（16384³）",
         "value": x10f0["ratio"], "ci": x10f0["ci95"], "cls": "d4"},
        {"en": "fp8-rowscale-4wave · AGPR pin (8192³)", "zh": "fp8-rowscale-4wave · AGPR pin（8192³）",
         "value": x10p1["ratio"], "ci": x10p1["ci95"], "cls": "d4"},
        row("x11-occdefault", "fp8-rowscale-4wave · occupancy hints", "fp8-rowscale-4wave · occupancy 提示", "d4"),
        row("x12-nosched", "rmsnorm-fwd · null control", "rmsnorm-fwd · 空白对照", "ctrl"),
        row("x15-null", "a16w16 · null control", "a16w16 · 空白对照", "ctrl"),
    ], 0.80, 1.10, title_id="p3")

    plate4 = B.bar_chart([
        row("h01-stratmaxmem", "rmsnorm-fwd · max-memory-clause", "rmsnorm-fwd · max-memory-clause", "head"),
        row("h02-stratmaxilp", "rmsnorm-fwd · max-ilp", "rmsnorm-fwd · max-ilp", "head"),
        row("h12-mischedoff", "rmsnorm-fwd · LLVM schedulers off", "rmsnorm-fwd · 关掉 LLVM 调度器", "off"),
        row("h03-stratmaxmem", "layernorm-fwd · max-memory-clause", "layernorm-fwd · max-memory-clause", "head"),
        row("h04-stratmaxilp", "layernorm-fwd · max-ilp", "layernorm-fwd · max-ilp", "head"),
        row("h05-stratmaxmem", "softmax-fwd · max-memory-clause", "softmax-fwd · max-memory-clause", "head"),
        row("h06-stratmaxilp", "softmax-fwd · max-ilp", "softmax-fwd · max-ilp", "head"),
        row("h07-stratmaxmem", "moe.reduce · max-memory-clause", "moe.reduce · max-memory-clause", "head"),
        row("h09-stratmaxmem", "moe gemm1 · max-memory-clause", "moe gemm1 · max-memory-clause", "head"),
        row("h08-stratmaxilp", "moe gemm1 · max-ilp", "moe gemm1 · max-ilp", "head"),
        row("h08b-stratmaxilp", "moe gemm1 · max-ilp (repeat)", "moe gemm1 · max-ilp（重复）", "head"),
        row("h10-stratmaxilp", "preshuffle-a8 · max-ilp", "preshuffle-a8 · max-ilp", "head"),
        row("h11-mischedoff", "preshuffle-a8 · LLVM schedulers off", "preshuffle-a8 · 关掉 LLVM 调度器", "off"),
    ], 0.85, 1.05, title_id="p4")

    plate5, _ = B.per_shape_plot([("x01-nosched", "s1", -4.5), ("x14-nogroup", "s2", -1.5),
                                  ("h10-stratmaxilp", "s3", 1.5), ("h11-mischedoff", "s4", 4.5)])
    plate6a = B.hbar_pairs([
        ("wider memory access", "更宽的访存", src["vectorize_or_memory"], "d5"),
        ("tile choice per shape", "按 shape 选 tile", src["dispatch_or_tiling"], "d5"),
        ("new algorithm", "新算法", src["algorithm_rewrite"], "d5"),
        ("more work splits", "更多工作切分", src["split_k_or_grid_fill"], "d5"),
        ("order setting", "顺序设置", src["schedule_or_sync_knob"], "prio"),
        ("register setting", "寄存器设置", src["register_or_occupancy_knob"], "d4"),
    ], title_id="p6a", maxv=1.0)
    plate6b = B.hbar_pairs([
        ("edit code", "改代码", by["edit_code"] / nc, "d5"),
        ("read code", "读代码", by["read_code"] / nc, "d5"),
        ("own timing, profiling", "自己测时、profile", by["own_timing_or_profile"] / nc, "head"),
        ("read IR or ISA", "读 IR 或 ISA", by["inspect_ir_or_isa"] / nc, "prio"),
        ("run ./evaluate", "跑 ./evaluate", by["evaluate"] / nc, "head"),
        ("clear the compile cache", "清编译缓存", by["cache_clear"] / nc, "off"),
        ("run tests", "跑测试", by["run_tests"] / nc, "head"),
    ], title_id="p6b", fmt="{:.1%}", maxv=0.30)
    plate7 = B.compile_chart()

    # ---------------- tables
    def res(label):
        d = B.arm(label)
        if d["verdict"] != "scored":
            f, n = B.tests(label)
            if f:
                en, zh = f"wrong · {f}/{n} tests", f"错误 · {f}/{n} 测试"
            elif d["verdict"] == "unscorable":
                en, zh = "stock failed", "stock 出错"
            else:
                en, zh = "wrong output", "输出错误"
            return f'<td class="bad"><span lang="en">{en}</span><span lang="zh">{zh}</span></td>'
        nf = d["noise_floor"] or 0
        cls = "good" if d["ci95"][0] > 1 + nf else ("bad" if d["ci95"][1] < 1 - nf else "v")
        return f'<td class="{cls}">{d["ratio"]:.3f}× {B.ci(label)}</td>'

    def trow(label, kernel, what_en, what_zh):
        isa_en, isa_zh = spill_note(label)
        return (f'<tr><td><code>{kernel}</code></td><td><span lang="en">{what_en}</span><span lang="zh">{what_zh}</span></td>'
                f'{res(label)}<td class="v"><span lang="en">{isa_en}</span><span lang="zh">{isa_zh}</span></td></tr>')

    fences_groups = ("fences + interleave groups", "fence + 交错组")
    abl_table = "\n".join([
        trow("x01-nosched", "gemm.preshuffle-a8", *fences_groups),
        trow("x14-nogroup", "gemm.preshuffle-a8", "interleave groups only", "只去交错组"),
        trow("x03-nosched", "gemm.mx-preshuffle-a8w8", *fences_groups),
        trow("x05-nosched", "attention.swa-gqa", *fences_groups),
        trow("x02-nosched", "gemm.a16w16", "fences", "fence"),
        trow("x02b-nosched", "gemm.a16w16", "fences (repeat)", "fence（重复）"),
        trow("x04-nosched", "attention.flash-fwd-dense", *fences_groups),
        trow("x13-nogroup", "attention.flash-fwd-dense", "interleave groups only", "只去交错组"),
        trow("x06-noprio", "gemm.fp8-rowscale-8wave", "<code>s_setprio</code>", "<code>s_setprio</code>"),
        trow("x07-noprio", "gemm.mxfp8-8wave", "<code>s_setprio</code>", "<code>s_setprio</code>"),
        trow("x08-noprio", "attention.flash-fwd-dense", "<code>s_setprio</code>", "<code>s_setprio</code>"),
        trow("x09-postmischedon", "attention.flash-fwd-dense", "the setting that turns off the post-RA scheduler",
             "关掉 post-RA 调度器的设置"),
        trow("x10-agprllvm", "gemm.fp8-rowscale-4wave", "AGPR pin (inline assembly)", "AGPR pin（inline 汇编）"),
        trow("x11-occdefault", "gemm.fp8-rowscale-4wave", "occupancy hints", "occupancy 提示"),
        trow("x12-nosched", "norm.rmsnorm-fwd", "nothing (null control)", "无（空白对照）"),
        trow("x15-null", "gemm.a16w16", "nothing, lines added (null control)", "无， 只加了那几行代码（空白对照）"),
    ])
    sched_off = ("LLVM schedulers off", "关掉 LLVM 调度器")
    mmc, ilp = "<code>max-memory-clause</code>", "<code>max-ilp</code>"
    head_table = "\n".join([
        trow("h01-stratmaxmem", "norm.rmsnorm-fwd", mmc, mmc),
        trow("h02-stratmaxilp", "norm.rmsnorm-fwd", ilp, ilp),
        trow("h12-mischedoff", "norm.rmsnorm-fwd", *sched_off),
        trow("h03-stratmaxmem", "norm.layernorm-fwd", mmc, mmc),
        trow("h04-stratmaxilp", "norm.layernorm-fwd", ilp, ilp),
        trow("h05-stratmaxmem", "norm.softmax-fwd", mmc, mmc),
        trow("h06-stratmaxilp", "norm.softmax-fwd", ilp, ilp),
        trow("h07-stratmaxmem", "moe.reduce", mmc, mmc),
        trow("h09-stratmaxmem", "moe.2stage-gemm1", mmc, mmc),
        trow("h08-stratmaxilp", "moe.2stage-gemm1", ilp, ilp),
        trow("h08b-stratmaxilp", "moe.2stage-gemm1", f"{ilp} (repeat)", f"{ilp}（重复）"),
        trow("h10-stratmaxilp", "gemm.preshuffle-a8", ilp, ilp),
        trow("h11-mischedoff", "gemm.preshuffle-a8", *sched_off),
    ])
    names = {"vector_add": "examples/01-vectorAdd", "preshuffle_fp8": "preshuffle FP8 GEMM",
             "fp8_gemm_4wave": "fp8_gemm_4wave 256×256", "flash_attn_gfx950": "flash_attn_gfx950 D128"}
    compile_table = "\n".join(
        f'<tr><td>{names[c["kernel"]]}</td><td class="v">{float(c["cold_compile_s"]):.2f} s</td>'
        f'<td class="v">{float(c["trace_s"]) + float(c["verify_print_s"]):.2f} s</td>'
        f'<td class="v">{float(c["mlir_passes_1_14_s"]):.2f} s</td><td class="v">{float(c["llvm_backend_s"]):.2f} s</td>'
        f'<td class="v">{float(c["warm_inmem_hit_call_s"]) * 1000:.0f} ms</td>'
        f'<td class="v">{int(c["origin_ir_ops"]):,} → {int(c["after_layout_lowering_ops"]):,}</td></tr>'
        for c in B.COMPILE)

    def th(en, zh):
        return f'<th lang="en">{en}</th><th lang="zh">{zh}</th>'

    exp_head = (th("Kernel", "Kernel") + th("What we removed", "去掉了什么")
                + th("Speed ratio [95% interval]", "速度比 [95% 区间]") + th("ISA at the largest shape", "最大 shape 的 ISA"))
    set_head = (th("Kernel", "Kernel") + th("Setting", "设置")
                + th("Speed ratio [95% interval]", "速度比 [95% 区间]") + th("ISA at the largest shape", "最大 shape 的 ISA"))

    def legend(items):
        spans = "".join(f'<span><i style="background:var(--{c})"></i><span lang="en">{en}</span><span lang="zh">{zh}</span></span>'
                        for c, en, zh in items)
        return f'<p class="legend">{spans}</p>'

    n_arms = len(B.SUMMARY)
    x01f0 = (B.shape("x01-nosched", "f0")["ratio"] - 1) * 100
    x01p1 = (B.shape("x01-nosched", "p1")["ratio"] - 1) * 100
    V, M, L = B.V042, B.MAIN, B.LLVM
    ast_rw = link(f"{M}/python/flydsl/compiler/ast_rewriter.py", "<code>ast_rewriter.py</code>")
    jitf = link(f"{M}/python/flydsl/compiler/jit_function.py#L1203-L1213", "<code>jit_function.py</code>")
    passes = link(f"{M}/include/flydsl/Dialect/Fly/Transforms/Passes.td#L29-L40", "<code>Passes.td</code>")
    claude = link(f"{M}/CLAUDE.md?plain=1#L192-L197", "<code>CLAUDE.md</code>")
    skill3 = link(f"{M}/.claude/skills/llvm/SKILL.md?plain=1#L114-L131", "llvm skill §3")
    skill8 = link(f"{M}/.claude/skills/llvm/SKILL.md?plain=1#L250-L263", "§8")
    fp4 = link(f"{M}/kernels/gemm/fp4_gemm_4wave.py#L82-L87", "<code>fp4_gemm_4wave.py</code>")
    moe = link(f"{M}/kernels/mega_moe_tp/gemm.py#L948-L951", "<code>mega_moe_tp/gemm.py</code>")
    siinstr = link(f"{L}/llvm/lib/Target/AMDGPU/SIInstrInfo.cpp#L4554-L4559", "<code>SIInstrInfo.cpp</code>")
    flash_src = link(f"{V}/kernels/attention/flash_attn_gfx950.py#L1027", "<code>flash_attn_gfx950.py</code>")
    arch = link(f"{V}/.claude/skills/llvm/references/arch-gfx950.md#L147", "<code>arch-gfx950.md</code>")
    init_py = link(f"{V}/kernels/__init__.py", "<code>kernels/__init__.py</code>")
    pr1368 = link("https://github.com/ROCm/FlyDSL/pull/1368", "#1368")
    pr1477 = link("https://github.com/ROCm/FlyDSL/pull/1477", "#1477")
    pr1436 = link("https://github.com/ROCm/FlyDSL/pull/1436", "#1436")
    pycute = link("https://github.com/NVIDIA/cutlass/tree/main/python/pycute", "pycute")
    jax = link("https://docs.jax.dev/en/latest/control-flow.html", "JAX control flow")
    isa_xml_en = link("https://gpuopen.com/machine-readable-isa/", "machine-readable ISA files")
    isa_xml_zh = link("https://gpuopen.com/machine-readable-isa/", "机器可读 ISA 文件")
    tensile = link("https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipblaslt/tensilelite/tensilelite/Toolchain/Component.py",
                   "<code>Component.py</code>")
    main_tree = link(M.replace("/blob/", "/tree/"), "fe81c407")
    data_dir = link(B.DATA, "<code>data/flydsl-compiler-redesign</code>")

    def df(name, en, zh):
        return (f'<li lang="en">{link(f"{B.DATAF}/{name}", f"<code>{name}</code>")}: {en}</li>\n'
                f'    <li lang="zh">{link(f"{B.DATAF}/{name}", f"<code>{name}</code>")}： {zh}</li>')

    plate2 = PLATE_PIPELINES.replace("{LLVM_SHARE}", llvm_share)

    head = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Experiment 008 · One Decision, One Decider: what the FlyDSL compiler should keep and remove</title>
<meta name="description" content="In FlyDSL, two parties decide register use and instruction order: the kernel author and LLVM. A plain-language analysis with a glossary, and {n_arms} experiments on MI355X GPUs that test a redesign with one decider for each decision.">
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
    <li><a href="#question"><span class="num">01</span><span lang="en">The question</span><span lang="zh">问题</span></a></li>
    <li><a href="#terms"><span class="num">02</span><span lang="en">Terms</span><span lang="zh">术语</span></a></li>
    <li><a href="#decisions"><span class="num">03</span><span lang="en">Five decisions</span><span lang="zh">五类决策</span></a></li>
    <li><a href="#redundancy"><span class="num">04</span><span lang="en">Repeated work</span><span lang="zh">重复的工作</span></a></li>
    <li><a href="#redesign"><span class="num">05</span><span lang="en">The redesign</span><span lang="zh">重新设计</span></a></li>
    <li><a href="#method"><span class="num">06</span><span lang="en">How we tested</span><span lang="zh">怎样检验</span></a></li>
    <li><a href="#ablation"><span class="num">07</span><span lang="en">A · Take a decision away</span><span lang="zh">A · 拿走一个决策</span></a></li>
    <li><a href="#headroom"><span class="num">08</span><span lang="en">B · Untuned settings</span><span lang="zh">B · 没人调过的设置</span></a></li>
    <li><a href="#agents"><span class="num">09</span><span lang="en">C · What agents did</span><span lang="zh">C · agent 做了什么</span></a></li>
    <li><a href="#overhead"><span class="num">10</span><span lang="en">D · Compile cost</span><span lang="zh">D · 编译开销</span></a></li>
    <li><a href="#verdict"><span class="num">11</span><span lang="en">What it means</span><span lang="zh">结论</span></a></li>
    <li><a href="#next"><span class="num">12</span><span lang="en">What could prove us wrong</span><span lang="zh">什么会推翻结论</span></a></li>
    <li><a href="#reproduce"><span class="num">13</span><span lang="en">Reproduce</span><span lang="zh">复现</span></a></li>
    <li><a href="#epilogue"><span class="num">14</span><span lang="en">Epilogue</span><span lang="zh">结语</span></a></li>
  </ol>
</nav>
<div>
<header class="masthead">
  <div class="kicker">Experiment 008</div>
  <h1>One Decision <span class="slash">/</span> One Decider</h1>
  {pair("FlyDSL is a Python language for writing AMD GPU kernels. AI agents now write many of these kernels. This post asks which parts of the FlyDSL compiler to keep and which parts to remove. Then it tests the answer with " + str(n_arms) + " experiments on MI355X GPUs.",
        "FlyDSL 是一种用 Python 写 AMD GPU kernel 的语言。 现在很多 kernel 由 AI agent 来写。 本文讨论 FlyDSL 编译器的哪些部分该保留、 哪些可以去掉。 然后在 MI355X GPU 上用 " + str(n_arms) + " 组实验检验这个答案。",
        cls="subtitle")}
  <div class="spec">
    <div><b lang="en">Hardware</b><b lang="zh">硬件</b>MI355X · gfx950 · overlay</div>
    <div><b lang="en">FlyDSL under test</b><b lang="zh">被测 FlyDSL</b><a href="{V.replace('/blob/', '/tree/')}">cb770e9a</a> (FlyBench v1)</div>
    <div><b lang="en">Code reading</b><b lang="zh">读代码</b>{main_tree} (v0.5.2)</div>
    <div><b lang="en">Timing</b><b lang="zh">计时</b><span lang="en">FlyBench v1 · 6 ABBA pairs</span><span lang="zh">FlyBench v1 · 6 组 ABBA</span></div>
    <div><b lang="en">Agent data</b><b lang="zh">agent 数据</b><span lang="en">{m['runs']} runs · {m['receipts']} receipts · {n_tr} traces</span><span lang="zh">{m['runs']} 次运行 · {m['receipts']} 张回执 · {n_tr} 份轨迹</span></div>
    <div><b lang="en">Data</b><b lang="zh">数据</b><a href="{B.DATA}">data/flydsl-compiler-redesign</a></div>
  </div>
</header>

<main class="article">
"""

    s01 = f"""
<section id="question">
  <h2><span class="n">01</span><span lang="en">The question</span><span lang="zh">问题</span></h2>
  {pair('My mentor asked me to think about the FlyDSL compiler chain. <a href="/sources/flydsl-compiler-in-the-loop.html">The previous post</a> asked where a compiler can help in the kernel optimization loop. This post asks a more basic question.',
        'mentor 让我想清楚 FlyDSL 的编译链。 <a href="/sources/flydsl-compiler-in-the-loop.html">上一篇</a>讨论的是编译器能在 kernel 优化循环的哪个环节帮忙。 这一篇问一个更基础的问题。')}
  {pair("FlyDSL lets an author write a GPU kernel in Python. A kernel is a function that runs on thousands of GPU threads at the same time. The FlyDSL compiler changes the Python code into machine instructions for the GPU. For this work, it uses two large compiler projects: MLIR and LLVM.",
        "FlyDSL 让作者用 Python 写 GPU kernel。 kernel 是在 GPU 上由几千个线程同时执行的函数。 FlyDSL 编译器把 Python 代码转换成 GPU 的机器指令。 为此， 它用了两个大型编译器项目： MLIR 和 LLVM。")}
  {pair("The question is: if we can remove MLIR, and even LLVM, what must a kernel language for agents keep, and what can it remove? A design argument is easy to write. Thus we also ran experiments that could show that the argument is wrong. We used the same GPUs and the same evaluator that score the agents.",
        "问题是： 如果可以去掉 MLIR， 甚至去掉 LLVM， 一个给 agent 用的 kernel 语言必须保留什么、 可以去掉什么？ 设计论证很容易写。 所以我们还做了实验， 这些实验有可能证明论证是错的。 实验用的是给 agent 打分的同一批 GPU 和同一个评测器。")}
  <div class="callout">
    <div lang="en">
      <p><b>Short answer.</b></p>
      <ol class="list">
        <li>The main problem is not the size of the compiler. The main problem is that two parties make the same decisions. The kernel author decides part of the instruction stream. Then LLVM decides the same part again with general rules (heuristics). An author who needs a specific result must override LLVM with special markers, inline assembly or LLVM patches.</li>
        <li>We removed one decision of the author at a time and let LLVM decide. {NUM.get(n_slow, n_slow)} kernels became {slow_lo:.0f}–{slow_hi:.0f}% slower. One more kernel became about 1% slower or less.</li>
        <li>In two kernels, the results became wrong. The markers that we removed were not only for speed. They also kept the results correct.</li>
        <li>AI agents got most of their speedups from other decisions, for example the tile size and the memory access. In {isa_share:.0f}% of the agent command logs, the agent read the compiler output to learn what the compiler did.</li>
        <li>Our proposal: each decision gets exactly one decider. The compiler reports each choice that it makes. MLIR is removed. LLVM is removed in two steps.</li>
      </ol>
    </div>
    <div lang="zh">
      <p><b>简短回答。</b></p>
      <ol class="list">
        <li>主要问题不是编译器太大， 而是同一件事有两方在决定。 kernel 作者决定了指令流的一部分。 然后 LLVM 用通用规则（启发式规则）把这一部分再决定一遍。 需要特定结果的作者， 只能用特殊标记、 inline 汇编或者 LLVM 补丁去覆盖 LLVM 的决定。</li>
        <li>我们每次去掉作者的一个决策， 让 LLVM 来决定。 结果 {n_slow} 个 kernel 慢了 {slow_lo:.0f}–{slow_hi:.0f}%。 另一个 kernel 慢了约 1% 或更少。</li>
        <li>有两个 kernel 的结果变错了。 被去掉的标记不只是为了速度， 也在保证结果正确。</li>
        <li>AI agent 的提速主要来自其他决策， 例如 tile 大小和访存方式。 在 {isa_share:.0f}% 的 agent 命令记录里， agent 读过编译器的输出， 为的是弄清编译器做了什么。</li>
        <li>我们的方案： 每个决策只有一个决策者。 编译器报告它做的每个选择。 去掉 MLIR。 分两步去掉 LLVM。</li>
      </ol>
    </div>
  </div>
</section>
"""

    s02 = f"""
<section id="terms">
  <h2><span class="n">02</span><span lang="en">Terms used in this post</span><span lang="zh">本文用到的术语</span></h2>
  {pair("You do not need a compiler background to read this post. The two tables below define the terms that it uses. Each term has only one meaning in this post.",
        "读这篇文章不需要编译器背景。 下面两张表定义了文中用到的术语。 每个术语在本文里只有一个意思。")}
  {pair("GPU and compiler terms", "GPU 和编译器术语", tag="h3")}
  {terms_table(TERMS_GPU)}
  {pair("Experiment terms", "实验术语", tag="h3")}
  {terms_table(TERMS_EXP)}
</section>
"""

    s03 = f"""
<section id="decisions">
  <h2><span class="n">03</span><span lang="en">Five decisions, two deciders</span><span lang="zh">五类决策，两个决策者</span></h2>
  {pair("The speed of an AMD GPU kernel depends on five kinds of decisions. This is true for each language that makes kernels, not only for FlyDSL:",
        "AMD GPU kernel 的速度取决于五类决策。 这一点对所有写 kernel 的语言都成立， 不只是 FlyDSL：")}
  <ul lang="en">
    <li><b>Work split:</b> how to cut the problem into tiles, and how many workgroups to start.</li>
    <li><b>Data movement:</b> which thread moves which data through which memory.</li>
    <li><b>Instruction choice:</b> which matrix instruction does the math.</li>
    <li><b>Register use:</b> which register holds each value. This sets the occupancy.</li>
    <li><b>Instruction order:</b> the order of the instructions, the position of the waits, and the wave priority.</li>
  </ul>
  <ul lang="zh">
    <li><b>工作划分：</b> 问题怎样切成 tile， 启动多少个 workgroup。</li>
    <li><b>数据搬运：</b> 哪个线程通过哪一级存储搬哪些数据。</li>
    <li><b>指令选择：</b> 用哪条矩阵指令做运算。</li>
    <li><b>寄存器使用：</b> 每个值放在哪种寄存器里。 它决定了 occupancy。</li>
    <li><b>指令顺序：</b> 指令的先后、 wait 的位置和 wave 优先级。</li>
  </ul>
  {pair("In FlyDSL, the author makes the first three decisions in Python. The compiler only translates them. The heuristics of LLVM make the last two decisions.",
        "在 FlyDSL 里， 前三类决策由作者在 Python 里做出， 编译器只负责翻译。 后两类决策由 LLVM 的启发式规则来做。")}
  <figure class="plate wide">
    <div class="plate-meta">Plate I · <span lang="en">five decisions and who makes them in FlyDSL today</span><span lang="zh">五类决策，以及今天在 FlyDSL 里由谁来做</span></div>
    {PLATE_DECISIONS}
    {pair("The author makes the first three decisions directly. LLVM makes the last two with its heuristics. When an author needs a specific result for the last two, the author overrides LLVM through the red path.",
          "前三类决策由作者直接做出。 后两类由 LLVM 用启发式规则来做。 作者如果需要后两类的特定结果， 就通过红色路径覆盖 LLVM 的决定。", cls="caption")}
  </figure>
  {pair(f"The red path is common. In the FlyDSL kernels at {main_tree}, 34 of 180 kernel files use fences (257 times in total). 44 files place waits by hand (186 times). 13 files use inline assembly, and 12 files set the wave priority.",
        f"红色路径很常见。 在 {main_tree} 的 FlyDSL kernel 里， 180 个 kernel 文件中有 34 个使用 fence（共 257 次）。 44 个文件手工放置 wait（共 186 次）。 13 个文件使用 inline 汇编， 12 个文件设置 wave 优先级。")}
  {pair(f'The kernel comments give the reasons. One kernel uses inline assembly so that LLVM does not see a memory operation and does not add a wait ({fp4}). Another kernel adds a fence and gives this reason: "the scheduler may hoist them above s_barrier: a race" ({moe}). That is, LLVM can move the reads before the barrier, and the kernel then reads data that is not ready.',
        f'kernel 里的注释写明了原因。 一个 kernel 用 inline 汇编， 是为了让 LLVM 看不到一次访存， 从而不插入 wait（{fp4}）。 另一个 kernel 加了 fence， 注释写的是： "调度器可能把它们提到 s_barrier 之前： 这是竞争"（{moe}）。 意思是： LLVM 可能把读取移到 barrier 之前， kernel 就会读到还没准备好的数据。')}
  {pair(f"Sometimes the override needs a change to LLVM itself. To keep one accumulator in AGPRs, FlyDSL patched the register allocator and the instruction selector of LLVM ({pr1368}). Then the patch was reverted, because it caused spills in a kernel of another team ({pr1477}).",
        f"有时覆盖 LLVM 需要修改 LLVM 本身。 为了把一个累加器固定在 AGPR 里， FlyDSL 修改了 LLVM 的寄存器分配和指令选择（{pr1368}）。 后来这个补丁被回滚， 因为它让另一个团队的 kernel 出现了 spill（{pr1477}）。")}
  {pair("This problem is larger for an agent than for a person. An agent improves a kernel with small edits and measurements. This method works only if a small edit has a small, local effect. A heuristic that decides again for the whole kernel breaks this condition.",
        '这个问题对 agent 比对人更严重。 agent 靠小改动加测量来改进 kernel。 这个方法只有在 "小改动只产生小的局部影响" 时才有效。 一个会对整个 kernel 重新做决定的启发式规则， 破坏了这个条件。')}
  {pair(f"Here is an example from our experiments. The FP8 8-wave GEMM has four <code>s_setprio</code> instructions. Their documented job is only to set the wave priority. When we removed them, the register use changed, and the kernel got 6 spills. The cause is one line in LLVM: LLVM treats <code>s_setprio</code> as a hard limit for instruction order ({siinstr}). Thus one hint has two effects, and the kernel source does not show the second effect.",
        f"下面是我们实验里的一个例子。 FP8 8-wave GEMM 里有 4 条 <code>s_setprio</code> 指令。 按文档， 它们只负责设置 wave 优先级。 我们去掉它们之后， 寄存器使用发生了变化， kernel 多出 6 个 spill。 原因是 LLVM 里的一行代码： LLVM 把 <code>s_setprio</code> 当作指令顺序的硬边界（{siinstr}）。 所以一个提示有两种效果， 而 kernel 源码里看不到第二种。")}
</section>
"""

    s04 = f"""
<section id="redundancy">
  <h2><span class="n">04</span><span lang="en">Where the compiler repeats work</span><span lang="zh">编译器在哪里重复工作</span></h2>
  {pair("The decision table also explains other repeated work in the compiler. In each item below, the compiler keeps the same information twice, or makes the same decision twice.",
        "这张决策表也能解释编译器里其他的重复工作。 下面每一项里， 编译器都把同一份信息保存了两次， 或者把同一个决定做了两次。")}
  {pair("Two languages in one function, with a hidden rule", "一个函数里有两种语言，靠隐藏规则区分", tag="h3")}
  {pair(f"A FlyDSL kernel mixes two kinds of Python. Some Python runs once, when the compiler reads the kernel. Other Python becomes loops and branches on the GPU. A tool of 1,928 lines decides which is which from the type of each value ({ast_rw}). The source code does not show this rule, and the rule can fail without an error.</p>\n  <p lang=\"en\">For example, {pr1436} fixed a case in which <code>if flag: return a</code>, followed by <code>return b</code>, always returned <code>b</code>. Because of this bug, one production kernel ignored one of its inputs from the day it was written. The agent guide of the repository lists four rules that exist only because of this tool ({claude}).",
        f"一个 FlyDSL kernel 里混着两种 Python。 一种在编译器读 kernel 时执行一次。 另一种会变成 GPU 上的循环和分支。 一个 1928 行的工具根据每个值的类型来判断哪段属于哪种（{ast_rw}）。 源码里看不出这条规则， 规则出错时也不报错。</p>\n  <p lang=\"zh\">例如， {pr1436} 修复了这样一个问题： <code>if flag: return a</code> 后面接 <code>return b</code> 时， 总是返回 <code>b</code>。 因为这个 bug， 一个生产 kernel 从写成那天起就一直忽略它的一个输入。 仓库给 agent 的指南列出了四条规则， 它们只因为这个工具才存在（{claude}）。")}
  {pair("Layouts built as compiler objects, then deleted", "layout 先建成编译器对象，再删掉", tag="h3")}
  {pair(f"Most layouts are known before the kernel runs. Thus the compiler can calculate them with simple integer arithmetic. Instead, FlyDSL records each layout step as an MLIR operation, and later deletes most of these operations. For <code>fp8_gemm_4wave</code>, the trace builds {int(fp8['origin_ir_ops']):,} operations, and only {int(fp8['after_layout_lowering_ops']):,} remain after the layout steps. This build-and-delete work is most of the {float(fp8['trace_s']):.2f} s of the trace. A pure-Python version of the same layout algebra, {pycute}, has about 760 lines.",
        f"大多数 layout 在 kernel 运行之前就已经确定， 所以编译器用简单的整数运算就能算出来。 但 FlyDSL 把每一步 layout 运算都记录成一个 MLIR 操作， 之后再删掉其中的大部分。 对 <code>fp8_gemm_4wave</code>， trace 建出 {int(fp8['origin_ir_ops']):,} 个操作， layout 步骤之后只剩 {int(fp8['after_layout_lowering_ops']):,} 个。 这些先建后删的工作占了 {float(fp8['trace_s']):.2f} s trace 时间的大部分。 同一套 layout 代数的纯 Python 版本 {pycute} 只有大约 760 行。")}
  {pair("Registers stored as memory, then converted back", "寄存器先当成内存，再转换回来", tag="h3")}
  {pair(f"FlyDSL first creates register data as memory objects in a special register address space. Then two compiler steps (about 900 lines) convert these objects back to register values before LLVM sees them ({passes}). This copies a habit of C++ CuTe, which declares register arrays. A compiler can treat a register tile as a value from the start.",
        f"FlyDSL 先把寄存器数据建成特殊寄存器地址空间里的内存对象。 然后用两个编译步骤（约 900 行）在交给 LLVM 之前把它们转换回寄存器值（{passes}）。 这是照搬 C++ CuTe 声明寄存器数组的习惯。 编译器完全可以从一开始就把寄存器 tile 当作一个值。")}
  {pair("Each call traces again, and each process compiles the launcher again", "每次调用都重新 trace，每个进程都重新编译启动代码", tag="h3")}
  {pair(f"Before each launch, a <code>@flyc.jit</code> call traces the launcher and the kernel, prints the result as text and hashes the text. Only then can it look in the compile cache ({jitf}). This makes the cache key correct. But a cache hit on <code>fp8_gemm_4wave</code> still costs {float(fp8['warm_inmem_hit_call_s']):.2f} s. Also, each process compiles the host launcher again with MLIR and LLVM at full optimization, 22–33 ms per kernel. The launcher only fills an argument buffer and calls <code>hipModuleLaunchKernel</code>.",
        f"每次启动前， <code>@flyc.jit</code> 都要先 trace 启动代码和 kernel， 把结果打印成文本， 再对文本做哈希。 然后才能查编译缓存（{jitf}）。 这样缓存 key 是正确的， 但 <code>fp8_gemm_4wave</code> 命中缓存的一次调用仍要 {float(fp8['warm_inmem_hit_call_s']):.2f} s。 另外， 每个进程都要用 MLIR 和 LLVM 以最高优化级别重新编译一次 host 端启动代码， 每个 kernel 22–33 ms。 而启动代码只做两件事： 填写参数缓冲区， 调用 <code>hipModuleLaunchKernel</code>。")}
  {pair("Settings reach LLVM through four paths, and some have no effect", "设置通过四条路径到达 LLVM，有的没有效果", tag="h3")}
  {pair(f"The LLVM skill file of the repository describes four paths that send settings from Python to LLVM. On AMD GPUs, one of these paths is ignored without a message. The compile-hint dictionary accepts misspelled keys, and a misspelled key still changes the cache key. An optimization-level variable is checked but never used ({skill3}, {skill8}). Our experiment found one more case: in <code>fp8_gemm_4wave</code>, the occupancy hints give the same ISA as no hints.",
        f"仓库里的 llvm skill 文件描述了从 Python 向 LLVM 传递设置的四条路径。 在 AMD GPU 上， 其中一条会被直接忽略， 没有任何提示。 compile-hint 字典接受拼错的 key， 而拼错的 key 仍会改变缓存 key。 一个优化级别变量会被检查， 但从来没有被使用（{skill3}、{skill8}）。 我们的实验又发现一个： 在 <code>fp8_gemm_4wave</code> 里， 写了 occupancy 提示和不写生成的 ISA 完全相同。")}
</section>
"""

    s05 = f"""
<section id="redesign">
  <h2><span class="n">05</span><span lang="en">The redesign: one decider for each decision</span><span lang="zh">重新设计：每个决策只有一个决策者</span></h2>
  {pair("The redesign follows from the decision table. Each decision has two options. In the first option, the author makes the decision, and the compiler obeys and checks it. In the second option, the author leaves the decision to the compiler. The compiler then makes it in a fixed, repeatable way and reports its choice. A decision never has two deciders.",
        "重新设计直接从决策表推出。 每个决策有两种选择。 第一种： 作者做决定， 编译器照做并检查。 第二种： 作者把决定交给编译器， 编译器用固定、 可重复的方式做出决定， 并报告它的选择。 一个决策永远不会有两个决策者。")}
  <figure class="plate wide">
    <div class="plate-meta">Plate II · <span lang="en">the current path, and the path after the redesign</span><span lang="zh">现在的路径，以及重新设计后的路径</span></div>
    {plate2}
    {pair("Top: the current path. It shows the costs from section 10, and the red path where the author overrides LLVM. Bottom: the redesign. Python only generates the kernel. A small instruction format close to the hardware (KIR) goes through a series of steps. Each step makes one decision and writes it into a report.",
          "上： 现在的路径。 图中标出了第 10 节测到的开销， 以及作者覆盖 LLVM 的红色路径。 下： 重新设计后的路径。 Python 只负责生成 kernel。 一种接近硬件的小型指令格式（KIR）依次经过若干步骤。 每个步骤只做一个决定， 并把它写进报告。", cls="caption")}
  </figure>
  {pair("Five design rules follow from this principle:", "由这个原则可以得出五条设计规则：")}
  <ol class="list" lang="en">
    <li><b>Python only generates the kernel.</b> Python loops and branches always run at compile time. GPU loops and branches use explicit functions: <code>fx.loop</code> and <code>fx.cond</code>. A GPU value in a Python <code>if</code> causes an error. JAX uses the same model ({jax}). This rule removes the 1,928-line tool and its four extra rules.</li>
    <li><b>Python calculates the layouts.</b> Most layouts are known at compile time, so Python calculates them. Each layout is then a value that you can print and test. Only the parts that depend on runtime values become compiler operations.</li>
    <li><b>A small instruction format close to the hardware, here called KIR.</b> The type of each value tells if all threads of a wave share it (an SGPR) or each thread has its own copy (a VGPR). A register tile records which thread holds which element. A memory copy returns a token, and each wait names the token that it waits for. Thus the compiler knows which data each wait protects, and it never moves a read before its wait.</li>
    <li><b>A repeatable backend that reports.</b> The backend selects instructions from a table. The table comes from AMD's {isa_xml_en}, which already cover CDNA3, CDNA4 and CDNA5. In explicit code, where the author sets the order, the backend keeps that order. In automatic code, where the author does not, it uses a standard list scheduler with a published table of instruction costs. It assigns registers and keeps the pins of the author. It adds waits from the tokens, and it uses a table to keep the gaps that hazards need. Each step writes its choices into a report that the agent can read.</li>
    <li><b>Only settings that cannot change the results.</b> The results then depend only on data flow and tokens. Thus the instruction order, the pins, the number of pipeline stages and the loop unrolling can change only the speed. Agents and autotuners can search these settings without a correctness check for each setting.</li>
  </ol>
  <ol class="list" lang="zh">
    <li><b>Python 只负责生成 kernel。</b> Python 的循环和分支永远在编译期执行。 GPU 上的循环和分支用显式函数 <code>fx.loop</code> 和 <code>fx.cond</code> 表示。 在 Python 的 <code>if</code> 里用 GPU 上的值会报错。 JAX 用的是同样的模型（{jax}）。 这条规则让那个 1928 行的工具和它带来的四条规则一起消失。</li>
    <li><b>layout 由 Python 计算。</b> 大多数 layout 在编译期就确定了， 所以由 Python 来算。 这样每个 layout 都是一个可以打印、 可以测试的值。 只有依赖运行时值的部分才变成编译器操作。</li>
    <li><b>一种接近硬件的小型指令格式， 这里叫 KIR。</b> 每个值的类型说明它是整个 wave 共享（SGPR）， 还是每个线程各有一份（VGPR）。 寄存器 tile 记录哪个线程持有哪个元素。 一次内存拷贝会返回一个 token， 每个 wait 写明它等待哪个 token。 所以编译器知道每个 wait 保护的是哪些数据， 永远不会把读取移到对应的 wait 之前。</li>
    <li><b>一个可重复、 会报告的后端。</b> 后端从一张表里选择指令。 这张表来自 AMD 的{isa_xml_zh}， 它已经覆盖 CDNA3、 CDNA4 和 CDNA5。 在显式代码里（由作者决定顺序）， 后端保持作者写下的顺序。 在自动代码里（作者不决定顺序）， 后端使用标准的列表调度方法和一张公开的指令开销表。 后端分配寄存器， 并保留作者的 pin。 它根据 token 插入 wait， 并按表格留出 hazard 需要的间隔。 每个步骤都把自己的选择写进 agent 能读的报告。</li>
    <li><b>只保留不能改变结果的设置。</b> 这样结果只取决于数据流和 token。 所以指令顺序、 pin、 流水线级数和循环展开只会改变速度。 agent 和 autotuner 可以搜索这些设置， 不需要对每个设置做一次正确性检查。</li>
  </ol>
  {pair("What this means for MLIR and LLVM:", "这对 MLIR 和 LLVM 意味着什么：")}
  <ul lang="en">
    <li><b>MLIR can be removed.</b> Its compiler steps take only {mlir_lo:.0f}–{mlir_hi:.0f}% of the cold compile of the large kernels (section 10). Its real costs are the trace of each operation, the build-and-delete layout work, and a C++ rebuild for each change to a compiler step.</li>
    <li><b>LLVM can be removed in two steps.</b> Step 1: for explicit kernels, stop using the heuristics of LLVM, and use LLVM only to encode the instructions. AMD's hipBLASLt TensileLite already works this way: Python writes assembly, and <code>amdclang++ -x assembler</code> encodes it ({tensile}). Step 2: when the new backend passes the kernel test set, generate an encoder from the ISA files, and stop using LLVM for these kernels.</li>
  </ul>
  <ul lang="zh">
    <li><b>MLIR 可以去掉。</b> 它的编译步骤只占大 kernel 冷编译时间的 {mlir_lo:.0f}–{mlir_hi:.0f}%（第 10 节）。 它真正的成本是逐个操作的 trace、 先建后删的 layout 工作， 以及每改一个编译步骤都要重新编译 C++。</li>
    <li><b>LLVM 分两步去掉。</b> 第一步： 对显式 kernel， 不再使用 LLVM 的启发式规则， 只用 LLVM 给指令编码。 AMD 的 hipBLASLt TensileLite 已经这样做了： Python 写汇编， <code>amdclang++ -x assembler</code> 负责编码（{tensile}）。 第二步： 等新后端通过 kernel 测试集之后， 从 ISA 文件生成编码器， 这些 kernel 就不再使用 LLVM。</li>
  </ul>
  <div class="callout note">
    {pair("<b>What we did not do.</b> We did not build KIR or the new backend. The experiments below test the facts that the design depends on. They do not measure the speed of the new design.",
          "<b>我们没有做的事。</b> 我们没有实现 KIR 和新后端。 下面的实验检验的是这个设计所依赖的事实， 不是新设计本身的速度。")}
  </div>
</section>
"""

    s06 = f"""
<section id="method">
  <h2><span class="n">06</span><span lang="en">How we tested it on the overlay</span><span lang="zh">怎样在 overlay 上检验</span></h2>
  {pair("The design depends on four premises. Each premise can be false. For each one, we name the result that would show that it is false:",
        "这个设计依赖四个前提。 每个前提都可能是错的。 对每个前提， 我们写出能证明它错误的结果：")}
  <ul lang="en">
    <li><b>A.</b> The decisions that authors make by hand about register use and instruction order give real speed. LLVM does not find this speed by itself. <i>False if:</i> removing these decisions changes nothing.</li>
    <li><b>B.</b> Where authors made no such decision, safe settings that nobody tuned still give extra speed. <i>False if:</i> no setting changes the time.</li>
    <li><b>C.</b> Agents spend most of their effort on things that a compiler could show or check. <i>False if:</i> agents rarely read compiler output, and their speedups come from compiler settings.</li>
    <li><b>D.</b> The compile work before each launch costs time in each iteration of an agent. <i>False if:</i> a cache hit costs no time.</li>
  </ul>
  <ul lang="zh">
    <li><b>A.</b> 作者手工做出的寄存器使用和指令顺序决策带来了实际的速度， LLVM 自己找不到这部分速度。 <i>证伪条件：</i> 去掉这些决策后什么都没变。</li>
    <li><b>B.</b> 在作者没有做这类决策的地方， 没人调过的安全设置仍能带来额外的速度。 <i>证伪条件：</i> 没有哪个设置能改变耗时。</li>
    <li><b>C.</b> agent 的大部分精力花在编译器本可以展示或检查的事情上。 <i>证伪条件：</i> agent 很少读编译器输出， 而且提速来自编译器设置。</li>
    <li><b>D.</b> 每次启动前的编译工作， 在 agent 的每次迭代里都要花时间。 <i>证伪条件：</i> 命中缓存不花时间。</li>
  </ul>
  {pair("How we measured experiments A and B", "实验 A 和 B 怎样测量", tag="h3")}
  <ul lang="en">
    <li>Each arm is one patch to FlyDSL <code>cb770e9a</code>, the version that FlyBench v1 uses.</li>
    <li>Arms that remove hints add a few lines to {init_py}. These lines change one kind of hint into an operation that does nothing, or they change the LLVM options. The register and occupancy arms change one line of the kernel.</li>
    <li>The FlyBench v1 evaluator scored each arm. It is the same program that scores agent submissions.</li>
    <li>The evaluator runs the stock kernel and the arm in new processes, in the order stock, arm, arm, stock (ABBA). It does this six times. This order cancels slow changes in GPU speed.</li>
    <li>Before each cold call, it reads 512 MB of other data. Thus the caches do not keep data from the previous call.</li>
    <li>It uses the public test shapes and new random shapes in each run.</li>
    <li>An arm must pass the tests of FlyDSL and a correctness check on the new shapes. If it fails, it gets no speed result.</li>
    <li>Each arm ran as one job on one MI355X GPU through the overlay. The job did not call an AI model.</li>
  </ul>
  <ul lang="zh">
    <li>每个 arm 是针对 FlyDSL <code>cb770e9a</code> 的一个补丁， 这是 FlyBench v1 使用的版本。</li>
    <li>去掉提示的 arm 在 {init_py} 里加几行代码。 这几行代码把某一类提示变成什么都不做的操作， 或者修改 LLVM 选项。 寄存器和 occupancy 两个 arm 修改 kernel 里的一行。</li>
    <li>每个 arm 都由 FlyBench v1 评测器打分， 也就是给 agent 提交打分的同一个程序。</li>
    <li>评测器在新进程里按 stock、 arm、 arm、 stock 的顺序（ABBA）运行， 一共重复 6 次。 这个顺序能抵消 GPU 速度的缓慢变化。</li>
    <li>每次冷调用前， 评测器先读 512 MB 其他数据， 这样缓存里不会留下上一次调用的数据。</li>
    <li>每次运行都使用公开的测试 shape 和新随机抽取的 shape。</li>
    <li>arm 必须通过 FlyDSL 自己的测试和新 shape 上的正确性检查， 否则不给速度结果。</li>
    <li>每个 arm 作为一个作业， 通过 overlay 在一张 MI355X GPU 上运行。 作业不调用 AI 模型。</li>
  </ul>
  {pair("Controls", "对照", tag="h3")}
  {pair(f"Each job also compiled the stock kernel and the arm at two shapes, and compared the ISA. All arms changed the ISA except three. The first is the occupancy arm (section 07). The other two are null controls: RMSNorm, which has no hints to remove, and the A16W16 GEMM with the added lines but nothing removed.</p>\n  <p lang=\"en\">The null controls measured {r['x12-nosched']:.3f}× and {r['x15-null']:.3f}×. This is the noise of our setup. Thus we cannot see effects smaller than about 0.5%.",
        f"每个作业还在两个 shape 上分别编译 stock kernel 和 arm， 并比较 ISA。 除了三个 arm， 其他 arm 都改变了 ISA。 第一个是 occupancy arm（第 07 节）。 另外两个是空白对照： 一个是没有提示可去的 RMSNorm， 另一个是加了那几行代码、 但什么都没去掉的 A16W16 GEMM。</p>\n  <p lang=\"zh\">两个空白对照测得 {r['x12-nosched']:.3f}× 和 {r['x15-null']:.3f}×。 这就是我们这套测量的噪声， 所以小于约 0.5% 的效果看不出来。")}
  {pair("The register arm changes code only for 256×256 tiles. Only two of its 19 timed shapes use these tiles. Thus we report this arm at these two shapes.",
        "寄存器 arm 只改变 256×256 tile 的代码。 它的 19 个计时 shape 里只有 2 个用这种 tile， 所以我们只在这两个 shape 上报告它。")}
  <div class="callout note">
    {pair("<b>Limit.</b> The overlay puts single-GPU jobs next to other jobs on the same node, so we did not measure on idle GPUs. The ABBA order removes slow drift between stock and arm. It does not remove noise from other jobs. The intervals below include this noise. The agent data in experiment C comes from earlier FlyBench runs, not from new agent jobs.",
          "<b>局限。</b> overlay 会把单卡作业和其他作业放在同一个节点上， 所以我们不是在空闲的 GPU 上测量的。 ABBA 顺序能消除 stock 和 arm 之间的缓慢漂移， 但消除不了其他作业带来的噪声。 下面的区间已经包含这部分噪声。 实验 C 的 agent 数据来自之前的 FlyBench 运行， 不是新跑的 agent 作业。")}
  </div>
</section>
"""

    s07 = f"""
<section id="ablation">
  <h2><span class="n">07</span><span lang="en">Experiment A · take one decision away from the author</span><span lang="zh">实验 A · 拿走作者的一个决策</span></h2>
  {pair("<b>What we did.</b> Each arm removes one decision that the author made about register use or instruction order. Then the default heuristics of LLVM make that decision. If these heuristics were as good as the author, each bar in Plate III would be at 1.0.",
        "<b>做了什么。</b> 每个 arm 去掉作者在寄存器使用或指令顺序上的一个决策， 改由 LLVM 的默认启发式规则来做。 如果这些规则和作者一样好， Plate III 里的每根条都应该在 1.0。")}
  <figure class="plate wide">
    <div class="plate-meta">Plate III · <span lang="en">speed ratio when one decision of the author is removed</span><span lang="zh">去掉作者的一个决策后的速度比</span></div>
    {plate3}
    {legend([("ink2", "instruction order: fences, groups", "指令顺序：fence、交错组"), ("violet", "wave priority", "wave 优先级"),
             ("amber", "register use", "寄存器使用"), ("hair", "null control", "空白对照")])}
    {pair("Each bar is the speed ratio over all timed and new shapes. The black line is the 95% interval over the six ABBA pairs. The two register rows show the only two shapes that this arm changes. A cross means that the tests of FlyDSL or the correctness check rejected the output.",
          "每根条是所有计时 shape 和新 shape 上的速度比。 黑线是 6 组 ABBA 的 95% 区间。 两行寄存器结果是这个 arm 唯一会改变的两个 shape。 叉号表示 FlyDSL 的测试或正确性检查拒绝了输出。", cls="caption")}
  </figure>
  <table>
    <tr>{exp_head}</tr>
    {abl_table}
  </table>
  {pair("Result 1: the decisions of the author give 5–16%, and LLVM does not find this speed", "结果 1：作者的决策值 5–16%，LLVM 找不到这部分速度", tag="h3")}
  {pair(f"Without the fences and interleave groups, the preshuffle GEMM became {pct('x01-nosched')}% slower ({r['x01-nosched']:.3f}×). The MX preshuffle GEMM became {pct('x03-nosched')}% slower, and the sliding-window attention became {pct('x05-nosched')}% slower. In the preshuffle GEMM, the interleave groups cause all of the loss. Without only the groups, the speed ratio is {r['x14-nogroup']:.3f}×, the same as without all hints.",
        f"去掉 fence 和交错组之后， preshuffle GEMM 慢了 {pct('x01-nosched')}%（{r['x01-nosched']:.3f}×）。 MX preshuffle GEMM 慢了 {pct('x03-nosched')}%， 滑窗 attention 慢了 {pct('x05-nosched')}%。 在 preshuffle GEMM 里， 全部损失都来自交错组： 只去掉交错组时速度比是 {r['x14-nogroup']:.3f}×， 和去掉全部提示一样。")}
  {pair(f"Without <code>s_setprio</code>, the two 8-wave GEMMs became {pct('x06-noprio')}% and {pct('x07-noprio')}% slower. For the first one, the ISA shows more than a priority change: the kernel got VGPR spills, and it had no spills before.",
        f"去掉 <code>s_setprio</code> 之后， 两个 8-wave GEMM 分别慢了 {pct('x06-noprio')}% 和 {pct('x07-noprio')}%。 对第一个 kernel， ISA 显示的不只是优先级变化： 原来没有 spill， 现在出现了 VGPR spill。")}
  {pair(f"In the FP8 4-wave GEMM, we let LLVM assign the registers of the accumulators, instead of the AGPR pins of the author. The kernel became {(1 - x10f0['ratio']) * 100:.0f}% slower at 16384³ and {(1 - x10p1['ratio']) * 100:.0f}% slower at 8192³. The spills went from {s_f0['meta'].get('vgpr_spill_count')} to {c_f0['meta'].get('vgpr_spill_count')}, and the moves between AGPRs and VGPRs went from {s_f0['v_accvgpr']:,} to {c_f0['v_accvgpr']:,}. The ISA also shows a fact that nobody had reported: even the pinned kernel has {s_f0['meta'].get('vgpr_spill_count')} spills at that size.",
        f"在 FP8 4-wave GEMM 里， 我们让 LLVM 来分配累加器的寄存器， 不再使用作者的 AGPR pin。 kernel 在 16384³ 上慢了 {(1 - x10f0['ratio']) * 100:.0f}%， 在 8192³ 上慢了 {(1 - x10p1['ratio']) * 100:.0f}%。 spill 从 {s_f0['meta'].get('vgpr_spill_count')} 个增加到 {c_f0['meta'].get('vgpr_spill_count')} 个， AGPR 和 VGPR 之间的搬运指令从 {s_f0['v_accvgpr']:,} 条增加到 {c_f0['v_accvgpr']:,} 条。 ISA 还显示了一个之前没人报告过的事实： 即使用了 pin， 这个尺寸上也有 {s_f0['meta'].get('vgpr_spill_count')} 个 spill。")}
  {pair("Without the occupancy hints of the same kernel, nothing changed. The ISA was the same at both shapes, so the default of LLVM already matched the hints. A hint with no effect does no harm. But no tool tells the author or the agent that the hint has no effect.",
        "去掉同一个 kernel 的 occupancy 提示后， 什么都没变。 两个 shape 上的 ISA 完全相同， 说明 LLVM 的默认值本来就和这些提示一致。 没有效果的提示不会造成损害， 但没有任何工具告诉作者或 agent 它没有效果。")}
  {pair('Result 2: in two kernels, the "scheduling" fences kept the results correct', '结果 2：两个 kernel 的"调度" fence 在保证结果正确', tag="h3")}
  {pair(f"Without the fences, the A16W16 GEMM failed {f02} of {n02} FlyDSL tests. The dense flash attention failed {f04} of {n04} tests: some log-sum-exp values (a value for each row that attention saves for later steps) were off by up to 0.43, and some outputs were NaN (not a number).",
        f"去掉 fence 之后， A16W16 GEMM 在 FlyDSL 的 {n02} 个测试里失败了 {f02} 个。 dense flash attention 的 {n04} 个测试里失败了 {f04} 个。 部分 log-sum-exp 值（attention 为后续步骤保存的每行数值）最多差 0.43， 部分输出是 NaN（无效数值）。")}
  {pair(f"We repeated the A16W16 arm in a new job. The same test failed, but with different wrong elements: {mis_a:,} and {mis_b:,} of {mis_total:,}. A rounding error gives the same wrong elements in each run. A race gives different ones. With the added lines but nothing removed, the same kernel passed all of its tests (null control, {x15['ratio']:.3f}×).",
        f"我们在新作业里重复了 A16W16 这个 arm。 失败的是同一个测试， 但错的元素不同： {mis_total:,} 个元素里分别错了 {mis_a:,} 个和 {mis_b:,} 个。 舍入误差每次错的是同样的元素， 竞争每次错的元素不同。 加了那几行代码、 但什么都不去掉时， 同一个 kernel 通过了全部测试（空白对照， {x15['ratio']:.3f}×）。")}
  {pair(f'A scheduling hint should change only the speed. Here it changed the results. We did not find which instruction moved.</p>\n  <p lang="en">But the kernels show the probable cause. They control the order of memory copies and shared-memory reuse with waits and barriers that the author placed by hand. LLVM does not connect these waits and barriers to the reads that they protect. A FlyDSL comment in another kernel describes this risk: "the scheduler may hoist them above s_barrier: a race" ({moe}).',
        f'调度提示本应只改变速度， 这里却改变了结果。 我们没有找出具体是哪条指令被移动了。</p>\n  <p lang="zh">但 kernel 本身指出了可能的原因： 它们用作者手工放置的 wait 和 barrier 来控制内存拷贝和共享内存复用的顺序。 LLVM 不会把这些 wait 和 barrier 与它们保护的读取联系起来。 FlyDSL 另一个 kernel 的注释写的正是这种风险： "调度器可能把它们提到 s_barrier 之前： 这是竞争"（{moe}）。')}
  {pair(f"In the flash attention, we also removed only the interleave groups and kept the fences. The kernel stayed correct, at {x13['ratio']:.3f}× {B.ci('x13-nogroup')}. Thus the fences, not the groups, keep the results correct.",
        f"在 flash attention 里， 我们还试了只去掉交错组、 保留 fence。 kernel 的结果仍然正确， 速度比是 {x13['ratio']:.3f}× {B.ci('x13-nogroup')}。 所以保证结果正确的是 fence， 不是交错组。")}
  {pair("This is the most important result of this post, and it is not about speed. Today, an autotuner or an agent cannot safely change a scheduling hint, because some of these hints are correctness rules that look like speed hints.",
        "这是本文最重要的结果， 而且它和速度无关。 今天， autotuner 或 agent 不能放心地修改调度提示， 因为其中一些提示看起来是为了速度， 实际上是保证正确性的规则。")}
  {pair("Result 3: on single shapes, the free order of LLVM is sometimes better", "结果 3：在单个 shape 上，LLVM 的自由排序有时更好", tag="h3")}
  {pair(f"The same preshuffle arm is {pct('x01-nosched')}% slower in total. But it is {x01f0:.1f}% faster at 16384³ and {x01p1:.1f}% faster at 8192³. Thus the interleave pattern of the author is a good fixed rule, but it is not the best rule for each shape. Plate V shows this for all public shapes.",
        f"同一个 preshuffle arm 整体慢了 {pct('x01-nosched')}%， 但在 16384³ 上快了 {x01f0:.1f}%， 在 8192³ 上快了 {x01p1:.1f}%。 所以作者的交错模板是一个不错的固定规则， 但不是每个 shape 上最好的规则。 Plate V 显示了所有公开 shape 的情况。")}
  {pair("Result 4: two small effects on the flash attention", "结果 4：flash attention 上的两个小效果", tag="h3")}
  {pair(f"Without <code>s_setprio</code>, the flash attention ran at {x08['ratio']:.3f}× {B.ci('x08-noprio')} in total. But it was up to {(1 - x08['min_shape_ratio']) * 100:.1f}% slower on single shapes, and {len(x08['regressions'])} shapes were slower by more than their noise level.",
        f"去掉 <code>s_setprio</code> 后， flash attention 整体速度比是 {x08['ratio']:.3f}× {B.ci('x08-noprio')}。 但在单个 shape 上最多慢 {(1 - x08['min_shape_ratio']) * 100:.1f}%， 有 {len(x08['regressions'])} 个 shape 的变慢超出了噪声下限。")}
  {pair(f"The author of this kernel turned off the post-RA scheduler of LLVM to protect the order that the author wrote ({flash_src}). The post-RA scheduler is a second LLVM step that changes the instruction order after register assignment. When we turned it on again, the kernel ran at {x09['ratio']:.3f}× {B.ci('x09-postmischedon')}. No shape was slower by more than its noise level, and the best shape was {(x09['max_shape_ratio'] - 1) * 100:.1f}% faster. The ISA has {nop_s - nop_c} fewer <code>s_nop</code> instructions ({nop_s} → {nop_c}). An <code>s_nop</code> is an empty instruction that fills a hazard gap.",
        f"这个 kernel 的作者关掉了 LLVM 的 post-RA 调度器， 用来保护作者写下的顺序（{flash_src}）。 post-RA 调度器是 LLVM 在分配寄存器之后再调整一次指令顺序的步骤。 我们重新打开它之后， kernel 的速度比是 {x09['ratio']:.3f}× {B.ci('x09-postmischedon')}。 没有 shape 的变慢超出噪声下限， 最好的 shape 快了 {(x09['max_shape_ratio'] - 1) * 100:.1f}%。 ISA 里少了 {nop_s - nop_c} 条 <code>s_nop</code>（{nop_s} → {nop_c}）。 <code>s_nop</code> 是填充 hazard 间隔的空指令。")}
  {pair(f"This gain is about the size of the {x09['noise_floor'] * 100:.1f}% noise level of this kernel. Thus FlyBench would not count it as an improvement. The author could not see if this setting helps. The notes of FlyDSL say that three related flash attention kernels use different values for this setting, with no comment ({arch}).",
        f"这个收益和这个 kernel {x09['noise_floor'] * 100:.1f}% 的噪声下限差不多大， 所以 FlyBench 不会把它算作提升。 作者看不到这个设置到底有没有帮助。 FlyDSL 自己的笔记写着： 三个相关的 flash attention kernel 在这个设置上取值不同。 这些 kernel 里也没有注释说明原因（{arch}）。")}
</section>
"""

    s08 = f"""
<section id="headroom">
  <h2><span class="n">08</span><span lang="en">Experiment B · settings that nobody tuned</span><span lang="zh">实验 B · 没人调过的设置</span></h2>
  {pair("<b>What we did.</b> The normalization, softmax and MoE kernels have no hints about instruction order. LLVM makes all of these decisions with its default heuristics. We tried two other LLVM scheduling strategies, <code>max-ilp</code> and <code>max-memory-clause</code>. We also turned off the instruction schedulers of LLVM. The instruction order is then close to the order in the source.",
        "<b>做了什么。</b> normalization、 softmax 和 MoE 这些 kernel 没有任何关于指令顺序的提示。 这些决策全部由 LLVM 用默认的启发式规则来做。 我们试了 LLVM 另外两种调度策略： <code>max-ilp</code> 和 <code>max-memory-clause</code>。 我们还关掉了 LLVM 的指令调度器， 这时指令顺序接近源码里的顺序。")}
  <figure class="plate wide">
    <div class="plate-meta">Plate IV · <span lang="en">speed ratio with one LLVM setting that nobody tuned</span><span lang="zh">换一个没人调过的 LLVM 设置后的速度比</span></div>
    {plate4}
    {pair("Green: a different <code>amdgpu-sched-strategy</code>. Red: the schedulers of LLVM turned off. A cross means that the evaluator rejected an output. In the repeated MoE run, the rejected output came from the stock kernel. Most bars are inside the noise level of each kernel.",
          "绿色： 换一种 <code>amdgpu-sched-strategy</code>。 红色： 关掉 LLVM 的调度器。 叉号表示评测器拒绝了某次输出。 在 MoE 的重复实验里， 被拒绝的输出来自 stock kernel。 大多数条都在各自 kernel 的噪声下限以内。", cls="caption")}
  </figure>
  <table>
    <tr>{set_head}</tr>
    {head_table}
  </table>
  {pair("<b>What we saw.</b>", "<b>看到了什么。</b>")}
  <ol class="list" lang="en">
    <li><b>For kernels without a hand-written order, the default of LLVM is good.</b> No strategy was faster by more than the noise level on the normalization kernels and the MoE reduce kernel. Without the schedulers, RMSNorm became {pct('h12-mischedoff')}% slower. Thus the automatic mode of a new compiler needs a real scheduler.</li>
    <li><b>For the hand-ordered preshuffle GEMM, the result is different.</b> Without the schedulers of LLVM, it ran at {r['h11-mischedoff']:.3f}× {B.ci('h11-mischedoff')}, faster than stock. With <code>max-ilp</code>, it ran at {r['h10-stratmaxilp']:.3f}×. But neither result is a clean gain: {len(h11['regressions'])} and {len(h10['regressions'])} shapes became slower by more than the noise level (worst {h11['min_shape_ratio']:.2f}× and {h10['min_shape_ratio']:.2f}×). Thus FlyBench would reject both arms. Even with the hints of the author, the heuristic scheduler of LLVM costs this kernel time. An explicit mode, which keeps the order of the author, predicts this result.</li>
    <li><b>The MoE GEMM sometimes returns wrong results, even without a change.</b> With <code>max-ilp</code>, one timed call failed the correctness check (cosine similarity {cosine('h08-stratmaxilp'):.3f}; the limit is 0.99). We repeated the arm. In the repeat, the stock kernel itself returned a wrong output (cosine similarity {cosine('h08b-stratmaxilp'):.3f}). Thus we cannot blame the setting. The cause is probably a race, but we did not find it. The case shows a different problem: the evaluator catches this bug only in some runs. A race check in the compiler can find this kind of bug before the kernel runs.</li>
  </ol>
  <ol class="list" lang="zh">
    <li><b>对没有手工排序的 kernel， LLVM 的默认值是好的。</b> 在 normalization 和 MoE reduce kernel 上， 没有哪种策略的提速超出噪声下限。 关掉调度器后， RMSNorm 慢了 {pct('h12-mischedoff')}%。 所以新编译器的自动模式需要一个真正的调度器。</li>
    <li><b>对手工排序的 preshuffle GEMM， 结果不同。</b> 关掉 LLVM 的调度器后， 速度比是 {r['h11-mischedoff']:.3f}× {B.ci('h11-mischedoff')}， 比 stock 快。 用 <code>max-ilp</code> 时是 {r['h10-stratmaxilp']:.3f}×。 但两者都不是干净的提升： 分别有 {len(h11['regressions'])} 个和 {len(h10['regressions'])} 个 shape 的变慢超出噪声下限（最差 {h11['min_shape_ratio']:.2f}× 和 {h10['min_shape_ratio']:.2f}×）， 所以 FlyBench 会拒绝这两个 arm。 即使有作者的提示， LLVM 的启发式调度器仍让这个 kernel 变慢。 保持作者顺序的显式模式预言的正是这个结果。</li>
    <li><b>MoE GEMM 即使不做任何改动， 有时也会返回错误结果。</b> 用 <code>max-ilp</code> 时， 有一次计时调用没有通过正确性检查（余弦相似度 {cosine('h08-stratmaxilp'):.3f}， 门限是 0.99）。 我们重复了这个 arm。 在重复实验里， stock kernel 自己返回了错误的输出（余弦相似度 {cosine('h08b-stratmaxilp'):.3f}）。 所以不能怪这个设置。 原因可能是竞争， 但我们没有找到。 这个例子说明了另一个问题： 评测器只能在部分运行里抓到这个 bug。 编译器里的竞争检查能在 kernel 运行之前发现这类 bug。</li>
  </ol>
  <figure class="plate wide">
    <div class="plate-meta">Plate V · <span lang="en">preshuffle-a8, shape by shape: no single order wins on all shapes</span><span lang="zh">preshuffle-a8 逐个 shape 看：没有哪种顺序在所有 shape 上都赢</span></div>
    {plate5}
    {legend([("ink2", "no fences or groups", "去掉 fence 和交错组"), ("red", "no groups", "去掉交错组"),
             ("teal", "max-ilp", "max-ilp"), ("amber", "LLVM schedulers off", "关掉 LLVM 调度器")])}
    {pair(f"Each dot is one public shape in one arm. 1.0 is the stock kernel. For each shape, pick the best of these four arms and stock. Then the {o_n} public shapes get {o_point:.3f}× on average (geometric mean). If we count an arm only where its interval is above 1.0, and use only the lower end of the interval, they get {o_strict:.3f}×.",
          f"每个点是一个公开 shape 在一个 arm 下的速度比。 1.0 是 stock kernel。 为每个 shape 在这四个 arm 和 stock 之间选出最好的一个， {o_n} 个公开 shape 平均得到 {o_point:.3f}×（几何平均）。 如果只在区间整体高于 1.0 时才计入某个 arm， 并且只计区间的下限， 平均得到 {o_strict:.3f}×。", cls="caption")}
  </figure>
  {pair("This choice for each shape gives the minimum gain that a search over safe order settings can find in this one kernel. Today, this search is not safe in kernels where the fences keep the results correct (experiment A).",
        "这种按 shape 的选择， 给出了在这一个 kernel 里搜索安全的顺序设置所能得到的最小收益。 今天， 在 fence 保证结果正确的 kernel 里， 这种搜索并不安全（见实验 A）。")}
</section>
"""

    s09 = f"""
<section id="agents">
  <h2><span class="n">09</span><span lang="en">Experiment C · what agents did in {cost['n']} attempts</span><span lang="zh">实验 C · agent 在 {cost['n']} 次尝试里做了什么</span></h2>
  {pair(f"<b>What we did.</b> The overlay has already run FlyBench many times. In each run, an agent gets one FlyDSL kernel and a time limit. It must make the kernel faster and keep the same results.</p>\n  <p lang=\"en\">We read all the records that still exist. These are {m['runs']} runs with {cost['n']} attempts, {m['receipts']} evaluator receipts, {rv['n']} independent reviews of the best patches and {n_tr} command logs with {nc:,} commands. We publish only totals.",
        f"<b>做了什么。</b> overlay 已经跑过很多次 FlyBench。 每次运行中， agent 拿到一个 FlyDSL kernel 和一个时间限制， 要在结果不变的前提下让 kernel 变快。</p>\n  <p lang=\"zh\">我们读了所有留存下来的记录。 其中有 {m['runs']} 次运行（共 {cost['n']} 次尝试）、 {m['receipts']} 张评测回执、 {rv['n']} 份对最好补丁的独立 review， 以及 {n_tr} 份命令记录（共 {nc:,} 条命令）。 我们只公开汇总数据。")}
  <figure class="plate wide">
    <div class="plate-meta">Plate VI · <span lang="en">where the accepted speedups came from, and what agents spent their commands on</span><span lang="zh">被认可的提速来自哪里，以及 agent 把命令花在了什么上</span></div>
    {pair(f"Share of the {rv['verdicts'].get('legit', 0)} reviews that accept the speedup, by the source that each review names. One patch can name more than one source.",
          f"在 {rv['verdicts'].get('legit', 0)} 份认可提速的 review 里， 提到每种来源的比例。 一个补丁可以有多个来源。", cls="caption")}
    {plate6a}
    {pair(f"Share of all {nc:,} commands, by purpose. Most of the other commands run scripts.",
          f"全部 {nc:,} 条命令按用途的比例。 其余命令大多是运行脚本。", cls="caption")}
    {plate6b}
  </figure>
  {pair(f"<b>Where the speedups come from.</b> The accepted speedups have a median of {rv['legit_reproduced_speedup_median']:.2f}×, as reproduced by the reviewer. They come from four kinds of change: wider memory access, a tile choice per shape, more work splits, and new algorithms. These changes are work split, data movement and instruction choice, the decisions that the author already makes. Order settings and register settings appear in only {src['schedule_or_sync_knob'] * 100:.0f}% and {src['register_or_occupancy_knob'] * 100:.0f}% of the reviews.</p>\n  <p lang=\"en\">Thus the redesign must keep what agents use: Python metaprogramming, layout algebra and atoms (ready-made blocks for copies and matrix instructions). It does not give agents a new source of speed.",
        f"<b>提速来自哪里。</b> 被认可的提速经 review 复现， 中位数是 {rv['legit_reproduced_speedup_median']:.2f}×。 它们来自四类改动： 更宽的访存、 按 shape 选择 tile、 更多的工作切分以填满 GPU， 以及新算法。 这些都属于工作划分、 数据搬运和指令选择， 也就是作者本来就在做的决策。 顺序设置和寄存器设置分别只出现在 {src['schedule_or_sync_knob'] * 100:.0f}% 和 {src['register_or_occupancy_knob'] * 100:.0f}% 的 review 里。</p>\n  <p lang=\"zh\">所以重新设计必须保留 agent 用到的东西： Python 元编程、 layout 代数和 atom（现成的拷贝和矩阵指令模块）。 它不会给 agent 带来新的提速来源。")}
  {pair(f"<b>Where the effort goes.</b> A median attempt used {cost['median_prompt_tokens'] / 1e6:.1f}M prompt tokens, {cost['median_turns']} turns and {cost['median_wall_s'] / 3600:.1f} hours. In {isa_share:.0f}% of the {n_tr} command logs, the agent read the IR or the ISA at least once. In {cc_share:.0f}%, it cleared the FlyDSL compile cache at least once ({by['cache_clear']:,} times in total). Agents do this when they cannot tell if the code that ran is the code that they wrote.</p>\n  <p lang=\"en\">The most common failure in the reviews was a slower result on shapes that the agent did not time ({rv['failed_checklist_items']['per-shape regressions vs stock']} of {rv['n']} reviews). A few reviews found problems that a compiler can find before the run. Some were correctness bugs, for example a missing barrier and an out-of-bounds read. Others were spills, for example a spill setting that made one shape 0.44× as fast. Nine reviews say that agents added their own launch caches to avoid the trace at each call.",
        f"<b>精力花在哪里。</b> 按中位数计， 一次尝试用了 {cost['median_prompt_tokens'] / 1e6:.1f}M prompt token、 {cost['median_turns']} 轮对话、 {cost['median_wall_s'] / 3600:.1f} 小时。 在 {n_tr} 份命令记录里， 有 {isa_share:.0f}% 显示 agent 至少读过一次 IR 或 ISA。 有 {cc_share:.0f}% 显示 agent 至少清过一次 FlyDSL 编译缓存（总共 {by['cache_clear']:,} 次）。 agent 在无法确认 \"运行的代码是不是自己写的代码\" 时， 就会这样做。</p>\n  <p lang=\"zh\">review 里最常见的失败， 是 agent 没有计时的 shape 变慢了（{rv['n']} 份里有 {rv['failed_checklist_items']['per-shape regressions vs stock']} 份）。 少数 review 发现了编译器在运行前就能发现的问题。 有些是正确性 bug， 例如漏掉的 barrier 和越界读取。 有些是 spill， 例如一个让某个 shape 只剩 0.44× 速度的 spill 设置。 有 9 份 review 提到， agent 为了避开每次调用的 trace， 自己加了启动缓存。")}
  <div class="callout">
    <div lang="en">
      <p><b>What agents need from the compiler</b> is mostly what the report and the checks of the redesign give:</p>
      <ul>
        <li>registers, spills and occupancy for each shape, before a GPU run;</li>
        <li>the waits that the compiler added, and the reason for each wait;</li>
        <li>a race check and a bounds check for shared memory;</li>
        <li>a correct cache key;</li>
        <li>a launch path without a new trace.</li>
      </ul>
      <p>None of these needs MLIR or the heuristics of LLVM. They need a compiler that is responsible for its own decisions.</p>
    </div>
    <div lang="zh">
      <p><b>agent 需要编译器提供的东西</b>， 大多正是重新设计里的报告和检查能给的：</p>
      <ul>
        <li>上 GPU 运行之前， 就知道每个 shape 的寄存器、 spill 和 occupancy；</li>
        <li>编译器插入了哪些 wait， 以及每个 wait 的原因；</li>
        <li>共享内存的竞争检查和越界检查；</li>
        <li>正确的缓存 key；</li>
        <li>不需要重新 trace 的启动路径。</li>
      </ul>
      <p>这些都不需要 MLIR， 也不需要 LLVM 的启发式规则。 它们需要的是一个对自己的决定负责的编译器。</p>
    </div>
  </div>
</section>
"""

    s10 = f"""
<section id="overhead">
  <h2><span class="n">10</span><span lang="en">Experiment D · the compile cost before each launch</span><span lang="zh">实验 D · 每次启动前的编译开销</span></h2>
  {pair("<b>What we did.</b> We measured the compile time on the CPU with FlyDSL 0.5.2. We compiled for gfx950 (MI355X) and did not run the kernels. We ran each case three times, and the runs were within 1% of each other.",
        "<b>做了什么。</b> 我们在 CPU 上用 FlyDSL 0.5.2 测量编译时间。 目标是 gfx950（MI355X）， 只编译， 不运行 kernel。 每种情况跑 3 次， 3 次结果相差在 1% 以内。")}
  <figure class="plate wide">
    <div class="plate-meta">Plate VII · <span lang="en">the parts of a cold compile, and the cost of a call on a cache hit</span><span lang="zh">冷编译由哪几部分组成，以及命中缓存时一次调用的开销</span></div>
    {plate7}
    {pair("Each bar shows the parts of one cold compile. The red line under each bar shows the time of one call on a cache hit.",
          "每根条显示一次冷编译的各个部分。 每根条下面的红线， 是命中缓存时一次调用的时间。", cls="caption")}
  </figure>
  <table>
    <tr>{th("Kernel", "Kernel")}{th("Cold compile", "冷编译")}{th("Trace + print", "trace + 打印")}{th("MLIR steps", "MLIR 步骤")}{th("LLVM backend", "LLVM 后端")}{th("Call on a cache hit", "命中缓存的一次调用")}{th("Objects built → kept", "建出的对象 → 留下的")}</tr>
    {compile_table}
  </table>
  {pair(f"<b>What we saw.</b> The LLVM backend is the largest part of a cold compile: {llvm_share} for the GEMM and the attention kernel. The MLIR steps are small. For an agent, the trace is the important cost, because it comes at each call. It also comes on a cache hit: {float(fp8['warm_inmem_hit_call_s']):.2f} s per call for the FP8 GEMM and {float(fa['warm_inmem_hit_call_s']) * 1000:.0f} ms for the attention kernel. Each new test process also spends about 0.7 s to import FlyDSL.",
        f"<b>看到了什么。</b> LLVM 后端是冷编译里最大的一块： 在 GEMM 和 attention kernel 上占 {llvm_share}。 MLIR 步骤占比很小。 对 agent 来说， 重要的是 trace， 因为每次调用都要付这笔时间。 命中缓存也一样： FP8 GEMM 每次调用 {float(fp8['warm_inmem_hit_call_s']):.2f} s， attention kernel {float(fa['warm_inmem_hit_call_s']) * 1000:.0f} ms。 另外， 每个新的测试进程还要花约 0.7 s 导入 FlyDSL。")}
</section>
"""

    s11 = f"""
<section id="verdict">
  <h2><span class="n">11</span><span lang="en">What the results mean</span><span lang="zh">结果说明了什么</span></h2>
  <ul lang="en">
    <li><b>Premise A is true, and stronger than we expected.</b> In most arms that changed the code, removing one decision of the author cost 5–16% of speed. The heuristics of LLVM did not find this speed. In two of the five kernels whose fences we removed, the fences kept the results correct.</li>
    <li><b>Premise B is partly true.</b> Where nobody ordered the instructions by hand, the default of LLVM is good, and an automatic mode needs a scheduler like it. Where an author did, the heuristic of LLVM still costs time. A choice of order for each shape gives {(o_strict - 1) * 100:.1f}–{(o_point - 1) * 100:.1f}% on the one kernel that we measured.</li>
    <li><b>Premise C is true, with a correction.</b> Agents frequently read compiler output, and they do not trust the compile cache. But their speedups come from decisions that they already make.</li>
    <li><b>Premise D is true.</b> The trace costs time in each iteration.</li>
  </ul>
  <ul lang="zh">
    <li><b>前提 A 成立， 而且比预想的更强。</b> 在大多数改变了代码的 arm 里， 去掉作者的一个决策会损失 5–16% 的速度。 LLVM 的启发式规则找不到这部分速度。 在我们去掉 fence 的五个 kernel 里， 有两个的 fence 在保证结果正确。</li>
    <li><b>前提 B 部分成立。</b> 在没人手工排序的地方， LLVM 的默认值是好的， 自动模式需要一个这样的调度器。 在作者手工排序的地方， LLVM 的启发式规则仍让 kernel 变慢。 在我们测的这一个 kernel 上， 按 shape 选择顺序能带来 {(o_strict - 1) * 100:.1f}–{(o_point - 1) * 100:.1f}%。</li>
    <li><b>前提 C 成立， 但要修正。</b> agent 经常读编译器输出， 也不信任编译缓存。 但它们的提速来自本来就由它们做的决策。</li>
    <li><b>前提 D 成立。</b> trace 在每次迭代里都要花时间。</li>
  </ul>
  {pair("<b>So, can the redesign make kernels much faster?</b> Not every kernel. The large, measured effect is in expert kernels whose authors already override LLVM. There, a compiler that always keeps the explicit order, pins and waits protects the 5–16% that these decisions give. It also makes a search over order settings safe.</p>\n  <p lang=\"en\">For simple kernels, the speed will stay about the same. The scheduler of LLVM is a good automatic mode until a new scheduler is as good. For agents, the larger gain is fewer wasted iterations, not a higher peak speed. Fewer runs only decode ISA or clear caches. Compile errors replace silent wrong results.",
        "<b>那么， 重新设计能不能让 kernel 快很多？</b> 不是每个 kernel 都能。 实测到的大效果在专家 kernel 上， 这些 kernel 的作者本来就在覆盖 LLVM 的决定。 在那里， 一个始终遵守显式顺序、 pin 和 wait 的编译器， 能保住这些决策带来的 5–16%， 也让顺序设置的搜索变得安全。</p>\n  <p lang=\"zh\">对简单 kernel， 速度基本不会变。 在新调度器达到同样水平之前， LLVM 的调度器是一个好的自动模式。 对 agent 来说， 更大的收益是浪费的迭代更少， 而不是峰值速度更高。 只为解读 ISA 或清缓存的运行会更少。 原本不报错的错误结果会变成编译错误。")}
</section>
"""

    s12 = f"""
<section id="next">
  <h2><span class="n">12</span><span lang="en">What could prove us wrong</span><span lang="zh">什么会推翻这个结论</span></h2>
  <ul lang="en">
    <li><b>An explicit-order backend that is slower.</b> Build KIR and the new backend for one GEMM. Use the FP8 4-wave GEMM, because all of its decisions are explicit. Compare its speed with the output of LLVM for the same source decisions. If the order of the author, with the pins kept, is slower than LLVM, then the scheduler of LLVM must stay in the explicit mode.</li>
    <li><b>Agents that do not use the report.</b> On the overlay, run the same FlyBench records with two arms. The only difference is a <code>report</code> command in one arm. The report gives registers, spills and occupancy for each shape, the waits, and the race and bounds checks. If the number of 1.2× results, the slower shapes and the tokens per record do not change, then the agent part of the argument is wrong.</li>
    <li><b>Correctness fences that are rare.</b> We found them in two of the five kernels whose fences we removed. A scan of all kernels, with one fence removed at a time, will show how common they are. If they are rare, waits based on tokens are less important than this post says.</li>
  </ul>
  <ul lang="zh">
    <li><b>显式顺序的后端更慢。</b> 为一个 GEMM 实现 KIR 和新后端。 选 FP8 4-wave GEMM， 因为它的决策全是显式的。 在源码决策相同的条件下， 和 LLVM 的输出比较速度。 如果 "作者的顺序加保留的 pin" 比 LLVM 慢， 那么显式模式下仍要保留 LLVM 的调度器。</li>
    <li><b>agent 不用报告。</b> 在 overlay 上用同样的 FlyBench record 跑两个 arm， 唯一的区别是其中一个有 <code>report</code> 命令。 报告给出每个 shape 的寄存器、 spill 和 occupancy， 插入的 wait， 以及竞争和越界检查。 如果达到 1.2× 的结果数、 变慢的 shape 数和每个 record 的 token 数都不变， 论证里关于 agent 的部分就是错的。</li>
    <li><b>保证正确性的 fence 其实很少。</b> 在我们去掉 fence 的五个 kernel 里发现了两个。 对所有 kernel 每次去掉一个 fence 扫一遍， 就能知道它们有多常见。 如果很少， 基于 token 的 wait 就没有本文说的那么重要。</li>
  </ul>
</section>
"""

    s13 = f"""
<section id="reproduce">
  <h2><span class="n">13</span><span lang="en">Reproduce</span><span lang="zh">复现</span></h2>
  {pair(f"All the numbers in this post come from the files in {data_dir}:", f"本文所有数字都来自 {data_dir} 里的文件：")}
  <ul>
    {df("arms.csv", "what each arm changes.", "每个 arm 改了什么。")}
    {df("results.csv", "the verdict, the speed ratio and the interval of each arm.", "每个 arm 的判定、 速度比和区间。")}
    {df("per_shape.csv", "the speed ratio of each arm on each shape.", "每个 arm 在每个 shape 上的速度比。")}
    {df("isa.csv", "the ISA of stock and arm at the dumped shapes.", "dump 的 shape 上 stock 和 arm 的 ISA。")}
    {df("compile_breakdown.csv", "the compile times of experiment D.", "实验 D 的编译时间。")}
    {df("agent_mining_summary.json", "the agent totals of experiment C.", "实验 C 的 agent 汇总数据。")}
    {df("failures.json", "the failed tests of each wrong arm.", "每个出错 arm 的失败测试。")}
    {df("summary.json", "all of the above in one file.", "以上全部内容合在一个文件里。")}
  </ul>
  {pair(f"The directory also has the {n_arms} patches (<code>patches/</code>) and the scripts that built and summarized them (<code>scripts/</code>). <code>scripts/make_page.py</code> builds this page. Internal paths are replaced by placeholders. Each patch applies to FlyDSL <code>cb770e9a</code>, and the FlyBench v1 evaluator scores it on one MI355X.",
        f"目录里还有 {n_arms} 个补丁（<code>patches/</code>）和构建、 汇总它们的脚本（<code>scripts/</code>）。 <code>scripts/make_page.py</code> 生成本页。 内部路径已经换成占位符。 每个补丁都可以打到 FlyDSL <code>cb770e9a</code> 上， 由 FlyBench v1 评测器在一张 MI355X 上打分。")}
</section>
"""

    s14 = f"""
<section id="epilogue">
  <h2><span class="n">14</span><span lang="en">Epilogue</span><span lang="zh">结语</span></h2>
  {pair("I started this work to show that LLVM is the problem. The experiments show something narrower and more useful. The heuristics of LLVM are good for the decisions that nobody makes. The problems start when an author makes these decisions, and the compiler cannot tell a speed preference from a correctness rule.</p>\n  <p lang=\"en\">A compiler for agents must keep this difference clear. Each decision has one decider. Each rule is visible as data. The compiler writes down each choice where the agent can read it.",
        "开始这项工作时， 我想证明 LLVM 是问题所在。 实验给出的结论更窄， 也更有用。 LLVM 的启发式规则很擅长做没人去做的决定。 问题出在作者做了这些决定， 而编译器分不清哪些是速度上的偏好， 哪些是正确性上的规则。</p>\n  <p lang=\"zh\">给 agent 用的编译器必须把这个区别保持清楚： 每个决策只有一个决策者； 每条规则都以数据的形式可见； 编译器做出的每个选择都写在 agent 读得到的地方。")}
</section>

</main>
<footer class="colophon">
  <div><span lang="en">Source: FlyDSL <code>cb770e9a</code> (experiments) and <code>fe81c407</code> (code reading); LLVM <code>941a04e69</code>; FlyBench v1 evaluator on the AMD Pilot v2 overlay, 2026-11-22. Revised 2026-11-23: plain-language text and a glossary.</span><span lang="zh">来源： FlyDSL <code>cb770e9a</code>（实验）与 <code>fe81c407</code>（读代码）； LLVM <code>941a04e69</code>； AMD Pilot v2 overlay 上的 FlyBench v1 评测器， 2026-11-22。 2026-11-23 修订： 改用浅白的文字， 并增加术语表。</span></div>
  <div><span lang="en">Typography: Epilogue · IBM Plex Serif · Victor Mono · Noto Serif SC. Palette: drafting paper, Prussian ink, vermilion for conflict, teal for a single decider.</span><span lang="zh">字体： Epilogue · IBM Plex Serif · Victor Mono · Noto Serif SC。 配色： 绘图纸、 普鲁士蓝墨、 表示冲突的朱红、 表示单一决策者的青色。</span></div>
  <div><span lang="en">Related: <a href="/sources/flydsl-compiler-in-the-loop.html">FlyDSL compiler in the loop</a> · <a href="/sources/together-compiler.html">Together Compiler</a> · <a href="/sources/flydsl.html">FlyDSL source reading</a>.</span><span lang="zh">相关： <a href="/sources/flydsl-compiler-in-the-loop.html">FlyDSL compiler in the loop</a> · <a href="/sources/together-compiler.html">Together Compiler</a> · <a href="/sources/flydsl.html">FlyDSL 源码阅读</a>。</span></div>
</footer>
</div>
</div>
<script>{JS}</script>
</body>
</html>
"""
    html = head + s01 + s02 + s03 + s04 + s05 + s06 + s07 + s08 + s09 + s10 + s11 + s12 + s13 + s14
    html = re.sub(r"([。，：；、])(?=[^\s<）」』”])", r"\1 ", html)
    return html


if __name__ == "__main__":
    html = render()
    B.OUT.write_text(html)
    print(f"wrote {B.OUT} ({len(html.encode()) / 1024:.0f} KB)")
