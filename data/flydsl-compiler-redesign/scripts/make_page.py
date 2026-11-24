"""Assemble the Experiment 008 page from computed numbers, charts and static plates."""

import re

import build_blog as B
from page_css import CSS, JS
from page_plates import PLATE_DECISIONS, PLATE_LOOP


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

# Text that the author writes. Each entry is a list of (English, Chinese) paragraphs.
# An empty list renders a draft marker in that section, and the build reports the page as a draft.
AUTHOR_TEXT = {
    "position": [],
    "toy": [],
}


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
    fp4 = link(f"{M}/kernels/gemm/fp4_gemm_4wave.py#L82-L87", "<code>fp4_gemm_4wave.py</code>")
    moe = link(f"{M}/kernels/mega_moe_tp/gemm.py#L948-L951", "<code>mega_moe_tp/gemm.py</code>")
    siinstr = link(f"{L}/llvm/lib/Target/AMDGPU/SIInstrInfo.cpp#L4554-L4559", "<code>SIInstrInfo.cpp</code>")
    flash_src = link(f"{V}/kernels/attention/flash_attn_gfx950.py#L1027", "<code>flash_attn_gfx950.py</code>")
    arch = link(f"{V}/.claude/skills/llvm/references/arch-gfx950.md#L147", "<code>arch-gfx950.md</code>")
    init_py = link(f"{V}/kernels/__init__.py", "<code>kernels/__init__.py</code>")
    pr1368 = link("https://github.com/ROCm/FlyDSL/pull/1368", "#1368")
    pr1477 = link("https://github.com/ROCm/FlyDSL/pull/1477", "#1477")
    pr945 = link("https://github.com/ROCm/FlyDSL/pull/945", "#945")
    v1_url = ("https://github.com/jhinpan/jinnpan.com/blob/d63784b86ee521af50c407c6819ec6325e0d987d"
              "/public/sources/flydsl-compiler-redesign.html")
    v1_en, v1_zh = link(v1_url, "in the git history"), link(v1_url, "保留在 git 历史里")
    main_tree = link(M.replace("/blob/", "/tree/"), "fe81c407")
    data_dir = link(B.DATA, "<code>data/flydsl-compiler-redesign</code>")

    def df(name, en, zh):
        return (f'<li lang="en">{link(f"{B.DATAF}/{name}", f"<code>{name}</code>")}: {en}</li>\n'
                f'    <li lang="zh">{link(f"{B.DATAF}/{name}", f"<code>{name}</code>")}： {zh}</li>')

    plate1 = PLATE_LOOP
    missing = []

    def author(key, todo_en, todo_zh):
        paras = AUTHOR_TEXT[key]
        if paras:
            return "\n  ".join(pair(en, zh) for en, zh in paras)
        missing.append(key)
        return f'<div class="callout draft">\n    {pair(todo_en, todo_zh)}\n  </div>'

    position = author(
        "position",
        "<b>Draft: the author writes this section.</b> It answers three questions in a few sentences. "
        "What limits the final code of a kernel today? What do we change in the way we work? What is not the goal?",
        "<b>草稿：这一节由作者来写。</b> 用几句话回答三个问题。 今天是什么限制了 kernel 的最终代码？ "
        "我们的工作方式要改什么？ 什么不是目标？")
    toy = author(
        "toy",
        "<b>Draft: not written, and not run.</b> The author chooses the kernel and the change. Then we run the three "
        "steps on one MI355X, and put the listings and the numbers here. This section has no measured result yet.",
        "<b>草稿：还没写，也还没跑。</b> 由作者选定 kernel 和改动。 然后我们在一张 MI355X 上跑完三步， "
        "把代码清单和数字放在这里。 这一节目前没有任何实测结果。")
    status_en = "RFC · draft 1" + (" · incomplete" if missing else "")
    status_zh = "RFC · 第 1 稿" + (" · 未完成" if missing else "")

    head = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Experiment 008 · RFC: co-develop FlyDSL kernels and the compiler</title>
<meta name="description" content="An RFC for FlyDSL on AMD GPUs: keep LLVM, and change the kernel code and the compiler together, so that a fixed compiler does not limit the final code. The evidence is {n_arms} experiments on MI355X GPUs.">
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
    <li><a href="#position"><span class="num">01</span><span lang="en">Position</span><span lang="zh">立场</span></a></li>
    <li><a href="#loop"><span class="num">02</span><span lang="en">The loop</span><span lang="zh">循环</span></a></li>
    <li><a href="#toy"><span class="num">03</span><span lang="en">Toy example</span><span lang="zh">小例子</span></a></li>
    <li><a href="#evidence"><span class="num">04</span><span lang="en">Evidence</span><span lang="zh">证据</span></a></li>
    <li><a href="#open"><span class="num">05</span><span lang="en">Open question</span><span lang="zh">未解决的问题</span></a></li>
    <li><a href="#not"><span class="num">06</span><span lang="en">Not proposed</span><span lang="zh">不提议的事</span></a></li>
    <li><a href="#next"><span class="num">07</span><span lang="en">Next steps</span><span lang="zh">下一步</span></a></li>
    <li><a href="#wrong"><span class="num">08</span><span lang="en">What could prove this wrong</span><span lang="zh">什么会推翻它</span></a></li>
    <li><a href="#ablation"><span class="num">A</span><span lang="en">Take a decision away</span><span lang="zh">拿走一个决策</span></a></li>
    <li><a href="#headroom"><span class="num">B</span><span lang="en">Untuned settings</span><span lang="zh">没人调过的设置</span></a></li>
    <li><a href="#agents"><span class="num">C</span><span lang="en">What agents did</span><span lang="zh">agent 做了什么</span></a></li>
    <li><a href="#overhead"><span class="num">D</span><span lang="en">Compile cost</span><span lang="zh">编译开销</span></a></li>
    <li><a href="#method"><span class="num">E</span><span lang="en">How we measured</span><span lang="zh">怎样测量</span></a></li>
    <li><a href="#terms"><span class="num">F</span><span lang="en">Terms</span><span lang="zh">术语</span></a></li>
    <li><a href="#reproduce"><span class="num">G</span><span lang="en">Reproduce</span><span lang="zh">复现</span></a></li>
  </ol>
</nav>
<div>
<header class="masthead">
  <div class="kicker">Experiment 008 · RFC</div>
  <h1>Kernel <span class="slash">/</span> Compiler</h1>
  {pair("FlyDSL is a Python language for writing AMD GPU kernels. LLVM changes each kernel into machine instructions. This RFC proposes to keep LLVM, and to change the kernel code and the compiler together, so that a fixed compiler does not limit the final code. The evidence is " + str(n_arms) + " experiments on MI355X GPUs. This is a draft for comments.",
        "FlyDSL 是一种用 Python 写 AMD GPU kernel 的语言。 LLVM 把每个 kernel 变成机器指令。 这份 RFC 提议保留 LLVM， 并让 kernel 代码和编译器一起改， 这样固定的编译器就不会限制最终代码。 证据是 MI355X GPU 上的 " + str(n_arms) + " 组实验。 这是一份征求意见的草稿。",
        cls="subtitle")}
  <div class="spec">
    <div><b lang="en">Status</b><b lang="zh">状态</b><span lang="en">{status_en}</span><span lang="zh">{status_zh}</span></div>
    <div><b lang="en">Hardware</b><b lang="zh">硬件</b>MI355X · gfx950 · overlay</div>
    <div><b lang="en">FlyDSL under test</b><b lang="zh">被测 FlyDSL</b><a href="{V.replace('/blob/', '/tree/')}">cb770e9a</a> (FlyBench v1)</div>
    <div><b lang="en">Code reading</b><b lang="zh">读代码</b>{main_tree} (v0.5.2)</div>
    <div><b lang="en">Timing</b><b lang="zh">计时</b><span lang="en">FlyBench v1 · 6 ABBA pairs</span><span lang="zh">FlyBench v1 · 6 组 ABBA</span></div>
    <div><b lang="en">Data</b><b lang="zh">数据</b><a href="{B.DATA}">data/flydsl-compiler-redesign</a></div>
  </div>
</header>

<main class="article">
"""

    s01 = f"""
<section id="position">
  <h2><span class="n">01</span><span lang="en">Position</span><span lang="zh">立场</span></h2>
  <div class="callout note">
    {pair(f"<b>Rewritten on 2026-11-23.</b> The first version of this page proposed to remove MLIR and LLVM, and to build a new backend. Reviewers did not accept that proposal, and the experiments on this page do not support it (section 06). This version keeps all the measurements and changes the proposal. The first version is {v1_en}.",
          f"<b>2026-11-23 重写。</b> 本页的第一版提议去掉 MLIR 和 LLVM， 并实现一个新后端。 审阅的人没有接受这个方案， 本页的实验也不支持它（第 06 节）。 这一版保留全部测量结果， 改掉了方案。 第一版{v1_zh}。")}
  </div>
  {position}
  {pair('This page uses terms such as wave, fence and spill. <a href="#terms">Appendix F</a> defines them.',
        '本页用到 wave、 fence、 spill 等术语。 <a href="#terms">附录 F</a> 有它们的定义。')}
</section>
"""

    s02 = f"""
<section id="loop">
  <h2><span class="n">02</span><span lang="en">The loop: generate, change, teach</span><span lang="zh">循环：生成、修改、教会编译器</span></h2>
  {pair("In this RFC, the compiler is one method to generate code. It is not the limit of the code. The loop has three steps. It starts with the compiler at version n.",
        "在这份 RFC 里， 编译器是生成代码的一种方法， 不是代码的上限。 循环有三步， 从第 n 版编译器开始。")}
  <figure class="plate wide">
    <div class="plate-meta">Plate I · <span lang="en">the loop for one kernel</span><span lang="zh">一个 kernel 的循环</span></div>
    {plate1}
    {pair("The final code changes first, in step 2. The compiler follows, in step 3. The red text is the open question of section 05.",
          "最终代码在第 2 步先变， 编译器在第 3 步跟上。 红字是第 05 节的未解决问题。", cls="caption")}
  </figure>
  <ol class="list" lang="en">
    <li><b>Generate.</b> The compiler builds the kernel from its source. The result is the final code: <code>code(n) = f(llvm(n))</code>. Here <code>f</code> is the kernel source, and <code>llvm(n)</code> is the compiler at version n.</li>
    <li><b>Change the code.</b> An agent or a person changes the final code directly, and then measures it: <code>code(n+1) = agent(code(n))</code>. The compiler does not limit this step.</li>
    <li><b>Teach the compiler.</b> If the new code is correct and faster, make the compiler produce the same code from the kernel source: <code>llvm(n+1) = f⁻¹(code(n+1))</code>. Section 05 lists the places where this change can live.</li>
  </ol>
  <ol class="list" lang="zh">
    <li><b>生成。</b> 编译器从 kernel 源码生成代码。 结果就是最终代码： <code>code(n) = f(llvm(n))</code>。 其中 <code>f</code> 是 kernel 源码， <code>llvm(n)</code> 是第 n 版编译器。</li>
    <li><b>改代码。</b> agent 或人直接修改最终代码， 然后测量： <code>code(n+1) = agent(code(n))</code>。 这一步不受编译器限制。</li>
    <li><b>教会编译器。</b> 如果新代码正确而且更快， 就让编译器从 kernel 源码生成同样的代码： <code>llvm(n+1) = f⁻¹(code(n+1))</code>。 第 05 节列出这个改动可以放在哪里。</li>
  </ol>
  {pair("Then go back to step 1, with the next kernel or with the next change to the compiler.",
        "然后回到第 1 步， 换下一个 kernel， 或者换下一个编译器改动。")}
</section>
"""

    s03 = f"""
<section id="toy">
  <h2><span class="n">03</span><span lang="en">A toy example</span><span lang="zh">一个小例子</span></h2>
  {pair("A toy example shows the three steps on one small kernel. It needs three listings:",
        "小例子在一个很小的 kernel 上走完三步。 它需要三份代码清单：")}
  <ol class="list" lang="en">
    <li>The kernel source, about ten lines, and the code that the compiler produces from it today. This is <code>code(n)</code>.</li>
    <li>The changed code, and its measured speed against <code>code(n)</code>. This is <code>code(n+1)</code>.</li>
    <li>The change that makes the compiler produce <code>code(n+1)</code> from the same source, and the check that no other kernel became slower.</li>
  </ol>
  <ol class="list" lang="zh">
    <li>kernel 源码（大约十行）， 以及今天的编译器从它生成的代码。 这是 <code>code(n)</code>。</li>
    <li>修改后的代码， 以及它相对 <code>code(n)</code> 的实测速度。 这是 <code>code(n+1)</code>。</li>
    <li>让编译器从同一份源码生成 <code>code(n+1)</code> 的改动， 以及其他 kernel 没有变慢的检查结果。</li>
  </ol>
  {toy}
</section>
"""

    s04 = f"""
<section id="evidence">
  <h2><span class="n">04</span><span lang="en">Evidence: a fixed compiler limits the final code today</span><span lang="zh">证据：固定的编译器今天就在限制最终代码</span></h2>
  {pair("The speed of an AMD GPU kernel depends on five kinds of decisions (Plate II). In FlyDSL, the author makes three of them in Python: the work split, the data movement and the instruction choice. The heuristics of LLVM make the other two: the register use and the instruction order.",
        "AMD GPU kernel 的速度取决于五类决策（Plate II）。 在 FlyDSL 里， 作者在 Python 里做出其中三类： 工作划分、 数据搬运和指令选择。 另外两类由 LLVM 的启发式规则来做： 寄存器使用和指令顺序。")}
  <figure class="plate wide">
    <div class="plate-meta">Plate II · <span lang="en">five decisions and who makes them in FlyDSL today</span><span lang="zh">五类决策，以及今天在 FlyDSL 里由谁来做</span></div>
    {PLATE_DECISIONS}
    {pair("The author makes the first three decisions directly. LLVM makes the last two with its heuristics. When an author needs a specific result for the last two, the author overrides LLVM through the red path.",
          "前三类决策由作者直接做出。 后两类由 LLVM 用启发式规则来做。 作者如果需要后两类的特定结果， 就通过红色路径覆盖 LLVM 的决定。", cls="caption")}
  </figure>
  {pair(f"The red path is common. In the FlyDSL kernels at {main_tree}, 34 of 180 kernel files use fences (257 times in total). 44 files place waits by hand (186 times). 13 files use inline assembly, and 12 files set the wave priority.",
        f"红色路径很常见。 在 {main_tree} 的 FlyDSL kernel 里， 180 个 kernel 文件中有 34 个使用 fence（共 257 次）。 44 个文件手工放置 wait（共 186 次）。 13 个文件使用 inline 汇编， 12 个文件设置 wave 优先级。")}
  {pair(f'The kernel comments give the reasons. One kernel uses inline assembly so that LLVM does not see a memory operation and does not add a wait ({fp4}). Another kernel adds a fence and gives this reason: "the scheduler may hoist them above s_barrier: a race" ({moe}). That is, LLVM can move the reads before the barrier, and the kernel then reads data that is not ready.',
        f'kernel 里的注释写明了原因。 一个 kernel 用 inline 汇编， 是为了让 LLVM 看不到一次访存， 从而不插入 wait（{fp4}）。 另一个 kernel 加了 fence， 注释写的是： "调度器可能把它们提到 s_barrier 之前： 这是竞争"（{moe}）。 意思是： LLVM 可能把读取移到 barrier 之前， kernel 就会读到还没准备好的数据。')}
  {pair("The red path is steps 2 and 3 of the loop, done by hand. The author wants a specific final code, and forces the compiler to produce it.",
        "红色路径就是手工完成的第 2 步和第 3 步。 作者想要某一份特定的最终代码， 就强制编译器生成它。")}
  {pair(f"How much speed does this path give? We removed one decision of the author at a time, and let LLVM decide (appendix A). {NUM.get(n_slow, n_slow)} kernels became {slow_lo:.0f}–{slow_hi:.0f}% slower. In two kernels, the results became wrong: the fences that we removed also kept the results correct.",
        f"这条路径值多少速度？ 我们每次去掉作者的一个决策， 让 LLVM 来决定（附录 A）。 {n_slow} 个 kernel 慢了 {slow_lo:.0f}–{slow_hi:.0f}%。 有两个 kernel 的结果变错了： 被去掉的 fence 同时在保证结果正确。")}
  <figure class="plate wide">
    <div class="plate-meta">Plate III · <span lang="en">speed ratio when one decision of the author is removed</span><span lang="zh">去掉作者的一个决策后的速度比</span></div>
    {plate3}
    {legend([("ink2", "instruction order: fences, groups", "指令顺序：fence、交错组"), ("violet", "wave priority", "wave 优先级"),
             ("amber", "register use", "寄存器使用"), ("hair", "null control", "空白对照")])}
    {pair("Each bar is the speed ratio over all timed and new shapes. The black line is the 95% interval over the six ABBA pairs. The two register rows show the only two shapes that this arm changes. A cross means that the tests of FlyDSL or the correctness check rejected the output.",
          "每根条是所有计时 shape 和新 shape 上的速度比。 黑线是 6 组 ABBA 的 95% 区间。 两行寄存器结果是这个 arm 唯一会改变的两个 shape。 叉号表示 FlyDSL 的测试或正确性检查拒绝了输出。", cls="caption")}
  </figure>
  {pair("Thus the default output of LLVM is not the final code that these kernels need. Today, each author closes this difference by hand, in one kernel at a time.",
        "所以 LLVM 的默认输出不是这些 kernel 需要的最终代码。 今天， 每个作者手工去补这个差距， 一次只补一个 kernel。")}
</section>
"""

    s05 = f"""
<section id="open">
  <h2><span class="n">05</span><span lang="en">Open question: step 3 must not slow other kernels</span><span lang="zh">未解决的问题：第 3 步不能让其他 kernel 变慢</span></h2>
  {pair("A change in a kernel source affects one kernel. A change in LLVM affects each kernel that LLVM compiles. Thus step 3 has a risk that step 2 does not have. Three results show that this risk is real.",
        "改一个 kernel 的源码， 只影响这一个 kernel。 改 LLVM， 会影响 LLVM 编译的每个 kernel。 所以第 3 步有第 2 步没有的风险。 三个结果说明这个风险是真实的。")}
  <ul lang="en">
    <li><b>One patch, two kernels.</b> To keep one accumulator in AGPRs, FlyDSL patched the register allocator and the instruction selector of LLVM ({pr1368}). Then the patch was reverted, because it caused spills in a kernel of another team ({pr1477}).</li>
    <li><b>One kernel, many shapes.</b> Without the fences and interleave groups of its author, the preshuffle GEMM is {pct('x01-nosched')}% slower in total. But it is {x01f0:.1f}% faster at 16384³ and {x01p1:.1f}% faster at 8192³ (Plate IV).</li>
    <li><b>Faster on average, slower on single shapes.</b> With the schedulers of LLVM turned off, the same kernel ran at {r['h11-mischedoff']:.3f}× in total. With <code>max-ilp</code>, it ran at {r['h10-stratmaxilp']:.3f}×. But {len(h11['regressions'])} and {len(h10['regressions'])} shapes became slower by more than the noise level, to {h11['min_shape_ratio']:.2f}× and {h10['min_shape_ratio']:.2f}× in the worst case (appendix B).</li>
  </ul>
  <ul lang="zh">
    <li><b>一个补丁， 两个 kernel。</b> 为了把一个累加器固定在 AGPR 里， FlyDSL 修改了 LLVM 的寄存器分配和指令选择（{pr1368}）。 后来这个补丁被回滚， 因为它让另一个团队的 kernel 出现了 spill（{pr1477}）。</li>
    <li><b>一个 kernel， 很多 shape。</b> 去掉作者的 fence 和交错组之后， preshuffle GEMM 整体慢了 {pct('x01-nosched')}%。 但它在 16384³ 上快了 {x01f0:.1f}%， 在 8192³ 上快了 {x01p1:.1f}%（Plate IV）。</li>
    <li><b>平均更快， 单个 shape 更慢。</b> 关掉 LLVM 的调度器后， 同一个 kernel 的整体速度比是 {r['h11-mischedoff']:.3f}×。 用 <code>max-ilp</code> 时是 {r['h10-stratmaxilp']:.3f}×。 但分别有 {len(h11['regressions'])} 个和 {len(h10['regressions'])} 个 shape 的变慢超出噪声下限， 最差到 {h11['min_shape_ratio']:.2f}× 和 {h10['min_shape_ratio']:.2f}×（附录 B）。</li>
  </ul>
  <figure class="plate wide">
    <div class="plate-meta">Plate IV · <span lang="en">preshuffle-a8, shape by shape: no single order wins on all shapes</span><span lang="zh">preshuffle-a8 逐个 shape 看：没有哪种顺序在所有 shape 上都赢</span></div>
    {plate5}
    {legend([("ink2", "no fences or groups", "去掉 fence 和交错组"), ("red", "no groups", "去掉交错组"),
             ("teal", "max-ilp", "max-ilp"), ("amber", "LLVM schedulers off", "关掉 LLVM 调度器")])}
    {pair(f"Each dot is one public shape in one arm. 1.0 is the stock kernel. For each shape, pick the best of these four arms and stock. Then the {o_n} public shapes get {o_point:.3f}× on average (geometric mean). If we count an arm only where its interval is above 1.0, and use only the lower end of the interval, they get {o_strict:.3f}×.",
          f"每个点是一个公开 shape 在一个 arm 下的速度比。 1.0 是 stock kernel。 为每个 shape 在这四个 arm 和 stock 之间选出最好的一个， {o_n} 个公开 shape 平均得到 {o_point:.3f}×（几何平均）。 如果只在区间整体高于 1.0 时才计入某个 arm， 并且只计区间的下限， 平均得到 {o_strict:.3f}×。", cls="caption")}
  </figure>
  {pair("A change from step 3 can live in three places. The list goes from local to global:",
        "第 3 步的改动可以放在三个地方。 下面从局部排到全局：")}
  <ol class="list" lang="en">
    <li><b>In the kernel source,</b> as an explicit decision: a fence, an interleave group, a pin. It affects one kernel. Authors use this place today (section 04).</li>
    <li><b>In a compiler option of one kernel.</b> A FlyDSL kernel can set LLVM options for itself. For example, the dense flash attention turns off the post-RA scheduler ({flash_src}). It affects one kernel, and all kernels still use one LLVM.</li>
    <li><b>In LLVM itself:</b> a heuristic, a pass or a patch. It affects all kernels.</li>
  </ol>
  <ol class="list" lang="zh">
    <li><b>放在 kernel 源码里，</b> 作为一个显式决策： fence、 交错组或 pin。 它只影响一个 kernel。 作者今天用的就是这里（第 04 节）。</li>
    <li><b>放在某个 kernel 的编译选项里。</b> FlyDSL kernel 可以给自己设置 LLVM 选项。 例如， dense flash attention 关掉了 post-RA 调度器（{flash_src}）。 它只影响一个 kernel， 所有 kernel 仍然共用一个 LLVM。</li>
    <li><b>放在 LLVM 本身：</b> 一条启发式规则、 一个 pass 或一个补丁。 它影响所有 kernel。</li>
  </ol>
  {pair("Places 1 and 2 are safe for other kernels. But the compiler learns nothing: the next kernel must find the same decision again. Place 3 helps each kernel, but it needs a gate.",
        "放在 1 和 2， 对其他 kernel 是安全的。 但编译器什么都没学到： 下一个 kernel 还得重新找到同样的决策。 放在 3， 对每个 kernel 都有帮助， 但需要一道关卡。")}
  <div class="callout">
    {pair("<b>A candidate gate, not decided.</b> Accept a change to LLVM only if all kernels and shapes pass the check that FlyBench already uses for a change to one kernel: each result is correct, and no shape is slower by more than its noise level. A change that does not pass stays in place 1 or 2.",
          "<b>一个候选的关卡，还没有定。</b> 只有当所有 kernel 和 shape 都通过 FlyBench 现在用于单个 kernel 改动的检查时， 才接受对 LLVM 的改动： 每个结果都正确， 没有 shape 的变慢超出它的噪声下限。 通不过的改动留在 1 或 2。")}
  </div>
  {pair("We have not measured how often a change passes this gate, or how many kernels and shapes the check needs. Section 07 lists this experiment.",
        "我们还没有测过一个改动通过这道关卡的概率， 也没有测过这个检查需要多少 kernel 和 shape。 第 07 节列出了这个实验。")}
</section>
"""

    s06 = f"""
<section id="not">
  <h2><span class="n">06</span><span lang="en">What this RFC does not propose</span><span lang="zh">这份 RFC 不提议什么</span></h2>
  <ul lang="en">
    <li><b>To remove LLVM.</b> Where no author decides the instruction order, the default of LLVM is good. On the normalization kernels and the MoE reduce kernel, no other scheduling strategy was faster by more than the noise level. Without the schedulers of LLVM, RMSNorm became {pct('h12-mischedoff')}% slower (appendix B). LLVM also fills the gaps that hardware hazards need, and FlyDSL follows upstream LLVM ({pr945}).</li>
    <li><b>To remove MLIR, or to build a new IR and backend.</b> We did not build them, and we did not measure them. No result on this page needs them.</li>
    <li><b>To move the layouts to Python.</b> Some layouts mix parts that are known at compile time with parts that are known only at run time. This change can make these layouts harder to write. We did not test it.</li>
    <li><b>To make the compile faster.</b> The work before each launch costs time (appendix D). But the test harness can solve this problem, and it is not part of this RFC.</li>
  </ul>
  <ul lang="zh">
    <li><b>去掉 LLVM。</b> 在作者没有决定指令顺序的地方， LLVM 的默认值是好的。 在 normalization 和 MoE reduce kernel 上， 没有其他调度策略的提速超出噪声下限。 关掉 LLVM 的调度器后， RMSNorm 慢了 {pct('h12-mischedoff')}%（附录 B）。 LLVM 还负责留出硬件 hazard 需要的间隔， 而且 FlyDSL 一直跟随上游 LLVM（{pr945}）。</li>
    <li><b>去掉 MLIR， 或者实现新的 IR 和后端。</b> 我们没有实现它们， 也没有测量它们。 本页没有哪个结果需要它们。</li>
    <li><b>把 layout 移到 Python 里。</b> 有些 layout 混着编译期已知的部分和运行时才知道的部分。 这个改动可能让这类 layout 更难写。 我们没有测试过。</li>
    <li><b>让编译更快。</b> 每次启动前的工作确实要花时间（附录 D）。 但这个问题可以在测试 harness 里解决， 不属于这份 RFC。</li>
  </ul>
</section>
"""

    s07 = f"""
<section id="next">
  <h2><span class="n">07</span><span lang="en">Next steps</span><span lang="zh">下一步</span></h2>
  <ol class="list" lang="en">
    <li><b>Run the toy example</b> of section 03 on one MI355X: all three steps, with the measured speed of each step.</li>
    <li><b>Measure the gate</b> of section 05. Take one change to LLVM that makes one kernel faster. Run all kernels and shapes with it, and count the ones that become wrong or slower. The patch of {pr1368} is a candidate.</li>
    <li><b>Build the harness.</b> Agents write a batch of kernels in one clean test flow. Then we read the development logs to see what the compiler lacks.</li>
  </ol>
  <ol class="list" lang="zh">
    <li><b>跑第 03 节的小例子。</b> 在一张 MI355X 上走完三步， 并测出每一步的速度。</li>
    <li><b>测第 05 节的关卡。</b> 选一个让某个 kernel 变快的 LLVM 改动。 带着它跑所有 kernel 和 shape， 数出结果变错或变慢的有多少。 {pr1368} 的补丁是一个候选。</li>
    <li><b>搭 harness。</b> agent 在一个干净的测试流程里写一批 kernel。 然后我们读开发日志， 看编译器还缺什么。</li>
  </ol>
</section>
"""

    s08 = f"""
<section id="wrong">
  <h2><span class="n">08</span><span lang="en">What could prove this wrong</span><span lang="zh">什么会推翻它</span></h2>
  <ul lang="en">
    <li><b>Step 2 gives little.</b> In earlier FlyBench runs, the accepted speedups of agents came from four kinds of change to the kernel source: wider memory access, a tile choice per shape, more work splits, and new algorithms (median {rv['legit_reproduced_speedup_median']:.2f}×). Order settings and register settings appear in only {src['schedule_or_sync_knob'] * 100:.0f}% and {src['register_or_occupancy_knob'] * 100:.0f}% of the reviews (appendix C). If direct changes to the final code give less than the noise level, on the toy kernel and on a batch of kernels, the loop is not worth its cost.</li>
    <li><b>Step 3 does not generalize.</b> If most changes fail the gate, each change stays in one kernel. Then the compiler does not improve, and the loop is only hand tuning with more steps.</li>
    <li><b>Step 2 breaks the results, and no test shows it.</b> In two of five kernels, fences that look like speed hints keep the results correct (appendix A). One MoE GEMM returned a wrong output in some runs, without any change (appendix B). If the tests miss such errors in changed code, step 2 needs a stronger check than the evaluator of today.</li>
  </ul>
  <ul lang="zh">
    <li><b>第 2 步的收益很小。</b> 在之前的 FlyBench 运行里， agent 被认可的提速来自对 kernel 源码的四类改动： 更宽的访存、 按 shape 选择 tile、 更多的工作切分， 以及新算法（中位数 {rv['legit_reproduced_speedup_median']:.2f}×）。 顺序设置和寄存器设置分别只出现在 {src['schedule_or_sync_knob'] * 100:.0f}% 和 {src['register_or_occupancy_knob'] * 100:.0f}% 的 review 里（附录 C）。 如果直接修改最终代码的收益在小例子和一批 kernel 上都低于噪声下限， 这个循环就不值它的成本。</li>
    <li><b>第 3 步推广不开。</b> 如果大多数改动过不了关卡， 每个改动就只能留在一个 kernel 里。 这样编译器没有进步， 循环只是步骤更多的手工调优。</li>
    <li><b>第 2 步把结果改错了， 而测试看不出来。</b> 在五个 kernel 里， 有两个的 fence 看起来是速度提示， 实际在保证结果正确（附录 A）。 一个 MoE GEMM 在没有任何改动的情况下， 有几次运行返回了错误的输出（附录 B）。 如果测试漏掉改动后代码里的这类错误， 第 2 步就需要比现在的评测器更强的检查。</li>
  </ul>
</section>
"""

    divider = f"""
<div class="divider">
  <span lang="en">Appendix · the measurements</span><span lang="zh">附录 · 测量数据</span>
  {pair("The RFC ends here. The appendices hold the data behind sections 04, 05, 06 and 08. The measurements are the same as in the first version.",
        "RFC 到这里结束。 附录是第 04、 05、 06、 08 节背后的数据。 测量结果和第一版相同。")}
</div>
"""

    app_a = f"""
<section id="ablation">
  <h2><span class="n"><span lang="en">Appendix A</span><span lang="zh">附录 A</span></span><span lang="en">Experiment A · take one decision away from the author</span><span lang="zh">实验 A · 拿走作者的一个决策</span></h2>
  {pair("<b>What we did.</b> Each arm removes one decision that the author made about register use or instruction order. Then the default heuristics of LLVM make that decision. If these heuristics were as good as the author, each bar in Plate III (section 04) would be at 1.0.",
        "<b>做了什么。</b> 每个 arm 去掉作者在寄存器使用或指令顺序上的一个决策， 改由 LLVM 的默认启发式规则来做。 如果这些规则和作者一样好， Plate III（第 04 节）里的每根条都应该在 1.0。")}
  <table>
    <tr>{exp_head}</tr>
    {abl_table}
  </table>
  {pair("Result 1: the decisions of the author give 5–16%, and LLVM does not find this speed", "结果 1：作者的决策值 5–16%，LLVM 找不到这部分速度", tag="h3")}
  {pair(f"Without the fences and interleave groups, the preshuffle GEMM became {pct('x01-nosched')}% slower ({r['x01-nosched']:.3f}×). The MX preshuffle GEMM became {pct('x03-nosched')}% slower, and the sliding-window attention became {pct('x05-nosched')}% slower. In the preshuffle GEMM, the interleave groups cause all of the loss. Without only the groups, the speed ratio is {r['x14-nogroup']:.3f}×, the same as without all hints.",
        f"去掉 fence 和交错组之后， preshuffle GEMM 慢了 {pct('x01-nosched')}%（{r['x01-nosched']:.3f}×）。 MX preshuffle GEMM 慢了 {pct('x03-nosched')}%， 滑窗 attention 慢了 {pct('x05-nosched')}%。 在 preshuffle GEMM 里， 全部损失都来自交错组： 只去掉交错组时速度比是 {r['x14-nogroup']:.3f}×， 和去掉全部提示一样。")}
  {pair(f"Without <code>s_setprio</code>, the two 8-wave GEMMs became {pct('x06-noprio')}% and {pct('x07-noprio')}% slower. For the first one, the ISA shows more than a priority change: the kernel got VGPR spills, and it had no spills before.",
        f"去掉 <code>s_setprio</code> 之后， 两个 8-wave GEMM 分别慢了 {pct('x06-noprio')}% 和 {pct('x07-noprio')}%。 对第一个 kernel， ISA 显示的不只是优先级变化： 原来没有 spill， 现在出现了 VGPR spill。")}
  {pair(f"This kernel, the FP8 8-wave GEMM, has four <code>s_setprio</code> instructions. Their documented job is only to set the wave priority. When we removed them, the register use changed, and the kernel got 6 spills. The cause is one line in LLVM: LLVM treats <code>s_setprio</code> as a hard limit for instruction order ({siinstr}). Thus one hint has two effects, and the kernel source does not show the second effect.",
        f"这个 kernel（FP8 8-wave GEMM）里有 4 条 <code>s_setprio</code> 指令。 按文档， 它们只负责设置 wave 优先级。 我们去掉它们之后， 寄存器使用发生了变化， kernel 多出 6 个 spill。 原因是 LLVM 里的一行代码： LLVM 把 <code>s_setprio</code> 当作指令顺序的硬边界（{siinstr}）。 所以一个提示有两种效果， 而 kernel 源码里看不到第二种。")}
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
  {pair("This result is not about speed. Today, an autotuner or an agent cannot safely change a scheduling hint, because some of these hints are correctness rules that look like speed hints.",
        "这个结果和速度无关。 今天， autotuner 或 agent 不能放心地修改调度提示， 因为其中一些提示看起来是为了速度， 实际上是保证正确性的规则。")}
  {pair("Result 3: on single shapes, the free order of LLVM is sometimes better", "结果 3：在单个 shape 上，LLVM 的自由排序有时更好", tag="h3")}
  {pair(f"The same preshuffle arm is {pct('x01-nosched')}% slower in total. But it is {x01f0:.1f}% faster at 16384³ and {x01p1:.1f}% faster at 8192³. Thus the interleave pattern of the author is a good fixed rule, but it is not the best rule for each shape. Plate IV (section 05) shows this for all public shapes.",
        f"同一个 preshuffle arm 整体慢了 {pct('x01-nosched')}%， 但在 16384³ 上快了 {x01f0:.1f}%， 在 8192³ 上快了 {x01p1:.1f}%。 所以作者的交错模板是一个不错的固定规则， 但不是每个 shape 上最好的规则。 Plate IV（第 05 节）显示了所有公开 shape 的情况。")}
  {pair("Result 4: two small effects on the flash attention", "结果 4：flash attention 上的两个小效果", tag="h3")}
  {pair(f"Without <code>s_setprio</code>, the flash attention ran at {x08['ratio']:.3f}× {B.ci('x08-noprio')} in total. But it was up to {(1 - x08['min_shape_ratio']) * 100:.1f}% slower on single shapes, and {len(x08['regressions'])} shapes were slower by more than their noise level.",
        f"去掉 <code>s_setprio</code> 后， flash attention 整体速度比是 {x08['ratio']:.3f}× {B.ci('x08-noprio')}。 但在单个 shape 上最多慢 {(1 - x08['min_shape_ratio']) * 100:.1f}%， 有 {len(x08['regressions'])} 个 shape 的变慢超出了噪声下限。")}
  {pair(f"The author of this kernel turned off the post-RA scheduler of LLVM to protect the order that the author wrote ({flash_src}). The post-RA scheduler is a second LLVM step that changes the instruction order after register assignment. When we turned it on again, the kernel ran at {x09['ratio']:.3f}× {B.ci('x09-postmischedon')}. No shape was slower by more than its noise level, and the best shape was {(x09['max_shape_ratio'] - 1) * 100:.1f}% faster. The ISA has {nop_s - nop_c} fewer <code>s_nop</code> instructions ({nop_s} → {nop_c}). An <code>s_nop</code> is an empty instruction that fills a hazard gap.",
        f"这个 kernel 的作者关掉了 LLVM 的 post-RA 调度器， 用来保护作者写下的顺序（{flash_src}）。 post-RA 调度器是 LLVM 在分配寄存器之后再调整一次指令顺序的步骤。 我们重新打开它之后， kernel 的速度比是 {x09['ratio']:.3f}× {B.ci('x09-postmischedon')}。 没有 shape 的变慢超出噪声下限， 最好的 shape 快了 {(x09['max_shape_ratio'] - 1) * 100:.1f}%。 ISA 里少了 {nop_s - nop_c} 条 <code>s_nop</code>（{nop_s} → {nop_c}）。 <code>s_nop</code> 是填充 hazard 间隔的空指令。")}
  {pair(f"This gain is about the size of the {x09['noise_floor'] * 100:.1f}% noise level of this kernel. Thus FlyBench would not count it as an improvement. The author could not see if this setting helps. The notes of FlyDSL say that three related flash attention kernels use different values for this setting, with no comment ({arch}).",
        f"这个收益和这个 kernel {x09['noise_floor'] * 100:.1f}% 的噪声下限差不多大， 所以 FlyBench 不会把它算作提升。 作者看不到这个设置到底有没有帮助。 FlyDSL 自己的笔记写着： 三个相关的 flash attention kernel 在这个设置上取值不同。 这些 kernel 里也没有注释说明原因（{arch}）。")}
</section>
"""

    app_b = f"""
<section id="headroom">
  <h2><span class="n"><span lang="en">Appendix B</span><span lang="zh">附录 B</span></span><span lang="en">Experiment B · settings that nobody tuned</span><span lang="zh">实验 B · 没人调过的设置</span></h2>
  {pair("<b>What we did.</b> The normalization, softmax and MoE kernels have no hints about instruction order. LLVM makes all of these decisions with its default heuristics. We tried two other LLVM scheduling strategies, <code>max-ilp</code> and <code>max-memory-clause</code>. We also turned off the instruction schedulers of LLVM. The instruction order is then close to the order in the source.",
        "<b>做了什么。</b> normalization、 softmax 和 MoE 这些 kernel 没有任何关于指令顺序的提示。 这些决策全部由 LLVM 用默认的启发式规则来做。 我们试了 LLVM 另外两种调度策略： <code>max-ilp</code> 和 <code>max-memory-clause</code>。 我们还关掉了 LLVM 的指令调度器， 这时指令顺序接近源码里的顺序。")}
  <figure class="plate wide">
    <div class="plate-meta">Plate V · <span lang="en">speed ratio with one LLVM setting that nobody tuned</span><span lang="zh">换一个没人调过的 LLVM 设置后的速度比</span></div>
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
    <li><b>For kernels without a hand-written order, the default of LLVM is good.</b> No strategy was faster by more than the noise level on the normalization kernels and the MoE reduce kernel. Without the schedulers, RMSNorm became {pct('h12-mischedoff')}% slower.</li>
    <li><b>For the hand-ordered preshuffle GEMM, the result is different.</b> Without the schedulers of LLVM, it ran at {r['h11-mischedoff']:.3f}× {B.ci('h11-mischedoff')}, faster than stock. With <code>max-ilp</code>, it ran at {r['h10-stratmaxilp']:.3f}×. But neither result is a clean gain: {len(h11['regressions'])} and {len(h10['regressions'])} shapes became slower by more than the noise level (worst {h11['min_shape_ratio']:.2f}× and {h10['min_shape_ratio']:.2f}×). Thus FlyBench would reject both arms. Even with the hints of the author, the heuristic scheduler of LLVM costs this kernel time.</li>
    <li><b>The MoE GEMM sometimes returns wrong results, even without a change.</b> With <code>max-ilp</code>, one timed call failed the correctness check (cosine similarity {cosine('h08-stratmaxilp'):.3f}; the limit is 0.99). We repeated the arm. In the repeat, the stock kernel itself returned a wrong output (cosine similarity {cosine('h08b-stratmaxilp'):.3f}). Thus we cannot blame the setting. The cause is probably a race, but we did not find it. The case shows a different problem: the evaluator catches this bug only in some runs.</li>
  </ol>
  <ol class="list" lang="zh">
    <li><b>对没有手工排序的 kernel， LLVM 的默认值是好的。</b> 在 normalization 和 MoE reduce kernel 上， 没有哪种策略的提速超出噪声下限。 关掉调度器后， RMSNorm 慢了 {pct('h12-mischedoff')}%。</li>
    <li><b>对手工排序的 preshuffle GEMM， 结果不同。</b> 关掉 LLVM 的调度器后， 速度比是 {r['h11-mischedoff']:.3f}× {B.ci('h11-mischedoff')}， 比 stock 快。 用 <code>max-ilp</code> 时是 {r['h10-stratmaxilp']:.3f}×。 但两者都不是干净的提升： 分别有 {len(h11['regressions'])} 个和 {len(h10['regressions'])} 个 shape 的变慢超出噪声下限（最差 {h11['min_shape_ratio']:.2f}× 和 {h10['min_shape_ratio']:.2f}×）， 所以 FlyBench 会拒绝这两个 arm。 即使有作者的提示， LLVM 的启发式调度器仍让这个 kernel 变慢。</li>
    <li><b>MoE GEMM 即使不做任何改动， 有时也会返回错误结果。</b> 用 <code>max-ilp</code> 时， 有一次计时调用没有通过正确性检查（余弦相似度 {cosine('h08-stratmaxilp'):.3f}， 门限是 0.99）。 我们重复了这个 arm。 在重复实验里， stock kernel 自己返回了错误的输出（余弦相似度 {cosine('h08b-stratmaxilp'):.3f}）。 所以不能怪这个设置。 原因可能是竞争， 但我们没有找到。 这个例子说明了另一个问题： 评测器只能在部分运行里抓到这个 bug。</li>
  </ol>
</section>
"""

    app_c = f"""
<section id="agents">
  <h2><span class="n"><span lang="en">Appendix C</span><span lang="zh">附录 C</span></span><span lang="en">Experiment C · what agents did in {cost['n']} attempts</span><span lang="zh">实验 C · agent 在 {cost['n']} 次尝试里做了什么</span></h2>
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
  {pair(f"<b>Where the speedups come from.</b> The accepted speedups have a median of {rv['legit_reproduced_speedup_median']:.2f}×, as reproduced by the reviewer. They come from four kinds of change: wider memory access, a tile choice per shape, more work splits, and new algorithms. These changes are work split, data movement and instruction choice, the decisions that the author already makes. Order settings and register settings appear in only {src['schedule_or_sync_knob'] * 100:.0f}% and {src['register_or_occupancy_knob'] * 100:.0f}% of the reviews.",
        f"<b>提速来自哪里。</b> 被认可的提速经 review 复现， 中位数是 {rv['legit_reproduced_speedup_median']:.2f}×。 它们来自四类改动： 更宽的访存、 按 shape 选择 tile、 更多的工作切分以填满 GPU， 以及新算法。 这些都属于工作划分、 数据搬运和指令选择， 也就是作者本来就在做的决策。 顺序设置和寄存器设置分别只出现在 {src['schedule_or_sync_knob'] * 100:.0f}% 和 {src['register_or_occupancy_knob'] * 100:.0f}% 的 review 里。")}
  {pair(f"<b>Where the effort goes.</b> A median attempt used {cost['median_prompt_tokens'] / 1e6:.1f}M prompt tokens, {cost['median_turns']} turns and {cost['median_wall_s'] / 3600:.1f} hours. In {isa_share:.0f}% of the {n_tr} command logs, the agent read the IR or the ISA at least once. In {cc_share:.0f}%, it cleared the FlyDSL compile cache at least once ({by['cache_clear']:,} times in total). Agents do this when they cannot tell if the code that ran is the code that they wrote.</p>\n  <p lang=\"en\">The most common failure in the reviews was a slower result on shapes that the agent did not time ({rv['failed_checklist_items']['per-shape regressions vs stock']} of {rv['n']} reviews). A few reviews found problems that a compiler can find before the run. Some were correctness bugs, for example a missing barrier and an out-of-bounds read. Others were spills, for example a spill setting that made one shape 0.44× as fast. Nine reviews say that agents added their own launch caches to avoid the trace at each call.",
        f"<b>精力花在哪里。</b> 按中位数计， 一次尝试用了 {cost['median_prompt_tokens'] / 1e6:.1f}M prompt token、 {cost['median_turns']} 轮对话、 {cost['median_wall_s'] / 3600:.1f} 小时。 在 {n_tr} 份命令记录里， 有 {isa_share:.0f}% 显示 agent 至少读过一次 IR 或 ISA。 有 {cc_share:.0f}% 显示 agent 至少清过一次 FlyDSL 编译缓存（总共 {by['cache_clear']:,} 次）。 agent 在无法确认 \"运行的代码是不是自己写的代码\" 时， 就会这样做。</p>\n  <p lang=\"zh\">review 里最常见的失败， 是 agent 没有计时的 shape 变慢了（{rv['n']} 份里有 {rv['failed_checklist_items']['per-shape regressions vs stock']} 份）。 少数 review 发现了编译器在运行前就能发现的问题。 有些是正确性 bug， 例如漏掉的 barrier 和越界读取。 有些是 spill， 例如一个让某个 shape 只剩 0.44× 速度的 spill 设置。 有 9 份 review 提到， agent 为了避开每次调用的 trace， 自己加了启动缓存。")}
</section>
"""

    app_d = f"""
<section id="overhead">
  <h2><span class="n"><span lang="en">Appendix D</span><span lang="zh">附录 D</span></span><span lang="en">Experiment D · the compile cost before each launch</span><span lang="zh">实验 D · 每次启动前的编译开销</span></h2>
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

    app_e = f"""
<section id="method">
  <h2><span class="n"><span lang="en">Appendix E</span><span lang="zh">附录 E</span></span><span lang="en">How we measured</span><span lang="zh">怎样测量</span></h2>
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
  {pair(f"Each job also compiled the stock kernel and the arm at two shapes, and compared the ISA. All arms changed the ISA except three. The first is the occupancy arm (appendix A). The other two are null controls: RMSNorm, which has no hints to remove, and the A16W16 GEMM with the added lines but nothing removed.</p>\n  <p lang=\"en\">The null controls measured {r['x12-nosched']:.3f}× and {r['x15-null']:.3f}×. This is the noise of our setup. Thus we cannot see effects smaller than about 0.5%.",
        f"每个作业还在两个 shape 上分别编译 stock kernel 和 arm， 并比较 ISA。 除了三个 arm， 其他 arm 都改变了 ISA。 第一个是 occupancy arm（附录 A）。 另外两个是空白对照： 一个是没有提示可去的 RMSNorm， 另一个是加了那几行代码、 但什么都没去掉的 A16W16 GEMM。</p>\n  <p lang=\"zh\">两个空白对照测得 {r['x12-nosched']:.3f}× 和 {r['x15-null']:.3f}×。 这就是我们这套测量的噪声， 所以小于约 0.5% 的效果看不出来。")}
  {pair("The register arm changes code only for 256×256 tiles. Only two of its 19 timed shapes use these tiles. Thus we report this arm at these two shapes.",
        "寄存器 arm 只改变 256×256 tile 的代码。 它的 19 个计时 shape 里只有 2 个用这种 tile， 所以我们只在这两个 shape 上报告它。")}
  <div class="callout note">
    {pair("<b>Limit.</b> The overlay puts single-GPU jobs next to other jobs on the same node, so we did not measure on idle GPUs. The ABBA order removes slow drift between stock and arm. It does not remove noise from other jobs. The intervals below include this noise. The agent data in experiment C comes from earlier FlyBench runs, not from new agent jobs.",
          "<b>局限。</b> overlay 会把单卡作业和其他作业放在同一个节点上， 所以我们不是在空闲的 GPU 上测量的。 ABBA 顺序能消除 stock 和 arm 之间的缓慢漂移， 但消除不了其他作业带来的噪声。 下面的区间已经包含这部分噪声。 实验 C 的 agent 数据来自之前的 FlyBench 运行， 不是新跑的 agent 作业。")}
  </div>
</section>
"""

    app_f = f"""
<section id="terms">
  <h2><span class="n"><span lang="en">Appendix F</span><span lang="zh">附录 F</span></span><span lang="en">Terms used in this post</span><span lang="zh">本文用到的术语</span></h2>
  {pair("You do not need a compiler background to read this post. The two tables below define the terms that it uses. Each term has only one meaning in this post.",
        "读这篇文章不需要编译器背景。 下面两张表定义了文中用到的术语。 每个术语在本文里只有一个意思。")}
  {pair("GPU and compiler terms", "GPU 和编译器术语", tag="h3")}
  {terms_table(TERMS_GPU)}
  {pair("Experiment terms", "实验术语", tag="h3")}
  {terms_table(TERMS_EXP)}
</section>
"""

    app_g = f"""
<section id="reproduce">
  <h2><span class="n"><span lang="en">Appendix G</span><span lang="zh">附录 G</span></span><span lang="en">Reproduce</span><span lang="zh">复现</span></h2>
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

    tail = f"""
</main>
<footer class="colophon">
  <div><span lang="en">Source: FlyDSL <code>cb770e9a</code> (experiments) and <code>fe81c407</code> (code reading); LLVM <code>941a04e69</code>; FlyBench v1 evaluator on the AMD Pilot v2 overlay, 2026-11-22. Rewritten 2026-11-23 as an RFC: the proposal changed, and the measurements did not.</span><span lang="zh">来源： FlyDSL <code>cb770e9a</code>（实验）与 <code>fe81c407</code>（读代码）； LLVM <code>941a04e69</code>； AMD Pilot v2 overlay 上的 FlyBench v1 评测器， 2026-11-22。 2026-11-23 重写为 RFC： 方案改了， 测量结果没有变。</span></div>
  <div><span lang="en">Typography: Epilogue · IBM Plex Serif · Victor Mono · Noto Serif SC. Palette: drafting paper, Prussian ink, vermilion for conflict and open questions, teal for the kernel author.</span><span lang="zh">字体： Epilogue · IBM Plex Serif · Victor Mono · Noto Serif SC。 配色： 绘图纸、 普鲁士蓝墨、 表示冲突和未解决问题的朱红、 表示 kernel 作者的青色。</span></div>
  <div><span lang="en">Related: <a href="/sources/flydsl-compiler-in-the-loop.html">FlyDSL compiler in the loop</a> · <a href="/sources/together-compiler.html">Together Compiler</a> · <a href="/sources/flydsl.html">FlyDSL source reading</a>.</span><span lang="zh">相关： <a href="/sources/flydsl-compiler-in-the-loop.html">FlyDSL compiler in the loop</a> · <a href="/sources/together-compiler.html">Together Compiler</a> · <a href="/sources/flydsl.html">FlyDSL 源码阅读</a>。</span></div>
</footer>
</div>
</div>
<script>{JS}</script>
</body>
</html>
"""
    html = (head + s01 + s02 + s03 + s04 + s05 + s06 + s07 + s08 + divider
            + app_a + app_b + app_c + app_d + app_e + app_f + app_g + tail)
    html = re.sub(r"([。，：；、])(?=[^\s<）」』”])", r"\1 ", html)
    return html, missing


if __name__ == "__main__":
    html, missing = render()
    B.OUT.write_text(html)
    state = f"DRAFT, author text missing: {', '.join(missing)}" if missing else "complete"
    print(f"wrote {B.OUT} ({len(html.encode()) / 1024:.0f} KB, {state})")
