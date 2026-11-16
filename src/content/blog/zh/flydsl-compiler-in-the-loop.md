---
title: "270 个调度屏障： 让编译器回到 FlyDSL 的 kernel 优化循环"
description: "FlyDSL 用一套可以打补丁的进程内 LLVM 编译 MI355X 和 MI455X 的 kernel， 但 180 个 kernel 文件里有 62 个在手工覆盖编译器的决定。 本文分析 kernel 优化循环在哪里断开、 compiler chain 的哪一层能修好每一处， 以及怎样用 CAKE 式的外层循环把手工覆盖变成编译器能力。"
date: 2026-11-16
tags: ["compiler", "FlyDSL", "LLVM", "MLIR", "kernel-optimization", "AMD", "agents", "MI355X"]
category: "Technical"
lang: "zh"
---

「多想想 compiler chain」 是一句好建议， 但它没有告诉你下一步做什么。 kernel 工程师知道怎么改 kernel。 那么改编译器是什么意思： 改哪一块、 怎么改， 又怎么知道改了有用？

先从决定 kernel 速度的因素说起。 算法和分块确定之后， 性能取决于少数几个决定： 执行哪些指令、 按什么顺序、 放在哪些寄存器里， 以及每个 wave 在哪里停下来等内存。 每个决定都有人负责， 要么是 kernel 作者， 要么是编译器。 compiler chain 真正重要的地方， 正是它做错决定、 或者做了决定却没人看得见的地方。 本文在 [FlyDSL](https://github.com/ROCm/FlyDSL)（AMD 用来写 MI355X 和 MI455X kernel 的 Python DSL） 里找出这些地方， 并讨论怎样让编译器跟着 kernel 一起进步， 而不是被绕开。

简短的结论： FlyDSL 掌握自己编译器的每一层， 而它的 kernel 在大约三分之一的文件里手工覆盖了后端的调度、 寄存器和等待决定。 最有价值的编译器工作， 是一次消除一种反复出现的手工覆盖， 并且只在所有 kernel 都认可时才接受改动。

文中所有 FlyDSL 引用都固定在 commit [`aae1553`](https://github.com/ROCm/FlyDSL/tree/aae155359ddc9b2bb1064ec8d391bc25aa4c0d3e)（2026 年 11 月 16 日）。

## 一套工具链， 而且归 FlyDSL 自己所有

FlyDSL 不调用 hipcc、 amdclang、 comgr 或 hipRTC。 它在 Python 进程里用自带的 MLIR 和 LLVM 编译： upstream `llvm-project` 固定在 `941a04e`（LLVM 24 开发版）， 外加 9 个 FlyDSL 补丁（[`llvm-build-info.json`](https://github.com/ROCm/FlyDSL/blob/aae155359ddc9b2bb1064ec8d391bc25aa4c0d3e/thirdparty/llvm-build-info.json)）。 pipeline 先用 FlyDSL 自己的 pass 降低 Fly dialect， 再把 GPU module 转成 ROCDL， 然后由 LLVM 做 `-O2` 优化、 生成 AMDGPU 代码并汇编（[`rocm.py`](https://github.com/ROCm/FlyDSL/blob/aae155359ddc9b2bb1064ec8d391bc25aa4c0d3e/python/flydsl/compiler/backends/rocm.py#L323-L378)）。 从 [#987](https://github.com/ROCm/FlyDSL/pull/987) 起， object 由链接进同一个库的 LLD 变成 HSA code object； 构建时没有带 LLD 的话， 就退回 ROCm 的 `ld.lld`。 ROCm 只提供 device bitcode 库、 备用链接器， 以及加载结果的 HIP runtime。

MI355X（gfx950） 和 MI455X（gfx1250， 即 MI450 系列） 走同一条路径。 target 属性只改变 chip 名称， 并把 gfx1250 标记为 wave32（[`gpu_module_targets`](https://github.com/ROCm/FlyDSL/blob/aae155359ddc9b2bb1064ec8d391bc25aa4c0d3e/python/flydsl/compiler/backends/rocm.py#L425-L438)）。 kernel 本身用的指令族不同： gfx950 上是 MFMA， gfx1250 上是 WMMA 和 tensor data mover（[`arch-gfx1250.md`](https://github.com/ROCm/FlyDSL/blob/aae155359ddc9b2bb1064ec8d391bc25aa4c0d3e/.claude/skills/llvm/references/arch-gfx1250.md)）。 gfx1250 也更依赖这些补丁： 9 个里有 6 个包含 gfx1250 专用的改动， 其中包括把 upstream LLVM 在 FlyDSL pin 之后才加入的 `gfx1250-strict` target 移植回来（[llvm-project #218543](https://github.com/llvm/llvm-project/pull/218543)）。

比细节更重要的是这个设计选择。 以二进制形式发布的编译器只能用开关来引导。 从源码构建、 前面有自己的 pass、 内部有自己补丁的编译器， 每一层都可以改， 而 FlyDSL 已经在经常这样做。 这是把编译器放进优化循环的前提。

<figure>
<div style="overflow-x:auto">
<svg viewBox="0 0 680 548" role="img" aria-labelledby="fc1-title" style="width:100%;min-width:600px;height:auto;font-family:inherit">
<title id="fc1-title">FlyDSL 的编译链， 以及每个阶段负责的决定</title>
<defs><marker id="fc1-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="currentColor"/></marker></defs>
<g fill="none" stroke="currentColor" stroke-opacity="0.45" stroke-width="1.2">
<rect x="16" y="16" width="250" height="56" rx="8"/>
<rect x="16" y="92" width="250" height="56" rx="8"/>
<rect x="16" y="168" width="250" height="56" rx="8"/>
<rect x="16" y="320" width="250" height="56" rx="8"/>
<rect x="16" y="396" width="250" height="56" rx="8"/>
</g>
<rect x="16" y="244" width="250" height="56" rx="8" stroke-width="2" style="fill:var(--color-accent);fill-opacity:0.1;stroke:var(--color-accent)"/>
<g stroke="currentColor" stroke-width="1.2" marker-end="url(#fc1-arrow)">
<line x1="141" y1="72" x2="141" y2="90"/>
<line x1="141" y1="148" x2="141" y2="166"/>
<line x1="141" y1="224" x2="141" y2="242"/>
<line x1="141" y1="300" x2="141" y2="318"/>
<line x1="141" y1="376" x2="141" y2="394"/>
</g>
<g fill="currentColor" font-size="15" font-weight="600">
<text x="32" y="40">Python kernel</text>
<text x="32" y="116">Fly MLIR passes</text>
<text x="32" y="192">LLVM IR 与 -O2</text>
<text x="32" y="268">AMDGPU 后端</text>
<text x="32" y="344">MC 汇编器 + LLD</text>
<text x="32" y="420">HIP runtime</text>
</g>
<g fill="currentColor" font-size="12" fill-opacity="0.65">
<text x="32" y="59">@flyc.kernel 源码</text>
<text x="32" y="135">FlyDSL 自己的 C++ pass</text>
<text x="32" y="211">upstream LLVM 中端</text>
<text x="32" y="287">upstream 代码 + 9 个 FlyDSL 补丁</text>
<text x="32" y="363">FlyDSL #987 起在进程内</text>
<text x="32" y="439">hipModuleLoadData， 启动</text>
</g>
<g fill="currentColor" font-size="13.5">
<text x="290" y="40">决定： 算法、 分块、 指令族</text>
<text x="290" y="116">决定： 把 layout 变成地址计算、 swizzle</text>
<text x="290" y="192">决定： 循环展开、 LICM、 CSE、 内联</text>
<text x="290" y="262">决定： 指令顺序、 寄存器、 等待点</text>
<text x="290" y="344">决定： 只有编码和 ELF， 不改 ISA</text>
<text x="290" y="420">加载 code object 并启动</text>
</g>
<g fill="currentColor" font-size="12" fill-opacity="0.65">
<text x="290" y="59">负责方： kernel 作者</text>
<text x="290" y="135">负责方： FlyDSL， 不用重编 LLVM</text>
<text x="290" y="211">负责方： upstream LLVM， 只能用开关引导</text>
<text x="290" y="279">负责方： LLVM 加 FlyDSL 补丁</text>
<text x="290" y="363">负责方： 同一个 LLVM， 或 ROCm ld.lld</text>
<text x="290" y="439">负责方： ROCm</text>
</g>
<text x="290" y="296" font-size="12.5" font-weight="600" style="fill:var(--color-accent)">180 个 kernel 文件中有 62 个在手工覆盖</text>
<g fill="currentColor" font-size="12.5" fill-opacity="0.7">
<text x="16" y="490">第 2 到第 5 阶段在 Python 进程内运行， 使用 FlyDSL 自带的 MLIR 和 LLVM</text>
<text x="16" y="510">（upstream llvm-project 941a04e 加 9 个补丁）。 MI355X（gfx950） 和 MI455X（gfx1250）</text>
<text x="16" y="530">走同一条路径， 只有 chip、 wave 大小和指令族不同。</text>
</g>
</svg>
</div>
<figcaption style="font-size:0.85em;opacity:0.75;text-align:center">图 1. FlyDSL 的编译链， 以及每个阶段负责的决定。 kernel 作者手工覆盖的三个决定都在 AMDGPU 后端。</figcaption>
</figure>

## kernel 在告诉我们什么

kernel 作者手工覆盖编译器， 要么是因为编译器决定错了， 要么是因为没法把作者的意图告诉编译器。 两种情况都会在源码里留下痕迹。 下面是 `kernels/` 下 180 个 Python 文件里的显式覆盖（统计方法见文末）：

| 覆盖方式 | 出现次数 | 作者在做什么 |
|---|---|---|
| `sched_barrier(0)` | 279 次 `sched_barrier` 调用中的 270 次， 35 个文件 | 禁止调度器把任何指令移过这个点 |
| `sched_mfma` / `sched_dsrd` / `sched_dswr` / `sched_vmem` | 66 行， 12 个文件 | 手写矩阵指令和访存指令的交错 |
| `s_waitcnt`、 `s_wait_*cnt` | 187 行， 45 个文件 | 手工决定 wave 在哪里等内存 |
| `s_setprio` | 49 行， 14 个文件 | 手工调高或调低 wave 优先级 |
| inline assembly | 29 行， 12 个文件 | 直接写编译器无法分析的指令 |
| `"enable-post-misched": False` | 4 个文件 | 关掉寄存器分配之后的调度 |

180 个文件中， 有 62 个至少包含其中一种。

作者为什么要禁止调度器移动任何东西？ 因为调度器优化的是一个通用目标， 在寄存器压力、 延迟和 occupancy 之间做取舍， 却不知道作者设计的 pipeline： 一条 global load 要在数据被用到之前， 提前若干条矩阵指令发出； 下一个 tile 的 LDS 写要放在循环末尾（[`kernel_tuning_guide.md`](https://github.com/ROCm/FlyDSL/blob/aae155359ddc9b2bb1064ec8d391bc25aa4c0d3e/docs/kernel_tuning_guide.md?plain=1#L311-L335)）。 手写的顺序编码了这条 pipeline， 屏障则保护它不被一个看不见它的优化器打乱。 同样的原因也让几个 attention kernel 关掉了 post-RA 调度， 用仓库文档的话说， 是为了保护手工调好的调度（[`mllvm-flags.md`](https://github.com/ROCm/FlyDSL/blob/aae155359ddc9b2bb1064ec8d391bc25aa4c0d3e/.claude/skills/llvm/references/mllvm-flags.md)）。

维持现状有三个代价。 每个 kernel、 每种 tile 形状、 每个架构都要重新付一次覆盖的成本。 一个完整的屏障还会阻止编译器优化它周围的代码。 而且正确的覆盖本身并不固定： FlyDSL 的调优文档记录， 手写交错在 async 「fat」 tile 上快 8–10%， 在部分 N 方向累加器很少的 f16 / bf16 tile 上反而更慢（[`kernel_tuning_guide.md`](https://github.com/ROCm/FlyDSL/blob/aae155359ddc9b2bb1064ec8d391bc25aa4c0d3e/docs/kernel_tuning_guide.md?plain=1#L336-L340)）。 一个随 tile 变化的决定， 不应该在每个文件里手写。

最清楚的例子在 11 月 16 日合入。 gfx950 的 FP4 和 FP8 GEMM 用 inline assembly 把矩阵累加器放在 AGPR 里， 因为编译器自己的固定原语 `sched_agpr` 不起作用： 累加器可能落进错误的寄存器文件， 指令选择也选了 VGPR 形式的 MFMA。 在 FP4 8192³ GEMM 上， 用这个原语要 674.8 µs， 用 assembly 只要 221.4 µs（[#1368](https://github.com/ROCm/FlyDSL/pull/1368)）。 修复是 LLVM 后端里的两处改动， 之后这个原语追平或超过了 assembly。 差距从来不在 kernel 里， 而在两个编译器决定里。 在编译器里修好之后， 下一个 kernel 不写 assembly 也能在 AGPR 里原地累加。

## 循环断开的四种方式

kernel 优化不管是人做还是 agent 做， 都是一个循环： 改 kernel、 编译、 测量、 保留更好的版本。 这个循环有四种不同的断法， 每一种都指向 compiler chain 的不同一层。 下面按修复成本从低到高排列。

### 看不见

一次编译只返回二进制或者报错。 想知道寄存器数、 spill、 occupancy， 或者某个 hint 有没有生效， 就得 dump 大约二十个中间文件， 再用脚本解析 assembly。 有些开关还会静默失效。 仓库自己的 LLVM 笔记记录了 4 个从未到达代码生成的开关； 一个未知的 `compile_hints` key 也会被默默接受， 同时还改变 cache key， 所以重新编译看起来像是真的生效了（[`llvm` skill](https://github.com/ROCm/FlyDSL/blob/aae155359ddc9b2bb1064ec8d391bc25aa4c0d3e/.claude/skills/llvm/SKILL.md)）。

修法是每次编译都输出一份结构化报告： 寄存器、 spill、 occupancy、 LDS、 指令数和 wait 数， 以及哪些 hint 真的到达了 IR。 LLVM 已经能通过 `kernel-resource-usage` remark 用 YAML 输出资源这一半， 其余部分可以从 FlyDSL 已经 dump 出来的 ISA 里取。 未知的 hint 应该直接报错。

这一项排在最前面， 因为其他所有修复都需要一个便宜的信号。 GPU 计时又贵又吵（同一份笔记测到 gfx950 上有 ±14% 的时钟噪声）， 而静态报告几毫秒就能筛掉一个会 spill 的候选。 FlyDSL 已经能把编译错误映射回出错的 Python 文件和行号（[`diagnostics.py`](https://github.com/ROCm/FlyDSL/blob/aae155359ddc9b2bb1064ec8d391bc25aa4c0d3e/python/flydsl/compiler/diagnostics.py#L30-L56)）， 但性能信息还没有这种待遇。

### 选不了

当最好的调度模板取决于 tile 时， 这个选择是搜索问题， 不是规则。 FlyDSL 的 autotuner 搜索 kernel 参数、 warp 数和 `waves_per_eu`（[`autotune.py`](https://github.com/ROCm/FlyDSL/blob/aae155359ddc9b2bb1064ec8d391bc25aa4c0d3e/python/flydsl/autotune.py#L130-L149)）。 LLVM 选项和模板开关都是手工写进 kernel 的。 把它们加入搜索空间， 再用上面的报告剪枝， 就能把按 tile 的猜测变成测量。

搜索有上限： 它只能在已有的选项里挑， 造不出缺少的那个选项。

### 表达不了

270 个屏障和 187 处手工 wait 并不是 457 个不同的想法。 它们是少数几个想法， 在每个 kernel 里重新写一遍； 一个被写了几百遍的想法， 就是一个缺失的抽象。 统计里能看到三个：

- **声明 pipeline， 而不是加屏障。** 作者只描述一次想要的调度（哪些 load 要提前多少条矩阵指令、 LDS 写放在哪里）， 由编译器生成调度提示， 并检查最终 ISA 是否遵守。 约束让编译器还能继续优化其余部分， 完整的屏障做不到。
- **从数据流推出 wait。** Fly 层能看到每个异步拷贝和消费它的操作， 所以能算出 wait 必须放在哪里， 并检查有没有漏掉。 在 gfx1250 上线期间， 漏掉或放错的 wait 反复造成静默的错误结果（[「单测通过， 模型出错」 里的缺口 5](/zh/blog/unit-pass-model-fail/)）。
- **用 API 而不是 assembly 控制寄存器文件。** [#1349](https://github.com/ROCm/FlyDSL/pull/1349) 加入了 `sched_bank` 和 `sched_agpr`， #1368 让 `sched_agpr` 真正生效。

为什么不让 LLVM 来做这些？ 因为 LLVM 看到的是指令， 不是 pipeline。 stage、 生产者和消费者只存在于 Fly 层， 到后端调度时已经消失了。 这也是这一层改起来便宜的原因： pass 就在 FlyDSL 仓库里， 改动不用重编 LLVM， 每次改动都能逐个 kernel 验证， 标准是速度相同或更快， 手工覆盖更少。

### 决定错了

有些决定只有后端能做： 最终指令顺序、 寄存器分配、 MFMA 形式、 wait 插入。 #1368 就是在这一层修好的。 这一层的上限最高， 因为一次修复能惠及所有 kernel； 成本也最高。 每次改动都要重编 LLVM， 而 LLVM 的变化会以难以预测的方式挪动 kernel 性能： #1349 刻意没有升级 LLVM pin， 不引入 1,807 个 upstream 提交， 理由就是避免改变 kernel 性能。 所以后端工作应该投到测量显示差距大的地方， 而且每次改动都要通过覆盖全部 kernel 的性能门禁。

### 量出上限

还有一个工具能把猜测变成数字。 FlyDSL 的 [`inject_asm`](https://github.com/ROCm/FlyDSL/blob/aae155359ddc9b2bb1064ec8d391bc25aa4c0d3e/docs/testing_benchmarking_guide.md?plain=1#L406-L431) 能把手工改过的 `.s` 重新汇编、 换进 cache， 再跑原来的测试， 并在相同条件下跑一次基线。 它现在只是调试工具。 如果由 agent 或强化学习策略来重排指令、 移动 wait， 它就变成了对某个二进制的最优调度的搜索。 [CuAsmRL](https://arxiv.org/abs/2501.08071) 在 NVIDIA SASS 上这样做， 让专家写的 kernel 平均快 9%， 最高快 26%。 结果绑定单个二进制和单个形状， 所以不是用来上线的。 它给出的是编译器留下了多大差距， 以及一个具体的 diff， 告诉你该改哪条启发式。

## 让编译器自己进步

来自 NVIDIA 和 CMU 的研究者提出的 [CAKE](https://arxiv.org/abs/2608.12629)， 把这件事做成了一个系统。 它靠三条原则支撑。 agent 修改的是类型化、 硬件决定显式的 IR， 而不是裸 CUDA。 编译器返回的是定位到具体位置的诊断， 而不是 pass / fail： 编译前的门禁检查同步安全、 硬件契约、 数据一致性和调度语义， 再加一个校准过的 cost model， 在任何 GPU 运行之前给候选排序（§3.1）。 harness 本身也在演化： 不透明的运行时崩溃变成 verifier 规则， 反复出现的非法 lowering 模式变成静态检查， 系统性的预测偏差变成校准目标（§3.2）。 在 B200 上从零开始的 Flash-KMeans 任务里， 同样预算下， Cake IR 的最好候选中位数达到调优基线的 1.144×， agent 直接写 CUDA / PTX 只有 0.928×。

映射到 FlyDSL 上， 就是两个循环：

<figure>
<div style="overflow-x:auto">
<svg viewBox="0 0 680 372" role="img" aria-labelledby="fc2-title" style="width:100%;min-width:600px;height:auto;font-family:inherit">
<title id="fc2-title">优化 kernel 的内层循环， 和改进编译器的外层循环</title>
<defs><marker id="fc2-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="currentColor"/></marker><marker id="fc2-arrow-accent" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--color-accent)"/></marker></defs>
<rect x="6" y="6" width="668" height="150" rx="10" fill="currentColor" fill-opacity="0.04"/>
<rect x="6" y="196" width="668" height="170" rx="10" fill="currentColor" fill-opacity="0.04"/>
<text x="20" y="30" font-size="13" font-weight="600" fill="currentColor" fill-opacity="0.85">内层循环： 优化一个 kernel</text>
<text x="20" y="356" font-size="13" font-weight="600" fill="currentColor" fill-opacity="0.85">外层循环： 改进编译器</text>
<g fill="none" stroke="currentColor" stroke-opacity="0.45" stroke-width="1.2">
<rect x="20" y="74" width="145" height="56" rx="8"/>
<rect x="185" y="74" width="145" height="56" rx="8"/>
<rect x="350" y="74" width="145" height="56" rx="8"/>
<rect x="515" y="74" width="145" height="56" rx="8"/>
</g>
<g stroke-width="1.6" style="fill:var(--color-accent);fill-opacity:0.1;stroke:var(--color-accent)">
<rect x="350" y="262" width="145" height="56" rx="8"/>
<rect x="185" y="262" width="145" height="56" rx="8"/>
<rect x="20" y="262" width="145" height="56" rx="8"/>
</g>
<g fill="none" stroke="currentColor" stroke-width="1.2" marker-end="url(#fc2-arrow)">
<path d="M165,102 H183"/>
<path d="M330,102 H348"/>
<path d="M495,102 H513"/>
<path d="M587,74 V54 H92 V72"/>
<path d="M422,130 V260"/>
</g>
<g fill="none" stroke-width="1.4" marker-end="url(#fc2-arrow-accent)" style="stroke:var(--color-accent)">
<path d="M350,290 H332"/>
<path d="M185,290 H167"/>
<path d="M92,262 V132"/>
</g>
<g fill="currentColor" font-size="14" font-weight="600">
<text x="34" y="98">编译</text>
<text x="199" y="98">报告</text>
<text x="364" y="98">测量</text>
<text x="529" y="98">改 kernel</text>
<text x="364" y="286">记录、 归类</text>
<text x="199" y="286">修复</text>
<text x="34" y="286">语料门禁</text>
</g>
<g fill="currentColor" font-size="11.5" fill-opacity="0.7">
<text x="34" y="117">MLIR + LLVM</text>
<text x="199" y="117">spill、 wait、 hint</text>
<text x="364" y="117">空闲 GPU， 多次重复</text>
<text x="529" y="117">agent 或工程师</text>
<text x="364" y="305">失败和手工覆盖</text>
<text x="199" y="305">先用最便宜的一层</text>
<text x="34" y="305">所有 kernel 都通过</text>
</g>
<g fill="currentColor" font-size="12" fill-opacity="0.75">
<text x="340" y="48" text-anchor="middle">重复， 保留最好的版本</text>
<text x="432" y="181">失败、 手工覆盖、 差距</text>
<text x="102" y="181">合并后， 之后的 kernel 都受益</text>
<text x="257" y="340" text-anchor="middle">开关 → Fly pass → LLVM 补丁</text>
</g>
</svg>
</div>
<figcaption style="font-size:0.85em;opacity:0.75;text-align:center">图 2. 内层循环优化一个 kernel。 外层循环把内层循环遇到的问题（失败、 手工覆盖、 测出的差距） 变成编译器改动， 并且只合并所有 kernel 都通过的改动。</figcaption>
</figure>

FlyDSL 已经有了大部分组件：

| CAKE 原则 | FlyDSL 现状 | 还缺什么 |
|---|---|---|
| 硬件显式的 IR | atom、 layout 和调度原语都是显式的 | layout 代码出错时难以定位 |
| 定位到具体位置的诊断 | 编译错误能映射到 Python 行号 | 结构化的性能诊断 |
| 自我演化的 harness | 仓库 skill、 LLVM 补丁流程、 对比 `main` 和最新 tag 的 benchmark CI（[`flydsl.yaml`](https://github.com/ROCm/FlyDSL/blob/aae155359ddc9b2bb1064ec8d391bc25aa4c0d3e/.github/workflows/flydsl.yaml)） | 失败和手工覆盖的记录； 覆盖全部 kernel 的编译器改动门禁 |

有一个设计差异值得注意。 CAKE 刻意不把 layout 作为一等抽象（§2.2）， 理由是 layout 演算让 agent 容易出错， 而且错误难以定位。 FlyDSL 恰好建立在 layout algebra 之上。 所以对 FlyDSL 来说， 最有价值的诊断， 是把 layout 错误翻译成 agent 能动手修的问题。

语料门禁不是官僚流程。 CAKE 指出， 没有经过语料验证的新 verifier 规则可能拒绝合法的 kernel， 没有合法性规则的新原语会让 IR 更难分析（§3.2）。 编译器改动会影响每个 kernel， 所以必须用每个 kernel 来评判。

自动化什么、 怎么自动化， 取决于要改的是什么：

| 自动化方式 | 改什么 | 例子 |
|---|---|---|
| 搜索 | 选项和 kernel 参数 | FlyDSL 的 autotuner |
| 学习型启发式 | 编译器里的单个决定 | [MLGO](https://arxiv.org/abs/2101.04808) 用训练出的策略替换了 LLVM 的 inlining-for-size 启发式， 代码最多小 7% |
| LLM 演化代码 + 自动评估 | kernel 或编译器启发式的源码 | [AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) 让 Gemini 的一个 kernel 快 23%， 在 GPU 指令层面让 FlashAttention 最多快 32.5% |
| ISA 上的强化学习 | 最终指令顺序 | CuAsmRL： 平均快 9%， 最高快 26% |

四种方式都先需要同一样东西： 一个可信的评估器， 也就是正确性测试， 加上在隔离 GPU 上重复足够多次、 能压过噪声的计时。 没有它， 自动化只会去优化噪声。

## 第一个实验： 编译器差距普查

在测出差距之前， 「该改哪一块」 始终是模糊的。 一次普查可以回答它：

1. 在 MI355X 上选 3 到 5 个重要的 kernel， 因为那里有 CI runner： 一个 GEMM、 一个 attention kernel 和一个 MoE kernel。
2. 每个 kernel 构建三个版本： 原样； 去掉手工覆盖并扫几种 LLVM 调度策略； 以及可选的、 用 `inject_asm` 搜索找到的最好调度。
3. 把每一段性能差距归到一个阶段： 调度、 寄存器分配、 wait、 指令选择或地址计算。
4. 把每类差距列成表： 它损失多少性能、 出现在几个 kernel 里、 由哪一层来修。

当这张表能回答 「先改哪一层、 能拿到多少」 时， 普查就完成了。 第一项修复里的结构化编译报告正是让普查变便宜的工具， 所以两者可以同时开始。

## 结语

只要对每个决定 kernel 速度的因素问一句 「是谁做的决定、 有没有人看得见」， 「多想想 compiler chain」 就不再模糊。 在 FlyDSL 里， kernel 自己给出了答案： 180 个文件里有 62 个在手工决定调度、 寄存器和等待。 每一处手工覆盖， 都是一份没人提交的编译器 bug 报告。 外层循环就是提交它们的方式： 记录、 归类， 在能表达它的最便宜的一层修好， 并且只合并整套 kernel 都认可的改动。

## 统计方法

数字是 [`aae1553`](https://github.com/ROCm/FlyDSL/tree/aae155359ddc9b2bb1064ec8d391bc25aa4c0d3e) 上 `kernels/` 目录的静态匹配结果。 其中包含少量注释， 所以应该把它看作指标， 而不是精确的调用次数：

```bash
git clone https://github.com/ROCm/FlyDSL.git && cd FlyDSL
git checkout aae155359ddc9b2bb1064ec8d391bc25aa4c0d3e
rg --files -g '*.py' kernels | wc -l               # 180 个 kernel 文件
rg -o 'sched_barrier\(\s*0\s*\)' kernels | wc -l   # 270 个完整屏障
rg -o 'sched_barrier\(' kernels | wc -l            # 279 次 sched_barrier 调用
rg -c 's_waitcnt|s_wait_[a-z]+cnt' kernels         # 每个文件里手工放置的 wait
rg -l 'sched_barrier|sched_group_barrier|sched_(mfma|dsrd|dswr|vmem)\(|s_setprio|s_waitcnt|s_wait_[a-z]+cnt|inline_asm|InlineAsm' kernels | wc -l   # 62 个文件
```

文中 8–10% 和 ±14% 两个数字来自 FlyDSL 自己的文档， 本文没有复测。 上面的方向都还没有实验数据； 普查是第一个能产生数字的步骤。
