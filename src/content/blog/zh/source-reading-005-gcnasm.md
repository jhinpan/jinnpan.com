---
title: "源码精读 005 — GCNasm， 六十四道公案为你补全那本读不完的 AMD ISA 手册"
description: "carlushuang 的 gcnasm 是 HIP 教程和 1200 页 CDNA3 ISA 手册之间罕见的中间地带： 64 个独立、 完整、 短小的 kernel， 把 AMD 高手亲手写的代码长什么样直接摊给你看。 6 小时认真读下来， 你会拿到一套能用的 MFMA / vmcnt 流水线 / DPP 跨 lane / 还有那条 LLVM 汇编器拒绝接受的指令的心智模型。"
date: 2026-05-17
tags: ["source-reading", "MLSys", "AMD", "CDNA3", "kernel-optimization"]
category: "Technical"
lang: "zh"
---

源码精读系列第五篇， 题材完全切换： 从 Python 重的 serving 系统跳到手写 GPU 汇编。 代码量不大（约 7.3 万行散在 64 个独立文件夹里）， 但它覆盖的优化模式 —— 决定 AMD CDNA3 上一个 kernel 是跑到 30% 还是 95% 的所有关键招式 —— 范围极广。 完整 HTML 深读在 [/sources/gcnasm.html](/sources/gcnasm.html)。

## 为什么读这个 repo， 为什么是现在

AMD GPU 编程一直存在一个文档断层。 一边是 HIP 教程： 友好、 可移植、 几乎完全没法表达那六七个把 kernel 从"能跑"变成"跑得快"的关键手法。 另一边是 CDNA3 ISA 手册： 权威、 翔实、 全文 1200 页， 但从不告诉你一个"完整的优化好的 kernel"长什么样。

[carlushuang 的 gcnasm](https://github.com/carlushuang/gcnasm) 是一座一块板一块板搭出来的桥 —— 64 个文件夹， 每个都是一个完整可跑的程序， 演示一种具体技术。 没有库、 没有中央抽象、 没有任何把这些例子串起来的构建系统。 每个目录就是 `build.sh` + `.s`/`.cc`（运气好还有 README）。 这个结构本身就是重点： 每个 kata 都可以一个小时之内读完、 编译、 跑起来、 改一改。

对于工作需要"生成 AMD kernel"的人 —— 不管是手写、 codegen 还是用自主 agent 来做 —— 这个 repo 是校准数据。 它回答了官方文档不回答的问题： *一个熟练 AMD 工程师亲手写的 kernel， 应该长什么样？*

## 五条值得带走的发现

**1. 汇编器有一个结构性盲区。** `buffer_load_dword … offen lds` 在 gfx942 硬件里是一条真实存在的指令。 LLVM MC 汇编器（ROCm 7.1 / clang 20）拒绝接受它。 carlushuang 在 `vector_add_kernel.s:147-158` 的解法是用 `.long 0xE0511000` 手写 MUBUF 字节。 这意味着任何"生成 .s 让 clang 汇编"的 agent， 在结构上无法产出这类 pattern —— 这对 kernel 生成系统的设计有直接影响。

**2. `vmcnt(3)` 的算账是工程口传， 不是手册可以推出来的。** vector_add 这个 kernel 任何时刻都保持精确的 3 条 memory ops 在飞 —— 上一个 half-iteration 的 2 个 prefetch + 1 个 store。 这个 3 不在 ISA 手册里， 它是从"每个循环边界上哪些 ops 需要排空"这个推理里掉出来的。 `README.md:410-452` 把算账过程写清楚了。 这类知识 agent 要么需要被显式喂入， 要么必须通过测量重新发现。

**3. 用 SRD 处理 OOB 让 exec mask 完全消失。** 把 buffer SRD 的 `num_records` 设成 `N * sizeof(element)`， 硬件就会自动让越界 load 返回 0、 让越界 store 静默丢弃。 `vector_add` 这个 kernel 自始至终不碰 `exec` 寄存器 —— 64 个 lane 无条件执行每一条指令， 由硬件接管边界处理。 等价 CUDA 实现里要塞六七条 `s_and_saveexec_b64`； AMD 这边一条都没有。 这个 pattern 在所有 production AMD kernel（aiter、 opus_* 全家）里随处可见， 是"用硬件特性换软件复杂度"最干净的例子之一。

**4. MFMA tile shape 决定下游的一切。** `matrix_core/matrix_core.cc` 给同一条 `32×32×8 fp16` MFMA 演示了三种 layout 策略： 标准（按列存）、 swap A/B（按行存 + `buffer_store_dwordx2`）、 swap+swizzle（按行存 + `buffer_store_dwordx4`）。 C 矩阵的 layout 不是自由选择 —— 它由你怎么把 A、 B 喂给 MFMA 决定的。 先设计 LDS 再考虑 MFMA， 是新手 AMD GEMM 卡在 30% 峰值的常见姿势。

**5. `co-exec` 是 repo 里最被低估的工具。** 一个 320 行的 Python 脚本， 用 `clang++ -x assembler` 直接编译 `.s`（不走 hipcc、 不过 linker、 不依赖 ROCm 工具链）， 跑生成出来的 `.hsaco`， 反汇编看结果 —— 全部在一次调用里完成。 编译步骤约 200ms， 对比 hipcc 的 8–15 秒。 对一个每天试 500 个 variant 的 kernel 生成 agent， 这是"70 分钟任务"和"28 小时任务"的差别。

## ★ 一条重塑心智模型的洞察

> 汇编器不是最低层。 它下面还有一层 —— 原始机器字节 —— 某些最优 pattern 只能在那一层表达。 这意味着任何"目标产出 `.s` 源码"的 kernel 生成系统都有一个结构性天花板： 任何需要手写 MUBUF 字节、 用内部 opcode、 或者用未公开寄存器类的优化， 从汇编层往上是不可见的。 实用解法是分层生成 —— DSL 用于快速探索、 intrinsics 用于有希望的候选、 raw 汇编（必要时 patch 字节）留给最后 5% 的极限优化。 repo 本身就是这么组织的（`matrix_core/` 是 HIP+intrinsic、 `matrix_core_asm/` 是 raw 汇编、 `opus_*` 是 DSL）， agent 系统大概也该这样分层。

## 完整 HTML 里有什么

深度阅读分八个模块：

- **M0 — 全景图**： 64 个 kata 的六维分类（数据形状 × 操作类型）。
- **M1 — 十分钟搞懂 CDNA3**： wavefront、 CU / SIMD、 VGPR / AGPR 统一寄存器文件、 vmcnt / lgkmcnt FIFO、 buffer SRD 解剖。
- **M2 — vector_add_asm**： 逐行走读 5 个经典优化 pattern。
- **M3 — bandwidth_memread**： roofline 标定工具（MI308X 实测 4.56 TB/s）和 dead-store 防优化技巧。
- **M4 — Matrix Core / MFMA**： 三种 layout 策略、 AGPR 调度、 `s_nop 16` 死时间。
- **M5 — DPP wave reduction**： 6 阶段蝶式归约、 6 个 DPP 控制码、 bpermute 兜底通道。
- **M6 — co-exec 和 measure_ips**： 给 kernel agent 用的迭代速度基础设施。
- **M7 — 杂项**： magic integer division、 FP8 / INT4 转换、 atomic CAS、 硬件探查。

外加 7 张手绘 SVG 图（repo 罗盘、 CU 解剖图、 vmcnt FIFO 时序、 roofline 图、 MFMA tile 图、 DPP 蝴蝶图、 buffer SRD 解剖）、 一节 "Reefs"（6 个陷阱）、 一节 "Red Lines"（3 个 kernel agent 设计的结构性追问）。

**→ 完整深度阅读在 [/sources/gcnasm.html](/sources/gcnasm.html)** —— 整体用"逻辑分析仪"美学（深硅黑底 + 荧光绿 + 琥珀 + 数据青 + 时钟品红， Newsreader / Manrope / Geist Mono 字体）， 所有图都是手写的 inline SVG。

---

*上一篇： [源码精读 004 — mini-SGLang](/blog/source-reading-004-mini-sglang/)。 下一篇大概率写 [aiter](https://github.com/ROCm/aiter)（gcnasm 的 `opus_*` 例子底下那个 production AMD kernel 库）或者 Triton-ROCm 的 codegen 流水线 —— 两个方向都是 gcnasm 这层地基上的自然延伸。*
