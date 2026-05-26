---
title: "源码精读 006 — FlyDSL， 一个用 MLIR 撑骨架的 layout 代数 Python DSL"
description: "AMD 的 FlyDSL 是 Fly dialect MLIR 编译器的 Python 前端， 把 layout 代数和 copy / MMA atom 一路 lower 到 CDNA3 / CDNA4 的 ROCDL。 examples 目录下 vectorAdd、 tiledCopy、 tiledMma、 preshuffle GEMM 四个例子构成一条严格的进阶阶梯， 按顺序读完， 你就掌握了 paged attention、 MoE GEMM、 flash attention 这些 production kernel 反复重组的所有零件。"
date: 2026-05-26
tags: ["source-reading", "MLSys", "AMD", "CDNA3", "MLIR", "kernel-optimization"]
category: "Technical"
lang: "zh"
---

源码精读系列第六篇。 上一篇 [gcnasm](/blog/source-reading-005-gcnasm/) 是下沉到手写 CDNA3 汇编， 这一篇方向完全相反 —— 上升到一个 Python DSL， 但底下有完整的 typed MLIR IR， layout 代数被做成一等公民， copy / MMA atom 可以拼出 production GEMM 而全程不用离开 Python 编辑器。 完整 HTML 深读在 [/sources/flydsl.html](/sources/flydsl.html)。

## 为什么读这个 repo， 为什么是现在

读一个现代 GPU kernel 你会有一种特殊的眩晕感： 你会意识到这段代码大部分不是在做算术， 而是在做 layout。 一个 4096 立方的 FP16 GEMM， 逐行看下来， 绝大部分是簿记 —— 哪个 thread 加载哪 8 个元素、 共享内存里它们落在哪、 MFMA 指令以什么顺序消费它们、 下一个 K 块的 prefetch 在当前 MFMA 的尾巴上以多大比例重叠。 真正的乘法是两行。 其余全是 layout。

[FlyDSL](https://github.com/ROCm/FlyDSL) —— *Flexible Layout pYthon DSL* —— 是 AMD 对这个观察的回应。 你用 `@flyc.kernel` + `@flyc.jit` 在 Python 里写 kernel， 但底下是一个真实的 MLIR dialect（叫做 *Fly* dialect）， 有自己的 `!fly.layout` 类型、 layout 代数 op、 一整套 lower 到 ROCDL 和 HSA fatbin 的 pass 管线。 思想上的母本是 NVIDIA CuTe； AMD 的特定贡献是把这套代数做成 typed IR， 而不是 C++ 模板。

我花了 4 小时把 `examples/` 下面 4 个例子（vectorAdd、 tiledCopy、 tiledMma、 preshuffle GEMM）读了一遍， 发现它们是一条严格的教学阶梯。 每一个都比上一个多引入恰好一个概念， 第四个例子基本就是 production CDNA3 / CDNA4 GEMM 用到的全部优化的紧凑版。 按顺序读它们， 是进入这个 codebase 最高效的路径。

## 五条值得带走的发现

**1. Layout 是 typed IR 概念， 不是模板字符串。** FlyDSL 的定义性选择是： `!fly.layout<(8,16):(1,8)>` 是一个真实的 MLIR 类型。 像 `fly.logical_divide`、 `fly.partition_S` 这些 op 消费和产出这个类型的值。 流水线第 3 阶段的 `fly-layout-lowering` pass 把这套代数物化成具体的地址算术。 这意味着任何 MLIR-aware 工具（autotuner、 分析 pass、 verifier）都可以把 layout 当数据来 reason， 而不是把它当模板参数当字符串看。

**2. `partition_S` + `retile` 是真正赚回来的抽象。** 例 03 里， `thr_copy_A.partition_S(bA)` 直接给你形状是 `(V, VM, VN)` 的 per-thread fragment —— 而且已经按底层 MFMA atom 的 lane 布局索引好了。 然后 `retile` 把同一段寄存器从"copy 视角"重新解读成"MMA 视角"， 零代价。 没有 `retile` 就要分配第二个 fragment 然后做寄存器到寄存器的拷贝。 MLIR pass 管线会把两个视角折叠成一份 VGPR 分配。

**3. Preshuffle 在 production kernel 里反复出现， 不是一次性的技巧。** 例 04 的 `shuffle_weight` 把戏 —— 在 host 端把 B 重排， 让一条普通的 `buffer_load_dwordx4` 出来的就直接是 MFMA lane-correct 的 VGPR —— 不是孤例。 `kernels/preshuffle_gemm.py`、 `blockscale_preshuffle_gemm.py`、 `moe_gemm_2stage.py` 都在不同 MFMA 形状和 dtype 上用同一个思路。 对于推理时不变的权重， 这个 trade 省掉 B 的整个 LDS 来回 —— 既省掉 `ds_write` 流量， 也省掉本该用来保证 B 读取无 bank conflict 的 swizzle。

**4. Scheduler 是最后那 30% 性能的所在。** 一个没有 `fx.rocdl.sched_*` 提示的 FlyDSL kernel 大概能跑到峰值 FLOPs 的 60–70%。 调好 scheduler（MFMA 数、 `ds_read` 数、 `buffer_load` 数、 以及在 hot loop 内的精确交错）之后， 同一个 kernel 可以跑到 90%+。 `kernels/preshuffle_gemm.py` 里的 scheduler 是按 (BM, BN, BK, MFMA-shape) 元组分别调过的， 通常用 `rocprofv3` 抓的 ATT trace 来调， 你改了 tile size 之后它会静默失效。 例 04 那 30 行的 `hot_loop_scheduler` 就是这种 artifact 的最小形态。

**5. CUDA Graph capture 默认就能用， 是设计出来的。** 启动路径过的是 `fly-gpu-stream-inject` —— 一个把用户传入的 stream 直接编进 launch 指令的 MLIR pass， 不依赖任何 TLS 变量。 对于把 kernel 批量塞进 captured graph 做回放的推理引擎（vLLM、 SGLang）， 这就是"FlyDSL kernel 能用"和"FlyDSL kernel 是需要特殊处理的边角"的区别。 例 01 的第二个测试就是把 kernel capture 进 `torch.cuda.CUDAGraph` 然后正确 replay。

## ★ 一条重塑心智模型的洞察

> "Layout" 不是对内存的描述， 而是一个函数。 `make_layout(shape, stride)` 定义了一个映射 *coord ↦ index*。 `composition`、 `logical_divide`、 `product` 是这个映射上的函数组合 / 分割 / 扩展。 一旦你用这个视角去读 FlyDSL， "代码说什么"和"kernel 在做什么"之间的距离会瞬间缩小一个数量级。 这种"函数之上的函数"的框架， 也解释了为什么这套代数能撑过编译 —— 每个 pass 都在 layout-as-function 这个表示上操作， 直到最后一个 lowering 阶段才把它具体化成地址算术。 整个 pass 管线其实是一序列的 layout 函数变换， 不是一序列的模板替换。

## 完整 HTML 里有什么

深度阅读分八个模块：

- **M0 — Compass**： 四个例子的阶梯、 行数、 概念递进。
- **M1 — Layout 代数十分钟入门**： shape、 stride、 layout、 divide、 slice、 TV layout。
- **M2 — 例 01 vectorAdd**： 最小可用 kernel； BufferCopy vs UniversalCopy； @flyc.jit / Constexpr。
- **M3 — 例 02 tiledCopy**： 完整的 TV layout； `partition_S/D`； `(V, VM, VN)` 这个 fragment 形状是怎么来的。
- **M4 — 例 03 tiledMma**： MFMA atom； `make_tiled_copy_A/B/C`； `retile` 的双视角戏法。
- **M5 — 例 04 preshuffle GEMM**： host 端预洗、 LDS XOR swizzle、 两阶段流水、 `hot_loop_scheduler`。
- **M6 — 编译流水线**： Python → MLIR → ROCDL → fatbin； JIT 缓存； `FLYDSL_DUMP_IR` 调试流程。
- **Reefs**： 6 个真实调试过的陷阱 —— 分支限定值、 SmemPtr._view_cache、 失效的 scheduler、 等等。
- **AMD notes**： FlyDSL 相对 Triton-ROCm 和 Composable Kernel 的位置； 一套可操作的 kernel 调优工作流。

外加 6 张手绘的 inline SVG： 编译流水线全景、 两阶段 divide 级联、 tiledCopy 的 TV layout 覆盖网格、 64×64 C tile 上的 MFMA wave 平铺、 preshuffle B 的前后对照、 软件流水的时序图（展示 prefetch 怎么和 MFMA 重叠）。

**→ 完整深度阅读在 [/sources/flydsl.html](/sources/flydsl.html)** —— 整体用"cyanotype 蓝图"美学（深海军蓝底 + 粉笔白墨水 + 锈橙 / 黄铜 / 玉绿 / 青蓝 标注， EB Garamond + IBM Plex Sans + JetBrains Mono 字体）， 所有图都是手写的 inline SVG。

## 主要参考资料 —— 配套阅读

1. **[FlyDSL repo](https://github.com/ROCm/FlyDSL)** —— 先看 `docs/layout_system_guide.md` 的完整 Quick Reference， 再看 `docs/kernel_authoring_guide.md` 的实战 pattern。 `kernels/` 下面那些 production kernel 是字典， 4 个例子是字母表。

2. **[NVIDIA CUTLASS CuTe](https://github.com/NVIDIA/cutlass/tree/main/include/cute)** —— 思想上的母本。 Layout 代数、 copy / MMA atom 设计、 `partition_S/D` idiom 全部来自 CuTe， 移植到 AMD 加了 MLIR 骨架。 CuTe 文档和 FlyDSL 对照着读， 可以看清哪些选择是通用的、 哪些是 AMD 特有的。

3. **[Categorical Foundations for CuTe Layouts（Colfax Research）](https://arxiv.org/abs/2601.05972)** —— 把 layout 代数当做一个 category 来形式化处理。 足够推出 FlyDSL 依赖的所有代数恒等式。 想扩展代数、 提自定义的 `product` 变体、 或者验证两个 layout 等价的时候读这本。

4. **[AMD Instinct MI300 CDNA3 ISA Reference](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)** —— FlyDSL lowering 产出的每一条指令的权威手册。 § 8（MUBUF）对应 `buffer_load`， § 10（MFMA）对应 matrix core， § 6（`s_waitcnt`）对应 scheduling 在控制的 vmcnt / lgkmcnt 机制。

5. **[MLIR 文档](https://mlir.llvm.org/)** —— 用来读 `FLYDSL_DUMP_IR` 输出。 FlyDSL 组合用的 `gpu`、 `arith`、 `scf`、 `memref`、 `vector`、 `rocdl` dialect 文档都在这里； `fly` dialect 自己的文档在 repo 内的 `include/flydsl/Dialect/Fly/IR/` 里。

6. **[Triton-ROCm](https://github.com/triton-lang/triton)** —— 大部分读者熟悉的另一个 AMD kernel DSL。 FlyDSL 用 `fx.rocdl.sched_*` 的显式控制权， 换来了 Triton 那种"调度不透明"的损失。 两个 DSL 并排读可以看清设计空间： 同一个目标硬件， 两种不同的控制接口。

第一次接触 AMD kernel 编程的合理阅读顺序大致是 **№ 4 的 CDNA3 § 2-3 建立硬件直觉 → 这篇 → № 2 的 CuTe 学代数 → FlyDSL 例 01-04 → № 3 范畴论文章（可选）→ `kernels/` 里的 production kernel**。 MLIR 文档和 Triton 是查阅， 不是顺读。

---

*上一篇： [源码精读 005 — GCNasm](/blog/source-reading-005-gcnasm/)。 下一篇大概率写 [aiter](https://github.com/ROCm/aiter) —— FlyDSL 测试基础设施引用的那个 production AMD kernel 库， 是 layout 代数往上自然延伸的一层。*
