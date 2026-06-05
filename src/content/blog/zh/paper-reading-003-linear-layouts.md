---
title: "论文精读 003 · Linear Layouts： 每个张量布局都是 F₂ 上的一个矩阵"
description: "精读《Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using F2》。 核心一招： GPU 索引本来就是位， 所以每个 Triton 布局 —— Blocked、 MMA、 swizzled、 sliced —— 都是 GF(2) 上的一个二元矩阵。 布局转换坍缩成 B⁻¹A， 广播变成一个零列， 无 bank 冲突的 swizzling 变成一次子空间搜索。 附 AMD 视角： mfma 布局同样是线性的。 手写 SVG 配图， 双语。"
date: 2026-06-05
tags: ["paper-reading", "triton", "layout", "kernel", "AMD", "MLSys"]
category: "Technical"
lang: "zh"
---

论文精读第 003 篇。 这一篇新机器很少、 重组很多 —— 正是我喜欢的那类论文。 《Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using F2》（[arXiv:2505.23819](https://arxiv.org/abs/2505.23819)）押了一个赌注： GPU 编译器需要的每一种张量布局， 都是 *二元域上的一个线性函数*， 而这一个事实几乎溶解掉了整个布局问题。 完整双语 HTML 深读在 [/sources/linear-layouts.html](/sources/linear-layouts.html)。

## 为什么值得读

张量布局， 就是从一个逻辑元素 —— `A[row, col]` —— 到承载它的物理资源的映射： 哪个寄存器、 在哪个线程、 哪个 warp， 或者共享内存的哪个字节。 搞错了， Tensor Core 读到的就是垃圾； 搞得不够优， 你就要为本不必要的数据搬运买单。 历来每个 DL 编译器都手写这一层： 每种布局一套接口方法、 每一对布局一段转换例程、 一串魔法 swizzle 常数。 Triton 自己的 bug tracker 把代价说得很直白 —— **所有提交的 bug 里有 12% 跟布局相关**。

对我们 AMD 这边来说， 这不是 Triton 的趣闻。 真正的 NV→AMD 移植不是语法翻译； 一个 CUDA kernel 只有在它的数据被重新 tile 成 64 宽 wavefront、 重新 swizzle 对齐到 LDS bank 之后， 才能在 MI300X 上跑得 *好*。 论文证明了 AMD 的 `mfma` 布局同样是线性的， 而它的两个旗舰算法都是硬件无关的。 所以我们要造的任何移植 agent 或 kernel 优化循环的布局层， 都可以从这个抽象出发， 而不是从一张命名特例的注册表出发。

## 五个值得带走的发现

1. **一个布局就是 F₂ 上的 `y = M·x`。** 把硬件索引的位（寄存器、 线程、 warp）拼成一个向量 `x`； 把张量坐标 `(i, j)` 写成向量 `y`； 布局就是一个 0/1 矩阵 `M`， 满足 `y = M·x`， 其中乘法是 AND、 求和是 XOR。 `M` 的每一行说明一个输出位如何由输入位 XOR 拼出来。 这就是整个抽象。

2. **转换坍缩成 `B⁻¹A`。** 旧做法需要每一对 `(源, 目标)` 一个专门的转换器 —— 一堆 O(n²) 数量的手写路径， 每个都是新的出 bug 机会。 有了矩阵， 把数据从布局 A 送到 B 就是 `B⁻¹∘A`， 由一个通用的 F₂ 高斯消元例程算一次。 那一整族二次方数量的转换器变成单个算法。 把结果按资源（寄存器/线程/warp）分解， 甚至告诉编译器数据 *需要* 往哪搬 —— warp 块是单位阵就意味着"没有 warp 间搬运"， 这是用 warp shuffle 取代往返共享内存的绿灯。

3. **两个干净的定义直接掉出来。** *分布式* 布局（Blocked、 MMA/wgmma/mfma、 Sliced）是带可选零列的置换矩阵 —— 而 **零列恰好就是广播**， 这以前是个顽固的 bug 来源。 *内存* 布局可逆， 每列有一个或两个 1； 两个 1 的情形是一个剪切 `I + C`， 其中 `C` 混入坐标位 —— 那个剪切 *就是* mma swizzling， 从它 `per_phase`/`max_phase`/`vec` 的咒语里被解出了真身。

4. **最优 swizzling 变成一次子空间搜索。** 一个 bank 冲突恰好是 `span(Seg) ∩ span(Thr) ≠ {0}`。 算法从线程访问子空间的 *补空间* 里构造段的基， 让不同的段撞到不同的 bank， 只有当安全空间耗尽时才借用冲突空间里的向量 —— 给出一个对 *任意* 布局、 在 *任意* 厂商上都可证明冲突最小的 swizzle。 同一套机器也生成最优的 warp shuffle。

5. **回报是鲁棒性优先、 速度其次。** 在 265 个真实 TritonBench case 上： 最高 1.40×、 平均 1.07×。 更响的数字是正确性和 micro 收益 —— 混合精度 matmul 通过率 46.6% → **100%**（784 个 case）、 LD/ST 向量化宽度最高 7×、 广播的共享内存 store −76%、 布局转换 shuffle 最高 3.93×、 gather 最高 14.20×。

## ★ 重置我心智模型的那一个洞见

> AMD–NVIDIA 的布局差距不是物理。 swizzle 的数学在 F₂ 上完全一样； NVIDIA 有而 AMD 没有的， 是十年积累的像 `ldmatrix` 这样、 能把最优布局变成一条指令的原语。 在 MI250 上框架是正确的， 但只拿到 1.00×–1.03× —— 被缺失的原语卡住， 而不是被代数卡住。 那是软件成熟度的差距， 而软件成熟度的差距， 恰恰是系统化搜索能补上的。 论文把搜索交到了我们手里。

## 完整深读里有什么

六张手写 SVG 配图、 EN/ZH 切换、 磷光示波器风格： 代数阶梯（群 → 环 → 域 → F₂）； 核心的 `y = M·x` 映射， 配一个手算的位向量例子； 16×16 引子例子， 追踪线程 t9 的寄存器 r1 落在哪； 四个算子（复合、 积、 左除、 右逆）； 分布式 vs 内存布局（置换 + 零列 vs 剪切）； 以及把 bank 冲突看成子空间交集， 配一次拆成四个无冲突事务的读取。 收尾一节把这一切映射到 MI300X 上一个具体的"先赢下来"的 kernel。

**→ 完整深读在 [/sources/linear-layouts.html](/sources/linear-layouts.html)** —— 近黑 CRT 辉光配色， 绿/青/品红磷光， 二元矩阵用位网格渲染。

---

*上一篇： [论文精读 002 · Kernel Design Agents](/zh/blog/paper-reading-002-kernel-design-agents/)。 系列： 精读那些喂给我们 AMD kernel-agent 目标的 MLSys 工作。*
