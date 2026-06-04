---
title: "源码精读 008 — 读懂这条 Trace： rocprof-compute-viewer 与 rocprofv3"
description: "一份 AMD 指令级 GPU profiler 的实战指南： rocprofv3 如何采集 Advanced Thread Trace， 以及 rocprof-compute-viewer 的每一块面板到底怎么看。"
date: 2026-06-04
tags: ["source-reading", "MLSys", "AMD", "ROCm", "profiling", "kernel-optimization"]
category: "Technical"
lang: "zh"
---

计数器告诉你一个 kernel 慢了。 Trace 告诉你是哪条指令在等， 以及在等什么。 大多数 GPU profiling 止步于聚合量 —— occupancy、 实测带宽、 一张扁平的 top-kernel 排行榜 —— 它们只回答 *多少*， 从不回答 *在哪*。 要优化一个手写 kernel， 你需要另一条轴： 单个 SIMD 上单个 wave 的指令流， 每行都盖着 cycle 开销。 在 AMD 上这条轴就是 **Advanced Thread Trace（ATT）**， 而掌管它的是两个开源工具。

完整 HTML 深读在： [/sources/rocprof-viewer.html](/sources/rocprof-viewer.html)。

## 为什么值得读

`rocprofv3`（[rocprofiler-sdk](https://github.com/ROCm/rocprofiler-sdk) 里的 CLI）负责武装硬件、 跑你的程序、 再解码原始 SQTT token 流。 [rocprof-compute-viewer](https://github.com/ROCm/rocprof-compute-viewer)（RCV）—— 一个 Qt 桌面程序 —— 把解码结果变成可导航的图景： 每个 SIMD 的瀑布时间线、 带注释的 ISA 列表、 hotspot 直方图、 计数器叠加。 它们是一条管线的前后两端， 而大多数人从没学会读后面这端。 如果你在 Instinct 上写 kernel， 这就是「它是 memory-bound」和「第 214 行的 `s_waitcnt vmcnt(0)` 吃掉了 1920 个 cycle， 在等上面两条指令的那次 `global_load`」之间的差别。

## 五个值得带走的发现

1. **ATT 是刻意切出的一片， 不是整块 GPU。** 采集窗口由 `att_target_cu`（典型 `1`）、 `att_simd_select`（4 个 SIMD 里选哪几个）、 `att_shader_engine_mask`、 以及 `att_buffer_size`（`0x6000000` = 每个 SE 96 MB）决定。 窗口太宽或 buffer 太小会把 trace 尾部丢掉， 且不报错 —— 如果某个 wave 的时间线戛然而止， 先加大 buffer 再相信它。

2. **解码会自动运行， 而文件名就是硬件地址。** `rocprofv3 --att` 之后， ROCprof Trace Decoder 产出一个 `ui_output_agent_<pid>_dispatch_<N>/` 目录， 里面是 per-wave JSON， 命名为 `se{SE}_sm{SIMD}_sl{slot}_wv{wave}.json`。 RCV 左侧面板的选择器暴露的正是这四个字段 —— 读 trace 字面意义上就是在选一个坐标。

3. **每块面板回答一个问题。** Explorer（哪个 *文件* 烧了 cycle —— 每个文件带一条 latency bar）、 Utilization（哪个指令 *类别* 占主导 —— VALU / VMEM / SCALAR / OTHER）、 Compute Unit（单个 wave 的一生， 按 SIMD-slot 拆）、 Global（所有 wave， 按 kernel 着色）、 ISA（哪条 *指令* 在等）。 记住这个映射， 一墙面板就变成了一棵决策树。

4. **ISA 视图的杀手锏是 waitcnt 箭头。** 在 AMD 上一次 load 的开销很少落在 load 本身 —— 它落在后面那条 stall 到数据返回为止的 `s_waitcnt vmcnt`/`lgkmcnt` 上。 RCV 按 wave 画出 **memory 到 waitcnt 的依赖箭头**， 把这对因果关系画了出来。 一条 `s_waitcnt` 上挂着粗 Latency bar， 又有箭头指回某条 `global_load` —— 这就是教科书式的 memory-latency stall； 同样的形状落在 `ds_read` 上， 就是 LDS 的那种。

5. **ATT 与 PMC 是两次运行。** ATT 给逐指令的 stall 计时， 但没有 cache/HBM 计数器； `--pmc` 给计数器， 但没有指令流， 而且一组采不全就直接失败。 采两遍， 按 kernel 对应。 PC sampling（`--pc-sampling-beta-enabled`， 目前只有 `host_trap`/`time`）是便宜的统计折中 —— 有覆盖面， 没因果。

## ★ 重塑我心智模型的那一条

> 计数器是 *判决*； ATT 是 *地址*。 roofline 说「memory-bound」。 ISA 视图给你那条吃掉 cycle 的 `s_waitcnt`， 以及它在等的那次 `global_load`。 优化从不发生在判决上 —— 它发生在地址上。 ATT 的整套仪式（窄窗口、 分开运行、 buffer 的算术）存在的全部意义， 就是把一个数字和一行你能改的代码之间的环闭上。

## 完整深读里有什么

五张手画 SVG： 从采集到查看的管线； mask/select 参数在硅片上切出的东西； `se_sm_sl_wv` 文件名解成硬件坐标； 标注好的 RCV 窗口（Explorer · 时间线 · ISA · side panel）； 以及展示 wave 何时安静下来的 Utilization 泳道。 外加完整的 `rocprofv3` 模式分类（trace / PMC / PC-sampling / ATT， 加五种输出格式）、 RCV 逐面板指南（含 Counters、 Wave States、 Occupancy、 Dispatches、 Summary）、 一套从「这个 kernel 慢」到一行源码的五次点击流程、 Qt/CMake/LLVM-C 构建与它的 decoder 依赖、 以及那些会让 ATT 读错的暗礁。

它落在真实 trace 上： 这台 MI350X 上 `flydsl-kernel-profiling` 里每个 FlyDSL kernel 都已经带了一个装着这种采集结果的 `att_viewer/` 目录。

**→ 完整深读在 [/sources/rocprof-viewer.html](/sources/rocprof-viewer.html)** —— 一套「逻辑分析仪控制台」美学： 近黑底、 phosphor 绿 / 琥珀 / 青 / 品红、 monospace 为主， 带 EN/ZH 切换。

---

*上一篇： [源码精读 007 — AITER MoE Tuner](/zh/blog/source-reading-007-aiter/)。 这是持续连载的一部分， 读 AMD GPU 性能背后的系统。*
