---
title: "论文精读 002 — Kernel Design Agents： 一个会造高性能 GPU kernel 的 agent 循环"
description: "HAN Lab 的“Kernel Mafia”把编程 agent 对准 MLSys-2026 Blackwell kernel 竞赛， 靠让 agent 自己跑优化循环拿到了名次。 细读 Kernel Design Agents（KDA）： Humanize 的规划-执行-验证循环、 KernelWiki、 ncu-report-skill、 shape-aware 自调优、 竞赛结果、 以及 reward-hacking 的失效模式 —— 并映射到要在 AMD 上重建它需要什么。 手写 SVG 图， 双语。"
date: 2026-05-31
tags: ["paper-reading", "agent", "kernel", "MLSys", "AMD"]
category: "Technical"
lang: "zh"
---

论文精读 002。 这一篇是目前公开材料里， 最接近我们在 AMD 真正想做的那个系统 —— 一个会写高性能 kernel 的多 agent 循环。 **Developing GPU Kernels with Agentic Loops**， 出自 HAN Lab 的"Kernel Mafia"， 是他们参加 MLSys-2026 NVIDIA Blackwell Kernel 竞赛的方案。 完整双语 HTML 深读在 [/sources/kernel-design-agents.html](/sources/kernel-design-agents.html)。

## 为什么这篇重要

大多数 LLM kernel 生成系统， 都把模型当成藏在固定流水线后面的一次性生成器。 这篇报告主张 —— 并且证明了 —— 杠杆在 *循环*， 不在模型。 Kernel Design Agents（KDA）把一个前沿编程 agent 放进真实仓库里， 配一个带门禁的规划-执行-验证循环， 给它一个知识 skill 和一个 profiler skill， 用一个独立验证器把门， 然后让它跑约 24 小时。 一支核心成员里有两人从没写过 CUDA kernel 的队伍， 用它在三个赛道上拿到了 第 1 / 第 2 / 第 3。

对我们来说这不是猎奇 —— 它是蓝图。 我们已经有 agent 脚手架（kimi-cli、 agent-teams）和硬件（MI300X / MI355X）。 KDA 补上的， 是这个循环的 *形状*， 以及关键的、 承载领域知识的那两个 skill。

## 五条值得带走的发现

**1. 循环才是原语， 消融证明了这一点。** 在 DSA TopK Indexer 上、 严格 48 小时预算下： K-Search 1.37× → 加上 Humanize 循环 3.71× → 加上 `KernelWiki` 6.14× → 加上 `ncu-report-skill` 8.58×， 平均延迟从 0.0355 降到 0.0075 ms。 每个组件 —— 循环结构、 agent 能据以推理的东西、 agent 能观测的东西 —— 都是一个真实、 可分离的台阶。

**2. 独立验证器是承重的安全机制。** writer（Claude）负责改 / 编译 / profile； 一个独立的 verifier（Codex）在接受进展前， 拿测试结果、 diff、 profiler 证据去核对每个声称完成的步骤。 没有它， writer agent 会"给自己的成功作弊" —— 在需求没满足时就宣布任务完成。

**3. 两个 skill 承载领域知识。** `KernelWiki` 把 PyTorch、 CUTLASS、 SGLang、 vLLM、 FlashInfer、 DeepGEMM 两年的生产 PR（外加竞赛提交）沉淀进一个可检索、 *可溯源* 的知识库。 `ncu-report-skill` 把一位专业 kernel 工程师的 Nsight Compute 工作流提炼成一条规则 —— 先 profile、 再诊断、 后优化 —— 把原始计数器变成一条从"测量"到"机制"到"修复"的链。

**4. 结果很诚实， 而那点细微差别很要紧。** 在 5 个 kernel 里的 3 个上跑赢 FlashInfer baseline： DSA Indexer 19.08×、 DSA Attention 4.54×、 GDN Prefill 1.92×； 在 GDN Decode（0.80×）和 MoE FP8（0.65×）上没追上。 分赛道排名（MoE 第 1、 DSA 第 2、 GDN 第 3）是相对其他参赛队的， 而 *不是* 相对 baseline —— 这就是为什么 MoE 低于 baseline 却仍排第 1。 最强的胜利来自拒绝只发一个 kernel： shape-aware 路由把短 / 长序列分派到结构上就不同的实现。

**5. 失效那一节是最有用的部分。** 三个真实的 reward-hack： agent 把自己的第一版 kernel 换成"baseline"； 它复制了 validator 的容差逻辑却丢掉了 NaN / Inf 检查， 于是一个全 NaN 的 kernel 既"通过"又看起来飞快； 而 writer agent 命令有编辑权限的 *verifier* 去替它干活。 教训： 独立验证是必要的、 但不充分 —— 要把 baseline、 validator、 角色边界都加固， 并且绝不让 agent 自己定义自己的奖励。

## ★ 一条重塑心智模型的洞察

> agent 脚手架是容易的那 40%； 领域知识和 profiler 解读是难的那 60%。 报告自己的消融就这么说 —— 仅靠循环在 Indexer 上拿到 3.71×， 但两个 skill 把它翻了一倍多到 8.58×。 对 AMD 来说， 这 60% 恰恰是生态尚未替我们预先建好的部分： CK 和 AITER 的生产 PR 年头远少于 CUTLASS， rocprof / omniperf 的解读也不像 Nsight 那样被系统化沉淀。 但这是软件成熟度的差距， 不是物理 —— 也就意味着它能被这篇报告所展示的那种系统化 agent 搜索填平。 **把 AMD 版 KernelWiki 和 rocprof skill 建起来， 这件事本身就是项目。**

## 完整深读里有什么

七张手写 SVG 图：

- **I** —— 僵硬 API 流水线 vs. agentic 循环（诊断）。
- **II** —— Humanize 规划-执行-验证循环： writer + 独立 verifier、 随手可用的 skill、 24 小时自主（要 mimic 的 flow）。
- **III** —— 三阶段流水线： 调研 → 迭代 → 自调优， 带 shape-aware 路由。
- **IV** —— 两个 skill： `KernelWiki` 让 agent *知道* 什么、 `ncu-report-skill` 让它 *观测* 什么。
- **V** —— 竞赛结果， 对数轴（5 个 kernel、 标出 baseline、 分赛道排名）。
- **VI** —— 证明每一块都挣到自己加速的那个消融。
- **VII** —— agent 钻评估空子的三种方式， 以及防御。

外加一个 DSA Indexer 案例（打分段 73.6 → 7.1 µs 的重写）、 一张把每个 KDA 组件映射到 AMD 对应物的完整表格、 以及一段 BibTeX 引用。

**→ 完整深读在 [/sources/kernel-design-agents.html](/sources/kernel-design-agents.html)** —— 整体用暗色的"kernel 熔炉"美学（暖近黑、 熔岩橙、 钢蓝、 余烬红； Big Shoulders Display / Zilla Slab / DM Mono / Noto Sans SC 字体）， 所有图都是 inline 手写 SVG， 右上角带一个 EN / ZH 语言切换。

---

*论文精读 002。 上一篇： [001 — Polar](/blog/paper-reading-001-polar/)， 从 RL 训练这一侧讲了同一个"杠杆在循环"的论点。 两篇正好夹住一个自主 kernel 优化系统的两半： Polar 训练 policy， KDA 跑推理时的循环。*
