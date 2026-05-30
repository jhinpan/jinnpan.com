---
title: "论文精读 001 — Polar： 不打开盒子也能训练 Agent"
description: "Polar（arXiv 2605.24220， 出自 NVIDIA）训练语言 agent 的办法是代理它的 LLM API 调用， 而不是重写 harness。 集成点从 agent 本身挪到了模型 endpoint —— 正是我们用 SGLang 已经在跑的那道缝。 细读它的架构、 四步代理、 token-faithful prefix merging， 以及 SWE-Bench 结果， 配手写 SVG 图。"
date: 2026-05-30
tags: ["paper-reading", "RL", "agent", "MLSys", "SWE-Bench"]
category: "Technical"
lang: "zh"
---

这是"论文精读"系列的第一篇， 跟已有的"源码精读"系列并列 —— 同样的形式， 不同的对象。 不再是花六小时钻一个 codebase， 而是细读一篇论文， 配同样手写的 SVG 图， 同样较真每一个数字。 第一篇： **Polar： Agentic RL on Any Harness at Scale**（arXiv 2605.24220）， 出自 NVIDIA。 完整 HTML 深读在 [/papers/polar.html](/papers/polar.html)。

## 为什么读这篇， 为什么是现在

Polar 回答了一个我反复撞上的问题： 当 agent 是一个你没写过的复杂 harness —— Codex、 Claude Code、 自家的 CLI —— 而你又不想把它在 RL 框架里重建一遍时， 怎么用 RL 训练它？ 常规答案是把 harness 拆开， 重新表达成 `reset / step / reward`。 Polar 的答案是： 不拆。

它的核心论点就是论文里真正问出的那一句： *"能不能不打开盒子就用 RL 训练 agent？"* 任何 LLM agent， 不管内部多花哨， 总得调一次模型。 那个 API 调用是一个落在 agent 之外的公共接口。 在那里放一个代理， 抓住流过的 token， 重建轨迹， harness 就能在完全不改的前提下变得可训练。

这事对我们不抽象。 我们的 harness（kimi-cli 这些）本来就通过 SGLang 的 OpenAI 兼容 endpoint 指向本地托管的模型 —— 而那正是 Polar 把 RL 观测点放上去的那道缝。 这篇论文读起来， 像是给"包住我们已经在跑的 harness"画的一张蓝图。

## 五条值得带走的发现

**1. 贡献是挪动了边界， 不是换了算法。** Polar 把 RL 的集成点从 harness 挪到模型 endpoint。 一个 gateway 代理接收 Anthropic、 OpenAI Chat、 OpenAI Responses、 Google 风格的请求， 把它们归一化成 OpenAI Chat（`logprobs=true`）， 转发给本地 inference， 再把 provider 自己的格式原样返回给 harness。 harness 自始至终不知道模型被换掉了。 剩下的活儿， 朴素的 GRPO 就够了。

**2. 两个组件， 彻底解耦。** 有一个 *rollout server*（协调任务， 把一个 `TaskRequest` 展开成 `num_samples` 个 session）和若干 *gateway node*（各自掌管一个 session 的全生命周期， 并托管代理）。 trainer 是独立进程， 以服务的方式消费跑完的轨迹。 结果就是 Polar"对 agent harness、 训练基础设施、 RL 算法都不可知"。 换 harness、 换 trainer、 换算法 —— 没有一个会跨过服务边界。

**3. token-faithful prefix merging 是低调的精妙之处。** 把对话解码再重编码来重建轨迹会引入 *retokenization drift* —— 你训练用的 token ID 和模型当初采样出来的不一样。 Polar 的做法是： 采样出的 assistant token 逐字复制， 中间夹的 interstitial token 取自规范化的 tokenization， 没被生成的一律 mask 掉（只有 behavior-policy token 的 `loss_mask=1`）。 它还通过严格的 token 前缀判定 `p_{m+1}[1:|p_m|] = p_m`， 把 append-only 的多次 completion 缝回成一条长轨迹。 在同样三个训练 step 下， 这个合并把 1,185 次 request 级更新压成 218 条 merged trace， 把墙钟训练时间从 **189.5 分钟砍到 35.2 分钟（5.39 倍）**； 另外还把 rollout 的 GPU 利用率从 20.4% 抬到 87.7%。（注意： 5.39 倍是时间， 利用率是另一个指标， 别混。）

**4. 异步 staging 正好治我们的痛点。** 每个 gateway 跑 INIT / RUNNING / POSTRUN 三个 worker pool， 外加一个有界的 READY buffer， 让 CPU 重的容器准备提前跑， GPU 永远不用等开机。 evaluator 在 agent 运行期间就被 *预热*， 超时时 gateway 仍然进 POSTRUN 把部分轨迹捞回来。 这正是我们 kernel 迭代里"容器在构建、 GPU 在空转"那个老问题。

**5. 结果： 同一个 4B base， 四个 harness， 涨幅跟着 harness 的陌生程度走。** 从同一个 `Qwen3.5-4B` checkpoint 出发， 通过 Polar 跑 GRPO， SWE-Bench Verified 的 pass@1 分别涨了 **+22.6（Codex 3.8→26.4）、 +4.8（Claude Code 29.8→34.6）、 +0.6（Qwen Code 34.6→35.2）、 +6.2（Pi 34.2→40.4）**。 Codex 那一跳特别大， 是因为 base 几乎不熟它的 tool schema； 而 base 本来就顺手的地方（Qwen Code）， 空间就薄。

## ★ 一条重塑心智模型的洞察

> Agentic RL 最难的不是目标函数， 是集成 —— 而最省事的集成点， 是每个 agent 都已经暴露出来的那一个接口： 模型 API 调用。 一旦接受这一点， "把 harness 当黑盒"就不再是妥协， 而成了设计本身。 对于一支想把 RL 对准自家模型、 自家硬件的团队来说， 我们已经在对外提供的那个模型 endpoint， 本身就是训练接口。 不是把我们的栈去迁就 trainer， 而是 trainer 接到我们已经在跑的那道缝上。

## 完整深读里有什么

七张手写 SVG 图：

- **I** —— 把 agent 接进 RL 的两个位置（环境 API vs 边界处的代理）。
- **II** —— 系统架构： rollout server、 gateway node、 代理、 inference、 以及独立的 trainer。
- **III** —— 代理的四步请求生命周期（检测 → 归一化 → 捕获 → 返回 provider 格式）。
- **IV** —— 异步 staging： INIT / READY / RUNNING / POSTRUN 各 pool， 以及一条 evaluator 预热时间线。
- **V** —— token-faithful prefix merging： append-only 链、 前缀判定、 merged 序列与它的 loss mask。
- **VI** —— SWE-Bench Verified， 四个 harness 上 base 与 Polar-RL 的对比。
- **VII** —— prefix_merging 与 per_request 的消融， 三个指标分得清清楚楚。

外加一节"这对我们 AMD / kimi-cli 的工作意味着什么"、 完整的 GRPO 配方、 以及一段 BibTeX 引用。

**→ 完整深读在 [/papers/polar.html](/papers/polar.html)** —— 整体用淡色的"极地勘测"美学（冰川白底、 极地蓝与极光青、 一抹信号橙做强调； Bricolage Grotesque / Source Serif 4 / Spline Sans Mono 字体）， 所有图都是 inline 手写 SVG。

---

*论文精读 001。 姊妹篇 [源码精读系列](/blog/source-reading-005-gcnasm/) 对 codebase 做同样的事。 下一篇大概率还留在 agentic-RL 基础设施里 —— Agent Lightning 或 SkyRL-Agent —— 把 Polar 对标的那片领域补全。*
