---
title: "PD 分离不是 kernel： GLM-5.3 上 PD 与 AMD monokernel 的结合"
description: "为什么在 MI355X 上只做 PD 分离不能让 GLM-5.3 的 decode 变快， 怎样让 AMD monokernel 跑在使用 MoRI 的 decode 服务上， 以及为什么在 InferenceX AgentX 上两份普通副本仍然更好。"
date: 2026-11-15
tags: ["inference", "SGLang", "PD-disaggregation", "monokernel", "MI355X", "GLM-5.3", "AgentX"]
category: "Technical"
lang: "zh"
---

TileRT 在 8 张 MI355X 上把 GLM-5.3 的单用户 decode 速度做到了 469 tok/s（[TileRT](https://www.tilert.ai/blog/tilert-amd-agentx.html)）。 做法是一次只处理一个请求， 用一个常驻的 decode kernel， 把 prefill 交给另一台服务： 一个 prefill 节点加一个 decode 节点， 也就是 1P1D。 我们用同样的方式跑 SGLang， 第一批数据按每 GPU 算比单节点慢了一倍左右。 这看起来像是 PD 分离不行， 其实不是。 它测出来的是 decode 服务上跑的是哪个 kernel。

这篇文章讲的是把 AMD 的 monokernel 放到 GLM-5.3 的 PD 分离后面： 要做什么， 换来了什么， 以及为什么在我们真正关心的那张图上， 它仍然没有赢过两份普通副本。 表格、图和完整命令见 [Experiment 007](/sources/glm53-pd-monokernel-mi355x.html)。

## PD 改变了什么， 没改变什么

从用户的感受出发。 单用户的 decode 速度等于 1 除以一个 decode 步骤的时间。 这个时间由执行这一步的 kernel 和同时参与这一步的请求数决定。 PD 分离对这两者都没有直接影响。 它把 prefill 挪到别的 GPU 上， 别人的长 prompt 就不会插进你的 decode 步骤之间。 这保护的是尾部： 慢步骤更少， token 间隔更稳定。 它不会让一个普通步骤变快。

代价是 GPU。 InferenceX 的 AgentX 图按每 GPU 吞吐量画， 部署里的每张 GPU 都算在内。 一个 8 卡模型做 1P1D， 要用 16 张卡。 除非 prefill 节点做的工作本来会拖慢 decode， 否则 PD 会让每 GPU 吞吐量减半。

所以当我们的 PD decode 服务只有 213 tok/s/user， 只有 TileRT 469 的一半， 和不开 monokernel 的普通 SGLang 一样时， 第一个问题不是 PD， 而是： decode 用的是哪个 kernel？

## 为什么 decode 服务走了慢路径

AMD 为 GLM-5.3 写的 monokernel（在 [sgl-amd-mono](https://github.com/sgl-project/sgl-amd-mono) 里）把一个 decode 层融合成一次启动， 只覆盖一小组配置。 范围小是有意的： 融合 kernel 对 cache 布局、批次形状和同步方式都有假设， 所以每个模型的 spec 都声明自己验证过哪些服务功能；只要正在使用的功能不在列表里， SGLang 就拒绝启动 monokernel。 PD 分离就是这样一个功能。 DeepSeek-V4.1 的 spec 声明了它（[sgl-amd-mono#70](https://github.com/sgl-project/sgl-amd-mono/pull/70)）， GLM-5.3 的没有。 所以我们的 PD decode 服务跑的是 SGLang 自己的路径， 单用户速度反映的正是这一点。

改动在 [sgl-amd-mono#136](https://github.com/sgl-project/sgl-amd-mono/pull/136)， 很小： 在 target 和 NextN draft 的 spec 里声明这个功能， 并且只放行使用 MoRI 传输后端的 decode 服务， 其他 PD 角色一律拒绝。

为什么只放行 decode？prefill 跑的是长分块， monokernel 本来就不是为它设计的。 prefill 服务继续走 SGLang 的路径， 而且在这个负载上它大部分时间都是空闲的。

为什么在 decode 上是安全的？kernel 只读 MLA KV cache 和 indexer 的 key cache。 SGLang 的 DSA 内存池会把这两块缓冲区都注册给传输引擎， 所以 MoRI 把每个新请求的 KV 写进的， 正好是 kernel 要读的那块内存。 kernel 本身没有改动。

为什么只放行 MoRI？因为只在它上面验证过： 每个节点 8 条 400G RoCE 链路。 放行别的后端， 就是没有证据的说法。

质量没有变化。 GSM8K 全部 1319 题， 开 decode monokernel 的 PD 得分 0.9742， 不开 monokernel 的单节点服务是 0.9750。 有 9 道题结果不同， 5 道是单节点对、4 道是 PD 对（McNemar p = 1.0）。

## monokernel 在 PD decode 服务上带来了什么

开了 monokernel 之后， 低并发下 PD decode 服务和单节点的 decode 一样快： c1 的单用户速度从 213 提高到 474 tok/s， token 间隔和单节点一样是 2.1 ms。

| 并发 | 单用户速度 P50， 关 → 开 | ITL P90 ms， 关 → 开 | 每 GPU 吞吐量， 关 → 开 |
|---|---|---|---|
| 1 | 213 → 474 | 4.99 → 2.32 | 1308 → 1448 |
| 2 | 205 → 467 | 5.84 → 3.11 | 1186 → 1373 |
| 4 | 208 → 466 | 6.09 → 3.96 | 1263 → 1706 |
| 10 | 148 → 162 | 8.25 → 7.84 | 4108 → 4231 |

c10 这一行第一次暴露了真正的限制。 monokernel 每步最多处理 2 个请求， 而在并发 10 下， 大多数步骤的请求数都超过这个数。

和开 monokernel 的单节点相比， 从 c2 起 PD 在尾部占优。 c4 时它的 ITL P90 是 3.96 ms， 单节点是 7.14 ms， 因为不再有别人的 prefill 和 decode 共用 GPU。 这正是 PD 分离承诺的好处， 而且确实存在。

## 真正该比的基线

但单节点不是合适的基线。 PD 用了 16 张卡， 公平的比较是把同样的 16 张卡换一种用法： 两台单节点服务放在 SGLang router 后面， 按缓存分流， 让每个会话的后续轮次回到保存着它前缀的那台节点。

| 并发 | 部署 | 单用户速度 P50 / P90 | TTFT P50 | 端到端延迟 P90 | 每 GPU 吞吐量 |
|---|---|---|---|---|---|
| 4 | 两节点 | 456 / 165 | 0.21 s | 9.7 s | 1876 |
| 4 | PD + monokernel | 466 / 252 | 0.24 s | 10.1 s | 1706 |
| 10 | 两节点 | 216 / 130 | 0.23 s | 9.4 s | 4492 |
| 10 | PD + monokernel | 162 / 128 | 0.34 s | 13.2 s | 4231 |

PD 在 c4 保住了尾部。 到 c10， 两份副本在单用户速度中位数、首 token 时间、端到端延迟和每 GPU 吞吐量上都领先， 尾部持平。

为什么？这些是 agentic coding 的 trace： 会话很长， 每一轮只在长上下文后面追加一点内容。 约 95% 的 prompt token 命中 prefix cache， 每一轮新的 prefill 只有几千个 token。 prefill 节点没什么事做， PD 消除的干扰也很小。 两份副本把 16 张卡全部用在 decode 上， 每份服务一半用户。 这样每台节点的批次也减半， 而批次小正是 monokernel 能起作用的情况。 只有 prefill 重到会拖累 decode 时， PD 分离才划算。 这里不是这种情况。

## 怎么读 AgentX 图

InferenceX 的 AgentX 图， x 轴是 P90 E2E 归一化交互性， y 轴是每 GPU 的总吞吐量。 在这张图上， TileRT 的点（98.4， 16 张卡 1534 tok/s/GPU）已经落在公开的 SGLang 单节点 monokernel 曲线左下方。 后者在 c1 是 128.4， 8 张卡 2834 tok/s/GPU。 batch 为 1 的 decode 换来了单用户速度， 代价是 GPU；而且 E2E 归一化交互性还把首 token 时间算了进去。

所以要超过的是我们自己的曲线。 本地数据里， 这条曲线的前沿由单节点 c2、两节点 c10 和单节点 c10 组成， 没有一个 PD 点到达前沿。

## 真正的杠杆： 覆盖率

公开曲线从 c4 的 122.8 掉到 c10 的 58.7。 decode 日志说明了原因： c10 时只有 35% 的 decode 步骤最多只有 2 个请求， 这是 monokernel 的上限， 更宽的步骤都走 SGLang 自己的路径。 12 行（3 个请求， 每个请求 4 个 MTP token）能覆盖 47% 的步骤， 16 行能覆盖 61%。

为什么只有 8 行？kernel 把一个 workgroup 的工作数据放在 LDS 里， 静态预算是 32768 个字（128 KiB）。 tail 和 NextN 入口阶段要把每一行的 bf16 hidden state 都放进去， GLM-5.3 的 hidden size 是 6144， 每行 3072 个字， 12 行就要 36864 个字。 每 8 行启动一次的办法对 tail 可行， 因为它每步只跑一次。 每一层都这样做就会重复读权重， 而 decode monokernel 存在的意义就是避免这件事。

我正在把它加宽到 12 行。 已经解决了 6 处限制： 宽度表；tail， 现在拆成 8 行和 4 行两次启动；NextN 入口， 限制在 8 行， 更宽的 draft 步骤回退；输入投影里一个流式读权重的捷径；本步新 KV 行的暂存区域；以及一个要求 2 的幂的 Triton 表 kernel。 dense 层在 12 行下和 SGLang 吻合。 第一个 MoE 层不行。 它从 up 和 gate 到 down 的交接是按两个 4 行组设计的；在 verify 批次里多行会选中相同的 expert， down 阶段等 FP8 中间结果时超时。 这个阶段需要重新划分行的分组， 这是下一步的工作。

## 给下一个人的建议

- 评估 PD 分离时， 基线要用同样多的 GPU。 否则测到的是“多了一台机器”， 而不是“换了一种分工”。
- 按每 GPU 吞吐量算， PD 要为 prefill 节点买单。 prefill 很重、并且和 decode 争抢资源时， PD 才划算；前缀复用很多的 agentic trace 恰恰相反。
- 对融合 decode kernel 来说， 它能处理多宽的步骤， 决定了曲线在哪里拐弯。 在 AgentX 上， 覆盖率比部署拓扑更重要。

这里用到的单节点 monokernel 曲线来自 [Experiment 006](/sources/glm53-agentx-hillclimb-mi355x.html)。 这篇文章的数据见 [Experiment 007](/sources/glm53-pd-monokernel-mi355x.html)。
