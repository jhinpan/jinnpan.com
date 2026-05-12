---
title: "源码精读 002 — SGLang， 推理引擎也是个四进程分布式系统"
description: "72.9 万行代码、 27 个 attention backend、 4006 行 scheduler、 把对话 prefix 变成 KV cache 命中的 radix 树。 一份对我 AMD 工作每天都在用的推理引擎的深度阅读。"
date: 2026-05-13
tags: ["source-reading", "MLSys", "inference", "sglang", "AMD"]
category: "Technical"
lang: "zh"
---

源码精读三部曲第二篇。 大约 6 小时穿过 SGLang 的 72.9 万行 codebase。 带 4 张 SVG 大图的完整 HTML 在 [/sources/sglang.html](/sources/sglang.html)； 这篇博客是要点。

## 为什么读这个

我每天在 MI300X 上用 SGLang —— serving Kimi-K2.5、 Qwen3-Coder-Next， 网格搜并行策略。 知道引擎内部把"调黑盒参数"变成"理解每个参数动哪个旋钮"。 任何做 AMD 推理工作的人最终都得碰 AITER attention backend； 这次阅读精确定位了那是哪里。

## SGLang 到底是什么

它自称"LLM 推理引擎"， 但只要看过 README 之外的源码， 它的架构其实更接近一个小型分布式系统：

- **Process A · TokenizerManager** —— HTTP 入口， 跑 HF tokenizer
- **Process B · Scheduler** —— `managers/scheduler.py`， 心脏， 4006 行在单一文件里
- **Process C · DetokenizerManager** —— 增量解码、 stop-string 检测
- **Process(es) D · ModelRunner** —— N 个进程做 tensor / expert 并行， 跑 `gpu_model_runner.py`（3607 行）

四者全靠 **ZeroMQ PUSH/PULL socket** 通信。 拓扑是环形： HTTP → Tokenizer → Scheduler → Worker → Scheduler → Detokenizer → Tokenizer → HTTP。 一个 TP=8 的部署有 **10 个 Python 进程**协作。

## 五条值得带走的发现

**1. Scheduler 有两个 event loop。** `event_loop_normal()` 和 `event_loop_overlap()`。 overlap 版本是杀手 —— GPU 跑 `forward(t)` 的时候， CPU 同时 plan `forward(t+1)`。 实现靠 CUDA stream + future 状态。 prefill-heavy 工作负载上通常 10-25% 吞吐提升， 因为 CPU planning 开销被完全吸收掉。

**2. RadixCache 是真的 radix 树， 不是哈希表。** `mem_cache/radix_cache.py` 定义了 `RadixKey`、 `TreeNode`、 `RadixCache`。 每个节点持有（token 序列， KV slot indices， ref_count， last_access）。 新请求沿树查最长 cached prefix， 命中部分直接复用 KV slot。 驱逐策略是 LRU + ref_count==0。 粒度比 vLLM 的 16-token block hash <em>更细</em>。

**3. 有 27 个 attention backend。** 全部实现一个 `AttentionBackend` 抽象基类， 在 `attention_registry.py` 注册。 AITER（给 AMD ROCm 的）单文件 3284 行。 FlashAttention 有 4 个变体。 FlashInfer 自成一系。 Triton 是跨平台 fallback。 这种多态是 SGLang 紧跟 attention 研究前沿的方式 —— 每篇新 kernel 论文加一个 backend， 不需要 fork。

**4. `sgl-kernel` 是个独立 wheel 包。** 有<em>五份</em> `pyproject_*.toml` —— CUDA、 ROCm、 CPU、 MUSA、 Metal。 同一份 C++/Triton 源码， 五个平台专属 wheel。 这把"kernel 改进"和"runtime 发布节奏"解耦 —— 还允许其他推理引擎复用这些 kernel。

**5. AITER backend 的"薄包装 + 厚模块"模式。** `AiterAttnBackend`（class， line 117）只有 200 行； 剩下的 3284 行是 `AiterIndicesUpdaterPrefill`、 `AiterMlaIndicesUpdaterPrefill`、 `AiterMultiStepDraftBackend` —— 全是"把 SGLang 的张量布局翻译成 AITER kernel 期望的格式"的 helper。 这正是 AMD 工作流经的集成点。

## ★ 重新框定了我心智模型的洞察

> 多进程推理引擎不是"恰好 spawn 子进程的线程池服务器"。 它们是<strong>有自己 RPC 协议的显式分布式系统</strong>。 SGLang 选 ZMQ 而不是 gRPC， 是因为在"同机多进程 + 高频小消息"场景下 ZMQ 更轻（没有 protobuf 序列化， 没有 HTTP/2 帧）。 代价是无 schema 版本演进 —— 但单一二进制部署能承受。
>
> 如果我要做自己的 kernel optimization agent 编排系统， 同样的取舍： gRPC（重一些， 但 schema 可演进）还是 ZMQ（轻， 但 schema 改动要全进程重启）？ SGLang 的答案是"没特殊原因就选轻的"。

## 完整阅读里有什么

四张手绘 SVG 大图： 四进程 + ZMQ 总线拓扑、 overlap 调度时间线（展示 CPU plan 如何藏在 GPU forward 之内）、 RadixCache 树（含两个真实请求共享 prefix 的样例）、 27 个 attention backend 的全景图（AITER 加重突出）。

**→ 完整深度阅读： [/sources/sglang.html](/sources/sglang.html)** —— 视觉方向定为"1972 年实验室笔记"： 奶油纸面、 深棕墨迹、 钴蓝和暗红点缀、 通篇衬线字体。

---

*上一篇： [源码精读 001 — SkyPilot](/blog/source-reading-001-skypilot)。 下一篇： [源码精读 003 — vLLM](/blog/source-reading-003-vllm)， 另一个推理引擎 —— 以及它的 PagedAttention 为什么选了完全不同的抽象。*
