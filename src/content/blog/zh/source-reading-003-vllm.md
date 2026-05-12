---
title: "源码精读 003 — vLLM， KV cache 变成虚拟内存的地方"
description: "63.3 万行代码、 20 个 attention backend、 7185 行 GPU model runner、 一次让引擎本身变成进程的架构重构。 推理引擎三部曲的收官。"
date: 2026-05-14
tags: ["source-reading", "MLSys", "inference", "vllm", "AMD"]
category: "Technical"
lang: "zh"
---

源码精读三部曲第三、 也是最后一篇。 大约 6 小时穿过 vLLM 的 63.3 万行代码。 完整 HTML 在 [/sources/vllm.html](/sources/vllm.html)。 这篇是收尾 —— 也是三篇里唯一对比比独立发现更重要的一篇。

## 为什么哪怕你日常用 SGLang， 这篇也值得看

vLLM 做了三件定义现代 LLM serving 的事： (1) **PagedAttention** —— KV cache 当虚拟内存， 用物理 block 表管理； (2) **continuous batching** —— prefill 和 decode 在同一 step 的 batch 里混跑； (3) **2024 年的 v1 重构** —— 把单体引擎拆成 `EngineCore` + `Worker` 双进程。 即便你日常主力是 SGLang， 你永远都在拿它跟 vLLM 做基准对比。 知道它的内部能告诉你哪些数据可信。

## v0 vs v1 —— 一个 repo 里的两套宇宙

代码库里两套都还在：

- **`vllm/engine/`** —— 老 v0， 单进程 `LLMEngine` 串行做所有事
- **`vllm/v1/`** —— 新 v1（v0.6 之后默认）， `EngineCore` + Worker 拆分， 异步是天生的

如果你在读老 issue、 老博客、 老 Stack Overflow 答案， 一半引用的是 v0 类， 这些类已经不在执行路径上了。 第一件要 grep 的事： `VLLM_USE_V1`。

## 五条值得带走的发现

**1. `EngineCore` 有三种部署形态 —— 一条继承链。** 继承结构： `EngineCore`（进程内） → `EngineCoreProc`（子进程， API server 默认） → `DPEngineCoreProc`（数据并行） → `EngineCoreActor`（Ray actor， 多机用）。 同一份逻辑， 三种运行模式。 OO 多态在系统问题上的漂亮应用。

**2. PagedAttention 用哈希链， 不用树。** `block_pool.py` 定义了 `BlockHashToBlockMap`。 每个 16-token block 的 hash 包含它父 block 的 hash —— 所以两个请求 prefix 完全一致时， 自然 hash 到同样的 block（共享物理 KV slot， 带引用计数）。 查找 O(1)； 代价是粒度（只能共享 16-token 对齐的 prefix）。

**3. Scheduler 每个 step 跑两阶段。** Phase A： `running` 请求继续 decode。 Phase B： 从 `waiting` admit 新请求， 做 chunked prefill。 两阶段在一次 `schedule()` 调用里交错（`v1/core/sched/scheduler.py:310`）。 KV cache 压力高峰时， 低优先级的 `running` 请求被 preempt 回 `waiting`。 这就是"continuous batching" —— 不是"一直在跑"， 而是"prefill 和 decode 在 batch 维度共存"。

**4. `gpu_model_runner.py` 是 7185 行 —— vLLM 最大单文件。** 它协调： model forward、 attention backend dispatch（20 个 backend 选一）、 KV cache 写入、 CUDA Graph capture/replay（按 shape bucket）、 LoRA adapter 切换、 投机解码（draft + verify）、 EP MoE 的 all-to-all、 混精度路径。 它之所以这么大， 跟 SGLang 4006 行 scheduler 同源 —— 协调的关注点太多， 拆开反而不舒服。

**5. ROCm 支持是三个 backend， 不是一个。** `rocm_aiter_fa.py`（1471 行， MI300X+ 默认）、 `rocm_aiter_unified_attn.py`（304 行， 较新的 unified prefill+decode）、 `rocm_attn.py`（545 行， 不依赖 AITER 的 fallback）。 加一个顶层 `vllm/_aiter_ops.py` 包装一次 AITER ops， 被三个 backend 共同消费。 这比 SGLang 单一 3284 行 `aiter_backend.py` <em>更模块化</em> —— 但找东西也更费力。

## ★ 读完两个引擎之后的总结

> SGLang 和 vLLM 解决同一个问题（高吞吐 LLM serving）， 但选了<strong>不同的核心抽象</strong>。 SGLang： token 级 prefix 树（细粒度， 灵活匹配， 树维护成本）。 vLLM： 16-token block 上的哈希链（粗粒度， O(1) 查找， 操作系统页表的类比）。 两者最终都走到了 continuous batching、 chunked prefill、 投机解码。 抽象的选择持续存在， 塑造了后续所有设计。
>
> 如果今天要为生产 serving 选一个： vLLM 适合通用 OpenAI 兼容 serving + 部署灵活性最强（进程内 / 子进程 / Ray）； SGLang 适合多轮对话、 结构化生成、 prefix-heavy workload。 AMD 上： 两者都能跑， 都依赖 AITER， 但 vLLM 的 wrapper 层（`_aiter_ops.py`）对加自定义 kernel 更友好。

## 完整阅读里有什么

四张手绘 SVG 大图： v0 vs v1 架构对比、 EngineCore × Worker 拓扑（带 IPC 标签）、 PagedAttention 图（两个请求共享物理 block 7 和 12）、 vLLM vs SGLang 六维度并排对比表。

**→ 完整深度阅读： [/sources/vllm.html](/sources/vllm.html)** —— 视觉方向定为"1745 年航海图"： 深海军蓝、 羊皮纸奶油色、 金色和绿松石点缀、 通篇 Garamond 斜体。

---

*三部曲到此终结。 完整阅读顺序：*
- *[源码精读 001 — SkyPilot](/blog/source-reading-001-skypilot)（编排层）*
- *[源码精读 002 — SGLang](/blog/source-reading-002-sglang)（推理， 树形）*
- *[源码精读 003 — vLLM](/blog/source-reading-003-vllm)（推理， 块形） ← 你在这里*

*三个 repo 加起来覆盖了现代 AI infra 完整栈， 从"在任意云上调度一个 job"到"以 5000 tok/s 服务 70B 模型"。 三篇读完会发现同样的五个工程模式 —— 不可变值对象、 集中状态管理、 异步 IPC、 scheduler/executor 拆分、 plugin 抽象 —— 在三个 repo 反复出现。 这种收敛本身就是教训。*
