---
title: "Source Reading 003 — vLLM, Where KV Cache Became Virtual Memory"
description: "633k lines, 20 attention backends, a 7,185-line GPU model runner, and an architecture rewrite that turned the engine itself into a process. The closing of the inference-engine trilogy."
date: 2026-05-14
tags: ["source-reading", "MLSys", "inference", "vllm", "AMD"]
category: "Technical"
lang: "en"
---

Third and final in the source-reading trilogy. ~6 hours through vLLM's 633k lines. Full HTML at [/sources/vllm.html](/sources/vllm.html). This is the wrap-up — and the only one where comparison matters more than absolute findings.

## Why this one matters even if you use SGLang

vLLM did three things that defined modern LLM serving: (1) **PagedAttention** — KV cache as virtual memory with physical block tables; (2) **continuous batching** — prefill and decode mixed in one step's batch; (3) **the v1 rewrite (2024)** — splitting the monolithic engine into `EngineCore` + `Worker` processes. Even if your daily driver is SGLang, you'll be benchmarking against vLLM forever. Knowing its internals tells you what numbers to trust.

## v0 vs v1 — two universes in one repo

The codebase still contains both:

- **`vllm/engine/`** — old v0, single-process `LLMEngine` doing everything serially
- **`vllm/v1/`** — new v1 (default since v0.6), `EngineCore` + Worker split, async-by-design

If you're reading old issues, blog posts, or Stack Overflow answers, half of them reference v0 classes that no longer drive the codepath. First thing to grep: `VLLM_USE_V1`.

## Five findings worth carrying

**1. `EngineCore` has three deployment forms — one class hierarchy.** Inheritance: `EngineCore` (in-process) → `EngineCoreProc` (subprocess, default for API server) → `DPEngineCoreProc` (data parallel) → `EngineCoreActor` (Ray actor, for multi-node). Same logic, three runtime modes. Beautiful application of OO polymorphism to a systems problem.

**2. PagedAttention uses hash chains, not trees.** `block_pool.py` defines `BlockHashToBlockMap`. Each 16-token block's hash includes its parent block's hash — so two requests with identical prefixes naturally hash to the same blocks (and share physical KV slots, with reference counting). Lookup is O(1); the cost is granularity (only 16-token-aligned prefixes share).

**3. The Scheduler runs two phases per step.** Phase A: continue decoding for `running` requests. Phase B: admit new requests from `waiting` with chunked prefill. The two interleave inside one `schedule()` call (line 310 of `v1/core/sched/scheduler.py`). When KV cache pressure peaks, low-priority `running` requests get preempted back to `waiting`. This is "continuous batching" — not "always running," but "prefill and decode coexist in the batch dimension."

**4. `gpu_model_runner.py` is 7,185 lines — vLLM's largest single file.** It coordinates: model forward, attention backend dispatch (out of 20 backends), KV cache writes, CUDA Graph capture/replay (per shape bucket), LoRA adapter swapping, speculative decoding (draft + verify), EP MoE all-to-all, mixed-precision paths. The reason it's huge is the same as SGLang's 4,006-line scheduler — they coordinate too many concerns to comfortably split.

**5. ROCm support is structured into three backends, not one.** `rocm_aiter_fa.py` (1,471 lines, default for MI300X+), `rocm_aiter_unified_attn.py` (304 lines, newer unified prefill+decode), `rocm_attn.py` (545 lines, AITER-free fallback). Plus a top-level `vllm/_aiter_ops.py` that wraps AITER ops once and is consumed by all three backends. This is *more modular* than SGLang's single 3,284-line `aiter_backend.py` — but harder to find your way through.

## ★ The takeaway from reading both engines

> SGLang and vLLM solve the same problem (high-throughput LLM serving) but chose **different core abstractions**. SGLang: prefix tree on tokens (fine-grained, flexible matching, tree maintenance cost). vLLM: hash chain on 16-token blocks (coarse-grained, O(1) lookup, OS-page-table analog). Both converged on continuous batching, chunked prefill, speculative decoding. The abstraction choice persists and shapes everything else.
>
> If you're picking one for production serving today: vLLM for general OpenAI-compatible serving with maximal deployment flexibility (in-process / subprocess / Ray); SGLang for multi-turn dialogue, structured generation, and aggressive prefix-heavy workloads. On AMD: both work, both lean on AITER, but vLLM's wrapper layer (`_aiter_ops.py`) is friendlier to add custom kernels through.

## What's in the full reading

Four hand-drawn SVG plates: v0 vs v1 architectural comparison, the EngineCore × Worker topology with IPC labels, a PagedAttention diagram showing two requests sharing physical blocks 7 and 12, and a side-by-side comparison table of vLLM vs SGLang on six axes.

**→ Full deep dive at [/sources/vllm.html](/sources/vllm.html)** — designed as an "1745 navigational chart": deep-ocean navy, parchment cream, gold and teal accents, italic Garamond throughout.

---

*The trilogy concludes here. Reading list:*
- *[Source Reading 001 — SkyPilot](/blog/source-reading-001-skypilot) (orchestration)*
- *[Source Reading 002 — SGLang](/blog/source-reading-002-sglang) (inference, tree-based)*
- *[Source Reading 003 — vLLM](/blog/source-reading-003-vllm) (inference, block-based) ← you are here*

*The three repos together cover the complete modern AI infra stack from "schedule a job on any cloud" to "serve a 70B model at 5000 tok/s." Reading all three reveals the same five engineering patterns — immutable value objects, centralized state, async IPC, scheduler/executor split, plugin abstractions — recurring across all three. That convergence is itself the lesson.*
