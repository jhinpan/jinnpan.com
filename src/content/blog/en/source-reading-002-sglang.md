---
title: "Source Reading 002 — SGLang, an Inference Engine That's Actually a Four-Process Distributed System"
description: "729,000 lines, 27 attention backends, a 4,006-line scheduler, and a radix tree that turns chat prefixes into KV cache hits. A deep reading of the inference engine I use daily at AMD."
date: 2026-05-13
tags: ["source-reading", "MLSys", "inference", "sglang", "AMD"]
category: "Technical"
lang: "en"
---

Second in the source-reading trilogy. ~6 hours through SGLang's 729k-line codebase. The full HTML deep dive with four SVG plates is at [/sources/sglang.html](/sources/sglang.html); this post is the highlights.

## Why this one

I use SGLang every day on MI300X — to serve Kimi-K2.5, Qwen3-Coder-Next, and to grid-search parallelism strategies. Knowing the engine internals turns "tune black-box flags" into "understand which knob does what." And anyone working on AMD inference eventually has to touch the AITER attention backend; this reading maps where exactly that is.

## What SGLang actually is

It's labelled an "LLM inference engine" but if you read past the README, the architecture is closer to a small distributed system:

- **Process A · TokenizerManager** — HTTP entry, runs the HF tokenizer
- **Process B · Scheduler** — `managers/scheduler.py`, the heart, 4,006 lines in one file
- **Process C · DetokenizerManager** — incremental detokenization, stop-string detection
- **Process(es) D · ModelRunner** — N processes for tensor / expert parallelism, run `gpu_model_runner.py` (3,607 lines)

All four communicate over **ZeroMQ PUSH/PULL sockets**. The topology is a ring: HTTP → Tokenizer → Scheduler → Worker → Scheduler → Detokenizer → Tokenizer → HTTP. A TP=8 deployment runs **10 Python processes** cooperating.

## Five findings worth carrying

**1. The Scheduler has two event loops.** `event_loop_normal()` and `event_loop_overlap()`. The overlap version is the killer: while GPU runs `forward(t)`, the CPU plans `forward(t+1)`. Implemented with CUDA streams + future state. This typically gives 10-25% throughput improvement on prefill-heavy workloads because CPU planning overhead gets completely absorbed.

**2. RadixCache is a real radix tree, not just a hash map.** `mem_cache/radix_cache.py` defines `RadixKey`, `TreeNode`, `RadixCache`. Each node holds (token sequence, KV slot indices, ref_count, last_access). New requests walk the tree to find the longest cached prefix, then re-use the KV slots for free. Eviction is LRU + ref_count==0. This is *finer-grained* than vLLM's 16-token block hashing.

**3. There are 27 attention backends.** All implementing one `AttentionBackend` abstract base, registered in `attention_registry.py`. AITER (for AMD ROCm) is a 3,284-line file by itself. FlashAttention has 4 variants. FlashInfer is its own family. Triton is the cross-platform fallback. The polymorphism is how SGLang stays at the bleeding edge of attention research — each new kernel paper gets a new backend, not a fork.

**4. `sgl-kernel` is a separate wheel package.** With *five* `pyproject_*.toml` files — CUDA, ROCm, CPU, MUSA, Metal. Same C++/Triton sources, five platform-specific wheels. This decouples "kernel improvements" from "runtime release cadence" — and lets other inference engines reuse the kernels.

**5. `skylet.py` (oh wait, wrong repo) — but the AITER backend has the same "thin wrapper + fat modules" pattern.** `AiterAttnBackend` (class, line 117) is 200 lines; the rest of the 3,284-line file is `AiterIndicesUpdaterPrefill`, `AiterMlaIndicesUpdaterPrefill`, `AiterMultiStepDraftBackend` — all "translate SGLang's tensor layout into what AITER kernels expect" helpers. This is the exact integration point AMD work flows through.

## ★ The insight that reframed my mental model

> Multi-process inference engines are not "thread-pool servers that happen to spawn workers." They are **explicit distributed systems with their own RPC**. SGLang chose ZMQ over gRPC because ZMQ is lighter (no protobuf serialization, no HTTP/2 framing) for the "same machine, multiple processes, high-frequency small messages" case. The cost is no schema versioning — but a single-binary deployment can afford that.
>
> If I build my own agent orchestration system for kernel optimization, the same trade-off applies: do I want gRPC (heavier, schema-evolving) or ZMQ (lighter, all-process-restart-on-schema-change)? SGLang says "go light unless you have a reason."

## What's in the full reading

Four hand-drawn SVG plates: the four-process topology with ZMQ wiring, the overlap scheduling timeline (showing how CPU plan hides inside GPU forward), the RadixCache tree with two real requests sharing a prefix, and the constellation of 27 attention backends with AITER highlighted.

**→ Full deep dive at [/sources/sglang.html](/sources/sglang.html)** — designed as a "1972 lab notebook": cream paper, dark warm ink, cobalt blue and crimson accents, all serif.

---

*Previous: [Source Reading 001 — SkyPilot](/blog/source-reading-001-skypilot). Next: [Source Reading 003 — vLLM](/blog/source-reading-003-vllm), the other inference engine — and why its PagedAttention chose a different abstraction.*
