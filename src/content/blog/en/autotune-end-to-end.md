---
title: "Autotune, End to End — From Triton to FlyDSL"
description: "How kernel autotuning is designed in Triton, used in aiter / quack / CuteDSL, and consumed by inference engines like SGLang — and, from all of that, how to design a real autotune path for FlyDSL. With a full HTML deep dive carrying six hand-drawn SVG plates."
date: 2026-06-30
tags: ["autotune", "GPU-kernels", "Triton", "CuTeDSL", "FlyDSL", "SGLang", "MLSys", "primer"]
category: "Technical"
lang: "en"
---

A GPU kernel has knobs that do not change *what* it computes, only how fast — tile sizes, the number of warps, software-pipeline stages, register limits, cluster width. The fastest setting of those knobs is not a constant. It depends on the input shape (a GEMM at `M=1` wants a different tile than at `M=8192`), on the dtype, and on the exact GPU. You cannot read the best config off a datasheet; you have to run the kernel and measure. That measure-and-pick process is autotuning, and every modern kernel DSL has some version of it.

This is a guide built by reading six real implementations from source — [Triton](https://github.com/triton-lang/triton) (the template everyone copies), [quack](https://github.com/Dao-AILab/quack) and [CuteDSL](https://github.com/NVIDIA/cutlass) (two DSLs that build on the idea), [aiter](https://github.com/ROCm/aiter) (offline tuning at a kernel library), [SGLang](https://github.com/sgl-project/sglang) (an inference engine that *consumes* tuned configs), and [FlyDSL](https://github.com/ROCm/FlyDSL) (which has the machinery but no adoption) — to answer one question: **how should FlyDSL's autotune actually be designed?**

> Full read: [/sources/autotune-end-to-end.html](/sources/autotune-end-to-end.html) — a bilingual HTML deep dive with six hand-drawn SVG plates (the autotune loop, in-place corruption, online vs offline timelines, cache-key anatomy, the proposed FlyDSL architecture, and the sequenced implementation steps).

## The loop everyone shares

Strip away the framework differences and every autotuner is the same four-step loop, wrapped in a cache. A **search space** of candidate configs is defined; each candidate is **compiled and benchmarked**; the fastest is **selected**; the winner is **cached** under a key so the next call with the same shape skips the search. The interesting engineering is all in the edges: what the key contains, how you avoid the search when you can, how you keep benchmarking from corrupting state, and whether the loop runs online (at first call) or offline (ahead of time).

## The one rule: in-place kernels corrupt the search

Autotuning runs the *same kernel* dozens of times to pick a winner. If the kernel writes its output in place or accumulates, those repeated runs corrupt the data and you measure garbage — and the "fastest" config you select may simply be the one that diverged fastest. The selection is wrong in a way no unit test on the kernel itself will catch.

Triton's answer is two hooks on the decorator: `reset_to_zero` (zero the named tensors before each rep) and `restore_value` (snapshot the named tensors, restore them after each rep). Both run around every benchmarked call, so each config is measured against the same clean inputs. This is the correctness soul of autotune, and it is the single most important thing a naive port forgets.

## Triton — the template

Triton's `@triton.autotune` ([runtime/autotuner.py](https://github.com/triton-lang/triton/blob/main/python/triton/runtime/autotuner.py), 511 lines) is the design quack and FlyDSL both descend from. Three things in it separate a real autotuner from a toy, and all three are worth stealing:

- **`@heuristics` — the skip-the-search door.** If a parameter can be *computed* from the inputs (a block size that should just be the next power of two above `N`), derive it instead of enumerating it. The cheapest search is the one you never run.
- **`prune_configs_by` — cut before you compile.** A `perf_model` + `top_k` and an `early_config_prune` run *before* any compilation, so you only pay to benchmark candidates that have a chance.
- **The cache key.** Triton's disk-cache key is a SHA-256 of five things — `triton_key()` (compiler version), the backend target hash, `fn.cache_key` (kernel source), cache-invalidating env vars, and shape/dtype. Each axis answers "when should I throw the cached config away?"

## quack — two-track configs

[quack](https://github.com/Dao-AILab/quack) is Tri Dao's kernel library on CuteDSL. Its autotuner ([quack/autotuner.py](https://github.com/Dao-AILab/quack/blob/main/quack/autotuner.py), 849 lines) is the most production-hardened reference, and unlike FlyDSL it is actually adopted — `rmsnorm` and `gemm` both wear `@autotune`. It adds three things:

- **Two-track configs.** Each kernel's `*_config.py` exposes both an analytical `get_default(N, dtype, arch)` (zero-search heuristic) and `get_all_fwd_configs()` (the exhaustive space). Day-to-day takes the heuristic; you only pay for the search when you explicitly autotune. This is the cure for an all-or-nothing autotuner.
- **Multi-layer cache with a source fingerprint.** In-memory dict → on-disk `.o` → a SHA-256 of every `quack/*.py` plus CUTLASS/Python versions (any source change invalidates) → an optional result cache. The tuning key also normalizes stride to `{0, 1, other}`.
- **Parallel precompile.** `_precompile` compiles all candidates in a subprocess pool to warm the cache before benchmarking — the practical answer to the slow-first-call problem.

## CuteDSL — autotune_jit

CuteDSL is the layer quack builds on. Its `autotune_jit(params_dict, update_on_change)` takes the parameter grid directly, sweeps the Cartesian product on first call, and caches the best kernel per tuning key. Two details transfer to FlyDSL: compilation is the cost (so `cute.compile()` once per config and reuse the executor — exactly FlyDSL's `build_*_module` situation), and the implicit JIT cache keys on a hash of the source *plus all DSL environment variables* — the same "env is part of the key" lesson Triton encodes.

## aiter and SGLang — offline, and the consumer

[aiter](https://github.com/ROCm/aiter) is the opposite paradigm: tuning is **offline and checked in**. A tuner reads an untuned shape list (`*_untuned_gemm.csv`), sweeps configs, and commits the winners to `*_tuned_gemm.csv`; serving reads from the CSV with no search, and a CI guard keeps the configs from regressing. Notably aiter already ships a FlyDSL-backed tuning path at `aiter/ops/flydsl/gemm_tune/`.

[SGLang](https://github.com/sgl-project/sglang) is not a kernel DSL — it is an inference engine that *consumes* tuned configs, and it shows what the offline artifact looks like at scale. Its fused-MoE configs are committed JSON whose *filename* is the lookup key (`E=8,N=14336,device_name=...,dtype=...json`), under a triton-version directory, with AMD `gfx950` variants alongside the NVIDIA ones. At runtime `get_moe_configs` looks up by device + shape and falls back to `get_default_config` on a miss. If FlyDSL kernels are to be tuned-and-consumed by an engine, this is the contract the offline artifact should target.

One nuance: training and inference tolerate first-run cost differently. A training run amortizes a one-time startup autotune over millions of steps, so JIT tuning is fine. Inference serving is latency-sensitive, so a slow first request per new shape is a visible regression — which is why engines lean on committed offline configs with a default fallback.

## Designing FlyDSL's autotune

FlyDSL already has an autotuner — [python/flydsl/autotune.py](https://github.com/ROCm/FlyDSL/blob/main/python/flydsl/autotune.py), 296 lines: a `Config`, an `Autotuner`, a disk cache, a `_make_key`, a `_prune`, and a public `@autotune`. The problem is not that it is missing; it is that **no kernel uses it** — well-built dead code. Reading it against the five references, the gap is clear:

| Capability | FlyDSL today | Target |
|---|---|---|
| Adoption | none | 1 kernel + offline config |
| Avoid-search path | none | heuristic default |
| In-place correctness | `reset_to_zero` only | + `restore_value` snapshot |
| Cache key | shape/dtype only | + arch + src fingerprint + compiler + env + stride |
| Offline artifact | none | device+shape config file |
| CI guard | none | GPU-free unit + regression gate |

The target architecture is **three tracks off one switch**: a heuristic default for zero-cost normal runs, an offline device+shape lookup for serving, and an online search (with `restore_value`) for dev and the long tail — all writing one cache keyed by the full key above, with the online search able to emit the offline config file.

## Implementation guide — sequenced

1. **Harden the key, add `restore_value`.** Before any kernel adopts autotune, fix the two correctness gaps: extend `_make_key` (currently shape/dtype only) with arch, a source fingerprint, the compiler/version, and env; and add `restore_value` alongside the existing `reset_to_zero`. Without this, autotuning a fused-add rmsnorm is incorrect.
2. **Two-track config for one kernel.** Pick softmax or rmsnorm (small bounded space). Give it `get_default(N, dtype, arch)` and `get_all_configs()`, quack-style. Normal callers get the heuristic for free; autotune is opt-in.
3. **Offline emit + lookup (SGLang-style).** Let the online search write a config file keyed by device + shape + dtype; a serving run with tuning off looks it up and falls back to the default on a miss.
4. **CI guard, GPU-free first.** Unit tests for `Config` serialization, cache-key construction, and pruning (no hardware needed), then an opt-in GPU integration test, then a tuned-config regression gate like aiter's.
5. **(optional) parallel precompile** if the online search proves too slow.

If you read this for one decision: do not start FlyDSL's autotune by adding configs. Start by making it *correct* and *cheap to not use* — the full cache key and `restore_value`, and a heuristic default. FlyDSL has the loop already; what it needs is the edges. These steps map onto FlyDSL issues [#770](https://github.com/ROCm/FlyDSL/issues/770) (autotune), [#769](https://github.com/ROCm/FlyDSL/issues/769) (rmsnorm bwd, which also needs tuning), and [#749](https://github.com/ROCm/FlyDSL/issues/749) (downstream consumption).
