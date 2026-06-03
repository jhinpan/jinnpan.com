---
title: "Source Reading 007 — AITER MoE Tuner, the Dispatch Board Behind Fused MoE"
description: "A reading of AITER's gemm_moe_tune.py: how one 4,259-line script turns MoE shapes into ASM, CK, CK-Tile, and FlyDSL candidate tasks, then writes production fused_moe configs."
date: 2026-06-03
tags: ["source-reading", "MLSys", "AMD", "ROCm", "MoE", "kernel-optimization"]
category: "Technical"
lang: "en"
---

Source Reading 007 is about AITER, but not the whole repo. The target is one dense file: `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py`. It is 4,259 lines long, and it is easy to mistake it for a benchmark script. It is not. It is closer to a dispatch-board compiler: it reads MoE shapes, expands them into candidate kernel tasks, runs those tasks through a shared multiprocessing executor, validates results against references, accounts for missing end-to-end costs, and finally writes rows that production `fused_moe` can consume.

Full HTML deep dive: [/sources/aiter.html](/sources/aiter.html).

## Why this file matters

MoE tuning is not just "pick the fastest GEMM." The production path has multiple hidden contracts:

- routing and sorting contract: top-k token-expert pairs must be grouped into expert-local blocks;
- quantization contract: per-token fp8, per-1x128 blockscale, per-1x32 MXFP4, and int4 paths all carry different scale layouts;
- inter-stage contract: stage1 may emit bf16, fp8, or fp4 activations depending on whether quant/cast is fused;
- backend contract: ASM, CK, CK-Tile, and FlyDSL do not support the same dtype and activation matrix;
- dispatch contract: the final `tuned_fmoe.csv` row is read by production `fused_moe`, not merely by the tuner.

The script exists because those contracts cannot be trusted implicitly. It makes them explicit enough to test.

## The main abstraction: task tuple

The load-bearing abstraction is the task tuple passed into `mp_tuner`. Every backend is normalized into roughly the same form:

```text
(tag,
 generate_data, generate_args,
 candidate_func, candidate_args,
 reference_func, reference_args,
 tolerances, optional_compare_fn)
```

This is why a single executor can run CK stage1, CK stage2, CK-Tile A8W4, FlyDSL FP4, FlyDSL int4, and ASM 1-stage candidates. Each candidate says how to generate data, what function to run, which keys to pull from the generated dictionary, and what reference should define correctness.

After that, the tuner becomes a table problem: collect `(info, us, err)` rows, group by shape, reject invalid candidates, combine stage1 and stage2 where `block_m` matches, append 1-stage candidates, add fairness costs, then choose the row with the lowest total time.

## The thing I would fix first

There is a concrete suspicious line in `calculate()`. It unpacks `stage`, then resets it to an empty string before the stage-specific FLOP/BW branches:

```python
key, stage, kernelName, block_m, us, err = results
...
stage = ""
if stage == "stage1":
    ...
elif stage == "stage2":
    ...
```

That means stage1/stage2 TFLOPS and bandwidth reporting fall through to the combined estimate. It does not change which kernel wins because selection is by `us`, but it does make per-stage derived metrics misleading. Until that is fixed, I would trust timing, correctness, and final config fields before trusting TFLOPS/BW.

## Short mental model

Read `gemm_moe_tune.py` as a compiler for tuning experiments:

- input language: `untuned_fmoe.csv`;
- IR: task tuples;
- runtime: `mp_tuner`;
- optimizer: `post_process()`;
- output artifact: `tuned_fmoe.csv`.

Once you use that model, the long file becomes navigable. You stop reading 4,259 lines as one script and instead read five contracts: shape, quantization, task, fairness, and production dispatch.

**→ Full deep dive: [/sources/aiter.html](/sources/aiter.html)**.
