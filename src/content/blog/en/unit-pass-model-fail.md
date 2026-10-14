---
title: "Green Test, Red Model: Why GPU Kernels Pass Unit Tests and Fail End to End"
description: "Why a GPU kernel can pass every unit test and still break an end-to-end model run: seven gaps between a test and a model, traced through public bugs from the gfx1250 bring-up and from FlyDSL v0.3 to v0.4, and a validation ladder that closes them."
date: 2026-10-13
tags: ["GPU-kernels", "validation", "testing", "inference", "gfx1250", "FlyDSL", "primer"]
category: "Technical"
lang: "en"
---

A kernel pull request usually arrives with two pieces of evidence: a green unit test and a speedup table. Both are real. Then the kernel ships inside a model, and one of three things happens. The model answers in garbage: on gfx1250, one server loaded, served, and printed "!" for every token, scoring 0.0 on GSM8K with no error anywhere in its log ([ATOM #1750](https://github.com/ROCm/ATOM/pull/1750)). A multi-GPU run hangs, or finishes and returns stale data. Or nothing happens for a week, until one request takes a path nobody tried. People who maintain kernel libraries say it in one line: *it passed the unit test and failed in the model.*

This is not one bug. It is a gap in what the test covers. A unit test checks a kernel at a handful of points: a few shapes, random inputs, one GPU, one stream, a fresh process. A model run is a distribution over a much larger space. The test proves correctness only at the points it checks, and a model spends almost all of its time somewhere else.

The cases here come from the place where the gap is widest: a new architecture. Every case dates from June to October 2026, while kernels for gfx1250, the architecture of AMD's MI450 series, were brought up across the kernel libraries and FlyDSL went from its [v0.3 roadmap](https://github.com/ROCm/FlyDSL/issues/632), which named "full gfx1250 support, validated end-to-end" as a focus, to the [v0.4.0 release](https://github.com/ROCm/FlyDSL/releases/tag/v0.4.0) on 14 October.

> Full read: [/sources/unit-pass-model-fail.html](/sources/unit-pass-model-fail.html) is a bilingual HTML version with seven hand-drawn plates and a filterable index of 32 cases. Updated 2026-10-14: the cases now come from the gfx1250 bring-up and from FlyDSL v0.3 to v0.4.

## The seven dimensions

Start from what must be true of any kernel. Its output depends on more than the values of its arguments:

1. **Shape**: which code paths and tile tails run.
2. **Values**: which numeric regimes it hits.
3. **Layout, dtype and hardware contract**: what the producer wrote, and what the machine guarantees.
4. **Execution context**: graph capture, streams, caches, buffers, process lifetime.
5. **Timing**: which wave arrives first.
6. **Scale**: how many GPUs take part, and in which partition mode.
7. **Software and hardware**: the toolchain, the build, the version, and the copy that actually runs.

A unit test is cheap because it pins most of them to convenient values: small tile-aligned shapes, `torch.randn` inputs, layouts produced by its own helper, eager mode on the default stream, one call, one GPU, the repository's own build. That is the right trade-off for a fast test. The cost is that each pinned value hides a regime where the kernel's behavior changes.

## Why gfx1250 opens every gap at once

A kernel encodes facts about the machine it was written for: how many lanes a wave has, which matrix shapes exist, how data moves into LDS, what a barrier waits for, what a cache bit means. It also encodes facts about its toolchain and its fleet. When the target changes, each of these becomes a place where code compiles, passes on the old hardware, and is wrong on the new one. gfx1250 changed most of them at once:

| Change | What the ported code assumed | What gfx1250 does | Source |
|---|---|---|---|
| Wave size | not RDNA, so 64 lanes | 32 lanes, although it is not RDNA | [FlyDSL #1024](https://github.com/ROCm/FlyDSL/pull/1024), [SGLang #41687](https://github.com/sgl-project/sglang/pull/41687) |
| Matrix tile | any tile the launcher accepts issues MMA | FP8 WMMA is 128 deep; a 64-deep tile issues none | [FlyDSL #1264](https://github.com/ROCm/FlyDSL/pull/1264) |
| Copy into LDS | the buffer-to-LDS load lowers everywhere | LLVM cannot lower it; TDM and async loads instead | [aiter #4987](https://github.com/ROCm/aiter/issues/4987) |
| Barrier | a plain barrier publishes LDS stores | `s_barrier_signal` with no `s_wait_dscnt` before it | [aiter #6344](https://github.com/ROCm/aiter/issues/6344) |
| Cache bits | the value 17 (sc0 sc1) keeps its meaning | 17 means device scope plus non-temporal | [FlyDSL #1214](https://github.com/ROCm/FlyDSL/pull/1214) |
| Compiler | nothing is added at the kernel entry | LLVM prepends an entry prologue | [FlyDSL #1088](https://github.com/ROCm/FlyDSL/pull/1088) |
| Async copy | a masked load keeps its bounds check | Triton 3.8 lowers it to an unchecked async load | [aiter #6081](https://github.com/ROCm/aiter/pull/6081), [SGLang #41868](https://github.com/sgl-project/sglang/pull/41868) |
| Partitions | one GPU is one device | NPS2 / DPX: peers need system-scope ordering | [aiter #4812](https://github.com/ROCm/aiter/pull/4812) |
| CI | CI runs the kernel on every PR | default runners are gfx942 and gfx950 | [FlyDSL #1199](https://github.com/ROCm/FlyDSL/pull/1199), [#1264](https://github.com/ROCm/FlyDSL/pull/1264) |

Two structural changes widen the gaps further. aiter's default PR CI runs its op tests on gfx942 and gfx950, and FlyDSL's gfx1250 PRs say "Tested on gfx1250 (no gfx1250 CI runner)", so correctness rests on each author's own machine. And machines differ: in [FlyDSL #1228](https://github.com/ROCm/FlyDSL/pull/1228), two gfx1250 machines with the same pinned LLVM passed the same correctness tests, while one split-K fence cost about 0.03 µs per workgroup on one and about 0.3 µs on the other.

## Gap 1: shapes and sizes

A kernel computes full tiles fast and handles the tail with masks or clamps. It computes addresses in 32-bit arithmetic, because 64-bit address math costs registers on every load. Both choices are correct inside a range and fail silently outside it.

The 32-bit choice has two cliffs: 2³¹ for a signed index and 2³² bytes for a byte offset. In [aiter #6402](https://github.com/ROCm/aiter/issues/6402), the gfx1250 BF16 GEMM that the default config selects is right at 7067 × 151936 (2,147,463,424 bytes, just under 2³¹). One row more, and every row from 7067 on is never written; 8192 × 151936 returned 332,463 non-finite values and no error. The FlyDSL kernel in the same issue fits its output under 2³², but the 202 padding rows of its last 256-row tile cross 2³², wrap, and overwrite rows 0 to 201. A cliff can also wait for the process to age: in [ATOM #2533](https://github.com/ROCm/ATOM/pull/2533), int32 KV row ids times the stride 576 wrap from row 3,728,270, which only a pool filled by prefix caching reaches. Decodes on every Kimi-K3 rank then answered "!!!!".

Tails fail the same quiet way. In [FlyDSL #1199](https://github.com/ROCm/FlyDSL/pull/1199), `K // tile_k` dropped the last 128 columns at K = 640, so the output was 60% off the full-K reference. [FlyDSL #1230](https://github.com/ROCm/FlyDSL/pull/1230) found the same floor at gpt-oss K = 2880, where only 64-wide K tiles were correct. In [aiter #6344](https://github.com/ROCm/aiter/issues/6344), the gfx1250 RMSNorm never wrote the last N % 8 elements of a row. In [FlyDSL #1264](https://github.com/ROCm/FlyDSL/pull/1264), a tile_k of 64, below the 128-deep FP8 WMMA, issued no WMMA and returned an all-zero C. Untuned shapes take fallback configs: in [aiter #4987](https://github.com/ROCm/aiter/issues/4987), DeepSeek-V4-Flash on gfx1250 hit a heuristic FlyDSL fallback that LLVM could not lower, and the server aborted.

**Test that catches it:** outputs just below and above 2³¹ and 2³² bytes, counting the last tile's padding; K and N that the tile does not divide, at real widths such as K = 2880; every tile the launcher accepts; and every production shape without a tuning entry.

## Gap 2: real data

Real prompts are short and long, router scores nearly tie, trained attention is peaked, and biases are not zero. In [ATOM #1750](https://github.com/ROCm/ATOM/pull/1750), prompts shorter than one KDA chunk went NaN on gfx1250, NaN logits made argmax pick token 0, and the server answered "!" forever with GSM8K 0.0. In [aiter #5801](https://github.com/ROCm/aiter/pull/5801), the gfx1250 TDM grouped GEMM left the N-tile index out of its bias pointer; fixing it restored gpt-oss-120b accuracy, and the review asked for a test with several N-tiles and distinct biases, because a uniform bias hides it.

The FlyDSL cases from v0.3 to v0.4 follow the same pattern. [FlyDSL #1213](https://github.com/ROCm/FlyDSL/pull/1213) found, with real GLM checkpoints, that packing expert IDs into router scores picked expert 92 where 226 was right. [FlyDSL #1020](https://github.com/ROCm/FlyDSL/pull/1020) found FP8 softmax underflow that returned 0.637 instead of 1.0, which "a random-init model will not" show. In [aiter #6127](https://github.com/ROCm/aiter/pull/6127), identical warmup tokens skewed FlyDSL MegaMoE routing until a lagging wave read a reused LDS slot and faulted.

**Test that catches it:** replay captured prompts, routing, biases and weights; cover both sides of every chunk and top-k length; add ties, near ties, one-hot routing, distinct biases and NaN on purpose.

## Gap 3: contracts

A kernel is correct relative to a contract, and on a new architecture the hardware is part of it. FlyDSL derived the wave size from an "is RDNA" check, so gfx1250 was compiled as wave64 until [FlyDSL #1024](https://github.com/ROCm/FlyDSL/pull/1024), shipped in v0.3.2. Downstream copies carried their own constant: SGLang's FlyDSL norm kernel failed instruction selection ([SGLang #37751](https://github.com/sgl-project/sglang/pull/37751)), and the DSA top-k in [SGLang #41687](https://github.com/sgl-project/sglang/pull/41687) returned wrong indices only for rows longer than top-k. [FlyDSL #1214](https://github.com/ROCm/FlyDSL/pull/1214) notes that the gfx950 cache-policy value 17 means device scope plus non-temporal on gfx1250.

Between kernels, the contract is a layout. In [aiter #5465](https://github.com/ROCm/aiter/pull/5465), the quant kernel and the gfx1250 grouped-MoE GEMM derived the scale layout separately and disagreed on the expert-parallel path; the rule is one layout, "chosen by the producer and reported to the consumer". In [aiter #6360](https://github.com/ROCm/aiter/pull/6360), a pandas merge wrote integers as floats, so 878 cells failed to parse and EP rows never matched, while runs that read the CSV files one at a time worked. In [FlyDSL #1253](https://github.com/ROCm/FlyDSL/pull/1253), the test quantizers rounded E8M0 scales to nearest, so the oracle and the kernel disagreed on 58% to 98.5% of blocks.

**Test that catches it:** feed each kernel from its real producer, assert the compiled wave size at the kernel entry, pass the layout from the producer, parse the merged config, and quantize the reference with the kernel's own rule.

## Gap 4: execution context

Graph capture freezes host decisions: in [aiter #4494](https://github.com/ROCm/aiter/pull/4494), a split-K semaphore zeroed outside the graph made Kimi-K3 hang at warmup, and "No test crossed split-K with capture." Production reuses buffers: the unwritten RMSNorm tails in aiter #6344 came back as garbage up to 1e38 through `torch.empty_like`, so FlyDSL #1264 now fills the memory past N with a signaling-NaN pattern and checks it. Caches must key on everything that changes code: the FlyDSL JIT key ignored a MegaMoE spec object (aiter #6127), the disk cache ignored the LLVM version ([FlyDSL #1088](https://github.com/ROCm/FlyDSL/pull/1088)), and a warm cache gave 22 failures against 2 on a cold one ([FlyDSL #1277](https://github.com/ROCm/FlyDSL/pull/1277)). In [FlyDSL #1220](https://github.com/ROCm/FlyDSL/pull/1220), a test that filled inputs on one stream and launched on another failed about once in 100,000 launches.

**Test that catches it:** capture and replay, keep the producer on the launch stream, run several shapes and a warm cache in one process, pre-fill outputs with a sentinel, and fill state the way a long-running server does.

## Gap 5: rare races

On gfx1250, waits are split by counter, and a plain barrier does not wait for LDS stores. In [aiter #6344](https://github.com/ROCm/aiter/issues/6344), each LDS store of the RMSNorm reduction was followed by `s_barrier_signal` with no `s_wait_dscnt`, so wide rows were scaled by about 1.156 or 1.41: 0 of 30 runs at 4096 × 16384, 2 of 30 at 4096 × 32768, 13 of 20 at 8192 × 65536. The fences in [aiter #6348](https://github.com/ROCm/aiter/pull/6348) make the compiler emit the wait. The compiler can also misplace waits: in [aiter #5418](https://github.com/ROCm/aiter/issues/5418), a wait one instruction too late clobbered an in-flight async load, and in [SGLang #41868](https://github.com/sgl-project/sglang/pull/41868), "Answers look fluent but are wrong." Hand-placed waits in FlyDSL kernels broke under the compiler's schedule in [aiter #6170](https://github.com/ROCm/aiter/pull/6170) and [#6368](https://github.com/ROCm/aiter/pull/6368), where "a single-run test passes".

The arithmetic: with per-call failure probability p, N calls fail at least once with probability 1 − (1 − p)ᴺ. At 2 in 30, a few dozen calls are enough. At 1 in 100,000, even odds need about 70,000 calls. At the 7 in 292,889,600 of [aiter #4841](https://github.com/ROCm/aiter/pull/4841), a 1,000-call test catches it 0.002% of the time. That is why [FlyDSL #1237](https://github.com/ROCm/FlyDSL/issues/1237) matters: its prototype LDS checker reported 12 of 12 injected sync bugs, while one launch's output check missed 8 of them.

**Test that catches it:** repeat with identical inputs and require bit-identical output, sweep the widest rows, run a synchronization sanitizer, check wait counts against the ISA, and bound every wait.

## Gap 6: multi-GPU

Multi-GPU protocols fail as hangs or stale reads. In [aiter #4812](https://github.com/ROCm/aiter/pull/4812), a relaxed ready flag on gfx1250 let a peer read input before the copy was visible system-wide: 127 of 128 outputs were wrong, and "The collective completes normally". Only the registered-buffer path above 4 MiB uses that flag, and only back-to-back calls with changing inputs show it. In [aiter #6446](https://github.com/ROCm/aiter/pull/6446), a size-derived slot stride hung vLLM CI on gfx950 (6 of 6 with a lagging rank, 0 of 114 on a quiet host); a review of the gfx1250 LL128 kernel in [aiter #5616](https://github.com/ROCm/aiter/pull/5616) found the same rule and warned that "Two sizes always land in the same bank and can miss the bug." In [FlyDSL #1214](https://github.com/ROCm/FlyDSL/pull/1214), the gfx1250 TP > 1 path "compiles but has never run".

**Test that catches it:** TP 2, TP 4 and EP in each partition mode, three message sizes per round, many queued collectives before one sync, bounded waits and a watchdog.

## Gap 7: the software you didn't test

The compiler adds code you never wrote: an LLVM pin bump built and was approved ([FlyDSL #1051](https://github.com/ROCm/FlyDSL/pull/1051)), and [FlyDSL #1088](https://github.com/ROCm/FlyDSL/pull/1088) traces the reported fault of every gfx1250 kernel to an entry prologue that upstream reworked three times in three weeks. Flags can fail to reach the code: FlyDSL v0.4.0 fixed four LLVM knobs that never reached codegen ([#1153](https://github.com/ROCm/FlyDSL/pull/1153)), a DAZ flag that never reached the hardware ([#1177](https://github.com/ROCm/FlyDSL/pull/1177)), and compile hints that reached only a discarded code object ([#1195](https://github.com/ROCm/FlyDSL/pull/1195)).

Production also runs someone else's copy. After FlyDSL v0.3.0 moved `buffer_ops`, SGLang's FlyDSL norm kernel silently fell back to a slow path, with "Nothing is logged" ([SGLang #33142](https://github.com/sgl-project/sglang/issues/33142)), and aiter's attention import raised on every architecture with FlyDSL 0.3.1 ([aiter #4701](https://github.com/ROCm/aiter/issues/4701)). A review of [aiter #6348](https://github.com/ROCm/aiter/pull/6348) found dispatch keyed on the build target instead of the live device. [aiter #5133](https://github.com/ROCm/aiter/issues/5133) found 141 pytest functions that never executed, 8 of them in a gfx1250 FlyDSL test, while CI recorded a pass.

**Test that catches it:** test the engine's image and pins on gfx1250, dispatch on the live device, put the compiler version in every cache key, require a flag to change the code object, add contract tests for downstream imports, and fail when nothing is collected.

## A validation ladder

Each rung catches a class of bugs the rung below cannot, and costs more:

| Rung | Closes | What to run |
|---|---|---|
| 1. Boundary unit tests | shapes | tails, 2³¹ / 2³² with the last tile, every accepted tile, untuned shapes, sentinels past the output |
| 2. Replay real inputs | data | captured prompts, routing and biases, short and long rows, ties, NaN |
| 3. Test the seams | contracts | real producer → consumer, wave size asserted, merged config, kernel-rule reference |
| 4. Production modes | context | graph replay, side streams, warm cache, reused buffers, full pools |
| 5. Repeat for determinism | timing | N calls bit-identical, wide rows, sanitizer, bounded waits |
| 6. Multi-GPU protocols | scale | TP 2 / TP 4 / EP, each partition mode, three sizes per round, watchdog |
| 7. Downstream and toolchain | software | gfx1250, the engine's pins, live-device dispatch, flags that change the code |
| 8. End-to-end model | all seven | fixed-seed accuracy (e.g. GSM8K) on the target arch, throughput; read the score, not the exit code |

The model at the top exercises all seven dimensions at once, which is why it is the last rung and not the only one. On gfx1250, also record the machine, the driver and the pins: a result on one machine is weaker evidence for the next one than it looks.

## Closing

"It passed the unit test and failed in the model" sounds like bad luck. It is not. A unit test is a sample; a model is the population. Every public bug above lived in a dimension the test held still: a row past byte 2³¹, a prompt shorter than one chunk, a wave size taken from the wrong family, a barrier that did not wait for its stores, a prologue the compiler added. A new architecture moves many of those dimensions at once. Name the dimension, and the missing test is usually obvious.
