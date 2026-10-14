---
title: "Green Test, Red Model: Why GPU Kernels Pass Unit Tests and Fail End to End"
description: "Why a GPU kernel can pass every unit test and still break an end-to-end model run: seven gaps between a test and a model, each traced through public bugs, and a validation ladder that closes them."
date: 2026-10-13
tags: ["GPU-kernels", "validation", "testing", "inference", "MLSys", "primer"]
category: "Technical"
lang: "en"
---

A kernel pull request usually arrives with two pieces of evidence: a green unit test and a speedup table. Both are real. Then the kernel ships inside a model, and one of three things happens. The model starts to answer in garbage. A multi-GPU run hangs with no error. Or nothing happens for a week, until one request takes a path nobody tried. People who maintain kernel libraries say it in one line: *it passed the unit test and failed in the model.*

This is not one bug. It is a gap in what the test covers. A unit test checks a kernel at a handful of points: a few shapes, random inputs, one GPU, one stream, a fresh process. A model run is a distribution over a much larger space. The test proves correctness only at the points it checks, and a model spends almost all of its time somewhere else.

> Full read: [/sources/unit-pass-model-fail.html](/sources/unit-pass-model-fail.html) is a bilingual HTML version with six hand-drawn plates: the seven gaps, the 32-bit cliffs, the contract chain, the probability of a rare race, a multi-rank slot overwrite, and the validation ladder.

## The seven dimensions

Start from what must be true of any kernel. Its output depends on more than the values of its arguments:

1. **Shape**: which code paths and tile tails run.
2. **Values**: which numeric regimes it hits.
3. **Layout and dtype**: the contract with whoever produced the input.
4. **Execution context**: graph capture, streams, caches, process lifetime.
5. **Timing**: which wave arrives first.
6. **Scale**: how many GPUs take part.
7. **Software and hardware**: the build, the version, and the copy that actually runs.

A unit test is cheap because it pins most of them to convenient values: small tile-aligned shapes, `torch.randn` inputs, layouts produced by its own helper, eager mode on the default stream, one call, one GPU, the repository's own build. That is the right trade-off for a fast test. The cost is that each pinned value hides a regime where the kernel's behavior changes.

Why not test everything? Because the space is a product: ten shapes × five value distributions × three layouts × four execution modes × three GPU counts is already 1,800 configurations, and timing is not a parameter you can set. The goal is not exhaustive testing. It is to place test points where behavior changes.

## Gap 1: shapes and sizes

A kernel is written for the common case and patched for the rest. It computes full tiles fast and handles the tail with masks or clamps. It computes addresses in 32-bit arithmetic, because 64-bit address math costs registers and instructions on every load. Both choices are correct inside a range and fail silently outside it.

The 32-bit choice has two cliffs: a signed 32-bit index wraps at 2³¹, and a 32-bit byte offset from a buffer's base wraps at 2³² bytes (4 GiB). In [FlyDSL #879](https://github.com/ROCm/FlyDSL/pull/879), the MoE stage-1 weight tensor of Kimi-K2.6 at TP 1 is 385 × (2·2048) × 7168 × 0.5 bytes, about 5.3 GiB. The kernel addresses it through one buffer resource with a 32-bit offset, so experts past 4 GiB read another expert's weights and the model prints garbage. At TP 2 the tensor is split to about 2.6 GiB, below the cliff, which is exactly why the bug only appears at TP 1.

The quiet version is worse. In [FlyDSL #1236](https://github.com/ROCm/FlyDSL/pull/1236), a (4, 2²⁹) tensor has 2³¹ elements; the total size, computed in int32, wraps negative, every lane writes to a wrong position, and because those positions are still in bounds, nothing reports an error. [FlyDSL #1020](https://github.com/ROCm/FlyDSL/pull/1020) failed with `struct.error` once a flattened FP8 attention shape reached 2³¹ elements, which at head-dim 128 and 64 heads is any sequence of 131,072 tokens or more. And a refactor in [FlyDSL #1131](https://github.com/ROCm/FlyDSL/pull/1131) passed its targeted tests, then failed the full CI on two FP16 tiles at M = 16 and M = 33, and was reverted in [#1189](https://github.com/ROCm/FlyDSL/pull/1189) the next day.

Every tuned kernel library also has a structural version of this gap: a shape with no entry in the tuning table falls back to a default or heuristic config. Tests usually run tuned shapes, so the fallback is often the least-tested code that production runs.

**Test that catches it:** tensors just below and above 2³¹ elements and 2³² bytes; the largest unsharded deployment, not the sharded one; tails in every dimension; and an explicit list of production shapes with no tuning entry.

## Gap 2: real data

Real activations and weights are not random. Router scores cluster and nearly tie. Trained attention is peaked. Warmup traffic can be identical tokens. Code that is exact for random inputs can fail on each of these.

[FlyDSL #1213](https://github.com/ROCm/FlyDSL/pull/1213) is the cleanest example. The MonoKernel from [#1204](https://github.com/ROCm/FlyDSL/pull/1204) passed its tests and was merged; one day later, validation with real GLM checkpoints in the ATOM engine found that packing expert IDs into router scores could pick the wrong expert when scores were close. A captured TP8 routing replay picked expert 92 where 226 was correct. With random scores, near-ties almost never happen.

[FlyDSL #1020](https://github.com/ROCm/FlyDSL/pull/1020) shows the numeric side. An FP8 attention kernel cast softmax probabilities to e4m3, which bottoms out at 2⁻⁹, so most probabilities underflowed while the normalizer still counted them. With V set to all ones, the output was 0.637 instead of 1.0. The PR says only peaked attention shows this, and a random-init model will not.

**Test that catches it:** replay captured activations, router scores and weights from a real model run, and add adversarial values on purpose: ties, near-ties, one-hot distributions, all tokens on one expert.

## Gap 3: contracts

A kernel is correct relative to a contract: layout, dtype, strides, index width, which rows to skip. A unit test hides contracts by construction, because its own helper writes the input and its own reference reads it. In the model, the producer is a checkpoint loader, a previous kernel, or someone else's framework.

#1213 again: real checkpoints used a weight-scale layout the kernel did not support; the reference followed the caller's default dtype, so a BF16 default changed the reference itself; and short contexts let a fused indexer select beyond the available context. In [FlyDSL #1234](https://github.com/ROCm/FlyDSL/pull/1234), an automated review noted that index buffers were read as raw int32 while PyTorch defaults to int64 (a review finding, not an observed failure).

**Test that catches it:** feed each kernel from its real producer, and validate dtype, shape, stride and layout at the kernel entry.

## Gap 4: execution context

Engines record kernels into CUDA or HIP graphs, use side streams, and keep one process alive for days. In [aiter #4494](https://github.com/ROCm/aiter/pull/4494), a split-K GEMM's semaphore workspace had the wrong owner under graph capture, and the replay spun forever: "the device wedges; it does not return a wrong number and it does not recover." Kimi-K3 at TP 8 hung during warmup. [FlyDSL #171](https://github.com/ROCm/FlyDSL/pull/171) fixed a capture segfault on PyTorch 2.9 that did not occur on 2.8. [FlyDSL issue #317](https://github.com/ROCm/FlyDSL/issues/317) and [aiter #6127](https://github.com/ROCm/aiter/pull/6127) both found JIT cache keys that let one configuration reuse another's compiled kernel.

**Test that catches it:** capture and replay inside the test, use a non-default stream, and call several shapes in one process before checking the last one.

## Gap 5: rare races

A race depends on which wave arrives first, so a test samples a probability. In [aiter #4841](https://github.com/ROCm/aiter/pull/4841), a candidate fix for a cross-block barrier race still failed 7 times in 292,889,600 row-launches, about once in 42 million; the PR says the race can wedge the GPU and stall a whole DP-attention service while its health endpoint keeps answering.

The arithmetic: with per-call failure probability p, N calls fail at least once with probability 1 − (1 − p)ᴺ. At p ≈ 2.4 × 10⁻⁸, a 1,000-call test catches it 0.002% of the time. A service at an assumed 10,000 calls per second passes 50% in about half an hour. In [aiter #6127](https://github.com/ROCm/aiter/pull/6127), identical warmup tokens skewed routing enough to expose an LDS slot-reuse race that failed within a few calls; random routing hid it. [FlyDSL #1235](https://github.com/ROCm/FlyDSL/pull/1235) notes plainly that "a test cannot force the wave order that triggers the race."

**Test that catches it:** call the same input many times and require bit-identical output, run a sanitizer where one exists, and stress with skewed inputs.

## Gap 6: multi-GPU

Multi-GPU protocols fail as hangs, not wrong numbers. In [aiter #6446](https://github.com/ROCm/aiter/pull/6446), an all-reduce derived its slot stride from the message size; when consecutive calls changed size, a fast rank overwrote a slow rank's unread slots, and the slow rank polled forever. It appeared in vLLM's CI as an RCCL watchdog timeout at TP 2. The repository's own multi-GPU CI ran the test without its size argument and never reached the TP 2 and TP 4 cases. Even the order of sizes matters: at TP 4, `[32768, 28800]` repeated never overlaps, while `[32768, 32768, 28800]` does. [FlyDSL #954](https://github.com/ROCm/FlyDSL/pull/954) fixed multi-rank dispatch hangs found in review and serving tests.

**Test that catches it:** TP 2, TP 4 and EP with message sizes that change between calls, under a watchdog.

## Gap 7: the software you didn't test

Production usually runs a downstream copy at a pinned version. [FlyDSL #1277](https://github.com/ROCm/FlyDSL/pull/1277) puts it directly: a change "could break a kernel vLLM runs in production while CI here stayed green," for example deleting a legacy allocator that only vLLM kernels still used. In [aiter #3683](https://github.com/ROCm/aiter/pull/3683), an import-time `KeyError` on an unsupported architecture took down every flash-attention import. [aiter #5133](https://github.com/ROCm/aiter/issues/5133) found 141 pytest functions that never executed while CI recorded a pass.

**Test that catches it:** test the engine's copy at its pinned version, add API contract tests for downstream imports, and make the runner fail when it collects nothing.

## Why a new architecture makes it worse

Every gap exists on mature hardware. On a new GPU architecture they widen at once: ported code carries old assumptions about wave size, bounds checks and barriers; compilers are young; tuning tables cover few shapes, so more traffic hits fallbacks; CI has few machines; and larger single-GPU memory lets unsharded tensors cross the 32-bit cliffs. No new kind of bug is needed, only more of the seven gaps open at the same time.

## A validation ladder

Each rung catches a class of bugs the rung below cannot, and costs more:

| Rung | Closes | What to run |
|---|---|---|
| 1. Boundary unit tests | shapes | tails, 2³¹ / 2³² sizes, untuned shapes, guard pages after tensors |
| 2. Replay real inputs | data | captured activations and scores, ties, skew |
| 3. Test the seams | contracts | real producer → consumer, checks at kernel entry |
| 4. Production modes | context | graph capture and replay, side streams, many shapes per process |
| 5. Repeat for determinism | timing | N calls bit-identical, sanitizer, skewed load |
| 6. Multi-GPU protocols | scale | TP 2 / TP 4 / EP, changing sizes, watchdog |
| 7. Downstream | software | the engine's copy and version, API contract tests |
| 8. End-to-end model | all seven | fixed-seed accuracy eval (e.g. GSM8K), throughput, single then multi-GPU |

The model at the top exercises all seven dimensions at once, which is why it is the last rung and not the only one. When it fails, the rung that should have caught the bug tells you which test to add.

## Closing

"It passed the unit test and failed in the model" sounds like bad luck. It is not. A unit test is a sample; a model is the population. Every public bug above lived in a dimension the test held still. Name the dimension, and the missing test is usually obvious.
