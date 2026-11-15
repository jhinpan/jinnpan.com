---
title: "Disaggregation Is Not a Kernel: PD Meets AMD Monokernels on GLM-5.3"
description: "Why prefill-decode disaggregation alone did not make GLM-5.3 decode faster on MI355X, what it took to run AMD monokernels on a MoRI decode server, and why two plain replicas still win on InferenceX AgentX."
date: 2026-11-15
tags: ["inference", "SGLang", "PD-disaggregation", "monokernel", "MI355X", "GLM-5.3", "AgentX"]
category: "Technical"
lang: "en"
---

TileRT decodes GLM-5.3 at 469 tokens per second per user on eight MI355X GPUs ([TileRT](https://www.tilert.ai/blog/tilert-amd-agentx.html)). It gets there with one request at a time, a persistent decode kernel, and prefill handed to a second server: one prefill node and one decode node, "1P1D". When we ran SGLang the same way, the first numbers came back about twice as slow per GPU as a single node. That looked like a verdict on disaggregation. It was not. It was a measurement of which kernel ran on the decode server.

This post is about putting AMD's monokernels behind PD disaggregation for GLM-5.3: what it took, what it bought, and why, on the chart we actually care about, it still did not beat two ordinary replicas. The tables, plates and exact commands are in [Experiment 007](/sources/glm53-pd-monokernel-mi355x.html).

## What PD changes, and what it does not

Start from what a user feels. Per-user decode speed is one over the time of a decode step. That time is set by the kernels that run the step and by how many requests share it. Prefill-decode disaggregation changes neither directly. It moves prefill to other GPUs, so a long prompt from someone else no longer lands between your decode steps. That protects the tail: fewer slow steps, steadier inter-token latency. It does not make a typical step faster.

The cost is GPUs. InferenceX's AgentX chart plots throughput per GPU and counts every GPU in the deployment. A 1P1D deployment of an eight-GPU model uses sixteen. Unless the prefill node does work that would otherwise slow decode down, PD halves tokens per GPU.

So when our PD decode server ran at 213 tok/s per user, half of TileRT's 469 and the same as plain SGLang without monokernels, the first question was not about disaggregation. It was: which kernel was decoding?

## Why the decode server ran the slow path

AMD's monokernels for GLM-5.3 (in [sgl-amd-mono](https://github.com/sgl-project/sgl-amd-mono)) fuse a decode layer into one launch for a narrow set of configurations. Narrow by design: a fused kernel makes assumptions about cache layouts, batch shapes and synchronization, so each model's spec declares the server features it was validated with, and SGLang refuses to start the monokernel when a feature in use is missing from that list. PD disaggregation is such a feature. DeepSeek-V4.1's spec declares it ([sgl-amd-mono#70](https://github.com/sgl-project/sgl-amd-mono/pull/70)); GLM-5.3's did not. Our PD decode server therefore ran SGLang's stock path, and its per-user speed showed exactly that.

The change, [sgl-amd-mono#136](https://github.com/sgl-project/sgl-amd-mono/pull/136), is small. It declares the feature on the target and NextN draft specs and refuses every PD role except a decode server that uses the MoRI transfer backend.

Why decode only? Prefill steps are long chunks, and the monokernel was never built for them. The prefill server keeps SGLang's path, and on this workload it is mostly idle anyway.

Why is it safe on decode? The kernels read the MLA KV cache and the indexer's key cache. SGLang's DSA memory pool registers both buffers with the transfer engine, so MoRI writes each new request's KV into exactly the memory the kernels read. The kernels themselves did not change.

Why MoRI only? Because that is where it was validated, over eight 400G RoCE links per node. Accepting another backend would be a claim without evidence.

Quality held. On all 1319 GSM8K questions, PD with the decode monokernel scored 0.9742 and single-node serving without monokernels 0.9750. Nine questions changed sides, five one way and four the other (McNemar p = 1.0).

## What the monokernel buys on a PD decode server

With the monokernel on, the PD decode server matches single-node decode at low concurrency: 474 tok/s per user at concurrency 1, up from 213, with the same 2.1 ms inter-token latency as one node.

| concurrency | tok/s/user P50, off → on | ITL P90 ms, off → on | tok/s/GPU, off → on |
|---|---|---|---|
| 1 | 213 → 474 | 4.99 → 2.32 | 1308 → 1448 |
| 2 | 205 → 467 | 5.84 → 3.11 | 1186 → 1373 |
| 4 | 208 → 466 | 6.09 → 3.96 | 1263 → 1706 |
| 10 | 148 → 162 | 8.25 → 7.84 | 4108 → 4231 |

The c10 row is the first hint of the real limit. The monokernel takes at most two requests per step, and at concurrency 10 most steps carry more.

Against one node with monokernels, PD now wins the tail from concurrency 2 up. At c4 its inter-token latency P90 is 3.96 ms against 7.14 ms, because nobody's prefill shares the decode GPUs. That is the benefit disaggregation promises, and it is real.

## The baseline that matters

But one node is the wrong baseline. PD spends sixteen GPUs, so the honest comparison is the same sixteen GPUs spent differently: two single-node servers behind SGLang's router with cache-aware routing, so each session's turns keep returning to the node that holds its prefix.

| concurrency | deployment | tok/s/user P50 / P90 | TTFT P50 | E2E latency P90 | tok/s/GPU |
|---|---|---|---|---|---|
| 4 | two nodes | 456 / 165 | 0.21 s | 9.7 s | 1876 |
| 4 | PD + monokernel | 466 / 252 | 0.24 s | 10.1 s | 1706 |
| 10 | two nodes | 216 / 130 | 0.23 s | 9.4 s | 4492 |
| 10 | PD + monokernel | 162 / 128 | 0.34 s | 13.2 s | 4231 |

PD keeps the tail at c4. At c10 the two replicas win on median speed, time to first token, end-to-end latency and tokens per GPU, and the tails are even.

Why? These are agentic coding traces: long sessions, each turn appending a little to a long context. About 95% of prompt tokens are prefix-cache hits, so each turn's new prefill is a few thousand tokens. The prefill node has little to do, and the interference PD removes is small. Two replicas spend all sixteen GPUs on decode, each for half the users. That also halves each node's batch, and small batches are exactly where the monokernel applies. Disaggregation pays when prefill is heavy enough to hurt decode. Here it is not.

## Reading the AgentX chart

InferenceX's AgentX chart puts P90 E2E-normalized interactivity on the x axis and total tokens per second per GPU on the y axis. On it, TileRT's point (98.4 at 1534 tok/s/GPU on sixteen GPUs) already sits below and to the left of the published single-node SGLang monokernel curve, which reaches 128.4 at 2834 tok/s/GPU on eight GPUs at concurrency 1. Batch-1 decode buys per-user speed and pays for it in GPUs, and E2E-normalized interactivity also folds in time to first token.

So the curve to beat is our own. In the local runs its frontier is one node at c2, two nodes at c10 and one node at c10. No PD point reaches it.

## The lever: coverage

The published curve falls from 122.8 at concurrency 4 to 58.7 at concurrency 10. The decode logs explain it. At c10, only 35% of decode steps carry two requests or fewer, the most the monokernel takes; every wider step runs SGLang's stock path. Twelve rows (three requests with four MTP tokens each) would cover 47% of steps, and sixteen rows 61%.

Why only eight rows? The kernel keeps a workgroup's working set in its local data share, with a static budget of 32,768 words (128 KiB). The tail and the NextN entry stage every row's bf16 hidden state there, 3,072 words a row at GLM-5.3's hidden size of 6,144, so twelve rows alone would need 36,864 words. One launch per eight rows works for the tail, which runs once a step. For every layer it would re-read the weights, and avoiding that is the reason a decode monokernel exists.

I am widening it to twelve rows. Six limits are gone: the width table; the tail, now an eight-row and a four-row launch; the NextN entry, capped at eight rows so wider draft steps fall back; a weight-streaming shortcut in the input projection; the staging region for this step's new KV rows; and a Triton table kernel that needed a power of two. Dense layers match SGLang at twelve rows. The first MoE layer does not. Its up/gate-to-down hand-off is laid out for two groups of four rows, and verify batches, where several rows pick the same experts, time out while the down stage waits for the FP8 intermediate tile. That stage needs its rows regrouped, and that is the next piece of work.

## What I would tell the next person

- A disaggregation result needs a baseline that spends the same GPUs. Otherwise it measures "one more node", not "a different split".
- Per-GPU throughput charges PD for its prefill node. PD pays off when prefill is expensive and competes with decode; prefix-heavy agentic traces are the opposite case.
- For a fused decode kernel, the width it accepts decides where the curve bends. On AgentX, coverage matters more than topology.

The single-node monokernel curve used here came out of [Experiment 006](/sources/glm53-agentx-hillclimb-mi355x.html). The data for this post is in [Experiment 007](/sources/glm53-pd-monokernel-mi355x.html).
