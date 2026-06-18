---
title: "Long-Sequence MoE RL Training: From First Principles to MI300X"
description: "A first-principles read of Yan Bai's long-sequence MoE RL optimizations — Path B recompute, linear cross-entropy, FSDP2, chunked expert-parallel overlap — and what each one means on AMD MI300X / MI355X."
date: 2026-06-18
tags: ["RL", "MoE", "long-context", "MLSys", "AMD"]
category: "Technical"
lang: "en"
---

Yan Bai published a sharp write-up on training Qwen3.5-35B at 128K context on 32 H100s for RL: [Long-Sequence MoE RL Training Optimization](https://iseekyan.github.io/zh/posts/qwen35-long-sequence-moe-rl/). It's worth reading on its own. This post is not a translation. I want to re-derive the moves from first principles, check the numbers, and then ask the question the original doesn't: what changes when the GPU underneath is an MI300X instead of an H100?

That last part is the whole point for me. Most of the long-context training playbook was written against 80 GB of HBM and NVLink. AMD parts have 2.4–3.6× the memory and a different interconnect, so some of these tricks get more important and some get less. The only way to know which is to understand *why* each one works, not just *that* it works.

## 1. RL is not pretraining, and that changes the target

The first thing the post gets right is a framing point, not a kernel. Pretraining is one giant job that runs for weeks on a fixed, maximal card count. There, the objective is pure MFU: every percent of the roofline you leave on the floor multiplies across the entire run, so people will pay any complexity cost (hand-tuned 4D parallelism, bespoke per-topology sweeps) to claw back utilization.

RL post-training has a different shape. Each step is rollout (autoregressive generation, memory- and latency-bound) followed by train (the policy update, compute-bound). And you don't run one job. You run many small experiments in parallel (reward tweaks, KL coefficients, data mixes), each on a handful of cards, spawning and dying as ideas come and go.

So the binding constraint flips. It is no longer "peak MFU of one job." It is robustness, simplicity, and the ability to redeploy onto whatever card count is free right now without re-deriving a parallelism plan. A config that runs at 90% of peak but boots reliably on any node count beats a 98%-MFU config that needs a fresh sweep every time the batch shape moves. Hold that thought; it justifies almost every decision that follows.

## 2. Path B: spend the cheap resource to buy the scarce one

At 128K context, activation memory is the problem. There are two ways out.

**Path A** keeps activations small by splitting the sequence across more cards: more context/sequence parallelism, so each rank holds a shorter slice. It saves memory, but it adds a parallel dimension, more collectives, and a topology you have to re-tune whenever the layout changes.

**Path B** does the opposite. It accepts *full activation recomputation* (regenerate activations in the backward pass instead of storing them) and uses the freed memory to *grow* the local sequence per card. You trade compute for memory and for simplicity: fewer ranks, fewer collectives, one fat sequence per GPU.

Here's the arbitrage, and it's worth doing the arithmetic. Normal training costs about 6N FLOPs per token (2N forward + 4N backward). Full recompute re-runs the forward inside the backward, so you pay 8N instead of 6N, a fixed **+33% FLOP tax**. That sounds bad until you look at where the achieved throughput sits: ~186 TFLOPs/GPU against an H100's ~989 TFLOPs dense bf16 peak is about **19% MFU**. The machine is ~81% idle on compute. Spending an abundant resource (FLOPs) to relieve the binding one (memory) is textbook roofline reasoning. And the longer local sequence is a bonus: bigger GEMMs run at better efficiency, and you launch fewer kernels, which matters precisely in the low-MFU regime where launch overhead bites.

The tell is the result the post reports: **37–38 GB peak on an 80 GB H100**. Path B didn't just fit; it left half the card empty, which is exactly the headroom you need to pack several RL experiments onto one node.

## 3. Linear cross-entropy: don't build a 32 GB tensor to get a scalar

The language-model head computes logits `Z = H Wᵀ` of shape `[tokens × vocab]`, then reduces them through softmax to a single cross-entropy loss per token. The naive implementation materializes `Z` *and* its gradient `dZ` as dense `[tokens × vocab]` tensors. That is almost pure waste: `Z` is an intermediate you immediately reduce away, and the only thing the backward pass actually needs is `dZ = softmax(Z) − onehot(target)`.

For the post's example config (65,536 tokens, vocab 124,160, bf16), one copy of the logits is:

```
65,536 × 124,160 × 2 bytes = 16.27 GB
```

You need the logits and their gradient co-resident in a naive backward, so it's ~2 copies: **32.55 GB**. (Nice coincidence worth knowing: a single fp32 copy is also 32.55 GB, since 4 bytes × one copy = 2 bytes × two copies. The number is the same either way.)

Fused linear cross-entropy never builds that tensor. It tiles the token axis into blocks; for each block it computes that slice of logits on the fly, reduces it with the same numerically-stable running-max + log-sum-exp recurrence FlashAttention uses, and accumulates the loss keeping only O(block) scalars. In the backward it *recomputes* the logit tile (cheap; the inputs are still resident), forms `dZ` for that tile directly, and immediately contracts it into `dWᵀ` and `dH`. The tile dies inside the loop. Peak vocab-dimension residency drops from `[tokens × vocab]` (16 GB) to `[block × vocab]` (sub-GB).

This is the Liger-Kernel "fused linear cross-entropy" pattern; Apple's "Cut Cross-Entropy" pushes it further by keeping tiles in SRAM and indexing only the target logit. The cost is one extra projection GEMM in the backward (cheap FLOPs again) to remove a 32 GB transient. Same trade as Path B, applied to the one kernel where activation memory, not compute, is the wall.

## 4. FSDP2: shard the memory you can't recompute away

Recompute and linear CE both attack *activation* memory. But there's a floor you can't recompute past: the persistent state. For mixed-precision Adam that's, per parameter, a bf16 weight + a gradient + an fp32 master weight + the two Adam moments, roughly **16–18 bytes per parameter** and completely independent of sequence length. You can't regenerate it; the only lever is to shard it across ranks.

FSDP1 sharded a flattened `FlatParameter`: it concatenated many tensors into one opaque buffer per unit, then chunked that buffer across the data-parallel group. The flat blob fights mixed dtypes, makes per-tensor handling awkward, and composes badly with other parallel axes because it has no idea which logical tensor maps to which mesh dimension.

FSDP2 replaces that with **per-parameter `DTensor` sharding**. Each parameter stays its own tensor, sharded on a named `DeviceMesh` axis, with grads and optimizer states inheriting the same sharding. Because the unit of sharding is the logical tensor on a named mesh dim, FSDP2 composes cleanly with tensor / context / expert parallelism on a shared mesh. A parameter is just a `DTensor` with placements across several dims. The compute mechanic is unchanged: all-gather the full parameter just-in-time, use it, free it. You trade a bit of communication for a smaller resident footprint.

The post reports static memory dropping from **55.91 GB to 47.03 GB**, a drop of 8.88 GB or about 16%. I can't reproduce the absolute numbers without the param count and DP degree, but the arithmetic is self-consistent and the magnitude is right for sharding params + grads + optimizer state across the global DP dimension. The reason it's worth doing here is the same RL-specific reason as everything else: once activations are crushed (recompute + linear CE), persistent state *is* the dominant resident cost, so sharding it is the highest-leverage move left.

## 5. Chunked expert-parallel overlap: hide the all-to-all under the GEMM

An MoE layer with expert parallelism has a fixed dataflow. The router picks experts per token; a **dispatch** all-to-all ships each token to the rank that owns its experts; each rank runs its local expert GEMMs; a **combine** all-to-all routes the outputs back. Done naively, these are three serial stages, and the all-to-all is a global, bandwidth-heavy collective during which the matrix cores sit idle:

```
Serial (per MoE layer):
  NIC : [ dispatch ]················[ combine ]
  CU  : ···········[ expert GEMM ]···········
  time ──────────────────────────────────────▶
  cost ≈ T_dispatch + T_gemm + T_combine
```

Chunking the *token* dimension breaks the dependency, because routing is per-token — each chunk is an independent unit of work. Split the local tokens into chunks and software-pipeline them: dispatch chunk `i+1` while the matrix cores compute chunk `i` while you combine chunk `i−1`. Communication and compute now occupy different hardware (the network engine vs the matrix cores), so they run concurrently:

```
Pipelined (token-chunked, 3 chunks):
  NIC : D0  D1  C0  D2  C1      C2
  CU  :     G0  G1      G2
  time ──────────────────────────────────────▶
  cost → max(T_comm, T_gemm) + one ramp + one drain
```

The post measures a **23.97% speedup at 64K sequence**. Invert it under a simple two-term model and the implied all-to-all fraction is ~19% of the un-overlapped layer time (`1 / (1 − 0.19) ≈ 1.24`), entirely plausible for that much token traffic. The detail I like: the win *grows* with sequence length, but not because there's proportionally more communication (comm and compute both scale linearly in tokens). It grows because longer sequences mean more chunks, and a deeper pipeline amortizes the fixed ramp/drain cost. With a 2048-token chunk, 16K → 32K → 64K is 8 → 16 → 32 chunks, and overlap efficiency `k/(k+1)` climbs 0.89 → 0.94 → 0.97.

## 6. The recipe: collapse the 4D sweep

The last move is the most underrated. A naive parallelism search is a 4D grid over PP / TP / EP / CP. Three values each is 81 configs, and in RL each one costs real GPU-hours and a healthy fraction OOM or die on a shape mismatch before they ever produce a number.

The post's insight is that three of the four dimensions each have a single dominant binding constraint, so you can solve them independently instead of jointly:

| Dim | Bound by | Rule |
|-----|----------|------|
| **EP** | expert count + load balance | Set by model size — pick EP so experts-per-rank lands in a sane range. Nearly discrete; little to sweep. |
| **CP** | activation memory (∝ local seq) | Set CP so local sequence hits a 16K–32K window. `CP=8 → 16K`, `CP=4 → 32K` from 128K context. One inequality, not a sweep. |
| **TP** | per-layer compute/comm (all-reduce every layer) | Only when a layer won't fit or saturate one rank. |
| **PP** | pipeline bubble `(P−1)/(microbatches+P−1)` | Only when chasing peak. |

That turns 81 configs into roughly one, plus a small PP tie-break. The few percent of MFU you give up buys a ~50–80× cut in config-search iterations, and in RL, iteration speed *is* the binding constraint. This is the same logic as Path B, one level up: optimize the loop, not the lap.

The engineering that holds it together is what the author calls **bumblebee / Megatron-Lite**: a design with tight primitive boundaries so each optimization (loss = linear CE, optimizer = FSDP2, comm = chunked EP overlap) is a swappable module, each verified against a paired baseline. That's not a footnote. A small change radius plus A/B-verifiable modules is exactly what lets you trust a stack you're constantly reconfiguring, which is the whole RL bet.

## 7. Reading the numbers honestly

Before porting anything, two notes so the comparison stays clean.

**Always use the dense peak.** The H100's bf16 tensor peak is 989 TFLOPs dense. The 1979 number you see quoted is with 2:4 structured sparsity, which never applies to dense training. Divide 186 by 1979 and you'd report a wrong ~9.4% MFU. The honest figure is ~18.8%.

**There are two honest MFUs, actually.** ~18.8% is hardware-FLOP MFU. But Path B's full recompute means ~33% of those FLOPs are the recompute forward, not model work. If the reported TFLOPs count recompute as useful, the model-FLOP rate you'd compare against a no-recompute baseline is closer to ~14%. Neither is wrong; they answer different questions. Just don't mix them.

Two things from the source I'm taking on faith rather than verifying: "Qwen3.5-35B" is past my knowledge horizon, and its quoted vocab of 124,160 doesn't match the Qwen3 vocab I know (151,936). Still, 124,160 = 970 × 128 is cleanly TP-aligned, so it's a sensible number for the math. Treat both as the post's stated config, not gospel.

## 8. Through an AMD lens

Now the part I actually care about. Three of these four tricks are framework-level and port to ROCm essentially unchanged: FSDP2 is PyTorch `DTensor`/`DeviceMesh`, linear CE is an algebra trick, the EP overlap is a scheduling pattern. But *how much each one matters* shifts, because the scarce resource is different.

| GPU | HBM | BW | Dense bf16 peak | 186 TFLOPs is |
|-----|-----|-----|----------------|---------------|
| H100 SXM | 80 GB | 3.35 TB/s | 989 TFLOPs | ~18.8% |
| MI300X | 192 GB | 5.3 TB/s | 1307 TFLOPs | ~14.2% |
| MI355X | 288 GB HBM3E | 8 TB/s | 2500 TFLOPs | ~7.5% |

The whole memory story relaxes on AMD. A 38 GB peak is 20% of an MI300X and 13% of an MI355X. So:

- **Path B gets *more* attractive, the recompute tax *less* necessary.** With 192–288 GB you can keep more activations resident and recompute *less* of the 33% tax, or grow the local sequence even further before CP is needed at all. The instinct ("grow the sequence, spend FLOPs") is right; the optimal recompute fraction is just lower. The compute headroom is also larger (186 TFLOPs is ~14% of MI300X and ~7.5% of MI355X dense peak), so the "spend FLOPs" side of every trade is even cheaper.
- **CP often shrinks or disappears.** The inequality that forces sequence splitting at 128K on 80 GB loosens by 2.4–3.6×. For many context lengths you can drop CP entirely, which removes a parallel dimension and a class of collectives. The recipe's "set CP for a 16K–32K local window" becomes "set CP only if you actually overflow."
- **FSDP2's memory motive weakens but its composition value stays.** You may shard across fewer ranks and keep more optimizer state local, freeing HBM for a longer sequence or for co-locating the rollout engine with training. The clean `DTensor` composition with the EP mesh is still worth it.
- **The collective tricks need re-profiling, not re-porting.** Every all-to-all and all-gather here runs over **RCCL, not NCCL**, intra-node over Infinity Fabric / xGMI rather than NVLink/NVSwitch. The chunked-EP overlap depends on the collective and the GEMM occupying independent streams; whether that overlap actually materializes is an empirical question on ROCm. And RCCL's topology differs from NVSwitch's flat bisection, so the realized comm fraction may be *higher* — which means overlap helps *more*, not less. Re-profile chunk size with rocprof; don't inherit the NVIDIA tile config.

One kernel-level note for the linear CE port: AMD wavefronts are **64-wide**, not 32. The per-block token tile and the softmax reduction should be sized to 64-lane waves and LDS. A direct port of an NVIDIA tile config will under-occupy. Liger-Kernel already has ROCm Triton paths for fused linear CE, but verify numerics against a paired torch baseline — the running-max correction is load-bearing, and a tile implementation that skips it overflows at bf16.

## 9. What I'd do on ROCm

If I were standing this up on MI300X with slime + Megatron (roughly the [framework choice I argued for earlier](/en/blog/nemo-rl-vs-slime/)), the order of operations follows the roofline, not the paper:

1. **Start without CP.** With 192 GB, see how long a local sequence fits with full recompute + linear CE before you need to split at all. Eliminating a parallel dimension is worth more than any kernel tweak.
2. **Port linear CE first, tuned for 64-wide waves.** It's the biggest single memory win and the most hardware-portable, and the LM head is the one place memory genuinely binds.
3. **Bring up FSDP2 over the EP mesh and verify composition against a baseline.** Per-parameter `DTensor` + custom MoE all-to-all has had rougher edges on ROCm than CUDA. Trust nothing here without an A/B check — which is exactly what the bumblebee primitive-boundary design is for.
4. **Re-profile chunked EP overlap on RCCL last.** It's the most interconnect-sensitive piece. Confirm with hipEvents that the collective and the GEMM actually run concurrently, and watch for RCCL kernels stealing CUs from the expert GEMM.

The deeper lesson isn't any one of these kernels. It's the stance: in RL post-training, the scarce resource is engineering robustness and iteration speed, not peak FLOPs. So you spend the abundant thing (compute) to buy the scarce thing (memory, simplicity, a config that always boots). On AMD, the ledger is even more lopsided in that direction. More HBM, more idle compute, a younger collective stack. That's a setup that rewards exactly this kind of "spend FLOPs, simplify the topology" thinking, as long as you re-measure instead of assuming the NVIDIA numbers carry over.

*Credit where due: the original analysis and all the measured numbers are [Yan Bai's](https://iseekyan.github.io/zh/posts/qwen35-long-sequence-moe-rl/). The first-principles re-derivation, the arithmetic checks, and the AMD mapping are mine.*
