---
title: "Source Reading 006 — FlyDSL, A Layout-Algebra Python DSL with an MLIR Spine"
description: "AMD's FlyDSL is the Python front-end for a Fly-dialect MLIR compiler that lowers layout algebra and copy/MMA atoms to ROCDL on CDNA3/CDNA4. Four examples — vectorAdd, tiledCopy, tiledMma, preshuffle GEMM — form a strict pedagogical ladder; reading them in order gives you every machinery that real production kernels (paged attention, MoE GEMM, flash attention) recombine."
date: 2026-05-26
tags: ["source-reading", "MLSys", "AMD", "CDNA3", "MLIR", "kernel-optimization"]
category: "Technical"
lang: "en"
---

A sixth entry in the source-reading series. After [gcnasm](/blog/source-reading-005-gcnasm/)'s descent into hand-written CDNA3 assembly, FlyDSL goes in the opposite direction — a Python DSL with a proper typed MLIR IR underneath, where layout algebra becomes a first-class concept and copy / MMA atoms compose into production GEMMs without leaving the Python editor. Full HTML deep dive at [/sources/flydsl.html](/sources/flydsl.html).

## Why this repo, why now

There is a particular vertigo that comes from reading a modern GPU kernel and realizing how much of it is layout, not arithmetic. A 4096-cube FP16 GEMM is, viewed line-by-line, mostly bookkeeping — which thread loads which 8 elements, where in shared memory they land, in what order the MFMA instructions consume them, when the next K-block's prefetch overlaps with the current MFMA tail. The actual multiplication is two lines. Everything else is layout.

[FlyDSL](https://github.com/ROCm/FlyDSL) — *Flexible Layout pYthon DSL* — is AMD's response to that observation. It is a Python DSL where you author kernels using <code>@flyc.kernel</code> + <code>@flyc.jit</code>, but underneath sits a real MLIR dialect (the *Fly* dialect) with a <code>!fly.layout</code> type, layout algebra ops, and a pass pipeline that lowers everything to ROCDL and a HSA fatbin. The intellectual parent is NVIDIA CuTe; the AMD-specific contribution is making the algebra a typed IR rather than C++ templates.

I spent four hours reading the four examples under `examples/` — vectorAdd, tiledCopy, tiledMma, preshuffle GEMM — and they turn out to be a strict pedagogical ladder. Each one adds exactly one concept to the previous, and the fourth example is essentially a compact version of every optimization that production CDNA3/CDNA4 GEMMs use. Reading them in order is the most efficient path into the codebase.

## Five findings worth carrying

**1. Layout is a typed IR concept, not a templated string.** FlyDSL's defining choice is that <code>!fly.layout&lt;(8,16):(1,8)&gt;</code> is a real MLIR type. Operations like <code>fly.logical_divide</code> and <code>fly.partition_S</code> consume and produce values of this type. The pass <code>fly-layout-lowering</code> at stage 3 of the pipeline materializes the algebra into concrete address arithmetic. This means any MLIR-aware tool (an autotuner, an analysis pass, a verifier) can reason about layouts as data, not as opaque template parameters.

**2. <code>partition_S</code> + <code>retile</code> is the abstraction that pays off.** In example 03, <code>thr_copy_A.partition_S(bA)</code> hands you the per-thread fragment with shape <code>(V, VM, VN)</code> — already correctly indexed for the underlying MFMA atom's lane layout. Then <code>retile</code> gives you the *same registers* viewed under the MMA layout instead of the copy layout, at zero cost. Without <code>retile</code> you'd need a second fragment and explicit register-to-register copies. The MLIR pass pipeline collapses both views into one VGPR allocation.

**3. Preshuffle is a recurring pattern across the production kernels.** Example 04's <code>shuffle_weight</code> trick — reshape B on the host so a plain <code>buffer_load_dwordx4</code> already lands MFMA-lane-correct in VGPRs — is not a one-off. <code>kernels/preshuffle_gemm.py</code>, <code>blockscale_preshuffle_gemm.py</code>, <code>moe_gemm_2stage.py</code> all use the same idea against different MFMA shapes and dtypes. For inference weights that never change, the trade saves the entire LDS round-trip on B, eliminating both the <code>ds_write</code> traffic and the swizzle that would otherwise be needed for bank-conflict-free B reads.

**4. Schedulers are where the last 30% lives.** A FlyDSL kernel without <code>fx.rocdl.sched_*</code> hints will land around 60–70% of peak FLOPs. With a tuned scheduler — count of MFMAs, count of <code>ds_read</code>s, count of <code>buffer_load</code>s, and the exact interleaving inside the hot loop — the same kernel can hit 90%+. The schedulers in <code>kernels/preshuffle_gemm.py</code> are tuned per (BM, BN, BK, MFMA-shape) tuple, typically with ATT traces from <code>rocprofv3</code>, and break silently when you change the tile size. The 30-line <code>hot_loop_scheduler</code> in example 04 is the minimum viable shape of this artifact.

**5. CUDA Graph capture works out of the box, by design.** The launch path goes through <code>fly-gpu-stream-inject</code>, an MLIR pass that threads the user-provided stream into the actual launch instead of consulting a thread-local variable. For an inference engine that batches kernels into a captured graph for replay (vLLM, SGLang), this is the difference between FlyDSL kernels being usable and being a corner case requiring special handling. Example 01 demonstrates this with a second test that captures the kernel into <code>torch.cuda.CUDAGraph</code> and replays it correctly.

## ★ The one insight that reframed my mental model

> "Layout" is not a description of memory; it is a function. <code>make_layout(shape, stride)</code> defines a map *coord ↦ index*. <code>composition</code>, <code>logical_divide</code>, <code>product</code> are function composition / partition / extension on that map. Once you read FlyDSL through this lens, the gap between "what the code says" and "what the kernel does" closes by an order of magnitude. The function-on-functions framing also explains why the algebra survives compilation — every pass operates on the layout-as-function representation until the final lowering stage materializes it into address arithmetic. The whole pass pipeline is a sequence of layout-function transformations, not a sequence of code-template substitutions.

## What's in the full reading

The HTML deep dive walks through eight modules:

- **M0 — Compass**: the four-example ladder, line counts, concept progression.
- **M1 — Layout algebra in ten minutes**: shape, stride, layout, divide, slice, TV layout.
- **M2 — Example 01 vectorAdd**: minimum viable kernel; BufferCopy vs UniversalCopy; @flyc.jit / Constexpr.
- **M3 — Example 02 tiledCopy**: TV layout in full; <code>partition_S/D</code>; the (V, VM, VN) fragment shape.
- **M4 — Example 03 tiledMma**: MFMA atoms; <code>make_tiled_copy_A/B/C</code>; <code>retile</code>'s two-view trick.
- **M5 — Example 04 preshuffle GEMM**: host preshuffle, LDS XOR swizzle, two-stage pipeline, <code>hot_loop_scheduler</code>.
- **M6 — Compile pipeline**: Python → MLIR → ROCDL → fatbin; the JIT cache; <code>FLYDSL_DUMP_IR</code> workflow.
- **Reefs**: six traps from real debugging — branch-only values, SmemPtr._view_cache, stale schedulers, &c.
- **AMD notes**: where FlyDSL sits relative to Triton-ROCm and Composable Kernel; a practical kernel-tuning workflow.

Plus six hand-coded SVG plates: the compilation pipeline, the two-stage divide cascade, the TV layout grid for tiledCopy, the MFMA wave-tiling of a 64×64 C tile, the preshuffle B before/after diagram, and the software-pipeline timing diagram showing how prefetch overlaps with MFMA.

**→ Full deep dive at [/sources/flydsl.html](/sources/flydsl.html)** — rendered in a cyanotype-blueprint aesthetic (deep navy ground with chalk-white ink and rust / brass / jade / teal annotation marks, EB Garamond + IBM Plex Sans + JetBrains Mono), with all diagrams hand-coded inline SVG.

## Primary references — what to read alongside

1. **[FlyDSL repo](https://github.com/ROCm/FlyDSL)** — start with `docs/layout_system_guide.md` for the complete Quick Reference, then `docs/kernel_authoring_guide.md` for practical patterns. The production kernels under `kernels/` are the dictionary that the four examples are the alphabet for.

2. **[NVIDIA CUTLASS CuTe](https://github.com/NVIDIA/cutlass/tree/main/include/cute)** — the intellectual parent. Layout algebra, copy/MMA atom design, and the <code>partition_S/D</code> idiom are all CuTe ideas, ported to AMD with an MLIR backbone. Reading CuTe docs alongside FlyDSL clarifies which choices are universal and which are AMD-specific.

3. **[Categorical Foundations for CuTe Layouts (Colfax Research)](https://arxiv.org/abs/2601.05972)** — formal treatment of layout algebra as a category. Sufficient to derive every algebraic identity FlyDSL relies on. Read this if you want to extend the algebra, propose custom <code>product</code> variants, or verify that two layouts are equivalent.

4. **[AMD Instinct MI300 CDNA3 ISA Reference](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)** — authoritative on every instruction FlyDSL's lowering emits. § 8 (MUBUF) for <code>buffer_load</code>, § 10 (MFMA) for the matrix core, § 6 (<code>s_waitcnt</code>) for the vmcnt / lgkmcnt mechanics that scheduling controls.

5. **[MLIR documentation](https://mlir.llvm.org/)** — for reading <code>FLYDSL_DUMP_IR</code> output. The <code>gpu</code>, <code>arith</code>, <code>scf</code>, <code>memref</code>, <code>vector</code>, and <code>rocdl</code> dialects that FlyDSL composes with are documented here; the <code>fly</code> dialect itself is documented in-repo under `include/flydsl/Dialect/Fly/IR/`.

6. **[Triton-ROCm](https://github.com/triton-lang/triton)** — the alternative AMD kernel-DSL most readers will know. FlyDSL trades Triton's opacity-around-scheduling for explicit control via <code>fx.rocdl.sched_*</code>. Reading them side-by-side clarifies the design space: same target hardware, different control surfaces.

The right reading order if you are new to AMD kernel programming is roughly: **№ 4 (CDNA3 § 2-3 for hardware intuition) → this writeup → № 2 (CuTe for the algebra) → FlyDSL examples 01-04 → № 3 (categorical paper, optional) → production kernels in `kernels/`**. The MLIR doc and Triton are lookups, not sequential reads.

---

*Previous: [Source Reading 005 — GCNasm](/blog/source-reading-005-gcnasm/). The next entry will likely cover [aiter](https://github.com/ROCm/aiter) — the production AMD kernel library that FlyDSL's tests reference, and a natural next layer up from the layout algebra explored here.*
