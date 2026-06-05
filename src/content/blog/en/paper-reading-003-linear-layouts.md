---
title: "Paper Reading 003 — Linear Layouts: every tensor layout is one matrix over F₂"
description: "A close read of 'Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using F2'. The trick: GPU indices are bits, so every Triton layout — Blocked, MMA, swizzled, sliced — is a binary matrix over GF(2). Layout conversion collapses to B⁻¹A, broadcast becomes a zero column, bank-conflict-free swizzling becomes a subspace search. With the AMD angle: mfma layouts are linear too. Hand-coded SVG plates, bilingual."
date: 2026-06-05
tags: ["paper-reading", "triton", "layout", "kernel", "AMD", "MLSys"]
category: "Technical"
lang: "en"
---

Paper Reading 003. This one is short on new machinery and long on reorganization — exactly the kind of paper I like. **Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using F2** ([arXiv:2505.23819](https://arxiv.org/abs/2505.23819)) makes one bet: every tensor layout a GPU compiler will ever need is a *linear function over the two-element field*, and that single fact dissolves most of the layout problem. Full bilingual HTML deep dive at [/sources/linear-layouts.html](/sources/linear-layouts.html).

## Why this matters

A tensor layout is the map from a logical element — `A[row, col]` — to the physical resource holding it: which register, in which thread, in which warp, or which byte of shared memory. Get it wrong and the Tensor Core reads garbage; get it suboptimal and you pay for data movement you never needed. Historically every DL compiler hand-wrote this layer: per-layout interface methods, per-pair conversion routines, magic swizzle constants. Triton's own bug tracker is blunt about the cost — **12% of all reported bugs were layout-related**.

For us at AMD this isn't a Triton curiosity. A real NV→AMD port isn't a syntax translation; a CUDA kernel only runs *well* on MI300X once its data is re-tiled for 64-wide wavefronts and re-swizzled for LDS banks. The paper proves AMD `mfma` layouts are linear too, and its two flagship algorithms are hardware-agnostic. So the layout layer of any porting agent or kernel-optimization loop we build can start from this abstraction instead of a registry of named special cases.

## Five findings worth carrying

1. **A layout is `y = M·x` over F₂.** Concatenate the hardware-index bits (register, thread, warp) into a vector `x`; write the tensor coordinate `(i, j)` as a vector `y`; the layout is a 0/1 matrix `M` with `y = M·x`, where multiply is AND and sum is XOR. Each row of `M` says how one output bit is XOR-built from input bits. That's the entire abstraction.

2. **Conversion collapses to `B⁻¹A`.** The old way needed a bespoke converter per `(source, target)` pair — an O(n²) pile of hand-written paths, each a fresh chance for a bug. With matrices, sending data from layout A to B is `B⁻¹∘A`, computed once by a generic F₂ Gaussian-elimination routine. The quadratic family becomes one algorithm. Decomposing the result by resource (reg/thread/warp) even tells the compiler *where* data must move — an identity warp block means "no inter-warp movement," the green light for a warp shuffle over a shared-memory round-trip.

3. **Two clean definitions fall out.** A *distributed* layout (Blocked, MMA/wgmma/mfma, Sliced) is a permutation matrix with optional zero columns — and a **zero column is exactly broadcasting**, which used to be a persistent bug source. A *memory* layout is invertible with one or two ones per column; the two-ones case is a shear `I + C` where `C` mixes coordinate bits — that shear *is* mma swizzling, demystified from its `per_phase`/`max_phase`/`vec` incantation.

4. **Optimal swizzling becomes a subspace search.** A bank conflict is precisely `span(Seg) ∩ span(Thr) ≠ {0}`. The algorithm builds the segment basis from the *complement* of the thread-access subspace, so distinct segments hit distinct banks, dipping into the conflicting space only when the safe space runs out — a provably minimal-conflict swizzle for *any* layout, on *any* vendor. The same machinery generates optimal warp shuffles.

5. **The payoff is robustness first, speed second.** On 265 real TritonBench cases: up to 1.40×, averaging 1.07×. The louder numbers are correctness and micro wins — mixed-precision matmul pass rate 46.6% → **100%** (784 cases), LD/ST vectorization up to 7× wider, broadcast shared-stores −76%, layout-conversion shuffle up to 3.93×, gather up to 14.20×.

## ★ The one insight that reframed my mental model

> The AMD–NVIDIA layout gap is not physics. The swizzle math is identical over F₂; what NVIDIA has and AMD lacks is a decade of primitives like `ldmatrix` that turn the optimal layout into one instruction. On the MI250 the framework was correct but gained only 1.00×–1.03× — gated by missing primitives, not by the algebra. That's a software-maturity gap, and software-maturity gaps are exactly what systematic search closes. The paper hands us the search.

## What's in the full reading

Six hand-coded SVG plates, EN/ZH toggle, phosphor-oscilloscope aesthetic: the algebra ladder (group → ring → field → F₂); the core `y = M·x` mapping with a worked bit-vector; the 16×16 motivating example traced for thread t9's register r1; the four operators (composition, product, left-division, right-inverse); distributed-vs-memory layout (permutation + zero column vs shear); and bank conflict as a subspace intersection with a read split into four conflict-free transactions. The closing section maps all of it onto a concrete first kernel to win on MI300X.

**→ Full deep dive at [/sources/linear-layouts.html](/sources/linear-layouts.html)** — near-black CRT-glow palette, lime/cyan/magenta phosphor, binary matrices rendered as bit grids.

---

*Previous: [Paper Reading 002 — Kernel Design Agents](/en/blog/paper-reading-002-kernel-design-agents/). Series: close reads of MLSys work that feeds our AMD kernel-agent goals.*
