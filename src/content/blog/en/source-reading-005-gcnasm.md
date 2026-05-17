---
title: "Source Reading 005 — GCNasm, Sixty-Four Katas for the AMD ISA Manual You Never Finished"
description: "carlushuang's gcnasm repo is the rare middle ground between HIP tutorials and the 1,200-page CDNA3 ISA manual: 64 short, self-contained kernels that show what hand-tuned AMD code actually looks like. Six hours of careful reading buys you a working mental model of MFMA, vmcnt pipelining, DPP cross-lane primitives, and the trick the LLVM assembler refuses to let you write."
date: 2026-05-17
tags: ["source-reading", "MLSys", "AMD", "CDNA3", "kernel-optimization"]
category: "Technical"
lang: "en"
---

A fifth entry in the source-reading series, shifting domain entirely: from Python-heavy serving systems to hand-written GPU assembly. The codebase is small (~73k LOC across 64 self-contained folders), but the surface area it covers — every optimization pattern that separates a 30% kernel from a 95% kernel on AMD CDNA3 — is enormous. Full HTML deep dive at [/sources/gcnasm.html](/sources/gcnasm.html).

## Why this repo, why now

There has always been a documentation chasm in AMD GPU programming. On one side: HIP tutorials, friendly and portable, almost entirely unable to express the half-dozen tricks that take a kernel from working to fast. On the other side: the CDNA3 ISA manual, authoritative, exhaustive, and never once showing you what a complete optimized kernel looks like.

[carlushuang's gcnasm](https://github.com/carlushuang/gcnasm) is a bridge built plank by plank — 64 folders, each one a complete program demonstrating one technique. No library, no central abstraction, no build system tying anything together. Every folder has a `build.sh`, an `.s` or `.cc` or both, and (if you're lucky) a README. That structure is the point: each kata is something you can read, compile, run, and modify in an hour.

For anyone whose work involves *generating* AMD kernels — by hand, by codegen, or by autonomous agent — this is calibration data. It answers a question the official docs do not: *what should a good kernel look like when a fluent AMD engineer writes it?*

## Five findings worth carrying

**1. The assembler has a structural blind spot.** `buffer_load_dword … offen lds` is a real instruction in the gfx942 hardware. The LLVM MC assembler (ROCm 7.1 / clang 20) refuses to accept it. carlushuang's workaround in `vector_add_kernel.s:147-158` hand-encodes the MUBUF bytes via `.long 0xE0511000`. Any agent that emits `.s` for clang to assemble will be unable to produce this kind of pattern — which has downstream implications for how kernel-generation systems should be designed.

**2. The `vmcnt(3)` arithmetic is engineering folklore, not derivable from docs.** The vector_add kernel keeps exactly three memory operations in flight at all times — two prefetches and one store from the previous half-iteration. The choice is not in the ISA manual; it falls out of careful reasoning about which operations need to be drained at each loop boundary. `README.md:410-452` walks through the accounting. This is the kind of knowledge agents have to either be taught explicitly or rediscover through measurement.

**3. OOB-by-SRD eliminates exec mask manipulation entirely.** Setting `num_records = N * sizeof(element)` in the buffer SRD makes hardware return zero for out-of-bounds loads and silently drop out-of-bounds stores. The `vector_add` kernel never touches the `exec` mask — all 64 lanes execute every instruction unconditionally, with hardware handling the boundary. The CUDA equivalent has half a dozen `s_and_saveexec_b64` instructions; the AMD version has none. This pattern is everywhere in production AMD kernels (aiter, opus_*) and is one of the cleanest examples of trading software complexity for hardware features.

**4. MFMA tile shape determines everything downstream.** `matrix_core/matrix_core.cc` shows three layout strategies for the same `32×32×8 fp16` MFMA: standard (column store), swap A/B (row store with `buffer_store_dwordx2`), and swap+swizzle (row store with `buffer_store_dwordx4`). The C-matrix layout is not a free choice — it's determined by how you fed A and B into the MFMA. Designing the LDS first and the MFMA later is how novice AMD GEMMs end up at 30% of peak.

**5. `co-exec` is the most under-appreciated tool in the repo.** A 320-line Python script that compiles `.s` assembly directly with `clang++ -x assembler` (no hipcc, no linker, no ROCm tooling), runs the resulting `.hsaco`, and disassembles it for inspection — all in one call. The compile step takes ~200 ms versus 8-15 seconds for hipcc. For a kernel-generation agent trying 500 variants per day, that's the difference between a 70-minute job and a 28-hour job.

## ★ The one insight that reframed my mental model

> The assembler is not the lowest level. There is a layer below it — raw machine bytes — where some optimal patterns are only expressible. This means a kernel-generation system that targets `.s` source has a structural ceiling: anything that requires hand-rolled MUBUF encoding, internal-only opcodes, or undocumented register classes is invisible from above. The pragmatic answer is layered emission — DSL for fast exploration, intrinsics for promising candidates, raw assembly with byte-level patches for the last 5%. This is how the repo itself is organized (HIP+intrinsic in `matrix_core/`, raw assembly in `matrix_core_asm/`, DSL in `opus_*`), and it's probably how an agent system should be structured too.

## What's in the full reading

The HTML deep dive walks through eight modules:

- **M0 — Topography** of all 64 katas, organized on a six-axis grid (data shape × operation kind).
- **M1 — CDNA3 in ten minutes**: wavefront, CU/SIMD, VGPR/AGPR unified file, vmcnt/lgkmcnt FIFOs, buffer SRD anatomy.
- **M2 — vector_add_asm**: line-by-line walk through the five canonical optimization patterns.
- **M3 — bandwidth_memread**: the roofline tool (4.56 TB/s measured on MI308X) and the dead-store trick.
- **M4 — Matrix Core / MFMA**: three layout strategies, AGPR scheduling, the `s_nop 16` dead-time.
- **M5 — DPP wave-reductions**: six-stage butterfly, the 6 DPP control codes, bpermute escape hatch.
- **M6 — co-exec and measure_ips**: the iteration-speed infrastructure for kernel agents.
- **M7 — Oddments**: magic integer division, FP8/INT4 conversion, atomic CAS, HW probing.

Plus seven hand-coded SVG plates (repo compass, CU exploded view, vmcnt FIFO timeline, roofline chart, MFMA tile diagram, DPP butterfly, buffer SRD anatomy), a "Reefs" section with six pitfalls, and a "Red Lines" section with three structural questions for kernel-agent design.

**→ Full deep dive at [/sources/gcnasm.html](/sources/gcnasm.html)** — rendered in a logic-analyzer aesthetic (phosphor green on silicon black + Newsreader/Manrope/Geist Mono), with all diagrams hand-coded inline SVG.

## Primary references — the AMD docs to read alongside

gcnasm only makes sense when you read it against the AMD specifications it implements against. Six load-bearing documents — the gap between "this instruction exists in the ISA" and "the LLVM assembler accepts it" is exactly the territory the repo maps:

1. **[AMD Instinct MI300 CDNA3 ISA Reference Guide (Aug 2025)](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)** — the 1,200-page authority for gfx942. § 8 (MUBUF/MTBUF/FLAT) covers M2's buffer loads, § 7 (Vector ALU) covers M5's DPP, § 10 (MFMA) covers M4's matrix-core families, § 6 (s_waitcnt encoding) covers the vmcnt mechanics this repo lives on.
2. **[AMD CDNA3 Architecture White Paper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf)** — XCD/AID chiplet topology, Infinity Fabric, HBM3, the unified VGPR+AGPR register file. Read this *before* the ISA reference so the instructions have hardware context (why MUBUF exists at all, why MFMA writes to AGPRs, what 304 CUs means physically).
3. **[AMD Instinct CDNA4 ISA Reference Guide (Aug 2025)](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf)** — for MI355X (gfx950). CDNA4 is a strict superset of CDNA3 with added fp8/fp6/fp4 MFMA shapes and an expanded MTBUF; diff this against № 1 to see what's new and what to retune.
4. **[LLVM AMDGPU Backend documentation](https://llvm.org/docs/AMDGPUUsage.html)** — the definitive source for what the assembler *actually* accepts (sometimes a strict subset of what the ISA defines — see Reef 1, the `buffer_load…lds` case). When LLVM contradicts AMD, your code compiles against LLVM, not against AMD. Also documents the `__builtin_amdgcn_*` intrinsics used throughout M2 and M4.
5. **[ROCm GPU architecture specifications](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html)** — quick reference for CU count, peak FP/INT throughput, HBM bandwidth, LDS size per SKU. Source of the roofline constants in M3 (the 4.56 TB/s measured on MI308X versus the ~5.3 TB/s peak comes straight from this table).
6. **[AMD GPUOpen ISA documentation hub](https://gpuopen.com/amd-gpu-architecture-programming-documentation/)** — index of every public AMD GPU ISA reference (GCN3-5, CDNA1-4, RDNA1-4). Bookmark for cross-architecture work or comparing how MFMA evolves across generations.

If you're coming to AMD GPU programming for the first time, the right order is **№ 2 → § 2-3 of № 1 → this writeup → § 6-10 of № 1 → № 4 alongside as you start to assemble**. The white paper builds intuition for the hardware shape; the early ISA chapters establish the register and memory model; this writeup gives you running examples to anchor against; the deep ISA chapters become readable once you have those anchors; the LLVM doc is your reference for what the toolchain actually accepts. № 3, № 5, № 6 are lookups, not sequential reads.

---

*Previous: [Source Reading 004 — mini-SGLang](/blog/source-reading-004-mini-sglang/). The next entry will likely cover [aiter](https://github.com/ROCm/aiter) (the production AMD kernel library that gcnasm's `opus_*` examples build on) or Triton-ROCm's codegen pipeline — both are natural extensions of the foundation laid here.*
