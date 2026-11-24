---
title: "FlyDSL"
description: "A Python DSL for GPU kernels on AMD Instinct GPUs, with typed MLIR underneath. I write and review kernels for MI355X and MI455X."
date: "2026-05-26"
repoURL: "https://github.com/ROCm/FlyDSL"
group: "now"
order: 1
role: "Contributor"
repo: "ROCm/FlyDSL"
reading: "/sources/flydsl.html"
---

FlyDSL is a Python DSL for expressing tiling, partitioning, data movement and kernel structure, compiled through MLIR and LLVM for AMD Instinct GPUs.

I write and review kernels for MI355X (gfx950) and MI455X (gfx1250): softmax, RMSNorm and GEMM tiles, plus the tests and benchmark gates around them.

Related reading: [FlyDSL source reading](/sources/flydsl.html), [270 Fences: the compiler in the FlyDSL kernel loop](/sources/flydsl-compiler-in-the-loop.html), [Experiment 008](/sources/flydsl-compiler-redesign.html).
