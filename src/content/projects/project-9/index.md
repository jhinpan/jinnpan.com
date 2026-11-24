---
title: "AITER"
description: "AI Tensor Engine for ROCm: the kernel library that serving engines call. I work on GEMM and MoE kernel selection and tuning for gfx1250."
date: "2026-09-01"
repoURL: "https://github.com/ROCm/aiter"
group: "now"
order: 4
role: "Contributor"
repo: "ROCm/aiter"
reading: "/sources/aiter.html"
---

AITER is the AI Tensor Engine for ROCm. Serving engines such as SGLang call its kernels on AMD GPUs.

I work on which kernel runs for which shape on gfx1250: grouped MoE tuning, BF16 GEMM selection, and the FlyDSL-backed kernels inside AITER.

Related reading: [AITER MoE tuner](/sources/aiter.html), [AITER CI kernel surface](/sources/aiter-ci-kernel-audit.html).
