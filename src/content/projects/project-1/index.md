---
title: "TritonForge"
description: "LLM-powered GPU kernel synthesis: Train models to convert PyTorch ops into optimized Triton kernels via SFT+RL."
date: "2025-02-01"
repoURL: "https://github.com/RLsys-Foundation/TritonForge"
---

TritonForge is an LLM-powered GPU kernel synthesis framework. It trains models to convert PyTorch operations into optimized Triton kernels through SFT and RL.

Key features:
- Multi-turn compilation feedback loop for iterative kernel refinement
- Cross-platform support for both NVIDIA and AMD GPUs
- Kernelbook and KernelBench integration for systematic evaluation
- SFT + RL training pipeline for teaching models to write high-performance kernels
