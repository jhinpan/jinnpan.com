---
title: "sgl-amd-mono"
description: "AMD monokernels for SGLang: fused per-model decode kernels for MI355X, built for GLM-5.3 and MiniMax."
date: "2026-10-16"
repoURL: "https://github.com/sgl-project/sgl-amd-mono"
group: "now"
order: 3
role: "Contributor"
repo: "sgl-project/sgl-amd-mono"
reading: "/sources/glm53-agentx-hillclimb-mi355x.html"
---

sgl-amd-mono holds the AMD monokernels that SGLang runs on MI355X: one fused kernel per model step instead of many small ones.

I work on the GLM-5.3 and MiniMax monokernels and measure each change at the model level.

Related reading: [GLM-5.3 monokernel hill-climbing on MI355X](/sources/glm53-agentx-hillclimb-mi355x.html), [PD x monokernel: GLM-5.3 on MI355X](/sources/glm53-pd-monokernel-mi355x.html).
