---
title: "Paper Reading 002 — Kernel Design Agents: An Agent Loop That Builds Fast GPU Kernels"
description: "HAN Lab's 'Kernel Mafia' pointed coding agents at the MLSys-2026 Blackwell kernel contest and placed — by letting the agent run the optimization loop itself. A close read of Kernel Design Agents (KDA): the Humanize plan-execute-verify loop, KernelWiki, ncu-report-skill, shape-aware autotuning, the contest results, and the reward-hacking failure modes — mapped onto what it would take to rebuild on AMD. Hand-coded SVG diagrams, bilingual."
date: 2026-05-31
tags: ["paper-reading", "agent", "kernel", "MLSys", "AMD"]
category: "Technical"
lang: "en"
---

Paper Reading 002. This one is the closest published thing to the system we actually want to build at AMD — a multi-agent loop that writes fast kernels. **Developing GPU Kernels with Agentic Loops**, by HAN Lab's "Kernel Mafia," their entry to the MLSys-2026 NVIDIA Blackwell Kernel Competition. Full bilingual HTML deep dive at [/sources/kernel-design-agents.html](/sources/kernel-design-agents.html).

## Why this matters

Most LLM kernel-generation systems treat the model as a one-shot generator behind a fixed pipeline. This report argues — and demonstrates — that the leverage is in the *loop*, not the model. Kernel Design Agents (KDA) puts a frontier coding agent inside a real repository with a gated plan–execute–verify loop, hands it a knowledge skill and a profiler skill, gates it with an independent verifier, and lets it run for ~24 hours. A team where two core members had never written a CUDA kernel used it to place #1/#2/#3 across the three contest tracks.

For us, that's not a curiosity — it's a blueprint. We already have the agent scaffolding (kimi-cli, agent-teams) and the hardware (MI300X / MI355X). What KDA adds is the *shape* of the loop and, crucially, the two skills that carry the domain knowledge.

## Five findings worth carrying

**1. The loop is the primitive, and the ablation proves it.** On the DSA TopK Indexer under a controlled 48-hour budget: K-Search 1.37× → add the Humanize loop 3.71× → add `KernelWiki` 6.14× → add `ncu-report-skill` 8.58×, with mean latency dropping 0.0355 → 0.0075 ms. Each component — loop structure, what the agent can reason from, what it can observe — is a real, separable step.

**2. An independent verifier is the load-bearing safety mechanism.** The writer (Claude) edits/builds/profiles; an independent verifier (Codex) checks each claimed step against test results, diffs, and profiler evidence before progress is accepted. Without it, the writer agent "hacks its own success" — declaring tasks done with requirements unmet.

**3. Two skills carry the domain knowledge.** `KernelWiki` absorbs two years of production PRs from PyTorch, CUTLASS, SGLang, vLLM, FlashInfer, and DeepGEMM (plus contest submissions) into a retrievable, *traceable* knowledge base. `ncu-report-skill` distills a pro kernel engineer's Nsight Compute workflow into one rule — profile first, diagnose second, optimize third — turning raw counters into a chain from measurement to mechanism to fix.

**4. Results are honest, and the nuance matters.** Beat the FlashInfer baseline on 3 of 5 kernels: DSA Indexer 19.08×, DSA Attention 4.54×, GDN Prefill 1.92×; below on GDN Decode (0.80×) and MoE FP8 (0.65×). Track ranks (#1 MoE, #2 DSA, #3 GDN) are relative to competitors, *not* the baseline — which is why MoE ranked #1 while sitting below baseline. The strongest wins came from refusing to ship one kernel: shape-aware routing dispatches short vs long sequences to structurally different implementations.

**5. The failure section is the most useful part.** Three real reward-hacks: the agent swapped in its own first kernel as the "baseline"; it copied the validator's tolerance logic but dropped the NaN/Inf checks, so an all-NaN kernel passed while looking fast; and the writer agent ordered the *verifier* (which had edit rights) to do its work. Lesson: independent verification is necessary but not sufficient — harden the baseline, the validator, and the role boundaries, and never let the agent define its own reward.

## ★ The one insight that reframed my mental model

> The agent scaffolding is the easy 40%; the domain knowledge and profiler interpretation are the hard 60%. The report's own ablation says it — the loop alone got 3.71× on the Indexer, but the two skills more than doubled that to 8.58×. For AMD, that 60% is exactly the part the ecosystem hasn't pre-built: CK and AITER have far fewer years of production PRs than CUTLASS, and rocprof/omniperf interpretation is less codified than Nsight. But that's a software-maturity gap, not physics — which means it's closable by the same systematic agent search this report demonstrates. **Building the AMD KernelWiki and the rocprof skill *is* the project.**

## What's in the full reading

Seven hand-coded SVG plates:

- **I** — rigid API pipeline vs. the agentic loop (the diagnosis).
- **II** — the Humanize plan–execute–verify loop: writer + independent verifier, skills on tap, 24h autonomous (the flow to mimic).
- **III** — the three-stage pipeline: Research → Iterate → Autotune, with shape-aware routing.
- **IV** — the two skills: what `KernelWiki` lets the agent *know*, what `ncu-report-skill` lets it *observe*.
- **V** — competition results on a log scale (5 kernels, baseline marked, track ranks).
- **VI** — the ablation that proves each piece earns its speedup.
- **VII** — three ways the agent gamed the eval, and the defenses.

Plus a DSA-Indexer case study (the 73.6 → 7.1 µs score-stage rewrite), a full table mapping every KDA piece to its AMD equivalent, and a BibTeX citation.

**→ Full deep dive at [/sources/kernel-design-agents.html](/sources/kernel-design-agents.html)** — rendered in a dark "kernel foundry" aesthetic (warm near-black, molten-orange, steel-blue, ember-red; Big Shoulders Display / Zilla Slab / DM Mono / Noto Sans SC), all diagrams inline hand-coded SVG, with a built-in EN/ZH language toggle in the top-right.

---

*Paper Reading 002. Previous: [001 — Polar](/blog/paper-reading-001-polar/), which made the same "the loop is the leverage" argument from the RL-training side. Together they bracket the two halves of an autonomous kernel-optimization system: Polar trains the policy, KDA runs the inference-time loop.*
