---
title: "Paper Reading 001 — Polar: Training Agents Without Opening the Box"
description: "Polar (arXiv 2605.24220, NVIDIA) trains language agents with RL by proxying their LLM API calls instead of rewriting the harness. The integration point moves from the agent to the model endpoint — the exact seam we already run with SGLang. A close read of the architecture, the four-step proxy, token-faithful prefix merging, and the SWE-Bench results, with hand-coded SVG diagrams."
date: 2026-05-30
tags: ["paper-reading", "RL", "agent", "MLSys", "SWE-Bench"]
category: "Technical"
lang: "en"
---

This kicks off a "Paper Reading" series alongside the existing Source Reading one — same format, different object. Instead of six hours inside a codebase, it's a close read of one paper, with the same hand-coded SVG diagrams and the same insistence on getting every number right. First up: **Polar: Agentic RL on Any Harness at Scale** (arXiv 2605.24220), out of NVIDIA. Full HTML deep dive at [/papers/polar.html](/papers/polar.html).

## Why this paper, why now

Polar answers a question I keep running into: how do you train an agent with RL when the agent is a complicated harness you didn't write — Codex, Claude Code, a homegrown CLI — and you don't want to rebuild it inside your RL framework? The usual answer is to tear the harness apart and re-express it as `reset/step/reward`. Polar's answer is to not.

Its thesis fits in one sentence the paper actually asks: *"Can we train agents with RL without opening the box?"* Every LLM agent, however baroque, has to call a model. That API call is a common interface sitting outside the agent. Put a proxy there, capture the tokens, reconstruct trajectories, and the harness becomes trainable while running completely unmodified.

This isn't abstract for us. Our harnesses already point at locally-hosted models through SGLang's OpenAI-compatible endpoint — which is exactly the seam Polar puts the RL observation point on. The paper reads like a blueprint for wrapping the harnesses we already run.

## Five findings worth carrying

**1. The contribution is a relocated boundary, not an algorithm.** Polar moves the RL integration point from the harness to the model endpoint. A gateway proxy accepts Anthropic, OpenAI Chat, OpenAI Responses, and Google-style requests, normalizes them to OpenAI Chat (`logprobs=true`), forwards to local inference, and returns the provider's own shape back to the harness. The harness never knows the model was swapped. Plain GRPO does the rest.

**2. Two components, total decoupling.** There's a *rollout server* (coordinates tasks, fans one `TaskRequest` into `num_samples` sessions) and *gateway nodes* (own each session's lifecycle and host the proxy). The trainer is a separate process that consumes finished trajectories as a service. The result: Polar is "agnostic to agent harnesses, training infrastructure, and RL algorithms." Swap the harness, the trainer, or the algorithm — none crosses the service boundary.

**3. Token-faithful prefix merging is the quietly clever part.** Reconstructing a transcript by decoding and re-encoding causes *retokenization drift* — the token IDs you train on differ from what the model sampled. Polar copies sampled assistant tokens verbatim, pulls interstitial tokens from canonical tokenization, and masks everything that wasn't generated (`loss_mask=1` only on behavior-policy tokens). It also stitches append-only completions back into one long trace via a strict token-prefix check, `p_{m+1}[1:|p_m|] = p_m`. That merging cut wall-clock training from **189.5 → 35.2 min (5.39×)** over the same three steps by collapsing 1,185 request-level updates into 218 merged traces — and separately lifted rollout GPU utilization from 20.4% to 87.7%.

**4. The async staging is built for exactly our bottleneck.** Each gateway runs INIT / RUNNING / POSTRUN worker pools plus a bounded READY buffer, so CPU-heavy container prep runs ahead and the GPU never waits on a boot. Evaluators get *prewarmed* during the agent run, and on timeout the gateway still enters POSTRUN to recover partial traces. This is the same "GPU idle while a container builds" problem we hit in the kernel loop.

**5. Results: same 4B base, four harnesses, gains track harness familiarity.** From one `Qwen3.5-4B` checkpoint, GRPO via Polar improved SWE-Bench Verified pass@1 by **+22.6 (Codex 3.8→26.4), +4.8 (Claude Code 29.8→34.6), +0.6 (Qwen Code 34.6→35.2), +6.2 (Pi 34.2→40.4)**. The Codex jump is huge because the base barely knew its tool schemas; where the base was already fluent, headroom was thin.

## ★ The one insight that reframed my mental model

> The hardest part of agentic RL isn't the objective, it's the integration — and the cheapest place to integrate is the one interface every agent already exposes: the model API call. Once you accept that, "treat the harness as a black box" stops being a compromise and becomes the design. For a team that wants RL pointed at its own models on its own hardware, the model endpoint we already serve *is* the training interface. We wouldn't retrofit our stack to a trainer; the trainer attaches to the seam we run.

## What's in the full reading

Seven hand-coded SVG plates:

- **I** — the two places to wire an agent into RL (env-API vs proxy-at-the-boundary).
- **II** — system architecture: rollout server, gateway nodes, proxy, inference, the independent trainer.
- **III** — the proxy's four-step request lifecycle (detect → normalize → capture → return provider shape).
- **IV** — asynchronous staging: the INIT/READY/RUNNING/POSTRUN pools and an evaluator-prewarm timeline.
- **V** — token-faithful prefix merging: append-only chains, the prefix relation, the merged sequence and its loss mask.
- **VI** — SWE-Bench Verified, base vs Polar-RL across all four harnesses.
- **VII** — the prefix_merging vs per_request ablation, three metrics kept distinct.

Plus a section on what it means for our AMD / kimi-cli work, the full GRPO recipe, and a BibTeX citation.

**→ Full deep dive at [/papers/polar.html](/papers/polar.html)** — rendered in a light "polar survey" aesthetic (glacier whites, polar-blue and aurora-teal, a single signal-orange accent; Bricolage Grotesque / Source Serif 4 / Spline Sans Mono), all diagrams inline hand-coded SVG.

---

*Paper Reading 001. The companion [Source Reading series](/blog/source-reading-005-gcnasm/) does the same thing for codebases. Next paper-reading entry will likely stay in agentic-RL infrastructure — Agent Lightning or SkyRL-Agent — to map the field Polar positions against.*
