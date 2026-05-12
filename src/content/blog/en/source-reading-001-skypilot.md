---
title: "Source Reading 001 — SkyPilot, 211,510 Lines of Multi-Cloud Orchestration"
description: "A six-and-a-half-hour reading of SkyPilot's source — the three-zone architecture, the DP+ILP optimizer, and what it takes to add a new cloud backend."
date: 2026-05-12
tags: ["source-reading", "infrastructure", "MLSys", "skypilot"]
category: "Technical"
lang: "en"
---

This is the first in a trilogy of source-reading deep dives. I spent ~6.5 hours reading SkyPilot end-to-end and produced a full HTML deep dive with seven hand-drawn SVG diagrams. This post is the appetizer; the full reading lives at [/sources/skypilot.html](/sources/skypilot.html).

## Why this repo

SkyPilot is the de-facto open-source layer for "run AI workloads on any cloud". One CLI command (`sky launch foo.yaml`), 24 supported cloud providers, automatic provisioning, spot recovery, model serving. I read it because I'm building a multi-agent kernel optimization system at AMD — and the closest design precedent for "uniform control plane over heterogeneous compute" lives in this codebase.

## Five findings worth carrying

**1. The repo is a three-zone distributed system, not a single binary.** Zone A is the client (CLI + SDK), Zone B is the FastAPI API server (which spawns one subprocess per `launch` request), Zone C is the provisioned compute (with a `skylet` daemon on every node — SkyPilot's kubelet analog). One `sky launch` traverses 9 Python process boundaries.

**2. `sky/cli.py` is a 10-line shim.** The real 7,954-line CLI lives at `sky/client/cli/command.py`. The top-level `sky/__init__.py` uses a facade pattern — every verb (`sky.launch`, `sky.exec`, ...) is re-exported from `sky.client.sdk` so users get the "feels-like-local-function" experience while the implementation crosses processes.

**3. The Optimizer is a DP + ILP hybrid.** Chain DAGs (the common case — one task with multiple resource candidates) go through dynamic programming in `O(N · R²)`. General DAGs go through PuLP/CBC integer linear programming with McCormick linearization of the bilinear `c[u] ⊗ c[v]` cost terms. This is textbook-level engineering — different algorithms for different problem structures.

**4. The 9-stage pipeline is `CLONE_DISK → OPTIMIZE → PROVISION → SYNC_WORKDIR → SYNC_FILE_MOUNTS → SETUP → PRE_EXEC → EXEC → DOWN`.** Each stage maps to one `backend.method()` call. The subtle bit: `OPTIMIZE` runs *outside* the per-cluster lock (because it's expensive), but a `planner` callback can re-fire it inside the lock if cached decisions become stale. Optimistic decision + lock-interior fallback.

**5. The "Ray" in `cloud_vm_ray_backend.py` is not distributed-training Ray.** It's [ray.io](https://ray.io)'s cluster launcher being borrowed for VM lifecycle management. SkyPilot was originally a Ray extension; the name persisted. The `sky/skylet/ray_patches/` subdirectory confirms — these are private patches to Ray's cluster launcher.

## ★ The one design pattern I'm stealing

> The repo has **four** different `state.py` files — `skylet/job_lib.py` (per-node), `jobs/state.py` (managed jobs, global), `serve/serve_state.py` (serving), `server/state.py` (API server). Four state machines layered to **isolate transient failures at the lowest level from policy decisions at the highest**. A spot preemption causes the node-level state to flip RUNNING → FAILED → RUNNING; the managed-job state stays RUNNING throughout. This "layered state machines absorb churn" pattern is directly applicable to multi-agent systems where individual workers fail but the agent task should remain stable.

## What you'll find in the full reading

Seven hand-drawn SVG plates: the architecture overview, a launch sequence diagram, the 9-stage pipeline timeline, the DAG → DP/ILP routing decision, the skylet anatomy, the two-level state machine for managed jobs, and a file-by-file map of "what would it take to add an AMD ROCm cloud backend." Five traps for new readers, three red-line questions to test comprehension.

**→ Read the full deep dive at [/sources/skypilot.html](/sources/skypilot.html)**

---

*Designed as a "1962 aerospace engineering manual meets contemporary architecture firm annual report" — dark navy with bone-white text, rust and brass accents, all diagrams hand-coded SVG (no Mermaid, no runtime deps).*

*Next in series: [Source Reading 002 — SGLang](/blog/source-reading-002-sglang) (an LLM inference engine that's also a four-process distributed system).*
