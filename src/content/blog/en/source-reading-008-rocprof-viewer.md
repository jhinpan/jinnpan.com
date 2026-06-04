---
title: "Source Reading 008 — Reading the Trace: rocprof-compute-viewer & rocprofv3"
description: "A field guide to AMD's instruction-level GPU profiler: how rocprofv3 captures Advanced Thread Trace, and how to read every panel of rocprof-compute-viewer."
date: 2026-06-04
tags: ["source-reading", "MLSys", "AMD", "ROCm", "profiling", "kernel-optimization"]
category: "Technical"
lang: "en"
---

A counter tells you a kernel is slow. A trace tells you which instruction is waiting, and for what. Most GPU profiling stops at aggregates — occupancy, achieved bandwidth, a flat top-kernel list — which answer *how much* but never *where*. To optimize a hand-written kernel you need the other axis: the instruction stream of a single wave on a single SIMD, with a cycle cost on every line. On AMD that axis is **Advanced Thread Trace (ATT)**, and two open-source tools own it.

Full HTML deep dive: [/sources/rocprof-viewer.html](/sources/rocprof-viewer.html).

## Why this matters

`rocprofv3` (the CLI inside [rocprofiler-sdk](https://github.com/ROCm/rocprofiler-sdk)) arms the hardware, runs your app, and decodes the raw SQTT token stream. [rocprof-compute-viewer](https://github.com/ROCm/rocprof-compute-viewer) (RCV) — a Qt desktop app — turns that decode into a navigable picture: per-SIMD waterfall timelines, an annotated ISA listing, a hotspot histogram, counter overlays. They are the front and back of one pipeline, and most people never learn to read the back. If you write kernels on Instinct, this is the difference between "it's memory-bound" and "line 214's `s_waitcnt vmcnt(0)` ate 1,920 cycles waiting on the `global_load` two instructions up."

## Five findings worth carrying

1. **ATT is a deliberate slice, not the whole GPU.** The capture window is set by `att_target_cu` (typically `1`), `att_simd_select` (which of 4 SIMDs), `att_shader_engine_mask`, and `att_buffer_size` (`0x6000000` = 96 MB per SE). Too wide a window or too small a buffer drops the tail of the trace with no error — if a wave's timeline ends abruptly, raise the buffer before trusting it.

2. **The decode runs automatically, and the filenames are the hardware address.** After `rocprofv3 --att`, the ROCprof Trace Decoder produces a `ui_output_agent_<pid>_dispatch_<N>/` directory of per-wave JSON named `se{SE}_sm{SIMD}_sl{slot}_wv{wave}.json`. RCV's left-panel selector exposes exactly those four fields — reading the trace is literally choosing a coordinate.

3. **Each panel answers one question.** Explorer (which *file* burned cycles — every file carries a latency bar), Utilization (which instruction *class* dominates — VALU / VMEM / SCALAR / OTHER), Compute Unit (a single wave's life, per SIMD-slot), Global (all waves, colored by kernel), ISA (which *instruction* waits). Knowing the mapping turns a wall of panels into a decision tree.

4. **The ISA view's killer feature is the waitcnt arrow.** On AMD the cost of a load is rarely on the load — it is on the later `s_waitcnt vmcnt`/`lgkmcnt` that stalls until the data lands. RCV draws **memory-to-waitcnt dependency arrows** per wave, making that causal pair visible. A fat Latency bar on an `s_waitcnt` with an arrow reaching back to a `global_load` is the textbook memory-latency stall; the same shape on a `ds_read` is the LDS one.

5. **ATT and PMC are separate runs.** ATT gives per-instruction stall timing but no cache/HBM counters; `--pmc` gives counters but no instruction stream, and fails outright if the set can't be collected in one pass. Capture twice, correlate by kernel. PC sampling (`--pc-sampling-beta-enabled`, currently `host_trap`/`time` only) is the cheap statistical middle ground — coverage without causality.

## ★ The one insight that reframed my mental model

> A counter is a *verdict*; ATT is an *address*. A roofline says "memory-bound." The ISA view shows the exact `s_waitcnt` that ate the cycles and the `global_load` it waited on. Optimization never happens on the verdict — it happens at the address. ATT's whole ceremony (the narrow window, the separate runs, the buffer math) exists to close the loop from a number to a line of code you can change.

## What's in the full reading

Five hand-coded SVG plates: the capture-to-view pipeline; what the mask/select parameters carve out of the silicon; the `se_sm_sl_wv` filename decoded into a hardware coordinate; the annotated RCV window (Explorer · timeline · ISA · side panel); and the Utilization lanes showing where a wave goes quiet. Plus the full `rocprofv3` mode taxonomy (trace / PMC / PC-sampling / ATT and the five output formats), a panel-by-panel guide to RCV (including Counters, Wave States, Occupancy, Dispatches, Summary), a five-click reading loop from "this kernel is slow" to a source line, the Qt/CMake/LLVM-C build and its decoder dependency, and the reefs that make ATT readings quietly wrong.

It is grounded in real traces: every FlyDSL kernel in `flydsl-kernel-profiling` on this MI350X already ships an `att_viewer/` directory of exactly these captures.

**→ Full deep dive at [/sources/rocprof-viewer.html](/sources/rocprof-viewer.html)** — a "logic-analyzer console" aesthetic: near-black, phosphor green / amber / cyan / magenta, monospace-forward, with an EN/ZH toggle.

---

*Previous: [Source Reading 007 — AITER MoE Tuner](/en/blog/source-reading-007-aiter/). Part of an ongoing series reading the systems behind AMD GPU performance.*
