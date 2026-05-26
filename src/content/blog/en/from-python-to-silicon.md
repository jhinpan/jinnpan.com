---
title: "From Python to Silicon — A Compiler & Arch Primer for the Working ML Engineer"
description: "You can write production ML systems for years without knowing what IR, MLIR, LLVM, ISA, or FFI actually mean. This is the patch — a bilingual primer for the undergrad-CS-but-skipped-compilers crowd, with a full HTML deep dive carrying six hand-drawn SVG plates."
date: 2026-05-26
tags: ["compiler", "computer-architecture", "MLIR", "LLVM", "ISA", "FFI", "MLSys", "primer"]
category: "Technical"
lang: "en"
---

You can write production ML systems for years without quite knowing what an IR is. You import `torch`, you call `torch.matmul`, the GPU lights up, a few milliseconds later there is a tensor on the other side. Somewhere between your Python file and that tensor sit at least eight layers of software, three forms of intermediate representation, two compiler frameworks, one device driver, and a chip that speaks an instruction set someone in Bangalore or Sunnyvale spent two years specifying.

Most of the time you do not need to know any of this. But every once in a while something underneath leaks — a kernel runs slower than it should, a new accelerator does not have a backend, a wheel built for CUDA 12.4 will not load on CUDA 12.6 — and the abstraction stops being free. This essay is for the working ML engineer who hit one of those leaks and realized they could not quite name the pieces.

**Full HTML deep dive (six hand-drawn SVG plates, EN / ZH toggle):** [/sources/from-python-to-silicon.html](/sources/from-python-to-silicon.html)

## Five words, thirty seconds each

**IR (Intermediate Representation)** — the data structure the compiler uses to reason about a program. More analyzable than source code (no syntactic sugar, no implicit conversions), more structured than machine code (variables, types, control flow markers). LLVM IR is an IR, MLIR's `linalg.matmul` is an IR, PyTorch's FX Graph is an IR. The word is badly overloaded — always pin down whose IR.

**MLIR (Multi-Level IR)** — a framework for building IRs. The core abstraction is a *dialect*: a named bundle of ops, types, and attributes. The standard distribution ships dozens of dialects, from high-level `linalg` (tensor ops) down to `llvm` (mirroring LLVM IR). After MLIR, most new ML compilers (Triton, FlyDSL, Torch-MLIR, IREE, OpenXLA) share one parser, one verifier, one pass manager. Infrastructure stopped being a tax.

**LLVM** — one of the most successful compiler infrastructure projects in history, now covering essentially everything that isn't Microsoft's MSVC. It does three things: provides one IR (LLVM IR), one set of backends (x86, ARM, RISC-V, AMDGPU, NVPTX, Wasm, SPIR-V, ...), and one optimizer. Rust, Swift, Julia, Zig, modern Fortran, Numba, every GPU kernel DSL — all run on LLVM.

**ISA (Instruction Set Architecture)** — the chip's contract with the outside world. Specifies which instructions are legal, what they do, the register file, the memory model. For ML, ISA matters more than ever — performance is now almost entirely determined by whether the compiler reaches the chip's matrix instructions: Intel AMX, ARM SME, NVIDIA Tensor Core (HMMA), AMD MFMA (CDNA) / WMMA (RDNA). The difference between 50 TFLOPS and 500 TFLOPS on the same silicon area lives here.

**FFI (Foreign Function Interface)** — the mechanism a program in one language uses to call functions in another. In PyTorch you write Python, it crosses through pybind11 into C++, then through the HIP / CUDA driver to the GPU. Every bridge has its own ABI, data-representation conventions, lifetime rules. Apache TVM-FFI is the cleanest contemporary attempt at "stable C ABI for ML kernels" — one wheel, loadable from PyTorch / JAX / CuPy / Paddle.

## The one sentence that ties them together

> Every layer rewrites the program from "easier to write" toward "easier to execute." IR is the data type the compiler uses to do those rewrites. MLIR is a framework for building IRs at many levels. LLVM is one specific IR plus a battle-tested set of backends. The ISA is the chip's spec — what the backend must emit. FFI is how higher and lower layers talk across language and runtime boundaries. Everything else is engineering.

## What's in the full HTML deep dive

10 chapters and 6 hand-drawn SVG plates, set in an "engineering monograph" aesthetic (warm parchment + oxblood + forest green + brass):

- **Plate I · The vertical stack** — from `torch.matmul` to a transistor switching, all 9 layers on one page
- **Plate II · What SSA actually is** — the same Python function as a control-flow graph in SSA form
- **Plate III · The MLIR dialect tower** — linalg / affine / scf / memref / gpu / rocdl / llvm as floors of one building
- **Plate IV · LLVM as the universal backend** — 8 frontends → LLVM IR → 8 targets, the hub-and-spoke that made the modern compiler ecosystem economical
- **Plate V · Matrix instructions across vendors** — Intel AMX, ARM SME, NVIDIA HMMA, AMD MFMA, AMD WMMA, RISC-V matrix — the fragmented "ML ISA war"
- **Plate VI · FFI bridges** — Python / C++ / GPU runtime / kernel — three bridges and three failure modes

A whole chapter walks a single FP16 4096-cube `torch.matmul(A, B)` on AMD MI300X from Python entry to the actual MFMA instructions retiring on a wavefront. This is where the vocabulary you just learned gets spent.

The final chapter — "Reefs" — collects the common confusions: which IR did you mean, LLVM IR ≠ MLIR, PTX is virtual / SASS is real, ISA ≠ ABI, "Tensor Core" is a brand not a category, "compiler" means at least three different things in ML, AOT vs JIT.

## Why I wrote it

The hard part of writing a primer is not finding material — it is landing it where the right reader needs it. What I want you to walk away with: the next time an ML workload doesn't hit its expected instructions, or a wheel won't load, or a new chip has no backend, you can name what just happened. Naming things is most of debugging.

Full read: [/sources/from-python-to-silicon.html](/sources/from-python-to-silicon.html) (bilingual toggle in the top-right; safe to share with colleagues either way).

---

*Companion: the other entries in this [Source Reading series](/blog/source-reading-006-flydsl/) read specific repos in this stack — SkyPilot (orchestration), SGLang and vLLM (inference engines), mini-SGLang (teaching version), gcnasm (AMD CDNA3 assembly), FlyDSL (layout-algebra Python DSL). Each is a worked example of one of the boxes in Plate I.*
