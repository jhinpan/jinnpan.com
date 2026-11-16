---
title: "270 Fences: Putting the Compiler Back in the FlyDSL Kernel Loop"
description: "FlyDSL compiles MI355X and MI455X kernels with one in-process LLVM it can patch, yet 62 of its 180 kernel files override that compiler by hand. Where the kernel optimization loop breaks, which layer of the compiler chain fixes each break, and how a CAKE-style outer loop could turn hand overrides into compiler features."
date: 2026-11-16
tags: ["compiler", "FlyDSL", "LLVM", "MLIR", "kernel-optimization", "AMD", "agents", "MI355X"]
category: "Technical"
lang: "en"
---

"Think about the compiler chain" is good advice that comes without a next step. A kernel engineer knows how to change a kernel. What would it mean to change the compiler: which part, how, and how would anyone know it helped?

Start from what sets a kernel's speed. Once the algorithm and the tiling are fixed, performance comes down to a few decisions: which instructions run, in what order, in which registers, and where each wave stops to wait for memory. Each of these decisions is made by someone, either the kernel author or the compiler. The compiler chain matters exactly where it makes a decision badly, or makes it in a way nobody can see. This post finds those places in [FlyDSL](https://github.com/ROCm/FlyDSL), the Python kernel DSL that AMD uses for MI355X and MI455X kernels, and asks how the compiler could improve together with the kernels instead of being worked around.

The short version: FlyDSL owns every layer of its compiler, and its kernels override the backend's scheduling, register and wait decisions in about a third of all kernel files. The most useful compiler work removes the need for those overrides, one recurring pattern at a time, and accepts a change only if every kernel agrees.

All FlyDSL references are pinned to commit [`aae1553`](https://github.com/ROCm/FlyDSL/tree/aae155359ddc9b2bb1064ec8d391bc25aa4c0d3e) (16 November 2026).

## One toolchain, and FlyDSL owns it

FlyDSL does not call hipcc, amdclang, comgr or hipRTC. It compiles inside the Python process with its own MLIR and LLVM: upstream `llvm-project` pinned at `941a04e` (LLVM 24 development), plus nine FlyDSL patches ([`llvm-build-info.json`](https://github.com/ROCm/FlyDSL/blob/aae155359ddc9b2bb1064ec8d391bc25aa4c0d3e/thirdparty/llvm-build-info.json)). The pipeline lowers the Fly dialect through FlyDSL's own passes, converts the GPU module to ROCDL, and then lets LLVM optimize at `-O2`, generate AMDGPU code and assemble it ([`rocm.py`](https://github.com/ROCm/FlyDSL/blob/aae155359ddc9b2bb1064ec8d391bc25aa4c0d3e/python/flydsl/compiler/backends/rocm.py#L323-L378)). The object becomes an HSA code object through an LLD linked into the same library since [#987](https://github.com/ROCm/FlyDSL/pull/987), or through ROCm's `ld.lld` when that LLD is not built in. ROCm contributes only the device bitcode libraries, the fallback linker and the HIP runtime that loads the result.

MI355X (gfx950) and MI455X (gfx1250, the MI450 series) take the same path. The target attribute changes the chip name and, for gfx1250, marks the target as wave32 ([`gpu_module_targets`](https://github.com/ROCm/FlyDSL/blob/aae155359ddc9b2bb1064ec8d391bc25aa4c0d3e/python/flydsl/compiler/backends/rocm.py#L425-L438)). The kernels themselves use different instruction families: MFMA on gfx950, WMMA and the tensor data mover on gfx1250 ([`arch-gfx1250.md`](https://github.com/ROCm/FlyDSL/blob/aae155359ddc9b2bb1064ec8d391bc25aa4c0d3e/.claude/skills/llvm/references/arch-gfx1250.md)). gfx1250 also leans harder on the patches: six of the nine carry gfx1250-specific changes, including a backport of the `gfx1250-strict` target that upstream LLVM added after FlyDSL's pin ([llvm-project #218543](https://github.com/llvm/llvm-project/pull/218543)).

The design choice matters more than the details. A compiler that ships as a binary can only be steered with flags. A compiler built from source, with your own passes in front of it and your own patches inside it, can be changed at every stage, and FlyDSL already does this routinely. That is the precondition for putting the compiler in the optimization loop at all.

<figure>
<div style="overflow-x:auto">
<svg viewBox="0 0 680 548" role="img" aria-labelledby="fc1-title" style="width:100%;min-width:600px;height:auto;font-family:inherit">
<title id="fc1-title">The FlyDSL compile chain and the decisions each stage owns</title>
<defs><marker id="fc1-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="currentColor"/></marker></defs>
<g fill="none" stroke="currentColor" stroke-opacity="0.45" stroke-width="1.2">
<rect x="16" y="16" width="250" height="56" rx="8"/>
<rect x="16" y="92" width="250" height="56" rx="8"/>
<rect x="16" y="168" width="250" height="56" rx="8"/>
<rect x="16" y="320" width="250" height="56" rx="8"/>
<rect x="16" y="396" width="250" height="56" rx="8"/>
</g>
<rect x="16" y="244" width="250" height="56" rx="8" stroke-width="2" style="fill:var(--color-accent);fill-opacity:0.1;stroke:var(--color-accent)"/>
<g stroke="currentColor" stroke-width="1.2" marker-end="url(#fc1-arrow)">
<line x1="141" y1="72" x2="141" y2="90"/>
<line x1="141" y1="148" x2="141" y2="166"/>
<line x1="141" y1="224" x2="141" y2="242"/>
<line x1="141" y1="300" x2="141" y2="318"/>
<line x1="141" y1="376" x2="141" y2="394"/>
</g>
<g fill="currentColor" font-size="15" font-weight="600">
<text x="32" y="40">Python kernel</text>
<text x="32" y="116">Fly MLIR passes</text>
<text x="32" y="192">LLVM IR at -O2</text>
<text x="32" y="268">AMDGPU backend</text>
<text x="32" y="344">MC assembler + LLD</text>
<text x="32" y="420">HIP runtime</text>
</g>
<g fill="currentColor" font-size="12" fill-opacity="0.65">
<text x="32" y="59">@flyc.kernel source</text>
<text x="32" y="135">FlyDSL's own C++ passes</text>
<text x="32" y="211">upstream LLVM middle end</text>
<text x="32" y="287">upstream code + 9 FlyDSL patches</text>
<text x="32" y="363">in process since FlyDSL #987</text>
<text x="32" y="439">hipModuleLoadData, launch</text>
</g>
<g fill="currentColor" font-size="13.5">
<text x="290" y="40">decides: algorithm, tiling, instruction family</text>
<text x="290" y="116">decides: layout to address math, swizzles</text>
<text x="290" y="192">decides: unrolling, LICM, CSE, inlining</text>
<text x="290" y="262">decides: instruction order, registers, waits</text>
<text x="290" y="344">decides: encoding and ELF only; ISA unchanged</text>
<text x="290" y="420">loads the code object and launches it</text>
</g>
<g fill="currentColor" font-size="12" fill-opacity="0.65">
<text x="290" y="59">owner: kernel author</text>
<text x="290" y="135">owner: FlyDSL, no LLVM rebuild needed</text>
<text x="290" y="211">owner: upstream LLVM, steered by flags</text>
<text x="290" y="279">owner: LLVM plus FlyDSL patches</text>
<text x="290" y="363">owner: same LLVM, or ROCm ld.lld</text>
<text x="290" y="439">owner: ROCm</text>
</g>
<text x="290" y="296" font-size="12.5" font-weight="600" style="fill:var(--color-accent)">overridden by hand in 62 of 180 kernel files</text>
<g fill="currentColor" font-size="12.5" fill-opacity="0.7">
<text x="16" y="490">Stages 2 to 5 run inside the Python process, in FlyDSL's own MLIR and LLVM</text>
<text x="16" y="510">(upstream llvm-project 941a04e plus 9 patches). MI355X (gfx950) and MI455X</text>
<text x="16" y="530">(gfx1250) take the same path; only the chip, wave size and instruction families change.</text>
</g>
</svg>
</div>
<figcaption style="font-size:0.85em;opacity:0.75;text-align:center">Figure 1. The FlyDSL compile chain and the decision each stage owns. The three decisions that kernel authors override by hand all live in the AMDGPU backend.</figcaption>
</figure>

## What the kernels are telling us

When a kernel author overrides the compiler, either the compiler decided badly or it could not be told what the author wanted. Both leave a trace in the source. These are the explicit overrides in the 180 Python files under `kernels/` (method at the end):

| Override | Occurrences | What the author is doing |
|---|---|---|
| `sched_barrier(0)` | 270 of 279 `sched_barrier` calls, 35 files | forbid the scheduler to move any instruction across this point |
| `sched_mfma` / `sched_dsrd` / `sched_dswr` / `sched_vmem` | 66 lines, 12 files | hand-write the interleave of matrix and memory instructions |
| `s_waitcnt`, `s_wait_*cnt` | 187 lines, 45 files | choose by hand where a wave waits for memory |
| `s_setprio` | 49 lines, 14 files | raise and lower wave priority by hand |
| inline assembly | 29 lines, 12 files | write instructions the compiler cannot analyze |
| `"enable-post-misched": False` | 4 files | switch off scheduling after register allocation |

62 of the 180 files contain at least one of them.

Why would an author forbid the scheduler to move anything? Because the scheduler optimizes a generic objective, trading register pressure against latency and occupancy, without knowing the pipeline the author designed: that a global load must issue several matrix instructions before its data is needed, and that the LDS writes for the next tile belong at the tail of the loop ([`kernel_tuning_guide.md`](https://github.com/ROCm/FlyDSL/blob/aae155359ddc9b2bb1064ec8d391bc25aa4c0d3e/docs/kernel_tuning_guide.md?plain=1#L311-L335)). The hand-written order encodes that pipeline, and the fence protects it from an optimizer that cannot see it. The same reasoning switches off post-RA scheduling in several attention kernels, "to protect a hand-tuned schedule" ([`mllvm-flags.md`](https://github.com/ROCm/FlyDSL/blob/aae155359ddc9b2bb1064ec8d391bc25aa4c0d3e/.claude/skills/llvm/references/mllvm-flags.md)).

Leaving it this way has three costs. An override is paid again for every kernel, every tile shape and every architecture. A full fence also stops the compiler from improving the code around it. And the right override is not even fixed: FlyDSL's tuning guide reports that the hand interleave wins 8–10% on async "fat" tiles and loses on some f16/bf16 tiles with few N accumulators ([`kernel_tuning_guide.md`](https://github.com/ROCm/FlyDSL/blob/aae155359ddc9b2bb1064ec8d391bc25aa4c0d3e/docs/kernel_tuning_guide.md?plain=1#L336-L340)). A decision that changes with the tile should not be written by hand in each file.

The clearest case merged on 16 November. The gfx950 FP4 and FP8 GEMMs kept their matrix accumulators in AGPRs with inline assembly, because the compiler's own pinning primitive, `sched_agpr`, did not hold: the accumulator could land in the wrong register file, and instruction selection picked the VGPR form of the MFMA. On an FP4 8192³ GEMM, the primitive took 674.8 µs against 221.4 µs for the assembly ([#1368](https://github.com/ROCm/FlyDSL/pull/1368)). The fix was two changes in the LLVM backend, after which the primitive matches or beats the assembly. The gap was never in the kernel. It was two compiler decisions, and once they are fixed in the compiler, the next kernel gets in-place AGPR accumulation without writing assembly.

## Four ways the loop breaks

Kernel optimization, by hand or by an agent, is a loop: change the kernel, compile, measure, keep the better version. The loop breaks in four distinct ways, and each points at a different layer of the chain. They are ordered here from the cheapest fix to the most expensive.

### It cannot see

A compile returns a binary or an error. To learn the register count, the spills, the occupancy, or whether a hint took effect, someone dumps about twenty intermediate files and runs a script over the assembly. Some knobs also fail silently. The repository's own LLVM notes record four knobs that never reached code generation, and an unknown `compile_hints` key is accepted without complaint while still changing the cache key, so the recompile looks real ([`llvm` skill](https://github.com/ROCm/FlyDSL/blob/aae155359ddc9b2bb1064ec8d391bc25aa4c0d3e/.claude/skills/llvm/SKILL.md)).

The fix is a structured report from every compile: registers, spills, occupancy, LDS, instruction and wait counts, and which hints reached the IR. LLVM can already emit the resource half as YAML through the `kernel-resource-usage` remark; the rest comes from the ISA that FlyDSL already dumps. Unknown hints should be errors.

This comes first because every other fix needs a cheap signal. GPU timing is expensive and noisy (the same notes measure ±14% clock noise on gfx950), while a static report rejects a spilling candidate in milliseconds. FlyDSL already maps compile errors back to the Python file and line that caused them ([`diagnostics.py`](https://github.com/ROCm/FlyDSL/blob/aae155359ddc9b2bb1064ec8d391bc25aa4c0d3e/python/flydsl/compiler/diagnostics.py#L30-L56)); performance facts get no such treatment yet.

### It cannot choose

When the best schedule template depends on the tile, the choice is a search problem, not a rule. FlyDSL's autotuner searches kernel parameters, the warp count and `waves_per_eu` ([`autotune.py`](https://github.com/ROCm/FlyDSL/blob/aae155359ddc9b2bb1064ec8d391bc25aa4c0d3e/python/flydsl/autotune.py#L130-L149)). LLVM options and template switches are written into kernels by hand. Adding them to the search space, and pruning it with the report above, turns a per-tile guess into a measurement.

Search has a ceiling: it only picks among options that exist. It cannot create the one that is missing.

### It cannot express

270 fences and 187 hand-placed waits are not 457 different ideas. They are a few ideas written again in every kernel, and an idea written hundreds of times is a missing abstraction. Three are visible in the counts:

- **A declared pipeline instead of fences.** The author states the intended schedule once (which loads run how many matrix instructions ahead, where the LDS writes go), and the compiler emits the scheduling hints and checks that the final ISA respects them. A constraint lets the compiler keep optimizing everything else; a full fence does not.
- **Waits derived from data flow.** The Fly level sees every asynchronous copy and the operation that consumes it, so it can compute where the waits must go and check that none is missing. Missing and misplaced waits were a recurring source of silent wrong answers during the gfx1250 bring-up ([Gap 5 in "Green Test, Red Model"](/en/blog/unit-pass-model-fail/)).
- **Register files as an API, not assembly.** [#1349](https://github.com/ROCm/FlyDSL/pull/1349) added `sched_bank` and `sched_agpr`, and #1368 made `sched_agpr` hold.

Why not ask LLVM to do all of this? Because LLVM sees instructions, not pipelines. Stages, producers and consumers exist at the Fly level and are gone by the time the backend schedules. That is also why this layer is cheap to change: the passes live in the FlyDSL repository, a change needs no LLVM rebuild, and each change can be checked kernel by kernel, with the same or better speed and fewer overrides.

### It decides wrong

Some decisions only the backend makes: the final instruction order, register allocation, the MFMA form, wait insertion. #1368 was fixed there. This layer has the highest ceiling, because one fix reaches every kernel, and the highest cost. Each change means rebuilding LLVM, and LLVM changes move kernel performance in ways that are hard to predict: #1349 kept the LLVM pin on purpose instead of pulling in 1,807 upstream commits, "to avoid ... shifting kernel performance". Backend work should therefore go where measurement shows a large gap, and every change must pass a performance gate across the whole kernel set.

### Measuring the ceiling

One more tool turns guesses into numbers. FlyDSL's [`inject_asm`](https://github.com/ROCm/FlyDSL/blob/aae155359ddc9b2bb1064ec8d391bc25aa4c0d3e/docs/testing_benchmarking_guide.md?plain=1#L406-L431) reassembles a hand-edited `.s`, swaps it into the cache and reruns the unchanged test, with a baseline run under the same conditions. Today it is a debugging aid. Driven by an agent or a reinforcement-learning policy that reorders instructions and moves waits, it becomes a search for the best schedule a given binary admits. [CuAsmRL](https://arxiv.org/abs/2501.08071) did this on NVIDIA SASS and improved expert kernels by 9% on average and up to 26%. The result is tied to one binary and one shape, so it is not what you ship. What it gives you is the size of the gap the compiler left, and a concrete diff that says which heuristic to change.

## Letting the compiler improve itself

[CAKE](https://arxiv.org/abs/2608.12629), from researchers at NVIDIA and CMU, turns this into a system. Three commitments carry it. Agents edit a typed IR in which hardware decisions are explicit, not raw CUDA. The compiler answers with localized diagnostics rather than a pass/fail bit: pre-compile gates for synchronization safety, hardware conformance, data consistency and schedule semantics, plus a calibrated cost model that ranks candidates before any GPU run (§3.1). And the harness itself evolves: "an opaque runtime crash becomes a verifier rule, a repeated illegal lowering pattern becomes a static check, a systematic misprediction becomes a calibration target" (§3.2). On a clean-start Flash-KMeans task on B200, the best Cake IR candidate reached a median 1.144× of a tuned baseline, against 0.928× when agents wrote CUDA/PTX directly with the same budget.

Mapped onto FlyDSL, that is two loops:

<figure>
<div style="overflow-x:auto">
<svg viewBox="0 0 680 372" role="img" aria-labelledby="fc2-title" style="width:100%;min-width:600px;height:auto;font-family:inherit">
<title id="fc2-title">The inner kernel-optimization loop and the outer compiler-improvement loop</title>
<defs><marker id="fc2-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="currentColor"/></marker><marker id="fc2-arrow-accent" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--color-accent)"/></marker></defs>
<rect x="6" y="6" width="668" height="150" rx="10" fill="currentColor" fill-opacity="0.04"/>
<rect x="6" y="196" width="668" height="170" rx="10" fill="currentColor" fill-opacity="0.04"/>
<text x="20" y="30" font-size="13" font-weight="600" fill="currentColor" fill-opacity="0.85">Inner loop: optimize one kernel</text>
<text x="20" y="356" font-size="13" font-weight="600" fill="currentColor" fill-opacity="0.85">Outer loop: improve the compiler</text>
<g fill="none" stroke="currentColor" stroke-opacity="0.45" stroke-width="1.2">
<rect x="20" y="74" width="145" height="56" rx="8"/>
<rect x="185" y="74" width="145" height="56" rx="8"/>
<rect x="350" y="74" width="145" height="56" rx="8"/>
<rect x="515" y="74" width="145" height="56" rx="8"/>
</g>
<g stroke-width="1.6" style="fill:var(--color-accent);fill-opacity:0.1;stroke:var(--color-accent)">
<rect x="350" y="262" width="145" height="56" rx="8"/>
<rect x="185" y="262" width="145" height="56" rx="8"/>
<rect x="20" y="262" width="145" height="56" rx="8"/>
</g>
<g fill="none" stroke="currentColor" stroke-width="1.2" marker-end="url(#fc2-arrow)">
<path d="M165,102 H183"/>
<path d="M330,102 H348"/>
<path d="M495,102 H513"/>
<path d="M587,74 V54 H92 V72"/>
<path d="M422,130 V260"/>
</g>
<g fill="none" stroke-width="1.4" marker-end="url(#fc2-arrow-accent)" style="stroke:var(--color-accent)">
<path d="M350,290 H332"/>
<path d="M185,290 H167"/>
<path d="M92,262 V132"/>
</g>
<g fill="currentColor" font-size="14" font-weight="600">
<text x="34" y="98">Compile</text>
<text x="199" y="98">Report</text>
<text x="364" y="98">Measure</text>
<text x="529" y="98">Edit kernel</text>
<text x="364" y="286">Record, group</text>
<text x="199" y="286">Fix</text>
<text x="34" y="286">Corpus gate</text>
</g>
<g fill="currentColor" font-size="11.5" fill-opacity="0.7">
<text x="34" y="117">MLIR + LLVM</text>
<text x="199" y="117">spills, waits, hints</text>
<text x="364" y="117">idle GPU, repeats</text>
<text x="529" y="117">agent or engineer</text>
<text x="364" y="305">failures, overrides</text>
<text x="199" y="305">cheapest layer first</text>
<text x="34" y="305">every kernel passes</text>
</g>
<g fill="currentColor" font-size="12" fill-opacity="0.75">
<text x="340" y="48" text-anchor="middle">repeat, keep the best version</text>
<text x="432" y="181">failures, overrides, gaps</text>
<text x="102" y="181">merged: every later kernel benefits</text>
<text x="257" y="340" text-anchor="middle">option → Fly pass → LLVM patch</text>
</g>
</svg>
</div>
<figcaption style="font-size:0.85em;opacity:0.75;text-align:center">Figure 2. The inner loop optimizes one kernel. The outer loop turns what the inner loop runs into (failures, hand overrides, measured gaps) into compiler changes, and merges only the changes that every kernel passes.</figcaption>
</figure>

FlyDSL already has most of the pieces:

| CAKE commitment | FlyDSL today | Missing |
|---|---|---|
| hardware-explicit IR | atoms, layouts and scheduling primitives are explicit | errors in layout code are hard to localize |
| localized diagnostics | compile errors map to Python lines | structured performance diagnostics |
| evolving harness | repository skills, an LLVM patch pipeline, CI that benchmarks against `main` and the latest tag ([`flydsl.yaml`](https://github.com/ROCm/FlyDSL/blob/aae155359ddc9b2bb1064ec8d391bc25aa4c0d3e/.github/workflows/flydsl.yaml)) | a log of failures and overrides; a compiler-change gate across the kernel set |

One design difference deserves attention. CAKE deliberately does not make layout a first-class abstraction (§2.2), because a layout calculus makes agent errors likely and hard to localize. FlyDSL is built on a layout algebra. For FlyDSL, then, the most valuable diagnostic is one that turns a layout error into a repair an agent can make.

The corpus gate is not bureaucracy. CAKE notes that a new verifier rule without corpus validation can reject valid kernels, and that a new primitive without legality rules makes the IR harder to analyze (§3.2). A compiler change touches every kernel, so it has to be judged on every kernel.

What gets automated, and how, depends on what is being changed:

| Automation | What it changes | Example |
|---|---|---|
| search | options and kernel parameters | FlyDSL's autotuner |
| learned heuristic | one decision inside the compiler | [MLGO](https://arxiv.org/abs/2101.04808) replaced LLVM's inlining-for-size heuristic with a trained policy: up to 7% smaller code |
| LLM-evolved code with an automatic evaluator | the source of a kernel or a compiler heuristic | [AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) sped up a Gemini kernel by 23%, and FlashAttention by up to 32.5% at the GPU-instruction level |
| reinforcement learning on the ISA | the final instruction order | CuAsmRL: 9% on average, up to 26% |

All four need the same thing first: an evaluator you can trust, meaning correctness tests plus timing on an isolated GPU with enough repeats to beat the noise. Without it, automation optimizes noise.

## A first experiment: a compiler-gap census

"Which part should we change?" stays vague until the gaps are measured. A census answers it:

1. Pick three to five important kernels on MI355X, where CI runners exist: a GEMM, an attention kernel and a MoE kernel.
2. Build each one three ways: as written; with the hand overrides removed and a sweep of LLVM scheduler strategies; and, optionally, with the best schedule that an `inject_asm` search finds.
3. Attribute each performance gap to one stage: scheduling, register allocation, waits, instruction selection or address arithmetic.
4. Tabulate each gap class: how much performance it costs, in how many kernels, and which layer would fix it.

The census is done when that table answers "which layer first, and for how much". The structured compile report from the first fix is the instrument that makes the census cheap, so the two can start together.

## Closing

"Think about the compiler chain" becomes concrete once you ask, for each decision that sets a kernel's speed, who made it and whether anyone can see it. In FlyDSL the kernels answer for themselves: scheduling, registers and waits are decided by hand in 62 of 180 files. Each override is a bug report against the compiler that nobody filed. The outer loop is a way to file them: record them, group them, fix each one at the cheapest layer that can express it, and merge only what the whole kernel set agrees with.

## Method

The counts are static matches in `kernels/` at [`aae1553`](https://github.com/ROCm/FlyDSL/tree/aae155359ddc9b2bb1064ec8d391bc25aa4c0d3e). They include a few comments, so read them as an indicator, not an exact call count:

```bash
git clone https://github.com/ROCm/FlyDSL.git && cd FlyDSL
git checkout aae155359ddc9b2bb1064ec8d391bc25aa4c0d3e
rg --files -g '*.py' kernels | wc -l               # 180 kernel files
rg -o 'sched_barrier\(\s*0\s*\)' kernels | wc -l   # 270 full fences
rg -o 'sched_barrier\(' kernels | wc -l            # 279 sched_barrier calls
rg -c 's_waitcnt|s_wait_[a-z]+cnt' kernels         # hand-placed waits, per file
rg -l 'sched_barrier|sched_group_barrier|sched_(mfma|dsrd|dswr|vmem)\(|s_setprio|s_waitcnt|s_wait_[a-z]+cnt|inline_asm|InlineAsm' kernels | wc -l   # 62 files
```

The 8–10% and ±14% figures come from FlyDSL's own documentation and were not re-measured for this post. None of the directions above has been tried yet; the census is the first step that would produce numbers.
