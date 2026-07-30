# Kimi-K3 KDA / attention / MoE time attribution — raw data archive

Source of record for Experiment 002. Nothing here is built by Astro (`data/` sits
outside `src/` and `public/`); this directory exists so the measurements and their
provenance survive the machine they were taken on.

## The question

Kimi-K3's text stack is 93 layers: 69 Kimi Delta Attention (KDA) layers and 24
full-attention (MLA) layers, laid out so that three of every four layers are KDA.
Three quarters of the layers is a share of *parameters and layers*, not a share of
*time*. This measures the latter, per layer type, across context length, in both
decode and prefill.

## Hardware and software

| | |
|---|---|
| Node | 1x 8 AMD Instinct MI355X (gfx950), 288 GB HBM3E each |
| Model | `moonshotai/Kimi-K3`, 93 layers, 896 experts (16 routed + 2 shared), MXFP4 experts / bf16 activations |
| Serving | SGLang `0.5.15.post1.dev20260723+g6c9fd0adc5`, TP8, `--attention-backend triton`, `--disable-radix-cache` |
| Linear attention | `linear_attn_backend=triton`, `mamba_backend=triton`, SSM state fp32 |
| ROCm | 7.2.0; aiter `dcd204ea`; `rocprof-trace-decoder` 0.1.6 installed for ATT decode |
| Dates | 2026-07-30 |

Decode and prefill were both measured with **no speculative decoding** and
**batch size 1**. That is the realistic operating point at long context (at 1M the
KV cache barely fits one sequence) and it removes two confounds — the draft model
and the accept length — that the deployed DSpark recipe would otherwise introduce.

## What was measured

| File | Phase | Points | Contents |
|---|---|---|---|
| `results/decode-composition.csv` | decode | 4K, 32K, 64K, 512K, 1M | ms/step by block type, device total, measured CUDA-graph latency |
| `results/decode-mechanism.csv` | decode | same | per-layer cost, KDA recurrence cost, MLA KV bytes and achieved bandwidth |
| `results/decode-block-internals.csv` | decode | same | every component inside each block (150 rows) |
| `results/prefill-composition.csv` | prefill | 1K, 4K, 8K, 32K | TTFT, throughput, ms by block type, per-token and per-layer cost |
| `results/prefill-mechanism-kernels.csv` | prefill | same | the one kernel carrying each mechanism |
| `att/instruction-stats.csv` | — | 1 kernel | per-instruction hitcount / latency / stall for the KDA decode kernel |
| `att/rcv-ui-output/` | — | 1 dispatch | decoded thread trace, opens directly in ROCprof Compute Viewer |

`results/*.json` are the unreduced intermediates the CSVs were rolled up from,
including `name_map_4k.json` (the ground-truth kernel-name → block-type map).

The raw chrome traces are too large to sit in this repo (52 MB for rank TP0 alone,
440 MB for all eight ranks) and live in a public gist instead:
<https://gist.github.com/jhinpan/fe0f4eefa9b302d16db77c1414a0dcce>. That gist also
carries `kernel-inventory.csv` — every distinct kernel, which block launches it, its
dispatch count per forward pass, and whether two blocks share it — plus
`overlap_check.py`, which tests whether KDA and full-attention kernels ever execute
concurrently. They do not, in either phase: layers are serial and each holds exactly
one attention, which is why `kimi_k3.py:1940` can hand the KDA path and the MLA gate
path the same alt stream.

## How a kernel was attributed to a layer type

This is the part that took the work, so the reasoning is recorded rather than just
the outcome.

Kernel names cannot do it alone. The Tensile GEMM names encode tile shapes, not
semantics, so a KDA `q_proj` and a shared-expert GEMM can carry byte-identical
names. `sglang-k3-profiler-ranges.patch` therefore adds opt-in
`torch.profiler.record_function` ranges (`K3/kda`, `K3/full_attn`, `K3/moe`) around
the three cost centres of `KimiK3DecoderLayer`, gated behind
`SGLANG_K3_PROF_RANGES` and compiled out to a no-op class otherwise.

For **prefill** that is the whole method: prefill never runs through a CUDA graph
in this configuration, so the profiler sees every kernel and the ranges attribute
them directly.

For **decode** it is not, because the decode forward *is* captured in a HIP graph
and the torch profiler cannot see kernels replayed from inside one — a decode trace
of the real server showed only the eager pre/post work. So the ranges were used
once, in eager mode, to *learn* the map from kernel name to block type
(`name_map.py` → `results/name_map_4k.json`), and that map was then applied to
graphs-on traces. 81% of non-collective dispatches have a name unique to one block;
the only genuinely shared compute kernel is the 1536→7168 `o_proj`, which KDA and
MLA issue 69 and 24 times per step with identical shapes, so splitting it 69:24 is
exact rather than approximate.

Every context point re-checks the structural invariants before its numbers are
used: exactly 69 KDA recurrence dispatches, 24 MLA stage-1 dispatches and 92
expert-GEMM dispatches per decode step, with an integral per-step count for every
mapped name. All five decode points pass with no unmapped kernel names.

## Three ways the measurement lied, and the corrections

1. **Eager-mode collectives are not communication.** With CUDA graphs off, each
   decode step issues 3372 kernels and per-rank launch jitter leaves aiter's
   all-reduce kernels spinning on their peers; the 187 collectives per step
   accumulate 73–121 ms of apparent duration against a 14 KB payload. They are
   excluded from the decode composition. The exclusion is calibrated: measured
   CUDA-graph latency divided by the eager compute-kernel sum is a steady 0.93–0.97
   across all five points, so real collectives plus gaps net out near zero. Prefill
   is the opposite case — a 16384-token chunk all-reduces 235 MB per layer and the
   surrounding kernels are millisecond-scale — so prefill collectives are reported,
   not dropped.

2. **The probe changed the thing it measured.** A first pass put the GPU at 29%
   busy during a 1K prefill and read that as host-launch-bound. It was the ranges:
   with them compiled in, 4K TTFT was 692 ms versus 333 ms without, while 32K was
   ~3.05 s either way. `prefill_ttft.py` re-measures TTFT on a server built without
   the annotations; summed kernel time then lands within 1% of TTFT at every size.
   Kernel *durations* are unaffected by the ranges, so the composition numbers were
   valid throughout — only the wall-clock column moved.

3. **`rocprofv3` could not profile the server.** Per-dispatch interception across
   all 8 TP ranks drove the scheduler past its 300 s watchdog with no output, and
   runtime `--attach` loaded but produced nothing. The ATT trace therefore comes
   from `kda_micro.py`, a standalone reproduction at the server's exact per-GPU
   shapes. That substitution is equivalent rather than approximate here: the KDA
   decode kernel reads a fixed `[128 x 128]` state per head and nothing in its
   launch geometry depends on context length.

## Reproducing

```bash
# decode composition (eager server, name-map attribution)
GRAPH=off RANGES=0 bash scripts/serve-prof.sh
bash scripts/sweep2.sh ctx

# decode latency anchor (same config, CUDA graphs on)
GRAPH=on RANGES=0 bash scripts/serve-prof.sh
python3 scripts/rp3_drive.py --out graphon_itl.json --max-new 40

# prefill composition (ranges on — they attribute directly)
GRAPH=on RANGES=1 bash scripts/serve-prof.sh
bash scripts/sweep-prefill.sh pf

# prefill wall-clock reference (ranges off — see correction 2)
GRAPH=on RANGES=0 bash scripts/serve-prof.sh
python3 scripts/prefill_ttft.py --out prefill_ttft.json

# roll-ups
python3 scripts/summarize.py --out summary.json
python3 scripts/derive.py --out derived.json
python3 scripts/summarize_prefill.py --out prefill_summary.json
```

The ATT trace, which needs `--att-shader-engine-mask 0xFF` because the kernel's 48
workgroups never land on the default target CU:

```bash
rocprofv3 --att --att-library-path /opt/rocm/lib --att-target-cu 0 \
  --att-shader-engine-mask 0xFF --att-gpu-index 0 --att-consecutive-kernels 8 \
  --kernel-include-regex "fused_recurrent_kda_packed_decode_kernel" \
  -d att_kda_64k -o kda_decode_64k -- python3 scripts/kda_micro.py --iters 12 --no-timing
```

## Caveat on precision

Each decode point is the mean of 24 consecutive steps on rank TP0, armed only
after prefill produced its first token, so the window is steady-state decode.
Prefill TTFT is the best of three after a warm first call, since aiter selects a
configuration on the fly for GEMM shapes it has not seen — the cold call runs
roughly 2x the warm one at 1K and 4K.

Composition percentages are stable to well under a point; the numbers most worth
treating as approximate are the achieved-bandwidth figures, which divide a measured
kernel time into an analytically computed byte count (24 layers x 576 dims x 2 B
per token for the MLA latent cache) and so inherit any error in that model of what
the kernel actually reads.

## `canvas/`

Archived copy of the Cursor Canvas source that presented this data. Kept as a
backup and as the input for the web rendering; it is **not** built by this site and
will not render from here. Cursor only renders `.canvas.tsx` from its managed
directory (`~/.cursor/projects/<workspace>/canvases/`), which is outside any repo —
hence this archive. It imports from `cursor/canvas`, which ships types only (no
runtime), so it cannot be bundled without substituting an implementation of those
components.
