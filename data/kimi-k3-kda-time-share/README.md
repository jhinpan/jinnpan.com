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

## Verifying the traces with Perfetto

Opened in Perfetto, the 32K prefill trace makes full attention look negligible,
which does not match the 37.6% reported here. It is worth reproducing the check
that resolved that, because it is the kind of doubt this data should be able to
answer without appealing to its own tooling.

`scripts/verify/` recomputes the published numbers along paths that share no code
with `bucketize.py`. Perfetto's `trace_processor` is a parser we did not write:

```bash
pip install perfetto              # downloads trace_processor_shell on first use
gunzip -c prefill-32k-TP0.trace.json.gz > pf32k.json
python3 scripts/verify/perfetto_verify.py pf32k.json
```

It reports the GPU timeline, kernel time by name, the `_fwd_kernel` dispatches one
by one, and — the part that matters below — the same `K3/*` ranges measured on the
host thread and on the device.

```bash
# rebuild prefill-composition.csv from the traces, two ways, against the published file
python3 scripts/verify/verify_composition.py traces/ --csv results/prefill-composition.csv

# split the 32K trace by chunked-prefill chunk
python3 scripts/verify/verify_chunks.py traces/pf32k/pf32k-TP-0.trace.json.gz

# ground-truth counts and launch geometry, straight from the chrome JSON
python3 scripts/verify/raw_totals.py traces/pf32k/pf32k-TP-0.trace.json.gz
```

**The composition holds.** The correlation-id walk and a second attribution built
from the `gpu_user_annotation` ranges the PyTorch profiler itself projects onto the
device timeline agree at every point: 37.56% versus 37.57% for full attention at
32K, and within 0.9 points everywhere else. Kernel by kernel the two disagree on
18.6 ms of 8961 dispatches (0.6%). Summing durations is safe here — the union of
all kernel intervals is 3051.00 ms against a naive sum of 3052.04 ms, so 0.034% of
the total runs concurrently.

**Two things make it look otherwise in the UI.**

The first is which track you are reading. Perfetto draws `K3/*` on both the CPU
thread and the GPU stream, and they answer opposite questions:

| range | CPU thread | | GPU device | |
|---|---|---|---|---|
| `K3/kda` | 1077.6 ms | 85.3% | 546.1 ms | 20.8% |
| `K3/moe` | 154.2 ms | 12.2% | 1035.5 ms | 39.5% |
| `K3/full_attn` | 31.1 ms | 2.5% | 1032.0 ms | 39.4% |

The host runs ahead of the device during prefill and blocks wherever the launch
queue happens to fill; with the GPU 99.8% busy across the 3.06 s span it is stalled
almost throughout. Host range width measures where it stalled. Every number in this
archive uses device-side kernel duration.

The second is that a 32K prefill is two 16384-token chunks and they do not look
alike. Chunk 1 has no prefix; chunk 2 attends to all of chunk 1.

| | chunk 1 (0.00–1.14 s) | chunk 2 (1.14–3.06 s) |
|---|---|---|
| full attention | 120.6 ms, 10.6% | 911.8 ms, 47.6% |
| MoE | 526.5 ms, 46.4% | 524.1 ms, 27.3% |
| KDA | 276.8 ms, 24.4% | 272.1 ms, 14.2% |

MoE and KDA barely move — both are linear in token count and both chunks carry
16384 tokens. Only attention changes, and 37.6% is the average of 10.6% and 47.6%.
`_fwd_kernel` occupies 6.1% of chunk 1's wall clock and 45.4% of chunk 2's.

**One caveat about the tool itself.** Perfetto's Chrome-JSON importer silently
drops slices it cannot place by nesting on a single track — 2 of 8961 dispatches
here, one of them a 35.4 ms `_fwd_kernel`. It therefore reports 47 dispatches at
880.73 ms where the truth is 48 at 916.14 ms, which is 24 MLA layers times 2
chunks. Use it for queries and structure; use the raw JSON for totals.

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

3. **The 8K→32K attention jump was misread as kernel inefficiency.** The first pass
   noted that MLA prefill attention grows faster than quadratically past one chunk
   and attributed part of it to the kernel. `chunk_diff.py` shows that is wrong:
   splitting the 32K trace at the forward-pass boundary, every kernel is identical
   between the two chunks except attention, and the recorded launch grids
   (`[1,12,128]` versus `[1,12,256]`, same 16384 extend tokens, so `BLOCK_M` 128
   versus 64) pin `Lq` at 192 versus 576. The two chunks run *different MLA forms* —
   decompressed MHA without a prefix, absorbed latent with one. Chunk 2 does 10.2x
   the FLOPs at 1.26x lower efficiency, which is the 12.8x. See
   `results/chunk-split-analysis.txt` and `scripts/chunk2_flops.py`.

   The follow-on: chunked prefix cache is the mechanism that would run the prefix
   part in the decompressed form, and `maybe_disable_chunked_prefix_cache` turns it
   off at load time because `triton` is not in
   `CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS`. The intended ablation
   (`--disable-chunked-prefix-cache`) would have been a no-op; the server log never
   prints "Chunked prefix cache is turned on".

4. **`rocprofv3` could not profile the server.** Per-dispatch interception across
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
