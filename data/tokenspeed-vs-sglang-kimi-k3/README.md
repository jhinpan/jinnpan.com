# TokenSpeed vs SGLang — Kimi-K3 on 8× MI355X

Source of record for **Experiment 003** (`/sources/tokenspeed-vs-sglang-kimi-k3.html`).

Serving Kimi-K3 with TokenSpeed on an 8× MI355X ROCm box and comparing it against the
existing SGLang deployment on the same node — end to end, and kernel by kernel.

**Headline:** In graph-off attribution, TokenSpeed has 17.6% less non-collective
device-kernel time and 30% fewer total launches. Four operation-level matched groups are faster;
the routed-expert GEMM row is slower but confounded by EP8/a16w4 versus TP8/a8w4.
TokenSpeed has 187 Iris layer all-reduces plus 2 RCCL dispatches; SGLang has 187
measured collectives. In a separate graph-on serving measurement TokenSpeed is 1.14x
slower at batch 1 and about 3.7x slower at concurrency 8/32. These measurements
prioritize the host/scheduler/overlap path, but do not yet localize the cause.

---

## Hardware and software

| | |
|---|---|
| Node | 8× AMD Instinct MI355X · gfx950 · 288 GiB HBM3E each |
| ROCm | 7.2 |
| Model | `moonshotai/Kimi-K3` — 2.78 T total / 105.4 B active · 93 layers (69 KDA + 24 full NoPE-MLA) · 896 routed experts top-16 + 2 shared · `situ` activation · routed experts `compressed-tensors` MXFP4 group-32, rest BF16 |
| TokenSpeed | `lightseekorg/tokenspeed` @ `d50bb481` · torch 2.11.0+rocm7.2 · own venv |
| SGLang | `DarkSharpness/sglang-kimi` @ `3d35b45f7` · torch 2.9.1+rocm7.2.0 |
| Measured | 2026-07-31 |

The SGLang kernel baseline is reused from **Experiment 002**
(`data/kimi-k3-kda-time-share/`), captured on the same node with the same model.

## Configuration — best-config vs best-config

Each stack runs its own recommended setup, so this is not a controlled A/B.

| Axis | TokenSpeed | SGLang |
|---|---|---|
| MoE parallelism | TP8 + EP8 | pure TP8 |
| Attention backend | `mla` (Gluon gfx950) | `triton` |
| MoE backend | `auto` → Gluon SiTU | AITER (`AITER_SITUV2_A8W4=1`) |
| KV cache | fp8 | bf16 |
| Scheduler | FlatKV (C++ FSM) | radix, disabled |
| Prefix caching | off | off |
| Speculative decoding | none | none (`nospec`) / DSpark |

`--kv-cache-dtype fp8` favours TokenSpeed, so it cannot explain any deficit found.

### TokenSpeed server profiles actually used

The serving table is not produced by one universal server boot. `MAXLEN` and
`MAXSEQS` are the only changing server parameters:

| Profile | `--max-model-len` | `--max-num-seqs` | Workloads | Reported `max_total_num_tokens` |
|---|---:|---:|---|---:|
| A | 8192 | 32 | 4K c1; 1K c1/8/32; kernel traces | 4,466,304 |
| B | 8192 | 128 | 1K, concurrency 32/64/128 | 3,556,992 |
| C | 16384 | 128 | 8K+1K, concurrency 1/128 | 3,556,992 |

The 8K-input/1K-output workload requires Profile C because total request length is
9,216 tokens. Raising `max-num-seqs` also reserves more graph state, which is why
Profiles B/C report a smaller token pool than Profile A.

> **Benchmark flag trap.** `--random-range-ratio` is inverted between the two harnesses.
> SGLang's `1` means exact lengths; TokenSpeed validates into `[0, 1)` and `0` means
> exact lengths. Both sides here use exact lengths.

## Two deviations from TokenSpeed's documented AMD recipe

1. `--disable-prefill-graph` — the documented recipe OOMs during prefill graph capture
   (free memory drains 19.9 GB → 0.03 GB, dies at 751 s). TokenSpeed's own MI35x CI perf
   config already passes this flag; `docs/recipes/models.md` omits it.
2. `--no-enable-prefix-caching` — matches SGLang's `--disable-radix-cache` baseline.

## Method — kernel traces

Both sides: batch 1, ISL 4096, TP8, no speculative decoding, **graph off**
(`--enforce-eager`), device-kernel duration summed per decode step, **collectives
excluded**.

- *Graph off* because individual kernels are not separately attributable inside a
  replayed HIP graph. The SGLang baseline was captured graph-off for the same reason.
- *Collective durations excluded* because in eager decode the all-reduce payload is a
  few KB and the kernel busy-waits, so its duration measures rank skew, not work
  (67.6 ms/step of apparent waiting against a 17.1 ms non-collective budget). Dispatch
  identities and counts are still reported: TokenSpeed has 187 Iris layer all-reduces
  plus 2 RCCL calls; SGLang has 187 measured collectives.
- *Proton was not usable*: on ROCm it calls `rocprofiler_force_configure`, which must run
  before HIP initialises, so attaching to a live server fails with `error 16` on every
  rank. The torch/roctracer profiler attaches at runtime and emits the same chrome-trace
  artifact the SGLang baseline used.

**Attribution is by launching source module, not name matching.** Name matching is wrong
here: `iris_stage_one_shot_allreduce_two_gluon_kernel` contains "gluon" but is a
collective; `_stage1_a16w4_situ_warp_gemv` contains "stage1" but is MoE.

Two kernels genuinely span blocks and are split by dispatch share:

- `_kimi3_projection_gemv_kernel` — 4 call sites per MoE layer (two latent H↔L
  projections, shared-expert down projection, router) plus `kimi3_qkvfab_projection` in
  `KimiLinearKDA`: 4 × 92 + 69 = **437**, matching the measured dispatch count exactly.
  Split 368 MoE / 69 KDA.
- The 93-dispatch vendor GEMM — the attention output projection, issued once per layer.
  Split 69 KDA / 24 MLA.

## Results

### Matched kernel groups (decode, bs 1, ISL 4096, µs/step)

| Kernel group | SGLang | TokenSpeed | Ratio |
|---|---:|---:|---|
| MLA attention math, 24 layers | 1212 | 438 | TokenSpeed 2.77x faster |
| MoE routing: top-16 of 896 + sort | 2956 | 454 | TokenSpeed 6.51x faster |
| Attention residual epilogue | 2288 | 1853 | TokenSpeed 1.23x faster |
| KDA recurrence + short conv, 69 layers | 744 | 650 | TokenSpeed 1.14x faster |
| Routed-expert quantized GEMMs | 1437 | 2391 | TokenSpeed 1.66x slower |

The expert-GEMM row is confounded: TokenSpeed runs EP8 (`a16w4`), SGLang pure TP8
(`a8w4`, half the activation bytes). Different shapes, different activation precision.

### Block composition (decode, bs 1, ISL 4096, µs/step)

| Block | TokenSpeed | SGLang | Δ |
|---|---:|---:|---:|
| MoE | 8085 | 10549 | −2464 |
| KDA linear attention | 2905 | 3542 | −637 |
| Full MLA attention | 1468 | 3100 | −1632 |
| Attention residual | 1865 | 2288 | −423 |
| Glue (norms, adds, casts, copies) | 2786 | 1290 | +1496 |
| **Total device compute** | **17108** | **20769** | **−3661** |

### The reconciliation

| Metric | SGLang | TokenSpeed |
|---|---:|---:|
| Device compute / decode step | 20769 µs | 17108 µs |
| Kernel launches / decode step | 3372 | 2375 |
| Collective dispatches / decode step | 187 | 189 (187 Iris + 2 RCCL) |
| End-to-end step, graph ON | 19.39 ms | 22.19 ms |

The first three rows are graph-off attribution; the final row is graph-on serving.
Fewer launches rule out excess launch **count**, not higher per-launch host cost,
synchronization, replay differences, collective latency, or poor overlap. A synchronized
host+GPU graph-on trace is required before assigning the deficit to a component.

### End-to-end serving (output tok/s, 8-GPU aggregate, exact lengths)

| ISL / OSL | Conc | TokenSpeed | SGLang nospec | SGLang DSpark |
|---|---:|---:|---:|---:|
| 1024 / 1024 | 1 | 44.73 | 51.40 | 109.84 |
| 1024 / 1024 | 8 | 84.97 | 311.79 | 472.00 |
| 1024 / 1024 | 32 | 226.80 | 847.87 | 949.73 |
| 1024 / 1024 | 64 | 440.58 | — | — |
| 1024 / 1024 | 128 | 487.19 | — | — |
| 4096 / 1024 | 1 | 43.98 | — | 89.30 |
| 8192 / 1024 | 1 | 44.55 | — | 74.29 |
| 8192 / 1024 | 128 | **failed** | 890.25 | — |

Sanity anchor: TokenSpeed's own CI reference for 4096/1024 at concurrency 1 is
43.05 tok/s per user on 8× MI350X (gated at 42,
`test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml`). We measured
43.98 on MI355X — 2% above, so the build is healthy.

### The 8K × 128 wedge

1 of 128 requests completed in 3600 s. FlatKV page pool saturates at 0.94 occupancy,
the scheduler logs **9,256** `flat retract ... to unwedge the pool` events, decode falls
to 0.02 tok/s with 106 requests queued. The engine reports
`max_total_num_tokens=3556992` while the scheduler config says `num_device_pages=3298`
at `block_size=128`, shared across four cache groups (`full_attention` + three
`linear_attention`, because 69 of 93 layers are KDA). SGLang on the same workload
reports a 1.29 M-token pool and zero retractions.

`--max-model-len 8192` in the documented recipe is not a hard ceiling — 16384 boots
cleanly (`max_req_input_len=16383`).

---

## Layout

```
results/
  block-comparison.csv                          block-level TokenSpeed vs SGLang + reconciliation rows
  launches-per-step.csv                         TokenSpeed kernel launches per block per decode step
  tokenspeed-kernels-attributed.csv             every TokenSpeed kernel with its block and source module
  tokenspeed-decode-block-composition.csv       raw block rollup, mean over 8 ranks
  tokenspeed-decode-kernels-cross-rank.csv      per-kernel mean/slowest-rank µs per step
  tokenspeed-decode-kernels-by-rank.csv         per-rank, per-kernel totals (8 ranks)
  serving/                                      tokenspeed bench serve JSON, one per workload point
scripts/
  ts-build.sh, ts-build-step6.sh                ROCm build into an isolated venv (step 6 skips mooncake)
  ts-bench.sh                                   serve + end-to-end benchmark sweep (takes /tmp/k3-grid.lock)
  ts-profile.sh                                 serve eager + arm torch GPU profiler + capture 8-rank traces
  ts_trace_report.py                            chrome trace -> per-kernel CSV
  ts_vs_sgl_kernels.py                          attribution + the TokenSpeed vs SGLang comparison tables
```

The 16 raw per-rank chrome traces (~6 MB each for DECODE) are **not** committed. They are
regenerated by `scripts/ts-profile.sh`; everything derived from them is in `results/`.

## Reproducing

```bash
# build (isolated venv — ts-build.sh reaches the known Mooncake failure at step 6;
# ts-build-step6.sh installs the dependency set without Mooncake and finishes runtime)
bash scripts/ts-build.sh
bash scripts/ts-build-step6.sh
pip uninstall -y triton && pip install --force-reinstall --no-deps \
  triton-rocm==3.6.0 --index-url https://download.pytorch.org/whl/rocm7.2

# Profile A — 1K/4K, concurrency 1–32
MAXLEN=8192 MAXSEQS=32 \
WORKLOADS="4096:1024:1:3:1 1024:1024:1:4:1 1024:1024:8:16:2 1024:1024:32:64:4" \
  bash scripts/ts-bench.sh

# Profile B — 1K, high concurrency
MAXLEN=8192 MAXSEQS=128 \
WORKLOADS="1024:1024:32:64:4 1024:1024:64:64:4 1024:1024:128:128:8" \
  bash scripts/ts-bench.sh

# Profile C — 8K input + 1K output
MAXLEN=16384 MAXSEQS=128 \
WORKLOADS="8192:1024:1:2:1 8192:1024:128:128:8" \
  bash scripts/ts-bench.sh

# kernel traces (graph off, GPU activities — Proton cannot attach on ROCm)
ISL=4096 NUM_STEPS=16 bash scripts/ts-profile.sh
python3 scripts/ts_trace_report.py <run>/traces/*-DECODE.trace.json.gz \
  --steps 16 --out <run>/report-decode
python3 scripts/ts_vs_sgl_kernels.py <run>/report-decode/kernels-cross-rank.csv <run>/compare
```

`ts-bench.sh` and `ts-profile.sh` both take `/tmp/k3-grid.lock` with `flock`, the same
mutex every other harness on this node uses, so they will refuse to start while another
run holds the box.
