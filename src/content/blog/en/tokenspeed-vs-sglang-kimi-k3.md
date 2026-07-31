---
title: "Serving Kimi-K3 with TokenSpeed on MI355X — and why faster kernels lost"
description: "Building TokenSpeed from source on an 8x MI355X ROCm box, serving Kimi-K3, and profiling it against SGLang kernel by kernel. TokenSpeed's Gluon kernels win every block — MLA decode by 2.77x, MoE routing by 6.5x — and it is still slower end to end. The four build pitfalls, the throughput numbers, and what the traces actually say."
date: 2026-07-31
tags: ["benchmark", "TokenSpeed", "SGLang", "Kimi-K3", "MoE", "AMD", "GPU-kernels", "inference", "MLSys"]
category: "Technical"
lang: "en"
---

TokenSpeed is LightSeek's inference engine, MIT-licensed and public since May 2026, and it advertises Kimi-K3 on AMD gfx950 as a first-class path: there is a `Dockerfile.amd`, native Gluon kernels for the MoE and MLA decode, and per-commit CI on an 8x MI35x runner. We already had a heavily instrumented SGLang deployment of the same model on the same box, so the obvious question was how the two compare — not just in tokens per second, but kernel by kernel.

The short version is that TokenSpeed's kernels are genuinely faster. Its Gluon MLA decode beats SGLang's Triton attention by **2.77x**, its fused router beats AITER's routing pipeline by **6.5x**, and its total device compute per decode step is **1.21x** lower. It also issues 30% fewer kernel launches. And it is still **1.14x slower** end to end at batch 1, **3.7x slower** at concurrency 8, and it does not survive the 8K-input workload at concurrency 128 at all.

That gap between "faster kernels" and "slower engine" is the whole story, and it is not where I expected to find it.

## 1. The setup

One node, 8x AMD Instinct MI355X (gfx950, 288 GiB each), ROCm 7.2. The model is `moonshotai/Kimi-K3` — 2.78 T total parameters, 105.4 B active, 93 layers of which **69 are KDA linear attention and 24 are full NoPE-MLA**, 896 routed experts with top-16 plus 2 shared, `situ` activation, routed experts stored as `compressed-tensors` MXFP4 (4-bit, group size 32) with everything else in BF16.

The two stacks run their own recommended configuration, which means the comparison is best-config against best-config rather than a controlled A/B:

| | TokenSpeed | SGLang |
| --- | --- | --- |
| MoE parallelism | TP8 + EP8 | pure TP8 |
| Attention backend | `mla` (Gluon gfx950) | `triton` |
| MoE backend | `auto` → Gluon SiTU | AITER (`AITER_SITUV2_A8W4=1`) |
| KV cache | fp8 | bf16 |
| Scheduler | FlatKV (C++ FSM) | radix, disabled for these runs |
| Prefix caching | off | off |

The fp8 KV cache is worth flagging because it works *in TokenSpeed's favour* — less bandwidth per token — so it cannot explain any deficit we find.

> **A trap that would have silently ruined the comparison.** Both benchmark harnesses descend from vLLM's `benchmark_serving.py` and emit identically-named metrics, but `--random-range-ratio` is inverted between them. SGLang's `1` means exact lengths; TokenSpeed's valid domain is $[0, 1)$ and `0` means exact lengths. Passing "1" to both would have compared exact lengths against a crash. Every number below uses exact lengths on both sides.

## 2. Four pitfalls between `git clone` and a running server

None of these are exotic. All four are load-bearing, and the documented recipe hits three of them.

### 2.1 A pinned dependency that was never published

`python/pyproject.toml` on `main` pins `tokenspeed-mooncake>=0.3.12.post20260725`. That package does not exist on PyPI — not an old version, no releases at all — so `pip install ./python` dies with `from versions: none`.

Mooncake is the KV transfer engine for prefill/decode disaggregation. Its imports are all lazy, inside `runtime/pd/` and `runtime/cache/storage/mooncake_store/`, and an aggregated single-node server launched with `--disable-kvstore` never reaches them. The fix is to install the runtime with the dependency list minus that one entry. Worth knowing that repo `main` is ahead of what PyPI can actually satisfy.

### 2.2 `xgrammar` silently replaces the ROCm Triton with the CUDA one

This is the subtle one. `torch 2.11.0+rocm7.2` requires `triton-rocm==3.6.0`. `xgrammar` declares a bare `triton` dependency. Both distributions install into the *same* `triton/` directory, so pip cheerfully installs stock `triton 3.7.1` from PyPI on top of the ROCm build, and whichever landed last wins:

```bash
$ python -c "import triton; print(triton.__version__)"
3.7.1          # stock PyPI build, not the ROCm one torch asked for
```

Nothing errors. You simply end up running a Triton that was not built for your platform. The fix is to uninstall `triton` and reinstall `triton-rocm==3.6.0` after everything else settles.

The Gluon gfx950 kernels turn out to be insulated from this, because `tokenspeed-kernel-amd` routes every Triton symbol through one indirection module:

```python
# tokenspeed_kernel_amd/_triton.py
import tokenspeed_triton as triton
import tokenspeed_triton.experimental.gluon.language as gl
from tokenspeed_triton.experimental.gluon.language.amd.cdna4 import (
    async_copy as cdna4_async_copy,
)
```

`tokenspeed_triton` is a separate module name (version 3.8.10) that never collides with `triton/`. That single point of indirection is why the vendored Gluon path survives a broken `triton` install — a design decision that pays for itself the first time someone's dependency resolver does something stupid.

### 2.3 The documented AMD recipe runs out of memory on boot

Following `docs/recipes/models.md` exactly, the server loads all 96 shards, captures decode graphs, and then dies at 751 seconds. The prefill graph capture is what kills it — you can watch the free memory drain bucket by bucket:

```
Capturing prefill buckets (bucket=2048 avail_mem=19.92 GB):   2%
Capturing prefill buckets (bucket=1024 avail_mem=14.64 GB):  22%
Capturing prefill buckets (bucket= 704 avail_mem= 4.65 GB):  35%
Capturing prefill buckets (bucket= 512 avail_mem= 0.03 GB):  42%
[FATAL ERROR]: HIP failure: 'out of memory'
```

After weights and the KV pool there are about 20 GB left per GPU, and capturing all 40 prefill buckets needs more than that. The fix is `--disable-prefill-graph`, which TokenSpeed's own MI35x CI perf config already passes — the docs recipe just omits it. (The repo's HEAD commit at the time was, fittingly, a fix for prefill-warmup OOMs.)

### 2.4 It cannot share a Python environment with SGLang

TokenSpeed hard-pins `torch==2.11.0` and `transformers==5.12.0`. The container runs a custom `torch 2.9.1+rocm7.2.0` build that SGLang depends on. There is no reconciling these; TokenSpeed goes in its own venv with its own 6.2 GB torch wheel. Budget the disk and the download.

With those four handled, the server boots in about 11 minutes (dominated by reading 1.5 TB of weights), reports `max_total_num_tokens=4466304`, and serves.

## 3. End-to-end throughput

All numbers below are exact-length random workloads, `temperature 0`, `ignore_eos`, prefix caching off, measured on the 8-GPU node. SGLang appears twice: `nospec` is the like-for-like comparison since TokenSpeed ran without speculative decoding, and `DSpark` is SGLang's best tuned result on this box.

| Workload (ISL/OSL) | Concurrency | TokenSpeed | SGLang nospec | SGLang DSpark |
| --- | --- | --- | --- | --- |
| 1024 / 1024 | 1 | 44.73 | 51.40 | 109.84 |
| 1024 / 1024 | 8 | 84.97 | 311.79 | 472.00 |
| 1024 / 1024 | 32 | 226.80 | 847.87 | 949.73 |
| 1024 / 1024 | 64 | 440.58 | — | — |
| 1024 / 1024 | 128 | 487.19 | — | — |
| 4096 / 1024 | 1 | 43.98 | — | 89.30 |
| 8192 / 1024 | 1 | 44.55 | — | 74.29 |
| 8192 / 1024 | 128 | **failed** | 890.25 | — |

Output tokens/s, aggregated over all 8 GPUs.

Single stream is respectable: 44.73 against SGLang's 51.40, and TokenSpeed's TPOT stays flat at 21.5–22.2 ms from 1 K to 8 K input, where DSpark's advantage decays as its accept length falls from 2.51 to 2.34. Concurrency is where it comes apart — 3.7x behind at both 8 and 32, and TokenSpeed's best result anywhere (487 tok/s at concurrency 128) is still below SGLang's 848 at concurrency 32.

> **This is not a broken build.** TokenSpeed's own CI reference for this exact workload — `test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml` — is 43.05 tok/s per user on 8x MI350X, gated at 42. We measured 43.98 on MI355X, 2% above their own number. Whatever is happening at concurrency, it is not a misconfiguration on our end.

The more telling detail is what that CI file gates. TokenSpeed's NVIDIA perf configs carry reference curves at concurrency 1, 2, 4, 8 and 16. The AMD K3 config has exactly one line, `perf_reference: {1: [42, 5.2]}`, and pins the server to `--max-num-seqs 1` and `--cudagraph-capture-sizes 1`. Upstream does not measure batched throughput on AMD at all, so nothing protects it from regressing.

### 3.1 The 8K workload at concurrency 128 does not degrade, it wedges

This is the workload SGLang was tuned around on this box: 890 tok/s output, 8012 tok/s total. TokenSpeed completed **1 of 128 requests in an hour**. The failure is structural, not slow:

- FlatKV page pool saturates at 94% occupancy
- the scheduler logs **9,256** `flat retract ... to unwedge the pool` events
- decode collapses to 0.02 tok/s with 106 requests stuck in the queue

The capacity arithmetic explains it. The engine advertises `max_total_num_tokens=3556992`, but the scheduler's own configuration says `num_device_pages=3298` at `block_size=128`, and that pool is shared across four cache groups — `full_attention` plus three `linear_attention` groups, because 69 of K3's 93 layers are KDA and each needs its own paged recurrent state. 128 requests at 9,216 tokens each overcommit that pool by a wide margin, and the scheduler thrashes instead of queueing. SGLang on the same workload reports a 1.29 M-token pool and zero retractions.

One thing that is *not* a limit: the `--max-model-len 8192` in the documented AMD recipe. Raising it to 16384 boots cleanly.

## 4. Kernel by kernel

Now the interesting part. We already had a full PyTorch-profiler kernel decomposition of SGLang decoding K3 at batch 1 on this box, so I captured the matching TokenSpeed traces and compared them directly.

**Methodology, matched on both sides.** Batch 1, ISL 4096, TP8, no speculative decoding, **graph off** — inside a replayed HIP graph individual kernels are not separately attributable, and the SGLang baseline was captured graph-off for the same reason. Both sides sum device-kernel duration per decode step and **exclude collectives**: in eager decode the all-reduce payload is a few KB and the kernel busy-waits, so its duration measures rank skew, not work.

TokenSpeed's Proton profiler could not be used here — on ROCm it calls `rocprofiler_force_configure`, which must run before HIP initialises, so attaching it to a live server fails with error 16. The torch/roctracer profiler attaches fine and, conveniently, produces the same chrome-trace artifact the SGLang baseline used.

Kernels were attributed to blocks by the source module that launches them, not by name matching. One kernel spans blocks: `_kimi3_projection_gemv_kernel` backs the two latent H↔L projections, the shared-expert down projection and the router in every MoE layer (4 × 92 = 368 dispatches) plus `kimi3_qkvfab_projection` in `KimiLinearKDA` (69), totalling **437 — exactly the measured dispatch count**. Its time is split by dispatch share. The 93-dispatch vendor GEMM is the attention output projection, split 69/24 between the two attention types.

### 4.1 Block composition per decode step

| Block | TokenSpeed (µs) | SGLang (µs) | Verdict |
| --- | --- | --- | --- |
| MoE | 8085 | 10549 | TokenSpeed 1.30x faster |
| KDA linear attention | 2905 | 3542 | TokenSpeed 1.22x faster |
| Full MLA attention | 1468 | 3100 | TokenSpeed 2.11x faster |
| Attention residual | 1865 | 2288 | TokenSpeed 1.23x faster |
| Glue (norms, adds, casts, copies) | 2786 | 1290 | TokenSpeed 2.16x **slower** |
| **Total device compute** | **17108** | **20769** | **TokenSpeed 1.21x faster** |

The two stacks bucket their per-block "misc" slightly differently, so read the block rows as indicative and the total as solid. For the load-bearing claims, here are the kernel groups where the mapping is unambiguous:

| Kernel group | SGLang (µs) | TokenSpeed (µs) | Ratio |
| --- | --- | --- | --- |
| MLA attention math, 24 layers | 1212 | 438 | **TokenSpeed 2.77x faster** |
| MoE routing: top-16 of 896, plus sort | 2956 | 454 | **TokenSpeed 6.51x faster** |
| Attention residual epilogue | 2288 | 1853 | TokenSpeed 1.23x faster |
| KDA recurrence + short conv, 69 layers | 744 | 650 | TokenSpeed 1.14x faster |
| Routed-expert quantized GEMMs | 1437 | 2391 | **TokenSpeed 1.66x slower** |

### 4.2 What actually wins, and why

**Gluon MLA decode is the clearest win.** SGLang runs MLA decode as a two-stage split-KV Triton attention, `_fwd_grouped_kernel_stage1` scanning the KV cache and `_fwd_kernel_stage2` reducing the splits — 50.5 µs per layer. TokenSpeed's `_mla_decode_gluon` plus its softmax/reduce-V companion does the same work in 18.3 µs per layer. At batch 1 with a 4 K context the attention is pure bandwidth, and the hand-written CDNA4 Gluon kernel simply streams the KV cache better than a generic split-K Triton template. This is the one place where "hand-written Gluon beats Triton" is unambiguously true.

**The biggest absolute win is routing, and it is a design choice rather than a kernel-quality one.** Selecting top-16 of 896 experts and ordering tokens for the grouped GEMM costs SGLang 2.96 ms per step across a pipeline of AITER kernels — `grouped_topk`, `opus_moe_sorting` in multiple phases, `fused_mx_quant_moe_sort`, `moe_reduction`. TokenSpeed does it in a single fused `_kimi3_sigmoid_bias_topk_kernel` for 0.45 ms. That one kernel is worth 2.5 ms per decode step, which is more than the entire end-to-end deficit. With 896 experts the routing metadata is large enough that materialising it between separate kernels costs more than computing it, and fusing the sigmoid, the bias and the top-k into one pass removes those round trips. Nothing about this is AMD-specific — it is the kind of win that transfers.

**AITER wins the expert GEMMs.** SGLang's `mfma_moe1`/`mfma_moe2` MXFP4 kernels run the routed experts in 1.44 ms against 2.39 ms for TokenSpeed's Gluon `_stage1_a16w4_situ_warp_gemv` and `_stage2_a16w4_warp_gemv_combine`. Read this one with care: TokenSpeed runs EP8, so each rank owns 112 whole experts and at batch 1 handles only the couple that its tokens route to, while SGLang runs pure TP8 where every rank does all 16 experts at one-eighth width. Those are different shapes, and the comparison is confounded by parallelism strategy, not purely by kernel quality. AITER's are also `a8w4` (8-bit activations) against Gluon's `a16w4`, so AITER is moving half the activation bytes.

**KDA is a wash, and both are Triton.** TokenSpeed's decode-time KDA is `ops/attention/triton/kda.py` — despite the "native AMD KDA implementation" framing in the docs, the batch-1 decode recurrence is a Triton kernel on both stacks, and they land within 14% of each other. This is consistent with the earlier finding that KDA at batch 1 is occupancy-bound rather than bandwidth- or compute-bound: 69 layers of a fixed-size recurrent state update, filling 48 of 256 CUs. No kernel language fixes that; only batching does.

**And TokenSpeed loses 1.5 ms to glue.** RMSNorms, `_add3`, fp8 copy kernels, `Memcpy DtoD`, a generic `_rowcta_gemv_kernel` — 2.79 ms against SGLang's 1.29 ms. SGLang fuses more of this away, most visibly into `add_rmsnorm_quant_kernel`, which folds the residual add, the norm and the quantisation into one pass.

### 4.3 The finding that matters

Put the three measurements side by side:

| Metric | SGLang | TokenSpeed |
| --- | --- | --- |
| Device compute per decode step | 20769 µs | **17108 µs** |
| Kernel launches per decode step | 3372 | **2375** |
| Collective dispatches per decode step | 187 | 187 |
| End-to-end step, graph on | **19.39 ms** | 22.19 ms |

TokenSpeed does less GPU work, in fewer launches, with the same number of collectives — and takes longer. Both of the cheap explanations are ruled out by measurement: it is not launch overhead, because it issues 30% fewer launches; it is not collective count, because both issue exactly 187 per step, one per layer block.

Whatever costs TokenSpeed those milliseconds happens *between* the kernels, not inside them. That is the scheduler, the Python execution plane, the host-side path that has to feed 93 layers of a hybrid model every 20 ms, and how much of the collective latency is actually exposed rather than overlapped. At concurrency 1 it costs 14%. At concurrency 8 the same class of overhead costs 3.7x, which is what you would expect from a per-step cost that does not amortise as the batch grows.

This reframes the AMD gap entirely. The kernels are not the problem — they are, measurably, the better half of this engine. TokenSpeed on gfx950 has a runtime problem wearing a kernel problem's clothes.

## 5. What I would take away

**For someone choosing an engine for K3 on MI355X today:** SGLang, and it is not close at concurrency. TokenSpeed is competitive for single-stream latency and nothing else yet.

**For someone working on TokenSpeed's AMD path:** the kernels are ahead. Profiling effort belongs in the scheduler and the host path, and the first thing to fix is that upstream CI does not measure concurrency on AMD at all — a throughput regression there is currently invisible.

**For someone working on SGLang's AMD path:** the fused router is 2.5 ms per decode step sitting on the table, larger than any other single gap in the trace, and it is a pure software win with no hardware dependency. The Gluon MLA decode kernel is worth another 0.8 ms.

**The transferable idea**, with the proper nouns deleted: when an engine is slower, the instinct is to profile the kernels, and the kernels are usually where the answer is. Here the trace said the opposite twice over — faster kernels, fewer launches, same collectives, slower engine — and the only reason that was visible at all is that both sides were measured with the same methodology on the same hardware on the same day. A benchmark that stops at tokens per second would have concluded "TokenSpeed's AMD kernels need work", which is precisely backwards.

## 6. Reproducing this

Build into an isolated venv, in this order, because `tokenspeed-kernel` resolves the AMD package during its native build:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip "setuptools<82" wheel cmake ninja
pip install --index-url https://download.pytorch.org/whl/rocm7.2 torch==2.11.0+rocm7.2

pip install --force-reinstall --no-deps ./tokenspeed-kernel-amd --no-build-isolation
TOKENSPEED_KERNEL_BACKEND=rocm pip install ./tokenspeed-kernel/python/ --no-build-isolation
pip install ./tokenspeed-scheduler/ \
  --config-settings=cmake.define.TOKENSPEED_FLAT_KVCACHE=ON   # K3 is FlatKV-only
# install ./python's dependencies minus tokenspeed-mooncake, then:
pip install -e ./python --no-build-isolation --no-deps
pip uninstall -y triton && pip install --force-reinstall --no-deps \
  triton-rocm==3.6.0 --index-url https://download.pytorch.org/whl/rocm7.2
```

Serve, with the two deviations from the documented recipe:

```bash
tokenspeed serve moonshotai/Kimi-K3 --served-model-name kimi-k3 --trust-remote-code \
  --max-model-len 8192 --kv-cache-dtype fp8 --tensor-parallel-size 8 \
  --mm-encoder-tp-mode data --enable-expert-parallel \
  --attention-backend mla --moe-backend auto \
  --gpu-memory-utilization 0.92 --max-num-seqs 32 --disable-kvstore \
  --disable-prefill-graph --no-enable-prefix-caching \
  --host 127.0.0.1 --port 8100
```

Benchmark with exact lengths — note the `0`:

```bash
tokenspeed bench serve --base-url http://127.0.0.1:8100 --model kimi-k3 \
  --tokenizer /path/to/Kimi-K3 --dataset-name random \
  --input-len 4096 --output-len 1024 --random-range-ratio 0 \
  --num-prompts 3 --max-concurrency 1 --num-warmups 1 \
  --ignore-eos --extra-body '{"temperature": 0}'
```

For the traces, add `--enforce-eager`, then arm the profiler with GPU activities (not Proton) and drive one request through it:

```bash
curl -sS -X POST http://127.0.0.1:8100/start_profile \
  -H 'Content-Type: application/json' \
  -d '{"num_steps": 16, "activities": ["GPU"],
       "profile_by_stage": true, "profile_id": "kimi-k3-tp8"}'
```

Each rank writes `<id>-TP<r>-{EXTEND,DECODE}.trace.json.gz`. Sum device-kernel duration per step, drop the collectives, and attribute the rest by launching module.
