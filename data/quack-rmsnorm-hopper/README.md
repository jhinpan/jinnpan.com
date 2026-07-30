# Quack RMSNorm on H100 / H200 — raw data archive

Source of record for Experiment No.001 ([`/sources/quack-rmsnorm-hopper.html`](../../public/sources/quack-rmsnorm-hopper.html)).
Nothing here is built by Astro (`data/` sits outside `src/` and `public/`); this
directory exists so the measurements and their provenance survive the container
they were taken in.

## What was measured

Quack's CuTe DSL RMSNorm forward and backward kernels, at repository commit
`8aabb38`, on one verified-idle GPU per host:

| | H200 | H100 |
|---|---|---|
| host | `hyper00-jhinpan` | `novita-h100-jhinpan` |
| GPU | NVIDIA H200, index 3 | NVIDIA H100 80GB HBM3, index 0 |
| idle check before launch | 0 MiB used, 0% util | 0 MiB used, 0% util |
| PyTorch / CUDA | 2.9.1+cu128 / 12.8 | 2.11.0+cu130 / 13.0 |
| CUTLASS DSL | 4.6.1 | 4.6.0 |
| date | 2026-07-30 | 2026-07-30 |

The matrix is the harness default: 9 shapes x 5 dtype/weight-mode combinations x
2 operations = 90 cells, times 3 providers = 270 rows per GPU.

Providers, all three timed in a single sweep so they share one roofline probe
and one thermal state:

| provider | entry point | config source |
|---|---|---|
| `quack_tuned` | `rmsnorm_fwd_tuned` / `rmsnorm_bwd_tuned` | `@autotune` exhaustive search, winner cached to `~/.quack/cache` |
| `quack` | `rmsnorm_fwd` / `rmsnorm_bwd` | hand-written analytical heuristic ladder |
| `torch` | `torch.nn.functional.rms_norm` | same-device reference |

## Headline result

Absolute figures at the widest row (`32768x8192`, bf16), best config path per
cell — these are the numbers an MI355X run should be placed next to. Logical
bytes are hardware-independent, so the same shape moves the same bytes and the
GB/s ratio is a real throughput ratio.

| | bytes moved | H200 | H100 |
|---|---|---|---|
| forward | 1024.0 MiB | 266.5 us · 4029.5 GB/s · 93.4% of roofline | 358.9 us · 2991.8 GB/s · 94.0% |
| backward | 1536.2 MiB | 407.1 us · 3956.5 GB/s · 91.7% | 540.6 us · 2979.6 GB/s · 93.6% |
| roofline | — | 4312.7 GB/s (2-read-1-write probe) | 3182.3 GB/s (write probe) |

Both kernels reach the low nineties as a fraction of measured achievable
bandwidth — but they get there differently.
Forward is already there on the default analytical config (93.4% H200 / 93.8%
H100) and autotuning does not improve it. Backward on the default config stalls
at 75.7% (H200) / 85.1% (H100) and only reaches 91.7% / 93.6% once searched,
because `_for_hopper_bwd` hardcodes `use_tma=False` and leaves `smem_stages` at
its default of 2 — neither of which the analytical path can vary.

Autotuning is not uniformly better: it regresses a minority of cells, worst case
−16.7%, because it scores candidates with an L2-cold benchmark while this harness
times steady-state calls through rotating buffers.

## Files

```
results/h200/    results.csv, environment.json, run.log
results/h100/    results.csv, environment.json, run.log
sweep-rmsnorm.sh            the exact command, with the idle-GPU guard
quack_tuned_provider.patch  adds the quack_tuned provider to the harness
```

`results.csv` is the harness's stable schema v2. `environment.json` records the
full command, the measured roofline, the contention canary and the git commit.

## Reproducing

```bash
bash sweep-rmsnorm.sh <repo-root> <idle-gpu-index> <output-dir>
```

The script refuses to start on a GPU that is not idle, and prints the resolved
`quack.rmsnorm` module path first. Both guards exist because those are the two
ways this measurement silently produces a plausible wrong answer:

1. **A busy GPU.** `peak_bw_pct` is normalised against a bandwidth probe taken at
   the start of the run. A co-tenant invalidates every ratio in the file.
2. **The wrong `quack`.** Running the harness by path puts `benchmarks/` at the
   front of `sys.path`, so a bare `import quack` resolves to the pip wheel in
   site-packages instead of the checkout under test. `PYTHONPATH` must point at
   the repository root.

Note that `environment.json` reports `versions.quack` from pip distribution
metadata, so it names the *installed wheel* even when `PYTHONPATH` correctly
wins. `git_commit` is the field to trust.

## Derived vs measured

Measured per cell: `median_us`, `p10_us`, `p90_us` from 12 timed samples after 3
warmup rounds, and `cold_compile_ms` for the first call (which for `quack_tuned`
contains the entire autotune search and is excluded from the steady-state
samples).

Derived: `logical_gbps = logical_bytes / median_us`, where `logical_bytes` is
provider-independent — forward counts reads of x and weight plus a write of y;
backward counts reads of x, dout, weight and fp32 rstd plus writes of dx and
dweight. `peak_bw_pct = logical_gbps / peak_bw_gbps * 100`.

`peak_bw_gbps` is the best of three device probes (copy, pure write,
two-read-one-write). **The two hosts did not peak on the same probe**, so
percent-of-peak is a within-host efficiency measure and the two columns are not
directly comparable to each other.

## Precision

Every cell passes an fp32 correctness gate before it is timed; a mismatch aborts
the sweep. Both archived runs closed with a quiet contention canary — 0.999 on
each — so percent-of-peak is measured against a roofline that still held at the
end of the sweep.

`environment.discarded-contended.json` is the first H100 attempt, kept for
provenance and **not** the source of any published number. Neighbours on GPUs 1,
3 and 4 of that node came online partway through and its canary closed at 0.872,
outside the quiet band. The sweep was re-run on the same idle GPU 0 once the
autotune cache was warm, which is why the replacement run was cheap: the winning
configs were already on disk, so only the timing had to be redone.

Run-to-run spread on the large shapes is roughly 1–3%; treat differences smaller
than that as noise, and do not compare percent-of-peak across the two hosts
(different roofline probes, see above).

## Regenerating the write-up

The figures in the HTML record are generated from these CSVs:

```bash
node ../../scripts/gen-rmsnorm-plates.mjs          # rewrite the plates and tables
node ../../scripts/gen-rmsnorm-plates.mjs --check  # non-zero if the page is stale
```
