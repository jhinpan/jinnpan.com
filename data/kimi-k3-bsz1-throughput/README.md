# Kimi-K3 dspark throughput sweeps — raw data archive

Source of record for the Kimi-K3 bsz=1 / concurrency benchmark. Nothing here is
built by Astro (`data/` sits outside `src/` and `public/`); this directory exists
so the measurements and their provenance survive the machine they were taken on.

## What was measured

18 points across three orthogonal sweeps, run 2026-07-29/30 on a single node with
8x MI355X, TP8.

| File | Sweep | Fixed | Varied |
|---|---|---|---|
| `results/sweep-ab-isl-osl.csv` | A | OSL 1024, concurrency 1 | ISL 128 → 65536 |
| `results/sweep-ab-isl-osl.csv` | B | ISL 1024, concurrency 1 | OSL 128 → 4096 |
| `results/sweep-c-concurrency.csv` | C | ISL/OSL 1024 | concurrency 1 → 48 |

## Server configuration

The winning dspark recipe from the launch-parameter grid search, reproduced from
`results/incumbent-dspark.env`:

```
MEM_FRAC=0.92
RADIX=0
CUDA_GRAPH_MAX_BS=256
CHUNKED_PREFILL=default
EXTRA_ARGS=--speculative-dspark-block-size 3
BEST_LABEL=p3-curve-p3-g3
```

Launched with `serve-k3-ext.sh dspark` on port 30100, which adds `--tp 8`,
`--attention-backend triton`, `--dtype bfloat16`, `--disable-radix-cache`, the
DSpark draft model, and the aiter environment. Boot reported
`max_total_num_tokens=1504168`, `max_running_requests=48`,
`available_gpu_mem=19.94 GB` — identical to the grid's `p5-dspark` run, which is
what confirms the recipe reproduced. sglang at `3d35b45f7`.

`results/accuracy_gate.md` is the accuracy gate for this recipe, kept here
because it is the reason `--mamba-ssm-dtype bfloat16` is *not* in the recipe
despite being the nospec lane's throughput winner.

## Reproducing

```bash
bash sweep-bsz1.sh <tag>          # sweeps A + B
bash sweep-conc-scaling.sh <tag>  # sweep C
```

Both scripts target port 30100, write to `bsz1_results/<tag>/results.csv`, and
extract metrics with `gridtools.py parse-bench` — the same parser the grid search
used, so rows are comparable with its `results.csv`. Both take
`/tmp/k3-grid.lock` so a grid run starting midway cannot tear the server down
underneath them.

## Derived metrics

Two columns quoted in the writeup are computed, not emitted by the benchmark:

- `step_ms = accept_len x tpot_med` — wall time of one speculative verify step.
- `prefill_tps = ISL / ttft_med` — prefill token rate.

## Caveat on precision

`--dataset-name random` emits random token IDs, and the draft model's acceptance
rate is content-sensitive. The same ISL/OSL 1024 point measures 101.92 tok/s with
4 requests (sweep A) and 109.84 tok/s with 8 requests (sweep C) — 7.8% apart,
with the throughput ratio almost exactly equal to the accept-length ratio. Treat
bsz=1 dspark figures as carrying 8–10% run-to-run variance at low sample counts,
and as a conservative lower bound relative to real text.

For reference, the grid search's `p3-lat-win` row on the same recipe recorded
111.58 tok/s at accept length 2.757, against sweep C's 109.84 and 2.693 — 1.6%
and 2.3% apart respectively.

## `canvas/`

Archived copies of the Cursor Canvas sources that presented this data. They are
kept as a backup and as the input for any web rendering; they are **not** built
by this site and will not render from here. Cursor only renders `.canvas.tsx`
from its managed directory (`~/.cursor/projects/<workspace>/canvases/`), which is
outside any repo — hence this archive. Both import from `cursor/canvas`, which
ships types only (no runtime), so they cannot be bundled without substituting an
implementation of those components.
