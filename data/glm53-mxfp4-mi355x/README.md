# GLM-5.3 FP8 vs MXFP4 on 8× MI355X — raw data archive

Source of record for Experiment No.004
([`/sources/glm53-mxfp4-mi355x.html`](../../public/sources/glm53-mxfp4-mi355x.html)).
Nothing here is built by Astro (`data/` sits outside `src/` and `public/`); this
directory exists so the measurements and their provenance survive the container
they were taken in — and, in this case, so that an **unfinished** campaign leaves
behind something a resume can be built on rather than a memory of what happened.

39 data files plus this README, 1.0 MB, no GPU needed to read any of it.

## What was measured

A controlled SGLang A/B between the official Z.ai FP8 checkpoints of GLM-5.3 and
the OneNexus MXFP4 conversions of the same models, on one node, inside one frozen
software point.

| | |
|---|---|
| node | `mia1-p02-g23` — 8× AMD Instinct MI355X, `gfx950:sramecc+:xnack-` |
| runtime | host ROCm 7.2.0, torch `2.9.1+rocm7.2.0.git7e1940d4`, Triton 3.6.0 |
| SGLang | `cc981fdebd74582b26a6e0e3c1274c910df72bc7` (Full) · `e122ff45e901e714f4610b1c3a535cecdd1706b4` (Flash) |
| AITER | `ea6868a02cf54e29730a38b9a3980e4342823482` |
| sgl-eval | `db1547d6098c791ecb3576353f8a5e9d06344e7c` |
| FlyDSL / sglang-kernel | `0.3.2.dev853` / `0.4.6.post1` |
| window | 2026-08-31 07:33:49 → 2026-09-01 04:36:07 UTC (21:02:18) |

**There is no container in this picture.** The node has no Docker, Podman,
Apptainer or Skopeo, so no ROCm 7.2.4 MI35x image was pulled or run. This is host
ROCm 7.2.0 plus a source overlay of the two frozen checkouts. Do not describe any
number in this directory as container-level validation.

Four variants were planned. Two finished, one finished on the accuracy side only,
one never started:

| variant | accuracy | performance |
|---|---|---|
| Flash FP8 | 1198 paired rows | 11/11 runs |
| Flash MXFP4 | 1198 paired rows | 11/11 runs |
| Full FP8 | 2017 rows | **3/11 runs — not publishable** |
| Full MXFP4 | not started | not started |

## Headline result

The Flash pair is the only complete comparison. Five thresholds were fixed in the
protocol before the first formal run; four are met and one is not.

| gate | line | measured | |
|---|---|---|---|
| per-benchmark accuracy delta | ≥ −2.0 pp | −1.52 pp (GPQA) | pass |
| combined correct-count recovery | ≥ 98% | 99.5301% | pass |
| length-finish-rate increase | ≤ 2.0 pp | **2.0202 pp (GPQA)** | **breach** |
| median total throughput ratio | ≥ 90% at each concurrency | 91.78% (c1) | pass |
| before/after canary drift | ≤ 5% | 2.30% (FP8 side) | pass |

Accuracy, from `accuracy/flash-accuracy-comparison.json`:

| set | FP8 | MXFP4 | delta | McNemar p | length finishes |
|---|---|---|---|---|---|
| GSM8K (500) | 489 / 97.80% | 490 / 98.00% | +0.20 pp | 1.000 | 1 → 3 |
| MMLU (500) | 437 / 87.40% | 434 / 86.80% | −0.60 pp | 0.690 | 37 → 39 |
| GPQA Diamond (198) | 138 / 69.70% | 135 / 68.18% | −1.52 pp | 0.7283 | 55 → 59 |
| combined (1198) | 1064 | 1059 | — | — | 93 → 101 |

`correct_count_recovery = 1059/1064 = 0.9953007518796992`. The net −5 is the
residue of 67 rows that changed side: 36 FP8-only-correct against 31
MXFP4-only-correct. Report the 2×2 cells, not just the delta.

Serving, median of three rounds, ISL 8192 / OSL 1024, from
`perf/flash-{fp8,mxfp4}-perf-summary.json`:

| concurrency | FP8 tok/s | MXFP4 tok/s | ratio | TPOT FP8 → MXFP4 |
|---|---|---|---|---|
| 1 | 124.93 | 114.66 | 91.78% | 71.81 → 77.97 ms |
| 8 | 943.52 | 870.06 | 92.21% | 75.27 → 81.63 ms |
| 32 | 3624.11 | 3350.44 | 92.45% | 75.62 → 82.08 ms |

Canary drift: FP8 +2.30%, MXFP4 +0.46%.

### The one breach, and why it is recorded as one

`length_rate_delta` on GPQA Diamond is `0.020202020202020204` against a 2.0 pp
ceiling. Truncation count on a 198-row set is an integer, so the measurable grid
is 0.50505 pp wide: the attainable increases either side of the ceiling are
1.5152 pp (three more truncations) and 2.0202 pp (four). **A 2.0 pp ceiling names
a value this instrument cannot report.** The threshold is mis-specified; the
reading is not wrong. The fix belongs in the protocol before the next run — state
the gate as a maximum number of additional truncations, or choose a sample size
that makes 2.0 pp resolvable — and not in a post-hoc relaxation of the line.

### Checkpoint size does not predict decode speed

Computed in `weights/*.json` directly from safetensors headers, never from a
config or model-card claim:

```
active_bytes = always_active_bytes + (top_k / num_experts) * routed_expert_bytes
```

| checkpoint | file | always active | routed share | active / token |
|---|---|---|---|---|
| Flash FP8 | 328.327 GB | 13.957 GB | 8.458 GB | 22.415 GB |
| Flash MXFP4 | 227.486 GB (−30.71%) | 16.574 GB | 5.379 GB | 21.953 GB (−2.06%) |
| Full FP8 | 755.617 GB | 18.729 GB | 22.655 GB | 41.383 GB |
| Full MXFP4 | 438.002 GB (−42.04%) | 31.141 GB | 12.032 GB | 43.174 GB (**+4.33%**) |

The baseline is FP8, already one byte per parameter, so relative to it the
quantized tensors roughly halve and the BF16 exclusions double — and the
exclusions are exactly the always-active trunk. On Full the trunk grows 12.4 GB
per token while the routed share only falls 10.6 GB.

Neither number predicts the clock: Flash MXFP4 reads 2.06% fewer bytes and runs
8.22% slower at c1. At TP8 each rank reads 2.80 GB in 71.81 ms — 39 GB/s against
HBM3E measured in thousands — so this operating point is not bandwidth bound and
the extra time is dequantization work, not bytes.

### Model card vs metadata (Full MXFP4)

`weights/full-mxfp4-shared-experts.json` records, per decoder layer, the stored
dtype of every shared-expert projection. The card claims the dense/shared MLP
projections stay BF16. The headers say layers 3–77 store `gate_proj`, `up_proj`
and `down_proj` as packed U8 with an FP4 scale; layers 0–2 have no shared-expert
projections; the only BF16 exclusion is the MTP head at layer 78. If the card
were right the always-active set would be larger still, so the sign of the result
above does not depend on which document is correct — only its magnitude does.

## Files

```
accuracy/
  flash-accuracy-comparison.json      paired 2x2 cells, McNemar exact p, length_rate_delta
  flash-fp8-accuracy-summary.json     per-set score, latency, tokens, finish reasons
  flash-mxfp4-accuracy-summary.json
  full-fp8-accuracy-summary.json      GSM8K 1319 / MMLU 500 / GPQA 198, error_rate 0
perf/
  flash-fp8-perf-summary.json         median/mean/p10/p90 over the three rounds + canary
  flash-mxfp4-perf-summary.json
  flash-fp8/     *.jsonl              11 runs: c1/c8/c32 x r1..r3, canary before + after
  flash-mxfp4/   *.jsonl              11 runs, identical protocol
  full-fp8-partial/                   3 runs + INTERMEDIATE_STOP.json — NOT PUBLISHABLE
weights/
  flash-fp8.json  flash-mxfp4.json    tensor bytes by dtype, active-weight decomposition
  full-fp8.json   full-mxfp4.json
  full-mxfp4-shared-experts.json      per-layer storage of the shared-expert projections
env/
  environment-preflight-20260901.json versions, GPU inventory, resolved import paths (pre-campaign)
  flash-fp8-environment.json          commits and versions recorded at that run's launch
  flash-mxfp4-environment.json        same, for the MXFP4 half of the Flash A/B
  full-fp8-environment.json           same, for Full FP8
  flash-fp8-sglang-sha.txt            the SHA the performance harness stamped independently
checkpoints.json                      repo, revision, index sha256, shard and tensor counts
```

Each performance `.jsonl` is one `sglang.bench_serving` record and embeds the
full `server_info` of the server it hit, so the launch flags are recoverable from
the measurement itself rather than from a separate note.

`perf/full-fp8-partial/INTERMEDIATE_STOP.json` is the disqualifying marker for
the three records beside it. It carries `publishable: false`, the list of what
was never run, and the resume rule. Read it before touching that directory.

## The exact commands

One server per variant, then accuracy, then eleven benchmark runs. Ports were
31101 (Flash FP8), 31102 (Flash MXFP4), 31103 (Full FP8).

The launch and `bench_serving` invocations below are reconstructed field by field
from the `server_info` and top-level keys recorded inside every `.jsonl`, so they
are checkable against the data. The `sgl-eval` invocation is reconstructed from
the `metrics.gen` block of each accuracy summary (`temperature`, `top_p`,
`min_p`, `reasoning_effort`, `max_tokens`, `seed`, `num_threads`,
`chat_template_kwargs`); the flag spellings are the ones that produce that block,
not a transcript of the shell history.

```bash
python3 -m sglang.launch_server \
    --model-path /data/GLM-5.3-Flash-MXFP4 \
    --tp-size 8 \
    --attention-backend dsa \
    --moe-runner-backend aiter \
    --kv-cache-dtype bfloat16 \
    --context-length 65536 \
    --mem-fraction-static 0.8 \
    --chunked-prefill-size 4096 \
    --max-running-requests 32 \
    --disable-radix-cache \
    --trust-remote-code \
    --model-loader-extra-config '{"enable_multithread_load": true}' \
    --served-model-name glm-5.3-flash-mxfp4 \
    --port 31102
```

The FP8 servers are the same command with the model path, served name and port
changed; `quantization` resolves to `null` for them and to `quark` for the MXFP4
checkpoints (visible in `server_info.quantization` of every `.jsonl`).

```bash
# accuracy — same three sets, same generation settings, both sides
for SET in gsm8k mmlu; do
  sgl-eval run "$SET" --base-url http://127.0.0.1:31102/v1 \
      --num-threads 32 --temperature 0 --top-p 0.95 --min-p 0 \
      --reasoning-effort max --max-tokens 8192 --seed 0
done
sgl-eval run gpqa --base-url http://127.0.0.1:31102/v1 \
    --num-threads 32 --temperature 0 --top-p 0.95 --min-p 0 \
    --reasoning-effort max --max-tokens 16384 --seed 0 --thinking
```

```bash
# performance — canary, three rounds at each concurrency, canary again
bench () {  # $1 concurrency  $2 round  $3 isl  $4 osl
  python3 -m sglang.bench_serving \
      --backend sglang --dataset-name random \
      --random-input-len "$3" --random-output-len "$4" --random-range-ratio 1.0 \
      --max-concurrency "$1" --num-prompts $(( $1 * 4 )) \
      --port 31102 \
      --output-file "perf-c$1-r$2-i$3-o$4.jsonl"
}

bench 1 canary-before 512 128
for R in 1 2 3; do for C in 1 8 32; do bench "$C" "$R" 8192 1024; done; done
bench 1 canary-after  512 128
```

`--num-prompts` is four per concurrency slot, which is why the records show 4, 32
and 128 completed requests at c1, c8 and c32.

## Derived vs measured

**Measured**, straight out of the harnesses: every field in the accuracy
summaries (`score`, `latency_seconds`, `total_completion_tokens`,
`finish_reasons`) and every field in the performance `.jsonl`
(`total_throughput`, `median_tpot_ms`, `median_ttft_ms`, `completed`,
`duration`).

**Derived**, computed over those files:

- `perf/*-perf-summary.json` — median, mean, std, p10, p90 and
  `relative_spread` across the three rounds of each concurrency, plus
  `canary.relative_delta = after/before − 1`.
- `accuracy/flash-accuracy-comparison.json` — the 2×2 cells, `absolute_delta`,
  `correct_count_recovery`, `length_rate_delta`, exact McNemar p, and the
  agreement rates. Pairing is by row index across the two runs, which is only
  valid because both sides ran the same dataset revision at `temperature=0` and
  `seed=0`.
- `weights/*.json` — everything except `path`, `num_layers`, `num_experts` and
  `top_k`; the byte counts come from summing safetensors header entries.

The gate verdicts on the HTML page are derived from the two files above and
nothing else.

## Precision, and what these files cannot support

1. **The Full performance numbers are not results.** A performance figure in this
   campaign is defined as the median of three rounds bracketed by two canaries
   with a post-run quiescence check. `perf/full-fp8-partial/` holds one round at
   two of three concurrencies and one opening canary. That is a different
   quantity printed in the same unit. Use it to plan the resume; never quote it.

2. **No error bar is attached to the accuracy scores.** Each set was run once per
   side at `temperature=0`, so the pairing is deterministic given identical
   software, but there is no repeat that would let a run-to-run interval be
   estimated. The McNemar p-values describe the paired disagreement, not the
   sampling variability of the score.

3. **The 43.94% Full FP8 GPQA score is half a truncation measurement.** 109 of
   198 responses hit the 16K output ceiling under `reasoning_effort=max`. The
   number is reproducible and it is not a model-quality figure.

4. **Round-to-round spread is small but not zero**, and it is smallest where the
   load is highest: `relative_spread` on total throughput is 1.12% / 0.82% /
   0.06% for FP8 at c1 / c8 / c32 and 0.80% / 0.44% / 0.18% for MXFP4. The
   MXFP4/FP8 ratios of 91.78–92.45% sit far outside those bands.

5. **The canary is the only contention control.** There is no co-tenant probe
   beyond it. The FP8 side moved +2.30% across its own sequence, which is a third
   of the effect being resolved — this is the measurement that makes single-round
   numbers indefensible, including the Full ones in item 1.

6. **Two different moments are recorded, and only one of them served.**
   `env/environment-preflight-20260901.json` is the node capture taken *before*
   the formal campaign, on the checkout as it then stood
   (`sglang_sha 959ca033ebb7…`, `aiter_sha ad27e15097bc…`). Each formal run then
   recorded its own commits at launch, and those are the ones that answered
   requests:

   | run | `sglang_sha` | `aiter_sha` |
   |---|---|---|
   | Flash FP8 | `e122ff45e901…` | `ea6868a02cf5…` |
   | Flash MXFP4 | `e122ff45e901…` | `ea6868a02cf5…` |
   | Full FP8 | `cc981fdebd74…` | `ea6868a02cf5…` |

   Those three files — `env/flash-fp8-environment.json`,
   `env/flash-mxfp4-environment.json`, `env/full-fp8-environment.json` — are the
   provenance for the table at the top of this file, and they agree with it
   exactly. The preflight file is not a contradiction of them; it is an earlier
   timestamp. When the two disagree, the per-run file is the one that ran.

## Resuming

Do not append to `perf/full-fp8-partial/`. Re-run the whole Full FP8 performance
sequence into a fresh directory so all three repeats and both canaries belong to
one uninterrupted run. The Full FP8 accuracy in `accuracy/` may be reused **only**
if the frozen checkpoint, SGLang, AITER, kernel runtime, datasets and generation
protocol are all unchanged — the moment `reasoning_effort`, `max_tokens` or the
sample size moves, both sides have to be re-measured.

Before resuming at full scale, run a 60–90 minute pilot on Full MXFP4: route
smoke, a fixed small sample of each accuracy set, one round at each concurrency.
The three ways the remaining six hours can be wasted — a loader failure, a
collapsed truncation rate, a throughput collapse — are all visible inside that
window.

And chain the stages. The largest avoidable cost in the original run was
1:57:03 of eight idle MI355X between the end of Full FP8 accuracy and a human
noticing it had finished. A completion marker per stage, a successor triggered by
that marker, and a watchdog that tears the server down when a marker is late
would have removed all of it.
