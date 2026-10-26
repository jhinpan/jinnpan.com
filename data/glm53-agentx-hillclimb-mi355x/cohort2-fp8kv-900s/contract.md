# GLM-5.3 on MI355X, InferenceX L1 (900 s) A/B contract, cohort 2: monokernel on the fp8 cache

Written 2026-10-25 before the first measured run of this cohort. Local only: nothing is
pushed or posted.

## Question

Cohort 1 (`../infx-l1-glm53-20261025/contract.md`) found no InferenceX lead for bf16 KV +
monokernel: the bf16 cache lengthened TTFT (b0). With glm53 reading and writing SGLang's
E4M3 MLA cache (local branch `glm53-fp8-kv`, commit `4d98bda` on sgl-amd-mono `9484c60`),
the candidate is the recipe plus one flag. Does it raise GLM-5.3 AgentX on InferenceX's
metrics?

## Cohort (fixed for every arm)

As cohort 1 (node mi355x-node, image `sha256:65a81d6b...4915`, SGLang
`fe1bd10837`, AITER `285579b6`, InferenceX `cfb7bdd0c` + the local config, AgentX
`agentx-harness 1.0.12`, corpus snapshot `23f152f6f0f9`, golden AL 2.99), except:

| Field | Value |
|---|---|
| sgl-amd-mono | `glm53-fp8-kv` `4d98bda` mounted over `/sgl-workspace/sgl-amd-mono` in container `infx-glm53-fp8`, for both arms |
| Arms | rendered into `arms/` by `render_arms.py` from cohort 1's `infx generate` output |

## Arms

| Arm | Delta from the recipe | Route log |
|---|---|---|
| base | none (fp8_e4m3 KV) | absent |
| cand8 | `monokernel-backend=amd` (fp8_e4m3 KV kept) | present |
| baseq, cand8q (quality) | base, cand8 with `--real-acceptance` | as base, cand8 |

## Correctness before timing (done 2026-10-25, phase `fp8kv`)

- The new e2e check `test_each_covered_verify_layer_within_its_numerics_with_fp8_kv`
  (MTP verify, 4 and 8 rows, past 2048 keys).
- A local decode check on the fp8 cache at widths 2, 4 and 8 (not committed).
- The bf16 verify check (regression).
- A negative control (the latent's fp8 byte pairs swapped) that must fail the fp8 check.

## Order

Round 1: c1 base, cand8; c2 base, cand8; c4 base, cand8. Round 2 reverses the whole
sequence. Then GSM8K baseq, cand8q at c2.

## Metrics and decision (unchanged from cohort 1)

- Primary, per concurrency: total tok/s/GPU, P90 E2E-normalized interactivity, P90 E2EL
  (`compare_agentx.py`, InferenceX field names), per round.
- Noise floor: the A/A pair base c1 r1 vs base c1 r2 (`--aa`). A lead needs the same sign
  in both rounds and a size beyond the A/A difference.
- Go for the upstream route: candidate ahead at c1 and c2 on P90 E2E-normalized
  interactivity or P90 E2EL with no baseline lead on throughput at the same points, c4
  not behind beyond noise, the quality gate passed, and every timing check passed.
- Supporting evidence, reported but not part of the decision: request-paired TTFT, ITL
  and E2EL against the mean of the two base runs (`l1_pair.py`), and the slowest 10 % /
  20 % of E2EL/OSL (the requests behind P90 E2E-normalized interactivity).

## Per-run validity (as amended in cohort 1)

Startup, health, served model name; AIPerf's profiling phase reaches its 900-s target;
the profiling window's failed requests plus drain cancellations within the 0.10
threshold; the InferenceX validator; the realized `accept len` median within 2.99 +-
0.05; the route log as the table above; no foreign GPU process or activity; memory
released after the stop.

## Quality gate

As cohort 1: InferenceX GSM8K, first 500 test questions, real acceptance, concurrency 2;
both arms at least 0.94 under every lm-eval filter; paired McNemar reported.

## Deviations from the official flow

As cohort 1, plus: the sgl-amd-mono change is a local, unreviewed branch.

## Stop conditions

As cohort 1.
