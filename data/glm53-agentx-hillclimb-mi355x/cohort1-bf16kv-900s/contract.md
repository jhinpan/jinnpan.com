# GLM-5.3 on MI355X, InferenceX L1 (900 s) A/B contract

Written 2026-10-25 before the first measured run. Local only: nothing is pushed or posted.

## Question

Can SGLang's GLM-5.3 AgentX result on 8x MI355X improve on the InferenceX path when the
recipe turns on the sgl-amd-mono glm53 monokernel (`--monokernel-backend amd`)?

## Cohort (fixed for every arm)

| Field | Value |
|---|---|
| Node | mi355x-node, 8x MI355X (gfx950), all 8 GPUs per run (TP8 EP1) |
| Image | `lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261024` = `sha256:65a81d6ba18d32e00161a0feed0b889fb604505daeb56ae3610da92dc52f4915` |
| SGLang | `fe1bd10837` (editable `/sgl-workspace/sglang/python`; package metadata says `g3ea7e210d2`, 3 commits older) |
| AITER | `285579b6` (`amd-aiter 0.1.26.dev144`) |
| sgl-amd-mono | `9484c60` (main after #42) mounted over `/sgl-workspace/sgl-amd-mono` in container `infx-glm53-l1`; the image's own `05e0bb02` in `infx-glm53-l1-img` (arm b1 only) |
| InferenceX | `cfb7bdd0c` plus the local, uncommitted config `glm5.3-fp8-mi355x-sglang-agentic-mtp` (branch `local/glm53-mi355x-sglang-l1`), rendered by `infx generate` |
| Recipe | copy of `glm5.2/sglang/mi355x-fp8-mtp/agentic.yaml` with GLM-5.3's served name and EAGLE geometry (3 steps, topk 1, 4 draft tokens); overrides c1, c2, c4 |
| Client | AgentX `agentx-harness 1.0.12`, corpus `semianalysisai/cc-traces-weka-062126` snapshot `23f152f6f0f9`, 900-s window, scenario locks as shipped |
| Acceptance | golden AL 2.99 (`infx.golden_al_distribution lookup glm5.3 mtp 3`), `SGLANG_SIMULATE_ACC_*` from `infx generate` |

## Arms (render_arms.py; each delta checked with `render_arms.py diff`)

| Arm | Delta from the recipe | Container | Route log |
|---|---|---|---|
| base | none (fp8_e4m3 KV) | infx-glm53-l1 | absent |
| cand | `kv-cache-dtype=auto` (bf16 KV), `monokernel-backend=amd` | infx-glm53-l1 | present |
| b0 (diagnostic) | `kv-cache-dtype=auto` | infx-glm53-l1 | absent |
| b1 (diagnostic) | as cand | infx-glm53-l1-img | present |
| baseq, candq (quality) | base, cand with `--real-acceptance` | infx-glm53-l1 | as base, cand |

The glm53 spec accepts only `auto` or `bfloat16` KV, so the candidate necessarily changes
the KV dtype; b0 separates the KV effect from the kernel effect. b1 shows whether the
image as shipped already carries the gain (recipe-only route, no SGLang pin bump).

## Order

Round 1: c1 base, cand; c2 base, cand; c4 base, cand. Round 2 reverses the whole
sequence. Then GSM8K baseq, candq at c2. Then b0 and b1 at c1 and c2 (one round).

## Coverage

The kernels run steps of up to 8 rows at TP8; MTP K=3 is 4 rows a request, so a step
with at most 2 running requests runs on the kernels. AgentX sessions think and idle, so
even c4 mostly runs 1-2 requests (an earlier AgentX c4 run had about 92 % of its verify
steps on the kernels); c4 is not an uncovered control. `l1_check.py runs` records each
run's share of decode steps with at most 2 running requests (`cov`).

## Metrics and decision

- Primary, per concurrency: total tok/s/GPU, P90 E2E-normalized interactivity, P90 E2EL
  (`compare_agentx.py`, InferenceX field names).
- Noise floor: the A/A pair base c1 r1 vs base c1 r2 (`--aa`). A lead needs the same
  sign in both rounds and a size beyond the A/A difference; otherwise "difference too
  small to establish a lead".
- Go for the upstream route: candidate ahead at c1 and c2 on interactivity or E2EL with
  no baseline lead on throughput at the same points, c4 not behind beyond noise, the
  quality gate passed, and every timing check passed.

## Per-run validity (an attempt that fails one is kept but is not a completed point)

Startup, health, served model name; the 900-s window; request accounting (successful,
warmup-dropped, error-dropped) within the 0.10 failed-request threshold; the InferenceX
validator; the realized `accept len` tight around 2.99 (a miss invalidates the arm); the
route log as the table above; no foreign GPU process or activity (gpumon every 10 s);
memory released after the stop.

Amendment 2026-10-25T07:50Z (after round 1, before round 2; validity rule only, decision rule
unchanged): "the 900-s window" means the client log shows AIPerf's profiling phase reaching
its 900-s target (a completion, or a drain timeout after the grace period).
`duration_seconds` spans the completed requests only (round 1: 879-928 s), so it is not
checked against 900 s. Requests cancelled by a drain timeout count with the profiling
window's failed requests (the validator's count) against the 0.10 threshold;
`records_error_dropped` is not used because it also counts warmup records (round 1: 41 of
its 42 error-dropped records were warmup records, all `InvalidInferenceResultError`, "no
responses with actual content").

## Quality gate

InferenceX GSM8K (lm-eval, 5-shot, greedy) on the first 500 test questions
(`EVAL_LIMIT=500`), real acceptance, concurrency 2. Both arms must reach InferenceX's
`glm5` threshold 0.94; report the paired difference (McNemar) on the same questions.
500 questions give about 1.1 pp standard error.

## Deviations from the official flow (all disclosed in the report)

1. The config and recipe are local and unmerged; the candidate mounts an sgl-amd-mono
   overlay, so its numbers are L1 only (an L2 run needs a published image).
2. The AgentX venv is built once for the cohort and reused; the official job rebuilds it.
3. Power telemetry is off (`ENABLE_AGENTX_POWER=0`).
4. GSM8K runs 500 of 1319 questions.
5. The server runs with `HF_HUB_OFFLINE=1` and a local model path; the client resolves
   the corpus online and records the snapshot.

## Stop conditions

A busy node (idle check fails) waits and retries; a route mismatch stops the queue; a
contaminated attempt reruns at most twice; any other failure reruns once, then stops the
queue with `state/ALERT.txt`.
