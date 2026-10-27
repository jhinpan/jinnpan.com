# Experiment 006 · GLM-5.3 monokernel hill-climbing on MI355X (retrospective)

Raw data behind `public/sources/glm53-agentx-hillclimb-mi355x.html`. Measured 2026-10-20 to 2026-10-27 (UTC).
Rebuild the page from this directory alone with `python3 data/glm53-agentx-hillclimb-mi355x/scripts/make_page.py`.

## Protocol

- Hardware: one node with 8x AMD Instinct MI355X (gfx950). Every run held all 8 GPUs under a lock after a
  30-sample idle check, sampled `amd-smi` every 10 s, and checked for foreign processes.
- Model: `zai-org/GLM-5.3` at `ba32f8c19c8738e2c57c71798c601d477a7bf844` (FP8), TP8 EP1, EAGLE MTP with
  3 steps, top-k 1, and 4 draft tokens, `--kv-cache-dtype fp8_e4m3` unless the arm says BF16.
- Image: `lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261024`
  (`sha256:65a81d6ba18d32e00161a0feed0b889fb604505daeb56ae3610da92dc52f4915`): SGLang `fe1bd108372c`,
  AITER `285579b6211b`, bundled sgl-amd-mono `05e0bb02`.
- Kernel arms: cohort 1 mounts sgl-amd-mono `9484c60` (main after #42, BF16 KV only); cohorts 2 and 3
  mount `0a107f1` (main after [#60](https://github.com/sgl-project/sgl-amd-mono/pull/60), FP8 KV).
  The monokernel is enabled with `--monokernel-backend amd`, the image's spelling of `--enable-amd-monokernel`.
- Client: InferenceX AgentX (`agentx-harness 1.0.12`, AIPerf) from `SemiAnalysisAI/InferenceX@cfb7bdd0c`
  plus the local config in `inferencex-config/`, corpus `semianalysisai/cc-traces-weka-062126` snapshot
  `23f152f6f0f9`. Windows: 900 s (cohorts 1 and 2), 3600 s (cohort 3). Synthetic acceptance in both arms:
  `SGLANG_SIMULATE_ACC_LEN=2.99`, `SGLANG_SIMULATE_ACC_METHOD=match-expected`,
  `SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token`.
- A fresh server per point and arm, with no generation request before the timed phase.
- GSM8K: InferenceX's eval (lm-evaluation-harness) with real MTP acceptance; 500 questions in cohorts 1
  and 2, all 1319 in cohort 3, plus a rerun of each arm at concurrency 10.
- Each cohort's decision rule is in its `contract.md`, written before its runs.

## Ablation

- BF16 ladder (`ladder-bf16-3600s/`): the PR campaign. Image `lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261019`,
  SGLang `e45cbce599`, 3600 s, BF16 KV in both arms (the monokernel accepted only BF16 KV before #60), simulated
  accept length 2.99. Arms: `mtpref` (normal path with MTP), `mtpmono` (#15), `mtpptopk` (#24), `mtpscore` (#42).
- FP8 ladder (`ladder-fp8-900s/`): the image and client of cohort 2, 900 s, the recipe's FP8 E4M3 KV. F1 is
  `05e0bb02` (#7, #15) with #60 cherry-picked (`421c124`; the e2e test file keeps only #60's FP8 check), F2 is
  `240d587` (#24) with #60 cherry-picked (`8671f43`), F3 is cohort 2's monokernel arm. Each build passed the FP8
  layer check (`fp8kv-validation/e2e_checks.json`). A new base run at concurrency 2 checks for drift.
- `ablation/ablation.json`: per run, the request-paired effects against the base runs of the same ladder, the
  exact accept length, the coverage, and `by_isl` (the paired ITL change in four input-length bins: below 64k,
  64k to 128k, 128k to 192k, 192k or more). Per ladder, `by_isl` also pools concurrency 1 and 2; F3 pools its two
  runs. `ablation/compare/`: compare_agentx.py outputs per ladder step.
- Compare the two ladders only within one input-length bin. The 900-s and 3600-s windows hold different shares of
  long requests (at concurrency 1: 37% and 52% at 192k or more), and #15 alone is slower above 192k.
- `prs/timelines.json`: the kernel-timeline tables of #24 and #42, parsed from the PR descriptions (`prs/pr*.md`).

## Metrics

- Headline (`analysis/compare_*.json`, InferenceX's `compare_agentx.py`): `tput_gpu` (total tok/s/GPU),
  `e2e_int_p90` (1 / P90 of E2EL / OSL, tok/s/user), `e2el_p90`, `ttft_p90`. A latency gain is
  base / candidate - 1. `noise_floor` comes from the A/A pair.
- Matched requests (`analysis/pair_effects.json`, `scripts/l1_pair.py`): requests present in every run
  with input lengths within 8 tokens; effect = 100 x (exp(median of log x - mean log of the base runs) - 1).
  `tail` repeats this on the slowest 10% and 20% of base requests by E2EL per output token.
- Coverage (`runs/*/run.json`, key `coverage`): share of the server's `Decode batch` log lines with at
  most 2 running requests, the steps the glm53 kernel serves, and the histogram by running requests.

## Layout

- `cohort1-bf16kv-900s/`, `cohort2-fp8kv-900s/`, `cohort3-fp8kv-3600s/`:
  `contract.md`; `analysis/*.json`; `runs/<run>/run.json` (server args, env, client, AgentX summary,
  coverage, GSM8K summary); `arms/` and `recipes/` (rendered InferenceX arms); `gsm8k/<run>.csv`
  (per-question correctness and response length).
- `cohort3-fp8kv-3600s/requests/<run>.csv`: every request of every 3600-s run (phase, conversation, turn,
  ISL, OSL, TTFT, ITL, E2EL, start time); `runs/<run>/aiperf_summary.json`: the AIPerf summary.
- `ttft-profile/summary.json`: one prefill of about 100k cached + 2048 new tokens, rank 0, FP8 E4M3 KV
  against BF16 KV, neither with the monokernel; TTFT samples and kernel totals.
- `fp8kv-validation/e2e_checks.json`: the sgl-amd-mono e2e layer checks of #60 (KV and stream cosine,
  violations), including the negative control.
- `references/`: published InferenceX rows (`refs.json`, `derived.json`, `osl_published.json`) from
  `https://inferencex.semianalysis.com/api/v1/benchmarks?model=GLM-5.2` and
  `/api/v1/derived-agentic-metrics`.
- `inferencex-config/`: the local `amd-master.yaml` diff and the `mi355x-fp8-mtp` recipe.
- `scripts/`: `make_page.py` (this page), `ablation_analysis.py` (both ladders), `l1_check.py` and `l1_pair.py`
  (validity and matched-request analysis; they read the uncommitted raw run directories).
- `index.json`: run index per cohort.

## Not committed

Raw AIPerf exports (about 475 MB per run), server logs, profiler traces (3.4 MB each), GSM8K samples with
model outputs (15 MB per run), and the raw API responses (re-fetchable at the URLs above). Local paths are
written as `$WORK`, `$HOME`, and `$CACHE`, and the host name as `mi355x-node`.
