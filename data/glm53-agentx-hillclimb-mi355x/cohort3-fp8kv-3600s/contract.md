# GLM-5.3 on MI355X, InferenceX ladder at 3600 s on the sgl-amd-mono #60 overlay (cohort 3)

Written 2026-10-26 before the first measured run. Local only: nothing is pushed or posted.

## Question

With sgl-amd-mono main `0a107f1` (#60: glm53 on SGLang's fp8_e4m3 MLA cache), what does the
GLM-5.3 MI355X SGLang agentic config score on InferenceX's own metrics at the official
3600-s window, with and without `--monokernel-backend amd`, over the ladder InferenceX's
GLM-5.2 MI355X FP8 SGLang config publishes without KV offload (c1, c2, c4, c10)? And does
c2, which cohort 2's 900-s windows left inside the noise, show a lead?

This is not the skill's L2: L2 needs a published image without an overlay, and no
published image carries #60 yet (SGLang pins sgl-amd-mono `05e0bb02`).

## Cohort

As cohort 2 (`../infx-l1-glm53-fp8-20261025/contract.md`), except:

| Field | Value |
|---|---|
| sgl-amd-mono | main `0a107f1` (tree `1c9ff3c`), worktree `trees/mono-fp8kv` at detached `0a107f1`, mounted in `infx-glm53-fp8` for both arms |
| Config | local `glm5.3-fp8-mi355x-sglang-agentic-mtp`, conc-list [1, 2, 4, 10]; the c10 override copied from the GLM-5.2 sibling (max-running 20, cuda-graph-max-bs-decode 20) |
| Window | 3600 s (`--duration 3600`), the official AgentX duration |

## Arms

base (the recipe, fp8_e4m3 KV) and cand8 (the recipe plus `monokernel-backend=amd`), each
rendered by `render_arms.py` from the same `infx generate` output (`recipes/`); baseq and
cand8q (`--real-acceptance`) for the quality gate.

## Order

c1 base, cand8; c2 base, cand8; c4 base, cand8; c10 base, cand8; then c2 base again (the
A/A pair with the first c2 base); then GSM8K baseq, cand8q at c10.

## Metrics and decision

- InferenceX metrics per point (`compare_agentx.py`): total tok/s/GPU, P90 E2E-normalized
  interactivity, P90 E2EL; noise floor = the c2 A/A pair. A lead needs a gain beyond the
  floor (one round per point at 3600 s).
- Report against published InferenceX rows as context only, never as an arm: GLM-5.3
  MI355X TileRT (c1, disaggregated, 16 GPUs), GLM-5.2 MI355X SGLang FP8 / FP4 and ATOM
  FP4 (same architecture, other weights and draft depth), GLM-5.3 B300 SGLang FP4.
- Supporting: request-paired TTFT / ITL / E2EL against the base runs of each point, and
  the slowest 10 % of E2EL/OSL; `pareto_coverage` on both curves.
- Covered points: c1 and c2 (at most 2 running requests, 8 rows); c4 mostly; c10 is
  mostly wider than 8 rows and runs the standard path (a control: it must not regress).

## Validity

As cohort 2, with the profiling phase target 3600 s.

## Quality gate

InferenceX GSM8K, all 1319 test questions, real acceptance, concurrency 10 (as InferenceX
CI runs agentic GSM8K at the highest concurrency of a group); both arms at least 0.94
under every lm-eval filter; paired McNemar reported.

## Stop conditions

As cohort 1.
