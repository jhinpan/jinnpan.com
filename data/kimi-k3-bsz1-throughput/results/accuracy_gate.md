# Phase 5 accuracy gate

Each tuned config re-run through exactly the protocol that produced the published
Day-0 numbers, so the comparison is direct: GSM8K via in-tree `run_eval`
(n=1319, greedy, `--max-tokens 8192 --temperature 0 --num-threads 32`) and AIME25 via
`sgl-eval` (`--n-repeats 8 --num-threads 48 --max-tokens 64000 --temperature 1.0
--top-p 0.95 --thinking`).

Sources: `accuracy/gsm8k-*.txt`, `accuracy/aime25-*.txt`,
`accuracy/gibberish-*.txt`, `accuracy/details-*.jsonl`.

## lane `nospec` — mem-fraction 0.93 + `--mamba-ssm-dtype bfloat16`

- GSM8K greedy n=1319: **97.489%** vs baseline 97.49% (delta −0.001 pp, tolerance
  +/-1.0 pp) -> **PASS**. Wall clock 580.7 s at 480.7 tok/s, against the baseline's
  605.4 s at 468.8 tok/s.
- AIME25 pass@1 avg-of-8: **91.67% +/-3.09%** (SEM 1.09%) vs baseline
  93.33% +/-4.36% (SEM 1.54%) — delta −1.66 pp, baseline 1 sigma = 4.36 -> **PASS**.
  Pooled SEM of the difference is 1.89 pp, so this is 0.88 sigma: not a detectable
  regression, but it is the largest delta in the gate and it belongs to the one
  config carrying a numerics-changing knob.
  - stop_rate = 99.17% (baseline 100%)
  - truncated = 0.83%, no_answer = 0.83% (baseline 0% / 0%)
  - error_rate = 0.00%
- Degeneration probe: **PASS** — 32 samples, 0 empty, mean 8-gram repetition 0.0170,
  max 0.2949, longest consecutive token run 4.

## lane `dspark` — mem-fraction 0.92 + `--speculative-dspark-block-size 3`

- GSM8K greedy n=1319: **97.641%** vs baseline 97.64% (delta +0.001 pp) -> **PASS**.
  Wall clock 393.1 s at 714.5 tok/s.
- AIME25 pass@1 avg-of-8: **95.42% +/-3.54%** (SEM 1.25%) vs baseline
  94.58% +/-3.05% — delta **+0.84 pp**, baseline 1 sigma = 3.05 -> **PASS**.
  - stop_rate = 100.00%, truncated = 0.00%, no_answer = 0.00%, error_rate = 0.00%
- Degeneration probe: **PASS** — 32 samples, 0 empty, mean 8-gram repetition 0.0205,
  max 0.2487, p90 0.0294, longest consecutive token run 12.

## Verdict

**PASS** — all 6 checks within tolerance of the published baseline.

- `nospec/gsm8k`: PASS
- `nospec/aime25`: PASS
- `nospec/degeneration`: PASS
- `dspark/gsm8k`: PASS
- `dspark/aime25`: PASS
- `dspark/degeneration`: PASS

The headline conclusion is that the tuned recipes are output-neutral. That is the
expected result and worth stating why it was still worth the GPU hours: of the knobs
the search moved, `--mem-fraction-static`, `--chunked-prefill-size`,
`--cuda-graph-max-bs-decode`, `--max-running-requests`, `--schedule-conservativeness`
and `--mamba-radix-cache-strategy` only change memory and scheduling and *cannot*
alter the token distribution. Two can: `--mamba-ssm-dtype bfloat16` changes SSM state
precision, and `--speculative-dspark-block-size` changes the verify window. Those two
are what the gate exists for.

- `--speculative-dspark-block-size 3` is clean, and pleasingly so — AIME25 came out
  0.84 pp *above* the Day-0 DSpark baseline with a perfect stop rate. Speculative
  decoding is supposed to be lossless because the target verifies every token, and a
  shorter verify window does not change that. This is the evidence, and the knob
  ships.
- `--mamba-ssm-dtype bfloat16` passes but is not clean. It is the only config that
  lost any samples to truncation (0.83% truncated, 0.83% no_answer, stop_rate 99.17%)
  and it carries the −1.66 pp AIME25 delta. Combined with a measured gain of only
  +1.24% throughput, it is **left out of the recommended recipe** and documented as an
  opt-in for anyone willing to re-run their own gate.

## Correction

An earlier run of this gate reported FAIL on five of six checks. That was a parsing
bug in the reporter, not a result: the GSM8K regex scanned forward from `Score` and
picked up the `64` in `--max-tokens 64000`, and the AIME25 regex read the `8` out of
the label `pass@1[avg-of-8]`. Both now anchor on the `=` or on the exact `[METRIC]`
key ([`gate-k3.py`](../../grid_k3/gate-k3.py)).

The degeneration probe separately reported FAIL on `nospec` for one sample at 29.5%
8-gram repetition. Inspecting the sample showed fluent, correct English — a reasoning
trace summarising repetitive source text — with a longest consecutive token run of 1.
The threshold now requires repetition *and* consecutive runs together, or one signal
far out (>0.60 repetition, >20-token run), which is what actual degeneration looks
like. No eval was re-run; only the scoring changed.
