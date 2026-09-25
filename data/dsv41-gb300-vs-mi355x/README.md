# Experiment 005 · DeepSeek-V4.1 Flash, BS=1, GB300 vs MI355X

Raw data behind `public/sources/dsv41-gb300-vs-mi355x.html`. Measured 2026-09-24/25 (UTC).

## Protocol (identical on both machines)

- Model `deepseek-ai/DeepSeek-V4.1-Flash@dba1be0a`, TP4 / EP4, each platform's cookbook cell:
  Low-Latency (DSpark block 5, decode graph cap 64, `--mem-fraction-static 0.8`) or
  High-Throughput (DSpark off); `--disable-radix-cache --random-seed 42` on both.
- Prompt and client: BBuf's random-dspark assets at
  https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/b001f347bab8468d1299fba754c15be9efa0f900/large-language-model/sglang/assets/deepseek-v41-kernel-journey/random-dspark
  (`prompt.json` sha256 `2331eb918994e153dc659a4bd97f5eca87af7751268a5d931a1863a865801307`,
  `benchmark.py` sha256 `8a9276345df9fa7ae71c3f11f73fb59ac02ffcbc82a47d6b25455346ece89952`):
  4,096 random ids, 1,024 output tokens, temperature 0, `ignore_eos`, cache flushed per request,
  one discarded warm-up and six timed rounds per server, a fresh server per launch.
- Modes: `off` (plain decode), `sim` (`SGLANG_RAGGED_VERIFY_MODE=static SGLANG_SIMULATE_ACC_LEN=5.5
  SGLANG_SIMULATE_ACC_METHOD=match-expected`), `real` (natural acceptance).

## Metrics (per timed round, see `analysis/analyze.py`)

- `output_tps` = (completion tokens - first-event tokens) / (last event - first event)  [BBuf's metric]
- `cycle_ms` = (elapsed_s - ttft_s) / verify_steps  [DSpark; independent of acceptance length]
- `step_ms` = (elapsed_s - ttft_s) / (output tokens - first-event tokens)  [plain decode]

## Layout

- `gb300/arms/<arm>/`: `rounds.csv` (per round, output text replaced by its sha256), `summary.json`,
  `status.json`, `manifest.json` (image digest, server command, env, docker args, nvidia-smi summary),
  `docker_run.txt`, `placement_*.txt` (scheduler CPUs, per-thread node, numa_maps), `gpu_samples.csv`
  (200 ms clocks/power in the NUMA control), `server_record_lines.txt`.
  - `s0-*` author's commit 835c3909 on `dev-dsv41` (tp4 = MoE TP4/EP1, ep4, nods = DSpark off), unbound;
    `n0-*` the same with SGLang NUMA binding (`--cap-add SYS_NICE`).
  - `s1a-*` main ffac53d779 (`lmsysorg/sglang@sha256:b8257f5c...`, dev-cu13) with cookbook cells, unbound;
    `n1a-*` the same, NUMA-bound; `e1-base-sim-*` / `e1-nice-sim-*` three-launch NUMA control.
  - `p2-real`, `p2-off`: nsys captures (100 verify cycles, 200 decode steps) with `attribution.json`
    from `gb300/scripts/attribute.py` (the .nsys-rep and sqlite exports are not committed: 115 MB / 320 MB).
- `gb300/scripts/`: the drivers (`arm.py`, `step*.sh`, `exp_numa.sh`, `profile_arm.py`, `attribute.py`).
- `gb300/system/`: lscpu, topology, nvidia-smi (serials removed).
- `mi355x/arms/<arm>/`: `rounds.csv`, `summary.json`, `status.json`, `manifest.json`, `processes.json`
  (every server process: allowed CPUs, NUMA node of resident pages), `server_record_lines.txt`.
  - `u-*` / `n-*` (session 1): SGLang's inherited `SGLANG_SET_CPU_AFFINITY=1` split affinity; `n-*` also
    pins the launcher process tree to node 1 with taskset (SGLang then re-pins the schedulers).
  - `s2-{split,node1,none}-*` (session 2): split affinity; SGLang affinity off plus the whole server on
    node 1; SGLang affinity off and no pinning.
- `mi355x/leases/<lease>/`: the exclusive eight-GPU lease of each session: idle-picker table, HBM
  canaries before and after (256 MiB device copy per GPU), KFD process lists, the lease verdict
  (`exit.json`) and every PID the 5 s process monitor saw (`process-monitor-summary.csv`).
  Session 2's verdict is `contaminated: true` because GPU 4's canary read 10% *faster* after the
  session (6.48 against 7.15 ms). The session was kept: every monitored PID appears and disappears
  inside one of our server launches, the other three canaries moved by under 0.3%, contention can
  only lengthen a copy, and the split-affinity cycles match session 1 within 0.2%.
- `mi355x/scripts/`: `run_arm.sh`, `run_arm2.sh`, `session*.sh`, `lease.sh` and the e2e824 launchers/env.
- `mi355x/prior-base-cycle-ledger/`: the BS=1 verify-cycle ledger from the previous MI355X base
  (genuine acceptance; profiled per-family shares scaled to unprofiled phase totals).
- `analysis/`: `arms.json` (every group), `attribution_compare.json` (both machines in one category
  scheme), `gb300_placement.json`, and the scripts that produced them and the page.

Hostnames, account names, IP addresses and machine-local paths are replaced by placeholders
(`$GB300_WORK`, `$CAMPAIGN`, `$E2E_CAMPAIGN`, `$MODEL_ROOT`, `gb300-tray`, `mi355x-node`, `<ip>`).
