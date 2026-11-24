# Experiment 008 · FlyDSL kernels and the compiler: measurements behind the co-development RFC

Raw data behind `public/sources/flydsl-compiler-redesign.html`. Measured 2026-11-22 and 2026-11-23 (UTC).
Rebuild the page from this directory alone with `python3 data/flydsl-compiler-redesign/scripts/make_page.py`.
The author's own text (position, toy example) is `AUTHOR_TEXT` in `scripts/make_page.py`; while an entry is empty,
the page shows a draft marker there and the build prints `DRAFT`.

## Protocol

- Kernel source: FlyDSL `cb770e9a96b3454fa9cbbbcdc16dceeda1d8e4e7` (v0.4.2), the pin that FlyBench v1 scores.
  Every arm is one patch in `patches/` against that tree, applied on top of the task's stock baseline.
- Scoring: the FlyBench v1 mode-A evaluator, the same one that scores agent submissions. Candidate and stock
  run in fresh worker processes for each of 6 ABBA pairs; each cold call follows a 512 MB cache-flushing
  read; warm blocks follow. Shapes: the record's public timed shapes plus freshly drawn ones (`t*`). Gates:
  FlyDSL's own tests for the kernel must give the stock kernel's pass counts, and every timed call must pass the
  task's output check.
- Hardware: one AMD Instinct MI355X (gfx950) per arm, one model-free job per arm through the AMD Pilot v2 job
  queue. The queue places single-GPU jobs beside other jobs on the same node. Pairing inside one job on one GPU
  removes drift between the two sides, but not noise from neighbours; the reported intervals include it.
- ISA: a separate step in each job compiled stock and arm at the first timed shape and at the largest shape
  with `FLYDSL_DEBUG_DUMP_ASM=1` and the runtime cache off, then `scripts/isa_compare.py` counted registers,
  spills, instructions and order. The null control `x12` produced byte-identical ISA.
- Compile cost (`compile_breakdown.csv`): CPU only, FlyDSL `fe81c407` (v0.5.2), target gfx950, compile only,
  3 runs per kernel within 1%, `scripts/measure_compile.py`.
- Agent corpus (`agent_mining_summary.json`): earlier FlyBench mode-A runs on the same fleet, read by
  `scripts/mine_reviews.py`. Only aggregates are published.

## Arms

- `x*` (experiment A): remove one explicit decision the kernel author made and hand it back to LLVM.
  `nosched` turns every `rocdl.sched_*` hint into a no-op; `nogroup` keeps `sched_barrier` and drops the
  interleave groups; `noprio` turns `s_setprio` into a no-op; `postmischedon` deletes the author's
  `enable-post-misched: False`; `agprllvm` replaces the AGPR-pinned MFMA atom with the plain one;
  `occdefault` deletes the `waves_per_eu` and `flat_work_group_size` hints; `null` installs the hint shim
  and changes nothing.
- `h*` (experiment B): change one LLVM option on kernels whose authors set no schedule hints:
  `amdgpu-sched-strategy` (`max-ilp`, `max-memory-clause`) or both machine schedulers off
  (`enable-misched=0`, `enable-post-misched=0`).
- `b` suffix: a repeat of the same patch in a new job.

## Metrics

- `speed_vs_stock`: stock time / arm time, the evaluator's geometric mean over all timed and drawn shapes.
  Below 1 means the arm is slower than the stock kernel. `ci95_lo` and `ci95_hi`: its 95% interval over the 6
  pairs. `noise_floor`: the record's A/A floor from FlyBench validation (stock against stock: the largest
  absolute log ratio plus half-width). FlyBench calls an arm improved only when `ci95_lo` exceeds
  1 + `noise_floor` and no timed shape regressed.
- `regressed_shapes`: timed shapes whose whole 95% interval lies below 1 minus that shape's noise floor.
- `verdict`: `scored` (correct, timed), `tests_failed` (FlyDSL's tests disagree with stock), `incorrect`
  (a timed call failed the output check), `unscorable` (the stock kernel itself failed a check in that job).

## Layout

- `arms.csv`: label, record, arm, what changed, patch path.
- `results.csv`: one row per arm with the metrics above and the evaluator's reason string.
- `per_shape.csv`: one row per arm and shape: speed, interval, stock and arm time in microseconds, drawn flag.
- `isa.csv`: per arm, dumped shape, kernel and side: VGPR, SGPR, spills, scratch bytes, instruction counts,
  and whether the instruction order equals stock.
- `summary.json`: everything above in one file; the page reads it.
- `patches/`: the exact patch of every arm.
- `census_fe81c407.txt`: the source census of FlyDSL `fe81c407` cited in section 04 (pipeline passes, LLVM
  patches, hint counts).
- `scripts/`: `build_experiments.py` (patches and evaluator scripts), `isa_compare.py`, `summarize.py`,
  `mine_reviews.py`, `measure_compile.py`, `census_and_isa.py`, and the page builder (`make_page.py`,
  `build_blog.py`, `page_css.py`, `page_plates.py`).

## Not committed

Evaluator receipts and logs, per-call timing samples, ISA dumps, agent transcripts and reviews, job IDs, host
names, and the scripts that submit and fetch jobs through the overlay's control plane. Local paths are written
as `$WORK`, `$HOME`, `$REPORTS`, `$AMDPILOT_DEPLOY`, and `$BENCH_RUNS`.
