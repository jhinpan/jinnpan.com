# Ranked per-cycle budget (BS1, DSpark block 5, genuine acceptance)

Denominator: unprofiled natural-generation cycle of the PA-PR baseline,
event-timed by `bs1_cycle` (`results/cycle-r1/1-baseline`, 4 ranks x 4,352 cycles):

| Segment | ms/cycle | Share |
|---|---:|---:|
| VERIFY graph (1,202 kernels) | 7.443 | 83.4% |
| DRAFT graph (145 graph kernels) | 0.790 | 8.9% |
| GPU idle, verify end -> draft start | 0.647 | 7.3% |
| GPU idle, draft end -> verify start | 0.040 | 0.4% |
| Sum | 8.921 | 100% |

Targets at unchanged acceptance: +5% needs 424 us/cycle, +10% needs 809 us/cycle.

## Cost model (measured, not assumed)

- HIP graph launch floor on MI355X/ROCm 7.2: 1.54-1.84 us per dependent trivial
  kernel, 1.8 us with a 16 KiB producer->consumer hand-off (`results/microbench-r2`).
- Removing one small kernel boundary saves about 2 us: MoE no-clear removed 40
  clears for 70-81 us in the fixed-state stage replay (BS1 campaign ledger item 5).
  One kernel per layer across 40 layers is therefore about 80 us/verify (0.9%).
- Kineto slows every graph kernel to at least ~4.2 us (`results/kineto-cal-r1`);
  profiled small-kernel durations overstate their unprofiled cost.
- Multi-stream graph branches lose packet capture and cost ~4x per kernel; a wait
  pending in another HW queue slows every dispatch of the running graph by
  ~1.34 us (`results/microbench-r2`, `results/queue-interference-r1.json`).
  Concurrency is not a BS1 lever on this stack.

## Ranked candidates

| Rank | Item | Budget us/cycle | Evidence | Status |
|---:|---|---:|---|---|
| 1 | Host-blocked GPU idle between verify and draft (HIP `publish_ready.synchronize()`, #26672) | 647 | `cycle-r1`, E3 | E6 single-queue scheduling: -526 us diagnostic; scored ABBA running |
| 2 | Residual eager DRAFT metadata launches after E6 | ~147 | `cycle-r2` candidate gap | after E6 lands |
| 3 | Draft custom collectives (12 calls, rank-arrival waits) | ~100-177 | ledger DRAFT row | depends on E6 host jitter change |
| 4 | Verify kernel-count reductions, ~80 us per kernel per layer | 80 each | cost model | MoE no-clear (40 clears), WO-A MXFP8 epilogue switch, WO-A partial+reduce, mHC pair, SiLU epilogue, router chain |
| 5 | Verify head/tail glue (84 kernels outside layers, ~410 us est.) | ~60-120 | ledger head/tail rows | consolidation |
| 6 | Verify-tail RCCL collectives (3 calls) | ~15-20 | ledger tail row | custom collective or replicated compute |
| 7 | MoE EP4 -> TP4 topology (slow-rank skew) | unknown | KW2 skew evidence | separate topology A/B |
| 8 | Above-floor compute: G1/G2, PA main, LM head BF16, shared down BF16 | per kernel | ledger | after structural items |

Full per-family ledger: `analysis/kernel-ledger.csv`, summary `analysis/kernel-ledger-summary.json`
(profiled durations scaled uniformly to the unprofiled phase totals; tiny-kernel
rows are upper bounds).
