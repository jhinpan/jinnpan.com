#!/usr/bin/env python3
"""Block-level kernel comparison: TokenSpeed vs SGLang, Kimi-K3 decode, bs=1, ISL 4096.

Both sides are graph-off device-kernel time summed per decode step, with
collectives excluded (in eager decode the all-reduce payload is tiny and the
kernel busy-waits, so its duration measures skew, not work — the SGLang
baseline excluded it for exactly this reason).

TokenSpeed kernels are attributed to blocks using the source modules that
launch them, not name heuristics. `_kimi3_projection_gemv_kernel` is the one
kernel shared across blocks: it backs kimi3_latent_projection (x2),
kimi3_shared_down_projection and kimi3_router_projection in the MoE layers
(4 x 92 = 368 dispatches) plus kimi3_qkvfab_projection in KimiLinearKDA
(69 dispatches) = 437, matching the measured dispatch count exactly. Its time
is split by dispatch share.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

# kernel name -> (block, note). Verified against the launching source module.
ATTRIB: dict[str, tuple[str, str]] = {
    # --- collectives (excluded from compute totals)
    "iris_stage_one_shot_allreduce_kernel": ("collective", "ops/communication/iris.py"),
    "iris_stage_one_shot_allreduce_two_gluon_kernel": ("collective", "ops/communication/iris.py"),
    "ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)": ("collective", "RCCL"),
    # --- MoE: Gluon a16w4 SiTU expert GEMMs
    "_stage1_a16w4_situ_warp_gemv": ("moe", "amd ops/moe/gluon_a16w4_situ_decode.py"),
    "_stage2_a16w4_warp_gemv_combine": ("moe", "amd ops/moe/gluon_a16w4_situ_decode.py"),
    "_kimi3_shared_situ_projection_gemv_kernel": ("moe", "ops/gemm/kimi3.py shared expert"),
    "_kimi3_sigmoid_bias_topk_kernel": ("moe", "ops/moe/triton/kimi3_sigmoid_topk.py"),
    # --- KDA linear attention
    "_kda_recurrent_decode_kernel": ("kda", "ops/attention/triton/kda.py"),
    "_causal_conv1d_update_kernel": ("kda", "ops/attention/triton/kda.py"),
    "_rmsnorm_gated_kernel": ("kda", "KDA gated output norm"),
    # --- full MLA attention
    "_mla_decode_gluon": ("full_attn", "amd ops/attention/gluon/mla_decode_gfx950.py"),
    "_mla_softmax_reducev_kernel": ("full_attn", "MLA softmax/reduce-V"),
    "_mla_nope_query_fp8_kernel": ("full_attn", "MLA NoPE query fp8"),
    "set_mla_kv_buffer_kernel": ("full_attn", "MLA KV write"),
    "_sigmoid_mul_kernel": ("full_attn", "MLA output gate (24/step)"),
    # --- attention residual
    "_attnres_partial_dual_kernel": ("attn_residual", "ops/activation/triton.py"),
    "_attnres_combine_kernel": ("attn_residual", "ops/activation/triton.py"),
    "_attnres_partial_kernel": ("attn_residual", "ops/activation/triton.py"),
}

# Vendor GEMMs attributed by dispatches/step: 69 -> KDA layers, 24 -> MLA layers.
VENDOR_BY_COUNT = {69: "kda", 24: "full_attn"}

# The 93-dispatch vendor GEMM is the attention output projection issued once per
# layer, so it spans both attention types (69 KDA + 24 MLA). SGLang's own study
# found the same structure: exactly one compute kernel shared between KDA and
# MLA, its output projection. Split by dispatch share to keep the block
# comparison like-for-like with SGLang, whose "projections" sit inside blocks.
VENDOR_SHARED_COUNT = 93
VENDOR_SHARED_SPLIT = {"kda": 69 / 93, "full_attn": 24 / 93}

# The one cross-block kernel, split by dispatch share (source-verified).
SHARED_GEMV = "_kimi3_projection_gemv_kernel"
SHARED_SPLIT = {"moe": 368 / 437, "kda": 69 / 437}

# SGLang baseline, ISL 4096, bs1, graph-off, us/step (decode-composition.csv +
# decode-block-internals.csv from data/kimi-k3-kda-time-share).
SGLANG = {
    "moe": 10548.5,
    "kda": 3542.3,
    "full_attn": 3100.1,
    "attn_residual": 2287.7,
    "other": 1290.3,
}
SGLANG_INTERNALS = {
    "moe": [("shared_expert_gemms", 3837.9), ("route_sort", 2956.4),
            ("misc", 2316.9), ("expert_gemms", 1437.3)],
    "kda": [("projections", 2235.2), ("recurrent_state_update", 447.3),
            ("conv_short", 297.2), ("misc", 294.5), ("gated_out_norm", 268.1)],
    "full_attn": [("projections", 1029.2), ("split_reduce_stage2", 904.8),
                  ("misc", 858.5), ("kv_scan_stage1", 307.6)],
    "attn_residual": [("_score_kernel", 1500.4), ("_combine_kernel", 787.3)],
}
SGLANG_STEP_GRAPH_ON_MS = 19.387          # graphon_itl.json, 4k
TOKENSPEED_STEP_GRAPH_ON_MS = 22.19       # measured TPOT, 4096/1024 conc 1


def main() -> None:
    src = Path(sys.argv[1])          # report-decode/kernels-cross-rank.csv
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(src.open()))
    blocks: dict[str, float] = {}
    launches: dict[str, float] = {}
    detail: list[tuple] = []
    unattributed: list[tuple[str, float]] = []

    for r in rows:
        name = r["kernel"]
        us = float(r["mean_us_per_step"])
        n = float(r["dispatches_per_step"])

        if name == SHARED_GEMV:
            for blk, frac in SHARED_SPLIT.items():
                blocks[blk] = blocks.get(blk, 0.0) + us * frac
                launches[blk] = launches.get(blk, 0.0) + n * frac
                detail.append((blk, us * frac, n * frac,
                               f"{name} [{round(n*frac)}/step share]",
                               "ops/gemm/kimi3.py Triton GEMV (bs1 path)"))
            continue

        if name.startswith("Cijk_") and round(n) == VENDOR_SHARED_COUNT:
            for blk, frac in VENDOR_SHARED_SPLIT.items():
                blocks[blk] = blocks.get(blk, 0.0) + us * frac
                launches[blk] = launches.get(blk, 0.0) + n * frac
                detail.append((blk, us * frac, n * frac,
                               f"{name[:44]}... [{round(n*frac)}/step share]",
                               "vendor GEMM, attention output projection"))
            continue

        if name in ATTRIB:
            blk, note = ATTRIB[name]
        elif name.startswith("Cijk_"):
            blk = VENDOR_BY_COUNT.get(round(n), "other")
            note = "hipBLASLt/Tensile vendor GEMM (gfx950)"
        else:
            blk = "other"
            note = ""
            if us > 50:
                unattributed.append((name, us))

        blocks[blk] = blocks.get(blk, 0.0) + us
        launches[blk] = launches.get(blk, 0.0) + n
        detail.append((blk, us, n, name, note))

    compute = {k: v for k, v in blocks.items() if k != "collective"}
    ts_total = sum(compute.values())
    sg_total = sum(SGLANG.values())

    # ---- headline block comparison
    with (out / "block-comparison.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(("block", "tokenspeed_us_per_step", "sglang_us_per_step",
                    "delta_us", "tokenspeed_vs_sglang", "tokenspeed_pct",
                    "sglang_pct"))
        for blk in ("moe", "kda", "full_attn", "attn_residual", "other"):
            ts = compute.get(blk, 0.0)
            sg = SGLANG.get(blk, 0.0)
            ratio = f"{sg/ts:.2f}x faster" if ts and sg > ts else (
                f"{ts/sg:.2f}x slower" if ts and sg else "-")
            w.writerow((blk, f"{ts:.1f}", f"{sg:.1f}", f"{ts-sg:+.1f}", ratio,
                        f"{100*ts/ts_total:.1f}", f"{100*sg/sg_total:.1f}"))
        w.writerow(("TOTAL compute", f"{ts_total:.1f}", f"{sg_total:.1f}",
                    f"{ts_total-sg_total:+.1f}",
                    f"{sg_total/ts_total:.2f}x", "100.0", "100.0"))
        w.writerow(("collective (eager busy-wait, excluded)",
                    f"{blocks.get('collective', 0):.1f}", "excluded", "", "", "", ""))
        w.writerow(("end-to-end step, graph ON",
                    f"{TOKENSPEED_STEP_GRAPH_ON_MS*1000:.0f}",
                    f"{SGLANG_STEP_GRAPH_ON_MS*1000:.0f}",
                    f"{(TOKENSPEED_STEP_GRAPH_ON_MS-SGLANG_STEP_GRAPH_ON_MS)*1000:+.0f}",
                    f"{TOKENSPEED_STEP_GRAPH_ON_MS/SGLANG_STEP_GRAPH_ON_MS:.2f}x slower",
                    "", ""))

    # ---- launches per step
    with (out / "launches-per-step.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(("block", "tokenspeed_dispatches_per_step"))
        for blk, n in sorted(launches.items(), key=lambda kv: -kv[1]):
            w.writerow((blk, f"{n:.0f}"))
        w.writerow(("TOTAL", f"{sum(launches.values()):.0f}"))

    # ---- per-kernel detail
    detail.sort(key=lambda d: -d[1])
    with (out / "tokenspeed-kernels-attributed.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(("block", "us_per_step", "dispatches_per_step", "kernel", "source"))
        for blk, us, n, name, note in detail:
            if us >= 1.0:
                w.writerow((blk, f"{us:.1f}", f"{n:.0f}", name, note))

    # ---- print
    print(f"=== Kimi-K3 decode, bs=1, ISL 4096, TP8, graph-off, us/step ===\n")
    print(f"{'block':<16}{'TokenSpeed':>12}{'SGLang':>12}{'delta':>12}   verdict")
    for blk in ("moe", "kda", "full_attn", "attn_residual", "other"):
        ts, sg = compute.get(blk, 0.0), SGLANG.get(blk, 0.0)
        v = f"TokenSpeed {sg/ts:.2f}x faster" if ts and sg > ts else (
            f"TokenSpeed {ts/sg:.2f}x slower" if ts and sg else "")
        print(f"{blk:<16}{ts:>12.1f}{sg:>12.1f}{ts-sg:>+12.1f}   {v}")
    print(f"{'TOTAL compute':<16}{ts_total:>12.1f}{sg_total:>12.1f}"
          f"{ts_total-sg_total:>+12.1f}   TokenSpeed {sg_total/ts_total:.2f}x faster")
    print(f"\ncollective (excluded, eager busy-wait): "
          f"{blocks.get('collective', 0):.0f} us/step")
    print(f"end-to-end graph-ON step: TokenSpeed {TOKENSPEED_STEP_GRAPH_ON_MS} ms "
          f"vs SGLang {SGLANG_STEP_GRAPH_ON_MS} ms "
          f"-> TokenSpeed {TOKENSPEED_STEP_GRAPH_ON_MS/SGLANG_STEP_GRAPH_ON_MS:.2f}x slower")
    print(f"\nTokenSpeed kernel launches per decode step: "
          f"{sum(launches.values()):.0f}")
    if unattributed:
        print("\nunattributed kernels >50 us/step (folded into 'other'):")
        for n, us in sorted(unattributed, key=lambda x: -x[1])[:10]:
            print(f"  {us:8.1f}  {n[:80]}")


if __name__ == "__main__":
    main()
