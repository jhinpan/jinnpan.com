#!/usr/bin/env python3
"""Derived quantities for the write-up: per-layer costs and achieved bandwidths.

Turns the measured per-step device times into the two numbers that explain the
whole result -- bytes touched per decode step by each attention mechanism, and the
fraction of MI355X HBM bandwidth that touching them achieves.
"""
from __future__ import annotations

import argparse
import json

# Kimi-K3 text config
N_KDA = 69
N_FULL = 24
N_MOE = 92
N_LAYERS = 93
HEADS = 96
HEAD_DIM = 128
KV_LORA_RANK = 512
QK_ROPE_DIM = 64
TP = 8
SSM_STATE_BYTES = 4  # fp32 (mamba_ssm_dtype default)
KV_BYTES = 2  # bf16 latent cache

# MI355X: 288 GB HBM3E, 8 TB/s peak
HBM_TBPS = 8.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="/sgl-workspace/workspace/kda_prof/results/summary.json")
    ap.add_argument("--itl", default="/sgl-workspace/workspace/kda_prof/results/graphon_itl.json")
    ap.add_argument("--out")
    args = ap.parse_args()

    with open(args.summary) as fh:
        rows = json.load(fh)["rows"]
    itl_by_label = {}
    try:
        with open(args.itl) as fh:
            for c in json.load(fh)["contexts"]:
                itl_by_label[c["label"]] = c
    except FileNotFoundError:
        pass

    # Per-token, per-rank footprint of each mechanism's decode-time state.
    # MLA keeps a compressed latent (kv_lora_rank + rope dims) that every head
    # reads, so it is replicated on each TP rank rather than sharded.
    mla_bytes_per_token = N_FULL * (KV_LORA_RANK + QK_ROPE_DIM) * KV_BYTES
    # KDA keeps a [head_dim x head_dim] recurrent state per head, sharded by TP,
    # and it does not grow with sequence length.
    kda_state_bytes = N_KDA * (HEADS // TP) * HEAD_DIM * HEAD_DIM * SSM_STATE_BYTES

    print(f"MLA latent KV per token per rank : {mla_bytes_per_token/1024:.1f} KiB "
          f"({N_FULL} layers x {KV_LORA_RANK + QK_ROPE_DIM} dims x {KV_BYTES}B)")
    print(f"KDA recurrent state per rank     : {kda_state_bytes/1e6:.1f} MB "
          f"({N_KDA} layers x {HEADS//TP} heads x {HEAD_DIM}x{HEAD_DIM} x fp32), "
          f"context-independent")
    print()

    out = []
    hdr = (f"{'ctx':>5} {'tokens':>9} {'KV read':>10} {'kv-scan':>9} {'achieved':>10} "
           f"{'% peak':>7} | {'KDA/layer':>10} {'FA/layer':>10} {'FA:KDA':>7} | "
           f"{'real ITL':>9} {'real/comp':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        isl = r["isl"]
        fa = r["detail_us_per_step"]["full_attn"]
        kv_scan_us = fa.get("kv_scan_stage1", 0.0)
        kv_bytes = isl * mla_bytes_per_token
        achieved = kv_bytes / (kv_scan_us * 1e-6) / 1e12 if kv_scan_us else 0.0

        kda_per_layer = r["groups_us_per_step"]["kda"] / N_KDA
        fa_per_layer = r["groups_us_per_step"]["full_attn"] / N_FULL

        itl = itl_by_label.get(r["label"], {}).get("itl_median_ms")
        ratio = (itl / (r["compute_us_per_step"] / 1000)) if itl else None

        rec = {
            "label": r["label"],
            "isl": isl,
            "kv_read_gb_per_step": kv_bytes / 1e9,
            "kv_scan_us_per_step": kv_scan_us,
            "kv_scan_achieved_tbps": achieved,
            "kv_scan_pct_peak": 100.0 * achieved / HBM_TBPS,
            "kda_us_per_layer": kda_per_layer,
            "full_attn_us_per_layer": fa_per_layer,
            "full_attn_over_kda_per_layer": fa_per_layer / kda_per_layer,
            "real_itl_ms": itl,
            "real_itl_over_compute": ratio,
            "compute_ms_per_step": r["compute_us_per_step"] / 1000,
            "groups_ms_per_step": {k: v / 1000 for k, v in r["groups_us_per_step"].items()},
            "groups_pct": r["groups_pct_of_compute"],
            "detail_us_per_step": r["detail_us_per_step"],
        }
        out.append(rec)
        print(f"{r['label']:>5} {isl:9d} {kv_bytes/1e9:8.2f}GB {kv_scan_us/1000:7.2f}ms "
              f"{achieved:8.2f}TB/s {100*achieved/HBM_TBPS:6.1f}% | "
              f"{kda_per_layer:9.1f}us {fa_per_layer:9.1f}us {fa_per_layer/kda_per_layer:6.1f}x | "
              f"{(f'{itl:7.2f}ms' if itl else '      --'):>9} "
              f"{(f'{ratio:8.3f}' if ratio else '      --'):>9}")

    kda_state_traffic_us = next(
        r["detail_us_per_step"]["kda"]["recurrent_state_update"] for r in rows
    )
    print()
    print(f"KDA recurrence kernel: {kda_state_traffic_us:.0f} us/step for {N_KDA} layers "
          f"= {kda_state_traffic_us/N_KDA:.2f} us/layer")
    print(f"  state traffic {2*kda_state_bytes/1e6:.1f} MB read+write per step -> "
          f"{2*kda_state_bytes/(kda_state_traffic_us*1e-6)/1e12:.2f} TB/s "
          f"({100*2*kda_state_bytes/(kda_state_traffic_us*1e-6)/1e12/HBM_TBPS:.1f}% of peak)")

    payload = {
        "hbm_peak_tbps": HBM_TBPS,
        "mla_latent_bytes_per_token_per_rank": mla_bytes_per_token,
        "kda_state_bytes_per_rank": kda_state_bytes,
        "rows": out,
    }
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=1)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
