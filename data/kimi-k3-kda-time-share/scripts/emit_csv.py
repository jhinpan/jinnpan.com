#!/usr/bin/env python3
"""Emit the citable CSVs for the data archive from the measurement JSONs."""
from __future__ import annotations

import csv
import json
import os
import sys

SRC = "/sgl-workspace/workspace/kda_prof/results"
N_KDA, N_FULL, N_MOE = 69, 24, 92


def w(path, header, rows):
    with open(path, "w", newline="") as fh:
        c = csv.writer(fh)
        c.writerow(header)
        c.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def main(out_dir: str) -> int:
    os.makedirs(out_dir, exist_ok=True)

    derived = json.load(open(f"{SRC}/derived.json"))
    rows = derived["rows"]

    w(
        f"{out_dir}/decode-composition.csv",
        ["context", "input_tokens", "decode_steps_profiled",
         "kda_ms_per_step", "full_attn_ms_per_step", "moe_ms_per_step",
         "attn_residual_ms_per_step", "other_ms_per_step",
         "device_compute_ms_per_step", "measured_ms_per_step_cudagraph",
         "kda_pct", "full_attn_pct", "moe_pct", "attn_residual_pct", "other_pct"],
        [
            [r["label"], r["isl"], 24,
             f"{r['groups_ms_per_step']['kda']:.4f}",
             f"{r['groups_ms_per_step']['full_attn']:.4f}",
             f"{r['groups_ms_per_step']['moe']:.4f}",
             f"{r['groups_ms_per_step']['attn_residual']:.4f}",
             f"{r['groups_ms_per_step']['other']:.4f}",
             f"{r['compute_ms_per_step']:.4f}",
             f"{r['real_itl_ms']:.3f}" if r["real_itl_ms"] else "",
             f"{r['groups_pct']['kda']:.2f}",
             f"{r['groups_pct']['full_attn']:.2f}",
             f"{r['groups_pct']['moe']:.2f}",
             f"{r['groups_pct']['attn_residual']:.2f}",
             f"{r['groups_pct']['other']:.2f}"]
            for r in rows
        ],
    )

    w(
        f"{out_dir}/decode-mechanism.csv",
        ["context", "input_tokens", "kda_us_per_layer", "full_attn_us_per_layer",
         "full_attn_over_kda", "kda_recurrence_us_per_step",
         "mla_kv_scan_us_per_step", "mla_kv_read_gb_per_step",
         "mla_kv_scan_achieved_tbps", "mla_kv_scan_pct_of_8tbps_peak"],
        [
            [r["label"], r["isl"],
             f"{r['kda_us_per_layer']:.2f}",
             f"{r['full_attn_us_per_layer']:.2f}",
             f"{r['full_attn_over_kda_per_layer']:.2f}",
             f"{r['detail_us_per_step']['kda']['recurrent_state_update']:.1f}",
             f"{r['detail_us_per_step']['full_attn'].get('kv_scan_stage1', 0):.1f}",
             f"{r['kv_read_gb_per_step']:.3f}",
             f"{r['kv_scan_achieved_tbps']:.3f}",
             f"{r['kv_scan_pct_peak']:.1f}"]
            for r in rows
        ],
    )

    # Inside-the-block breakdowns, decode, one row per (context, block, component)
    detail_rows = []
    for r in rows:
        for block, parts in r["detail_us_per_step"].items():
            total = sum(parts.values())
            for comp, us in sorted(parts.items(), key=lambda kv: -kv[1]):
                detail_rows.append([r["label"], r["isl"], block, comp,
                                    f"{us:.1f}", f"{100*us/total:.2f}"])
    w(f"{out_dir}/decode-block-internals.csv",
      ["context", "input_tokens", "block", "component", "us_per_step", "pct_of_block"],
      detail_rows)

    pf = json.load(open(f"{SRC}/prefill_summary.json"))["rows"]
    ttft = {r["label"]: r for r in json.load(open(f"{SRC}/prefill_ttft.json"))["rows"]}
    w(
        f"{out_dir}/prefill-composition.csv",
        ["context", "input_tokens", "chunks", "ttft_ms_measured", "tokens_per_s",
         "gpu_kernel_ms_total", "gpu_busy_over_ttft", "collective_ms",
         "kda_ms", "full_attn_ms", "moe_ms", "other_ms",
         "kda_pct", "full_attn_pct", "moe_pct", "other_pct",
         "kda_us_per_token", "full_attn_us_per_token", "moe_us_per_token",
         "kda_ms_per_layer", "full_attn_ms_per_layer", "full_attn_over_kda"],
        [
            [r["label"], r["isl"], int(r["chunks"]),
             f"{ttft[r['label']]['ttft_s_min']*1000:.0f}",
             f"{ttft[r['label']]['tok_per_s']:.0f}",
             f"{r['total_gpu_ms']:.1f}",
             f"{r['total_gpu_ms']/(ttft[r['label']]['ttft_s_min']*1000):.3f}",
             f"{r['collective_ms']:.1f}",
             f"{r['groups_ms']['kda']:.1f}",
             f"{r['groups_ms']['full_attn']:.1f}",
             f"{r['groups_ms']['moe']:.1f}",
             f"{r['groups_ms'].get('other',0)+r['groups_ms'].get('dense_mlp',0):.1f}",
             f"{r['groups_pct_of_compute']['kda']:.2f}",
             f"{r['groups_pct_of_compute']['full_attn']:.2f}",
             f"{r['groups_pct_of_compute']['moe']:.2f}",
             f"{r['groups_pct_of_compute'].get('other',0)+r['groups_pct_of_compute'].get('dense_mlp',0):.2f}",
             f"{r['us_per_token']['kda']:.2f}",
             f"{r['us_per_token']['full_attn']:.2f}",
             f"{r['us_per_token']['moe']:.2f}",
             f"{r['groups_ms']['kda']/N_KDA:.4f}",
             f"{r['groups_ms']['full_attn']/N_FULL:.4f}",
             f"{(r['groups_ms']['full_attn']/N_FULL)/(r['groups_ms']['kda']/N_KDA):.3f}"]
            for r in pf
        ],
    )

    w(
        f"{out_dir}/prefill-mechanism-kernels.csv",
        ["context", "input_tokens", "mla_prefill_attn_kernel_ms",
         "kda_chunked_kernels_ms", "mla_kernel_pct_of_all_prefill_gpu_time"],
        [
            [r["label"], r["isl"],
             f"{r['mla_prefill_kernel_ms']:.1f}",
             f"{r['kda_chunk_kernels_ms']:.1f}",
             f"{100*r['mla_prefill_kernel_ms']/r['total_gpu_ms']:.2f}"]
            for r in pf
        ],
    )

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
