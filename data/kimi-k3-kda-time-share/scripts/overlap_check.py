#!/usr/bin/env python3
"""Do KDA and full-attention kernels ever execute concurrently?

Two things could be meant by "overlap": sharing a kernel (same code running for
both blocks) and overlapping in time (both resident on the GPU at once). This
answers the second empirically -- take the kernels that only a KDA layer launches
and the kernels that only a full-attention layer launches, and look for any pair
whose device intervals intersect.

  ./overlap_check.py traces/ctx64k/ctx64k-TP-0.trace.json.gz
"""
from __future__ import annotations

import collections
import gzip
import json
import sys

KDA_ONLY_DECODE = {
    "fused_recurrent_kda_packed_decode_kernel",
    "_causal_conv1d_update_kernel",
    "layer_norm_gated_fwd_kernel",
    "hgemm_bf16_16x64x128x6_SPK2_W1x2x1_BLDS1_TN_AS1_0",
}
MLA_ONLY_DECODE = {
    "_fwd_grouped_kernel_stage1",
    "_fwd_kernel_stage2",
    "hgemm_bf16_16x64x64x7_SPK7_W1x2x1_BLDS1_TN_AS1_0",
    "hgemm_bf16_16x64x64x5_SPK4_W1x2x1_BLDS1_TN_AS1_0",
}
KDA_ONLY_PREFILL = {
    "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
    "chunk_kda_fwd_kernel_intra_token_parallel",
    "chunk_gla_fwd_kernel_o",
    "chunk_kda_fwd_kernel_inter_solve_fused",
    "_causal_conv1d_fwd_kernel",
}
MLA_ONLY_PREFILL = {"_fwd_kernel"}


def load(path):
    with gzip.open(path, "rt") as fh:
        return json.load(fh)


def main(path: str) -> int:
    trace = load(path)
    kernels = [e for e in trace["traceEvents"] if e.get("cat") == "kernel"]
    names = {e["name"] for e in kernels}
    prefill = "_fwd_kernel" in names and "_fwd_grouped_kernel_stage1" not in names
    kda_set = KDA_ONLY_PREFILL if prefill else KDA_ONLY_DECODE
    mla_set = MLA_ONLY_PREFILL if prefill else MLA_ONLY_DECODE

    streams = collections.Counter(
        (e.get("args") or {}).get("stream") for e in kernels
    )
    print(f"{path}")
    print(f"  phase inferred      : {'prefill' if prefill else 'decode'}")
    print(f"  kernel events       : {len(kernels)}")
    print(f"  distinct streams    : {dict(streams)}")

    def spans(nameset):
        out = []
        for e in kernels:
            if e["name"] in nameset:
                out.append((float(e["ts"]), float(e["ts"]) + float(e.get("dur", 0)),
                            e["name"], (e.get("args") or {}).get("stream")))
        out.sort()
        return out

    kda, mla = spans(kda_set), spans(mla_set)
    print(f"  KDA-only dispatches : {len(kda)}")
    print(f"  MLA-only dispatches : {len(mla)}")

    # Sweep both sorted lists once looking for intersecting intervals.
    overlaps, i = [], 0
    for a0, a1, an, ast in kda:
        while i < len(mla) and mla[i][1] <= a0:
            i += 1
        j = i
        while j < len(mla) and mla[j][0] < a1:
            overlaps.append((an, mla[j][2], min(a1, mla[j][1]) - max(a0, mla[j][0])))
            j += 1
    print(f"  overlapping pairs   : {len(overlaps)}")
    if overlaps:
        for o in overlaps[:5]:
            print(f"      {o[2]:.2f} us  {o[0][:40]} || {o[1][:40]}")

    # Gap between the last KDA kernel of a layer group and the next MLA kernel,
    # to show they are strictly sequential rather than merely non-overlapping.
    merged = sorted([(s, e, "KDA") for s, e, _, _ in kda]
                    + [(s, e, "MLA") for s, e, _, _ in mla])
    trans = [merged[k + 1][0] - merged[k][1]
             for k in range(len(merged) - 1) if merged[k][2] != merged[k + 1][2]]
    if trans:
        trans.sort()
        print(f"  KDA→MLA / MLA→KDA transitions: {len(trans)}, "
              f"min gap {trans[0]:.2f} us, median {trans[len(trans)//2]:.2f} us")
    return 0


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
        print()
