#!/usr/bin/env python3
"""Emit the per-block kernel inventory: which kernels belong to which mechanism.

Answers, in citable form, whether KDA and full attention share any code. Decode
comes from the ground-truth name map; prefill from the range-attributed traces,
where attribution is direct because prefill does not run through a CUDA graph.
"""
from __future__ import annotations

import csv
import json
import sys

SRC = "/sgl-workspace/workspace/kda_prof/results"
DECODE_STEPS = 24
LABEL = {"K3/kda": "KDA", "K3/full_attn": "full_attn", "K3/moe": "MoE",
         "K3/dense_mlp": "dense_MLP", "other": "outside_blocks"}


def main(out: str) -> int:
    rows = []

    for r in json.load(open(f"{SRC}/name_map_4k.json")):
        split = {LABEL.get(b, b): n for b, n in r["split"].items()}
        owners = [b for b, n in split.items() if n]
        rows.append([
            "decode", r["name"], r["cls"],
            "|".join(f"{b}:{n//DECODE_STEPS}" for b, n in split.items()),
            "shared" if len(owners) > 1 else "exclusive",
            owners[0] if len(owners) == 1 else "",
        ])

    for tag, ctx in (("pf8k", "8k"), ("pf32k", "32k")):
        d = json.load(open(f"{SRC}/{tag}.json"))
        seen: dict[str, dict[str, int]] = {}
        for bucket, bb in d["buckets"].items():
            for k in bb["top_kernels"]:
                seen.setdefault(k["name"], {})[LABEL.get(bucket, bucket)] = k["n"]
        for name, split in seen.items():
            owners = list(split)
            rows.append([
                f"prefill_{ctx}", name, "",
                "|".join(f"{b}:{n}" for b, n in split.items()),
                "shared" if len(owners) > 1 else "exclusive",
                owners[0] if len(owners) == 1 else "",
            ])

    with open(out, "w", newline="") as fh:
        c = csv.writer(fh)
        c.writerow(["phase", "kernel_name", "class", "dispatches_per_forward_by_block",
                    "sharing", "exclusive_owner"])
        c.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")

    dec = [r for r in rows if r[0] == "decode"]
    shared = [r for r in dec if r[4] == "shared"]
    print(f"decode: {len(dec)} distinct kernels, {len(shared)} shared across blocks")
    for r in shared:
        print(f"   {r[3]:<58} {r[1][:56]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
