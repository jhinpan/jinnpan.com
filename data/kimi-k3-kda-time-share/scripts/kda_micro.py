#!/usr/bin/env python3
"""Standalone KDA decode-step reproduction at the server's exact per-GPU shapes.

Purpose is twofold: an independent latency check on the in-situ number, and a
small, single-kernel process that `rocprofv3 --att` can thread-trace (the full
server could not be traced -- per-dispatch interception across 8 TP ranks drove
the scheduler past its watchdog).

The shapes are context-independent by construction, which is the whole point of a
linear-attention layer: the kernel reads a fixed [V, K] state per head, applies a
per-K decay and one delta-rule update, and writes the state back. Nothing in the
launch depends on how many tokens preceded it, so a trace taken here is the same
trace the 64K decode executes.

  ./kda_micro.py --iters 200
  rocprofv3 --att --att-target-cu 1 --kernel-include-regex "kda_packed_decode" \
      -d att_out -o kda -- ./kda_micro.py --iters 3 --no-timing
"""
from __future__ import annotations

import argparse
import json
import sys

import torch

# Kimi-K3 text config, sharded over TP8:
#   linear_attn_config.num_heads = 96 -> 12 heads per GPU
#   linear_attn_config.head_dim  = 128  (K = V = 128)
#   gate_lower_bound             = -5.0
#   short_conv_kernel_size       = 4
NUM_HEADS_GLOBAL = 96
TP = 8
HEAD_DIM = 128
GATE_LOWER_BOUND = -5.0
CONV_KERNEL = 4
N_KDA_LAYERS = 69
STATE_SLOTS = 8


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()

    from sglang.kernels.ops.attention.fla.fused_recurrent import (
        fused_recurrent_kda_packed_decode,
    )

    dev = torch.device("cuda:0")
    torch.manual_seed(0)

    H = HV = NUM_HEADS_GLOBAL // TP
    K = V = HEAD_DIM
    B = args.batch

    mixed_qkv = torch.randn(B, 2 * H * K + HV * V, device=dev, dtype=torch.bfloat16)
    a = torch.randn(B, HV * K, device=dev, dtype=torch.bfloat16)
    b = torch.randn(B, HV, device=dev, dtype=torch.bfloat16)
    A_log = torch.randn(HV, device=dev, dtype=torch.float32)
    dt_bias = torch.randn(HV * K, device=dev, dtype=torch.float32)
    # The pool the server keeps per sequence: fp32 by default (mamba_ssm_dtype).
    state = torch.randn(STATE_SLOTS, HV, V, K, device=dev, dtype=torch.float32)
    out = torch.empty(B, 1, HV, V, device=dev, dtype=torch.bfloat16)
    idx = torch.zeros(B, device=dev, dtype=torch.int32)

    state_bytes = HV * V * K * state.element_size()
    print(
        f"per-GPU KDA layer: heads={HV} K=V={K}  state={state_bytes/1024:.0f} KiB/layer"
        f"  ({N_KDA_LAYERS} layers -> {N_KDA_LAYERS*state_bytes/1e6:.1f} MB read+write per step)",
        flush=True,
    )

    def one():
        fused_recurrent_kda_packed_decode(
            mixed_qkv=mixed_qkv, a=a, b=b, A_log=A_log, dt_bias=dt_bias,
            scale=K ** -0.5, initial_state=state, out=out, ssm_state_indices=idx,
            use_qk_l2norm_in_kernel=True, lower_bound=GATE_LOWER_BOUND,
        )

    for _ in range(3):
        one()
    torch.cuda.synchronize()

    if args.no_timing:
        for _ in range(args.iters):
            one()
        torch.cuda.synchronize()
        print(f"ran {args.iters} untimed iterations (trace mode)", flush=True)
        return 0

    # Back-to-back launches, so the number is a per-dispatch device time
    # comparable with the per-layer cost extracted from the server trace.
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        one()
    end.record()
    torch.cuda.synchronize()
    per_call_us = start.elapsed_time(end) * 1000.0 / args.iters

    result = {
        "heads_per_gpu": HV,
        "head_dim": K,
        "batch": B,
        "state_bytes_per_layer": state_bytes,
        "iters": args.iters,
        "per_call_us": per_call_us,
        "implied_per_step_us_69_layers": per_call_us * N_KDA_LAYERS,
        "state_traffic_gb_per_s": 2 * state_bytes / (per_call_us * 1e-6) / 1e9,
    }
    print(
        f"fused_recurrent_kda_packed_decode: {per_call_us:.2f} us/call\n"
        f"  -> {per_call_us*N_KDA_LAYERS/1000:.3f} ms/step for {N_KDA_LAYERS} KDA layers\n"
        f"  state traffic {result['state_traffic_gb_per_s']:.0f} GB/s "
        f"(read+write of {state_bytes/1024:.0f} KiB)",
        flush=True,
    )
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
