#!/usr/bin/env python3
"""HIP graph per-kernel fixed cost, host submission cost and branch concurrency.

One process measures one runtime configuration because CLR reads its flags at
initialization. Device time uses a two-replay backlog followed by eight timed
replays per event pair, so small graphs are not timed against an idle queue.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import time

import torch
import triton
import triton.language as tl


@triton.jit
def _bump(x_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(x_ptr + offs, tl.load(x_ptr + offs, mask=mask) + 1, mask=mask)


def bump(buf, programs):
    _bump[(programs,)](buf, programs * 64, BLOCK=64)


@triton.jit
def _copy_add(src, dst, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(dst + offs, tl.load(src + offs) + 1)


@triton.jit
def _stream_sum(w_ptr, out_ptr, per_prog, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    acc = tl.zeros([BLOCK], tl.float32)
    base = w_ptr + pid * per_prog
    for i in range(0, per_prog, BLOCK):
        acc += tl.load(base + i + tl.arange(0, BLOCK)).to(tl.float32)
    tl.store(out_ptr + pid * BLOCK + tl.arange(0, BLOCK), acc)


def capture(body):
    body()  # compile and warm outside capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()
    torch.cuda.synchronize()
    return graph


def time_graph(graph, rounds, timed=8, backlog=2):
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    device_us, host_us = [], []
    for _ in range(rounds):
        for _ in range(backlog):
            graph.replay()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        t0 = time.perf_counter()
        for _ in range(timed):
            graph.replay()
        t1 = time.perf_counter()
        end.record()
        torch.cuda.synchronize()
        device_us.append(start.elapsed_time(end) * 1000 / timed)
        host_us.append((t1 - t0) * 1e6 / timed)
    return dict(device_us=device_us, host_us=host_us,
                device_median_us=statistics.median(device_us),
                host_median_us=statistics.median(host_us))


def fit(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    slope = sum((x - mx) * (y - my) for x, y in points) / sum((x - mx) ** 2 for x in xs)
    return dict(per_kernel_us=slope, intercept_us=my - slope * mx)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--rounds", type=int, default=31)
    parser.add_argument("--quick", action="store_true", help="profiler run: one size per case")
    args = parser.parse_args()
    torch.cuda.init()
    dev = torch.device("cuda")
    flags = {k: v for k, v in os.environ.items() if k.startswith(("DEBUG_", "AMD_", "ROC_", "GPU_", "HIP_", "HSA_"))}
    result = dict(label=args.label, flags=flags, device=torch.cuda.get_device_name(0),
                  hip=torch.version.hip, torch=torch.__version__, triton=triton.__version__, cases={})
    sizes = [128] if args.quick else [1, 8, 32, 128, 512, 1024]

    # 1. Dependent single-stream chains of tiny Triton kernels (1 and 256 programs).
    for programs in (1, 256):
        buf = torch.zeros(programs * 64, device=dev, dtype=torch.float32)
        rows = {}
        for n in sizes:
            graph = capture(lambda: [bump(buf, programs) for _ in range(n)])
            rows[n] = time_graph(graph, args.rounds)
            del graph
        result["cases"][f"triton_chain_p{programs}"] = dict(rows=rows,
            fit=fit([(n, r["device_median_us"]) for n, r in rows.items() if n >= 32]) if not args.quick else None)

    # 2. PyTorch elementwise chain (the verify head/tail glue kernels are of this kind).
    small = torch.zeros(64, device=dev, dtype=torch.int32)
    rows = {}
    for n in sizes:
        graph = capture(lambda: [small.add_(1) for _ in range(n)])
        rows[n] = time_graph(graph, args.rounds)
        del graph
    result["cases"]["torch_add_chain"] = dict(rows=rows,
        fit=fit([(n, r["device_median_us"]) for n, r in rows.items() if n >= 32]) if not args.quick else None)

    # 3. Two independent branches (fork/join) versus one chain with the same kernel count.
    side = [torch.cuda.Stream() for _ in range(2)]
    bufs = [torch.zeros(64, device=dev, dtype=torch.float32) for _ in range(2)]
    rows = {}
    for n in ([128] if args.quick else [32, 128, 512]):
        def branches():
            main = torch.cuda.current_stream()
            for s, b in zip(side, bufs):
                s.wait_stream(main)
                with torch.cuda.stream(s):
                    for _ in range(n // 2):
                        bump(b, 1)
            for s in side:
                main.wait_stream(s)
        graph = capture(branches)
        rows[n] = time_graph(graph, args.rounds)
        del graph
    result["cases"]["two_branch_p1"] = dict(rows=rows)

    # 3b. Fork/join ladder: per rung one kernel on the capture stream, then two
    # independent sibling kernels on side streams, joined by the next rung.
    rows = {}
    for rungs in ([32] if args.quick else [32, 128]):
        def ladder():
            main = torch.cuda.current_stream()
            for _ in range(rungs):
                bump(bufs[0], 1)
                for s, b in zip(side, (bufs[0], bufs[1])):
                    s.wait_stream(main)
                with torch.cuda.stream(side[0]):
                    bump(ladder_a, 1)
                with torch.cuda.stream(side[1]):
                    bump(ladder_b, 1)
                for s in side:
                    main.wait_stream(s)
        ladder_a = torch.zeros(64, device=dev, dtype=torch.float32)
        ladder_b = torch.zeros(64, device=dev, dtype=torch.float32)
        graph = capture(ladder)
        rows[rungs] = time_graph(graph, args.rounds)
        rows[rungs]["kernels"] = 3 * rungs
        del graph
    result["cases"]["fork_join_ladder_p1"] = dict(rows=rows)

    # 3c. Producer->consumer chain through 16 KiB buffers (activation-sized glue work).
    ring = [torch.zeros(16 * 256, device=dev, dtype=torch.float32) for _ in range(2)]
    rows = {}
    for n in ([128] if args.quick else [32, 128, 512]):
        graph = capture(lambda: [_copy_add[(16,)](ring[i % 2], ring[(i + 1) % 2], BLOCK=256) for i in range(n)])
        rows[n] = time_graph(graph, args.rounds)
        del graph
    result["cases"]["producer_consumer_16k"] = dict(rows=rows,
        fit=fit([(n, r["device_median_us"]) for n, r in rows.items()]) if not args.quick else None)

    # 3d. Each kernel streams a distinct cold 1 MiB slice of a 1 GiB bf16 region (64 programs).
    wbig = torch.ones(512 * 1024 * 1024, device=dev, dtype=torch.bfloat16)
    wout = torch.zeros(64 * 1024, device=dev, dtype=torch.float32)
    slice_elems = 512 * 1024
    rows = {}
    for n in ([128] if args.quick else [32, 128, 512]):
        def stream_chain():
            for i in range(n):
                off = (i * 7919 % 1024) * slice_elems
                _stream_sum[(64,)](wbig[off:off + slice_elems], wout, slice_elems // 64, BLOCK=1024)
        graph = capture(stream_chain)
        rows[n] = time_graph(graph, args.rounds)
        del graph
    result["cases"]["cold_stream_1mib"] = dict(rows=rows, bytes_per_kernel=slice_elems * 2,
        fit=fit([(n, r["device_median_us"]) for n, r in rows.items()]) if not args.quick else None)
    del wbig

    # 4. A small BF16 GEMV-shaped matmul chain, roughly the shared-expert gate/up shape.
    x = torch.randn(6, 5120, device=dev, dtype=torch.bfloat16)
    w = torch.randn(5120, 1152, device=dev, dtype=torch.bfloat16)
    out = torch.empty(6, 1152, device=dev, dtype=torch.bfloat16)
    rows = {}
    for n in ([40] if args.quick else [1, 40]):
        graph = capture(lambda: [torch.mm(x, w, out=out) for _ in range(n)])
        rows[n] = time_graph(graph, args.rounds)
        del graph
    result["cases"]["gemv_6x5120x1152_bf16"] = dict(rows=rows, weight_bytes=w.numel() * 2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    summary = {k: (v.get("fit") or {k2: round(r["device_median_us"], 3) for k2, r in v["rows"].items()})
               for k, v in result["cases"].items()}
    print(json.dumps(dict(label=args.label, summary=summary)), flush=True)


if __name__ == "__main__":
    main()
