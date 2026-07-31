#!/usr/bin/env python3
"""Re-derive the prefill composition from the trace using Perfetto's own importer.

The point is independence. `bucketize.py` parses the chrome JSON itself, walks
correlation ids to host-side `record_function` ranges, and sums `dur`. If that walk
is wrong, re-running it cannot reveal the error. Here trace_processor does the
parsing and SQL does the arithmetic, and three separate quantities are computed for
every block:

  sum-of-durations   what the write-up reports
  union-of-intervals what the GPU wall clock actually spends there (overlap-free)
  gpu_user_annotation  what PyTorch projects onto the GPU track, which is what a
                       human scrolling Perfetto actually looks at

  ./tp_verify.py pf32k-TP0.json
"""
from __future__ import annotations

import argparse
import sys

from perfetto.trace_processor import TraceProcessor

BLOCKS = ("K3/kda", "K3/full_attn", "K3/moe", "K3/dense_mlp")


def rows(tp, sql):
    return list(tp.query(sql))


def table(title, rs, cols, widths=None, limit=None):
    print(f"\n=== {title} ===")
    if not rs:
        print("  (no rows)")
        return
    widths = widths or [28] * len(cols)
    print("  " + "  ".join(c.ljust(w)[:w] for c, w in zip(cols, widths)))
    print("  " + "  ".join("-" * w for w in widths))
    for r in rs[: limit or len(rs)]:
        cells = []
        for c, w in zip(cols, widths):
            v = getattr(r, c)
            s = f"{v:,.2f}" if isinstance(v, float) else str(v)
            cells.append(s.ljust(w)[:w] if not isinstance(v, (int, float)) else s.rjust(w))
        print("  " + "  ".join(cells))


def union_ms(intervals: list[tuple[int, int]]) -> float:
    """Wall-clock covered by a set of [ts, ts+dur) intervals, in ms."""
    if not intervals:
        return 0.0
    intervals.sort()
    total = 0
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s > cur_e:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    total += cur_e - cur_s
    return total / 1e6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    tp = TraceProcessor(trace=args.trace)

    # ---- 1. the GPU device timeline, with no attribution involved at all -------
    r = rows(tp, """
        select count(*) as n, sum(dur)/1e6 as sum_ms,
               (max(ts+dur) - min(ts))/1e6 as span_ms
        from slice where category in ('kernel','gpu_memcpy','gpu_memset')
    """)[0]
    gpu_intervals = [(x.ts, x.ts + x.dur) for x in rows(tp, """
        select ts, dur from slice
        where category in ('kernel','gpu_memcpy','gpu_memset')
    """)]
    busy = union_ms(gpu_intervals)
    print("=== GPU device timeline (category = kernel/memcpy/memset) ===")
    print(f"  dispatches            {r.n:,}")
    print(f"  sum of durations      {r.sum_ms:,.2f} ms")
    print(f"  union of intervals    {busy:,.2f} ms   "
          f"(overlap inflates the sum by {r.sum_ms - busy:,.2f} ms = "
          f"{100*(r.sum_ms-busy)/r.sum_ms:.2f}%)")
    print(f"  first->last kernel    {r.span_ms:,.2f} ms")
    print(f"  GPU busy / span       {busy/r.span_ms:.3f}")

    # ---- 2. kernel time by name, the claim that needs no attribution ----------
    table("GPU time by kernel name (top)", rows(tp, f"""
        select
          case when length(name) > 58 then substr(name,1,55)||'...' else name end as kernel,
          count(*) as n, sum(dur)/1e6 as ms,
          100.0*sum(dur)/(select sum(dur) from slice
                          where category in ('kernel','gpu_memcpy','gpu_memset')) as pct
        from slice where category in ('kernel','gpu_memcpy','gpu_memset')
        group by name order by ms desc limit {args.top}
    """), ["kernel", "n", "ms", "pct"], [60, 6, 12, 8])

    # ---- 3. _fwd_kernel, dispatch by dispatch ---------------------------------
    fwd = rows(tp, """
        select ts, dur/1e6 as ms from slice
        where name = '_fwd_kernel' and category = 'kernel' order by ts
    """)
    if fwd:
        ms = [x.ms for x in fwd]
        print(f"\n=== _fwd_kernel (the MLA prefill attention kernel) ===")
        print(f"  dispatches   {len(ms)}")
        print(f"  total        {sum(ms):,.2f} ms")
        print(f"  min / median / max   {min(ms):.3f} / "
              f"{sorted(ms)[len(ms)//2]:.3f} / {max(ms):.3f} ms")
        print("  per-dispatch ms, in trace order:")
        for i in range(0, len(ms), 12):
            print("    " + " ".join(f"{v:7.2f}" for v in ms[i:i+12]))

    # ---- 4. what a human sees on the GPU track: gpu_user_annotation -----------
    ann = rows(tp, """
        select name, count(*) as n, sum(dur)/1e6 as sum_ms
        from slice where category = 'gpu_user_annotation'
        group by name order by sum_ms desc
    """)
    table("gpu_user_annotation on the GPU track (what Perfetto draws as K3/* bands)",
          ann, ["name", "n", "sum_ms"], [40, 8, 14])
    for b in BLOCKS:
        iv = [(x.ts, x.ts + x.dur) for x in rows(tp, f"""
            select ts, dur from slice
            where category = 'gpu_user_annotation' and name = '{b}'
        """)]
        if iv:
            print(f"  {b:<16} union of its GPU bands = {union_ms(iv):9,.2f} ms")

    # ---- 5. host-side ranges, which are NOT the GPU cost ----------------------
    table("user_annotation on the CPU thread (host-side record_function ranges)",
          rows(tp, """
        select name, count(*) as n, sum(dur)/1e6 as sum_ms, avg(dur)/1e3 as avg_us
        from slice where category = 'user_annotation' and name like 'K3/%'
        group by name order by sum_ms desc
    """), ["name", "n", "sum_ms", "avg_us"], [24, 8, 14, 14])

    tp.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
