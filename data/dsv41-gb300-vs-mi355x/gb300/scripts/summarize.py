#!/usr/bin/env python3
"""Summarize BBuf-client arms: pooled median over startups (warmups excluded), per-startup
medians, and per-step time (decode time after the first event / (verify steps - 1), or per
token without speculation)."""
import argparse
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

WORK = Path('$GB300_WORK')


def load(arm):
    ms = json.loads((WORK / 'results' / arm / 'bench' / 'measurements.json').read_text())
    rows = []
    for r in ms:
        if r['warmup']:
            continue
        dec = r['elapsed_s'] - r['ttft_s']
        v = r['verify_steps']
        step_ms = dec / (v - 1) * 1e3 if v else dec / (r['output_tokens'] - r['first_event_tokens']) * 1e3
        rows.append({'tps': r['output_tps'], 'acc': r['accept_length'], 'step_ms': step_ms,
                     'ttft_ms': r['ttft_s'] * 1e3})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('pattern', help='regex over arm names; the config is the name minus its -<startup> suffix')
    a = p.parse_args()
    groups = defaultdict(list)
    for d in sorted((WORK / 'results').iterdir()):
        if re.search(a.pattern, d.name) and (d / 'bench' / 'measurements.json').exists():
            groups[d.name.rsplit('-', 1)[0]].append(d.name)
    for cfg, arms in groups.items():
        pooled = []
        per = []
        for arm in arms:
            rows = load(arm)
            pooled += rows
            per.append(f"{arm.rsplit('-', 1)[1]}={st.median(r['tps'] for r in rows):.1f}")
        acc = [r['acc'] for r in pooled if r['acc']]
        print(f"{cfg:22s} n={len(pooled):2d} median={st.median(r['tps'] for r in pooled):7.1f} "
              f"[{min(r['tps'] for r in pooled):.1f}, {max(r['tps'] for r in pooled):.1f}] "
              f"per-startup: {' '.join(per)} | step_ms median={st.median(r['step_ms'] for r in pooled):.2f} "
              f"acc median={st.median(acc) if acc else float('nan'):.3f} "
              f"ttft_ms median={st.median(r['ttft_ms'] for r in pooled):.0f}")


if __name__ == '__main__':
    main()
