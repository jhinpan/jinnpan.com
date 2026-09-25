#!/usr/bin/env python3
"""Per-kernel time per verify cycle from an nsys SQLite export (TP0), for chosen categories.

Uses attribute.py's categories and cycle boundaries (one launch of the main graph per cycle), and
reports every kernel name's summed duration and launch count per cycle, verify graph only.
"""
import argparse
import collections
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attribute import classify  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sqlite")
    p.add_argument("--out", required=True)
    p.add_argument("--categories", default="moe,attention")
    a = p.parse_args()
    want = set(a.categories.split(","))
    c = sqlite3.connect(a.sqlite)
    names = dict(c.execute("select id, value from StringIds"))
    rows = c.execute("select start, end, graphId, demangledName, shortName, graphNodeId from CUPTI_ACTIVITY_KIND_KERNEL "
                     "where deviceId=0 order by start").fetchall()
    graph_time, first_node = collections.Counter(), {}
    for s, e, g, dn, sn, node in rows:
        if g is not None:
            graph_time[g] += e - s
            first_node.setdefault(g, node)
    main_graph = max(graph_time, key=graph_time.get)
    bounds = sorted(s for s, e, g, dn, sn, node in rows if g == main_graph and node == first_node[main_graph])
    n = len(bounds) - 1
    lo, hi = bounds[0], bounds[-1]
    acc, cnt = collections.Counter(), collections.Counter()
    for s, e, g, dn, sn, node in rows:
        if g != main_graph or not lo <= s < hi:
            continue
        full = names.get(dn) or names.get(sn) or ""
        cat, sub = classify(full)
        if cat in want:
            key = (cat, sub, (names.get(sn) or full)[:80])
            acc[key] += (e - s) / 1e3
            cnt[key] += 1
    out = [dict(category=k[0], sub=k[1], kernel=k[2], us_per_cycle=v / n, calls_per_cycle=cnt[k] / n)
           for k, v in sorted(acc.items(), key=lambda kv: -kv[1])]
    Path(a.out).write_text(json.dumps(dict(sqlite=a.sqlite, cycles=n, kernels=out), indent=2) + "\n")
    for x in out:
        print(f"{x['category']:10s} {x['sub']:14s} {x['us_per_cycle']:8.1f} us {x['calls_per_cycle']:6.1f}/cycle  "
              f"{x['us_per_cycle'] / x['calls_per_cycle']:6.2f} us/call  {x['kernel']}")


if __name__ == "__main__":
    main()
