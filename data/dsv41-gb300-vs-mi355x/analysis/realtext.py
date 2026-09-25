#!/usr/bin/env python3
"""The real-text BS=1 contract on both machines: acceptance length and cost per verify.

MI355X: the two base-arm servers of the e2e824 campaign (s2-bs1/1-base, 6-base; the four
chat-encoded 4,096-token prompts, 24 samples each). GB300: the same prompts and client
replayed on current main at the default stream layout, NUMA-bound (w3-rt-a, w3-rt-b).
wall_per_verify_proxy_ms includes host and scheduler time, so it is compared only with itself.
"""
import hashlib
import json
import statistics as st
from pathlib import Path

CAMP = Path(__file__).resolve().parents[1]
E2E = Path("$E2E_CAMPAIGN/results/s2-bs1")
KEYS = ("accepted_length", "wall_per_verify_proxy_ms", "stream_decode_tps", "request_output_tps")


def load(paths):
    rows = []
    for launch, p in paths:
        for line in p.open():
            r = json.loads(line)
            if not r.get("scored"):
                continue
            d = r.get("derived", r)
            out = r.get("output_sha256") or hashlib.sha256(
                json.dumps(r["output_ids"], separators=(",", ":")).encode()).hexdigest()
            rows.append(dict(launch=launch, index=r["index"], server_ttft_ms=r["server_ttft_ms"],
                             output_sha256=out, **{k: d[k] for k in KEYS}))
    return rows


def summary(rows):
    med = lambda k, rs=rows: st.median(r[k] for r in rs)
    return dict(requests=len(rows), launches=sorted({r["launch"] for r in rows}),
                **{f"{k}_median": med(k) for k in KEYS + ("server_ttft_ms",)},
                accepted_length_range=[min(r["accepted_length"] for r in rows),
                                       max(r["accepted_length"] for r in rows)],
                accepted_length_by_prompt={i: med("accepted_length", [r for r in rows if r["index"] == i])
                                           for i in range(4)},
                accepted_length_by_launch={l: med("accepted_length", [r for r in rows if r["launch"] == l])
                                           for l in sorted({r["launch"] for r in rows})},
                distinct_outputs_by_prompt_and_launch={f"{l}/{i}": len({r["output_sha256"] for r in rows
                                                                        if r["launch"] == l and r["index"] == i})
                                                       for l in sorted({r["launch"] for r in rows}) for i in range(4)})


def main():
    mi = load([("1-base", E2E / "1-base/requests.jsonl"), ("6-base", E2E / "6-base/requests.jsonl")])
    gb = load([(x, CAMP / f"gb300/results/w3-rt-{x}/realtext/requests.jsonl") for x in "ab"])
    out = dict(mi355x=summary(mi), gb300=summary(gb), rows=dict(mi355x=mi, gb300=gb))
    (CAMP / "analysis/realtext.json").write_text(json.dumps(out, indent=2) + "\n")
    for name in ("gb300", "mi355x"):
        s = out[name]
        print(f"{name:7} n={s['requests']} AL={s['accepted_length_median']:.3f} "
              f"range={s['accepted_length_range'][0]:.2f}-{s['accepted_length_range'][1]:.2f} "
              f"verify={s['wall_per_verify_proxy_ms_median']:.3f} ms decode={s['stream_decode_tps_median']:.1f} tok/s "
              f"by prompt={ {k: round(v, 2) for k, v in s['accepted_length_by_prompt'].items()} }")


if __name__ == "__main__":
    main()
