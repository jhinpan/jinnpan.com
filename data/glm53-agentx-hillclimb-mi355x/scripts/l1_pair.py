#!/usr/bin/env python3
"""Request-paired effects in the GLM-5.3 InferenceX L1 cohort (phase infxl1).

Requests match on (conversation_id, turn_index) with input lengths within PREFIX_TOKENS: each
run draws its own FIRST_TURN_PREFIX (cache busting), which shifts a conversation's lengths by
a few tokens (1-3 observed).
For each run X at one concurrency: the median over the requests X shares with both base runs
of log(X) - mean(log base r1, log base r2), per metric. The base runs themselves show half
the A/A difference with opposite signs. Requests within one run share its server, so the run
is the unit of comparison, not the request. Also per run: the exact accept length
(sglang:generation_tokens / sglang:spec_verify_calls) and the server's cached-token share.

    python3 l1_pair.py [--cohort 1|2] [--json OUT]
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "runs"
ID = "agentx-glm53-l1-c{c}-{arm}-r{r}"
# cohort -> (run-id template, arms, candidate arm)
COHORTS = {
    "1": (ID, (("base", 1), ("base", 2), ("cand", 1), ("cand", 2), ("b0", 1), ("b1", 1)), "cand"),
    "2": ("agentx-glm53-l1f-c{c}-{arm}-r{r}", (("base", 1), ("base", 2), ("cand8", 1), ("cand8", 2)), "cand8"),
    "3": ("agentx-glm53-l2ov-c{c}-{arm}-r{r}", (("base", 1), ("base", 2), ("cand8", 1)), "cand8"),
}
CONCS = {"1": (1, 2, 4), "2": (1, 2, 4), "3": (1, 2, 4, 10)}
METRICS = ("time_to_first_token", "inter_token_latency", "request_latency")
ARMS = (("base", 1), ("base", 2), ("cand", 1), ("cand", 2), ("b0", 1), ("b1", 1))
PREFIX_TOKENS = 8
CAND = "cand"


def load(rid: str) -> dict | None:
    art = RUNS / rid / "agentx/aiperf_artifacts"
    if not (art / "profile_export.jsonl").exists() or not (RUNS / rid / "state.json").exists():
        return None
    if json.loads((RUNS / rid / "state.json").read_text())["status"] != "done":
        return None
    m = json.loads((art / "server_metrics_export.json").read_text())["metrics"]
    total = lambda k: sum(s["stats"].get("total", 0) for s in m[k]["series"])  # noqa: E731
    reqs = {}
    for line in open(art / "profile_export.jsonl"):
        r = json.loads(line)
        if r["metadata"]["benchmark_phase"] != "profiling" or r.get("error"):
            continue
        v = {k: x["value"] for k, x in r["metrics"].items() if isinstance(x, dict) and "value" in x}
        reqs[(r["metadata"]["conversation_id"], r["metadata"]["turn_index"])] = v
    return dict(id=rid, reqs=reqs, accept=total("sglang:generation_tokens") / total("sglang:spec_verify_calls"),
                cached=total("sglang:cached_tokens") / total("sglang:prompt_tokens"))


def effects(x: dict, *bases: dict) -> dict:
    """Against the mean log of the base runs (two in cohorts 1 and 2, one or two in 3)."""
    runs = (x, *bases)
    isl = lambda r, k: r["reqs"][k]["input_sequence_length"]  # noqa: E731
    keys = [k for k in set.intersection(*(set(r["reqs"]) for r in runs))
            if max(isl(r, k) for r in runs) - min(isl(r, k) for r in runs) <= PREFIX_TOKENS]
    out = {"requests": len(keys)}
    for m in METRICS:
        d = [math.log(x["reqs"][k][m]) - statistics.fmean(math.log(b["reqs"][k][m]) for b in bases)
             for k in keys if all(r["reqs"][k].get(m) for r in runs)]
        out[m] = 100 * (math.exp(statistics.median(d)) - 1) if d else None
    return out


def tail(c: int, r: int, arm: str = "cand") -> list[dict]:
    """The requests behind P90 E2E-normalized interactivity: E2EL/OSL per matched request,
    ranked by the base runs' mean; the arm's median change on all, the slowest 20 % and 10 %."""
    bases = [b for b in (load(ID.format(c=c, arm="base", r=rr)) for rr in (1, 2)) if b]
    x = load(ID.format(c=c, arm=arm, r=r))
    if not (x and bases):
        return []
    runs = (x, *bases)
    isl = lambda R, k: R["reqs"][k]["input_sequence_length"]  # noqa: E731
    keys = [k for k in set.intersection(*(set(R["reqs"]) for R in runs))
            if max(isl(R, k) for R in runs) - min(isl(R, k) for R in runs) <= PREFIX_TOKENS
            and all(R["reqs"][k].get("output_sequence_length") for R in runs)]
    norm = lambda R, k: R["reqs"][k]["request_latency"] / R["reqs"][k]["output_sequence_length"]  # noqa: E731
    lbase = {k: statistics.fmean(math.log(norm(b, k)) for b in bases) for k in keys}
    order = sorted(keys, key=lbase.get)
    out = []
    for label, sel in (("all", order), ("slowest 20%", order[int(0.8 * len(order)):]),
                       ("slowest 10%", order[int(0.9 * len(order)):])):
        def chg(m):
            # a one-token output has no inter-token latency (0), as in effects()
            return 100 * (math.exp(statistics.median(
                math.log(x["reqs"][k][m]) - statistics.fmean(math.log(b["reqs"][k][m]) for b in bases)
                for k in sel if all(R["reqs"][k].get(m) for R in runs))) - 1)
        out.append(dict(conc=c, run=f"{arm}-r{r}", subset=label, requests=len(sel),
                        osl=statistics.median(bases[0]["reqs"][k]["output_sequence_length"] for k in sel),
                        ttft_share=statistics.median(bases[0]["reqs"][k]["time_to_first_token"] / bases[0]["reqs"][k]["request_latency"]
                                                     for k in sel),
                        e2el_per_token=100 * (math.exp(statistics.median(math.log(norm(x, k)) - lbase[k] for k in sel)) - 1),
                        ttft=chg("time_to_first_token"), itl=chg("inter_token_latency")))
    return out


def main() -> int:
    global ID, ARMS, CAND
    cohort = sys.argv[sys.argv.index("--cohort") + 1] if "--cohort" in sys.argv else "1"
    ID, ARMS, CAND = COHORTS[cohort]
    rows = []
    for c in CONCS[cohort]:
        runs = {(a, r): load(ID.format(c=c, arm=a, r=r)) for a, r in ARMS}
        bases = [x for (a, _), x in runs.items() if a == "base" and x]
        if not bases:
            continue
        for (a, r), x in runs.items():
            if x:
                rows.append(dict(conc=c, run=f"{a}-r{r}", ok=len(x["reqs"]), accept=x["accept"],
                                 cached=x["cached"], **effects(x, *bases)))
    f = lambda v: "-" if v is None else f"{v:+.1f}%"  # noqa: E731
    print("conc | run | ok requests | exact accept len | cached share | matched with both base runs | "
          "TTFT vs base mean | ITL vs base mean | E2EL vs base mean")
    for r in rows:
        print(f"c{r['conc']} | {r['run']} | {r['ok']} | {r['accept']:.3f} | {r['cached']:.3f} | {r['requests']} | "
              f"{f(r['time_to_first_token'])} | {f(r['inter_token_latency'])} | {f(r['request_latency'])}")
    tails = [t for c in CONCS[cohort] for r in (1, 2) for t in tail(c, r, CAND)]
    print("\nconc | run | subset (by base E2EL/OSL) | requests | median OSL | TTFT share of E2EL (base r1) | "
          "E2EL/OSL change | TTFT change | ITL change")
    for t in tails:
        print(f"c{t['conc']} | {t['run']} | {t['subset']} | {t['requests']} | {t['osl']:.0f} | {t['ttft_share']:.2f} | "
              f"{f(t['e2el_per_token'])} | {f(t['ttft'])} | {f(t['itl'])}")
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps({"effects": rows, "tail": tails}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
