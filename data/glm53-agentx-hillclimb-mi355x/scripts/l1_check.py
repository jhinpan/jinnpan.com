#!/usr/bin/env python3
"""Validity checks and comparisons for the GLM-5.3 InferenceX L1 cohorts (``--cohort``: 1 =
phase infxl1, bf16 KV candidate; 2 = phase infxl1fp8, fp8 KV candidate).

    l1_check.py runs        one line per run: status, window, accounting, validator,
                            realized accept len, route, share of decode steps the kernels cover
    l1_check.py compare     compare_agentx.py per round and per diagnostic pair; noise floor
                            = the A/A pair base c1 r1 vs base c1 r2
    l1_check.py gsm8k       the quality arms' GSM8K, paired on the same questions (McNemar)
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
# cohort -> (plan phase, AgentX run-id prefix, GSM8K run-id prefix, candidate arm, output dir)
COHORTS = {
    "1": ("infxl1", "agentx-glm53-l1", "gsm8k-glm53-l1-c2", "cand", ROOT / "infx-l1-glm53-20261025"),
    "2": ("infxl1fp8", "agentx-glm53-l1f", "gsm8k-glm53-l1f-c2", "cand8",
          ROOT / "infx-l1-glm53-fp8-20261025"),
    "3": ("infxl2ov", "agentx-glm53-l2ov", "gsm8k-glm53-l2ov-c10", "cand8",
          ROOT / "infx-l2ov-glm53-20261026"),
}
WINDOWS = {"1": 900, "2": 900, "3": 3600}
# cohort -> (rounds, concurrencies, the A/A pair's concurrency)
DESIGN = {"1": ((1, 2), (1, 2, 4), 1), "2": ((1, 2), (1, 2, 4), 1), "3": ((1,), (1, 2, 4, 10), 2)}
ROUNDS, CONCS, AA_CONC = DESIGN["1"]
PHASE, PREFIX, GSM8K, CAND, OUT = COHORTS["1"]
COMPARE = Path.home() / ".cursor/skills/inferencex-perf-submission/scripts/compare_agentx.py"
GOLDEN_AL = 2.99
AL_TOL = 0.05  # median realized accept len within GOLDEN_AL +- AL_TOL (contract)
WINDOW_S = 900
FAILED_FRACTION = 0.10
COVERED_REQS = 2  # 4 rows a request (MTP K=3); the TP8 kernels take up to 8 rows
DECODE = re.compile(r"Decode batch, #running-req: (\d+),.*?accept len: ([\d.]+)")
# AIPerf's profiling phase: a 900-s target, then a drain; a drain past its grace cancels the
# in-flight requests. duration_seconds spans the completed requests only.
PHASE_START = re.compile(r"^(\d\d):(\d\d):([\d.]+) NOTICE +Phase profiling \(profiling\) started.*target: ([\d.]+)s", re.M)
PHASE_DONE = re.compile(r"^(\d\d):(\d\d):([\d.]+) NOTICE +Phase profiling \(profiling\) complete \| "
                        r"completed=([\d,]+), cancelled=([\d,]+), errors=([\d,]+)", re.M)
PHASE_TIMEOUT = re.compile(r"^(\d\d):(\d\d):([\d.]+) WARNING +Phase profiling timed out.*?completed=([\d,]+), "
                           r"cancelled=([\d,]+), in_flight=([\d,]+)", re.M)


def clock(h, m, s) -> float:
    return int(h) * 3600 + int(m) * 60 + float(s)


def phase(log: str) -> dict:
    """The profiling phase's target, elapsed time to its end line, and drain cancellations."""
    st, done, to = PHASE_START.search(log), PHASE_DONE.search(log), PHASE_TIMEOUT.search(log)
    if not st or not (done or to):
        return {}
    end = done or to
    elapsed = (clock(*end.groups()[:3]) - clock(*st.groups()[:3])) % 86400
    n = lambda g: int(g.replace(",", ""))  # noqa: E731  AIPerf prints 1,411
    cancelled = n(done.group(5)) if done else n(to.group(5)) + n(to.group(6))
    return {"target": float(st.group(4)), "elapsed": elapsed, "drain_cancelled": cancelled,
            "timed_out": bool(to)}


def dig(d, *path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def aggregate(rid: str) -> dict | None:
    p = RUNS / rid / "agentx_out" / f"{rid}.json"
    return json.loads(p.read_text()) if p.exists() else None


def check(run: dict) -> dict:
    rid = run["id"]
    d = RUNS / rid
    st = json.loads((d / "state.json").read_text()) if (d / "state.json").exists() else {}
    row = {"id": rid, "status": st.get("status", "pending"), "attempts": st.get("attempts", 0),
           "corpus": ",".join(r[:12] for r in st.get("corpus_rev", [])), "problems": []}
    if row["status"] != "done":
        return row
    info = json.loads((d / "server_info.json").read_text())
    if bool(info["route"]) != run["expect_route"]:
        row["problems"].append(f"route {info['route']} but expected {run['expect_route']}")
    steps = [(int(n), float(a)) for n, a in DECODE.findall((d / "server.log").read_text(errors="replace"))]
    if steps:
        als = sorted(a for _, a in steps)
        row["al_med"] = statistics.median(als)
        row["al_p5"], row["al_p95"] = als[int(0.05 * (len(als) - 1))], als[int(0.95 * (len(als) - 1))]
        row["covered"] = sum(n <= COVERED_REQS for n, _ in steps) / len(steps)
        row["max_running"] = max(n for n, _ in steps)
    simulated = "SGLANG_SIMULATE_ACC_LEN" in run["env"]
    if simulated and abs(row.get("al_med", 0) - GOLDEN_AL) > AL_TOL:
        row["problems"].append(f"realized accept len {row.get('al_med')} misses {GOLDEN_AL}")
    if run["client"]["kind"] != "agentx_rendered":
        return row
    agg = aggregate(rid)
    if agg is None:
        row["problems"].append("no aggregate JSON")
        return row
    acc = agg.get("request_accounting", {})
    row.update(duration=dig(agg, "request_metrics", "throughput", "duration_seconds"),
               ok=agg.get("num_requests_successful"), total=agg.get("num_requests_total"),
               warmup=acc.get("records_warmup_dropped"), err=acc.get("records_error_dropped"),
               tput=dig(agg, "request_metrics", "throughput", "per_gpu", "total_tput_tps"),
               int90=dig(agg, "request_metrics", "latency", "e2e_norm_intvty", "p90"),
               e2el90=dig(agg, "request_metrics", "latency", "e2el", "p90"))
    log = (d / "agentx_client.log").read_text(errors="replace")
    ph = phase(log)
    row.update(phase_s=ph.get("elapsed"), drain=ph.get("drain_cancelled"))
    if not ph or ph["target"] != WINDOW_S or ph["elapsed"] < WINDOW_S:
        row["problems"].append(f"profiling phase did not run its {WINDOW_S}-s target: {ph}")
    # records_error_dropped also counts warmup primers (one token, no content); the validator
    # counts the profiling window only
    v = re.search(r"Validated aiperf request error rate: ([\d.]+)/([\d.]+) =", log)
    if v is None or re.search(r"^ERROR: ", log, re.M):
        row["problems"].append("validator did not pass")
    else:
        failed, done = float(v.group(1)) + (row["drain"] or 0), float(v.group(2)) + (row["drain"] or 0)
        row["window_failed"] = int(failed)
        if done and failed / done > FAILED_FRACTION:
            row["problems"].append(f"{failed:g}/{done:g} window requests failed or were cancelled")
    return row


def plan() -> list[dict]:
    return [r for r in json.loads((ROOT / "configs/plan.json").read_text()) if r["phase"] == PHASE]


def fmt(v, spec):
    return "-" if v is None else format(v, spec)


def cmd_runs() -> int:
    rows = [check(r) for r in plan()]
    print(f"{'run':<34} {'status':<14} {'span s':>6} {'phase s':>7} {'ok/total':>9} {'wu':>3} {'err':>3} {'drn':>3} "
          f"{'tok/s/GPU':>9} {'P90 int':>8} {'P90 E2EL':>8} {'AL med [p5,p95]':>18} {'cov':>5} {'maxrun':>6}  problems")
    for r in rows:
        al = f"{fmt(r.get('al_med'), '.2f')} [{fmt(r.get('al_p5'), '.2f')},{fmt(r.get('al_p95'), '.2f')}]"
        print(f"{r['id']:<34} {r['status']:<14} {fmt(r.get('duration'), '.0f'):>6} {fmt(r.get('phase_s'), '.0f'):>7} "
              f"{fmt(r.get('ok'), 'd')}/{fmt(r.get('total'), 'd'):<4} {fmt(r.get('warmup'), 'd'):>3} "
              f"{fmt(r.get('err'), 'd'):>3} {fmt(r.get('drain'), 'd'):>3} {fmt(r.get('tput'), '.1f'):>9} {fmt(r.get('int90'), '.1f'):>8} "
              f"{fmt(r.get('e2el90'), '.1f'):>8} {al:>18} {fmt(r.get('covered'), '.2f'):>5} "
              f"{fmt(r.get('max_running'), 'd'):>6}  {'; '.join(r['problems']) or ('ok' if r['status'] == 'done' else '')}")
    (OUT / "run_checks.json").write_text(json.dumps(rows, indent=1))
    return 0


def agg_path(rid: str) -> str:
    return str(RUNS / rid / "agentx_out" / f"{rid}.json")


def cmd_compare() -> int:
    valid = {r["id"] for r in (check(x) for x in plan()) if r["status"] == "done" and not r["problems"]}
    aa = [f"{PREFIX}-c{AA_CONC}-base-r1", f"{PREFIX}-c{AA_CONC}-base-r2"]
    if not set(aa) <= valid:
        print("the A/A pair is not complete and valid yet")
        return 1
    pairs = [(f"round {r}: base vs {CAND}", "base", CAND, f"r{r}", CONCS) for r in ROUNDS]
    pairs += [] if CAND != "cand" else [("diagnostic: base vs b0 (bf16 KV alone)", "base", "b0", "r1", (1, 2)),
              ("diagnostic: b0 vs cand (the kernels alone)", "b0", "cand", "r1", (1, 2)),
              ("diagnostic: b1 vs cand (05e0bb02 vs 9484c60)", "b1", "cand", "r1", (1, 2))]
    rc = 0
    for title, a, b, rnd, concs in pairs:
        ids_a = [f"{PREFIX}-c{c}-{a}-{rnd}" for c in concs]
        ids_b = [f"{PREFIX}-c{c}-{b}-{rnd}" for c in concs]
        ok_a, ok_b = [i for i in ids_a if i in valid], [i for i in ids_b if i in valid]
        if not ok_a or not ok_b:
            print(f"\n## {title}: not run or not valid yet")
            continue
        tag = title.split(":")[0].replace(" ", "")
        argv = [sys.executable, str(COMPARE), "--baseline", *map(agg_path, ok_a),
                "--candidate", *map(agg_path, ok_b), "--aa", *map(agg_path, aa),
                "--json", str(OUT / f"compare_{tag}_{a}_{b}.json")]
        if a == "base" and b == CAND:
            argv += ["--pareto-input", str(OUT / f"pareto_{tag}.json")]
        print(f"\n## {title}")
        r = subprocess.run(argv, capture_output=True, text=True)
        print(r.stdout + r.stderr)
        rc |= r.returncode
    return rc


GSM8K_GATE = 0.94  # InferenceX infx/evals/thresholds.yaml, family glm5


def gsm8k_scores(rid: str) -> dict[str, dict[int, int]]:
    """Per lm-eval filter (InferenceX validates every `exact_match,<filter>`), doc_id -> 0/1."""
    files = sorted((RUNS / rid / "gsm8k").rglob("samples_gsm8k*.jsonl"))
    if len(files) != 1:
        raise SystemExit(f"{rid}: expected one samples_gsm8k*.jsonl, found {files}")
    scores: dict[str, dict[int, int]] = {}
    # file iteration splits at "\n" only; str.splitlines also splits at U+2028 inside outputs
    for line in open(files[0], encoding="utf-8"):
        s = json.loads(line)
        per = scores.setdefault(s["filter"], {})
        if s["doc_id"] in per:
            raise SystemExit(f"{rid}: doc {s['doc_id']} twice under filter {s['filter']}")
        per[s["doc_id"]] = int(s["exact_match"])
    return scores


def mcnemar_p(b: int, c: int) -> float:
    from math import comb
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def cmd_gsm8k() -> int:
    arms = gsm8k_scores(f"{GSM8K}-baseq"), gsm8k_scores(f"{GSM8K}-{CAND}q")
    if set(arms[0]) != set(arms[1]):
        raise SystemExit(f"the arms report different filters: {sorted(arms[0])} vs {sorted(arms[1])}")
    res = {"gate": GSM8K_GATE, "filters": {}}
    for f in sorted(arms[0]):
        base, cand = arms[0][f], arms[1][f]
        if set(base) != set(cand):
            raise SystemExit(f"{f}: the arms answered different questions: {len(base)} vs {len(cand)}")
        n = len(base)
        only_b = sum(base[i] and not cand[i] for i in base)
        only_c = sum(cand[i] and not base[i] for i in base)
        res["filters"][f] = {"n": n, "base": sum(base.values()) / n, "cand": sum(cand.values()) / n,
                             "only_base_right": only_b, "only_cand_right": only_c,
                             "mcnemar_p": mcnemar_p(only_b, only_c)}
    res["pass"] = all(min(v["base"], v["cand"]) >= GSM8K_GATE for v in res["filters"].values())
    print(json.dumps(res, indent=1))
    (OUT / "gsm8k_paired.json").write_text(json.dumps(res, indent=1))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=("runs", "compare", "gsm8k"))
    p.add_argument("--cohort", choices=sorted(COHORTS), default="1")
    a = p.parse_args()
    global PHASE, PREFIX, GSM8K, CAND, OUT, WINDOW_S, ROUNDS, CONCS, AA_CONC
    PHASE, PREFIX, GSM8K, CAND, OUT = COHORTS[a.cohort]
    WINDOW_S = WINDOWS[a.cohort]
    ROUNDS, CONCS, AA_CONC = DESIGN[a.cohort]
    return {"runs": cmd_runs, "compare": cmd_compare, "gsm8k": cmd_gsm8k}[a.cmd]()


if __name__ == "__main__":
    sys.exit(main())
