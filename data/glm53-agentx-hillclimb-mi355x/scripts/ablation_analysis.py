#!/usr/bin/env python3
"""Per-PR ablation of the glm53 monokernel on GLM-5.3 (InferenceX AgentX, TP8, MTP 3-1-4).

Two ladders:
  bf16: the PR campaign (image 20261019, SGLang e45cbce599, 3600 s, BF16 KV in both arms):
        the normal path (mtpref) against #15 (mtpmono), #24 (mtpptopk) and #42 (mtpscore).
  fp8:  the recipe's FP8 E4M3 KV (image 20261024, SGLang fe1bd10837, 900 s): cohort 2's base
        against F1 = #7 #15 (+#60), F2 = F1 + #24 (+#60), F3 = F2 + #42 + #60 (cohort 2's cand8).

Per run: the request-paired effects of l1_pair.effects (median log change against the base runs),
the exact accept length, and the share of decode-batch log lines with at most 2 running requests.
Headline metrics come from InferenceX's compare_agentx.py.

    python3 ablation_analysis.py OUT.json
"""
import json
import math
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path.home() / "amdmono-sa"
RUNS = ROOT / "runs"
sys.path.insert(0, str(ROOT / "scripts"))
import l1_pair  # noqa: E402

COMPARE = Path.home() / ".cursor/skills/inferencex-perf-submission/scripts/compare_agentx.py"
CONCS = (1, 2, 4)
BF16 = {"base": ["agentx-glm53-tp8-c{c}-mtpref"],
        "steps": [("#15", "agentx-glm53-tp8-c{c}-mtpmono"), ("#24", "agentx-glm53-tp8-c{c}-mtpptopk"),
                  ("#42", "agentx-glm53-tp8-c{c}-mtpscore")]}
FP8 = {"base": ["agentx-glm53-l1f-c{c}-base-r1", "agentx-glm53-l1f-c{c}-base-r2"],
       "steps": [("F1", "agentx-glm53-abl-c{c}-f1-r1"), ("F2", "agentx-glm53-abl-c{c}-f2-r1"),
                 ("F3", "agentx-glm53-l1f-c{c}-cand8-r1"), ("F3", "agentx-glm53-l1f-c{c}-cand8-r2")],
       "drift": [(2, "agentx-glm53-abl-c2-base-r3")],
       "aa": ["agentx-glm53-l1f-c1-base-r1", "agentx-glm53-l1f-c1-base-r2"]}


EDGES = (0, 64_000, 128_000, 192_000, 10 ** 9)


def request_diffs(x, bases, m="inter_token_latency"):
    """(input length, log x - mean log base) per request that x shares with every base run."""
    runs = (x, *bases)
    isl = lambda r, k: r["reqs"][k]["input_sequence_length"]  # noqa: E731
    keys = [k for k in set.intersection(*(set(r["reqs"]) for r in runs))
            if max(isl(r, k) for r in runs) - min(isl(r, k) for r in runs) <= l1_pair.PREFIX_TOKENS]
    out = []
    for k in keys:
        if all(r["reqs"][k].get(m) for r in runs):
            base_log = sum(math.log(b["reqs"][k][m]) for b in bases) / len(bases)
            out.append((sum(isl(b, k) for b in bases) / len(bases), math.log(x["reqs"][k][m]) - base_log))
    return out


def by_isl(diffs):
    rows = []
    for lo, hi in zip(EDGES, EDGES[1:]):
        d = sorted(v for i, v in diffs if lo <= i < hi)
        if d:
            med = d[len(d) // 2] if len(d) % 2 else (d[len(d) // 2 - 1] + d[len(d) // 2]) / 2
            rows.append({"lo": lo, "hi": hi, "n": len(d), "itl": 100 * (math.exp(med) - 1)})
        else:
            rows.append({"lo": lo, "hi": hi, "n": 0, "itl": None})
    return rows


def coverage(rid):
    n = small = 0
    for line in open(RUNS / rid / "server.log", errors="replace"):
        if "Decode batch" in line:
            m = re.search(r"#running-req: (\d+)", line)
            if m:
                n += 1
                small += int(m.group(1)) <= 2
    return round(small / n, 4) if n else None


def routed(rid):
    text = (RUNS / rid / "server.log").read_text(errors="replace")
    m = re.search(r"monokernel \S+ on gfx950 runs target_verify[^\n]*", text)
    return m.group(0) if m else None


def agg(rid):
    return str(RUNS / rid / "agentx_out" / f"{rid}.json")


def done(rid):
    st = RUNS / rid / "state.json"
    return st.exists() and json.loads(st.read_text())["status"] == "done" and Path(agg(rid)).exists()


def headline(base_ids, cand_ids, noise_args, out):
    argv = [sys.executable, str(COMPARE), "--baseline", *map(agg, base_ids), "--candidate", *map(agg, cand_ids),
            *noise_args, "--json", str(out)]
    r = subprocess.run(argv, capture_output=True, text=True)
    if r.returncode not in (0, 1) or not out.exists():
        return {"error": (r.stdout + r.stderr)[-600:]}
    return json.loads(out.read_text())


def ladder(spec, noise_args, tmp):
    res = {"effects": [], "headline": {}, "by_isl": {}}
    pooled = {}
    for c in CONCS:
        bases = [l1_pair.load(b.format(c=c)) for b in spec["base"]]
        if not all(bases):
            continue
        for name, tmpl in spec["steps"]:
            rid = tmpl.format(c=c)
            if not done(rid):
                continue
            x = l1_pair.load(rid)
            e = l1_pair.effects(x, *bases)
            diffs = request_diffs(x, bases)
            if c in (1, 2):
                pooled.setdefault(name, []).extend(diffs)
            res["effects"].append({"step": name, "conc": c, "run": rid, **e, "accept": x["accept"],
                                   "coverage": coverage(rid), "route": routed(rid), "by_isl": by_isl(diffs)})
        for c2, rid in spec.get("drift", []):
            if c2 == c and done(rid):
                x = l1_pair.load(rid)
                res["effects"].append({"step": "base rerun", "conc": c, "run": rid, **l1_pair.effects(x, *bases),
                                       "accept": x["accept"], "coverage": None, "route": routed(rid)})
    res["by_isl"] = {name: by_isl(d) for name, d in pooled.items()}
    for name, tmpl in spec["steps"]:
        cands = [tmpl.format(c=c) for c in CONCS if done(tmpl.format(c=c))]
        bases = [spec["base"][0].format(c=c) for c in CONCS if done(tmpl.format(c=c))]
        if cands:
            key = f"{name} {tmpl.split('{c}')[-1]}"
            res["headline"][key] = headline(bases, cands, noise_args, tmp / f"cmp_{key.replace(' ', '_').replace('#', '')}.json")
    return res


def main():
    out = Path(sys.argv[1])
    tmp = out.parent / "cmp"
    tmp.mkdir(parents=True, exist_ok=True)
    result = {"bf16": ladder(BF16, ["--noise", "0.027"], tmp),
              "fp8": ladder(FP8, ["--aa", *map(agg, FP8["aa"])], tmp)}
    out.write_text(json.dumps(result, indent=1))
    for lad in ("bf16", "fp8"):
        print(f"== {lad}")
        for e in result[lad]["effects"]:
            print(f"  c{e['conc']} {e['step']:10s} n={e['requests']:4d} TTFT {e['time_to_first_token']:+6.1f}% "
                  f"ITL {e['inter_token_latency']:+6.1f}% E2EL {e['request_latency']:+6.1f}% "
                  f"accept {e['accept']:.3f} cov {e['coverage']} route {'yes' if e['route'] else 'no'}")


if __name__ == "__main__":
    main()
