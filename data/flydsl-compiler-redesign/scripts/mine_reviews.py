"""Mine FlyBench review bundles (overlay): what agents changed, what went wrong, and what it cost.

Inputs: $BENCH_RUNS/bench-flybench-*/{plan.json,review/index.jsonl,
review/verdicts/*.json,review/traces/*.md}. Output: mining/*.json and a printed summary.
"""

import collections
import glob
import json
import os
import re
from pathlib import Path

ROOT = "$BENCH_RUNS"
OUT = Path(__file__).resolve().parent / "mining"
OUT.mkdir(exist_ok=True)

# What the speedup came from, read from the reviewer's one-sentence summary (lower-cased).
SOURCES = {
    "dispatch_or_tiling": r"dispatch|tile|tiling|retun|retil|block size|block geometry|cta|planner|bucket|geometry|kernel for|routes|sends",
    "split_k_or_grid_fill": r"split-k|split-kv|splitk|split k|under-?fill|under-?occup|occupancy|persistent|grid",
    "vectorize_or_memory": r"vectori|128-bit|16-byte|vec4|vec8|wider|coalesc|non-temporal|lds|register-resident|registers|prefetch|pipelin|ring",
    "algorithm_rewrite": r"rewrite|replaces|fuses|fused|single-pass|single pass|two-kernel|atomics|reduction",
    "schedule_or_sync_knob": r"sched|post-ra|post-misched|misched|lgkmcnt|waitcnt|setprio|priority|interleav",
    "register_or_occupancy_knob": r"agpr|waves_per_eu|vgpr|spill",
}

# Failure classes, read from summary + checklist evidence.
FAILURES = {
    "regresses_off_timed_shapes": r"regress|slower|0\.[0-9]+x",
    "race_or_missing_barrier": r"missing barrier|race|barrier corrupt",
    "out_of_bounds": r"out-of-bounds|out of bounds|\boob\b",
    "numerics": r"nan|inf\b|overflow|precision|max-subtraction|max-free|fast_fp_math",
    "spill": r"spill",
    "host_overhead_or_cache": r"host[- ]side|host overhead|launcher|jit fast path|nested-jit|cached host",
    "cache_gaming": r"cold-only|cold cache|cache hint|non-temporal",
}

# Agent commands by purpose (trace lines inside ``` blocks).
COMMANDS = {
    "evaluate": r"\./evaluate\b",
    "own_timing_or_profile": r"rocprof|abba|torch\.profiler|timeit|bench\.py|benchmark|perf_counter|do_bench",
    "inspect_ir_or_isa": r"FLYDSL_DUMP_IR|DUMP_ASM|final_isa|llvm-objdump|--disassemble|\bamdgcn\b|\bvgpr|\bsgpr|\bagpr|spill_count|scratch_size|\.s\b",
    "cache_clear": r"rm -rf [^\n]*flydsl|ENABLE_CACHE=0|\.flydsl/cache",
    "read_code": r"^\s*(cat|sed -n|head|tail|rg|grep|less|wc|ls|find)\b",
    "edit_code": r"apply_patch|sed -i|cat >|python3? - <<|tee |perl -pi|write_text",
    "run_tests": r"\bpytest\b",
}


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def hits(table, text):
    return [k for k, p in table.items() if re.search(p, text, re.I | re.M)]


reviews, index_rows, traces = [], [], []
for run_dir in sorted(glob.glob(f"{ROOT}/bench-flybench-*")):
    run = os.path.basename(run_dir)
    plan = load(f"{run_dir}/plan.json") or {}
    model = (plan.get("model") or {}).get("fleet_name")
    harness = (plan.get("harness") or {}).get("name")
    idx_path = f"{run_dir}/review/index.jsonl"
    if os.path.exists(idx_path):
        for line in open(idx_path):
            try:
                row = json.loads(line)
            except Exception:
                continue
            agent = row.get("agent") or {}
            tok = agent.get("tokens") or {}
            index_rows.append({"run": run, "model": model, "harness": harness, "record": row.get("record"),
                               "verdict": row.get("verdict"), "ratio": (row.get("speedup") or {}).get("ratio"),
                               "turns": agent.get("turns"), "tool_calls": agent.get("tool_calls"),
                               "prompt_tokens": tok.get("prompt_tokens"), "completion_tokens": tok.get("completion_tokens"),
                               "wall_s": agent.get("wall_s"), "lines_added": row.get("lines_added"),
                               "lines_removed": row.get("lines_removed")})
    for vf in sorted(glob.glob(f"{run_dir}/review/verdicts/*.json")):
        if vf.endswith((".receipt.json", "summary.json")):
            continue
        v = load(vf) or {}
        if "verdict" not in v:
            continue
        evidence = " ".join(str((c or {}).get("evidence", "")) for c in (v.get("checklist") or {}).values())
        summary = v.get("summary") or ""
        failed_items = sorted(k for k, c in (v.get("checklist") or {}).items() if (c or {}).get("result") == "fail")
        failed_text = " ".join(str((v.get("checklist") or {})[k].get("evidence", "")) for k in failed_items)
        reviews.append({"run": run, "model": model, "record": v.get("record"), "verdict": v.get("verdict"),
                        "reproduced_ratio": v.get("reproduced_ratio"), "summary": summary,
                        "sources": hits(SOURCES, summary), "failures_summary": hits(FAILURES, summary),
                        "failed_items": failed_items, "failures_failed_items": hits(FAILURES, summary + " " + failed_text)})
    for tf in sorted(glob.glob(f"{run_dir}/review/traces/*.md")):
        text = open(tf, errors="replace").read()
        blocks = re.findall(r"```\n(.*?)```", text, re.S)
        cats = collections.Counter()
        for b in blocks:
            for k in hits(COMMANDS, b):
                cats[k] += 1
        traces.append({"run": run, "model": model, "record": Path(tf).stem, "commands": len(blocks),
                       "by_purpose": dict(cats)})

(OUT / "reviews.json").write_text(json.dumps(reviews, indent=1))
(OUT / "review_index.json").write_text(json.dumps(index_rows, indent=1))
(OUT / "traces.json").write_text(json.dumps(traces, indent=1))

print(f"reviews={len(reviews)} verdicts={dict(collections.Counter(r['verdict'] for r in reviews))}")
legit = [r for r in reviews if r["verdict"] == "legit"]
for name, group in (("all reviewed", reviews), ("legit only", legit)):
    src = collections.Counter(s for r in group for s in r["sources"])
    print(f"\n[{name}] n={len(group)} speedup source (share of reviews mentioning it):")
    for k, v in src.most_common():
        print(f"  {k:28s} {v:3d}  {v / len(group):.0%}")
ITEMS = {"1": "cold-cache-only tricks", "2": "shape-specific dispatch", "3": "caching across calls",
         "4": "tolerance exploitation", "5": "work outside timed window", "6": "changes outside kernel files",
         "7": "per-shape regressions vs stock"}
items = collections.Counter(i for r in reviews for i in r["failed_items"])
print(f"\n[failed checklist items] n={len(reviews)}")
for k, v in sorted(items.items()):
    print(f"  {k} {ITEMS.get(k, '?'):32s} {v:3d}  {v / len(reviews):.0%}")
fails = collections.Counter(f for r in reviews for f in r["failures_failed_items"])
print(f"\n[failure classes in summary + failed-item evidence] n={len(reviews)}")
for k, v in fails.most_common():
    print(f"  {k:28s} {v:3d}  {v / len(reviews):.0%}")
ratios = [r["reproduced_ratio"] for r in legit if isinstance(r["reproduced_ratio"], (int, float))]
ratios.sort()
if ratios:
    print(f"\nlegit reproduced speedup: median {ratios[len(ratios) // 2]:.2f}x, min {ratios[0]:.2f}x, max {ratios[-1]:.2f}x")

cost = [r for r in index_rows if r["prompt_tokens"]]
if cost:
    def med(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2]
    print(f"\n[agent cost per record] n={len(cost)} median prompt tokens {med([r['prompt_tokens'] for r in cost]) / 1e6:.1f}M, "
          f"median turns {med([r['turns'] or 0 for r in cost])}, median wall {med([r['wall_s'] or 0 for r in cost]) / 3600:.1f} h")
    by_model = collections.defaultdict(list)
    for r in cost:
        by_model[r["model"]].append(r)
    for m, rs in by_model.items():
        print(f"  {m}: n={len(rs)} median tokens {med([x['prompt_tokens'] for x in rs]) / 1e6:.1f}M "
              f"turns {med([x['turns'] or 0 for x in rs])} wall {med([x['wall_s'] or 0 for x in rs]) / 3600:.1f} h")

tot = collections.Counter()
ncmd = 0
for t in traces:
    tot.update(t["by_purpose"])
    ncmd += t["commands"]
print(f"\n[agent commands] traces={len(traces)} commands={ncmd}")
for k, v in tot.most_common():
    print(f"  {k:22s} {v:6d}  {v / max(ncmd, 1):.1%} of commands")
