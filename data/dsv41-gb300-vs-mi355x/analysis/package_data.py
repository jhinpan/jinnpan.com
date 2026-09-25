#!/usr/bin/env python3
"""Copy the evidence behind Experiment 005 into jinnpan.com/data/<slug>/, sanitized.

Hostnames, account names, tailnet/public IPs and machine-local paths are replaced by
placeholders; files are scanned for credential-shaped strings before anything is written.
Per-round CSVs replace the full-response measurement files (the responses are 0.5 MB each
and carry nothing the timing analysis needs beyond an output hash).
"""
import csv
import datetime
import hashlib
import io
import json
import re
import shutil
import sys
from pathlib import Path

CAMP = Path("$CAMPAIGN")
E2E = Path("$E2E_CAMPAIGN")
FLOOR = Path("$BS1_FLOOR")
DEST = Path("$WORKSPACE/jinnpan.com/data/dsv41-gb300-vs-mi355x")

REPLACE = [
    ("$GB300_WORK", "$GB300_WORK"),
    ("$GB300_MODEL_DIR", "$GB300_MODEL_DIR"),
    ("$GB300_HOME", "$GB300_HOME"),
    ("$CAMPAIGN", "$CAMPAIGN"),
    ("$E2E_CAMPAIGN", "$E2E_CAMPAIGN"),
    ("$PA_PR", "$PA_PR"),
    ("$BS1_FLOOR", "$BS1_FLOOR"),
    ("$BS1_STUDY", "$BS1_STUDY"),
    ("$WORKSPACE", "$WORKSPACE"),
    ("$MORI", "$MORI"),
    ("$CAMPAIGN_CACHE", "$CAMPAIGN_CACHE"),
    ("$MODEL_ROOT", "$MODEL_ROOT"),
    ("gb300-tray", "gb300-tray"),
    ("mi355x-node", "mi355x-node"),
    ("$USER", "$USER"),
]
IP = re.compile(r"\b(?!127\.0\.0\.1\b)(?!0\.0\.0\.0\b)\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")
# Split literals keep this file from matching its own patterns when it is archived.
SECRET = re.compile(r"(hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|"
                    r"AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|" + "HF_" + r"TOKEN=[^\s'\"<]{8,}|"
                    r"(?i:authorization: bearer)\s+\S+)")
# nvidia-smi -q board identity lines are hardware serials, not measurement data.
DROP_LINE = re.compile(r"^\s*(Serial Number|GPU UUID|Board Part Number|GPU Part Number|Module ID|"
                       r"Chassis Serial Number|GPU PDI)\b")


README = """# Experiment 005 · DeepSeek-V4.1 Flash, BS=1, GB300 vs MI355X

Raw data behind `public/sources/dsv41-gb300-vs-mi355x.html`. Measured 2026-09-24/25 (UTC).

## Protocol (identical on both machines)

- Model `deepseek-ai/DeepSeek-V4.1-Flash@dba1be0a`, TP4 / EP4, each platform's cookbook cell:
  Low-Latency (DSpark block 5, decode graph cap 64, `--mem-fraction-static 0.8`) or
  High-Throughput (DSpark off); `--disable-radix-cache --random-seed 42` on both.
- Prompt and client: BBuf's random-dspark assets at
  https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/b001f347bab8468d1299fba754c15be9efa0f900/large-language-model/sglang/assets/deepseek-v41-kernel-journey/random-dspark
  (`prompt.json` sha256 `2331eb918994e153dc659a4bd97f5eca87af7751268a5d931a1863a865801307`,
  `benchmark.py` sha256 `8a9276345df9fa7ae71c3f11f73fb59ac02ffcbc82a47d6b25455346ece89952`):
  4,096 random ids, 1,024 output tokens, temperature 0, `ignore_eos`, cache flushed per request,
  one discarded warm-up and six timed rounds per server, a fresh server per launch.
- Modes: `off` (plain decode), `sim` (`SGLANG_RAGGED_VERIFY_MODE=static SGLANG_SIMULATE_ACC_LEN=5.5
  SGLANG_SIMULATE_ACC_METHOD=match-expected`), `real` (natural acceptance).

## Metrics (per timed round, see `analysis/analyze.py`)

- `output_tps` = (completion tokens - first-event tokens) / (last event - first event)  [BBuf's metric]
- `cycle_ms` = (elapsed_s - ttft_s) / verify_steps  [DSpark; independent of acceptance length]
- `step_ms` = (elapsed_s - ttft_s) / (output tokens - first-event tokens)  [plain decode]

## Layout

- `gb300/arms/<arm>/`: `rounds.csv` (per round, output text replaced by its sha256), `summary.json`,
  `status.json`, `manifest.json` (image digest, server command, env, docker args, nvidia-smi summary),
  `docker_run.txt`, `placement_*.txt` (scheduler CPUs, per-thread node, numa_maps), `gpu_samples.csv`
  (200 ms clocks/power in the NUMA control), `server_record_lines.txt`.
  - `s0-*` author's commit 835c3909 on `dev-dsv41` (tp4 = MoE TP4/EP1, ep4, nods = DSpark off), unbound;
    `n0-*` the same with SGLang NUMA binding (`--cap-add SYS_NICE`).
  - `s1a-*` main ffac53d779 (`lmsysorg/sglang@sha256:b8257f5c...`, dev-cu13) with cookbook cells, unbound;
    `n1a-*` the same, NUMA-bound; `e1-base-sim-*` / `e1-nice-sim-*` three-launch NUMA control.
  - `p2-real`, `p2-off`: nsys captures (100 verify cycles, 200 decode steps) with `attribution.json`
    from `gb300/scripts/attribute.py` (the .nsys-rep and sqlite exports are not committed: 115 MB / 320 MB).
- `gb300/scripts/`: the drivers (`arm.py`, `step*.sh`, `exp_numa.sh`, `profile_arm.py`, `attribute.py`).
- `gb300/system/`: lscpu, topology, nvidia-smi (serials removed).
- `mi355x/arms/<arm>/`: `rounds.csv`, `summary.json`, `status.json`, `manifest.json`, `processes.json`
  (every server process: allowed CPUs, NUMA node of resident pages), `server_record_lines.txt`.
  - `u-*` / `n-*` (session 1): SGLang's inherited `SGLANG_SET_CPU_AFFINITY=1` split affinity; `n-*` also
    pins the launcher process tree to node 1 with taskset (SGLang then re-pins the schedulers).
  - `s2-{split,node1,none}-*` (session 2): split affinity; SGLang affinity off plus the whole server on
    node 1; SGLang affinity off and no pinning.
- `mi355x/leases/<lease>/`: the exclusive eight-GPU lease of each session: idle-picker table, HBM
  canaries before and after (256 MiB device copy per GPU), KFD process lists, the lease verdict
  (`exit.json`) and every PID the 5 s process monitor saw (`process-monitor-summary.csv`).
  Session 2's verdict is `contaminated: true` because GPU 4's canary read 10% *faster* after the
  session (6.48 against 7.15 ms). The session was kept: every monitored PID appears and disappears
  inside one of our server launches, the other three canaries moved by under 0.3%, contention can
  only lengthen a copy, and the split-affinity cycles match session 1 within 0.2%.
- `mi355x/scripts/`: `run_arm.sh`, `run_arm2.sh`, `session*.sh`, `lease.sh` and the e2e824 launchers/env.
- `mi355x/prior-base-cycle-ledger/`: the BS=1 verify-cycle ledger from the previous MI355X base
  (genuine acceptance; profiled per-family shares scaled to unprofiled phase totals).
- `analysis/`: `arms.json` (every group), `attribution_compare.json` (both machines in one category
  scheme), `gb300_placement.json`, and the scripts that produced them and the page.

Hostnames, account names, IP addresses and machine-local paths are replaced by placeholders
(`$GB300_WORK`, `$CAMPAIGN`, `$E2E_CAMPAIGN`, `$MODEL_ROOT`, `gb300-tray`, `mi355x-node`, `<ip>`).
"""


def clean(text: str) -> str:
    for old, new in REPLACE:
        text = text.replace(old, new)
    text = IP.sub("<ip>", text)
    text = re.sub(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", "GPU-<uuid>", text)
    text = "\n".join(l for l in text.splitlines() if not DROP_LINE.match(l)) + ("\n" if text.endswith("\n") else "")
    hit = SECRET.search(text)
    if hit:
        raise SystemExit(f"refusing to write: credential-shaped string {hit.group(0)[:12]}...")
    return text


def put(rel: str, text: str):
    path = DEST / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(clean(text))


def put_file(rel: str, src: Path):
    if src.exists():
        put(rel, src.read_text(errors="replace"))


def rounds_csv(measurements: Path) -> str:
    rows = json.loads(measurements.read_text())
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["repeat", "warmup", "output_tps", "accept_length", "verify_steps", "ttft_s", "elapsed_s",
                "first_event_tokens", "output_tokens", "max_repeated_16gram", "output_text_sha256"])
    for r in rows:
        text = (r.get("response") or {}).get("text", "")
        w.writerow([r["repeat"], r["warmup"], r["output_tps"], r.get("accept_length"), r.get("verify_steps"),
                    r["ttft_s"], r["elapsed_s"], r["first_event_tokens"], r["output_tokens"],
                    r.get("max_repeated_16gram"), hashlib.sha256(text.encode()).hexdigest()])
    return buf.getvalue()


def monitor_summary(src: Path) -> str:
    seen = {}
    for line in src.open():
        r = json.loads(line)
        for d in r["devices"]:
            for p in d.get("process_list") or []:
                info = p["process_info"]
                if isinstance(info, dict):
                    s = seen.setdefault(info["pid"], [r["time"], r["time"], set()])
                    s[1] = r["time"]
                    s[2].add(d["gpu"])
    utc = lambda t: datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = ["pid,first_utc,last_utc,amd_smi_gpus"]
    for pid, (first, last, gpus) in sorted(seen.items(), key=lambda kv: kv[1][0]):
        rows.append(f"{pid},{utc(first)},{utc(last)},{' '.join(map(str, sorted(gpus)))}")
    return "\n".join(rows) + "\n"


def gb300_manifest(src: Path) -> str:
    m = json.loads(src.read_text())
    keep = {k: m.get(k) for k in ("name", "image", "image_id", "server_cmd", "env", "docker_args", "port",
                                   "bench", "started_utc", "sample_gpu", "num_steps", "trigger_after")}
    pre = m.get("nvidia_smi_pre", "")
    keep["nvidia_smi_pre_summary"] = [l.strip() for l in pre.splitlines()
                                      if re.search(r"Driver Version|CUDA Version|Current Power Limit|"
                                                   r"Max Power Limit|^\s+SM\s+:|Graphics\s+:", l)][:12]
    if "idle_check" in m:
        keep["idle_check_gpus"] = m["idle_check"].get("gpus")
    return json.dumps(keep, indent=2) + "\n"


def main():
    if DEST.exists():
        shutil.rmtree(DEST)
    # GB300 arms
    for arm in sorted((CAMP / "gb300/results").iterdir()):
        if not arm.is_dir() or arm.name == "_aborted":
            continue
        base = f"gb300/arms/{arm.name}"
        if (arm / "bench/measurements.json").exists():
            put(f"{base}/rounds.csv", rounds_csv(arm / "bench/measurements.json"))
        for f in ("bench/summary.json", "status.json", "docker_run.txt", "placement_ready.txt",
                  "placement_post.txt", "server_record_lines.txt", "gpu_samples.csv", "attribution.json"):
            put_file(f"{base}/{Path(f).name}", arm / f)
        if (arm / "manifest.json").exists():
            put(f"{base}/manifest.json", gb300_manifest(arm / "manifest.json"))
    for f in sorted((CAMP / "gb300/scripts").glob("*")):
        if f.is_file():
            put_file(f"gb300/scripts/{f.name}", f)
    for f in ("lscpu.txt", "topo.txt", "uname.txt", "nvlink-gpu0.txt", "nvidia-smi-q.txt"):
        put_file(f"gb300/system/{f}", CAMP / "gb300/records/system" / f)
    # MI355X arms
    for arm in sorted((CAMP / "results").iterdir()):
        if not arm.is_dir() or arm.name.startswith("gpu-claim"):
            continue
        base = f"mi355x/arms/{arm.name}"
        if (arm / "bench/measurements.json").exists():
            put(f"{base}/rounds.csv", rounds_csv(arm / "bench/measurements.json"))
        for f in ("bench/summary.json", "status.json", "manifest.json", "processes.json"):
            put_file(f"{base}/{Path(f).name}", arm / f)
        log = (arm / "server.log")
        if log.exists():
            keep = [l for l in log.read_text(errors="replace").splitlines()
                    if len(l) < 600 and re.search(r"is running on CPUs|Load weight end|Memory pool end|"
                                                  r"Capture .* CUDA graph end|attention backend|moe runner|"
                                                  r"flashinfer|aiter_sparse|DSV4 SWA sizing|SIMULATE|"
                                                  r"NUMA|numa|affinity", l)]
            put(f"{base}/server_record_lines.txt", "\n".join(keep) + "\n")
    for lease in ("gpu-claim-session1-a2", "gpu-claim-session2"):
        src = CAMP / "results" / lease
        for f in sorted(src.glob("canary-*.json")):
            put_file(f"mi355x/leases/{lease}/{f.name}", f)
        for name in ("picker.txt", "command.json", "exit.json", "kfd-before.json", "kfd-after.json"):
            put_file(f"mi355x/leases/{lease}/{name}", src / name)
        put(f"mi355x/leases/{lease}/process-monitor-summary.csv", monitor_summary(src / "process-monitor.jsonl"))
    for f in ("run_arm.sh", "run_arm2.sh", "session1.sh", "session2.sh", "launch_ll_sim.sh", "lease.sh",
              "snapshot_procs.py", "wait_free.py"):
        put_file(f"mi355x/scripts/{f}", CAMP / "bench" / f)
    for f in ("env.sh", "launch_ll.sh", "launch_ht.sh"):
        put_file(f"mi355x/scripts/e2e824/{f}", E2E / "bench" / f)
    put_file("mi355x/scripts/e2e824/rt-base.json", E2E / "sources/rt-base.json")
    for f in ("kernel-ledger.csv", "kernel-ledger-summary.json", "budget.md"):
        put_file(f"mi355x/prior-base-cycle-ledger/{f}", FLOOR / "analysis" / f)
    # Analysis
    for f in ("arms.json", "gb300_placement.json", "attribution_compare.json", "analyze.py",
              "compare_attribution.py", "package_data.py", "build_page.py"):
        put_file(f"analysis/{f}", CAMP / "analysis" / f)
    put("README.md", README)
    n = sum(1 for _ in DEST.rglob("*") if _.is_file())
    size = sum(p.stat().st_size for p in DEST.rglob("*") if p.is_file())
    print(f"{n} files, {size / 1e6:.2f} MB under {DEST}")


if __name__ == "__main__":
    sys.exit(main())
