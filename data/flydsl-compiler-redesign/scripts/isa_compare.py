"""Compare stock vs candidate ISA per arm from results/<job>/flybench-eval/<label>/zz-isa.tar.gz.

Positive control: an arm that changes nothing in the ISA cannot change the time. Writes isa_stats.json.
"""

import collections
import hashlib
import io
import json
import re
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
META = re.compile(r"^\s+\.(vgpr_count|agpr_count|sgpr_count|vgpr_spill_count|sgpr_spill_count|"
                  r"private_segment_fixed_size|group_segment_fixed_size):\s+(\d+)", re.M)


def parse_isa(text):
    meta = {k: int(v) for k, v in META.findall(text)}
    ops = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith((".", ";", "//")) or s.endswith(":"):
            continue
        op = s.split()[0]
        if re.match(r"^[a-z][a-z0-9_]+$", op) and "_" in op:
            ops.append(s.split(";")[0].strip())
    mnem = collections.Counter(o.split()[0] for o in ops)

    def grp(p):
        return sum(v for k, v in mnem.items() if re.match(p, k))

    return {"meta": meta, "insts": len(ops), "mfma": grp(r"v_(mfma|smfmac|wmma)"), "s_waitcnt": grp(r"s_waitcnt"),
            "s_nop": grp(r"s_nop"), "s_setprio": grp(r"s_setprio"), "v_accvgpr": grp(r"v_accvgpr_"),
            "scratch_ops": grp(r"scratch_|buffer_store_dword") if meta.get("private_segment_fixed_size") else 0,
            "order_sha": hashlib.sha256("\n".join(ops).encode()).hexdigest()[:16],
            "multiset_sha": hashlib.sha256("\n".join(sorted(ops)).encode()).hexdigest()[:16]}


def load_arm(tar_path):
    """{(side, shape): {kernel_name: stats}} plus the logs' return codes."""
    out, rcs = {}, {}
    with tarfile.open(tar_path) as tar:
        for m in tar.getmembers():
            parts = Path(m.name).parts  # isa/<side>/<shape>/...
            if len(parts) < 4 or not m.isfile():
                continue
            side, shape = parts[1], parts[2]
            data = tar.extractfile(m).read().decode(errors="replace")
            if parts[-1] == "log.txt":
                rc = re.findall(r"rc=(-?\d+)\s*$", data)
                rcs[(side, shape)] = int(rc[-1]) if rc else None
            elif parts[-1].endswith("final_isa.s"):
                kernel = parts[-2]
                out.setdefault((side, shape), {})[kernel] = parse_isa(data)
    return out, rcs


def main(job="pilot"):
    root = HERE / "results" / job / "flybench-eval"
    exps = {e["label"]: e for e in json.loads((HERE / "experiments.json").read_text())}
    report = {}
    for label in sorted(exps):
        tar_path = root / label / "zz-isa.tar.gz"
        if not tar_path.exists():
            print(f"{label}: no ISA tarball")
            continue
        apply_txt = (root / label / "isa" / "apply.txt").read_text() if (root / label / "isa" / "apply.txt").exists() else ""
        stats, rcs = load_arm(tar_path)
        shapes = sorted({s for _side, s in stats} | {s for _side, s in rcs})
        rows = []
        for shape in shapes:
            st, ca = stats.get(("stock", shape), {}), stats.get(("cand", shape), {})
            for kernel in sorted(set(st) | set(ca)):
                a, b = st.get(kernel), ca.get(kernel)
                rows.append({"shape": shape, "kernel": kernel, "stock": a, "cand": b,
                             "same_order": bool(a and b and a["order_sha"] == b["order_sha"]),
                             "same_multiset": bool(a and b and a["multiset_sha"] == b["multiset_sha"])})
        report[label] = {"record": exps[label]["record"], "arm": exps[label]["arm"],
                         "apply_rc": re.findall(r"rc=(-?\d+)", apply_txt)[-1:] or None,
                         "rcs": {f"{k[0]}:{k[1]}": v for k, v in rcs.items()}, "kernels": rows}
        changed = sum(1 for r in rows if not r["same_order"])
        def fmt(s):
            if not s:
                return "-"
            m = s["meta"]
            return (f"v{m.get('vgpr_count')}/a{m.get('agpr_count')}/s{m.get('sgpr_count')} spill{m.get('vgpr_spill_count')} "
                    f"scr{m.get('private_segment_fixed_size')} n{s['insts']} wait{s['s_waitcnt']} nop{s['s_nop']} "
                    f"prio{s['s_setprio']} acc{s['v_accvgpr']}")
        print(f"\n{label:18s} {exps[label]['record']:32s} rc={report[label]['rcs']} kernels={len(rows)} order_changed={changed}")
        for r in rows[:4]:
            print(f"   {r['shape']:4s} {r['kernel'][:38]:38s} stock[{fmt(r['stock'])}]")
            print(f"   {'':4s} {'':38s} cand [{fmt(r['cand'])}] same_order={r['same_order']} same_multiset={r['same_multiset']}")
    (HERE / "results" / job / "isa_stats.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["pilot"]))
