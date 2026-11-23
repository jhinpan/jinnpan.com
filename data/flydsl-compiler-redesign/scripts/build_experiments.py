"""Build the overlay experiment for the FlyDSL compiler redesign post.

Two families of arms, each a patch against FlyDSL cb770e9a (FlyBench v1's pin):

- ablation (x*): remove one explicit decision the kernel author made (schedule hints, wave priority,
  AGPR pinning, occupancy, a disabled LLVM pass) and let LLVM's default heuristics decide instead.
- headroom (h*): kernels whose authors made no such decision; try one LLVM scheduling knob nobody tuned.

Every arm is semantics-preserving by construction, so FlyBench's correctness gate is a sanity check,
not a filter. Outputs: patches/, attempts/, scripts/eval-<label>.sh, scripts/pilot.sh.
"""

import base64
import hashlib
import json
import lzma
import subprocess
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
V042 = HERE.parent / "FlyDSL-v042"
FBT_ROOT = Path("$AMDPILOT_DEPLOY/flybench-tasks")
sys.path.insert(0, str(FBT_ROOT))
BASE_TREE = "e84519537d54c3c3f977e86f47c45e16d1eb707a"
RUN = "flydsl-redesign-exp-20261122"

SCHED = ["sched_barrier", "sched_group_barrier", "sched_mfma", "sched_vmem", "sched_dsrd", "sched_dswr"]

# arm -> (rocdl functions made no-ops, llvm_options keys dropped, llvm_options added, source edits)
ARMS = {
    "nosched": (SCHED, [], {}, []),
    "nogroup": ([n for n in SCHED if n != "sched_barrier"], [], {}, []),
    "noprio": (["s_setprio"], [], {}, []),
    "postmisched_on": ([], ["enable-post-misched"], {}, []),
    "misched_off": ([], [], {"enable-misched": False, "enable-post-misched": False}, []),
    "strat_maxilp": ([], [], {"amdgpu-sched-strategy": "max-ilp"}, []),
    "strat_maxmem": ([], [], {"amdgpu-sched-strategy": "max-memory-clause"}, []),
    "null": ([], [], {}, []),
    "agpr_llvm": ([], [], {}, [("kernels/gemm/fp8_gemm_4wave.py",
                                "        mfma = Mfma16x16x128AGPR(N_TILES_A, N_TILES_B)\n",
                                "        mfma = Mfma16x16x128(N_TILES_A, N_TILES_B)\n")]),
    "occ_default": ([], [], {}, [("kernels/gemm/fp8_gemm_4wave.py",
                                  '            value_attrs={"rocdl.waves_per_eu": 1, "rocdl.flat_work_group_size": "256,256"},\n',
                                  "")]),
}

EXPERIMENTS = [
    # ablation: the author decided, now LLVM decides
    ("x01", "gemm.preshuffle-a8@A", "nosched"),
    ("x02", "gemm.a16w16@A", "nosched"),
    ("x03", "gemm.mx-preshuffle-a8w8@A", "nosched"),
    ("x04", "attention.flash-fwd-dense@A", "nosched"),
    ("x05", "attention.swa-gqa@A", "nosched"),
    ("x06", "gemm.fp8-rowscale-8wave@A", "noprio"),
    ("x07", "gemm.mxfp8-8wave@A", "noprio"),
    ("x08", "attention.flash-fwd-dense@A", "noprio"),
    ("x09", "attention.flash-fwd-dense@A", "postmisched_on"),
    ("x10", "gemm.fp8-rowscale-4wave@A", "agpr_llvm"),
    ("x11", "gemm.fp8-rowscale-4wave@A", "occ_default"),
    ("x12", "norm.rmsnorm-fwd@A", "nosched"),  # null control: no hints to remove
    ("x13", "attention.flash-fwd-dense@A", "nogroup"),  # keep full fences, drop interleave groups
    ("x14", "gemm.preshuffle-a8@A", "nogroup"),
    ("x02b", "gemm.a16w16@A", "nosched"),  # replicate: is the wrong output deterministic?
    ("x15", "gemm.a16w16@A", "null"),  # null control on the same kernel: are its tests stable?
    # headroom: no author decision, one untuned LLVM knob
    ("h01", "norm.rmsnorm-fwd@A", "strat_maxmem"),
    ("h02", "norm.rmsnorm-fwd@A", "strat_maxilp"),
    ("h03", "norm.layernorm-fwd@A", "strat_maxmem"),
    ("h04", "norm.layernorm-fwd@A", "strat_maxilp"),
    ("h05", "norm.softmax-fwd@A", "strat_maxmem"),
    ("h06", "norm.softmax-fwd@A", "strat_maxilp"),
    ("h07", "moe.reduce@A", "strat_maxmem"),
    ("h08", "moe.2stage-gemm1@A", "strat_maxilp"),
    ("h08b", "moe.2stage-gemm1@A", "strat_maxilp"),  # replicate: was the wrong output deterministic?
    ("h09", "moe.2stage-gemm1@A", "strat_maxmem"),
    ("h10", "gemm.preshuffle-a8@A", "strat_maxilp"),
    ("h11", "gemm.preshuffle-a8@A", "misched_off"),
    ("h12", "norm.rmsnorm-fwd@A", "misched_off"),
]

SHIM = '''

# --- experiment arm "{arm}" (flydsl-redesign overlay experiment); not a kernel change ---
def _experiment_arm():
    import flydsl.expr.rocdl as _rocdl
    from flydsl.compiler import jit_function as _jf

    def _noop(*args, **kwargs):
        return None

    for _name in {noop!r}:
        if hasattr(_rocdl, _name):
            setattr(_rocdl, _name, _noop)
    _drop, _add = {drop!r}, {add!r}
    if _drop or _add:
        _orig = _jf.JitFunction._effective_compile_hints

        def _effective(self):
            hints = dict(_orig(self) or {{}})
            opts = dict(hints.get("llvm_options") or {{}})
            for _key in _drop:
                opts.pop(_key, None)
            opts.update(_add)
            if opts:
                hints["llvm_options"] = opts
            else:
                hints.pop("llvm_options", None)
            return hints

        _jf.JitFunction._effective_compile_hints = _effective


_experiment_arm()
'''


def git(*args, cwd=V042):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def make_patch(label, arm):
    noop, drop, add, edits = ARMS[arm]
    git("checkout", "--", ".")
    if noop or drop or add or arm == "null":
        init = V042 / "kernels/__init__.py"
        init.write_text(init.read_text() + SHIM.format(arm=arm, noop=noop, drop=drop, add=add))
    for rel, old, new in edits:
        p = V042 / rel
        text = p.read_text()
        assert text.count(old) == 1, (label, rel, "edit anchor must match exactly once")
        p.write_text(text.replace(old, new))
    diff = git("diff")
    git("checkout", "--", ".")
    assert diff.strip(), label
    out = HERE / "patches" / f"{label}.patch"
    out.write_text(diff)
    return out


def packed(raw: bytes) -> str:
    return "\n".join(textwrap.wrap(base64.b64encode(lzma.compress(raw, preset=9)).decode(), 76))


ISA_STEP = r'''
echo "[isa] {job} start $(date -u +%FT%TZ)"
python - "${{OUT}}/{job}/recovery.patch" {record} "${{SCR}}/isa-{job}" "${{OUT}}/{job}/isa" "${{FLYDSL}}" <<'PY'
import json, os, subprocess, sys, shutil, tarfile
from pathlib import Path
from fbt import runner
from fbt.catalog import load
patch, record_id, scratch, out, flydsl = sys.argv[1:6]
scratch, out = Path(scratch), Path(out)
out.mkdir(parents=True, exist_ok=True)
rec = load().by_id()[record_id]
trees = runner.stage_trees(rec, runner.Setup(flydsl=Path(flydsl), scratch=scratch))
applied = subprocess.run(["git", "-C", str(trees.candidate), "apply", patch], capture_output=True, text=True)
(out / "apply.txt").write_text(applied.stdout + applied.stderr + f"\nrc={{applied.returncode}}\n")
shapes = {{s.shape_id: s.params for s in rec.visible_shapes}}
timed = [s for s in rec.timed_shapes if s in shapes]
def _size(sid):
    n = 1
    for v in shapes[sid].values():
        if isinstance(v, int) and not isinstance(v, bool):
            n *= max(v, 1)
    return n
largest = max(timed, key=_size)
picked = list(dict.fromkeys([timed[0], largest if largest != timed[0] else timed[-1]]))
DUMP = """
import json, sys, torch
from pathlib import Path
from fbt.worker import load_task
task = load_task(Path(sys.argv[1]))
shape, dist = json.loads(sys.argv[2]), sys.argv[3]
inputs = task.make_inputs(shape, dist, 0, "cuda")
outputs = {{n: torch.empty(d, dtype=t, device="cuda") for n, (d, t) in task.output_specs(shape).items()}}
launch = task.prepare(shape, inputs, outputs)
task.run(launch, inputs, outputs, shape)
torch.cuda.synchronize()
print("ok")
"""
adapter = Path("/opt/flybench/flybench-tasks/data") / rec.interface["adapter"]
for side, tree in (("stock", trees.stock), ("cand", trees.candidate)):
    for sid in picked:
        d = out / side / sid
        d.mkdir(parents=True, exist_ok=True)
        env = {{**os.environ, "FLYDSL_DEBUG_DUMP_ASM": "1", "FLYDSL_DUMP_DIR": str(d / "dump"),
               "FLYDSL_RUNTIME_ENABLE_CACHE": "0",
               "PYTHONPATH": f"{{tree}}:/opt/flybench/flybench-tasks"}}
        r = subprocess.run([sys.executable, "-c", DUMP, str(adapter), json.dumps({{"id": sid, **shapes[sid]}}),
                            rec.distributions[0]], env=env, capture_output=True, text=True, timeout=1200)
        (d / "log.txt").write_text(r.stdout[-4000:] + "\n" + r.stderr[-12000:] + f"\nrc={{r.returncode}}\n")
with tarfile.open(out.parent / "zz-isa.tar.gz", "w:gz") as tar:
    tar.add(out, arcname="isa")
shutil.rmtree(out / "stock", ignore_errors=True)
shutil.rmtree(out / "cand", ignore_errors=True)
shutil.rmtree(scratch, ignore_errors=True)
PY
echo "[isa] {job} done $(date -u +%FT%TZ)"
'''


def eval_script(label, record, patch):
    from fbt import bundle

    attempts = [{"record": record, "agent_job": label, "patch": str(patch), "base_tree": BASE_TREE}]
    (HERE / "attempts" / f"{label}.json").write_text(json.dumps(attempts, indent=1))
    text = bundle.evaluator_script(attempts, run=RUN, timeout_s=2700)
    tail = 'echo "[eval] done $(date -u +%FT%TZ)"\n'
    assert text.endswith(tail)
    text = text[: -len(tail)] + ISA_STEP.format(job=label, record=record) + tail
    assert len(text.encode()) < bundle.MAX_ARGUMENT_BYTES
    out = HERE / "scripts" / f"eval-{label}.sh"
    out.write_text(text)
    return out


def pilot_script(items):
    """One job: every arm's ISA against stock at two timed shapes, no timing. The positive control."""
    from fbt import bundle

    expected = {record: BASE_TREE for _label, record, _arm, _patch in items}
    lines = [bundle.EVALUATOR_HEAD.format(run=RUN + "-pilot", expected=json.dumps(expected, sort_keys=True))]
    for label, record, _arm, patch in items:
        raw = patch.read_bytes()
        lines += [f'mkdir -p "${{OUT}}/{label}"',
                  f"base64 -d <<'PATCH_{label}' | xz -d > \"${{OUT}}/{label}/recovery.patch\"",
                  packed(raw), f"PATCH_{label}",
                  f'echo "{hashlib.sha256(raw).hexdigest()}  ${{OUT}}/{label}/recovery.patch" | sha256sum -c --quiet - || exit 3']
        lines.append(ISA_STEP.format(job=label, record=record))
    lines.append('echo "[pilot] done $(date -u +%FT%TZ)"')
    text = "\n".join(lines) + "\n"
    assert len(text.encode()) < 128 * 1024, len(text.encode())
    out = HERE / "scripts" / "pilot.sh"
    out.write_text(text)
    return out


def main():
    for d in ("patches", "attempts", "scripts"):
        (HERE / d).mkdir(exist_ok=True)
    assert not git("status", "--porcelain").strip(), "v0.4.2 worktree must be clean"
    items = []
    for exp_id, record, arm in EXPERIMENTS:
        label = f"{exp_id}-{arm.replace('_', '')}"
        patch = make_patch(label, arm)
        items.append((label, record, arm, patch))
        eval_script(label, record, patch)
    pilot_script(items)
    (HERE / "experiments.json").write_text(json.dumps(
        [{"label": l, "record": r, "arm": a, "patch": str(p)} for l, r, a, p in items], indent=1))
    print(f"{len(items)} arms; scripts in {HERE / 'scripts'}")


if __name__ == "__main__":
    sys.exit(main())
