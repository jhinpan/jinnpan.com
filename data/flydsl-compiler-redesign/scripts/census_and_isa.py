"""Reproduce the kernel-corpus census, final-ISA statistics, and origin-IR op histogram.

Usage: python census_and_isa.py <FlyDSL checkout> <dump root with */17_final_isa.s> <dump dir with 00_origin.mlir>
"""

import collections
import glob
import re
import subprocess
import sys

repo, isa_root, ir_dir = sys.argv[1:4]

PATTERNS = {
    "layout_api": r"\b(make_layout|logical_divide|zipped_divide|tiled_divide|flat_divide|make_tiled_copy|make_tiled_mma|"
    r"partition_S|partition_D|partition_A|partition_B|partition_C|make_buffer_tensor|composition|crd2idx|"
    r"make_fragment_like|thr_slice|get_slice|make_identity_layout|logical_product|raked_product|blocked_product)\b",
    "raw_buffer_ops": r"\b(buffer_ops\.|create_buffer_resource|buffer_load\(|buffer_store\(|raw_buffer_load|"
    r"raw_buffer_store|raw_ptr_buffer)",
    "sched_barrier": r"\bsched_barrier\(",
    "sched_group": r"\b(sched_group_barrier|sched_mfma|sched_dsrd|sched_dswr|sched_vmem|sched_valu|sched_salu|iglp_opt)\(",
    "manual_wait": r"\b(s_waitcnt|s_wait_\w*cnt|waitcnt\w*|s_wait_tensorcnt|wait_dscnt|wait_vmcnt|wait_loadcnt)\b",
    "s_setprio": r"\bs_setprio\b",
    "inline_asm": r"\b(inline_asm|InlineAsmOp|asm_volatile|llvm\.inline_asm)\b",
    "readfirstlane": r"\breadfirstlane\b",
    "range_constexpr": r"\brange_constexpr\(",
    "imports_flydsl._mlir": r"flydsl\._mlir",
}

files = [
    f
    for f in subprocess.check_output(["git", "-C", repo, "ls-files", "kernels"]).decode().split()
    if f.endswith(".py")
]
occ, nfiles = collections.Counter(), collections.Counter()
for f in files:
    s = open(f"{repo}/{f}", encoding="utf-8", errors="ignore").read()
    for k, p in PATTERNS.items():
        n = len(re.findall(p, s))
        if n:
            occ[k] += n
            nfiles[k] += 1
print(f"== kernel corpus census: {len(files)} .py files under kernels/")
for k in PATTERNS:
    print(f"{k:22s} occurrences={occ[k]:5d} files={nfiles[k]:4d}")

print("\n== final ISA statistics (gfx950, compile-only)")
union = set()
for path in sorted(glob.glob(f"{isa_root}/*/*/17_final_isa.s")):
    text = open(path).read()
    ops = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith((".", ";", "//")) or s.endswith(":"):
            continue
        op = s.split()[0]
        if re.match(r"^[a-z][a-z0-9_]+$", op) and "_" in op:
            ops.append(op)
    cnt = collections.Counter(ops)
    union |= set(cnt)

    def grp(p):
        return sum(v for k, v in cnt.items() if re.match(p, k))

    meta = dict(re.findall(r"^\s+\.(vgpr_count|agpr_count|vgpr_spill_count|private_segment_fixed_size):\s+(\d+)", text, re.M))
    print(
        f"{path.split('/')[-3]:18s} insts={len(ops):5d} distinct={len(cnt):3d} mfma={grp(r'v_(mfma|smfmac|wmma)'):5d} "
        f"s_nop={grp(r's_nop'):4d} s_waitcnt={grp(r's_waitcnt'):4d} v_accvgpr_*={grp(r'v_accvgpr_'):4d} meta={meta}"
    )
print(f"union of distinct mnemonics: {len(union)}")

print("\n== origin IR op histogram")
s = open(glob.glob(f"{ir_dir}/*/00_origin.mlir")[0]).read()
ops = collections.Counter(re.findall(r'^\s*(?:%\S+\s*=\s*)?"?([a-z_]+\.[a-z_.0-9]+)"?[ (<{]', s, re.M))
print(f"size={len(s) / 1e6:.2f} MB ops~{sum(ops.values())}")
dialects = collections.Counter()
for k, v in ops.items():
    dialects[k.split(".")[0]] += v
print("by dialect:", dict(dialects.most_common()))
for k, v in ops.most_common(12):
    print(f"{v:7d} {k}")
