"""Stage-level compile-time breakdown for FlyDSL kernels (CPU only, COMPILE_ONLY=1)."""

import json
import os
import sys
import time

os.environ.setdefault("FLYDSL_GPU_ARCH", "gfx950")
os.environ["COMPILE_ONLY"] = "1"
os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FlyDSL-main")
sys.path.insert(0, REPO)

t_import0 = time.perf_counter()
import torch  # noqa: E402

import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
from flydsl._mlir import ir  # noqa: E402
from flydsl._mlir.passmanager import PassManager  # noqa: E402
from flydsl.compiler import jit_function as jfm  # noqa: E402

t_import = time.perf_counter() - t_import0

T = {}


def acc(name, dt):
    T[name] = T.get(name, 0.0) + dt


def wrap(obj, attr, name):
    orig = getattr(obj, attr)

    def inner(*a, **k):
        t0 = time.perf_counter()
        try:
            return orig(*a, **k)
        finally:
            acc(name, time.perf_counter() - t0)

    setattr(obj, attr, inner)


LAST_JF = []
_orig_trace = jfm.JitFunction._trace


def _trace(self, *a, **k):
    LAST_JF[:] = [self]
    t0 = time.perf_counter()
    try:
        return _orig_trace(self, *a, **k)
    finally:
        acc("trace", time.perf_counter() - t0)


jfm.JitFunction._trace = _trace
wrap(jfm, "_artifact_cache_key", "cache_key")
_orig_prepare = jfm.MlirCompiler.prepare.__func__


def _prepare(cls, module, **kw):
    t0 = time.perf_counter()
    out = _orig_prepare(cls, module, **kw)
    acc("prepare(verify+asm)", time.perf_counter() - t0)
    T["origin_ir_chars"] = len(out[0])
    return out


jfm.MlirCompiler.prepare = classmethod(_prepare)

STAGE = {}
OPS = {}


def count_ops(module):
    n = 0

    def visit(op):
        nonlocal n
        n += 1
        return ir.WalkResult.ADVANCE

    module.operation.walk(visit)
    return n


def _run_pipeline(module, fragments, *, verifier, print_after_all):
    OPS["00_origin"] = count_ops(module)
    for i, frag in enumerate(fragments):
        pm = PassManager.parse(f"builtin.module({frag})")
        pm.enable_verifier(verifier)
        t0 = time.perf_counter()
        jfm._run_pass_manager(pm, module)
        dt = time.perf_counter() - t0
        label = f"{i + 1:02d}_{jfm._stage_label_from_fragment(frag)}"
        STAGE[label] = STAGE.get(label, 0.0) + dt
        if frag.startswith(("fly-layout-lowering", "convert-fly-to-rocdl", "reconcile-unrealized-casts")):
            OPS[label] = count_ops(module)


jfm._run_pipeline = _run_pipeline
wrap(jfm.MlirCompiler, "compile", "lower+codegen(total)")


def engine_cost(artifact):
    """Host-side ExecutionEngine build + entry lookup (no GPU module load)."""
    from flydsl._mlir.execution_engine import ExecutionEngine
    from flydsl.compiler.jit_executor import _resolve_runtime_libs

    ctx = jfm._create_mlir_context()
    with ctx:
        t0 = time.perf_counter()
        mod = ir.Module.parse(artifact._ir_text)
        t1 = time.perf_counter()
        eng = ExecutionEngine(mod, opt_level=3, shared_libs=_resolve_runtime_libs())
        t2 = time.perf_counter()
        ptr = eng.raw_lookup(artifact._entry)
        t3 = time.perf_counter()
    return {
        "artifact_ir_chars": len(artifact._ir_text),
        "parse_artifact_ir": t1 - t0,
        "engine_create_O3": t2 - t1,
        "entry_lookup(codegen)": t3 - t2,
        "lookup_ok": bool(ptr),
    }


def run_case(name):
    if name == "vector_add":
        sys.path.insert(0, os.path.join(REPO, "examples"))
        import importlib.util

        spec = importlib.util.spec_from_file_location("vadd", os.path.join(REPO, "examples", "01-vectorAdd.py"))
        src = open(os.path.join(REPO, "examples", "01-vectorAdd.py")).read()
        # Execute only the definitions (stop before the module-level demo).
        cut = src.find("\nM, N = 100, 1000")
        mod = {"__name__": "vadd", "__file__": spec.origin}
        exec(compile(src[: cut if cut > 0 else None], spec.origin, "exec"), mod)
        a = torch.zeros(4096, 4096, dtype=torch.float32)
        return mod["vector_add"], (a, a.clone(), a.clone())
    if name.startswith("preshuffle_fp8"):
        from kernels.gemm.preshuffle_gemm import compile_preshuffle_gemm

        M, N, K, tm, tn, tk = 5120, 5120, 8320, 64, 256, 128
        fn = compile_preshuffle_gemm(N=N, K=K, tile_m=tm, tile_n=tn, tile_k=tk, in_dtype="fp8")
        args = (
            torch.zeros(M * N, dtype=torch.float16),
            torch.zeros(M * K, dtype=torch.int8),
            torch.zeros(N * K, dtype=torch.int8),
            torch.zeros(M, dtype=torch.float32),
            torch.zeros(N, dtype=torch.float32),
            torch.empty(0, dtype=torch.float16),
            M,
            N,
            0,
        )
        return fn, args
    if name == "fp8_gemm_4wave":
        from kernels.gemm.fp8_gemm_4wave import compile_fp8_gemm_4w

        M, N, K = 4096, 4096, 4096
        fn = compile_fp8_gemm_4w(K=K, BLOCK_M=256, BLOCK_N=256)
        args = (
            torch.zeros(M * K, dtype=torch.float8_e4m3fn),
            torch.zeros(N * K, dtype=torch.float8_e4m3fn),
            torch.zeros(M * N, dtype=torch.bfloat16),
            torch.zeros(M, dtype=torch.float32),
            torch.zeros(N, dtype=torch.float32),
            M,
            N,
            fx.Stream(None),
        )
        return fn, args
    if name == "flash_attn_gfx950":
        from kernels.attention.flash_attn_gfx950 import build_flash_attn_dualwave_swp_module

        B, S, H, D = 1, 4096, 16, 128
        fn = build_flash_attn_dualwave_swp_module(num_heads=H, head_dim=D, causal=True)
        q, k, v, o = (torch.zeros(B, S, H, D, dtype=torch.bfloat16) for _ in range(4))
        return fn, (q, k, v, o, B, S)
    raise SystemExit(f"unknown case {name}")


def main():
    name = sys.argv[1]
    fn, args = run_case(name)
    t0 = time.perf_counter()
    fn(*args)
    cold = time.perf_counter() - t0
    cold_T = dict(T)
    T.clear()
    t0 = time.perf_counter()
    fn(*args)
    warm = time.perf_counter() - t0
    warm_T = dict(T)
    art = LAST_JF[0]._last_compiled[1]
    eng = engine_cost(art)
    out = {
        "case": name,
        "import_s": round(t_import, 3),
        "cold_call_s": round(cold, 3),
        "cold_breakdown_s": {k: round(v, 4) if isinstance(v, float) else v for k, v in cold_T.items()},
        "per_pass_s": {k: round(v, 4) for k, v in STAGE.items()},
        "op_counts": OPS,
        "warm_inmem_hit_call_s": round(warm, 4),
        "warm_breakdown_s": {k: round(v, 4) if isinstance(v, float) else v for k, v in warm_T.items()},
        "host_engine": {k: round(v, 4) if isinstance(v, float) else v for k, v in eng.items()},
    }
    print("RESULT " + json.dumps(out))


if __name__ == "__main__":
    main()
