"""Experiment-only: run DeepseekV2MoE's shared experts on the forward stream.

On CUDA, SGLang main overlaps the shared experts with the routed experts on a side stream
whenever the MoE layer holds an alt_stream, which it always does, and no environment switch
turns that off (SGLANG_OPT_USE_MULTI_STREAM_OVERLAP covers the attention, mHC-statistics,
routed-quant and draft streams only). This file reaches the scheduler processes only through
the PYTHONPATH of the serialized-stream arms; the installed package is not modified.
"""
import importlib.abc
import importlib.util
import os
import sys

_SYSTEM = "/usr/lib/python3.12/sitecustomize.py"
if os.path.exists(_SYSTEM):
    exec(compile(open(_SYSTEM).read(), _SYSTEM, "exec"))

_TARGET = "sglang.srt.models.deepseek_v2"


class _SerialSharedExperts(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != _TARGET:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        run = spec.loader.exec_module

        def exec_module(module):
            run(module)
            init = module.DeepseekV2MoE.__init__

            def __init__(self, *args, **kwargs):
                init(self, *args, **kwargs)
                self.alt_stream = None

            module.DeepseekV2MoE.__init__ = __init__
            print(f"[serial-moe] DeepseekV2MoE.alt_stream disabled in pid {os.getpid()}",
                  file=sys.stderr, flush=True)

        spec.loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, _SerialSharedExperts())
