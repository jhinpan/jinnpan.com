# dsv41-amd-main e2e824dc58 + AITER acf8fdf9 (no Docker patches, no EP4 CSV) on MI355X GPUs 4-7.
export CAMPAIGN=$E2E_CAMPAIGN
: "${DSV41_GPU_LOCK_RECORD:?an active GPU lease is required}"
: "${BS1_SGLANG_ROOT:?runtime SGLang tree (sources/rt-*)}"
export PATH=$CAMPAIGN_CACHE/env/bin:$PATH
export PYTHONPATH="${BS1_AITER_ROOT:-$CAMPAIGN/sources/aiter-acf8fdf9-e2e}:$BS1_SGLANG_ROOT/python:$CAMPAIGN_CACHE/sources/sglang-kernel-build/python:$MORI/python"
export HIP_VISIBLE_DEVICES=4,5,6,7
export CUDA_VISIBLE_DEVICES=4,5,6,7
unset ROCR_VISIBLE_DEVICES

# Caches private to this contract, so nothing compiled against the previous AITER pin is reused.
export XDG_CACHE_HOME=$CAMPAIGN/cache
export SGLANG_CACHE_DIR=$CAMPAIGN/cache/sglang
export SGLANG_JIT_CACHE_DIR=$CAMPAIGN/cache/sglang/jit
export TRITON_CACHE_DIR=$CAMPAIGN/cache/triton
export TORCH_EXTENSIONS_DIR=$CAMPAIGN/cache/torch_extensions
export AITER_JIT_DIR=${DSV41_AITER_JIT_DIR:-$CAMPAIGN/cache/aiter}  # one JIT cache per AITER tree
export PYTHONPYCACHEPREFIX=$CAMPAIGN/cache/pycache
export TMPDIR=$CAMPAIGN/cache/tmp
mkdir -p "$TMPDIR"
export HF_HOME=$CAMPAIGN_CACHE/cache/huggingface
export HF_HUB_CACHE=$CAMPAIGN_CACHE/cache/huggingface/hub
export HF_HUB_OFFLINE=1

# docs/src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx, MI350X cells.
export SGLANG_USE_AITER=1
export SGLANG_MOE_PADDING=1
export AITER_FLYDSL_FORCE_REDUCE=1
export ROCM_QUICK_REDUCE_QUANTIZATION=NONE
# sgl-project/sglang#39857 serving environment.
export TRITON_HIP_USE_ASYNC_COPY=0
export SGLANG_USE_ROCM700A=0
export AITER_BF16_FP8_MOE_BOUND=0
# docker/rocm.Dockerfile runtime environment.
export HIP_FORCE_DEV_KERNARG=1
export SGLANG_DISABLE_CUDNN_CHECK=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export AITER_USE_SYSTEM_TRITON=1
export AMDGPU_TARGET=gfx950
export GPU_ARCHS=gfx950
export PYTORCH_ROCM_ARCH=gfx950

unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD SGLANG_SIMULATE_ACC_TOKEN_MODE
unset SGLANG_USE_1STAGE_ALLREDUCE SGLANG_ENABLE_DETERMINISTIC_INFERENCE
unset AITER_CUSTOM_AR_MAX_SIZE AITER_CUSTOM_AR_MIN_SIZE AITER_CONFIG_FMOE AITER_CONFIG_GEMM_BF16
