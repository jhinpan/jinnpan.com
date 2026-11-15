# GLM-5.3 PD × monokernel on MI355X — how the runs were made

Every number in this directory comes from these commands. Host names, addresses and local paths are
replaced by placeholders (`<prefill-host>`, `<decode-host>`, `/models`).

## Cohort

| Field | Value |
|---|---|
| Hardware | 2 nodes x 8 MI355X (gfx950), bare metal; 8x AMD Pensando Pollara 400G NICs per node, RoCE v2 |
| Image | `lmsysorg/sglang-rocm:v0.5.23-rocm720-mi35x-20261105@sha256:33efb7dc35be2a099a21cd14e8d50eddaad0808b7ac71177f3aa1bdc990cfdec` (SGLang `1305b1830a`, sgl-amd-mono `e52803f`) |
| Model | `zai-org/GLM-5.3` FP8 at `ba32f8c19c8738e2c57c71798c601d477a7bf844` |
| Recipe | InferenceX `glm5.3-fp8-mi355x-sglang-agentic-mtp`: TP8, MTP (EAGLE 3 steps, top-k 1, 4 draft tokens), DSA `tilelang`, FP8 KV cache |
| Client | InferenceX `5dde68ef` (`python -m infx.bench agentic`, agentx-harness 1.0.13), `semianalysisai/cc-traces-weka-062126` |
| Window | 900 s after 10 warmup requests per lane, fresh servers for every point (local L1, not an InferenceX result) |

## Server (agg, one node; the `mono` arm adds `--enable-amd-monokernel`)

```bash
SGLANG_OPT_USE_TOPK_V2=false SGLANG_TIMEOUT_KEEP_ALIVE=900 \
SGLANG_SIMULATE_ACC_LEN=2.94 SGLANG_SIMULATE_ACC_METHOD=match-expected \
SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token \
python3 -m sglang.launch_server --model-path /models/GLM-5.3-FP8 --host 127.0.0.1 --port 31000 \
  --tensor-parallel-size 8 --expert-parallel-size 1 --kv-cache-dtype fp8_e4m3 \
  --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 --dsa-decode-backend tilelang --dsa-prefill-backend tilelang \
  --chunked-prefill-size 32768 --mem-fraction-static 0.85 --enable-metrics \
  --max-running-requests <2 x conc> --cuda-graph-max-bs-decode <2 x conc> \
  --reasoning-parser auto --tool-call-parser auto --trust-remote-code --watchdog-timeout 1800 \
  --served-model-name zai-org/GLM-5.3
```

The quality arms drop the three `SGLANG_SIMULATE_ACC_*` variables (real MTP acceptance).

## PD 1P1D over MoRI

Both roles take the agg arguments above plus:

```bash
# both roles
MORI_IB_GID_INDEX=1 NCCL_IB_DISABLE=1 SGLANG_HOST_IP=<role host>
--disaggregation-transfer-backend mori --disaggregation-ib-device rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7

# prefill (<prefill-host>, no monokernel)
--host 0.0.0.0 --port 31000 --disaggregation-mode prefill --disaggregation-bootstrap-port 8998

# decode (<decode-host>); pdr and pdk add the radix cache, pdk adds the monokernel
--host 127.0.0.1 --port 31001 --disaggregation-mode decode \
  --disaggregation-decode-enable-radix-cache --enable-amd-monokernel
```

For `pdk`, the decode container mounts `sgl_amd_mono/glm53/spec.py` and `sgl_amd_mono/glm53/draft.py` of
[sgl-project/sgl-amd-mono#136](https://github.com/sgl-project/sgl-amd-mono/pull/136) over the image's
copies (the image's versions equal `main` at `572ddf1`). The image's `libionic` provider supports kernel ABI 4
only, so the PD containers mount the host's provider when the host's `ionic_rdma` uses ABI 1.

```bash
python3 -m sglang_router.launch_router --pd-disaggregation \
  --prefill http://<prefill-host>:31000 8998 --decode http://127.0.0.1:31001 \
  --host 127.0.0.1 --port 31100 --health-check-timeout-secs 120
```

With the default 5-s health-check timeout, a prefill that runs long cold chunks can fail three checks in a row,
and the router then refuses every request ("No available prefill workers").

## Two agg nodes behind one router (same 16 GPUs as PD)

```bash
# the agg server with --enable-amd-monokernel on both nodes (<prefill-host> binds 0.0.0.0)
python3 -m sglang_router.launch_router --worker-urls http://<prefill-host>:31000 http://127.0.0.1:31000 \
  --policy cache_aware --host 127.0.0.1 --port 31100 --health-check-timeout-secs 120
```

## Client

```bash
cd InferenceX/inferencex-e2e   # 5dde68ef
DURATION=900 CONC=<n> AIPERF_EXPERIMENTAL_FAST=0 AIPERF_SERVER_URL=http://127.0.0.1:<31000 or 31100> \
AIPERF_SERVER_METRICS_URLS=<every server's /metrics> MODEL=zai-org/GLM-5.3 SCENARIO_TYPE=agentic-coding \
python -m infx.bench agentic

# quality: lm-eval GSM8K, all 1319 questions, strict-match, concurrency 2
python -m infx.bench eval --endpoint http://127.0.0.1:<port> --concurrency 2
```

`tok_s_per_gpu` in `agentx_l1.csv` divides by every GPU of the deployment (8 or 16).
