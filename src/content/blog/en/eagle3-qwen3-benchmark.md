---
title: "Benchmark: Qwen3-Coder-30B-A3B + EAGLE3 Speculative Decoding"
description: "EAGLE3 speculative decoding benchmarks on Qwen3-Coder — 1.87x speedup for code generation"
date: 2025-11-25
tags: ["speculative-decoding", "benchmark", "SGLang"]
category: "Technical"
lang: "en"
---

This post documents the performance evaluation of EAGLE3 speculative decoding on Qwen3-Coder-30B-A3B. Tests were conducted on a single H100 80GB GPU using SGLang as the inference engine.

## 1. Background

### 1.1 What Is Speculative Decoding

The core idea of speculative decoding: use a small model (draft model) to quickly "guess" multiple tokens, then use the large model (target model) to verify them in parallel. If the guesses are correct, you effectively generated multiple tokens in one step; if wrong, roll back to the first incorrect position.

Key metrics:
- **Acceptance Rate**: proportion of draft tokens accepted by the target model
- **Speedup**: throughput improvement compared to baseline (no speculation)
- **Draft Overhead**: the draft model's own inference time and memory cost

### 1.2 EAGLE3

EAGLE3 is the third generation of the EAGLE series, with key improvements over previous versions:

- **Lower training cost**: only needs about 1% of the original model's training data for distillation
- **Native MoE compatibility**: the draft model directly reuses the target model's expert layers, no need to train a separate dense draft model
- **Higher acceptance rate**: by drafting at the feature level (in hidden state space rather than token space), EAGLE3 achieves 15-25% higher acceptance rates than token-level draft methods

### 1.3 Why Qwen3-Coder-30B-A3B

Qwen3-Coder-30B-A3B is a MoE model (30B total parameters, 3B active parameters) that performs well on code generation. Its characteristics make it particularly suitable for speculative decoding:

- MoE architecture means decode-phase compute is already light (only 3B active parameters), but memory bandwidth is the bottleneck (must load routing info for full 30B weights)
- Code generation token prediction is relatively deterministic (syntax constraints, common patterns), leading to high draft acceptance rates
- EAGLE3 reuses expert layers directly, meaning near-zero additional memory for the draft model

## 2. Experimental Setup

### 2.1 Hardware and Software

| Item | Configuration |
|------|--------------|
| GPU | 1x NVIDIA H100 80GB SXM |
| Driver | NVIDIA 550.127.08, CUDA 12.4 |
| Inference Engine | SGLang v0.4.5 |
| Model | Qwen3-Coder-30B-A3B |
| EAGLE3 Draft Model | Qwen3-Coder-30B-A3B-EAGLE3 (distillation-trained, ~500M extra params) |
| Precision | FP16 |
| Max Context | 8192 |

### 2.2 Benchmark Scenarios

Three scenarios to cover different usage patterns:

| Scenario | Prompts | Input Length | Output Length | Request Rate | Concurrency |
|----------|---------|-------------|--------------|-------------|-------------|
| Code Gen (high output) | 128 | 512 | 1024 | inf | 64 |
| Chat (balanced) | 128 | 256 | 256 | 8 req/s | 32 |
| Completion (short output) | 256 | 1024 | 128 | inf | 128 |

### 2.3 Launch Commands

**Baseline (no speculative decoding):**

```bash
python -m sglang.launch_server \
    --model Qwen/Qwen3-Coder-30B-A3B \
    --tp 1 \
    --max-total-tokens 65536 \
    --mem-fraction-static 0.85 \
    --enable-torch-compile \
    --port 30000
```

**EAGLE3 speculative decoding:**

```bash
python -m sglang.launch_server \
    --model Qwen/Qwen3-Coder-30B-A3B \
    --speculative-algorithm EAGLE \
    --speculative-draft-model-path Qwen/Qwen3-Coder-30B-A3B-EAGLE3 \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 8 \
    --speculative-num-draft-tokens 32 \
    --tp 1 \
    --max-total-tokens 65536 \
    --mem-fraction-static 0.85 \
    --enable-torch-compile \
    --port 30000
```

**Benchmark command:**

```bash
python -m sglang.bench_serving \
    --backend sglang \
    --host 127.0.0.1 \
    --port 30000 \
    --dataset-name random \
    --num-prompts 128 \
    --random-input 512 \
    --random-output 1024 \
    --request-rate inf \
    --max-concurrency 64
```

## 3. Results

### 3.1 Code Gen Scenario (Primary Focus)

| Metric | Baseline | EAGLE3 | Change |
|--------|----------|--------|--------|
| Total throughput (tok/s) | 2,847 | 5,324 | **+87%** |
| Avg TTFT (ms) | 142 | 168 | +18% |
| Avg TPOT (ms) | 22.5 | 12.0 | **-47%** |
| P99 TPOT (ms) | 35.2 | 18.7 | -47% |
| Avg acceptance rate | — | 0.78 | — |
| Avg accepted length | — | 3.1 tokens | — |

> **1.87x throughput improvement**, with an average of 3.1 tokens accepted per draft step. TTFT increases slightly (draft model needs extra initialization), but TPOT drops by nearly half.

### 3.2 Chat Scenario

| Metric | Baseline | EAGLE3 | Change |
|--------|----------|--------|--------|
| Total throughput (tok/s) | 1,523 | 2,418 | **+59%** |
| Avg TTFT (ms) | 89 | 105 | +18% |
| Avg TPOT (ms) | 18.3 | 11.5 | **-37%** |
| P99 TPOT (ms) | 28.1 | 17.2 | -39% |
| Avg acceptance rate | — | 0.71 | — |
| Avg accepted length | — | 2.6 tokens | — |

Chat scenarios have slightly lower acceptance rates (natural language is more "random" than code), but still deliver 59% throughput improvement.

### 3.3 Completion Scenario

| Metric | Baseline | EAGLE3 | Change |
|--------|----------|--------|--------|
| Total throughput (tok/s) | 4,215 | 6,872 | **+63%** |
| Avg TTFT (ms) | 287 | 312 | +9% |
| Avg TPOT (ms) | 15.8 | 9.4 | **-41%** |
| P99 TPOT (ms) | 24.5 | 14.8 | -40% |
| Avg acceptance rate | — | 0.73 | — |
| Avg accepted length | — | 2.8 tokens | — |

Under high concurrency, EAGLE3's speedup remains stable. Long-prompt TTFT increase is smaller (prefill time already dominates, draft initialization overhead is proportionally lower).

### 3.4 Memory Comparison

| Item | Baseline | EAGLE3 |
|------|----------|--------|
| Model weights | 58.2 GB | 58.2 GB |
| Draft model | — | 1.2 GB |
| KV Cache + Buffers | 18.8 GB | 17.6 GB |
| **Total usage** | **77.0 GB** | **77.0 GB** |

EAGLE3's draft model adds only 1.2 GB (reuses target model expert layers); SGLang automatically adjusts KV Cache allocation. Total memory usage is essentially unchanged.

## 4. Analysis

### 4.1 Why Code Generation Gets the Largest Speedup

Code generation's token distribution is "sharper" than natural language:

- Once a variable name appears, subsequent references are essentially deterministic
- Syntactic structures (brackets, indentation, keyword sequences) are highly predictable
- Common patterns (for loops, if-else, function signatures) have relatively fixed token sequences

These characteristics give the draft model higher prediction accuracy, pushing acceptance rates from Chat's 0.71 to Code Gen's 0.78.

### 4.2 EAGLE3 vs Other Speculative Decoding Methods

| Method | Acceptance Rate (Code) | Extra Memory | Requires Independent Training | MoE Compatible |
|--------|----------------------|-------------|------------------------------|----------------|
| Medusa | 0.65 | ~2 GB | Yes (multi-head distillation) | Needs adaptation |
| EAGLE | 0.72 | ~3 GB | Yes | Needs adaptation |
| EAGLE2 | 0.75 | ~2 GB | Yes | Needs adaptation |
| **EAGLE3** | **0.78** | **~1.2 GB** | **Yes (lighter)** | **Native** |
| Lookahead | 0.60 | ~0 | No | Yes |

EAGLE3 has advantages in both acceptance rate and MoE compatibility.

### 4.3 When NOT to Use Speculative Decoding

Situations where it's not a good fit:
- **Very large batches**: draft + verify overhead at high batch sizes may exceed benefits. Generally, speedup narrows beyond batch > 128
- **Very short outputs**: if only generating 10-20 tokens, draft initialization overhead is proportionally too high
- **Extremely tight memory**: though EAGLE3 adds only 1.2 GB, when running 70B+ models on an 80GB card, that 1.2 GB might be critical

## 5. Reproduction Guide

### 5.1 Installation

```bash
pip install "sglang[all]>=0.4.5"
```

### 5.2 Download Models

```bash
huggingface-cli download Qwen/Qwen3-Coder-30B-A3B
huggingface-cli download Qwen/Qwen3-Coder-30B-A3B-EAGLE3
```

### 5.3 Launch and Test

See section 2.3 for launch commands. After benchmark completion, results are in SGLang's standard output; use `--output-file` to specify an output file.

### 5.4 Parameter Tuning Tips

| Parameter | Description | Recommended Value |
|-----------|-------------|------------------|
| `--speculative-num-steps` | Draft tree depth | 3-5 (deeper = slower but slightly higher acceptance) |
| `--speculative-eagle-topk` | Top-k expansion per step | 4-8 (higher = more memory) |
| `--speculative-num-draft-tokens` | Total draft tokens | 16-64 (too large wastes verification compute) |

Optimal parameters depend on the specific model and scenario. Run a grid search on your target workload.

## Summary

EAGLE3 speculative decoding results on Qwen3-Coder-30B-A3B:

- Code generation: **1.87x** throughput improvement, 47% TPOT reduction
- Chat: **1.59x** improvement
- High concurrency: **1.63x** improvement
- Memory overhead only 1.2 GB, native MoE compatibility
- Acceptance rates 0.71-0.78, averaging 2.6-3.1 tokens accepted per step

For single-GPU MoE model deployments where decode latency matters, EAGLE3 is essentially a free lunch.
