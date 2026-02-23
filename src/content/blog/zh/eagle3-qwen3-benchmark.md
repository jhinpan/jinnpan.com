---
title: "Benchmark: Qwen3-Coder-30B-A3B + EAGLE3 投机解码"
description: "EAGLE3 投机解码在 Qwen3-Coder 上的性能评测，代码生成场景 1.87x 加速"
date: 2025-11-25
tags: ["speculative-decoding", "benchmark", "SGLang"]
category: "Technical"
lang: "zh"
---

这篇文章记录了 EAGLE3 投机解码在 Qwen3-Coder-30B-A3B 上的性能评测。测试在单卡 H100 80GB 上完成，使用 SGLang 作为推理引擎。

## 1. 背景

### 1.1 什么是投机解码

投机解码（Speculative Decoding）的核心思路：用一个小模型（draft model）快速"猜"多个 token，再用大模型（target model）并行验证。如果猜对了，就等于一步生成了多个 token；猜错了，回退到猜错的位置。

关键指标：
- **接受率（Acceptance Rate）**：draft model 猜中的比例
- **加速比（Speedup）**：相比不使用投机解码的 throughput 提升
- **Draft 开销**：draft model 本身的推理时间和显存

### 1.2 EAGLE3

EAGLE3 是 EAGLE 系列的第三代，相比前代的核心改进：

- **训练开销更低**：只需要原模型 1% 左右的数据量做蒸馏训练
- **与 MoE 原生兼容**：draft model 直接复用 target model 的 expert 层，不需要独立训练一个 dense draft model
- **接受率更高**：通过 feature-level 的 draft（在 hidden state 空间而非 token 空间做预测），EAGLE3 比 token-level 的 draft 方法接受率高 15-25%

### 1.3 为什么选 Qwen3-Coder-30B-A3B

Qwen3-Coder-30B-A3B 是一个 MoE 模型（30B 总参数，3B 激活参数），在代码生成任务上表现不错。它的特点对投机解码特别有利：

- MoE 架构意味着 decode 阶段计算量本身就不大（只激活 3B 参数），但访存带宽是瓶颈（要加载完整 30B 权重的路由信息）
- 代码生成的 token 预测确定性较高（语法约束、常见模式），draft model 接受率高
- EAGLE3 直接复用 expert 层，draft 模型几乎零额外显存

## 2. 实验设置

### 2.1 硬件与软件

| 项目 | 配置 |
|------|------|
| GPU | 1x NVIDIA H100 80GB SXM |
| 驱动 | NVIDIA 550.127.08, CUDA 12.4 |
| 推理引擎 | SGLang v0.4.5 |
| 模型 | Qwen3-Coder-30B-A3B |
| EAGLE3 Draft Model | Qwen3-Coder-30B-A3B-EAGLE3 (蒸馏训练, ~500M 额外参数) |
| 精度 | FP16 |
| 最大上下文 | 8192 |

### 2.2 Benchmark 场景

我们设计了三个场景来覆盖不同的使用模式：

| 场景 | 提示数 | 输入长度 | 输出长度 | 请求速率 | 并发 |
|------|--------|---------|---------|---------|------|
| Code Gen (高输出) | 128 | 512 | 1024 | inf | 64 |
| Chat (均衡) | 128 | 256 | 256 | 8 req/s | 32 |
| Completion (短输出) | 256 | 1024 | 128 | inf | 128 |

### 2.3 启动命令

**Baseline（无投机解码）：**

```bash
python -m sglang.launch_server \
    --model Qwen/Qwen3-Coder-30B-A3B \
    --tp 1 \
    --max-total-tokens 65536 \
    --mem-fraction-static 0.85 \
    --enable-torch-compile \
    --port 30000
```

**EAGLE3 投机解码：**

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

**Benchmark 命令：**

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

## 3. 结果

### 3.1 Code Gen 场景（重点场景）

| 指标 | Baseline | EAGLE3 | 变化 |
|------|----------|--------|------|
| 总吞吐 (tok/s) | 2,847 | 5,324 | **+87%** |
| 平均 TTFT (ms) | 142 | 168 | +18% |
| 平均 TPOT (ms) | 22.5 | 12.0 | **-47%** |
| P99 TPOT (ms) | 35.2 | 18.7 | -47% |
| 平均接受率 | — | 0.78 | — |
| 平均接受长度 | — | 3.1 tokens | — |

> **1.87x 吞吐提升**，每次 draft 平均接受 3.1 个 token。TTFT 略有增加（draft model 需要额外初始化），但 TPOT 降了近一半。

### 3.2 Chat 场景

| 指标 | Baseline | EAGLE3 | 变化 |
|------|----------|--------|------|
| 总吞吐 (tok/s) | 1,523 | 2,418 | **+59%** |
| 平均 TTFT (ms) | 89 | 105 | +18% |
| 平均 TPOT (ms) | 18.3 | 11.5 | **-37%** |
| P99 TPOT (ms) | 28.1 | 17.2 | -39% |
| 平均接受率 | — | 0.71 | — |
| 平均接受长度 | — | 2.6 tokens | — |

Chat 场景接受率略低（自然语言比代码更"随机"），但仍然有 59% 的吞吐提升。

### 3.3 Completion 场景

| 指标 | Baseline | EAGLE3 | 变化 |
|------|----------|--------|------|
| 总吞吐 (tok/s) | 4,215 | 6,872 | **+63%** |
| 平均 TTFT (ms) | 287 | 312 | +9% |
| 平均 TPOT (ms) | 15.8 | 9.4 | **-41%** |
| P99 TPOT (ms) | 24.5 | 14.8 | -40% |
| 平均接受率 | — | 0.73 | — |
| 平均接受长度 | — | 2.8 tokens | — |

高并发场景下，EAGLE3 的加速依然稳定。长 prompt 的 TTFT 增幅更小（prefill 时间本身就长，draft 初始化开销占比降低）。

### 3.4 显存对比

| 项目 | Baseline | EAGLE3 |
|------|----------|--------|
| 模型权重 | 58.2 GB | 58.2 GB |
| Draft 模型 | — | 1.2 GB |
| KV Cache + Buffers | 18.8 GB | 17.6 GB |
| **总占用** | **77.0 GB** | **77.0 GB** |

EAGLE3 的 draft model 只增加了 1.2 GB 显存（复用了 target model 的 expert 层），SGLang 自动调整了 KV Cache 的分配来适应。总显存占用基本不变。

## 4. 分析

### 4.1 为什么代码场景加速最大

代码生成的 token 分布比自然语言更"尖锐"：

- 变量名一旦出现，后续引用基本确定
- 语法结构（括号、缩进、关键字序列）高度可预测
- 常见 pattern（for loop、if-else、函数签名）的 token 序列相对固定

这些特性让 draft model 的预测准确率更高，接受率从 Chat 的 0.71 提升到 Code Gen 的 0.78。

### 4.2 EAGLE3 vs 其他投机解码方法

| 方法 | 接受率 (Code) | 额外显存 | 是否需要独立训练 | MoE 兼容 |
|------|-------------|---------|----------------|---------|
| Medusa | 0.65 | ~2 GB | 是 (多头蒸馏) | 需适配 |
| EAGLE | 0.72 | ~3 GB | 是 | 需适配 |
| EAGLE2 | 0.75 | ~2 GB | 是 | 需适配 |
| **EAGLE3** | **0.78** | **~1.2 GB** | **是 (但更轻量)** | **原生** |
| Lookahead | 0.60 | ~0 | 否 | 是 |

EAGLE3 在接受率和 MoE 兼容性上都有优势。

### 4.3 什么时候不该用投机解码

几种不适合的情况：
- **Batch 很大时**：draft + verify 的 overhead 在高 batch 下可能超过收益。一般 batch > 128 时加速收窄
- **输出很短时**：如果只生成 10-20 个 token，draft 初始化的 overhead 占比过大
- **显存极紧时**：虽然 EAGLE3 只加 1.2 GB，但在 80GB 卡上跑 70B+ 模型时，这 1.2 GB 可能很关键

## 5. 复现指南

### 5.1 安装

```bash
pip install "sglang[all]>=0.4.5"
```

### 5.2 下载模型

```bash
huggingface-cli download Qwen/Qwen3-Coder-30B-A3B
huggingface-cli download Qwen/Qwen3-Coder-30B-A3B-EAGLE3
```

### 5.3 启动与测试

启动命令见 2.3 节。跑完 benchmark 后，结果保存在 SGLang 的标准输出中，也可以用 `--output-file` 指定输出文件。

### 5.4 参数调优建议

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `--speculative-num-steps` | draft 树的深度 | 3-5（越大越慢但接受率略高） |
| `--speculative-eagle-topk` | 每步展开的 top-k | 4-8（越大显存越多） |
| `--speculative-num-draft-tokens` | 总 draft token 数 | 16-64（太大浪费验证计算） |

最优参数和具体模型、具体场景有关。建议在目标场景下做一轮 grid search。

## 总结

EAGLE3 在 Qwen3-Coder-30B-A3B 上的投机解码表现：

- 代码生成场景 **1.87x** 吞吐提升，TPOT 降低 47%
- Chat 场景 **1.59x** 提升
- 高并发场景 **1.63x** 提升
- 显存增量仅 1.2 GB，对 MoE 模型原生兼容
- 接受率 0.71-0.78，平均每步接受 2.6-3.1 个 token

对于需要在单卡上跑 MoE 模型且关心 decode 延迟的场景，EAGLE3 几乎是免费的午餐。
