---
title: "NeMo-RL vs slime: RL Training Framework Comparison"
description: "Deep comparison of two RL training frameworks: algorithms, engineering quality, MoE support, and ROCm compatibility"
date: 2025-11-10
tags: ["RL", "training", "framework"]
category: "Technical"
lang: "en"
---

This post compares two RL training frameworks for LLMs: NVIDIA's NeMo-RL and the community-driven slime. I cover algorithm support, engineering quality, MoE readiness, and ROCm compatibility, then give selection recommendations.

## 1. Background

### 1.1 Why We Need RL Training Frameworks

SFT training works fine with HuggingFace Transformers or DeepSpeed alone. But RL training (RLHF/GRPO/PPO) involves coordinating multiple roles:

- **Actor**: generates rollouts (needs inference capability)
- **Critic / Reward Model**: scores outputs (another model or rule-based)
- **Reference Model**: computes KL constraint (prevents drifting too far)
- **Trainer**: updates actor parameters based on rewards and KL

The scheduling, communication, and memory management of these roles is far more complex than SFT, requiring dedicated frameworks.

### 1.2 Candidate Frameworks

| Framework | Maintainer | First Release | Stars | Core Positioning |
|-----------|-----------|--------------|-------|-----------------|
| NeMo-RL | NVIDIA | 2024-Q3 | ~2.5K | Production-grade RL training, deep NeMo/Megatron integration |
| slime | Community (ByteDance & academia) | 2024-Q4 | ~1.8K | Lightweight, flexible RL training, research-friendly |
| OpenRLHF | Community | 2023-Q2 | ~5K | Early framework, PPO/DPO |
| TRL | HuggingFace | 2022-Q4 | ~10K | Entry-level, Transformers ecosystem |

This post focuses on NeMo-RL and slime, as they represent the best current options for engineering quality and MoE support.

## 2. Feature Matrix

| Feature | NeMo-RL | slime |
|---------|---------|-------|
| PPO | Complete (GAE, clipping) | Complete |
| GRPO | Supported | Supported |
| DPO / SimPO | Supported | Supported |
| REINFORCE (w/ baseline) | Supported | Supported |
| Custom reward function | Yes, via config | Yes, via Python callable |
| Rule-based reward (code exec, math verify) | Built-in sandbox | Built-in + external API |
| Online RL (generate + train) | Supported | Supported |
| Offline RL (pre-generated rollouts) | Supported | Supported |
| Multi-turn RL | Limited | Full (conversation tree) |

> **Key difference:** NeMo-RL's algorithm implementations are more mature (validated at scale internally at NVIDIA), but less customizable (config-driven). slime is more research-friendly (transparent code, easy to fork and modify), but less validated at very large scale (1000+ GPUs).

## 3. Architecture Comparison

### 3.1 NeMo-RL

```
┌─────────────────────────────────────────┐
│              NeMo-RL Controller          │
│  (Hydra config → DAG of tasks)          │
├─────────────────────────────────────────┤
│                                         │
│  ┌─────────┐  ┌─────────┐  ┌────────┐  │
│  │ Actor   │  │ Critic  │  │ Ref    │  │
│  │(Megatron│  │(Megatron│  │(vLLM / │  │
│  │ + vLLM) │  │  Core)  │  │ static)│  │
│  └─────────┘  └─────────┘  └────────┘  │
│        ↕ NCCL / Gloo ↕                  │
│  ┌──────────────────────────────────┐   │
│  │     Megatron-Core Distributed    │   │
│  │     (TP/PP/DP/EP sharding)       │   │
│  └──────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

Characteristics:
- Built on Megatron-Core for distribution, native TP/PP/EP support
- Actor inference (rollout generation) uses vLLM engine
- Training updates use Megatron's optimizer
- Config-driven (Hydra YAML); changing algorithms means modifying config, not code

### 3.2 slime

```
┌─────────────────────────────────────────┐
│           slime Orchestrator            │
│  (Python script → Ray actors)           │
├─────────────────────────────────────────┤
│                                         │
│  ┌──────────┐  ┌──────────┐             │
│  │ Rollout  │  │ Trainer  │             │
│  │ Workers  │  │ Workers  │             │
│  │ (SGLang /│  │(DeepSpeed│             │
│  │  vLLM)   │  │  ZeRO)   │             │
│  └──────────┘  └──────────┘             │
│        ↕ Ray Object Store ↕             │
│  ┌──────────────────────────────────┐   │
│  │       Ray Cluster + NCCL         │   │
│  └──────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

Characteristics:
- Ray-based scheduling; Rollout and Training are decoupled into independent Ray actor groups
- Rollout workers can use SGLang or vLLM (swappable engines)
- Training workers use DeepSpeed ZeRO (simple but sufficient)
- Code-driven (Python scripts); modify algorithms by editing code directly

## 4. MoE Support

This is particularly important for our project since our target models (Kimi-K2.5, Qwen3-Coder-Next) are all MoE.

| Dimension | NeMo-RL | slime |
|-----------|---------|-------|
| MoE training | Native (Megatron-Core MoE) | Via DeepSpeed MoE |
| Expert Parallelism | Native EP | Relies on DeepSpeed EP |
| MoE + TP + PP | Full combination | EP + TP available, PP limited |
| Expert load balancing loss | Built-in | Manual addition needed |
| Token drop policy | Configurable (capacity factor) | Manual implementation needed |
| MoE rollout inference | vLLM MoE | SGLang MoE |

> **NeMo-RL is more mature for MoE.** Megatron-Core's MoE implementation is extensively validated; EP + TP + PP three-dimensional parallelism works out of the box. slime's DeepSpeed MoE support is newer, and reliability at large scale (384 experts, TP=8, EP=8) needs your own validation.

## 5. ROCm Compatibility

This is another critical dimension. Our hardware is MI300X/MI355X running ROCm 7.0.

| Dimension | NeMo-RL | slime |
|-----------|---------|-------|
| Official ROCm support | No (NVIDIA framework) | Partial (community PRs) |
| Dependency ROCm compatibility | Megatron-Core: has forks | DeepSpeed: official support |
| NCCL vs RCCL | NCCL only | Configurable for RCCL |
| Triton kernels | CUDA Triton | Can swap Triton-ROCm |
| FlashAttention | CUDA FA2 | Can swap CK FA |
| Practical usability | Needs extensive patching | Moderate patching |

> **Neither framework works out-of-the-box on ROCm.** But slime is easier to port because: (1) shorter dependency chain (Ray + DeepSpeed vs the entire Megatron-Core suite); (2) DeepSpeed officially supports ROCm; (3) the inference engine can be SGLang (good ROCm support). NeMo-RL's deep dependency on Megatron-Core makes ROCm adaptation significantly more work.

## 6. DX and Reproducibility

| Dimension | NeMo-RL | slime |
|-----------|---------|-------|
| Installation | `pip install nemo-rl` (but Megatron-Core needs separate install) | `pip install slime-rl` |
| Minimal runnable example | ~50 lines YAML | ~30 lines Python |
| Documentation | Good (NVIDIA-style, complete but dense) | Moderate (README + examples) |
| Wandb/TensorBoard | Built-in | Built-in |
| Checkpoint format | Megatron format (needs conversion) | HuggingFace format (use directly) |
| Paper reproduction | Has benchmark suite | Has recipe scripts |
| Debug friendliness | Hard (many Megatron layers) | Good (simple code) |

> **slime has better DX.** HuggingFace-format checkpoints mean no format conversion needed to interface with inference engines. NeMo-RL's Megatron checkpoints need conversion to HF format for SGLang/vLLM inference, which is sometimes painful (especially for MoE models).

## 7. Migration Plan

Starting from scratch, here's a selection flowchart:

![RL framework decision tree: choose based on hardware, MoE model, scale, and PP requirements](/blog/diagrams/nemo-rl-vs-slime-decision-tree-en.svg)

For our scenario (AMD MI300X/MI355X, MoE models, research iteration focus):

1. **Short-term (1-2 months)**: Use slime + SGLang + DeepSpeed on ROCm
2. **Mid-term (3-6 months)**: Contribute ROCm + CK optimization patches to slime, establish benchmark baselines
3. **Long-term**: If needing very large scale (1000+ GPU), evaluate feasibility of a NeMo-RL ROCm fork

## 8. Verdict

| Dimension | Winner | Reason |
|-----------|--------|--------|
| Algorithm maturity | NeMo-RL | Large-scale validation inside NVIDIA |
| MoE support | NeMo-RL | Megatron-Core MoE is more complete |
| ROCm compatibility | slime | Shorter dependency chain, DeepSpeed official ROCm support |
| Developer experience | slime | HF checkpoints, Python-driven, debug-friendly |
| Customizability | slime | Transparent code, easy to fork |
| Large-scale reliability | NeMo-RL | Validated on 1000+ GPUs |

No absolute winner. NeMo-RL suits production-grade large-scale training on NVIDIA hardware; slime suits research iteration and AMD hardware. For us, slime is the right choice today, but thorough validation on MoE + ROCm scenarios is needed.
