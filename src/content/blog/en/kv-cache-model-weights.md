---
title: "KV Cache & Model Weights"
description: "Understanding KV Cache vs Model Weights — the first step to LLM inference optimization"
date: 2025-06-23
tags: ["inference", "KV-cache", "LLM"]
category: "Technical"
lang: "en"
---

Understanding the distinction between KV Cache and Model Weights is the first step to grasping LLM inference optimization. This post breaks down both concepts from the training and inference perspectives.

## Core Concepts at a Glance

| Concept | Essence | Created During | Lifecycle | Size Depends On |
|---------|---------|---------------|-----------|----------------|
| Model Weights | Network parameters | Training | Long-term, stored on disk/GPU | Model architecture (layers, hidden size) |
| KV Cache | Intermediate computation cache | Inference | Short-term, created/destroyed per request | Sequence length, batch size |

## 1. Training Phase: Birth of Model Weights

### 1.1 What Are Model Weights

Model Weights are all learnable parameters in a neural network. For a Transformer model, these include:

- **Embedding layer**: token embedding $W_E \in \mathbb{R}^{V \times d}$, positional encoding parameters
- **Attention layers**: $W^Q, W^K, W^V \in \mathbb{R}^{d \times d_k}$, $W^O \in \mathbb{R}^{d \times d}$
- **FFN layers**: $W_1 \in \mathbb{R}^{d \times d_{ff}}$, $W_2 \in \mathbb{R}^{d_{ff} \times d}$
- **LayerNorm**: $\gamma, \beta \in \mathbb{R}^{d}$
- **LM Head**: usually shares weights with the embedding layer (weight tying)

### 1.2 Parameter Count Estimation

Using Llama-7B as an example:

| Component | Formula | Llama-7B (d=4096, L=32) |
|-----------|---------|------------------------|
| Embedding | $V \times d$ | 32000 * 4096 = 131M |
| QKV projection (per layer) | $3 \times d \times d$ | 3 * 4096 * 4096 = 50.3M |
| Output projection (per layer) | $d \times d$ | 4096 * 4096 = 16.8M |
| FFN (per layer) | $2 \times d \times d_{ff} + d \times d_{ff}$ | ~90M (with gate) |
| LayerNorm (per layer) | $2 \times d$ | 8K |
| **Total** | | **~6.7B** |

### 1.3 The Training Process

Training aims to adjust these weights to minimize loss:

$$
\theta^* = \arg\min_\theta \sum_{(x,y) \in \mathcal{D}} \mathcal{L}(f_\theta(x), y)
$$

Each update step:

$$
\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta \mathcal{L}
$$

After training, weights are saved to disk (e.g., safetensors format) and loaded into GPU memory for inference.

## 2. Inference Phase: Birth of KV Cache

### 2.1 Why KV Cache Is Needed

Autoregressive generation is token-by-token:

1. Input the prompt, get the first output token
2. Append the output token to the sequence, run forward pass again
3. Repeat until generation ends

The problem: every time a new token is generated, attention needs the current query against **all previous tokens' keys and values**. Without caching, each token would require recomputing K and V for all preceding tokens — $O(N^2)$ redundant work.

### 2.2 The KV Cache Solution

The core idea: **previous tokens' K and V don't change, so just cache them.**

```python
class AttentionWithKVCache(nn.Module):
    def forward(self, x, kv_cache=None):
        B, T, C = x.shape

        # Compute Q, K, V for current token
        q = self.w_q(x)  # (B, T, d)
        k = self.w_k(x)  # (B, T, d)
        v = self.w_v(x)  # (B, T, d)

        if kv_cache is not None:
            # Concatenate with previously cached K, V
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k], dim=1)  # (B, T_prev + T, d)
            v = torch.cat([v_prev, v], dim=1)  # (B, T_prev + T, d)

        # Attention: query is only current token, key/value spans full history
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(attn_weights, dim=-1)
        output = torch.matmul(attn_weights, v)

        # Update cache
        new_kv_cache = (k, v)
        return output, new_kv_cache
```

### 2.3 KV Cache Size

Per-layer KV Cache size:

$$
\text{KV Cache per layer} = 2 \times B \times N \times n_h \times d_h \times \text{bytes}
$$

Where:
- $2$: one each for K and V
- $B$: batch size
- $N$: current sequence length
- $n_h$: number of attention heads
- $d_h$: dimension per head ($d_{model} / n_h$)

Total KV Cache = per layer * number of layers $L$:

$$
\text{Total KV Cache} = 2 \times B \times N \times n_h \times d_h \times L \times \text{bytes}
$$

**KV Cache estimate for Llama-7B:**

| Parameter | Value |
|-----------|-------|
| Layers $L$ | 32 |
| Heads $n_h$ | 32 |
| Head dimension $d_h$ | 128 |
| Sequence length $N$ | 4096 |
| Batch size $B$ | 1 |
| Precision | FP16 (2 bytes) |
| **Total KV Cache** | 2 * 1 * 4096 * 32 * 128 * 32 * 2 = **2 GB** |

> **The sequence length effect:** KV Cache scales linearly with sequence length. For the same model, going from 4K to 128K context, KV Cache grows from 2 GB to 64 GB — exceeding the 7B model weights themselves (~14 GB FP16). This is why in long-context scenarios, KV Cache rather than model weights becomes the memory bottleneck.

## 3. The Relationship Between Model Weights and KV Cache

They're not independent — KV Cache is computed from Model Weights:

```
Input Token --> Embedding (weights) --> Hidden State
Hidden State --> W_K (weights) --> Key   --> cached in KV Cache
Hidden State --> W_V (weights) --> Value --> cached in KV Cache
Hidden State --> W_Q (weights) --> Query --> not cached, used once
```

The essential relationship:
- **Model Weights** define the computation graph (how to get K, V from inputs)
- **KV Cache** stores computation results (the actual K, V vectors)
- Weights are static, shared across all requests
- KV Cache is dynamic, independent per request

## 4. Advanced Topics

### 4.1 Optimization Techniques

**Model Weights optimizations:**

| Technique | Principle | Memory Savings |
|-----------|----------|---------------|
| Quantization (INT8/INT4) | Reduce weight precision | 2-4x |
| Weight Tying | Share embedding and LM Head | ~saves $V \times d$ |
| Pruning | Remove unimportant weights | Depends on sparsity rate |
| LoRA | Low-rank adaptation, keeps original weights frozen | Major savings during training |

**KV Cache optimizations:**

| Technique | Principle | Memory Savings |
|-----------|----------|---------------|
| Multi-Query Attention (MQA) | All heads share K/V | $n_h$ times |
| Grouped-Query Attention (GQA) | Group-shared K/V | $n_h / g$ times |
| KV Cache quantization | INT8/FP8 cache storage | 2x |
| PagedAttention | Paged management, reduces fragmentation | Reduces waste |
| Sliding Window | Keep only last $w$ tokens in cache | $N/w$ times |
| Token Eviction | Dynamically evict unimportant tokens | Depends on policy |

### 4.2 Common Interview Questions

**Q: What's in GPU memory during inference?**

A: Three main components:
1. Model Weights: static, loaded at startup
2. KV Cache: dynamic, grows with requests
3. Activation memory: intermediate activations (temporary during forward pass, much smaller than the other two)

**Q: Why does larger batch size improve throughput?**

A: Model Weights are shared across all requests. Going from batch 1 to 32, weights memory stays constant (read once), but Tensor Core utilization goes from very low to high (larger matrices, higher compute density). KV Cache grows linearly with batch, but as long as memory allows, throughput scales roughly linearly.

**Q: What's the difference between Prefill and Decode phases?**

A:
- **Prefill (first pass)**: processes the entire prompt, generates all KV Cache at once. Compute-intensive, high GPU utilization.
- **Decode (per-token generation)**: each step processes one token, reads from KV Cache. Memory-intensive, low GPU utilization.
- The different characteristics of these two phases are the foundation for scheduling optimizations in inference engines like SGLang and vLLM.

## 5. Llama1 Attention KV Cache Code Walkthrough

Here's the KV Cache-enabled Attention implementation from Llama1:

```python
class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # KV Cache initialization
        self.cache_k = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_heads, self.head_dim)
        )
        self.cache_v = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_heads, self.head_dim)
        )

    def forward(self, x, start_pos, freqs_cis, mask=None):
        bsz, seqlen, _ = x.shape

        # Compute Q, K, V using Model Weights
        xq = self.wq(x)  # (B, T, n_heads * head_dim)
        xk = self.wk(x)
        xv = self.wv(x)

        # Reshape to (B, T, n_heads, head_dim)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

        # Apply RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # Update KV Cache
        self.cache_k[:bsz, start_pos:start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos:start_pos + seqlen] = xv

        # Read full K, V from cache (including all previous tokens)
        keys = self.cache_k[:bsz, :start_pos + seqlen]
        values = self.cache_v[:bsz, :start_pos + seqlen]

        # Transpose for attention: (B, n_heads, T, head_dim)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)

        # Reshape back and project
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)
```

Key points in this code:

1. `self.cache_k` and `self.cache_v` pre-allocate space for the maximum sequence length at initialization
2. `start_pos` tracks the current write position, only writing new token K/V each step
3. Reading uses `[:start_pos + seqlen]` to fetch all historical K/V
4. Q is only the current token, K/V spans the full history — this is the core KV Cache pattern

> **Note:** This is Llama1's simplified implementation. Production inference engines (like SGLang, vLLM) use PagedAttention to manage KV Cache, avoiding memory waste from pre-allocation.

## Summary

- Model Weights are products of training, defining "how to compute"
- KV Cache is a product of inference, storing intermediate attention results
- For short sequences, weights dominate memory; for long sequences, KV Cache becomes the bottleneck
- Optimization strategies differ: weights use quantization/pruning; KV Cache uses GQA/PagedAttention/eviction
- Understanding this distinction is a prerequisite for understanding inference system (SGLang, vLLM) scheduling logic
