---
title: "KV Cache 与模型权重"
description: "理解 KV Cache 和 Model Weights 的区别与关联，掌握大模型推理优化的第一步"
date: 2025-06-23
tags: ["inference", "KV-cache", "LLM"]
category: "Technical"
lang: "zh"
---

理解 KV Cache 和 Model Weights 的区别与关联， 是搞懂大模型推理优化的第一步。 这篇文章从训练阶段和推理阶段两个角度， 拆解这两个核心概念。

## 核心概念一览

| 概念 | 本质 | 生成阶段 | 生命周期 | 大小与什么相关 |
|------|------|---------|---------|--------------|
| Model Weights | 网络参数 | 训练阶段学到 | 长期， 存在磁盘 / GPU | 模型架构（层数、hidden size） |
| KV Cache | 中间计算缓存 | 推理阶段动态生成 | 短期， 随请求创建和销毁 | 序列长度、batch size |

## 1. 训练阶段： Model Weights 的诞生

### 1.1 什么是 Model Weights

Model Weights（模型权重）是神经网络中所有可学习参数的总称。 对于一个 Transformer 模型， 这些参数包括：

- **Embedding 层**： token embedding $W_E \in \mathbb{R}^{V \times d}$， 位置编码参数
- **Attention 层**： $W^Q, W^K, W^V \in \mathbb{R}^{d \times d_k}$， $W^O \in \mathbb{R}^{d \times d}$
- **FFN 层**： $W_1 \in \mathbb{R}^{d \times d_{ff}}$， $W_2 \in \mathbb{R}^{d_{ff} \times d}$
- **LayerNorm**： $\gamma, \beta \in \mathbb{R}^{d}$
- **LM Head**： 通常和 embedding 层共享权重（weight tying）

### 1.2 参数量估算

以 Llama-7B 为例：

| 组件 | 参数量公式 | Llama-7B (d=4096, L=32) |
|------|----------|------------------------|
| Embedding | $V \times d$ | 32000 * 4096 = 131M |
| QKV projection (per layer) | $3 \times d \times d$ | 3 * 4096 * 4096 = 50.3M |
| Output projection (per layer) | $d \times d$ | 4096 * 4096 = 16.8M |
| FFN (per layer) | $2 \times d \times d_{ff} + d \times d_{ff}$ | ~90M (with gate) |
| LayerNorm (per layer) | $2 \times d$ | 8K |
| **总计** | | **~6.7B** |

### 1.3 训练过程

训练的目标是调整这些权重使得 loss 最小化：

$$
\theta^* = \arg\min_\theta \sum_{(x,y) \in \mathcal{D}} \mathcal{L}(f_\theta(x), y)
$$

每一步更新：

$$
\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta \mathcal{L}
$$

训练完成后， 权重被保存到磁盘（如 safetensors 格式）， 推理时加载到 GPU 显存。

## 2. 推理阶段： KV Cache 的诞生

### 2.1 为什么需要 KV Cache

自回归生成的过程是逐 token 的：

1. 输入 prompt， 得到第一个输出 token
2. 把输出 token append 到序列， 再次前向传播
3. 重复直到生成结束

问题在于： 每次生成新 token 时， attention 需要用当前 query 与**所有之前 token 的 key 和 value** 做计算。 如果不缓存， 每个 token 都要重新计算之前所有 token 的 K 和 V， 这是 $O(N^2)$ 的重复工作。

### 2.2 KV Cache 的解决方案

KV Cache 的核心想法： **之前 token 的 K 和 V 不会变， 缓存起来就行。 **

```python
class AttentionWithKVCache(nn.Module):
    def forward(self, x, kv_cache=None):
        B, T, C = x.shape

        # 计算当前 token 的 Q, K, V
        q = self.w_q(x)  # (B, T, d)
        k = self.w_k(x)  # (B, T, d)
        v = self.w_v(x)  # (B, T, d)

        if kv_cache is not None:
            # 拼接之前缓存的 K, V
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k], dim=1)  # (B, T_prev + T, d)
            v = torch.cat([v_prev, v], dim=1)  # (B, T_prev + T, d)

        # Attention 计算：query 只有当前 token，key/value 是全部历史
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(attn_weights, dim=-1)
        output = torch.matmul(attn_weights, v)

        # 更新 cache
        new_kv_cache = (k, v)
        return output, new_kv_cache
```

### 2.3 KV Cache 的大小

每一层的 KV Cache 大小：

$$
\text{KV Cache per layer} = 2 \times B \times N \times n_h \times d_h \times \text{bytes}
$$

其中：
- $2$： K 和 V 各一份
- $B$： batch size
- $N$： 当前序列长度
- $n_h$： attention head 数量
- $d_h$： 每个 head 的维度（$d_{model} / n_h$）

总 KV Cache = per layer * 层数 $L$：

$$
\text{Total KV Cache} = 2 \times B \times N \times n_h \times d_h \times L \times \text{bytes}
$$

**Llama-7B 的 KV Cache 估算： **

| 参数 | 值 |
|------|-----|
| 层数 $L$ | 32 |
| Head 数 $n_h$ | 32 |
| Head 维度 $d_h$ | 128 |
| 序列长度 $N$ | 4096 |
| Batch size $B$ | 1 |
| 精度 | FP16 (2 bytes) |
| **KV Cache 总量** | 2 * 1 * 4096 * 32 * 128 * 32 * 2 = **2 GB** |

> **序列长度的影响： ** KV Cache 与序列长度成正比。 同样的模型， 从 4K 到 128K 上下文， KV Cache 从 2 GB 涨到 64 GB， 直接超过了 7B 模型权重本身（~14 GB FP16）。 这就是为什么长上下文场景下， KV Cache 而不是模型权重成为显存瓶颈。

## 3. Model Weights 与 KV Cache 的关系

两者不是独立的 —— KV Cache 是由 Model Weights 计算得来的：

```
Input Token --> Embedding (weights) --> Hidden State
Hidden State --> W_K (weights) --> Key   --> 缓存到 KV Cache
Hidden State --> W_V (weights) --> Value --> 缓存到 KV Cache
Hidden State --> W_Q (weights) --> Query --> 不缓存，只用一次
```

本质关系：
- **Model Weights** 定义了计算图（怎么从输入得到 K、V）
- **KV Cache** 存储了计算结果（具体的 K、V 向量）
- Weights 是静态的， 所有请求共享
- KV Cache 是动态的， 每个请求独立

## 4. 进阶话题

### 4.1 优化技巧

**针对 Model Weights 的优化： **

| 技术 | 原理 | 显存节省 |
|------|------|---------|
| 量化 (INT8 / INT4) | 降低权重精度 | 2-4x |
| Weight Tying | Embedding 和 LM Head 共享 | ~节省 $V \times d$ |
| Pruning | 删除不重要的权重 | 取决于稀疏率 |
| LoRA | 低秩适配， 不改原始权重 | 训练时大幅节省 |

**针对 KV Cache 的优化： **

| 技术 | 原理 | 显存节省 |
|------|------|---------|
| Multi-Query Attention (MQA) | 所有 head 共享 K/V | $n_h$ 倍 |
| Grouped-Query Attention (GQA) | 分组共享 K/V | $n_h / g$ 倍 |
| KV Cache 量化 | INT8 / FP8 存储 cache | 2x |
| PagedAttention | 分页管理， 减少碎片 | 减少浪费 |
| Sliding Window | 只保留最近 $w$ 个 token 的 cache | $N/w$ 倍 |
| Token Eviction | 动态淘汰不重要的 token | 取决于策略 |

### 4.2 面试常见问题

**Q: 推理时 GPU 显存里都装了什么？**

A: 主要三部分：
1. Model Weights： 静态， 启动时加载
2. KV Cache： 动态， 随请求增长
3. Activation memory： 中间激活值（forward pass 临时使用， 量远小于前两者）

**Q: 为什么 batch size 越大， throughput 越高？**

A: Model Weights 是所有请求共享的。 batch 从 1 到 32， weights 的显存开销不变（只读一次）， 但 GPU 的 Tensor Core 利用率从极低到较高（矩阵更大， 计算密度更高）。 KV Cache 随 batch 线性增长， 但只要显存装得下， throughput 近似线性提升。

**Q: Prefill 和 Decode 阶段有什么区别？**

A:
- **Prefill（首次填充）**： 处理整个 prompt， 一次性生成所有 KV Cache。 计算密集， GPU 利用率高。
- **Decode（逐 token 生成）**： 每步只处理一个 token， 从 KV Cache 读取历史。 访存密集， GPU 利用率低。
- 这两个阶段的特性差异是 SGLang、vLLM 等推理引擎做调度优化的基础。

## 5. Llama1 Attention KV Cache 代码详解

下面是 Llama1 中带 KV Cache 的 Attention 实现：

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

        # KV Cache 初始化
        self.cache_k = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_heads, self.head_dim)
        )
        self.cache_v = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_heads, self.head_dim)
        )

    def forward(self, x, start_pos, freqs_cis, mask=None):
        bsz, seqlen, _ = x.shape

        # 用 Model Weights 计算 Q, K, V
        xq = self.wq(x)  # (B, T, n_heads * head_dim)
        xk = self.wk(x)
        xv = self.wv(x)

        # Reshape to (B, T, n_heads, head_dim)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

        # 应用 RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # 更新 KV Cache
        self.cache_k[:bsz, start_pos:start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos:start_pos + seqlen] = xv

        # 从 Cache 中读取完整的 K, V（包括之前所有 token）
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

代码中的关键点：

1. `self.cache_k` 和 `self.cache_v` 在初始化时预分配了最大序列长度的空间
2. `start_pos` 跟踪当前写入位置， 每次只写入新 token 的 K/V
3. 读取时用 `[:start_pos + seqlen]` 获取全部历史 K/V
4. Q 只有当前 token， K/V 是全部历史 —— 这就是 KV Cache 的核心模式

> **注意： ** 这是 Llama1 的简化实现。 生产级推理引擎（如 SGLang、vLLM）使用 PagedAttention 来管理 KV Cache， 避免预分配带来的显存浪费。

## 总结

- Model Weights 是训练阶段的产物， 定义了模型"怎么计算"
- KV Cache 是推理阶段的产物， 存储了 attention 的中间结果
- 短序列场景， weights 占显存大头；长序列场景， KV Cache 反而是瓶颈
- 优化两者的策略不同： weights 靠量化 / 剪枝， KV Cache 靠 GQA / PagedAttention / eviction
- 理解这个区分， 是搞懂 LLM 推理系统（SGLang、vLLM）调度逻辑的前提
