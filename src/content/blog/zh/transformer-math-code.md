---
title: "Transformer 详解 (数学 + 代码)"
description: "从数学原理、代码实现、逻辑结构三个维度拆解 Transformer 的 Self-Attention、LayerNorm 和 MLP"
date: 2025-03-09
tags: ["transformer", "deep-learning", "MLSys"]
category: "Technical"
lang: "zh"
---

这篇文章从数学公式、PyTorch 代码、逻辑结构三个角度拆解 Transformer 的核心组件： Self-Attention、LayerNorm 和 MLP。 每个组件先给公式， 再给代码， 最后讲它在整个架构中的角色。

## 1. 宏观架构

一个标准的 Transformer Encoder 层由以下组件堆叠：

```
Input
  ↓
LayerNorm → Multi-Head Self-Attention → Residual Add
  ↓
LayerNorm → MLP (FFN) → Residual Add
  ↓
Output
```

这是 Pre-LN 架构（GPT-2、Llama 等采用）。 原始 Transformer 用 Post-LN（LayerNorm 在 residual 之后）， 但 Pre-LN 训练更稳定。

完整的一层：

$$
\begin{aligned}
x' &= x + \text{MultiHeadAttn}(\text{LN}(x)) \\
\text{output} &= x' + \text{MLP}(\text{LN}(x'))
\end{aligned}
$$

## 2. Self-Attention： 数学与代码

### 2.1 Scaled Dot-Product Attention

**数学公式：**

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

分步骤拆解：

1. **线性投影**： $Q = XW^Q$， $K = XW^K$， $V = XW^V$
2. **相似度计算**： $S = QK^T \in \mathbb{R}^{N \times N}$
3. **缩放**： $S = S / \sqrt{d_k}$（防止 softmax 梯度消失）
4. **归一化**： $A = \text{softmax}(S)$， 每行归一化
5. **加权求和**： $O = AV$

> **为什么要缩放 $\sqrt{d_k}$？** 假设 $q$ 和 $k$ 的每个分量都是均值 0、方差 1 的独立随机变量。 那么 $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ 的方差是 $d_k$。 当 $d_k$ 很大时， $q \cdot k$ 的绝对值很大， softmax 的输出趋近于 one-hot， 梯度趋近于零。 除以 $\sqrt{d_k}$ 把方差归一化为 1。

**PyTorch 代码：**

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (B, N_q, d_k)
    K: (B, N_k, d_k)
    V: (B, N_k, d_v)
    mask: (B, N_q, N_k) or broadcastable, True = masked (不参与计算)
    """
    d_k = Q.shape[-1]

    # Step 1: 相似度计算 + 缩放
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    # scores: (B, N_q, N_k)

    # Step 2: 掩码（可选，用于 causal attention）
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    # Step 3: Softmax 归一化
    attn_weights = F.softmax(scores, dim=-1)
    # attn_weights: (B, N_q, N_k), 每行和为 1

    # Step 4: 加权求和
    output = torch.matmul(attn_weights, V)
    # output: (B, N_q, d_v)

    return output, attn_weights
```

### 2.2 Multi-Head Attention

**数学公式：**

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
$$

$$
\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)
$$

参数维度：
- $W_i^Q, W_i^K \in \mathbb{R}^{d_{model} \times d_k}$， 其中 $d_k = d_{model} / h$
- $W_i^V \in \mathbb{R}^{d_{model} \times d_v}$， 通常 $d_v = d_k$
- $W^O \in \mathbb{R}^{h \cdot d_v \times d_{model}}$

> **为什么用多头？** 单头 attention 只能学一种"关注模式"。 多头让模型同时关注不同位置的不同类型的信息（如语法关系、语义相似度、位置邻近等）。

**PyTorch 代码：**

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # 三个投影矩阵合并成一个大矩阵（效率更高）
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        """
        x: (B, N, d_model)
        """
        B, N, _ = x.shape

        # Step 1: 线性投影，得到 Q, K, V
        qkv = self.W_qkv(x)  # (B, N, 3 * d_model)
        qkv = qkv.reshape(B, N, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, N, d_k)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        # Step 2: Scaled Dot-Product Attention（每个 head 独立）
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores: (B, n_heads, N, N)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        # attn_output: (B, n_heads, N, d_k)

        # Step 3: 拼接所有 head 的输出
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(B, N, self.d_model)
        # attn_output: (B, N, d_model)

        # Step 4: 输出投影
        output = self.W_o(attn_output)
        return output
```

### 2.3 Causal Mask

自回归模型（GPT、Llama）需要确保位置 $i$ 只能看到位置 $\leq i$ 的 token， 通过上三角掩码实现：

```python
def create_causal_mask(seq_len):
    """返回 (1, 1, N, N) 的 causal mask, True = 需要被 mask"""
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)

# 使用：
# mask = create_causal_mask(N)
# scores = scores.masked_fill(mask, float('-inf'))
```

## 3. LayerNorm： 数学与代码

### 3.1 数学公式

给定输入向量 $x \in \mathbb{R}^d$：

$$
\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

其中：
- $\mu = \frac{1}{d} \sum_{i=1}^{d} x_i$（均值， 沿 hidden dimension 计算）
- $\sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2$（方差）
- $\gamma, \beta \in \mathbb{R}^d$ 是可学习的缩放和偏移参数
- $\epsilon$ 是数值稳定项（如 $10^{-5}$）

### 3.2 LayerNorm vs BatchNorm

| 维度 | LayerNorm | BatchNorm |
|------|-----------|-----------|
| 归一化方向 | 沿 hidden dimension (d) | 沿 batch dimension (B) |
| 统计量 | 每个 sample 独立计算 | 跨 batch 计算 |
| 推理时 | 无需额外统计量 | 需要 running mean / var |
| 对 batch size 敏感 | 否 | 是 |
| 适用场景 | NLP、序列模型 | CV、固定大小输入 |

> **为什么 Transformer 用 LayerNorm 不用 BatchNorm？** 两个原因： (1) 序列长度可变， batch 内不同样本的长度不同， BatchNorm 的统计量不稳定；(2) 自回归推理是逐 token 的（batch=1）， BatchNorm 的 running statistics 不适用。

### 3.3 PyTorch 代码

```python
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        """
        x: (B, N, d_model)
        """
        # 沿最后一个维度（hidden dimension）计算均值和方差
        mean = x.mean(dim=-1, keepdim=True)      # (B, N, 1)
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # (B, N, 1)

        # 归一化
        x_norm = (x - mean) / torch.sqrt(var + self.eps)  # (B, N, d_model)

        # 缩放和偏移
        return self.gamma * x_norm + self.beta
```

### 3.4 RMSNorm

Llama 等现代模型使用 RMSNorm（Root Mean Square Normalization）， 去掉了均值中心化：

$$
\text{RMSNorm}(x) = \gamma \odot \frac{x}{\text{RMS}(x) + \epsilon}
$$

$$
\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2}
$$

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x / rms)
```

> **RMSNorm 的优势：** 去掉均值计算后减少了一次 reduce 操作。 在 GPU 上， reduce 操作涉及线程间同步， 是性能瓶颈之一。 实测 RMSNorm 比 LayerNorm 快约 10-15%， 且对模型质量几乎无影响。

## 4. MLP / FFN： 数学与代码

### 4.1 标准 FFN

**数学公式：**

$$
\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2
$$

- $W_1 \in \mathbb{R}^{d_{model} \times d_{ff}}$： up projection， 扩展维度
- $W_2 \in \mathbb{R}^{d_{ff} \times d_{model}}$： down projection， 压缩回去
- $d_{ff}$ 通常是 $4 \times d_{model}$

**GELU 激活函数：**

$$
\text{GELU}(x) = x \cdot \Phi(x) \approx 0.5x\left(1 + \tanh\left[\sqrt{2/\pi}(x + 0.044715x^3)\right]\right)
$$

### 4.2 SwiGLU FFN

Llama、Qwen 等使用 SwiGLU 变体， 加了一个 gate：

$$
\text{SwiGLU}(x) = W_2 \cdot \left[\text{SiLU}(W_{gate} x) \odot (W_{up} x)\right]
$$

- $W_{gate} \in \mathbb{R}^{d_{model} \times d_{ff}}$： gate projection
- $W_{up} \in \mathbb{R}^{d_{model} \times d_{ff}}$： up projection
- $W_2 \in \mathbb{R}^{d_{ff} \times d_{model}}$： down projection
- $\text{SiLU}(x) = x \cdot \sigma(x)$

> **SwiGLU 为什么好？** 经验上 SwiGLU 比 GELU FFN 在同样参数量下效果更好。 代价是多了一个 gate 投影矩阵（参数量从 $2 \times d \times d_{ff}$ 变成 $3 \times d \times d_{ff}$）， 但通常把 $d_{ff}$ 缩小一些来保持总参数量不变（如 Llama 的 $d_{ff} = 2/3 \times 4d$）。

**PyTorch 代码：**

```python
class SwiGLU_FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        """
        x: (B, N, d_model)
        """
        gate = F.silu(self.w_gate(x))   # (B, N, d_ff)
        up = self.w_up(x)               # (B, N, d_ff)
        return self.w_down(gate * up)    # (B, N, d_model)
```

### 4.3 MLP 的参数量

| FFN 类型 | 参数量 | 示例 (d=4096, d_ff=11008) |
|---------|--------|--------------------------|
| Standard (GELU) | $2 \times d \times d_{ff}$ | 90.2M |
| SwiGLU | $3 \times d \times d_{ff}$ | 135.3M |
| SwiGLU (adjusted $d_{ff}$) | $3 \times d \times \frac{8d}{3}$ | 134.2M |

## 5. 完整 Encoder 层

把上面三个组件组装起来：

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, norm_eps=1e-5):
        super().__init__()
        self.attn_norm = RMSNorm(d_model, eps=norm_eps)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ffn_norm = RMSNorm(d_model, eps=norm_eps)
        self.ffn = SwiGLU_FFN(d_model, d_ff)

    def forward(self, x, mask=None):
        """
        x: (B, N, d_model)
        Pre-LN architecture with residual connections
        """
        # Sub-layer 1: Attention
        residual = x
        x = self.attn_norm(x)
        x = self.attn(x, mask=mask)
        x = residual + x

        # Sub-layer 2: FFN
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = residual + x

        return x
```

数据流的形状变化：

```
Input: (B, N, d_model) = (32, 2048, 4096)
  ↓ RMSNorm
(32, 2048, 4096)
  ↓ Multi-Head Attention (h=32, d_k=128)
    Q, K, V: (32, 32, 2048, 128)  -- 每个 head
    Attention: (32, 32, 2048, 2048)  -- 注意力矩阵
    Output: (32, 2048, 4096)        -- 拼接 + 投影后
  ↓ Residual Add
(32, 2048, 4096)
  ↓ RMSNorm
(32, 2048, 4096)
  ↓ SwiGLU FFN
    Gate: (32, 2048, 11008)  -- 扩展
    Up:   (32, 2048, 11008)
    Down: (32, 2048, 4096)   -- 压缩回来
  ↓ Residual Add
Output: (32, 2048, 4096)
```

## 6. 面试常见问题

**Q: 为什么 Transformer 用加法 residual connection 而不是 concatenation？**

A: 加法保持维度不变（不像 concatenation 会倍增维度）， 使得任意层数的堆叠成为可能。 数学上， residual connection 让梯度可以直接流过（$\partial(x + f(x))/\partial x = 1 + \partial f/\partial x$）， 缓解深层网络的梯度消失问题。

**Q: Attention 的 $O(N^2)$ 具体体现在哪？**

A: 体现在 $QK^T$ 这一步。 $Q \in \mathbb{R}^{N \times d_k}$, $K \in \mathbb{R}^{N \times d_k}$， 相乘得到 $\mathbb{R}^{N \times N}$ 的注意力矩阵。 存储这个矩阵需要 $O(N^2)$ 空间， 计算它需要 $O(N^2 d_k)$ 时间。

**Q: Multi-Head Attention 的参数量是多少？**

A: $3 \times d_{model}^2 + d_{model}^2 = 4d_{model}^2$。 三个 QKV 投影矩阵各 $d_{model} \times d_{model}$， 输出投影 $d_{model} \times d_{model}$。 注意这和 head 数量无关（head 数量只影响 $d_k$， 不影响总参数量）。

**Q: Pre-LN 和 Post-LN 的区别？**

A:
- Post-LN（原始 Transformer）： $x + \text{LN}(\text{SubLayer}(x))$。 梯度经过 LN 时会被缩放， 深层时训练不稳定， 需要 warmup。
- Pre-LN： $x + \text{SubLayer}(\text{LN}(x))$。 residual 路径上没有非线性变换， 梯度流更稳定。 缺点是深层（> 100 层）时可能出现梯度爆炸， 但对于常见深度（32-96 层）效果很好。

**Q: 为什么 Embedding 和 LM Head 常常共享权重（Weight Tying）？**

A: 两者维度相同（$V \times d_{model}$）， 共享可以减少参数量。 直觉上， 如果两个词在 embedding 空间中相近， 那么在输出时它们的概率也应该相近。 实测对模型质量几乎没有负面影响， 节省了 $V \times d_{model}$ 个参数（对于 Llama-7B， 这是 ~131M 参数）。
