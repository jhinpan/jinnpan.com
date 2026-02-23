---
title: "Transformer Deep Dive (Math + Code)"
description: "Deconstructing Transformer's Self-Attention, LayerNorm, and MLP from math, code, and architecture perspectives"
date: 2025-03-09
tags: ["transformer", "deep-learning", "MLSys"]
category: "Technical"
lang: "en"
---

This post deconstructs the core components of the Transformer — Self-Attention, LayerNorm, and MLP — from three angles: math formulas, PyTorch code, and architectural role. For each component, I'll show the formula, the code, and then explain its role in the overall architecture.

## 1. Macro Architecture

A standard Transformer Encoder layer stacks these components:

```
Input
  ↓
LayerNorm → Multi-Head Self-Attention → Residual Add
  ↓
LayerNorm → MLP (FFN) → Residual Add
  ↓
Output
```

This is the Pre-LN architecture (used in GPT-2, Llama, etc.). The original Transformer uses Post-LN (LayerNorm after residual), but Pre-LN trains more stably.

One complete layer:

$$
\begin{aligned}
x' &= x + \text{MultiHeadAttn}(\text{LN}(x)) \\
\text{output} &= x' + \text{MLP}(\text{LN}(x'))
\end{aligned}
$$

## 2. Self-Attention: Math and Code

### 2.1 Scaled Dot-Product Attention

**Math:**

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

Step by step:

1. **Linear projections**: $Q = XW^Q$, $K = XW^K$, $V = XW^V$
2. **Similarity computation**: $S = QK^T \in \mathbb{R}^{N \times N}$
3. **Scaling**: $S = S / \sqrt{d_k}$ (prevents softmax gradient vanishing)
4. **Normalization**: $A = \text{softmax}(S)$, normalized per row
5. **Weighted sum**: $O = AV$

> **Why scale by $\sqrt{d_k}$?** Assume each component of $q$ and $k$ is an independent random variable with mean 0 and variance 1. Then $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ has variance $d_k$. When $d_k$ is large, $q \cdot k$ magnitudes are large, softmax outputs approach one-hot, and gradients approach zero. Dividing by $\sqrt{d_k}$ normalizes the variance to 1.

**PyTorch code:**

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (B, N_q, d_k)
    K: (B, N_k, d_k)
    V: (B, N_k, d_v)
    mask: (B, N_q, N_k) or broadcastable, True = masked
    """
    d_k = Q.shape[-1]

    # Step 1: Similarity computation + scaling
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    # scores: (B, N_q, N_k)

    # Step 2: Masking (optional, for causal attention)
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    # Step 3: Softmax normalization
    attn_weights = F.softmax(scores, dim=-1)
    # attn_weights: (B, N_q, N_k), each row sums to 1

    # Step 4: Weighted sum
    output = torch.matmul(attn_weights, V)
    # output: (B, N_q, d_v)

    return output, attn_weights
```

### 2.2 Multi-Head Attention

**Math:**

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
$$

$$
\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)
$$

Parameter dimensions:
- $W_i^Q, W_i^K \in \mathbb{R}^{d_{model} \times d_k}$, where $d_k = d_{model} / h$
- $W_i^V \in \mathbb{R}^{d_{model} \times d_v}$, typically $d_v = d_k$
- $W^O \in \mathbb{R}^{h \cdot d_v \times d_{model}}$

> **Why multiple heads?** A single head can only learn one "attention pattern." Multiple heads let the model attend to different types of information at different positions simultaneously (e.g., syntactic relations, semantic similarity, positional proximity).

**PyTorch code:**

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Merge three projection matrices into one (more efficient)
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        """
        x: (B, N, d_model)
        """
        B, N, _ = x.shape

        # Step 1: Linear projection to get Q, K, V
        qkv = self.W_qkv(x)  # (B, N, 3 * d_model)
        qkv = qkv.reshape(B, N, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, N, d_k)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        # Step 2: Scaled Dot-Product Attention (independent per head)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores: (B, n_heads, N, N)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        # attn_output: (B, n_heads, N, d_k)

        # Step 3: Concatenate all head outputs
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(B, N, self.d_model)
        # attn_output: (B, N, d_model)

        # Step 4: Output projection
        output = self.W_o(attn_output)
        return output
```

### 2.3 Causal Mask

Autoregressive models (GPT, Llama) need to ensure position $i$ can only see tokens at positions $\leq i$, implemented via an upper-triangular mask:

```python
def create_causal_mask(seq_len):
    """Returns (1, 1, N, N) causal mask, True = should be masked"""
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)

# Usage:
# mask = create_causal_mask(N)
# scores = scores.masked_fill(mask, float('-inf'))
```

## 3. LayerNorm: Math and Code

### 3.1 Math

Given input vector $x \in \mathbb{R}^d$:

$$
\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

Where:
- $\mu = \frac{1}{d} \sum_{i=1}^{d} x_i$ (mean, computed along hidden dimension)
- $\sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2$ (variance)
- $\gamma, \beta \in \mathbb{R}^d$ are learnable scale and shift parameters
- $\epsilon$ is a numerical stability term (e.g., $10^{-5}$)

### 3.2 LayerNorm vs BatchNorm

| Dimension | LayerNorm | BatchNorm |
|-----------|-----------|-----------|
| Normalization axis | Along hidden dimension (d) | Along batch dimension (B) |
| Statistics | Computed independently per sample | Computed across batch |
| At inference | No extra statistics needed | Needs running mean/var |
| Batch size sensitive | No | Yes |
| Use case | NLP, sequence models | CV, fixed-size inputs |

> **Why does Transformer use LayerNorm instead of BatchNorm?** Two reasons: (1) Variable sequence lengths mean different samples in a batch have different lengths, making BatchNorm statistics unstable; (2) Autoregressive inference is token-by-token (batch=1), where BatchNorm's running statistics don't apply well.

### 3.3 PyTorch Code

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
        # Compute mean and variance along the last dimension (hidden)
        mean = x.mean(dim=-1, keepdim=True)      # (B, N, 1)
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # (B, N, 1)

        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)  # (B, N, d_model)

        # Scale and shift
        return self.gamma * x_norm + self.beta
```

### 3.4 RMSNorm

Modern models like Llama use RMSNorm (Root Mean Square Normalization), which removes mean centering:

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

> **RMSNorm advantage:** Removing the mean computation eliminates one reduce operation. On GPUs, reduce operations involve thread synchronization, which is a performance bottleneck. Benchmarks show RMSNorm is ~10-15% faster than LayerNorm with negligible impact on model quality.

## 4. MLP / FFN: Math and Code

### 4.1 Standard FFN

**Math:**

$$
\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2
$$

- $W_1 \in \mathbb{R}^{d_{model} \times d_{ff}}$: up projection, expands dimension
- $W_2 \in \mathbb{R}^{d_{ff} \times d_{model}}$: down projection, compresses back
- $d_{ff}$ is typically $4 \times d_{model}$

**GELU activation function:**

$$
\text{GELU}(x) = x \cdot \Phi(x) \approx 0.5x\left(1 + \tanh\left[\sqrt{2/\pi}(x + 0.044715x^3)\right]\right)
$$

### 4.2 SwiGLU FFN

Llama, Qwen, and similar models use the SwiGLU variant with a gate:

$$
\text{SwiGLU}(x) = W_2 \cdot \left[\text{SiLU}(W_{gate} x) \odot (W_{up} x)\right]
$$

- $W_{gate} \in \mathbb{R}^{d_{model} \times d_{ff}}$: gate projection
- $W_{up} \in \mathbb{R}^{d_{model} \times d_{ff}}$: up projection
- $W_2 \in \mathbb{R}^{d_{ff} \times d_{model}}$: down projection
- $\text{SiLU}(x) = x \cdot \sigma(x)$

> **Why SwiGLU?** Empirically, SwiGLU outperforms GELU FFN at the same parameter count. The cost is an extra gate projection (parameters go from $2 \times d \times d_{ff}$ to $3 \times d \times d_{ff}$), but usually $d_{ff}$ is reduced to keep total parameter count constant (e.g., Llama uses $d_{ff} = 2/3 \times 4d$).

**PyTorch code:**

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

### 4.3 MLP Parameter Count

| FFN Type | Parameter Count | Example (d=4096, d_ff=11008) |
|----------|----------------|------------------------------|
| Standard (GELU) | $2 \times d \times d_{ff}$ | 90.2M |
| SwiGLU | $3 \times d \times d_{ff}$ | 135.3M |
| SwiGLU (adjusted $d_{ff}$) | $3 \times d \times \frac{8d}{3}$ | 134.2M |

## 5. Complete Encoder Layer

Assembling all three components:

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

Shape flow through the layer:

```
Input: (B, N, d_model) = (32, 2048, 4096)
  ↓ RMSNorm
(32, 2048, 4096)
  ↓ Multi-Head Attention (h=32, d_k=128)
    Q, K, V: (32, 32, 2048, 128)   -- per head
    Attention: (32, 32, 2048, 2048) -- attention matrix
    Output: (32, 2048, 4096)        -- after concat + projection
  ↓ Residual Add
(32, 2048, 4096)
  ↓ RMSNorm
(32, 2048, 4096)
  ↓ SwiGLU FFN
    Gate: (32, 2048, 11008)  -- expanded
    Up:   (32, 2048, 11008)
    Down: (32, 2048, 4096)   -- compressed back
  ↓ Residual Add
Output: (32, 2048, 4096)
```

## 6. Common Interview Questions

**Q: Why does Transformer use additive residual connections instead of concatenation?**

A: Addition preserves dimensions (unlike concatenation which doubles them), making arbitrary layer stacking possible. Mathematically, residual connections let gradients flow directly ($\partial(x + f(x))/\partial x = 1 + \partial f/\partial x$), mitigating vanishing gradients in deep networks.

**Q: Where exactly does the $O(N^2)$ in attention come from?**

A: It comes from the $QK^T$ step. $Q \in \mathbb{R}^{N \times d_k}$, $K \in \mathbb{R}^{N \times d_k}$, their product is an $\mathbb{R}^{N \times N}$ attention matrix. Storing this matrix requires $O(N^2)$ space; computing it requires $O(N^2 d_k)$ time.

**Q: What's the parameter count of Multi-Head Attention?**

A: $3 \times d_{model}^2 + d_{model}^2 = 4d_{model}^2$. Three QKV projections each $d_{model} \times d_{model}$, plus one output projection $d_{model} \times d_{model}$. Note this is independent of head count (head count only affects $d_k$, not total parameters).

**Q: Difference between Pre-LN and Post-LN?**

A:
- Post-LN (original Transformer): $x + \text{LN}(\text{SubLayer}(x))$. Gradients get scaled by LN, causing instability in deep networks, requiring warmup.
- Pre-LN: $x + \text{SubLayer}(\text{LN}(x))$. The residual path has no nonlinear transformations, so gradient flow is more stable. Downside: gradient explosion possible in very deep networks (>100 layers), but works well for typical depths (32-96 layers).

**Q: Why do Embedding and LM Head often share weights (Weight Tying)?**

A: Both have the same dimensions ($V \times d_{model}$), so sharing reduces parameter count. Intuitively, if two words are close in embedding space, their output probabilities should also be similar. Empirically, weight tying has nearly no negative impact on quality while saving $V \times d_{model}$ parameters (for Llama-7B, that's ~131M parameters).
