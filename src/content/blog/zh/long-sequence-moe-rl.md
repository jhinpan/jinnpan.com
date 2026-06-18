---
title: "长序列 MoE RL 训练：从第一性原理到 MI300X"
description: "从第一性原理重新推导 Yan Bai 的长序列 MoE RL 优化（Path B 重计算、 linear cross-entropy、 FSDP2、 chunked expert-parallel overlap）， 以及它们在 AMD MI300X / MI355X 上各自意味着什么。"
date: 2026-06-18
tags: ["RL", "MoE", "long-context", "MLSys", "AMD"]
category: "Technical"
lang: "zh"
---

Yan Bai 写了一篇很扎实的文章， 讲在 32 张 H100 上做 Qwen3.5-35B 的 128K 长序列 RL 训练： [长序列 MoE RL 训练优化](https://iseekyan.github.io/zh/posts/qwen35-long-sequence-moe-rl/)。 原文值得自己读一遍。 这篇不是翻译。 我想做的是从第一性原理把这些招数重新推一遍， 核对数字， 然后问一个原文没问的问题： 当底下的卡从 H100 换成 MI300X， 哪些东西会变？

对我来说最后这一点才是重点。 长上下文训练这套打法， 基本都是针对 80 GB HBM 和 NVLink 写出来的。 AMD 的卡有 2.4–3.6 倍的显存， 互联也不一样， 所以有些招会变得更重要， 有些会变得没那么必要。 想分清哪个是哪个， 唯一的办法是搞懂每个招*为什么*管用， 而不是只知道它管用。

## 1. RL 不是 pretrain， 优化目标完全不同

原文做对的第一件事是一个 framing， 而不是某个 kernel。 Pretrain 是一个跑几周的巨型任务， 卡数固定且拉满。 在那里目标就是纯 MFU： 你在 roofline 上漏掉的每一个百分点， 都会乘以整个训练周期的长度。 所以人们愿意付任何复杂度代价（手调 4D 并行、 针对每种拓扑做专门的 sweep）只为把利用率抠回来。

RL post-training 的形态完全不同。 每一步是 rollout（自回归生成， memory-bound 且 latency-bound）接着 train（策略更新， compute-bound）。 而且你不是跑一个任务。 你是并行跑很多个小实验（调 reward、 调 KL 系数、 换数据配比）， 每个只占几张卡， 想法来了就起、 没用就停。

于是 binding constraint 翻转了。 它不再是「某一个任务的峰值 MFU」， 而是鲁棒性、 简单性， 以及能把任务随时重新部署到当下空闲的任意卡数上、 不用重新推导一套并行方案。 一个跑在峰值 90% 但在任意节点数上都能可靠启动的配置， 胜过一个 98% MFU、 但 batch shape 一变就得重新 sweep 的配置。 记住这句话， 后面几乎每个决定都是它推出来的。

## 2. Path B： 花便宜的资源去买稀缺的资源

在 128K 上下文下， 瓶颈是 activation 显存。 出路有两条。

**Path A** 通过把序列切到更多卡上来压小 activation： 加大 context / sequence parallelism， 让每张卡只持有更短的一段。 它省显存， 但代价是多一个并行维度、 更多 collective， 以及一套每次 layout 变化都得重调的拓扑。

**Path B** 反着来。 它接受*完整的 activation 重计算*（在 backward 里重新生成 activation， 而不是把它存下来）， 然后用省下来的显存去*加长*单卡的本地序列。 你拿 compute 换 memory、 换简单： 更少的 rank、 更少的 collective、 每张卡一段又粗又长的序列。

这里的套利逻辑值得把算术写出来。 正常训练每个 token 大约花 6N FLOPs（forward 2N + backward 4N）。 完整重计算会在 backward 里再跑一遍 forward， 所以你付的是 8N 而不是 6N， 一笔固定的 **+33% FLOP 税**。 听起来很亏， 直到你看实际吞吐落在哪里： ~186 TFLOPs/GPU 对上 H100 的 ~989 TFLOPs dense bf16 峰值， 大约是 **19% MFU**。 机器在 compute 上空着大约 81%。 花一个充裕的资源（FLOPs）去缓解那个真正卡你的资源（memory）， 是教科书式的 roofline 推理。 而更长的本地序列还是个赠品： 更大的 GEMM 跑得更高效， kernel launch 次数也更少， 这恰恰在 launch overhead 咬人的低 MFU 区间里最值钱。

证据就是原文报的结果： **80 GB 的 H100 上峰值只用了 37–38 GB**。 Path B 不只是塞进去了， 它还空出了半张卡， 而这正是你把好几个 RL 实验塞到一个节点上所需要的余量。

## 3. Linear cross-entropy： 别为了一个标量去建一个 32 GB 的张量

语言模型的 head 计算 logits `Z = H Wᵀ`， 形状是 `[tokens × vocab]`， 然后通过 softmax 把它归约成每个 token 一个 cross-entropy loss。 朴素实现会把 `Z` *以及*它的梯度 `dZ` 都实例化成稠密的 `[tokens × vocab]` 张量。 这几乎是纯浪费： `Z` 是一个你马上就要归约掉的中间量， 而 backward 真正需要的只有 `dZ = softmax(Z) − onehot(target)`。

按原文的示例配置（65,536 个 token、 vocab 124,160、 bf16）， 一份 logits 就是：

```
65,536 × 124,160 × 2 bytes = 16.27 GB
```

朴素 backward 里 logits 和它的梯度要同时驻留， 所以是约 2 份： **32.55 GB**。（一个值得记住的巧合： 一份 fp32 的 logits 也是 32.55 GB， 因为 4 bytes × 1 份 = 2 bytes × 2 份。 怎么算都是这个数。）

Fused linear cross-entropy 根本不建那个张量。 它把 token 轴切成 block； 对每个 block， 它在线计算这一片 logits， 用和 FlashAttention 同样的数值稳定 running-max + log-sum-exp 递推把它归约掉， 只保留 O(block) 个标量来累加 loss。 在 backward 里它*重新计算*这一片 logits（很便宜， 输入还驻留着）， 直接为这一片构造 `dZ`， 然后立刻把它收缩进 `dWᵀ` 和 `dH`。 这一片在循环内部就消亡了。 vocab 维度的峰值驻留从 `[tokens × vocab]`（16 GB）降到 `[block × vocab]`（不到 1 GB）。

这就是 Liger-Kernel 的「fused linear cross-entropy」模式； Apple 的「Cut Cross-Entropy」更进一步， 把 tile 留在 SRAM 里、 只索引 target logit。 代价是 backward 里多一次 projection GEMM（又是便宜的 FLOPs）， 用来换掉一个 32 GB 的临时张量。 和 Path B 一样的交易， 只是用在了那个唯一被 activation 显存（而非 compute）卡住的 kernel 上。

## 4. FSDP2： 把你重计算不掉的显存切开

重计算和 linear CE 攻的都是 *activation* 显存。 但有一层地板你重计算不过去： 持久状态。 对 mixed-precision Adam 来说， 每个参数要存一个 bf16 权重 + 一份梯度 + 一份 fp32 master 权重 + 两个 Adam moment， 大约 **每参数 16–18 bytes**， 完全和序列长度无关。 你没法重新生成它； 唯一的杠杆是把它切到各个 rank 上。

FSDP1 切的是一个扁平化的 `FlatParameter`： 它把很多张量拼进每个 unit 一个不透明的 buffer， 再把这个 buffer 切到 data-parallel group 上。 这个扁平 blob 跟混合 dtype 打架、 让逐张量处理变得别扭， 而且和其它并行轴组合得很差， 因为它根本不知道哪个逻辑张量对应哪个 mesh 维度。

FSDP2 把它换成**逐参数的 `DTensor` 切分**。 每个参数仍是它自己的张量， 切在某个命名的 `DeviceMesh` 轴上， 梯度和 optimizer state 继承同样的切分。 因为切分的单位是命名 mesh 维度上的逻辑张量， FSDP2 能在一个共享 mesh 上和 tensor / context / expert parallelism 干净地组合。 一个参数就是一个在多个维度上都有 placement 的 `DTensor`。 compute 的机制没变： 即时 all-gather 出完整参数、 用掉、 释放。 你拿一点通信换更小的驻留占用。

原文报的静态显存从 **55.91 GB 降到 47.03 GB**， 降了 8.88 GB， 约 16%。 没有参数量和 DP degree， 我没法复现这个绝对值， 但算术是自洽的， 而且这个量级对于「把 params + grads + optimizer state 切到全局 DP 维度上」是对的。 在这里值得做的理由， 和其它一切是同一个 RL 专属理由： 一旦 activation 被压扁（重计算 + linear CE）， 持久状态*就是*占用的大头， 所以切它是剩下杠杆里最高的那个。

## 5. Chunked expert-parallel overlap： 把 all-to-all 藏到 GEMM 底下

带 expert parallelism 的 MoE 层有一套固定的 dataflow。 router 为每个 token 选 expert； 一次 **dispatch** all-to-all 把每个 token 送到持有它 expert 的 rank； 每个 rank 跑本地的 expert GEMM； 一次 **combine** all-to-all 把输出送回去。 朴素地做， 这是三个串行阶段， 而 all-to-all 是一个全局的、 吃带宽的 collective， 它跑的时候 matrix core 在干等：

```
串行（每个 MoE 层）:
  NIC : [ dispatch ]················[ combine ]
  CU  : ···········[ expert GEMM ]···········
  time ──────────────────────────────────────▶
  cost ≈ T_dispatch + T_gemm + T_combine
```

切 *token* 维度能打破这个依赖， 因为 routing 是逐 token 的： 每个 chunk 都是一个独立的工作单元。 把本地 token 切成若干 chunk， 然后软件流水： 在 matrix core 算 chunk `i` 的同时 dispatch chunk `i+1`、 同时 combine chunk `i−1`。 通信和计算现在占的是不同的硬件（网络引擎 vs matrix core）， 所以它们能并发：

```
流水化（按 token 切成 3 个 chunk）:
  NIC : D0  D1  C0  D2  C1      C2
  CU  :     G0  G1      G2
  time ──────────────────────────────────────▶
  cost → max(T_comm, T_gemm) + 一次 ramp + 一次 drain
```

原文测到 **64K 序列下 23.97% 的加速**。 用一个简单的双项模型反推， 隐含的 all-to-all 占比约是未重叠层时间的 19%（`1 / (1 − 0.19) ≈ 1.24`）， 对这么大的 token 流量来说完全合理。 我喜欢的细节是： 这个收益*随序列变长而变大*， 但不是因为通信占比变大了（通信和计算都随 token 线性增长）。 它变大是因为更长的序列意味着更多的 chunk， 而更深的流水把固定的 ramp / drain 成本摊薄了。 用 2048 的 chunk， 16K → 32K → 64K 是 8 → 16 → 32 个 chunk， overlap 效率 `k/(k+1)` 从 0.89 → 0.94 → 0.97 一路爬。

## 6. Recipe： 把 4D sweep 塌缩掉

最后这一招最被低估。 朴素的并行搜索是一个 PP / TP / EP / CP 的 4D 网格。 每维三个值就是 81 个配置， 而在 RL 里每个都要烧真实的 GPU 时， 还有相当一部分在产出任何数字之前就 OOM 或者因为 shape 不对而挂掉。

原文的洞见是： 四个维度里有三个各自有一个*主导*的 binding constraint， 所以你可以独立求解、 而不是联合求解：

| 维度 | 被什么约束 | 规则 |
|-----|----------|------|
| **EP** | expert 数 + 负载均衡 | 由模型规模定： 选一个让 experts-per-rank 落在合理区间的 EP。 近乎离散， 没什么好 sweep 的。 |
| **CP** | activation 显存（∝ 本地序列） | 让 CP 把本地序列压到 16K–32K 窗口。 从 128K 上下文算， `CP=8 → 16K`、 `CP=4 → 32K`。 一个不等式， 不是一次 sweep。 |
| **TP** | 每层 compute / comm（每层一次 all-reduce） | 只在某层装不下或喂不饱一张卡时才用。 |
| **PP** | 流水气泡 `(P−1)/(microbatches+P−1)` | 只在追峰值时才用。 |

这就把 81 个配置变成大约 1 个， 外加一个小小的 PP 收尾微调。 你让出的那几个百分点 MFU， 换来的是配置搜索迭代次数的 ~50–80 倍削减， 而在 RL 里， 迭代速度*就是* binding constraint。 这和 Path B 是同一个逻辑， 只是上升了一层： 优化那个循环， 而不是那一圈。

把这一切兜住的工程， 是作者称之为 **bumblebee / Megatron-Lite** 的设计： 一套有着紧致 primitive 边界的架构， 让每个优化（loss = linear CE， optimizer = FSDP2， comm = chunked EP overlap）都成为一个可替换的模块， 每个都对着一个成对的 baseline 来验证。 这不是脚注。 小的改动半径加上可 A/B 验证的模块， 恰恰是你能信任一个被你不停重新配置的 stack 的前提， 而这正是整个 RL 的赌注。

## 7. 老老实实读数字

在把任何东西往 AMD 上搬之前， 有两点要先说清楚， 好让比较保持干净。

**永远用 dense 峰值。** H100 的 bf16 tensor 峰值是 dense 989 TFLOPs。 你常看到的 1979 那个数是带 2:4 结构化稀疏的， 而稀疏从不适用于 dense 训练。 用 186 除以 1979， 你会报出一个错误的 ~9.4% MFU。 老实的数字是 ~18.8%。

**其实有两个老实的 MFU。** ~18.8% 是 hardware-FLOP MFU。 但 Path B 的完整重计算意味着这些 FLOPs 里有约 33% 是那次重计算 forward， 不是 model work。 如果报出的 TFLOPs 把重计算算作有效工作， 那么你拿去和一个不重计算的 baseline 比的 model-FLOP 速率， 更接近 ~14%。 两个都不算错； 它们回答的是不同的问题。 只是别把它们混着用。

原文里有两点我是当作既定前提、 而非自己核实过的： 「Qwen3.5-35B」超出了我的知识时点， 而它给的 vocab 124,160 和我知道的 Qwen3 vocab（151,936）对不上。 不过 124,160 = 970 × 128， 对 TP 是对齐的， 所以作为算术输入是个合理的数。 把这两个都当成原文给定的配置， 别当真理。

## 8. 换上 AMD 的视角

现在到我真正关心的部分。 这四招里有三招是框架层面的， 搬到 ROCm 上基本不用改： FSDP2 是 PyTorch 的 `DTensor` / `DeviceMesh`， linear CE 是一个代数技巧， EP overlap 是一个调度模式。 但*每一招有多重要*会变， 因为稀缺资源不一样了。

| GPU | HBM | 带宽 | Dense bf16 峰值 | 186 TFLOPs 占 |
|-----|-----|-----|----------------|---------------|
| H100 SXM | 80 GB | 3.35 TB/s | 989 TFLOPs | ~18.8% |
| MI300X | 192 GB | 5.3 TB/s | 1307 TFLOPs | ~14.2% |
| MI355X | 288 GB HBM3E | 8 TB/s | 2500 TFLOPs | ~7.5% |

整个显存故事在 AMD 上都松了下来。 38 GB 的峰值在 MI300X 上是 20%， 在 MI355X 上是 13%。 于是：

- **Path B 变得*更*划算， 重计算税*更*没必要。** 有了 192–288 GB， 你可以让更多 activation 驻留、 少重计算掉那 33% 税里的一部分， 或者在根本用不上 CP 之前把本地序列拉得更长。 那个直觉（拉长序列、 花 FLOPs）是对的， 只是最优的重计算比例更低。 compute 余量也更大（186 TFLOPs 是 MI300X dense 峰值的 ~14%、 MI355X 的 ~7.5%）， 所以每笔交易里「花 FLOPs」那一侧更便宜了。
- **CP 经常缩小甚至消失。** 在 80 GB 上 128K 那个逼着你切序列的不等式， 松了 2.4–3.6 倍。 很多上下文长度下你可以完全不用 CP， 这就少了一个并行维度和一类 collective。 recipe 里的「把 CP 设到 16K–32K 本地窗口」变成了「只在真溢出时才设 CP」。
- **FSDP2 的省显存动机弱了， 但它的组合价值还在。** 你也许可以切到更少的 rank 上、 把更多 optimizer state 留在本地， 把 HBM 腾出来给更长的序列、 或者把 rollout 引擎和训练放在一起。 那个和 EP mesh 干净组合的 `DTensor` 仍然值。
- **那些 collective 招数需要重新 profile， 而不是重新 port。** 这里每一次 all-to-all 和 all-gather 走的都是 **RCCL， 不是 NCCL**， 节点内走 Infinity Fabric / xGMI 而不是 NVLink / NVSwitch。 chunked-EP overlap 依赖 collective 和 GEMM 占用各自独立的 stream； 这个重叠在 ROCm 上是否真的发生， 是个经验问题。 而且 RCCL 的拓扑和 NVSwitch 的平坦 bisection 不同， 所以实际的通信占比可能*更高*， 这意味着 overlap 帮助*更大*， 而不是更小。 用 rocprof 重新 profile chunk size； 别照搬 NVIDIA 的 tile 配置。

linear CE 那一招还有一个 kernel 层面的注意点： AMD 的 wavefront 是 **64 宽**， 不是 32。 每个 block 的 token tile 和 softmax 归约应该按 64 lane 的 wave 和 LDS 来设。 直接照搬 NVIDIA 的 tile 配置会 under-occupy。 Liger-Kernel 已经有 fused linear CE 的 ROCm Triton 路径了， 但要对着一个成对的 torch baseline 验证数值： running-max 修正是承重的， 一个跳过它的 tile 实现在 bf16 下会溢出。

## 9. 我会在 ROCm 上怎么做

如果让我在 MI300X 上用 slime + Megatron 把这套立起来（大致就是我[之前论证过的框架选择](/zh/blog/nemo-rl-vs-slime/)）， 操作顺序跟的是 roofline， 不是论文：

1. **先不上 CP。** 有 192 GB， 先看看在完整重计算 + linear CE 下、 不切序列能塞下多长的本地序列。 干掉一个并行维度， 比任何 kernel 微调都值。
2. **先 port linear CE， 按 64 宽 wave 调。** 它是单项最大的显存收益， 也最硬件可移植， 而且 LM head 是唯一一个显存真正卡你的地方。
3. **在 EP mesh 上把 FSDP2 拉起来， 对着 baseline 验证组合。** 逐参数 `DTensor` + 自定义 MoE all-to-all 在 ROCm 上的坑比 CUDA 多。 这里什么都别信、 都做 A/B， 而这恰恰就是 bumblebee primitive 边界设计的用处。
4. **最后在 RCCL 上重新 profile chunked EP overlap。** 它是对互联最敏感的一块。 用 hipEvents 确认 collective 和 GEMM 真的在并发跑， 并盯着 RCCL kernel 别从 expert GEMM 那里偷 CU。

更深的教训不是这些 kernel 里的任何一个。 它是那个立场： 在 RL post-training 里， 稀缺资源是工程鲁棒性和迭代速度， 不是峰值 FLOPs。 所以你花那个充裕的东西（compute）去买那个稀缺的东西（memory、 简单、 一个总能启动的配置）。 在 AMD 上， 这本账在这个方向上更失衡。 更多 HBM、 更多空闲 compute、 一个更年轻的 collective stack。 这是一个恰好奖励这种「花 FLOPs、 简化拓扑」思路的局面 —— 前提是你重新测量， 而不是假设 NVIDIA 的数字能照搬过来。

*该署名的署名： 原始分析和所有实测数字都是 [Yan Bai](https://iseekyan.github.io/zh/posts/qwen35-long-sequence-moe-rl/) 的。 第一性原理的重新推导、 算术核对、 以及 AMD 映射是我的。*
