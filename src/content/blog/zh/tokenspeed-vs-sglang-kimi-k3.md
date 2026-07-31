---
title: "用 TokenSpeed 在 MI355X 上服务 Kimi-K3： 为什么更快的 kernel 反而输了"
description: "在 8x MI355X 的 ROCm 容器里从源码把 TokenSpeed 编出来服务 Kimi-K3， 再跟 SGLang 逐个 kernel 对着 profile。 TokenSpeed 的 Gluon kernel 每一块都赢——MLA decode 快 2.77x， MoE routing 快 6.5x——但端到端仍然更慢。 四个构建的坑、 吞吐数字， 以及 trace 到底说了什么。"
date: 2026-07-31
tags: ["benchmark", "TokenSpeed", "SGLang", "Kimi-K3", "MoE", "AMD", "GPU-kernels", "inference", "MLSys"]
category: "Technical"
lang: "zh"
---

TokenSpeed 是 LightSeek 的推理引擎， MIT 协议， 2026 年 5 月开源。 它把 Kimi-K3 on AMD gfx950 当成一等公民： 有 `Dockerfile.amd`， MoE 和 MLA decode 都有原生 Gluon kernel， 还有跑在 8x MI35x runner 上的 per-commit CI。 我们手上正好有同一台机器、 同一个模型、 已经 profile 得很透的 SGLang 部署， 那自然就想问： 这两者到底差多少——不只是 tokens per second， 而是逐个 kernel 地看。

结论的短版本是： TokenSpeed 的 kernel 确实更快。 它的 Gluon MLA decode 比 SGLang 的 Triton attention 快 **2.77x**， 它融合过的 router 比 AITER 那条 routing 流水线快 **6.5x**， 每个 decode step 的总 device compute 低 **1.21x**， 而且 kernel launch 次数还少 30%。 然后它端到端在 batch 1 上慢 **1.14x**， 在并发 8 上慢 **3.7x**， 在 8K 输入、 并发 128 这档上直接活不下来。

「kernel 更快」和「引擎更慢」之间的这个缺口才是整篇文章的主题， 而它出现的位置跟我原本预期的完全不一样。

## 1. 实验环境

一台机器， 8 张 AMD Instinct MI355X（gfx950， 每张 288 GiB）， ROCm 7.2。 模型是 `moonshotai/Kimi-K3`——2.78 T 总参数、 105.4 B 激活、 93 层里 **69 层是 KDA 线性注意力、 24 层是 full NoPE-MLA**、 896 个 routed expert 走 top-16 外加 2 个 shared、 `situ` 激活， routed expert 以 `compressed-tensors` MXFP4（4-bit， group size 32）存储， 其余部分都是 BF16。

两套栈各跑各自推荐的配置， 所以这是 best-config 对 best-config， 而不是受控 A/B：

| | TokenSpeed | SGLang |
| --- | --- | --- |
| MoE 并行 | TP8 + EP8 | 纯 TP8 |
| Attention backend | `mla`（Gluon gfx950） | `triton` |
| MoE backend | `auto` → Gluon SiTU | AITER（`AITER_SITUV2_A8W4=1`） |
| KV cache | fp8 | bf16 |
| 调度器 | FlatKV（C++ FSM） | radix， 本次关闭 |
| 前缀缓存 | 关 | 关 |

fp8 KV cache 值得单独点出来， 因为它是*对 TokenSpeed 有利*的——每个 token 更省带宽——所以它解释不了后面任何一处劣势。

> **一个会悄无声息毁掉整个对比的坑。** 两边的 benchmark harness 都源自 vLLM 的 `benchmark_serving.py`， 输出的指标名字一模一样， 但 `--random-range-ratio` 的语义是反的。 SGLang 的 `1` 表示精确长度； TokenSpeed 的取值域是 $[0, 1)$， `0` 才表示精确长度。 两边都传 1 的话， 就是拿精确长度去跟一个直接报错的配置比。 下文所有数字两边都用精确长度。

## 2. 从 `git clone` 到服务起来， 中间有四个坑

没有一个是稀奇古怪的问题。 四个都是必须绕过去的， 而官方文档里的 recipe 会踩中其中三个。

### 2.1 一个从未发布过的依赖被硬钉住

`main` 上的 `python/pyproject.toml` 把 `tokenspeed-mooncake` 钉在 `>=0.3.12.post20260725`。 这个包在 PyPI 上根本不存在——不是版本旧， 而是一个 release 都没有——于是 `pip install ./python` 直接死在 `from versions: none`。

Mooncake 是 prefill / decode 分离用的 KV 传输引擎。 它的 import 全是惰性的， 藏在 `runtime/pd/` 和 `runtime/cache/storage/mooncake_store/` 里， 而一个用 `--disable-kvstore` 起的单机聚合式服务永远走不到那儿。 解法就是把依赖列表里那一条摘掉再装。 值得记住的是： 仓库 `main` 已经跑在 PyPI 能满足的范围之前了。

### 2.2 xgrammar 会悄悄把 CUDA 版 triton 盖到 ROCm 版上

这个坑很隐蔽。 `torch 2.11.0+rocm7.2` 要求 `triton-rocm==3.6.0`， 而 `xgrammar` 声明了一个裸的 `triton` 依赖。 两个发行包装进的是*同一个* `triton/` 目录， 于是 pip 会心安理得地把 PyPI 上的 `triton 3.7.1` 装到 ROCm 版之上， 谁后落地谁赢：

```bash
$ python -c "import triton; print(triton.__version__)"
3.7.1          # PyPI 的通用构建， 不是 torch 要的那个 ROCm 版
```

全程不报错。 你只是在跑一个并非为你的平台构建的 Triton。 解法是等其它都装完之后， 卸掉 `triton` 再重装 `triton-rocm==3.6.0`。

有意思的是 Gluon gfx950 那批 kernel 对此完全免疫， 因为 `tokenspeed-kernel-amd` 把所有 Triton 符号都收敛到了一个间接层：

```python
# tokenspeed_kernel_amd/_triton.py
import tokenspeed_triton as triton
import tokenspeed_triton.experimental.gluon.language as gl
from tokenspeed_triton.experimental.gluon.language.amd.cdna4 import (
    async_copy as cdna4_async_copy,
)
```

`tokenspeed_triton` 是一个独立的模块名（版本 3.8.10）， 永远不会跟 `triton/` 撞车。 正是这一层间接让 vendored 的 Gluon 路径能在 `triton` 被装坏的情况下照常工作——这个设计决策在别人的依赖求解器第一次干蠢事的时候就回本了。

### 2.3 文档里的 AMD recipe 一启动就 OOM

严格按 `docs/recipes/models.md` 来， 服务会把 96 个 shard 全部加载完、 decode graph 也捕获完， 然后在第 751 秒死掉。 凶手是 prefill graph 捕获， 显存是一个 bucket 一个 bucket 被抽干的：

```
Capturing prefill buckets (bucket=2048 avail_mem=19.92 GB):   2%
Capturing prefill buckets (bucket=1024 avail_mem=14.64 GB):  22%
Capturing prefill buckets (bucket= 704 avail_mem= 4.65 GB):  35%
Capturing prefill buckets (bucket= 512 avail_mem= 0.03 GB):  42%
[FATAL ERROR]: HIP failure: 'out of memory'
```

权重和 KV pool 占完之后每张卡只剩约 20 GB， 而把 40 个 prefill bucket 全捕获完需要的比这多。 解法是加 `--disable-prefill-graph`——TokenSpeed 自己的 MI35x CI perf 配置本来就带这个 flag， 只是文档的 recipe 漏了。 （当时仓库 HEAD 的那个 commit， 恰好就是在修 prefill warmup 的 OOM。）

### 2.4 它没法跟 SGLang 共用一个 Python 环境

TokenSpeed 硬钉 `torch==2.11.0` 和 `transformers==5.12.0`。 而容器里跑的是 SGLang 依赖的定制 `torch 2.9.1+rocm7.2.0`。 这两者没有调和余地； TokenSpeed 只能进自己的 venv， 带自己那个 6.2 GB 的 torch wheel。 磁盘和下载时间要预留出来。

这四个处理掉之后， 服务大约 11 分钟启动完成（时间几乎全花在读 1.5 TB 权重上）， 报告 `max_total_num_tokens=4466304`， 然后正常服务。

## 3. 端到端吞吐

下面所有数字都是精确长度的 random 负载、 `temperature 0`、 `ignore_eos`、 关前缀缓存， 在这台 8 卡机上实测。 SGLang 出现两次： `nospec` 是同类对比， 因为 TokenSpeed 这边没开投机解码； `DSpark` 是 SGLang 在这台机器上调优后的最好成绩。

| 负载（ISL/OSL） | 并发 | TokenSpeed | SGLang nospec | SGLang DSpark |
| --- | --- | --- | --- | --- |
| 1024 / 1024 | 1 | 44.73 | 51.40 | 109.84 |
| 1024 / 1024 | 8 | 84.97 | 311.79 | 472.00 |
| 1024 / 1024 | 32 | 226.80 | 847.87 | 949.73 |
| 1024 / 1024 | 64 | 440.58 | — | — |
| 1024 / 1024 | 128 | 487.19 | — | — |
| 4096 / 1024 | 1 | 43.98 | — | 89.30 |
| 8192 / 1024 | 1 | 44.55 | — | 74.29 |
| 8192 / 1024 | 128 | **失败** | 890.25 | — |

单位是输出 tokens/s， 8 卡整机聚合。

单流的表现是拿得出手的： 44.73 对 SGLang 的 51.40， 而且 TokenSpeed 的 TPOT 从 1 K 到 8 K 输入一直稳在 21.5–22.2 ms； 相比之下 DSpark 的优势会随输入变长而衰减， 接受长度从 2.51 掉到 2.34。 真正崩掉的是并发： 在 8 和 32 上都落后 3.7x， 而且 TokenSpeed 全场最好的成绩（并发 128 时的 487 tok/s）仍然低于 SGLang 并发 32 时的 848。

> **这不是我们把它装坏了。** TokenSpeed 自己针对这个负载的 CI 参考值——`test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml`——是 8x MI350X 上每用户 43.05 tok/s， 门槛设在 42。 我们在 MI355X 上实测 43.98， 比他们自己的数字还高 2%。 并发上出的问题， 不是我们这边配错了。

更能说明问题的是那个 CI 文件 gate 了什么。 TokenSpeed 的 NVIDIA perf 配置都带着并发 1、 2、 4、 8、 16 的完整参考曲线； 而 AMD 的 K3 配置只有一行 `perf_reference: {1: [42, 5.2]}`， 并且把服务端钉死在 `--max-num-seqs 1` 和 `--cudagraph-capture-sizes 1`。 上游在 AMD 上根本没有测过批量吞吐， 所以也没有任何东西能防止它退化。

### 3.1 8K 负载在并发 128 上不是变慢， 是卡死

这正是 SGLang 在这台机器上重点调优的负载： 输出 890 tok/s、 总吞吐 8012 tok/s。 TokenSpeed **一小时内只完成了 128 个请求中的 1 个**。 这是结构性失败， 不是慢：

- FlatKV 页池占用打到 94%
- 调度器打印了 **9,256** 次 `flat retract ... to unwedge the pool`
- 解码塌到 0.02 tok/s， 106 个请求卡在队列里

容量的算术能解释这一切。 引擎对外报告 `max_total_num_tokens=3556992`， 但调度器自己的配置写的是 `num_device_pages=3298`、 `block_size=128`， 而且这个页池要被四个 cache group 共享——`full_attention` 加三个 `linear_attention` 组， 因为 K3 的 93 层里有 69 层是 KDA， 每层都要自己的 paged recurrent state。 128 个各 9,216 token 的请求把这个池子远远超订， 于是调度器不是优雅排队而是持续抖动。 SGLang 在同一负载上报告的是 129 万 token 的池子， 零 retract。

有一点*不是*限制： 文档 AMD recipe 里写的 `--max-model-len 8192`。 改成 16384 能正常启动。

## 4. 逐个 kernel 看

现在到有意思的部分。 我们手上已经有一份 SGLang 在这台机器上 batch 1 解码 K3 的完整 PyTorch profiler kernel 分解， 所以我把 TokenSpeed 对应的 trace 抓下来直接对比。

**两边对齐的方法学。** Batch 1、 ISL 4096、 TP8、 无投机解码、 **关 graph**——在被 replay 的 HIP graph 内部， 单个 kernel 是无法单独归因的， SGLang 那份 baseline 也正是出于同样原因用 graph-off 抓的。 两边都是把每个 decode step 的 device kernel 时长求和， 并且**排除 collective**： 在 eager 解码下 all-reduce 的 payload 只有几 KB， kernel 是在忙等， 它的时长量的是 rank 之间的偏斜而不是工作量。

TokenSpeed 自带的 Proton profiler 在这里用不了——它在 ROCm 上会调 `rocprofiler_force_configure`， 而这个调用必须发生在 HIP 初始化之前， 所以挂到一个已经跑起来的服务上会以 error 16 失败。 torch / roctracer 那条路可以正常挂载， 而且顺带的好处是： 它产出的 chrome trace 跟 SGLang baseline 是同一种产物。

kernel 到 block 的归因是按**发起它的源码模块**做的， 不是靠名字匹配。 有一个 kernel 跨 block： `_kimi3_projection_gemv_kernel` 同时支撑每个 MoE 层里的两个 latent H↔L projection、 shared expert 的 down projection 和 router（4 × 92 = 368 次 dispatch）， 外加 `KimiLinearKDA` 里的 `kimi3_qkvfab_projection`（69 次）， 合计 **437——与实测 dispatch 数完全吻合**。 它的时间按 dispatch 占比拆分。 那个 93 次 dispatch 的 vendor GEMM 是 attention 的输出投影， 按 69 / 24 拆给两种注意力。

### 4.1 每个 decode step 的 block 构成

| Block | TokenSpeed（µs） | SGLang（µs） | 结论 |
| --- | --- | --- | --- |
| MoE | 8085 | 10549 | TokenSpeed 快 1.30x |
| KDA 线性注意力 | 2905 | 3542 | TokenSpeed 快 1.22x |
| Full MLA 注意力 | 1468 | 3100 | TokenSpeed 快 2.11x |
| Attention residual | 1865 | 2288 | TokenSpeed 快 1.23x |
| 胶水（norm、 add、 cast、 copy） | 2786 | 1290 | TokenSpeed **慢** 2.16x |
| **device compute 合计** | **17108** | **20769** | **TokenSpeed 快 1.21x** |

两套栈对每个 block 内部 misc 的归类方式略有差异， 所以 block 那几行当作指示性数字看， 合计那行才是硬的。 下面这些是映射毫无歧义的 kernel 组， 也是所有关键论断的依据：

| Kernel 组 | SGLang（µs） | TokenSpeed（µs） | 比值 |
| --- | --- | --- | --- |
| MLA attention 本体， 24 层 | 1212 | 438 | **TokenSpeed 快 2.77x** |
| MoE routing： 896 选 16 加排序 | 2956 | 454 | **TokenSpeed 快 6.51x** |
| Attention residual 收尾 | 2288 | 1853 | TokenSpeed 快 1.23x |
| KDA 递推 + short conv， 69 层 | 744 | 650 | TokenSpeed 快 1.14x |
| Routed expert 量化 GEMM | 1437 | 2391 | **TokenSpeed 慢 1.66x** |

### 4.2 到底谁赢在哪里， 为什么

**Gluon MLA decode 是最干净的一场胜利。** SGLang 把 MLA decode 做成两阶段的 split-KV Triton attention： `_fwd_grouped_kernel_stage1` 扫 KV cache， `_fwd_kernel_stage2` 归约各个 split， 每层 50.5 µs。 TokenSpeed 的 `_mla_decode_gluon` 加上它的 softmax / reduce-V 伙伴， 同样的活儿每层 18.3 µs。 在 batch 1、 4 K 上下文这个点上 attention 是纯带宽问题， 手写的 CDNA4 Gluon kernel 就是比一个通用 split-K Triton 模板更会流式读 KV cache。 这是「手写 Gluon 胜过 Triton」唯一一处无可争议成立的地方。

**绝对值最大的一场胜利在 routing， 而且它是设计选择而非 kernel 质量。** 从 896 个 expert 里选 top-16 并为 grouped GEMM 排好 token 顺序， SGLang 要花 2.96 ms/step， 摊在一串 AITER kernel 上——`grouped_topk`、 分多个 phase 的 `opus_moe_sorting`、 `fused_mx_quant_moe_sort`、 `moe_reduction`。 TokenSpeed 用一个融合的 `_kimi3_sigmoid_bias_topk_kernel` 花 0.45 ms 做完。 单这一个 kernel 就值 2.5 ms/step， 比整个端到端的差距还大。 在 896 个 expert 的规模下， routing 的元数据已经大到「在多个 kernel 之间把它物化出来」比「把它算出来」还贵， 而把 sigmoid、 bias 和 top-k 融进一趟就把这些往返全省了。 这件事跟 AMD 一点关系都没有——它是那种能迁移到任何平台的赢法。

**Expert GEMM 是 AITER 赢。** SGLang 的 `mfma_moe1` / `mfma_moe2` MXFP4 kernel 跑完 routed expert 用 1.44 ms， TokenSpeed 的 Gluon `_stage1_a16w4_situ_warp_gemv` 加 `_stage2_a16w4_warp_gemv_combine` 用 2.39 ms。 这一条要小心读： TokenSpeed 跑 EP8， 每个 rank 拥有 112 个完整 expert， batch 1 时只处理自己 token 路由到的那么一两个； SGLang 跑纯 TP8， 每个 rank 都要做全部 16 个 expert， 但只做 1/8 的宽度。 这是不同的 shape， 所以这个比较被并行策略污染了， 不能单纯归因于 kernel 质量。 另外 AITER 那边是 `a8w4`（8-bit 激活）， Gluon 是 `a16w4`， AITER 搬的激活字节数只有一半。

**KDA 基本打平， 而且两边都是 Triton。** TokenSpeed 解码期的 KDA 走的是 `ops/attention/triton/kda.py`——尽管文档里说的是「原生 AMD KDA 实现」， batch 1 解码的递推在两套栈上都是 Triton kernel， 差距在 14% 以内。 这跟之前那个结论是自洽的： KDA 在 batch 1 上是 occupancy-bound， 既不是带宽也不是算力受限——69 层固定大小的 recurrent state 更新， 只填满了 256 个 CU 里的 48 个。 换什么 kernel 语言都救不了这个， 只有加大 batch 能救。

**而 TokenSpeed 在胶水上输掉 1.5 ms。** RMSNorm、 `_add3`、 fp8 copy kernel、 `Memcpy DtoD`、 一个通用的 `_rowcta_gemv_kernel`——合计 2.79 ms， 对 SGLang 的 1.29 ms。 SGLang 把更多这类东西融掉了， 最显眼的是 `add_rmsnorm_quant_kernel`， 一趟里同时做完残差加、 norm 和量化。

### 4.3 真正重要的那个发现

把三个测量并排放在一起：

| 指标 | SGLang | TokenSpeed |
| --- | --- | --- |
| 每 decode step 的 device compute | 20769 µs | **17108 µs** |
| 每 decode step 的 kernel launch 数 | 3372 | **2375** |
| 每 decode step 的 collective dispatch 数 | 187 | 187 |
| 端到端 step（开 graph） | **19.39 ms** | 22.19 ms |

TokenSpeed 干的 GPU 活更少、 launch 更少、 collective 数量一样——然后更慢。 两个最省事的解释都被测量排除掉了： 不是 launch overhead， 因为它的 launch 少 30%； 也不是 collective 数量， 因为两边都恰好是每 step 187 次， 每个 layer block 一次。

那些毫秒是花在 kernel *之间*的， 不是 kernel 内部。 那就是调度器、 Python 执行平面、 每 20 ms 要喂饱一个 93 层混合模型的主机侧路径， 以及 collective 延迟里有多少是暴露出来而没有被 overlap 掉的。 在并发 1 上这笔开销值 14%； 在并发 8 上同一类开销值 3.7x——这正是一个「不随 batch 增大而摊薄的 per-step 成本」应该表现出来的样子。

这把 AMD 上的差距整个重新定义了。 kernel 不是问题所在——它们可测量地是这个引擎更强的那一半。 TokenSpeed 在 gfx950 上有的是一个穿着 kernel 问题外衣的 runtime 问题。

## 5. 我会带走的几条结论

**如果现在要在 MI355X 上给 K3 选引擎：** 选 SGLang， 而且在并发上差得不近。 TokenSpeed 目前只在单流延迟上有竞争力。

**如果在做 TokenSpeed 的 AMD 路径：** kernel 是领先的。 profiling 的力气该花在调度器和主机侧路径上， 而第一件要修的事是上游 CI 在 AMD 上完全没有测并发——现在那里的吞吐退化是不可见的。

**如果在做 SGLang 的 AMD 路径：** 那个融合 router 是白放在桌上的 2.5 ms/step， 比 trace 里任何其它单项差距都大， 而且是纯软件收益、 不依赖任何硬件特性。 Gluon MLA decode kernel 还能再拿 0.8 ms。

**把专有名词删掉之后剩下的那条：** 当一个引擎更慢时， 本能是去 profile kernel， 而且答案通常确实在 kernel 里。 这次 trace 连着两次给出相反的答案——kernel 更快、 launch 更少、 collective 一样多、 引擎更慢——而这件事之所以能被看见， 唯一的原因是两边在同一台机器、 同一天、 用同一套方法学被测量。 一个止步于 tokens per second 的 benchmark 会得出「TokenSpeed 的 AMD kernel 需要优化」这个结论， 而这恰好是反的。

## 6. 复现

装进独立 venv， 顺序不能乱， 因为 `tokenspeed-kernel` 在自己的 native build 阶段会去解析 AMD 包：

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip "setuptools<82" wheel cmake ninja
pip install --index-url https://download.pytorch.org/whl/rocm7.2 torch==2.11.0+rocm7.2

pip install --force-reinstall --no-deps ./tokenspeed-kernel-amd --no-build-isolation
TOKENSPEED_KERNEL_BACKEND=rocm pip install ./tokenspeed-kernel/python/ --no-build-isolation
pip install ./tokenspeed-scheduler/ \
  --config-settings=cmake.define.TOKENSPEED_FLAT_KVCACHE=ON   # K3 只能跑 FlatKV
# 先装 ./python 的依赖（去掉 tokenspeed-mooncake）， 然后：
pip install -e ./python --no-build-isolation --no-deps
pip uninstall -y triton && pip install --force-reinstall --no-deps \
  triton-rocm==3.6.0 --index-url https://download.pytorch.org/whl/rocm7.2
```

起服务， 带上相对文档 recipe 的两处偏离：

```bash
tokenspeed serve moonshotai/Kimi-K3 --served-model-name kimi-k3 --trust-remote-code \
  --max-model-len 8192 --kv-cache-dtype fp8 --tensor-parallel-size 8 \
  --mm-encoder-tp-mode data --enable-expert-parallel \
  --attention-backend mla --moe-backend auto \
  --gpu-memory-utilization 0.92 --max-num-seqs 32 --disable-kvstore \
  --disable-prefill-graph --no-enable-prefix-caching \
  --host 127.0.0.1 --port 8100
```

用精确长度跑 benchmark——注意那个 `0`：

```bash
tokenspeed bench serve --base-url http://127.0.0.1:8100 --model kimi-k3 \
  --tokenizer /path/to/Kimi-K3 --dataset-name random \
  --input-len 4096 --output-len 1024 --random-range-ratio 0 \
  --num-prompts 3 --max-concurrency 1 --num-warmups 1 \
  --ignore-eos --extra-body '{"temperature": 0}'
```

要抓 trace 的话加 `--enforce-eager`， 然后用 GPU activities（不是 Proton）武装 profiler， 再驱动一个请求穿过去：

```bash
curl -sS -X POST http://127.0.0.1:8100/start_profile \
  -H 'Content-Type: application/json' \
  -d '{"num_steps": 16, "activities": ["GPU"],
       "profile_by_stage": true, "profile_id": "kimi-k3-tp8"}'
```

每个 rank 会写出 `<id>-TP<r>-{EXTEND,DECODE}.trace.json.gz`。 按 step 求和 device kernel 时长， 去掉 collective， 剩下的按发起它的模块归因。
