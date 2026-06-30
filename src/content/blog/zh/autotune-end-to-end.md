---
title: "Autotune 全流程 · 从 Triton 到 FlyDSL"
description: "kernel autotune 在 Triton 里怎么设计、 在 aiter / quack / CuteDSL 里怎么用、 又怎么被 SGLang 这类推理引擎消费 —— 然后从这一切出发, 给 FlyDSL 设计一条真正的 autotune 路径。 配一份带 6 张手绘 SVG 图的 HTML 深读。"
date: 2026-06-30
tags: ["autotune", "GPU-kernels", "Triton", "CuTeDSL", "FlyDSL", "SGLang", "MLSys", "primer"]
category: "Technical"
lang: "zh"
---

一个 GPU kernel 有一些旋钮, 拧它们不改变算出来的*结果*, 只改变快慢 —— tile 大小、 warp 数、 软件流水线级数、 寄存器上限、 cluster 宽度。 这些旋钮的最优值不是常数。 它取决于输入形状 (GEMM 在 `M=1` 和 `M=8192` 想要的 tile 完全不同)、 取决于 dtype、 取决于具体那块 GPU。 你没法从数据手册上读出最优配置; 你必须把 kernel 跑一遍、 测出来。 这个"测量再挑选"的过程就是 autotune, 每个现代 kernel DSL 都有它的某种版本。

这篇 guide 从源码精读六个真实实现 —— [Triton](https://github.com/triton-lang/triton) (大家都在抄的范本)、 [quack](https://github.com/Dao-AILab/quack) 和 [CuteDSL](https://github.com/NVIDIA/cutlass) (两个在这想法上扩展的 DSL)、 [aiter](https://github.com/ROCm/aiter) (kernel 库的离线 tuning)、 [SGLang](https://github.com/sgl-project/sglang) (一个*消费*已 tune 配置的推理引擎)、 以及 [FlyDSL](https://github.com/ROCm/FlyDSL) (有机制但没人用) —— 用它们回答一个问题: **FlyDSL 的 autotune 到底应该怎么设计?**

> 完整阅读: [/sources/autotune-end-to-end.html](/sources/autotune-end-to-end.html) —— 一份中英双语的 HTML 深读, 带 6 张手绘 SVG 图 (autotune 循环、 原地污染、 在线 vs 离线时间线、 cache-key 解剖、 FlyDSL 架构提案、 排好序的实现步骤)。

## 每个人共享的循环

剥掉框架差异, 每个 autotuner 都是同一个四步循环, 外面套一层缓存。 先定义一个候选配置的**搜索空间**; 每个候选**编译并 benchmark**; 挑出**最快的**; 把赢家按一个 key **缓存**起来, 这样下次同样形状的调用就跳过搜索。 有意思的工程全在边角: key 里放什么、 怎么在能省的时候省掉搜索、 怎么让 benchmark 不污染状态、 以及循环是在线跑 (首次调用时) 还是离线跑 (提前)。

## 一条铁律: 原地 kernel 会污染搜索

autotune 会把*同一个 kernel* 跑几十遍来挑最快的。 如果这个 kernel 原地写输出、 或者做累加, 这些重复运行会污染数据, 你测到的就是垃圾 —— 你选出来的"最快"配置可能只是漂移得最快的那个。 这个选择是错的, 而且 kernel 自己的单元测试一个都抓不到。

Triton 的解法是装饰器上的两个钩子: `reset_to_zero` (每遍之前把指定张量清零) 和 `restore_value` (把指定张量存档, 每遍之后还原)。 两者都在每次被 benchmark 的调用前后跑, 这样每个配置都是在同样干净的输入上测的。 这是 autotune 正确性的灵魂, 也是草率移植最容易忘的一件事。

## Triton — 范本

Triton 的 `@triton.autotune` ([runtime/autotuner.py](https://github.com/triton-lang/triton/blob/main/python/triton/runtime/autotuner.py), 511 行) 是 quack 和 FlyDSL 共同继承的设计。 里面有三样东西把真正的 autotuner 和玩具区分开, 三样都值得偷:

- **`@heuristics` —— 跳过搜索的那扇门。** 如果一个参数能从输入*算出来* (比如 block size 就该取大于 `N` 的下一个 2 的幂), 就推导它而不是枚举。 最便宜的搜索是你根本没跑的那次。
- **`prune_configs_by` —— 编译前先砍。** 一个 `perf_model` + `top_k` 和一个 `early_config_prune` 都在任何编译*之前*跑, 这样你只为有机会的候选付 benchmark 的成本。
- **cache key。** Triton 的磁盘缓存 key 是五样东西的 SHA-256 —— `triton_key()` (编译器版本)、 backend 目标 hash、 `fn.cache_key` (kernel 源码)、 失效性 env、 shape/dtype。 每个维度都回答"我什么时候该把缓存的配置扔掉?"

## quack — 双轨配置

[quack](https://github.com/Dao-AILab/quack) 是 Tri Dao 基于 CuteDSL 的 kernel 库。 它的 autotuner ([quack/autotuner.py](https://github.com/Dao-AILab/quack/blob/main/quack/autotuner.py), 849 行) 是最经得起生产考验的参考, 而且不像 FlyDSL, 它真的被采用了 —— `rmsnorm` 和 `gemm` 都戴着 `@autotune`。 它加了三样:

- **双轨配置。** 每个 kernel 的 `*_config.py` 同时暴露一个解析式的 `get_default(N, dtype, arch)` (零搜索启发式) 和 `get_all_fwd_configs()` (穷举空间)。 日常走启发式; 只有你显式 autotune 时才付搜索成本。 这正是"全有或全无"型 autotuner 的解药。
- **带源码指纹的多层缓存。** 内存 dict → 磁盘 `.o` → 每个 `quack/*.py` 加 CUTLASS/Python 版本的 SHA-256 (任何源码改动都失效) → 可选的结果缓存。 tuning key 还把 stride 规整成 `{0, 1, other}`。
- **并行预编译。** `_precompile` 用子进程池并行编译所有候选、 在 benchmark 之前焐热缓存 —— 首跑慢这个问题的实用答案。

## CuteDSL — autotune_jit

CuteDSL 是 quack 构建其上的那一层。 它的 `autotune_jit(params_dict, update_on_change)` 直接接受参数网格, 首次调用时遍历笛卡尔积, 按 tuning key 缓存最优 kernel。 两个细节可以搬到 FlyDSL: 编译才是成本 (所以每个配置 `cute.compile()` 一次、 复用 executor —— 正是 FlyDSL 的 `build_*_module` 处境), 以及隐式 JIT 缓存的 key 是源码*加上所有 DSL 环境变量*的哈希 —— 和 Triton 编码的"env 是 key 的一部分"是同一课。

## aiter 和 SGLang — 离线, 以及消费者

[aiter](https://github.com/ROCm/aiter) 是相反的范式: tuning 是**离线的、 入库的**。 一个 tuner 读未 tune 的形状列表 (`*_untuned_gemm.csv`), 扫配置, 把赢家提交到 `*_tuned_gemm.csv`; 服务时从 CSV 读、 没有搜索, 一个 CI 守卫防止配置退化。 值得注意的是 aiter 已经自带一条 FlyDSL 后端的 tuning 路径, 在 `aiter/ops/flydsl/gemm_tune/`。

[SGLang](https://github.com/sgl-project/sglang) 不是 kernel DSL —— 它是一个*消费*已 tune 配置的推理引擎, 它展示了离线产物在规模上长什么样。 它的 fused-MoE 配置是入库的 JSON, *文件名*就是查找 key (`E=8,N=14336,device_name=...,dtype=...json`), 放在按 triton 版本分的目录下, NVIDIA 的旁边就是 AMD `gfx950` 变体。 运行时 `get_moe_configs` 按设备 + 形状查、 miss 时 fallback 到 `get_default_config`。 如果 FlyDSL 的 kernel 要被一个引擎 tune-then-consume, 这就是离线产物应该瞄准的契约。

一个微妙点: 训练和推理对首跑成本的容忍度不同。 训练把一次性的启动 autotune 摊薄到几百万步上, 所以 JIT tuning 没问题。 推理服务对延迟敏感, 每个新形状的首个请求慢就是看得见的退化 —— 这正是引擎倚重入库的离线配置 + 默认兜底的原因。

## 设计 FlyDSL 的 autotune

FlyDSL 已经有一个 autotuner —— [python/flydsl/autotune.py](https://github.com/ROCm/FlyDSL/blob/main/python/flydsl/autotune.py), 296 行: 一个 `Config`、 一个 `Autotuner`、 一个磁盘缓存、 一个 `_make_key`、 一个 `_prune`、 和一个公开的 `@autotune`。 问题不是它缺失; 而是**没有 kernel 用它** —— 构建良好的死代码。 把它对着五个参考读, 差距很清楚:

| 能力 | FlyDSL 今天 | 目标 |
|---|---|---|
| 采用 | 无 | 1 kernel + 离线配置 |
| 免搜路径 | 无 | 启发式默认 |
| 原地正确性 | 只有 `reset_to_zero` | + `restore_value` 存档 |
| cache key | 只有 shape/dtype | + arch + 源码指纹 + 编译器 + env + stride |
| 离线产物 | 无 | 设备+形状 配置文件 |
| CI 守卫 | 无 | 无 GPU 单测 + 回归门 |

目标架构是**一个开关引出三条轨**: 启发式默认给零成本日常运行、 离线 设备+形状 查表给服务、 在线搜索 (带 `restore_value`) 给开发和长尾 —— 三者都写同一个按上面完整 key 索引的缓存, 在线搜索还能产出离线配置文件。

## 实现 guide — 排好序

1. **加固 key, 加上 `restore_value`。** 在任何 kernel 采用之前先修两个正确性缺口: 扩展 `_make_key` (当前只有 shape/dtype), 加上 arch、 源码指纹、 编译器/版本、 env; 并在已有的 `reset_to_zero` 之外加上 `restore_value`。 没有这个, autotune fused-add rmsnorm 就是错的。
2. **给一个 kernel 做双轨配置。** 挑 softmax 或 rmsnorm (空间小而有界)。 给它 `get_default(N, dtype, arch)` 和 `get_all_configs()`, quack 风格。 普通调用者免费拿启发式; autotune 是 opt-in。
3. **离线产出 + 查表 (SGLang 风格)。** 让在线搜索写一个按 设备 + 形状 + dtype 建 key 的配置文件; 一个关掉 tuning 的服务运行查它、 miss 时 fallback 到默认。
4. **CI 守卫, 先做无 GPU 的。** 单测覆盖 `Config` 序列化、 cache-key 构造、 剪枝 (不需要硬件), 然后一个 opt-in 的 GPU 集成测试, 然后一个仿 aiter 的 tuned-config 回归门。
5. **(可选) 并行预编译**, 如果在线搜索太慢。

如果你为一个决定读这篇: 不要从加配置开始做 FlyDSL 的 autotune。 从让它*正确*且*不用它也便宜*开始 —— 完整的 cache key 和 `restore_value`、 加一个启发式默认。 FlyDSL 已经有那个循环了; 它需要的是边角。 这些步骤对应 FlyDSL 的 issue [#770](https://github.com/ROCm/FlyDSL/issues/770) (autotune)、 [#769](https://github.com/ROCm/FlyDSL/issues/769) (rmsnorm bwd, 它也需要 tuning)、 和 [#749](https://github.com/ROCm/FlyDSL/issues/749) (下游消费)。
