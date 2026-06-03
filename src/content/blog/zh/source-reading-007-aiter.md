---
title: "源码精读 007 — AITER MoE Tuner， fused MoE 背后的 dispatch board"
description: "解读 AITER 的 gemm_moe_tune.py： 一个 4259 行脚本如何把 MoE shape 展开成 ASM、 CK、 CK-Tile、 FlyDSL 候选任务， 再写成 production fused_moe config。"
date: 2026-06-03
tags: ["source-reading", "MLSys", "AMD", "ROCm", "MoE", "kernel-optimization"]
category: "Technical"
lang: "zh"
---

源码精读 007 写 AITER， 但不是泛泛读整个 repo。 目标是一份很密的文件： `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py`。 它有 4259 行， 很容易被误读成一个 benchmark 脚本。 其实不是。 它更像一个调参实验的 dispatch-board compiler： 读 MoE shape， 展开成候选 kernel task， 交给共享的多进程 executor， 和 reference 比正确性， 补上端到端路径里缺失的 cost， 最后写出 production `fused_moe` 能消费的配置行。

完整 HTML 深读在： [/sources/aiter.html](/sources/aiter.html)。

## 为什么读这个文件

MoE tuning 不是“选最快 GEMM”这么简单。 production 路径背后有几层隐藏 contract：

- routing / sorting contract： top-k token-expert pair 必须被整理成 expert-local block；
- quantization contract： per-token fp8、 per-1x128 blockscale、 per-1x32 MXFP4、 int4 路径各自有不同 scale layout；
- inter-stage contract： stage1 可能输出 bf16、 fp8 或 fp4 activation， 取决于 quant / cast 有没有 fused；
- backend contract： ASM、 CK、 CK-Tile、 FlyDSL 支持的 dtype 和 activation 矩阵并不相同；
- dispatch contract： 最终的 `tuned_fmoe.csv` 行会被 production `fused_moe` 读取， 不是只服务 tuner 自己。

这个脚本存在的原因， 就是这些 contract 不能只靠默认相信。 它把这些 contract 显式化到足够能测试。

## 主抽象： task tuple

真正撑住脚本的是传给 `mp_tuner` 的 task tuple。 每条 backend 路径最终都会被规整成类似这样的形态：

```text
(tag,
 generate_data, generate_args,
 candidate_func, candidate_args,
 reference_func, reference_args,
 tolerances, optional_compare_fn)
```

这就是为什么一个 executor 可以同时跑 CK stage1、 CK stage2、 CK-Tile A8W4、 FlyDSL FP4、 FlyDSL int4 和 ASM 1-stage 候选。 每个候选都说明： 怎么生成数据， 跑哪个函数， 从生成出来的 dictionary 里取哪些 key， 以及哪个 reference 定义正确性。

到了这一步， tuner 就变成一个表格问题： 收集 `(info, us, err)` 行， 按 shape 分组， 剔除非法候选， 在 `block_m` 匹配时组合 stage1 和 stage2， 把 1-stage 候选作为备选加入， 补上 fairness cost， 最后选总时间最低的一行。

## 我会优先修的点

`calculate()` 里有一个很具体的可疑点。 它先解包了 `stage`， 然后在 stage-specific FLOP / BW 分支之前又把它重置成空字符串：

```python
key, stage, kernelName, block_m, us, err = results
...
stage = ""
if stage == "stage1":
    ...
elif stage == "stage2":
    ...
```

这意味着 stage1 / stage2 的 TFLOPS 和 bandwidth 报告会落到 combined estimate。 它不会改变 winner， 因为选型靠的是 `us`， 但会让 per-stage 派生指标误导人。 在这个点修好之前， 我会优先相信 timing、 correctness 和最终 config 字段， 而不是 TFLOPS / BW。

## 最短心智模型

可以把 `gemm_moe_tune.py` 读成一个调参实验编译器：

- 输入语言： `untuned_fmoe.csv`；
- IR： task tuple；
- runtime： `mp_tuner`；
- optimizer： `post_process()`；
- 输出 artifact： `tuned_fmoe.csv`。

一旦用这个模型去读， 4259 行长文件就不再是一整坨脚本。 它变成五个 contract： shape、 quantization、 task、 fairness、 production dispatch。

**→ 完整深度阅读： [/sources/aiter.html](/sources/aiter.html)**。
