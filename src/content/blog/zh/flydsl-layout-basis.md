---
title: "FlyDSL 笔记 · Layout 之下的 BasisAttr"
description: "上一篇 FlyDSL 精读把 layout 代数收在了五个词上， 这篇补的是更深一层的子概念 —— Fly_Basis 类型和 BasisAttr。 是什么、 为什么 layout 需要、 以及 mentor 给我的'完善它'任务从哪条线进入最干净。"
date: 2026-05-26
tags: ["flydsl", "MLIR", "AMD", "layout", "kernel-optimization"]
category: "Technical"
lang: "zh"
---

[上一篇 FlyDSL 源码精读](/blog/source-reading-006-flydsl/) 写完之后， 我 mentor 让我"基于对 layout 的熟悉去完善一下 basisType"。 我才意识到自己虽然能把 layout 代数解释给别人听， 但对 layout *之下* 的那层东西其实没真懂。 这篇是补丁。

## Layout 的三层直觉， 补一下

精读那篇的 § M1 把 layout 收在了五个词上： shape、 stride、 layout、 divide、 slice， 解释成"coord tuple 到 linear index 的函数"也是对的。 但如果要去改 layout 系统的基础设施， 这五个词不够， 还要再补三层直觉：

**(a) Shape 和 stride 是两个独立的自由度。** Shape 决定定义域； stride 决定"每一维移动一格在内存里跨多远"。 同一个 `shape = (4, 8)`， 配上 `(1, 4)` 是列主、 配上 `(8, 1)` 是行主、 配上 `(0, 1)` 是行广播、 配上 `(1E0, 1E1)` —— 这最后一个就是 BasisAttr 的世界。

**(b) Layout 不是数据， 是一段"地址生成函数"。** FlyDSL kernel 里你不会写 `A[i, j]`， 你看到的是 `logical_divide` / `partition_S` / `slice`。 这是因为 layout 在 MLIR 里是一等公民 —— `!fly.layout<(8,16):(1,8)>` 是一个真实的类型。 pass 管线第 3 阶段 `fly-layout-lowering` 才把符号 layout 折叠成 `i*1 + j*8` 这样的地址算术。

**(c) Layout 代数是函数复合的代数。** `composition(A, B)(x) = A(B(x))`； `logical_divide(A, tiler)` 是把 A 拆成 "tile-内 layout × tile-间 grid layout"； `make_layout_tv(thr, val)` 是构造一个 `(thread_id, value_id) → tile coord` 的映射。 每个操作都在改造这个映射函数本身， 不是在搬数据。

## BasisAttr 到底是什么

文档里几乎没提它。 定义在 `include/flydsl/Dialect/Fly/IR/FlyAttrDefs.td:112`：

```cpp
def Fly_BasisAttr : Fly_Attr<"Basis", "basis", [...]> {
  let parameters = (ins
    Fly_IntAttr:$value,
    ArrayRefParameter<"int32_t">:$modes
  );
}
```

打印格式是 `value E mode0 E mode1 ...`， 所以你会看到 `1E0`、 `1E1`、 `2E0E1`。 这里的 E 来自标准基向量的 e_0、 e_1 那个 e。

**展开形式** （见 `lib/Dialect/Fly/Utils/IntTupleUtils.cpp:745` 的 `intTupleBasis2Tuple`）：

- `1E0` → `(1)` —— 1-tuple 第 0 位是 1
- `1E1` → `(0, 1)` —— 2-tuple 第 1 位是 1
- `1E2` → `(0, 0, 1)` —— 3-tuple 第 2 位是 1
- `2E0E1` → `((0, 2))` —— 嵌套 tuple， path `[0][1]` 处是 2

所以 BasisAttr 就是 **一个稀疏的"标准基向量"表示**： 用 `(value, modes)` 紧凑编码一个绝大部分都是 0、 只在指定 path 上有 `value` 的（可嵌套）IntTuple。

## 为什么 layout 需要它

普通 layout 的 stride 是标量， 把 coord 映射到 **flat int**： `(i, j)` → `i + 4*j`。 但有时候你需要 stride 保留**结构**， 让映射变成 coord → coord tuple。

最直接的场景是 identity layout。 `make_identity_layout((M, N))` 产出的 stride 是 `(1E0, 1E1)`， 完整类型是 `!fly.layout<(4, 8) : (1E0, 1E1)>`（见 `tests/mlir/LayoutAlgebra/construction.mlir:89`）。 把 `(i, j)` 喂进这个 layout， 得到的不是 flat int 而是 `(i, j)` 自己。 这只能用 basis stride 写出来。

它在 `make_identity_layout`、 `right_inverse`、 `coord_swizzle` 这些需要保留 coord 结构的运算里反复出现。 `tests/mlir/LayoutAlgebra/coord_swizzle.mlir` 整个文件几乎只用 `(1E0, 1E1, 1E2)` 这种 basis stride。

类型层面有两个东西， 容易混：

- `Fly_BasisAttr` —— MLIR attribute， 藏在 `IntTupleAttr` 的叶子里
- `Fly_Basis` —— MLIR type（`FlyTypeDefs.td:10`）， 把 attribute 包成一等公民类型

Mentor 说的"basisType"严格来说是后者， 但实际改起来两个往往一起动。

## "完善 BasisAttr"可以做什么

把现在的 BasisAttr 算子表面跟 IntAttr 对一下， 缺口非常明显：

| 算子 | IntAttr | BasisAttr |
|---|---|---|
| `+`， `-` | ✓ | ✗ |
| `*` | ✓ | ✓ （只有 × IntAttr） |
| `/`， `%` | ✓ | ✗ |
| `<`， `<=`， `>`， `>=` | ✓ | ✗ |
| `intMin`， `intMax` | ✓ | ✗ |
| `intShapeDiv` | ✓ | ✗ |
| `isStaticValue(v)` | ✓ | ✗ （只有 `isStatic`） |

更关键的是 `lib/Dialect/Fly/Utils/IntTupleUtils.cpp:105-121` 的 `IntTupleBuilder<IntTupleAttr>::add`： 凡是遇到 basis， 它先调 `intTupleBasis2Tuple` 把 basis 展开成 dense tuple 再做加法。 这意味着 layout lowering 之后产生的 IR 比理论上能产生的更"胖"。

四个可能的方向：

1. **算子补全** —— `lib/Dialect/Fly/Utils/IntUtils.cpp:277-304` 加上 `+`、 `-`、 `/`、 `%`、 `<` 等。 改动最局部， 有现成的 IntAttr 参照可循， 容易写 unit test。
2. **避免不必要的 dense 展开** —— 在 `IntTupleBuilder` 上加 basis-aware fast path， 让 layout 代数在 basis 上不退化。
3. **`Fly_Basis` type 上的 API 补全** —— 现在 type 只有 `depth()` 和 `isStatic()`（`FlyTypeDefs.td:10-25`）， 可能要把 `getValue()`、 `getModes()` 直接暴露到 type 层， 减少 callsite 的样板。
4. **Layout 代数里的 basis-aware 简化** —— composition / divide / product 在 basis stride 下的平凡化。

我倾向先做 (1)， 因为它边界最清楚， 也可以直接用 `tests/mlir/LayoutAlgebra/` 下现成的 lit 测验。 (2) 是真正能提升 IR 质量的事， 但要先有 (1) 才有意义。

## 写在最后

写这篇之前我以为我懂 layout 了。 写完才发现， 你能把一个抽象解释给别人听， 跟你能去改这个抽象的实现， 不是同一件事。 后者要求你顺着这个抽象往下走一层， 看到那些它"假装不存在"的细节 —— 比如 BasisAttr。

完整的 FlyDSL 系统级精读还是在 [/sources/flydsl.html](/sources/flydsl.html)。
