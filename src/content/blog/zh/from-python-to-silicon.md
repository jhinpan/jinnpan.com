---
title: "从 Python 到硅片 · 给 ML 工程师的编译器与体系结构小科普"
description: "你可以把生产级 ML 系统写好几年， 都不知道 IR、 MLIR、 LLVM、 ISA、 FFI 这些词指什么。 这一篇是补丁 —— 写给本科 CS 念过、 但是 Compiler 和 Computer Arch 没好好上过的 ML 工程师。 配一份带 6 张 SVG 图、 中英双语的 HTML 深读。"
date: 2026-05-26
tags: ["compiler", "computer-architecture", "MLIR", "LLVM", "ISA", "FFI", "MLSys", "primer"]
category: "Technical"
lang: "zh"
---

你可以把生产级 ML 系统写好几年， 都不太知道 IR 是什么。 你 import 一个 `torch`， 调一个 `torch.matmul`， GPU 亮一下， 几毫秒后另一头就出来一个 tensor。 你的 Python 文件和那个 tensor 之间， 至少夹着八层软件、 三种 IR、 两套编译器框架、 一份设备驱动、 和一颗讲着指令集（由某个在 Bangalore 或 Sunnyvale 的人花了两年定义的指令集）的芯片。

大多数时候你不需要知道这些。 但偶尔会有什么从下面漏出来 —— 一个 kernel 比预期慢、 一个新加速器没 backend、 一个为 CUDA 12.4 编的 wheel 在 CUDA 12.6 装不上 —— 抽象就不再免费。 这一篇是给那种"漏过一次"才意识到自己讲不出这些零件名字的 ML 工程师的科普。

**完整 HTML 深读（带 6 张手绘 SVG、 EN / ZH 切换）：** [/sources/from-python-to-silicon.html](/sources/from-python-to-silicon.html)

## 五个词， 三十秒说清楚

**IR （Intermediate Representation， 中间表示）** —— 编译器拿来推理用的数据结构。 比源码更好分析（没有语法糖、 没有隐式转换）， 比机器码更有结构（有变量名、 有类型、 有控制流）。 LLVM IR 是 IR， MLIR 里的 `linalg.matmul` 是 IR， PyTorch 的 FX Graph 也是 IR。 这个词被严重重载， 上下文里要先问"谁的 IR"。

**MLIR （Multi-Level IR）** —— 一个用来造 IR 的框架。 核心抽象是 *dialect*： 一组带名字的 op、 type、 attribute。 标准发行版里有几十个 dialect， 从最高层的 `linalg`（tensor 算子）一路到最低层的 `llvm`（LLVM IR 的镜像）。 MLIR 之后大部分新 ML 编译器（Triton、 FlyDSL、 Torch-MLIR、 IREE、 OpenXLA）共享同一份 parser、 verifier、 pass manager —— 基建不再是税。

**LLVM** —— 史上最成功的编译器基建项目之一， 现在覆盖几乎所有非微软 MSVC 的世界。 它做三件事： 一种 IR（LLVM IR）、 一组后端（x86、 ARM、 RISC-V、 AMDGPU、 NVPTX、 Wasm、 SPIR-V 等等）、 一个 optimizer。 Rust、 Swift、 Julia、 Zig、 现代 Fortran、 Numba、 每个 GPU kernel DSL —— 都在用 LLVM。

**ISA （Instruction Set Architecture， 指令集架构）** —— 芯片对外的契约。 规定哪些指令合法、 每条做什么、 寄存器是什么、 内存模型如何。 对 ML 来说， ISA 现在比以往任何时候都重要 —— 性能几乎完全取决于编译器是否打到芯片的专门矩阵指令： Intel AMX、 ARM SME、 NVIDIA Tensor Core (HMMA)、 AMD MFMA (CDNA) / WMMA (RDNA)。 同一片硅 50 TFLOPS 和 500 TFLOPS 的差别就在这里。

**FFI （Foreign Function Interface， 外部函数接口）** —— 一种语言写的程序调用另一种语言函数的机制。 PyTorch 里你写 Python， 底下走 pybind11 进 C++， 再走 HIP / CUDA driver 到 GPU。 每道桥都有自己的 ABI、 数据表示约定、 生命周期管理。 Apache TVM-FFI 是当前最干净的"为 ML kernel 提供稳定 C ABI"的例子 —— 一个 kernel 库发一个 wheel， PyTorch / JAX / CuPy / Paddle 都能装得上。

## 把这五个词串起来的那一段

> 每一层都把程序从"好写"改写成"好执行"。 IR 是编译器做这些改写时用的数据结构。 MLIR 是一个用来在多层造 IR 的框架。 LLVM 是一种具体的 IR 加上一整套打磨多年的 backend。 ISA 是芯片的规范 —— backend 必须吐出符合它的指令。 FFI 是上下两层跨过语言和运行时边界对话的方式。 其余都是工程细节。

## 完整 HTML 里有什么

10 章 + 6 张手绘 SVG plate， 美学定为"工程物理专著"（米黄羊皮纸 + 酒红 + 森林绿 + 黄铜）：

- **Plate I · 纵向栈全景** —— 从 `torch.matmul` 到晶体管翻转， 9 层一张图
- **Plate II · SSA 是什么** —— 同一个 Python 函数变成 SSA 控制流图
- **Plate III · MLIR dialect 高塔** —— linalg / affine / scf / memref / gpu / rocdl / llvm 一栋楼七层
- **Plate IV · LLVM 中心辐射** —— 8 个前端 → LLVM IR → 8 个 target
- **Plate V · 各厂矩阵指令对比** —— Intel AMX、 ARM SME、 NVIDIA HMMA、 AMD MFMA、 AMD WMMA、 RISC-V Matrix
- **Plate VI · FFI 桥** —— Python / C++ / GPU runtime / kernel 四界之间的三道桥

还有一节是"一个 matmul 一路下去"， 把一个 FP16 4096 立方的 `torch.matmul(A, B)` 在 AMD MI300X 上从 Python 入口追到 wavefront 上的 MFMA 指令发完。 把前面学的词全部花掉。

最后是"礁石"那一节， 收集了几个在对话里反复让人困住的混淆点 —— "IR" 是哪个 IR、 LLVM IR ≠ MLIR、 PTX 是虚拟的 SASS 才是真实的、 ISA ≠ ABI、 "Tensor Core" 是品牌不是品类、 ML 里"编译器"至少指三种东西、 AOT vs JIT。

## 为什么写这个

写一篇科普的最难处不是材料多， 是要在"对的人"那里讲对。 这一篇我想交付的是： 下次你跑一个 ML workload， 看到它没打到预期的指令、 或者一个 wheel 装不上、 或者一个新硬件没 backend 的时候， 你能立刻给那件事起一个名字。 而 debug 这件事大半就是命名。

完整阅读： [/sources/from-python-to-silicon.html](/sources/from-python-to-silicon.html) （配 EN / ZH 切换， 直接 share 给同事都能读）。

---

*配套： 这个 [Source Reading 系列](/blog/source-reading-006-flydsl/) 里其他几集精读了这个栈里的具体仓库 —— SkyPilot（编排）、 SGLang / vLLM（推理引擎）、 mini-SGLang（教学版）、 gcnasm（AMD CDNA3 汇编）、 FlyDSL（layout 代数 Python DSL）。 每一集都是 Plate I 中某一格的"动手版"。*
