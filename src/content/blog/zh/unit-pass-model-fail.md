---
title: "单测通过， 模型出错： GPU kernel 为什么会在端到端里坏掉"
description: "为什么一个 GPU kernel 能通过所有单测， 却在端到端的模型运行里出错： 测试和模型之间的七个缺口， 每个都用 gfx1250 上线期间和 FlyDSL v0.3 到 v0.4 期间的公开 bug 说明， 以及一套逐级关闭这些缺口的验证阶梯。"
date: 2026-10-13
tags: ["GPU-kernels", "validation", "testing", "inference", "gfx1250", "FlyDSL", "primer"]
category: "Technical"
lang: "zh"
---

一个 kernel PR 通常带着两样证据： 一个通过的单测， 一张加速比表格。 两样都是真的。 然后这个 kernel 进入模型， 接下来会发生三种情况之一。 模型开始输出乱码： 在 gfx1250 上， 有一个服务器正常加载、 正常服务， 却对每个 token 都输出 「!」， GSM8K 得分 0.0， 日志里没有任何报错（[ATOM #1750](https://github.com/ROCm/ATOM/pull/1750)）。 多卡运行挂住， 或者正常结束却返回陈旧的数据。 或者一周都相安无事， 直到某个请求走进一条没人试过的路径。 维护 kernel 库的人会把它总结成一句话： *单测过了， 跑模型出问题。*

这不是某一个 bug， 而是测试覆盖范围上的缺口。 单测只在少数几个点上检查 kernel： 几个形状、 随机输入、 一张卡、 一个 stream、 一个新进程。 模型运行则是一个大得多的空间上的分布。 测试只能证明它检查过的点正确， 而模型几乎所有时间都在别处。

本文的案例来自缺口最大的地方： 新架构。 每个案例都发生在 2026 年 6 月到 10 月之间： 这几个月里， 各个 kernel 库在为 gfx1250（AMD MI450 系列的架构） 移植和上线 kernel， FlyDSL 则从 [v0.3 路线图](https://github.com/ROCm/FlyDSL/issues/632)（其重点之一是 「完整支持 gfx1250， 并做端到端验证」） 走到了 10 月 14 日发布的 [v0.4.0](https://github.com/ROCm/FlyDSL/releases/tag/v0.4.0)。

> 完整阅读： [/sources/unit-pass-model-fail.html](/sources/unit-pass-model-fail.html) 是一份中英双语的 HTML 版本， 带 7 张手绘图和一个可以筛选的 32 个案例的索引。 2026-10-14 更新： 案例改为来自 gfx1250 上线期间和 FlyDSL v0.3 到 v0.4 期间。

## 七个维度

先从任何 kernel 都成立的事实出发。 kernel 的输出不只取决于参数的值：

1. **形状**： 决定哪些代码路径和 tile 尾块会执行。
2. **数值**： 决定会碰到哪些数值区间。
3. **布局、 dtype 和硬件契约**： 生产方写了什么， 机器又保证了什么。
4. **执行环境**： graph 捕获、 stream、 缓存、 buffer、 进程的生命周期。
5. **时序**： 哪个 wave 先到。
6. **规模**： 有几张 GPU 参与， 以及用什么分区模式。
7. **软件和硬件**： 工具链、 构建、 版本， 以及实际运行的那份副本。

单测之所以便宜， 是因为它把大多数维度固定在方便的值上： 小的、 按 tile 对齐的形状， `torch.randn` 生成的输入， 测试自己的辅助函数生成的布局， 默认 stream 上的 eager 模式， 调用一次， 一张卡， 仓库自己的构建。 对快速测试来说， 这是正确的取舍。 代价是每个被固定的值都藏起了一个 kernel 行为会改变的区间。

## 为什么 gfx1250 让所有缺口同时变大

kernel 里写死了它所针对的那台机器的事实： 一个 wave 有几个 lane、 有哪些矩阵形状、 数据怎样搬进 LDS、 barrier 等待什么、 一个 cache 位是什么意思。 它还写死了工具链和机群的事实。 目标一换， 每一条都会变成一个能编译、 在旧硬件上能通过、 到新硬件上却出错的地方。 gfx1250 一次改变了其中大部分：

| 变化 | 移植过来的代码的假设 | gfx1250 的实际行为 | 来源 |
|---|---|---|---|
| 波宽 | 不是 RDNA， 所以是 64 lane | 32 lane， 但它不是 RDNA | [FlyDSL #1024](https://github.com/ROCm/FlyDSL/pull/1024)、 [SGLang #41687](https://github.com/sgl-project/sglang/pull/41687) |
| 矩阵 tile | launcher 接受的 tile 都会发出 MMA | FP8 WMMA 的 K 深 128； 64 深的 tile 一条都不发 | [FlyDSL #1264](https://github.com/ROCm/FlyDSL/pull/1264) |
| 拷贝进 LDS | buffer 直写 LDS 的 load 处处能 lower | LLVM 无法 lower 它； 改用 TDM 和 async load | [aiter #4987](https://github.com/ROCm/aiter/issues/4987) |
| barrier | 普通 barrier 就能发布 LDS 写入 | `s_barrier_signal` 之前没有 `s_wait_dscnt` | [aiter #6344](https://github.com/ROCm/aiter/issues/6344) |
| cache 位 | 值 17（sc0 sc1） 含义不变 | 17 表示 device scope 加 non-temporal | [FlyDSL #1214](https://github.com/ROCm/FlyDSL/pull/1214) |
| 编译器 | kernel 入口不会多出代码 | LLVM 在入口前加一段 prologue | [FlyDSL #1088](https://github.com/ROCm/FlyDSL/pull/1088) |
| async copy | 带 mask 的 load 仍有越界检查 | Triton 3.8 把它变成没有检查的 async load | [aiter #6081](https://github.com/ROCm/aiter/pull/6081)、 [SGLang #41868](https://github.com/sgl-project/sglang/pull/41868) |
| 分区 | 一张卡就是一个设备 | NPS2 / DPX： peer 之间需要 system scope 的顺序 | [aiter #4812](https://github.com/ROCm/aiter/pull/4812) |
| CI | 每个 PR 的 CI 都会跑这个 kernel | 默认 runner 只有 gfx942 和 gfx950 | [FlyDSL #1199](https://github.com/ROCm/FlyDSL/pull/1199)、 [#1264](https://github.com/ROCm/FlyDSL/pull/1264) |

还有两处结构性的变化让缺口进一步变大。 aiter 默认的 PR CI 只在 gfx942 和 gfx950 上跑算子测试， FlyDSL 的 gfx1250 PR 写着 「在 gfx1250 上测试（没有 gfx1250 CI runner）」， 所以正确性只能靠每个作者自己的机器。 机器之间也不一样： 在 [FlyDSL #1228](https://github.com/ROCm/FlyDSL/pull/1228) 里， 两台用同一个锁定 LLVM 的 gfx1250 机器都通过了同样的正确性测试， 但同一个 split-K fence 在一台上每个 workgroup 约 0.03 µs， 在另一台上约 0.3 µs。

## 缺口 1： 形状与尺寸

kernel 高速处理完整的 tile， 用 mask 或 clamp 处理尾块。 它用 32 位算术计算地址， 因为 64 位地址运算在每次读取时都要多花寄存器。 这两个选择在一个范围内都是对的， 超出范围就会悄无声息地出错。

32 位的选择有两道悬崖： 有符号索引的 2³¹， 字节偏移的 2³² 字节。 在 [aiter #6402](https://github.com/ROCm/aiter/issues/6402) 里， 默认配置在 gfx1250 上选中的 BF16 GEMM， 在 7067 × 151936（2,147,463,424 字节， 刚好低于 2³¹） 时是对的。 多一行， 第 7067 行之后的每一行就都不会被写； 8192 × 151936 返回了 332,463 个非有限值， 却没有任何报错。 同一个 issue 里的 FlyDSL kernel， 输出本身在 2³² 以内， 但最后一个 256 行 tile 的 202 行填充越过 2³²， 回绕后覆盖了第 0 到 201 行。 悬崖也可以等进程 「变老」 才出现： 在 [ATOM #2533](https://github.com/ROCm/ATOM/pull/2533) 里， int32 的 KV 行号乘以步长 576， 从第 3,728,270 行起回绕， 只有被 prefix caching 填满的池才会走到。 之后 Kimi-K3 每个 rank 的 decode 都输出 「!!!!」。

尾块也是这样悄悄出错的。 在 [FlyDSL #1199](https://github.com/ROCm/FlyDSL/pull/1199) 里， `K // tile_k` 在 K = 640 时丢掉了最后 128 列， 输出和完整 K 的参考实现相差 60%。 [FlyDSL #1230](https://github.com/ROCm/FlyDSL/pull/1230) 在 gpt-oss 的 K = 2880 上发现了同样的向下取整， 只有 64 宽的 K tile 是对的。 在 [aiter #6344](https://github.com/ROCm/aiter/issues/6344) 里， gfx1250 上的 RMSNorm 从不写一行的最后 N % 8 个元素。 在 [FlyDSL #1264](https://github.com/ROCm/FlyDSL/pull/1264) 里， 比 128 深的 FP8 WMMA 还小的 64 深 tile 一条 WMMA 都不发， 返回全零的 C。 未调优的形状会走回退配置： 在 [aiter #4987](https://github.com/ROCm/aiter/issues/4987) 里， gfx1250 上的 DeepSeek-V4-Flash 走进了 LLVM 无法 lower 的启发式 FlyDSL 回退 kernel， 服务器直接退出。

**能抓住它的测试：** 输出刚好低于和高于 2³¹、 2³² 字节， 并把最后一个 tile 的填充算进去； tile 不能整除的 K 和 N， 用真实宽度， 例如 K = 2880； launcher 接受的每一种 tile； 以及每一个没有调优表项的生产形状。

## 缺口 2： 真实数据

真实的 prompt 有短有长， 路由分数几乎相等， 训练过的 attention 很尖锐， bias 不是零。 在 [ATOM #1750](https://github.com/ROCm/ATOM/pull/1750) 里， gfx1250 上比一个 KDA chunk 短的 prompt 变成 NaN， NaN 的 logits 让 argmax 选中 token 0， 服务器一直输出 「!」， GSM8K 0.0。 在 [aiter #5801](https://github.com/ROCm/aiter/pull/5801) 里， gfx1250 的 TDM grouped GEMM 在 bias 指针里漏了 N-tile 编号； 修复后 gpt-oss-120b 的精度恢复， 评审要求补一个多 N-tile、 bias 互不相同的测试， 因为一样的 bias 会把它藏起来。

FlyDSL v0.3 到 v0.4 期间的案例也是同一个模式。 [FlyDSL #1213](https://github.com/ROCm/FlyDSL/pull/1213) 用真实的 GLM checkpoint 发现， 把 expert ID 打包进路由分数会在应选 226 时选成 92。 [FlyDSL #1020](https://github.com/ROCm/FlyDSL/pull/1020) 发现 FP8 softmax 下溢， 返回 0.637 而不是 1.0， 而 「随机初始化的模型不会」 暴露它。 在 [aiter #6127](https://github.com/ROCm/aiter/pull/6127) 里， 相同的预热 token 让 FlyDSL MegaMoE 的路由倾斜， 一个落后的 wave 读到被复用的 LDS 槽位， 随即出错。

**能抓住它的测试：** 回放抓取的 prompt、 路由、 bias 和权重； 覆盖每个 chunk 和 top-k 长度的两侧； 刻意加入相等、 几乎相等、 one-hot 路由、 不同的 bias 和 NaN。

## 缺口 3： 接口契约

kernel 相对一份契约正确， 在新架构上， 硬件也是契约的一部分。 FlyDSL 原来按 「是不是 RDNA」 推断波宽， 所以 gfx1250 一直被编成 wave64， 直到 v0.3.2 发布的 [FlyDSL #1024](https://github.com/ROCm/FlyDSL/pull/1024)。 下游副本还带着自己的常量： SGLang 用 FlyDSL 写的 norm kernel 在指令选择阶段失败（[SGLang #37751](https://github.com/sgl-project/sglang/pull/37751)）， [SGLang #41687](https://github.com/sgl-project/sglang/pull/41687) 里的 DSA top-k 只在比 top-k 长的行上返回错误索引。 [FlyDSL #1214](https://github.com/ROCm/FlyDSL/pull/1214) 指出， gfx950 的 cache-policy 值 17 在 gfx1250 上表示 device scope 加 non-temporal。

kernel 之间的契约是布局。 在 [aiter #5465](https://github.com/ROCm/aiter/pull/5465) 里， 量化 kernel 和 gfx1250 的 grouped MoE GEMM 各自推断 scale 布局， 在 expert 并行路径上不一致； 规则是只保留一种布局， 「由生产方选定并告诉消费方」。 在 [aiter #6360](https://github.com/ROCm/aiter/pull/6360) 里， pandas 合并把整数写成浮点， 878 个单元格解析失败， EP 行永远匹配不上， 而逐个读取 CSV 文件的运行一切正常。 在 [FlyDSL #1253](https://github.com/ROCm/FlyDSL/pull/1253) 里， 测试量化器把 E8M0 scale 四舍五入到最近值， 参考实现和 kernel 在 58% 到 98.5% 的 block 上不一致。

**能抓住它的测试：** 用真实的生产方喂每个 kernel； 在 kernel 入口断言编译出的波宽； 由生产方传下布局； 解析合并后的配置； 用 kernel 自己的规则量化参考实现。

## 缺口 4： 执行环境

graph 捕获会把 host 端的决定冻结下来： 在 [aiter #4494](https://github.com/ROCm/aiter/pull/4494) 里， 一个在 graph 之外清零的 split-K semaphore 让 Kimi-K3 在预热时挂住， 而 「没有测试同时覆盖 split-K 和捕获」。 生产环境会复用 buffer： aiter #6344 里没被写的 RMSNorm 尾部经过 `torch.empty_like` 返回的是最大到 1e38 的垃圾值， 所以 FlyDSL #1264 现在会在 N 之外填上 signaling NaN 的位模式并检查它。 缓存的 key 必须覆盖所有会改变代码的因素： FlyDSL 的 JIT key 忽略了 MegaMoE 的配置对象（aiter #6127）， 磁盘缓存忽略了 LLVM 版本（[FlyDSL #1088](https://github.com/ROCm/FlyDSL/pull/1088)）， 热缓存下有 22 个失败， 冷缓存下只有 2 个（[FlyDSL #1277](https://github.com/ROCm/FlyDSL/pull/1277)）。 在 [FlyDSL #1220](https://github.com/ROCm/FlyDSL/pull/1220) 里， 一个在一个 stream 上填输入、 在另一个 stream 上启动的测试， 大约每 100,000 次启动失败一次。

**能抓住它的测试：** 捕获并回放； 让生产方和启动在同一个 stream 上； 同一个进程里跑多种形状和热缓存； 输出预先填上哨兵值； 像长时间运行的服务器那样把状态填满。

## 缺口 5： 偶发竞争

在 gfx1250 上， 等待按计数器拆开， 普通的 barrier 不会等 LDS 写入完成。 在 [aiter #6344](https://github.com/ROCm/aiter/issues/6344) 里， RMSNorm 归约的每次 LDS 写入后面紧跟 `s_barrier_signal`， 中间没有 `s_wait_dscnt`， 于是宽行被放大约 1.156 倍或 1.41 倍： 4096 × 16384 时 30 次里 0 次， 4096 × 32768 时 30 次里 2 次， 8192 × 65536 时 20 次里 13 次。 [aiter #6348](https://github.com/ROCm/aiter/pull/6348) 加的 fence 让编译器生成了这个等待。 编译器也可能把等待放错： 在 [aiter #5418](https://github.com/ROCm/aiter/issues/5418) 里， 一条晚了一条指令的等待覆盖了还在进行的 async load； 在 [SGLang #41868](https://github.com/sgl-project/sglang/pull/41868) 里， 「回答看起来很流畅， 但是错的」。 FlyDSL kernel 里手写的等待在 [aiter #6170](https://github.com/ROCm/aiter/pull/6170) 和 [#6368](https://github.com/ROCm/aiter/pull/6368) 里被编译器的调度打破， 后者写明 「只跑一次的测试会通过」。

算术是这样的： 单次调用出错的概率是 p， N 次调用至少出错一次的概率是 1 − (1 − p)ᴺ。 30 次出错 2 次时， 几十次调用就够了。 十万分之一时， 要约 70,000 次调用才有一半的机会。 [aiter #4841](https://github.com/ROCm/aiter/pull/4841) 里是 292,889,600 次出错 7 次， 1,000 次调用的测试只有 0.002% 的机会抓到它。 这就是 [FlyDSL #1237](https://github.com/ROCm/FlyDSL/issues/1237) 的意义： 它的 LDS 检查器原型报出了全部 12 个注入的同步 bug， 而单次启动的输出检查漏掉了其中 8 个。

**能抓住它的测试：** 用同一个输入重复调用， 要求输出逐位一致； 扫最宽的行； 跑同步 sanitizer； 对照 ISA 检查等待计数； 给每个等待设上限。

## 缺口 6： 多卡

多卡协议出错时表现为挂起或陈旧读取。 在 [aiter #4812](https://github.com/ROCm/aiter/pull/4812) 里， gfx1250 上 relaxed 的 ready 标志让 peer 在拷贝对整个系统可见之前就读取输入： 128 个输出错 127 个， 而 「集合通信正常结束」。 只有 4 MiB 以上的 registered-buffer 路径用这个标志， 也只有输入不断变化的连续调用才会暴露它。 在 [aiter #6446](https://github.com/ROCm/aiter/pull/6446) 里， 按大小计算的槽位步长让 gfx950 上的 vLLM CI 挂起（有 rank 落后时 6 次全挂， 安静主机上 114 次一次都没有）； [aiter #5616](https://github.com/ROCm/aiter/pull/5616) 对 gfx1250 LL128 kernel 的评审发现了同样的规则， 并提醒 「两种大小总是落在同一个 bank 里， 可能漏掉这个 bug」。 在 [FlyDSL #1214](https://github.com/ROCm/FlyDSL/pull/1214) 里， gfx1250 的 TP > 1 路径 「能编译， 但从没运行过」。

**能抓住它的测试：** 每种分区模式下的 TP 2、 TP 4 和 EP， 每轮三种消息大小， 一次同步之前排队很多次集合通信， 有上限的等待和看门狗。

## 缺口 7： 没测到的软件

编译器会加你从没写过的代码： 一次 LLVM 版本升级能编译、 也通过了评审（[FlyDSL #1051](https://github.com/ROCm/FlyDSL/pull/1051)）， 而 [FlyDSL #1088](https://github.com/ROCm/FlyDSL/pull/1088) 把报告中 gfx1250 上每个 kernel 都出错的问题追到一段入口 prologue， 上游三周内三次重写了它。 开关可能根本到不了代码： FlyDSL v0.4.0 修好了从未进入代码生成的四个 LLVM 开关（[#1153](https://github.com/ROCm/FlyDSL/pull/1153)）、 从未到达硬件的 DAZ 开关（[#1177](https://github.com/ROCm/FlyDSL/pull/1177)）， 以及只进入了被丢弃的 code object 的编译提示（[#1195](https://github.com/ROCm/FlyDSL/pull/1195)）。

生产环境运行的还可能是别人的副本。 FlyDSL v0.3.0 移走 `buffer_ops` 之后， SGLang 用 FlyDSL 写的 norm kernel 悄悄退回慢路径， 「没有任何日志」（[SGLang #33142](https://github.com/sgl-project/sglang/issues/33142)）； 装上 FlyDSL 0.3.1 后， aiter 的 attention 导入在所有架构上都报错（[aiter #4701](https://github.com/ROCm/aiter/issues/4701)）。 对 [aiter #6348](https://github.com/ROCm/aiter/pull/6348) 的评审发现， 分派按构建目标而不是实际设备进行。 [aiter #5133](https://github.com/ROCm/aiter/issues/5133) 发现 141 个 pytest 函数从未执行， 其中 8 个在 gfx1250 的 FlyDSL 测试里， CI 却记为通过。

**能抓住它的测试：** 在 gfx1250 上测引擎的镜像和锁定的版本； 按实际设备分派； 把编译器版本放进每个缓存 key； 要求开关必须改变 code object； 给下游导入的 API 加契约测试； 一个用例都没收集到时直接失败。

## 验证阶梯

每一级都能抓到下一级抓不到的一类 bug， 成本也更高：

| 级别 | 关闭的缺口 | 要跑什么 |
|---|---|---|
| 1. 边界单测 | 形状 | 尾块、 含最后一个 tile 的 2³¹ / 2³²、 每种可接受的 tile、 未调优形状、 输出之外的哨兵值 |
| 2. 回放真实输入 | 数据 | 抓取的 prompt、 路由和 bias、 长短不同的行、 相等、 NaN |
| 3. 测接缝 | 契约 | 真实生产方 → 消费方、 断言波宽、 合并后的配置、 按 kernel 规则构建的参考实现 |
| 4. 生产执行模式 | 环境 | graph 回放、 旁路 stream、 热缓存、 复用的 buffer、 填满的池 |
| 5. 重复跑， 要求确定性 | 时序 | N 次逐位一致、 宽行、 sanitizer、 有上限的等待 |
| 6. 多卡协议 | 规模 | TP 2 / TP 4 / EP、 每种分区模式、 每轮三种大小、 看门狗 |
| 7. 下游与工具链 | 软件 | gfx1250、 引擎锁定的版本、 按实际设备分派、 必须改变代码的开关 |
| 8. 端到端模型 | 全部七个 | 在目标架构上做固定种子的精度评测（例如 GSM8K）、 吞吐； 看分数， 不看退出码 |

顶端的模型同时覆盖全部七个维度， 所以它是最后一级， 而不是唯一的一级。 在 gfx1250 上， 还要记下机器、 驱动和锁定的版本： 一台机器上的结果， 对下一台机器的说明力比看上去要弱。

## 结语

「单测过了， 跑模型出问题」 听起来像运气不好， 其实不是。 单测是样本， 模型是总体。 上面每一个公开 bug， 都待在测试固定不动的某个维度里： 第 2³¹ 字节之后的一行、 比一个 chunk 短的 prompt、 按错误的架构家族推断的波宽、 不等写入完成的 barrier、 编译器加的一段 prologue。 新架构会同时移动很多个维度。 说出是哪个维度， 缺的那个测试通常就一目了然。
