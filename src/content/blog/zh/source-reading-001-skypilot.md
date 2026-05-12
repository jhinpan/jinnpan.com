---
title: "源码精读 001 — SkyPilot， 21 万行的多云编排"
description: "六个半小时把 SkyPilot 源码读穿。 三区架构、 DP+ILP 混合优化器、 9-stage 执行流水线，以及加一个新云后端要改哪些文件。"
date: 2026-05-12
tags: ["source-reading", "infrastructure", "MLSys", "skypilot"]
category: "Technical"
lang: "zh"
---

这是源码精读三部曲的第一篇。 我花了大约 6.5 小时读完 SkyPilot， 产出了一份带 7 张手绘 SVG 图的完整 HTML 深度分析。 这篇博客是开胃菜； 完整阅读在 [/sources/skypilot.html](/sources/skypilot.html)。

## 为什么读这个 repo

SkyPilot 是"把 AI 工作负载跑在任意云上"的开源事实标准。 一条 `sky launch foo.yaml`， 接 24 个云厂商， 自动 provisioning、 spot 恢复、 模型 serving 全套。 我读它的原因是手头正在做 AMD 的 multi-agent kernel optimization 系统， 而"统一控制平面接管异构计算"这件事最近的工程先例就在这个 codebase 里。

## 五条值得带走的发现

**1. 这个仓库是个三区分布式系统， 不是单一二进制。** Zone A 是客户端（CLI + SDK）， Zone B 是 FastAPI API server（每个 launch 请求 spawn 一个子进程）， Zone C 是 provisioned 计算节点（每个节点跑一个 skylet 守护进程， SkyPilot 版本的 kubelet）。 一条 `sky launch` 跨越 9 个 Python 进程边界。

**2. `sky/cli.py` 是一个 10 行的薄壳。** 真正 7,954 行的 CLI 在 `sky/client/cli/command.py`。 顶层 `sky/__init__.py` 用了门面模式 —— 每个动词 API（`sky.launch`、 `sky.exec`、 ...）都从 `sky.client.sdk` 重导出， 让用户调用感觉是本地函数， 实现却跨进程跨网络。

**3. Optimizer 是 DP + ILP 混合。** Chain DAG（最常见情况， 单个 task 有多种资源候选）走动态规划， 复杂度 `O(N · R²)`。 一般 DAG 走 PuLP/CBC 的 ILP， 把双线性的 `c[u] ⊗ c[v]` 代价项用 McCormick 线性化打平。 这是教科书级工程实践 —— 不同问题结构用不同算法。

**4. 9-stage 流水线是 `CLONE_DISK → OPTIMIZE → PROVISION → SYNC_WORKDIR → SYNC_FILE_MOUNTS → SETUP → PRE_EXEC → EXEC → DOWN`。** 每个 stage 对应一个 `backend.method()` 调用。 微妙之处： `OPTIMIZE` 在 per-cluster 锁<em>之外</em>跑（因为耗时不能阻塞别的请求）， 但带一个 `planner` callback 给 backend， 锁内如果发现缓存决策过期可以重新跑。 乐观决策 + 锁内兜底。

**5. `cloud_vm_ray_backend.py` 里的 "Ray" 不是分布式训练 Ray。** 是借了 [ray.io](https://ray.io) 的 cluster launcher 来管 VM 生命周期。 SkyPilot 早期是 Ray 的扩展， 名字延续下来。 `sky/skylet/ray_patches/` 子目录可以证实 —— 那是他们对 Ray cluster launcher 的私有 patch。

## ★ 我从这次精读里偷走的一个设计模式

> 整个仓库有<strong>四个</strong>不同的 `state.py` —— `skylet/job_lib.py`（节点级）、 `jobs/state.py`（managed jobs 全局）、 `serve/serve_state.py`（serving）、 `server/state.py`（API server）。 四层状态机分级， <strong>把底层瞬态故障和顶层策略决策隔离开</strong>。 一次 spot 抢占导致节点级状态 RUNNING → FAILED → RUNNING， 但 managed-job 顶层状态全程保持 RUNNING。 这种"分层状态机吸收抖动"的模式直接可以搬到 multi-agent 系统 —— worker 偶尔失败， agent 任务级状态应该稳定。

## 完整阅读里有什么

7 张手绘 SVG 大图： 整体架构、 launch 时序、 9-stage 流水线时间线、 DAG → DP/ILP 路由决策、 skylet 解剖、 managed jobs 的两层状态机、 以及"如果要加 AMD ROCm 云后端要改哪些文件"的文件级地图。 五个新人易踩的坑， 三个贯穿全文的红线问题。

**→ 完整深度阅读： [/sources/skypilot.html](/sources/skypilot.html)**

---

*视觉方向定为"1962 年航空工程手册 × 当代建筑事务所年报"—— 深海军蓝底色、 骨白正文、 铁锈和黄铜点缀。 所有图都是手写 SVG（不用 Mermaid， 零运行时依赖）。*

*下一篇： [源码精读 002 — SGLang](/blog/source-reading-002-sglang)（一个 LLM 推理引擎， 同时也是个四进程分布式系统）。*
