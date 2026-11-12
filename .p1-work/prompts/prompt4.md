# P1 v8 Controller 与 DP 工程深化执行 Prompt

你负责在 vLLM Runtime 中完成一项边界清楚、可验证、适合简历讲解的 KV reclamation
Controller 与 DP 路由增量。本轮是实施任务，不是只制定计划。先完整阅读适用的
`AGENTS.md`、[prompt3](prompt3.md)、[详细计划](../plans/engineering-controller-dp-final-plan.md)、
[status](../plans/status.md)、[engineering closeout](../reports/engineering-closeout.md)，
以及计划中列出的 Controller/DP 报告、协议和源码。先核对当前源码与工作树，再实施
尚未完成的目标；文档里的示例不是已经实现的接口或结果。

## 本轮目标

### 1. 有界单请求回收与最多双 victim

新增显式 opt-in 的 protected Controller，默认仍用 fixed。保留旧 fixed/demand/retry
模式的行为和历史结果，不为得到新结果改写旧报告。

protected Controller 至少有两个独立约束：单次删除不超过当前私有物理 KV 的 20%；
相对本次调度边界的逻辑已计算私有上下文至少保留 50%。以上只是第一版实验参数，
不是最优性结论或质量承诺。`retention` 保持原来按当前物理 E 计算的语义，禁止把它
悄悄重解释为逻辑累计比例。实现前确认 `E`、`L`、共享 prefix `P`、scheduled `q`、
block size 与 preempt/recompute 的实际源码语义；不把未运行的 prompt 或未来输出算入
逻辑下限。沿用现有 `min_keep`、sink/recent、共享页保护和 policy binding。

只在真实 allocation miss 触发。先选择能在保护边界内满足真实缺口 D 的单 victim；
若单 victim 不足，才允许两个 victim。组合需要先完整计算，优先减小多余释放页并让
两个请求回收后的逻辑保留率尽量平衡，再比较 zero-copy、复制字节和稳定顺序。超过
两个请求仍无法满足 D 时，沿原生等待/preemption 处理，不突破下限，不持续增加
headroom。精确按 block 边界计算释放页数。

多 victim 必须先形成完整 Scheduler 计划，然后再创建可能修改 hash/canonical state
的 intents。每个 victim 携带自己的 policy binding、source row/E 和 target keep；TP
rank 执行相同的 victim 集合、顺序和计划。全 rank GPU completion 及现有实际等待完成、
Scheduler 验证整批结果后，才一起 commit/free。验证页面真实复用及 Graph replay。
设备 mutation 后失败沿现有 fail-stop 规则暴露；不加静默 rollback、自动 recompute 或
catch-all fallback。

### 2. DP pressure-aware tie-break（有条件）

沿用现有 `known_work` 和 pending ADD watermark，不重造已有状态链。先确认
`DPLoadSnapshot.kv_cache_usage` 的生产、传播、快照年龄和实际含义，并用低扰动 trace
关联 work score、每副本 usage、入站 prompt、选路结果、后续本地 miss/reclaim/preempt，
以及 peer 是否仍有容量。

只有轨迹重复证明：有效新鲜快照下 work score 同分，当前路由反复选择较高 usage 副本，
随后发生本地 miss/抢占，而低 usage peer 有可服务余量，才实现 opt-in usage tie-break。
新信号只在 work score 相同时偏向较低 usage；不添加权重，不改变有明确 work score
差异的路由，不迁移活跃请求，不跨 replica 借页。没有进入证据就记录停止理由，并
继续完成 known_work 与 protected Controller 的组合验证。

### 3. 最终统一复测

按详细计划定义的范围，对简历主张所需的 correctness、micro 和 Serving 做一次最终
源码口径评估。正确性覆盖 CPU 边界、单/双 victim、TP2 ranks、continuation、completion、
commit/free/rewrite/reuse、Graph replay、DP pending ADD 和条件 tie-break。micro 分开
记录 Controller 决策、完整 reclaim、GPU compaction、victim/page/copy bytes 及路由开销。
Serving 使用固定压力 workload 比较 native/fixed/protected；DP2 使用现有 W1/W0 比较
requests/known_work，pressure tie-break 仅在条件进入后对比 known_work，并做一次必要的
组合场景验证。正式运行沿 prompt3 的预热、重复轮次、反向复核、输入和归档约束。

TTFT `<= 2s` 与 TPOT `<= 80ms` 保持为本项目 SLO，不放宽，也不称为行业通用标准。
逐请求同时满足两项才算通过；报告请求通过数、原有 goodput 口径、TTFT/TPOT 分布、
输出吞吐、preemption/recompute、每副本路由与 usage。小样本分位数只作描述，不宣称
统计显著性。不得用 micro 收益代替 Serving 结果。

不重跑废弃的预算/权重参数扫描，不扩大冻结的 85 条质量样本，不对有损策略声称质量
保持。现有 CB1 `0/24`、60 次抢占、DP 原始 DP4 `PARTIAL` 以及历史 FAIL/PARTIAL 均
保留。新结果单独归档并说明配置，不覆盖历史结论。

## 工程和执行要求

- 按 [prompt3](prompt3.md) 继承全部环境、依赖、GPU/CPU、run_guard、显存线、测试、
  实验归档和 Git 规则；它是外部环境与治理的 source of truth。本 Prompt 只定义本轮
  需求，不复制或修改 prompt3 的固定环境。
- 沿用既有 Worktree/branch。当前计划基线曾看到 `scheduler.py` 的用户未提交注释，
  开始时重新检查并原样保留所有用户差异；不 reset、amend、rebase、覆盖或把用户
  dirty 文件混入本轮 commit。
- GPU 只用物理 GPU1 做单卡；双卡 TP2/DP2 使用 GPU0、GPU1，禁止 GPU2。TP2/DP2
  正式验证使用现有 CPU20/21 affinity helper。所有 GPU 工作通过 `run_guard.py` 共用锁
  串行执行，不并发启动服务；合计显存达到 44 GiB 停止、45 GiB 硬上限。
- 模型固定 Qwen3-8B BF16，使用已有 Python/CUDA/PyTorch/Triton 环境。不得升级依赖、
  重建环境、下载模型、换小模型做正式结果、清理他人任务或绕过既有 Serving wrapper。
- 代码复用当前 Scheduler/Worker/BlockPool/KeepPlan/compactor/DP trace 模式。只增加
  完成本目标需要的状态和接口，不建设通用策略框架、额外评分器、任意权重系统或
  过度防御层。先写清 contract、所有者、mutation/completion/commit 边界；真实
  invariant 失败要直接暴露。
- 测试按实际风险选择：扩展现有 CPU suite 覆盖 floor、cap、页取整、确定性、pair
  feasible/infeasible、无 intent mutation、DP watermark 和同分 tie-break；GPU/TP2
  验证必须实际继续生成并复用页面，Graph replay 不能只用 CPU 模拟代替。按 prompt3
  执行适用的 lint/type checks；不为形式完整跑无关大套件，不安装新测试依赖。
- 每项实现先 correctness，再 micro，再 Serving。diagnostic/profile 与正式 Serving
  分离；正式数据关闭逐步 trace 和高扰动 profiler。记录实际源码/input/config hash、
  命令、退出码、GPU UUID、affinity、pool 与 NVML 数据。没有可比证据就报告 PARTIAL/
  STOPPED，不推断收益。
- 阶段性本地提交仅暂存本轮精确文件，遵守 `AGENTS.md` 的 commit attribution/sign-off。
  `.p1-work/` 若被 ignore，只对本轮明确文档使用 `git add -f <file>`。不要自动 push、
  不创建上游 PR，不改 ignore 规则。

按 `B0 → protected 单 victim → 最多双 victim TP/Graph 正确性 → DP 诊断/条件 tie-break
→ 最终统一复测 → X0` 推进。普通实现判断自主完成，不逐阶段等待确认。DP 条件不成立
不阻塞其他工作。新增 Controller、DP、最终评估报告，并更新 `plans/status.md` 和
`reports/delivery.md` 导航。最终分别报告 correctness、micro、Serving/SLO，提交列表，
条件阶段的停止依据，以及未完成和历史 FAIL/PARTIAL；不把简历项目包装成未经证明的
普遍优化。
