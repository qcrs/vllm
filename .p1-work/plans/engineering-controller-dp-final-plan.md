# Controller 与 DP 工程深化及统一评估计划

日期：2026-11-07。工作树：`p1/v8-physical-kv-reclamation`。计划基线 HEAD：
`80564d1b668dfe185b00f97c7d101a907118174c`。该基线只用于说明本文形成时的状态，
实施前必须重新记录实际 HEAD、工作树差异和实验配置。

## 目标与判断

本轮围绕一个可用于简历、也能由源码和运行结果支撑的工程故事：在物理 KV 紧张时，
对单请求回收量设置边界；当一个请求的可回收容量不足时，最多由两个请求分担；在
TP/Graph 执行链上安全完成搬移、提交、释放和复用；DP 继续使用已验证的 known-work
路由，并仅在轨迹证明 `kv_cache_usage` 能纠正重复错选时，增加一个 opt-in 的平局裁决。

`TTFT <= 2s`、`TPOT <= 80ms` 继续作为本项目当前 workload 的对比 SLO。它们是项目
验收口径，不宣称为所有模型、硬件或服务场景通用的行业标准。现有 workload、模型、
pool 和请求规模固定时，保留阈值才能比较历史结果；不因新实现达不到阈值而放宽。
每个请求仅在 TTFT、TPOT 两项都满足时算作 SLO 通过，同时报告逐请求结果、分布、
goodput 和吞吐。小样本 P95/P99 只描述该批样本，不用于声称统计显著性。若未来
产品目标改变，应以新的产品负载和延迟预算重新定义 SLO，不能回写本轮结果。

本轮不是全面重做 vLLM，也不以证明新策略一定比 fixed 快为目标。目标是实现逻辑
清楚、边界可测的改动，完成简历主张所需的一轮统一复测，并如实保留负结果。

## 当前证据与边界

| 范围 | 已有证据 | 本轮如何处理 |
| --- | --- | --- |
| Controller fixed | 现有正式压力对照优于 retry 类实验方案；fixed 继续作默认与基线 | 保留语义，作为新策略的对照 |
| CB1 demand/retry | 尾页和已知分配顺序的预测能覆盖不少首次 retry，但正式结果仍为 `0/24` SLO、60 次抢占；早期 demand 也出现过大量反复抢占 | 历史 FAIL 原样保留，不重启参数扫描，不把预测正确等同 Serving 收益 |
| 单请求公平性 | 当前候选下限基于本次物理 KV 长度 E；重复回收时仍可能逐次削弱同一请求 | 增加独立单次上限和逻辑累计下限，fixed 仍默认 |
| DP known-work | pending ADD watermark 修复已验证；W1 正反向均 `24/24`，W0 两种路由均 `24/24`，但 W0 的名义吞吐提升没有超过基线波动 | 统一复测其适用 workload；保持 `requests` 默认 |
| DP pressure | `DPLoadSnapshot` 已包含 `kv_cache_usage`，但它只是副本压力近似量，现有路由没有使用 | 先观察真实决策与后续 miss；只在有效同分场景反复出现错选时实现平局裁决 |
| TP/Graph | 当前 reclaim 协议已包含策略绑定、TP completion、Scheduler commit/free；multi-victim 将扩大同一执行边界 | 只扩展多请求回收协议并验证 TP2/Graph，不做新的 TP scorer 优化或策略热更新 |
| 有损质量 | 冻结质量样本尚未证明有损策略保持模型质量 | 不扩样；不把 token/payload 正确性或保留比例描述成质量保证 |

实施前复读 [prompt3](../prompts/prompt3.md)、[当前状态](status.md)、
[engineering closeout](../reports/engineering-closeout.md)、
[Controller 正式结果](../reports/engineering-refinement-controller.md)、
[DP 正式结果](../reports/engineering-refinement-dp.md)、
[DP 协议](../notes/dp-protocol.md) 和
[Controller 设计提案](vLLM_Physical_KV_Reclamation_Controller_Design_Proposal.md)。
源码和实验原始数据是实现及结论的最终依据；本文是实施范围，不是已完成报告。

## Controller 设计

### 语义与边界

Controller 只在真实 allocation miss 上运行。保留当前 fixed、demand、retry 等模式
及其历史结果；新增的受保护模式显式 opt-in，不改变默认。`retention` 继续沿用旧
语义，不能静默改成相对于原始逻辑上下文的累计比例。

候选计算统一在 Scheduler 当前决策状态完成。实现前逐项确认以下量的源码来源、时序
和单位，不能用总请求长度或未来 `max_tokens` 替代已经计算的上下文：

| 量 | 含义 |
| --- | --- |
| `E` | 本次 reclaim source 中实际可处理的物理 KV token 数，包含本轮已确认纳入的调度 token |
| `L` | 同一调度边界上实际已计算的逻辑上下文长度；不含尚未执行的 prompt 或未来输出 |
| `P` | 不可回收的共享 prefix / protected prefix 长度 |
| `B` | KV block 的 token 数 |
| `D` | 本次真实 allocation shortfall，作为硬目标 |

新保护模式使用两个独立约束。初始实验值为每次最多删除当前私有物理 KV 的 20%，并
至少保留逻辑私有已计算上下文的 50%。这两个数只作为第一版工程边界，不做权重或
百分比扫描，也不宣称最优。若实现发现现有配置需要采用别的合法初值，必须先记录
源码依据和对应实验配置。

```text
single_call_floor = P + ceil(0.80 * max(0, E - P))
logical_floor = P + ceil(0.50 * max(0, L - P))
K_floor = max(
    existing_min_keep_and_sink_recent_protection,
    existing_retention_floor_in_its_current_E_domain,
    single_call_floor,
    logical_floor,
)
```

`K_floor` 只规定可保留 token 数量；策略仍决定保留哪些 token。保护区和页取整沿用
现有 `KeepPlan`、policy binding、shared-prefix 约束和 block row 语义。每个候选动作
按真实 block 数计算实际释放页数，不以 token 差值代替物理容量。逻辑保留率只用于
分担压力的选择，不是质量分数或质量保证：

```text
retained_ratio = (K - P) / max(1, L - P)
```

实施时需确认 `L` 在部分 reclaim、decode、preempt/recompute 生命周期中可正确表达
同一逻辑上下文。优先复用 Request 已有状态；只有源码证明该状态会丢失必要的累计
上下文时，才增加最小的 per-request provenance。不得复制完整请求历史或再造一套
生命周期状态。

### 有界候选与选择

对每个候选生成满足 `K >= K_floor` 的最大合法保留量，得到其最大允许释放页数。按
下列顺序制定一次完整计划：

1. 如果一个候选在所有保护约束内就能覆盖硬缺口 `D`，优先单 victim；在满足缺口的
   单 victim 计划中，优先选择回收后 `retained_ratio` 较高的请求。
2. 只有没有单 victim 可覆盖 `D` 时，才枚举最多两个候选的有界组合。组合先满足
   `D`，再尽量减小超出 `D` 的页数，并最大化受影响请求中的最低回收后保留率；
   随后才比较 zero-copy、复制字节和稳定顺序。
3. 对单 victim 或双 victim 的等价结果，执行成本排序只作次级规则；最后使用到达
   时间和 request ID 形成确定性 tie-break。不引入多项加权分数。
4. 不为满足软 headroom 无限增加 victim。若无一个或两个 victim 能在保护边界内
   覆盖硬缺口，保持原生等待/preemption 行为，不降低下限、不挑第三个 victim。

计划必须使用 block/page 边界精确计算。对固定的释放页目标，取满足目标的最大合法
`K`，并记录预测与实际释放页数。每次回收记录 `E`、`L`、`P`、`K`、当前及回收后
逻辑保留率、victim、释放页、复制字节和累计回收次数。

### Multi-victim 执行协议

先由 Scheduler 选定完整的 1–2 victim 计划，再生成会影响 suffix hash 或其他 canonical
状态的 reclaim intent。一个逻辑 batch 中每个 victim 带有自己的 Request policy binding、
source row、source E、target keep 和稳定顺序。禁止在选第二个 victim 时重新扫描或
依赖已经部分修改的 Pool 状态。

Worker 接收同一 batch；TP ranks 必须执行相同的 victim 集合、顺序和 keep plan。所有
rank 完成后，由 Scheduler 验证整批结果身份、源状态和目标行，再统一提交 canonical
row/E 并释放相应页面。GPU 工作未完成前不能 commit/free；批次完成后必须实际分配并
改写回收页。复用现有 Graph 外 maintenance、persistent buffer 和 completion wait，
验证回收后 Graph replay。

所有可在执行前验证的候选和 batch contract 都应在第一次 mutation 前检查。若 GPU
mutation 后发生错误，沿现有 fail-stop 边界暴露错误；不做静默回滚、部分 commit、
自动 recompute 或额外 fallback。单 victim 与双 victim 均验证 abort、finished、
continuation 和页 reuse 的现有语义。该协议不等同于设备侧可回滚的原子事务。

## DP 增量设计

已有 `known_work` 与 pending ADD watermark 属于当前实现，不在本轮重造。先追踪完整
的 Scheduler → stats → EngineCore/Coordinator → Frontend route 链，确认：快照身份和
新鲜度、每个 client 的 ADD watermark、pending work 计入和撤销、sticky/abort/finished
语义，以及 `kv_cache_usage` 的准确来源和数值意义。

### 证据先行

新增一个低扰动诊断字段集，复用现有 trace/归档能力，不建设通用 telemetry 框架：

- 每个选路决策的 replica work score、snapshot sequence/生成时间/年龄和已知 pending ADD；
- 每个候选副本的 `kv_cache_usage`、所选副本和入站 prompt 长度；
- 后续本地 allocation miss、reclaim、preemption/recompute，以及对端是否保持更低使用率和可服务余量。

只检查符合当前新鲜度合同的快照。重点判断是否重复出现：work score 相同、usage
明显不同、当前选择了较高使用率副本，随后该副本发生本地 miss/抢占，而低使用率
peer 有余量。单次不同选择、快照陈旧、跨时段忙闲差异或仅仅“路由更平均”均不构成
进入证据。也要检查 usage 是否把可驱逐 prefix cache 或其他不可比状态当作持续压力。

如果找不到重复且可复现的同分错选，就停止 DP 新代码，只交付 known_work 与受保护
Controller 的组合验证和 trace 结论。若证据成立，新增显式 opt-in 的 pressure tie-break：
仅在已有 known_work work score 相同时优先选择较低 `kv_cache_usage`；usage 缺失或
快照不满足既有新鲜度条件时沿用已有 tie-break。不开启 usage 加权、不重排有明确
work score 差异的副本、不增加权重搜索、不迁移活动请求、不跨 replica 借用 KV pool。

压力 tie-break 的正式对照只与 `known_work` 比较，以隔离新信号；同时保留 `requests`
作为服务默认基线。W1 混合长短请求是主要诊断，W0 均匀请求作为不应明显退化的
保护对照。使用当前 DP2 workload、arrival、pool、model 和 affinity 约束；不为得到
正结果改变输入。

## 统一最终评估

“全部重测”限定为：重测简历主张直接涉及的正确性、micro 和 Serving，并在所有新增
Controller/DP 修改完成后的同一源码状态、同一固定配置族下采集。不恢复旧参数扫描、
不扩大支持矩阵、不重启 85 条质量样本。

| 层次 | 固定验证范围 | 必须回答的问题 |
| --- | --- | --- |
| CPU 正确性 | 现有 KV reclaim/Pool/Scheduler suites；DP route/watermark suites | cap、累计下限、页边界和选择顺序是否正确；无覆盖时原生行为是否保留；pending ADD 是否只计一次；usage tie-break 是否只影响有效同分 |
| TP2/GPU 正确性 | Qwen3-8B BF16、单/双 victim、继续生成、全 rank completion、commit/free/rewrite/reuse、Graph replay | 所有 rank 是否执行相同 batch；设备工作完成前是否不释放；页面是否真实复用；Graph 是否继续正常运行 |
| Controller micro | 固定 shapes 与同一 pool 条件下，fixed 和 protected；分别测计划决策、完整 reclaim、GPU compaction、victim 数、释放页、复制字节 | 一次上限和累计下限是否真的减少单请求删除；新增计划开销和完整路径成本是多少 |
| DP route micro | requests、known_work，以及仅在条件进入时的 known_work+usage tie-break | Frontend 选路时间/额外开销是否有限；新字段是否只用于既定 tie-break |
| TP2 Serving | `native`、`fixed`、`protected` 在同一代表性压力负载下 | SLO、goodput、吞吐、抢占/重算和单请求 TTFT/TPOT 是否有变化；多 victim 是否减少过度压缩而未制造新退化 |
| DP2 Serving | `requests` 与 `known_work` 的 W1/W0；条件进入时再比较 `known_work` 与 usage tie-break | W1 的已有收益在最终源码是否复现；W0 是否有明显退化；压力信号是否改善请求级结果 |
| 组合检查 | DP2 W1 + known_work + protected local reclaim；若 pressure tie-break 已进入，再在同一场景开启它 | 路由和 replica-local reclaim 能否共同工作；每个 replica 保持独立 pool；是否有本地 miss/抢占变化 |

正式对照沿用 prompt3 的完整预热、每组 3 个 timed round、资源 guard 和基本运行顺序
复核。反向启动仅在主对照出现正向收益、明显漂移或顺序依赖时按 prompt3 补一次；
不能只选最快轮次。诊断 trace 不算正式 Serving 数据。没有证据支持的 TP scorer
T0/T1 新优化不在本轮展开；如要将历史 T1 micro 数字列入统一简历材料，应使用现有
benchmark 里可复现的同版本对照；若只能从旧源码/不同轮次取得 T0，则把旧数字明确
标为历史结果，不为凑“同一轮”重造第二套 scorer。

各类结果分别报告：

- 正确性：CPU suite、实际 GPU/TP/Graph 生命周期，具体通过行为和支持边界；
- micro：选择开销、完整 reclaim、设备 compaction、释放页/复制字节/受影响 victim；
- Serving：逐请求 TTFT/TPOT、SLO 通过数、既有口径的 goodput、输出吞吐、preempt/recompute、每副本分配与 usage、峰值显存。

不得把 micro 节省直接写成 Serving 提升。既有 CB1 的 `0/24`、60 次抢占、DP 原始
DP4 `PARTIAL`、TP/Serving 历史 FAIL/PARTIAL 继续可见；新结果按自身固定输入和配置
单独判定，不覆盖旧记录。输出 token/payload 的正确性不代表模型质量已验证。

## 实施顺序与本地交付

1. **B0：基线冻结。** 复核适用 `AGENTS.md`、prompt3、实际 HEAD/branch/diff、ignored
   工作、环境、Serving wrapper、GPU UUID/映射、CPU affinity、模型/校准文件、token IDs
   与 hash、pool 和命令。保存所有结果所需的源码和输入 hash。冻结是记录状态，不是
   checkout、reset 或覆盖任何用户改动。
2. **C1：保护语义与单 victim。** 实现纯预算/候选 helper 和 opt-in 配置，先通过最小
   CPU 行为测试，再跑真实 TP2 single-victim continuation/reuse。
3. **C2：最多双 victim 批次。** 只有单 victim 保护测试通过后，扩展 request/worker
   批次契约；通过 TP2 所有 rank 和 Graph replay。若已有 protocol 无法支撑而需要
   改共享公开 contract，记录必要范围，保持功能简单，不引入通用调度框架。
4. **DP0：低扰动 pressure 诊断。** 复用 known_work/pending ADD。满足同分错选证据才
   实现 usage tie-break；否则记录 STOPPED 和原因，不影响最终组合测试。
5. **E0：统一评估。** 使用最终源码执行表中必要 correctness、micro、TP2/DP2 Serving
   和组合检查。正式指标按固定场景采集，不扩质量集、不扫参数。
6. **X0：文档与阶段提交。** 新增 Controller、DP 和最终复测报告，更新
   `plans/status.md`、`reports/delivery.md` 导航；按实现边界分成本地提交。所有 commit
   只暂存本轮明确文件，保留已有用户修改；不 push、不创建上游 PR。

建议阶段提交：Controller 预算/保护与 CPU 测试；multi-victim 执行与 TP2/Graph 验证；
DP 诊断及（条件成立时）tie-break；最终结果和状态导航。报告真实 commit、实际命令、
退出码、运行配置、结果、限制和停止结论。新实验进入 prompt3 约定的
`/data_ssd/qcrs/experiments/p1-v8` 归档树，不覆盖历史目录；源码仓库只提交必要的
普通文本、代码和测试，不提交模型、trace、缓存、二进制或实验软链接。

`.p1-work/` 受仓库 ignore 规则影响；如本计划或 Prompt 需要进入 Git，只对这两个
明确文件使用 `git add -f <path>`，不得强制添加整个 `.p1-work/`。

## 停止与结论标准

- 任一保护约束无法在真实 source state 下可靠计算，先修正状态定义；不能用保守但
  错误的逻辑长度或重复状态掩盖问题。
- 一个或两个 victim 无法满足硬缺口时，保持原生 fallback；不突破保护比例、不扩至
  第三个 victim、不通过增加 reserve 强行制造成功。
- 实际执行预测页数和 Scheduler commit 页数不一致，或任何 rank/Graph/allocator
  生命周期不一致，先修正确性，不进入正式性能结论。
- DP 没有重复、有效同分错选证据时，停止 pressure router 实现。路由分布更均衡本身
  不算收益。
- SLO 未过则报告 FAIL；可以先分析一轮进行优化，如还是FAIL 就记录FAIL达到 SLO 但 throughput/goodput 没有超过同配置波动时，只报告
  正确性或局部 micro，不写 Serving 提速。
- 这是一项简历工程项目。优先交付可解释、可复测、跨 Scheduler/Worker/TP/Graph/DP
  生命周期完整的技术实现；停止没有证据的调参、抽象和兼容层。
