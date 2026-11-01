# 下一阶段工程深化完成审计

2026-11-01，按 [prompt3](../prompts/prompt3.md) 从 `04a89d6037` 推进至 X0。
B0、CB0/CB1、TP2a/TP2b、PL0/PL1/PL2、DP3/DP4 已实施并完成各自必要验证；
CB2 未满足进入条件，STOPPED / NOT_RUN。正确性、micro 和 Serving 分别验收，
没有将本轮交付归并为全场景性能 PASS。

## 阶段结论与停止依据

| 阶段 | 实现与正确性 | micro / Serving / 停止结论 |
| --- | --- | --- |
| B0 | 固定分支、用户 patch、源码/环境、输入及比较方法 | [初始归档](../experiments/Engineering-Next-B0/20261101-093618-initial)，没有回退到基准 HEAD |
| CB0 | IMPLEMENTED / CORRECTNESS_PASS：45 次候选决策与 commit 关联、释放预测一致，43 次首次 retry 成功 | 低扰动诊断仍有 host 开销；inclusive 时间不充作回收增量成本 |
| CB1 | IMPLEMENTED / CORRECTNESS_PASS：预算求解、真实 Pool、TP2 q=0/1 continuation/free/reuse | fixed 18/24 SLO、约101 tok/s；demand 0/24、约11.13 tok/s、468 次 timed 抢占，SLO_FAIL；只保留显式实验选项 |
| CB2 | STOPPED / NOT_RUN | 新 demand 轨迹320次候选均覆盖硬缺口 D；没有重复无覆盖进入证据，不实施 skip_insufficient |
| TP2a | IMPLEMENTED / CORRECTNESS_PASS：有序层累加融合、独立 reference、真实 TP2 lifecycle | 相对 T1 完整 reclaim micro 缩短约8.5–13%；Serving 无稳定收益，15/24 SLO_FAIL |
| TP2b | IMPLEMENTED / CORRECTNESS_PASS：四层分组、完整 heads SUM/reference、near-cutoff 计划及36层 payload | 相对 TP2a 完整 micro 缩短约40–52%；Serving 正反向无稳定收益，grouped 两次均17/24 SLO_FAIL |
| PL0/PL1 | IMPLEMENTED / CORRECTNESS_PASS：三策略共用执行、请求/intent 不可变绑定、full 禁止回收、preempt 保版 | 真实 Random TP2及混合策略验证；不新增质量或 Serving 收益结论 |
| PL2 | IMPLEMENTED / CORRECTNESS_PASS：全rank prepare→一致确认→publish→新ADD绑定→无引用且设备完成后退役 | 真实 TP1/TP2延迟准备、旧intent、混合三策略/full、continuation/free/reuse通过；限定单engine |
| DP3 | IMPLEMENTED / CORRECTNESS_PASS：已知工作量/usage 快照贯通，counts 不变也发布，完成归零 | W1 两轮以新鲜快照和实际排队/工作量重复满足 DP4 门槛 |
| DP4 | IMPLEMENTED / CORRECTNESS_PASS：显式 known_work 路由，原 affinity/F/tie 保持 | Serving PARTIAL；W1 正向吞吐+19.081%、反向仅+3.884%且前两轮更慢；W0 -1.563%在≤5%预算内，W1仍 SLO_FAIL；停止扩大实验 |
| X0 | 完成源码、原始证据、复算、导航与本地提交审计 | 不追加GPU，不扩旧矩阵；检查与最终HEAD由下述manifest记录 |

分项事实与全部轮次见 [Controller](engineering-next-controller.md)、
[TP](engineering-next-tp.md)、[Policy](engineering-next-policy.md)、
[DP](engineering-next-dp.md)。DP4 不满足稳定超出重复波动的 Serving 收益判据，
`requests` 仍默认；CB1 不通过添加未经对照的 reserve 掩盖负结果。局部收益成立不等于
Serving收益成立，也不因负结果继续盲扫输入、pool、batch或kernel分组。

## 当前调用链、状态 owner 与支持域

| 能力 | 源码入口与新增行为 | 保持的边界 |
| --- | --- | --- |
| 缺口预算 | [kv_reclamation.py](../../vllm/v1/kv_reclamation.py) 的 `demand_reclaim_candidates` / `retarget_reclaim_candidate`；[Scheduler](../../vllm/v1/core/sched/scheduler.py) 单次候选扫描 | G为软目标、D为当前硬缺口；以 fixed 结果作 K_floor，求最大合法K后才查 exact whole-block；不降低保护区或跨victim用绝对K推断质量 |
| TP评分 | [kv_scoring.py](../../vllm/v1/worker/gpu/kv_scoring.py) 的 `TriAttentionScorer` / `GroupedTriAttentionScorer` | eligible域 population variance、唯一head owner、按层FP32 FMA、全局score SUM、TP root stable selection/broadcast；reduce-root只在benchmark |
| 策略绑定 | [EngineCore](../../vllm/v1/engine/core.py) 处理ADD绑定 `PolicyBinding(version,config,calibration_sha256)`，经 `NewRequestData` 传到slot/intent/result | Scheduler预算与Worker执行同版；preempt/recompute保版，EOS/abort/slot复用不串配置；策略不改Pool/hash |
| 在线更新 | `EngineCore.update_reclamation_policy`→executor collective RPC→[Worker](../../vllm/v1/worker/gpu/kv_reclamation.py) `prepare_policy`，全rank版本/config/artifact SHA一致才发布 | 新ADD以处理顺序取新版，旧请求/intent pin旧版；正常输出实际等待及canonical commit/free后，`post_step`才检查无引用并全rank同步、释放、ACK退役 |
| DP快照/路由 | Scheduler `get_dp_load_snapshot`→普通EngineCore输出队列→[Coordinator](../../vllm/v1/engine/coordinator.py)各rank latest→[Frontend](../../vllm/v1/engine/core_client.py) `_apply_load_stats` / `get_core_engine_for_request` | U只计已知未计算项，含在途、不预测max_tokens；counts立即、load-only按100ms变化发布；S=C+U/batch显式选择，无迁移/借页/usage惩罚 |

Scheduler继续拥有canonical E与Block对象，Worker只持执行镜像。logical L用于原token、
RoPE、sampling与hash，physical E用于allocation、slot和attention。回收仍遵循
forward→score/KeepPlan→KV/row/E写入→全rankcompletion→实际输出等待→
canonical commit→free→真实reuse。Prefix共享页/hash保护、source row/E与结果身份、
单intent及一次deferred retry、Graph active bounds/padding和mutation后fail-stop保留。
调用链细节见 [压力](../source-analysis/m2-pressure.md)、
[评分](../source-analysis/m2-scoring.md)、[TP](../source-analysis/tp-runtime.md)、
[DP](../source-analysis/dp-runtime.md)；稳定测量协议见 [DP协议](../notes/dp-protocol.md)。

PL2不是静态registry：资源按版本在Worker准备并实测退役；Tri资源每rank释放
610,304 allocated bytes，CUDA缓存不等于NVML立即归还。prepare失败零发布，但可能
留下未发布资源至engine关闭，不承诺跨rank事务回滚。支持单engine TP1/TP2，DPLB
广播与带Coordinator的EngineCoreProc明确拒绝更新。切默认full只影响新请求，不能
恢复旧请求已删KV。feature-on仍拒绝resumable/streaming-input，full也不绕过入口。
Codec、量化、offload、跨池迁移、TP2×DP2与新backend均未加入支持域。

## 验证、数值与测量范围

| 类别 | 关键证据 | 可复核入口 |
| --- | --- | --- |
| 组合CPU | 56 passed / 227 deselected，覆盖 pressure / reclamation / DPLB / dense / MoE 的受影响行为 | [X0归档](../experiments/Engineering-Next-Closeout/20261101-1137-x0) 的 `validation/run-cpu.py` 与 `cpu-combined.log` |
| CB预算/动作 | 真实Pool释放；q=0/1各一次8B TP2生命周期；320次demand预测与实际release一致，首次retry仅8次成功 | Controller报告链接原始run；X0 `analysis-manifest.json`记录重新关联命令 |
| TP独立正确性 | TP2a GPU reference 9项；TP2b cutoff/boundary 15项；真实TP2完整heads及模型continuation/reuse | TP报告的 `Engineering-Next-TP-Check` 原始run、两rank validation及attention reference |
| TP完整micro | 三个真实Runtime形状、合成KV，3次warmup、10次Events；每样本取两rankmax再median | `Engineering-Next-TP-Micro/tp2a-summary.json` / `tp2b-summary.json`及X0独立复算 |
| Policy安全更新 | CPU异常/缺rank/重复rank/config/hash不一致均零发布；真实TP2延迟rank和旧intent，TP1混合策略；每rank prepare-end≤publish、local-done≤commit ACK<retire | [独立复算脚本](../scripts/analyze_policy_updates.py)；PL2验证目录 `retirement-recomputed.json`及X0复算 |
| DP控制链/正式A/B | 真实ZMQ与DP2 HTTP，W1/W0预注册输入；6次正式run、每次3×8，三对各24条输出一致 | [DP4分析manifest](../experiments/Engineering-Next-DP-Formal/dp4-analysis-manifest.json)，记录输入/source/helper/hash、真实argv/exit、全部route/work/age |
| 静态/文档 | 适用pre-commit、manual mypy-3.12，Markdown本地链接/anchor、symbols、hash及working/index diff检查 | X0 `validation/`；初次失败也保留 |

CPU runner只将两个无关顶层Llama tokenizer fixture替换为已有本地Qwen3，未下载资源；
不称完整上游suite通过。PL2更新时间/首个新请求step包含延迟注入、JIT、payload检查
与同步，只作受扰动冷开销记录，不当成TTFT或Serving A/B。CPU不能替代设备完成，
诊断不能替代正式结果。

TP2b batched/逐层统计最大差2.384e-7、score最大差5.96e-8，受测near-cutoff计划
严格相同；初次逐bit统计要求FAIL保留，不承诺所有输入逐bit等价。四层grouped的
raw/stat新增2,359,296+512 bytes/rank，原589,824 bytes单层raw仍存在；设备活动
110→47含18次DtoD copy，不能只报29个计算kernel。micro不含正常模型forward，
两个增量的改善不可直接相加为Serving加速。

所有本轮GPU运行经同一guard锁串行，单卡GPU1、双卡GPU0,1，项目显存合计
44GiB停止/45GiB硬上限。41次guard运行中37 PASS、4 FAIL，无资源阈值触发；
最大NVML合计42,832,232,448 bytes。Worker主线程放置CPU20/21，单卡PL2 CPU21；
不声称服务所有线程绑定。实际命令、环境、HEAD/diff及guard记录见每run的
`invocation.json` / `result.json`，被忽略或间接helper的额外快照由wrapper补存。

四个guard FAIL分别为计数hook未保留RPC签名、初次逐bit统计要求、内部warmup缺
policy binding，以及诊断JSON无法序列化嵌套binding；后续修正与重跑均保留。
guard PASS只表示运行和资源通过，不覆盖正式SLO_FAIL。CB/TP正式重算细分及DP4
正式reclaim/首次retry次数仍为UNKNOWN，不能复制诊断数字填充。DP4实际重算和
Prom抢占均为0，输入/工作恒等式有单独校验。

旧 [工程收尾](engineering-closeout.md)、SinkRecent pressure 17/24 FAIL、T1 pressure
15/24 FAIL、C0时间校准PARTIAL、M2/Prefix/TP/DP历史FAIL和profile PARTIAL均保留。
85条质量样本冻结，有损质量保持未证明；输出一致、payload正确、删除更少均不能
代替任务质量结论。H2O仍仅历史源码审查，本轮不恢复importance/probe或质量矩阵。

## 本地提交、用户改动与归档

| 阶段 | 本地提交 |
| --- | --- |
| CB0 | `7957f67b01` |
| CB1 | `2d7f477b04` |
| TP2a | `5dd3a576d5` |
| TP2b | `a33e09e4cc` |
| PL0/PL1 | `392c2166c7` |
| PL2 | `17fcb1f96c` |
| DP3 | `a97ceb20a3` |
| DP4 | `77696ce852` |

X0自身提交的最终HEAD、上述完整SHA和trailers由提交后生成的
[completion-manifest](../experiments/Engineering-Next-Closeout/20261101-1137-x0/completion-manifest.json)
记录，避免在正文嵌入自引用SHA。复算的真实argv/exit、源码/输入/派生hash见同目录
`analysis-manifest.json`；全部41次guard审计和历史失败索引见 `guard-audit.json`。
大产物只在 `/data_ssd/qcrs/experiments/p1-v8`，不进入Git；模型和依赖未更换。

status/delivery只提交本轮新增导航，工作区保留初始用户正文；抽去本轮插入后与B0
备份逐字/hash一致。未跟踪详细计划只追加结果导航，不提交全文。prompt1/2/3与独立
Review原文保持；reference软链接有效且不改目标，B0未记录其目标内容hash的部分
不冒称已有初始hash。最终index为空，tracked dirty仅剩两个原用户文件。沿用原分支，
全部为新增本地commit，未reset/amend/rebase、push或创建上游PR。

## 简历与面试表述

可使用以下表述并保留量化范围：

> 在既有vLLM物理KV回收Runtime上补齐缺口预算、TP评分执行和策略版本生命周期。
> 将请求/intent pin到不可变配置，经全TP rank资源准备与一致确认后更新默认，
> 在旧请求及设备执行结束后安全退役资源，验证Qwen3-8B TP1/TP2混合策略与回收页
> 继续生成、真实复用。将评分按四层分组并保持全局SUM与唯一KeepPlan，在三个真实
> 形状、合成KV的完整回收micro中相对上一增量缩短约40–52%。打通dense DP已知
> 工作量反馈，并完成显式路由的有限正反向对照，报告收益波动及停止依据。

既有物理回收主体和D1上游学习移植不重复归为本轮原创；T1历史33–38%仅micro。
本轮CB1负结果、TP Serving无稳定收益、DP4性能PARTIAL和PL2支持域/失败资源限制
需可解释。接口数量、删除量或路由更均匀均不单列为用户性能提升。
