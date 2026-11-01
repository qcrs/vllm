# 工程深化完成审计与项目叙事

2026-11-01，本轮按D1→T0/T1→C0→H0源码审查完成。D1负载链修复与T1有限融合
保留；条件路线D2、C1/C2、H1没有触发，H0 GPU probe未运行。没有重启M0→DP、
扩大85条质量样本或把所有候选优化变成必做功能。

## 实际交付与停止判断

| 阶段 | 实现/证据状态 | 性能或停止结论 | 详细报告 |
| --- | --- | --- | --- |
| D1 | PASS：dense发布→Coordinator独立汇总→Frontend归零及本地in-flight下界 | native和uniform各版本均24/24 SLO、零抢占；未测得可分辨性能收益 | [DP负载链](engineering-dp-load-chain.md) |
| D2 | NOT_RUN：最小诊断与离线公式回放已做 | 仅timed末次建议变化，warmup未重复；未证明持续错选或更好路由 | [D2停止依据](engineering-dp-load-chain.md#最小诊断与d2反事实) |
| T0/T1 | PASS：融合normalization/GQA/head reduction，独立reference与真实TP2 continuation/free/reuse通过 | 三个实测shape的micro完整reclaim缩短约33–38%；Serving无稳定收益，pressure各15/24仍FAIL | [TP评分链](engineering-tp-scoring.md) |
| TP通信/裁剪 | NOT_RUN：Runtime未修改；Reduce-root只在benchmark比较 | score collective约.07ms，完整链无稳定收益；保护元素比例不足以支持继续裁剪 | [T0/T1口径](engineering-tp-scoring.md#t0t1测量口径) |
| C0 | PARTIAL：29次requester/victim/completion/commit/retry关联完成 | 24次首次retry成功；5次仍preempt后重算admission，不能当retry成功；增量时间校准不足 | [Controller成本](engineering-controller-cost.md) |
| C1/C2 | NOT_RUN | 无重复时间错选或匹配native动作成本证据，保留已有排序和动作边界 | [Controller停止依据](engineering-controller-cost.md#c1c2停止理由与保留能力) |
| H0/H1 | H0源码审查完成；GPU probe/H1 NOT_RUN | FA2 out/LSE不提供per-key重要性；官方实现物化Q×K，未得到满足约束的统计方案 | [H2O可行性](engineering-h2o-feasibility.md) |

“PASS”只对应表内实现和验证；不是全场景性能或质量保证。旧SinkRecent pressure
17/24 FAIL、旧M2/Prefix/TP/DP负结果及profile PARTIAL原样保留。T1不能解释或
修复不执行score SUM的SinkRecent TTFT；本轮TriAttention pressure也因TTFT超标
保持FAIL，SLO仍为TTFT≤2s且TPOT≤80ms。

## 调用链、owner与本轮修改

D1复用Scheduler→普通`EngineCoreProc._maybe_publish_request_counts`→原输出通道→
`DPCoordinatorProc.process_input_socket`→`DPLBAsyncMPClient`。Scheduler输出
`(running,waiting)`，Frontend使用`[waiting,running]`；dense按各rank latest汇总，
MoE保留step/wave。路由`max(4W+R,F)`中的F来自已有`reqs_in_flight`，不另造请求表。
这是参考上游#49204后适配本分支的正确性修复，不是原创Router算法。

T1在`TriAttentionScorer.shared_scores`内保留eligible域`var_mean`与逐层累加，
以`_normalize_reduce_scores`融合中间张量操作，增加36KiB/rank的`layer_total`。
真实测量形状、stride、logical L来源及数值限制已写入
[scoring源码记录](../source-analysis/m2-scoring.md)和[TP协议](../notes/tp-protocol.md)。
benchmark隔离阶段及完整reclaim，Serving wrapper补实际源码/hash归档，分析脚本
复算每轮/每请求结果。C0只扩展离线`analyze_tp_ttft.py`；没有Controller热路径改动。

Scheduler仍拥有canonical E与Block对象，Worker只维护执行镜像。回收继续遵循
forward→score/plan→KV/row/E写入→全rank completion→AsyncOutput实际等待→
canonical commit→free→真实reuse。logical L仍用于RoPE/sampling/hash，physical E
决定allocation/attention。Prefix保护、Graph bounds/padding、结果身份与mutation后
fail-stop没有削弱；未因本轮正确性轨迹扩大未测组合的支持范围。

## 验证与可复核证据

| 检查 | 结果与范围 | 原始证据 |
| --- | --- | --- |
| D1 CPU控制链 | 7 passed / 27 deselected；真实编码、ZMQ独立汇总及路由/affinity | [初始归档](../experiments/Engineering/20261031-232434-initial-state)中的`run-d1-cpu-tests.py`/日志 |
| T1单卡score reference | 7 passed / 14 deselected；含非2幂group、partial tile、零方差/ties | [scoring-reference](../experiments/Engineering-TP-Check/20261031-234629-scoring-reference) |
| T1真实TP2 score/plan | 3 cases×2 ranks；最大score误差约5.364e-6，plan一致 | [collectives-reference](../experiments/Engineering-TP-Check/20261031-234833-fused-collectives-reference) |
| T1真实模型生命周期 | 2次Tri回收、全36层payload、继续decode、释放页rewrite/reuse；每rank47次FP32 attention reference | [continuation-reuse](../experiments/Engineering-TP-Check/20261031-234918-fused-real-continuation-reuse) |
| D1/TP正式同配置对照 | guard均exit0；DP四次、TP pressure正反向四次和short两次，全部轮次保留 | [DP Formal](../experiments/Engineering-DP-Formal)、[TP Formal](../experiments/Engineering-TP-Formal) |
| CPU复算 | 三条实际argv均exit0；TP聚合、扩展TTFT关联、Controller outcome audit | [收尾归档](../experiments/Engineering/20261101-0832-closeout)的`analysis-manifest.json` |
| 静态与文档 | 适用pre-commit、mypy-3.12、Markdown本地链接、symbol/hash/diff检查 | 同一收尾归档的`validation/`；初次失败也保留 |

D1原suite因未缓存Llama tokenizer collection失败，runner仅替换两个无关顶层
tokenizer资源为已有本地Qwen3；不称完整上游suite通过。所有GPU由guard串行运行，
单卡GPU1、双卡GPU0,1，未使用GPU2；Worker主线程CPU20/21，未声称全服务线程绑核。
项目显存峰值与44/45GiB合计限制按各报告归档，本轮没有安装升级依赖或下载模型。

复算命令完整路径、源码和派生结果hash见`analysis-manifest.json`；每个GPU run的
`invocation.json`保存真实命令、HEAD/diff、环境及参数。收尾验证不重跑GPU。
最终benchmark的设备API/workspace元数据与实测harness有收尾差异，原文件保留；
scorer最终SHA仍为`88428bb1a655b8633762c0553e2ea6652aa76973f99f7997e8ec0c58576b5f5d`。

首组TP baseline缺dirty scorer即时快照，事后恢复文件及provenance明确标注；
反向A/B和T0/T1保存实际源码，不将恢复文件冒称即时证据。C0只使用已被profiler
扰动的旧SinkRecent trace，不能由其inclusive耗时拟合可靠的增量成本或动作反事实。
H0仅源码审查，既无GPU成本FAIL，也无H2O质量优越性结论。

## 本地提交与用户工作保留

D1提交为`3c9ee4117c`，T1为`be492ebfa9`。C0/H0/导航由本报告所在收尾提交记录；最终HEAD、各文件
hash、检查命令/退出码及提交列表保存在收尾归档的`completion-manifest.json`。
沿用原分支，未reset/amend/rebase、push或创建上游PR。

初始仅`status.md`和`delivery.md`有tracked dirty内容。本轮将新的当前结论置顶，
旧正文标记为带日期的历史记录；index只纳入本轮新增说明。初始用户patch保留在
working tree，与备份逐字比对；不会把用户原有计划声明代为提交。原有未跟踪
prompt2、工程计划及路线图全文不纳入新贡献，计划只追加结果链接。独立Review
与四份reference原文不改，模型、trace、二进制及大产物不进入Git。

## 可用于简历与面试的表述

可使用以下表述，并同时保留限制：

> 在vLLM MRV2实现logical/physical KV长度分离，将共享KeepPlan、Triton搬移、
> 全rank completion和Scheduler commit/free接入真实allocation miss，验证回收页
> 在Qwen3-8B TP2中的继续生成与真实复用。参考上游设计修复dense DP跨进程负载
> 发布与本地在途下界；随后融合TriAttention归一化/GQA归约，在三个真实Runtime
> 形状、合成KV的microbenchmark中将完整回收耗时降低约33–38%。

面试需进一步说明：物理回收Runtime是本轮之前已有的项目主体；D1是上游学习移植，
T1是本轮有限内核增量。Serving正反向没有稳定收益，不能写成吞吐提高33–38%；
DP无性能突破，Controller时间校准PARTIAL，H2O接入NOT_RUN。85条探索性质量样本
继续冻结，token一致、retained payload正确与固定retention都不能证明有损质量保持。
