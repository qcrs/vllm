# P1 v8 工程交付索引

<!-- ENGINEERING-NEXT-X0-NAV-BEGIN -->

## 2026-11-01 prompt3实施完成：当前交付

当前结论以 [下一阶段工程收尾](engineering-next-closeout.md) 为准；
从B0推进至X0，完成Controller缺口预算、TP两项评分增量、逐请求策略绑定、
真实PL2安全更新与dense DP工作量反馈/显式路由。CB2缺进入证据而停止；
TP2b、DP4均按门槛实施，已给有限对照和停止结论。

正确性：受影响CPU、真实Qwen3-8B TP1/TP2及DP2控制链通过；PL2有全rank准备、
旧请求/intent保版和安全退役证据，限定单engine。micro：TP2a/TP2b完整回收分别
相对前增量缩短约8.5–13%/40–52%，仅三个真实形状、合成KV。Serving：CB1 demand
明显负收益；TP无稳定收益、SLO_FAIL；DP4正反向变化不稳定，PARTIAL、W1 SLO_FAIL。

详细报告：[Controller](engineering-next-controller.md)、[TP](engineering-next-tp.md)、
[Policy](engineering-next-policy.md)、[DP](engineering-next-dp.md)。最终提交、命令/hash、
41次guard及4次失败审计见收尾归档；85条质量样本冻结，旧FAIL/PARTIAL不改写。
分阶段本地提交已完成，未push或创建上游PR，原用户待提交正文保留。

以下为本轮实施前的导航和历史交付，按对应日期及配置理解。

<details>
<summary>prompt3实施前的计划及历史记录（原文保留）</summary>

<!-- ENGINEERING-NEXT-X0-HISTORY -->

## 2026-11-01工程深化交付

本轮完成dense DP负载闭环修复和一项TP评分归约融合，相关正确性与真实Runtime
轨迹通过。三个真实shape、合成KV的micro完整reclaim耗时缩短约33–38%；
**Serving正反向没有稳定收益**，TriAttention pressure各15/24 SLO，TTFT仍FAIL。
D1主对照及均匀对照均24/24，性能差异落在波动内。SLO与质量边界没有放宽。

| 交付 | 源码/协议 | 事实与停止依据 |
| --- | --- | --- |
| D1与条件D2 | [DP调用链](../source-analysis/dp-runtime.md)、[DP协议](../notes/dp-protocol.md) | [DP负载闭环](engineering-dp-load-chain.md)：D1保留，D2 NOT_RUN |
| T0/T1 | [评分调用链](../source-analysis/m2-scoring.md)、[TP协议](../notes/tp-protocol.md) | [TP评分报告](engineering-tp-scoring.md)：有限局部收益，端到端无稳定收益 |
| C0与条件C1/C2 | [既有pressure链](../source-analysis/m2-pressure.md) | [Controller成本](engineering-controller-cost.md)：C0 PARTIAL，C1/C2 NOT_RUN |
| 可选H0/H1 | 绑定官方commit与当前FA2源码 | [H2O可行性](engineering-h2o-feasibility.md)：源码审查完成，GPU probe/H1 NOT_RUN |
| 完成审计与简历叙事 | [当前状态](../plans/status.md)、[工程计划](../plans/engineering-optimization-experiment-plan.md) | [工程收尾](engineering-closeout.md)：归因、验证、provenance及本地提交 |

既有物理回收Runtime仍是项目主体：L/E分离、共享KeepPlan、全rank completion、
canonical commit/free、真实reuse以及Prefix/Graph边界按原Gate交付。D1是参考上游
后的正确性修复，T1是本轮有限内核增量，不能把局部33–38%写成Serving吞吐收益。
旧SinkRecent17/24 FAIL与全部历史负结果保留；85条质量样本冻结，token一致不
证明有损质量保持。未扩展原支持域，未push或创建上游PR。

新分析命令/源码/hash与静态检查保存于
[收尾归档](../experiments/Engineering/20261101-0832-closeout)。初始用户dirty说明
保留在working tree；本轮仅提交自己的新增内容。以下折叠旧正文保留历史支持契约、
证据导航与用户计划说明，日期为2026-10-31，不覆盖上面的当前阶段结论。

<details>
<summary>2026-10-31功能收尾与计划记录（历史）</summary>

当前完成限定范围的功能收尾：R1请求生命周期误放行已封口，R2和D1–D4已修订。
本次修改、验证命令和剩余边界集中在 [closeout-fixes.md](closeout-fixes.md)。
[独立收尾Review](independent-closeout-review.md) 保留 `252c747ad6` 发现时的原文，
其中“待修复”是历史基准；当前状态由 [status](../plans/status.md) 维护。

后续按用户授权完成O1的稳定请求ID/step/CUDA关联诊断，见
[TP TTFT诊断](tp-ttft-diagnostic.md)。采样偏离原正式零抢占轨迹，未据此修改Runtime
或宣称性能改善；原TP pressure SLO FAIL保留。

工作树为 `worktrees/p1-v8-physical-kv-reclamation`，分支为
`p1/v8-physical-kv-reclamation`，vLLM基线为v0.26.0 `568afb3a13`。
沿用现有Python3.10/Torch2.11+cu129/Triton3.6环境；模型实验仅Qwen3-8B BF16。
大产物位于 `/data_ssd/qcrs/experiments/p1-v8`，各run保留命令、源码状态、参数、
日志和退出码；历史失败与原始产物未覆盖。

## 当前支持契约

logical L用于RoPE、sampling和原token/hash chain，physical E用于实际allocation、
KV slot和attention长度。Scheduler持有canonical E和Block对象，Worker持有执行镜像。
回收在post-forward/maintenance完成边界执行，GPU写入完成后才commit row/E、
free removed blocks；后续请求真实复用原生Pool中的页。

feature-on明确拒绝所有 `resumable=True` 和streaming-input，包括retention=1。
`Scheduler.add_request` 在入队、分配或session mutation前失败；`AsyncLLM.add_request`
在消费输入、注册输出或发送请求前清晰报错。feature-off保留原生能力。
preempt释放KV、清零L/E后recompute仍受支持，与保留KV的streaming session重入不同。
具体调用链与owner见 [m1-runtime](../source-analysis/m1-runtime.md)。

当前Gate覆盖MRV2、单在途、BF16、FA2/NHD/B16、单dense full-attention group、
generate runner。SinkRecent、Random、TriAttention-inspired shared scoring接入真实
Runtime；Prefix保护完整共享页，Graph消费持久physical buffer，TP2等待全rank完成，
dense DP2保持本地Pool和原生affinity。各组合只以实际轨迹为准。

Async scheduling/spec/connector、PP/DCP/PCP、SWA/hybrid、非标准位置偏置、
fine-grained shared partial page、prefill-time回收及TP2×DP2未获支持或未验证。
不由独立Gate推断全组合通过，也不改变原设计的治理状态或生产默认配置。

## 当前性能与质量边界

SLO保持 **TTFT ≤ 2s 且 TPOT ≤ 80ms**：

- TP eager开销0.868%只支持本机CPU20/21、short/eager、8GiB/rank、batch2048的
  ABBA有限样本预算通过。历史无绑核14.33%/6.871%仍为FAIL，不外推默认放置。
- TP pressure batch1024为17/24、零抢占，七个TTFT超标，SLO仍FAIL。
- DP skewed batch512为24/24，只覆盖本机CPU20/21、固定6/2路由及对应配置。
  历史batch2048的18/24保留FAIL，不归因为router改进或跨池共享。

历史M2/Prefix压力FAIL、TP其他配置FAIL、Graph/TP旧profile PARTIAL均保留在
各自报告。新profile完整不追溯改变旧记录；diagnostic/profile不并入正式吞吐。
累计归还blocks可包含同一页多次复用，不等于唯一物理页数或同时新增容量。

85条质量样本保持冻结，只是探索性结果，未证明有损质量保持。
token一致性或按KeepPlan搬移正确不能替代质量保证。限定功能收尾的O1–O3判断见
[修复报告](closeout-fixes.md)；其后O1仅补一轮诊断，O2/O3仍未实施，当前停止理由见
[TP TTFT诊断](tp-ttft-diagnostic.md)，未扩展模型、质量或性能矩阵。

## 实现与证据导航

| 范围 | 源码学习入口 | 实验事实与历史负结果 |
| --- | --- | --- |
| M0 / native与allocator | [m0-call-chain](../source-analysis/m0-call-chain.md) | [m0](m0.md) |
| M1 / L/E、搬移、completion、reuse | [m1-runtime](../source-analysis/m1-runtime.md) | [m1](m1.md) |
| M2 / pressure与多策略评分 | [m2-pressure](../source-analysis/m2-pressure.md)、[m2-scoring](../source-analysis/m2-scoring.md) | [pressure](m2-pressure.md)、[scoring](m2-scoring.md)、[质量与Serving](m2-evaluation.md) |
| Prefix / P与hash ownership | [prefix-runtime](../source-analysis/prefix-runtime.md) | [正确性](prefix.md)、[Serving](prefix-serving.md) |
| Graph / active bounds与padding | [graph-runtime](../source-analysis/graph-runtime.md) | [正确性](graph.md)、[Serving](graph-serving.md) |
| TP / distinct-GQA与全rank完成 | [tp-runtime](../source-analysis/tp-runtime.md) | [正确性](tp.md)、[Serving](tp-serving.md) |
| DP / 本地页池与affinity | [dp-runtime](../source-analysis/dp-runtime.md) | [正确性](dp.md)、[Serving](dp-serving.md) |
| 历史功能与性能收尾 | [final-review](final-review.md) | [performance-followup](performance-followup.md) |
| 本次R1/R2/D1–D4 | [独立Review基准](independent-closeout-review.md) | [closeout-fixes](closeout-fixes.md) |
| 后续O1请求级TTFT诊断 | [tp-runtime](../source-analysis/tp-runtime.md) | [tp-ttft-diagnostic](tp-ttft-diagnostic.md) |

本页只维护当前结论和导航；[status](../plans/status.md) 维护当前状态与未决项，
reports保存实验事实与历史负结果，notes保存协议，source-analysis解释当前调用链、
状态所有者和invariant。历史plan只指向结果，不重复维护实验数表。

本次必要CPU回归28项通过，未执行GPU continuation；先前扩大选择时的上游fixture/
async插件失败及适用静态检查均在 [closeout-fixes](closeout-fixes.md) 单独记录。
旧154项、GPU轨迹、microbenchmark和完整profile仍按原报告范围使用，不并入本轮测试数。

</details>

</details>
