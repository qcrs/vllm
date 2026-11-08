# P1 v8 工程交付索引

<!-- PROTECTED-STABLE-NAV-BEGIN -->

## Protected-only 稳定版

稳定版主结果与边界见[Protected KV 回收结果](protected-stable-results.md)。运行配置只有
native（不配置回收）和 protected（配置回收）；保留独立的 KV 保留/压缩策略与正确性测试。
本页仅整理现有 GPU1 单卡、GPU1/2 TP2 结果，不是新一轮性能测量；历史对照和负结果仍保留
在对应评估归档中。

<!-- PROTECTED-STABLE-NAV-END -->

<!-- ENGINEERING-RETEST12-NAV-BEGIN -->

## 2026-11-08 GPU1/2空闲基线与压力复测

见[收尾](engineering-retest-closeout.md)和[完整评估](engineering-retest-evaluation.md)。
GPU1单卡及GPU1/2 TP2、DP2对照已完成，未改Runtime源码。CPU/DP fixture、TP2 Graph与
真实页复用、DP2生命周期正确性通过；protected第二轮zero-copy micro为0.342ms，不能
代表端到端加速。TP1 burst protected吞吐低42.59%、SLO21/24降至9/24；TP2 burst吞吐
低约28%，但SLO9/24升至15/24，正反向一致。W1 known_work路由收益在DP正反向复现，
protected相对fixed在小池更慢。W0反向请求启动失败，保留PARTIAL。

SLO为TTFT≤2s且TPOT≤80ms。GPU1/2空闲批次29次guard为28 PASS/1 FAIL、峰值
39.888672GiB；正式窗口无外部GPU计算。测试后GPU2出现其他用户占用，未触碰。旧矩阵
和85条质量样本未扩大，历史FAIL/PARTIAL保留；本地分阶段提交，未push或创建PR。

<!-- ENGINEERING-RETEST12-NAV-END -->

<!-- ENGINEERING-PREFILL-NAV-BEGIN -->

## 2026-11-08 Prefill公平性：GPU0同卡基线

见[收尾](engineering-prefill-closeout.md)与[统一评估](engineering-prefill-evaluation.md)。
protected admission/retry为页对齐prompt预留首decode必要空间；wrapper透传已有
512-token限额并关联首token阶段。保留两轮门控、20%/50%、完整prefill预留及backfill。

CPU92+DP本地13项及静态检查PASS。GPU0/CPU20正反向8组各24/24 SLO、零抢占/重算。
protected短TTFT下降74.80%/73.73%，长8192 TTFT增加约23%、仍<2s，吞吐下降
1.84%/1.07%；同512下距fixed吞吐0.28%/0.02%。定位到计算预算公平性，未宣称
新Controller算法、kernel或TP通信加速。限额显式推荐，默认保留，停止扩大调参。

本轮micro、真实TP2/DP2 NOT_RUN，GPU1其他任务保留；项目峰值17.564453GiB。
历史FAIL/PARTIAL、冻结85条样本与用户patch保留，未push或创建PR。

<!-- ENGINEERING-PREFILL-NAV-END -->

<!-- ENGINEERING-TWO-ROUND-NAV-BEGIN -->

## 2026-11-08 两轮protected回收与单卡/TP2/DP2验证

见[收尾](engineering-two-round-closeout.md)、[统一评估](engineering-two-round-evaluation.md)
和[Controller](engineering-two-round-controller.md)。已实现最多两轮可行路径、真实
decode完成门控、受益等待者容量窗口及近完成自然释放；20%/50%和最多双victim不变。
默认仍rounds1/fixed/requests，Worker与全TP completion/commit协议复用。

Correctness：CPU118+DP fixture15项、真实单卡/TP2/DP2通过，含双victim、延迟
completion、Graph、改写复用、abort/continuation。Micro两轮约3.79/3.80ms，
fixed约3.20ms，不宣称执行提速。单卡相对r1吞吐+16.79%/+26.63%，SLO18/24→
23/24、24/24；TP2吞吐改善但SLO18/24→14/24、12/24，未超过fixed。DP同路由
两组均21/24且零回收/miss，不能归因Controller；usage tie-break仍STOPPED / NOT_RUN。

仅物理GPU0/1、23次guard为20 PASS/3 FAIL、合计峰值39.888672GiB，GPU2及他人任务
保留。独立诊断8/8只证明路径，未复现正式慢路径；正式负结果保留。
SLO仍TTFT≤2s且TPOT≤80ms；未重跑旧矩阵、未扩大85条质量样本，用户原patch和
历史正文保留，本地阶段提交完成，未push或创建PR。

<!-- ENGINEERING-TWO-ROUND-NAV-END -->

<!-- ENGINEERING-BACKFILL-NAV-BEGIN -->

## 2026-11-08 Admission backfill / TP 瓶颈复核

见[收尾](engineering-backfill-closeout.md)与[评估](engineering-backfill-evaluation.md)。
实现显式、最多两个成功admission的FCFS backfill及DP capacity guard交互修正，
CPU本地fixture107项、lint/type checks通过。TP2 GPU1/2与等效容量单卡GPU1均
6/24→18/24 SLO；fixed均24/24且吞吐更高。短请求全部恢复达标，8192-token仍0/6。
单卡足以复现主要容量/队头阻塞问题，当前无证据归因TP collective；TP2 backfill
零回收，不能宣称物理回收收益。micro本轮NOT_RUN，DP usage tie-break仍停止。
backfill默认0，长请求容量及总吞吐目标未解决，prefill中删除KV按用户要求暂缓。
GPU0未用、六次guard PASS；用户改动/冻结样本/历史FAIL/PARTIAL保留，未push或创建PR。

<!-- ENGINEERING-BACKFILL-NAV-END -->

<!-- ENGINEERING-PROGRESS-NAV-BEGIN -->

## 2026-11-08 Controller 进展修正与 TP/DP 复核交付

机制与完整结果见[收尾](engineering-progress-closeout.md)、
[统一评估](engineering-progress-evaluation.md)、
[Controller](engineering-progress-controller.md)与[DP](engineering-progress-dp.md)。
已实现完整prefill admission/净页预算、合法整页重映射优先与同victim内有界回收合并。

Correctness：CPU99项、DP本地fixture15项，真实TP2单/双victim、延迟completion、
Graph、页改写复用与DP abort前等待者进展通过。Micro：小额复制缺口修复，但不同
释放量不能作为等动作加速。TP同源码三轮零抢占/重算，protected73.359tok/s、6/24，
仍较fixed101.548tok/s、18/24低27.76%；容量不足有直接证据，按计划停止调参。

DP已有known_work在固定W1正反向吞吐+16.09%/+21.52%，均24/24；W0无稳定收益。
本轮没有新增DP Runtime，收益不归因Controller。usage tie-break缺进入证据而停止；
组合24/24但压力动作全0。SLO仍TTFT≤2s且TPOT≤80ms，fixed/requests继续默认。

本轮只用物理GPU1/2，GPU0未用；29次guard为27 PASS/2 FAIL、合计峰值39.888672GiB。
用户改动、历史与开发FAIL/PARTIAL、85条质量样本保留；未重跑旧矩阵，未push或创建PR。
最终hash、命令索引、资源与Git保护审计见收尾的最终B0。下方历史正文原样保留。

<!-- ENGINEERING-PROGRESS-NAV-END -->

<!-- ENGINEERING-PROTECTED-NAV-BEGIN -->

## 2026-11-07 protected Controller / DP 交付

本轮已从B0推进至X0，交付有单次cap和逻辑累计floor的protected预算、最多双victim
批次及TP2/Graph/真实页复用验证。结论见[本轮收尾](engineering-protected-closeout.md)，
机制见[Controller](engineering-protected-controller.md)、[DP](engineering-protected-dp.md)，
完整correctness/micro/Serving见[统一评估](engineering-protected-evaluation.md)。

Correctness：CPU 77+15项通过，真实Qwen3-8B TP2单/双victim、延迟completion、
36层payload、Graph与真实页面rewrite/reuse通过；DP2独立池/abort/peer continuation通过。
Micro只交付成本测量，未证明加速。Serving保持TTFT≤2s且TPOT≤80ms：TP2三组均
0/24，protected较fixed观测吞吐−88.60%；DP W1/W0正向差异未稳定复现，单项均
SLO_FAIL。组合24/24但零回收，不能归因protected；固定预算与requests继续默认。

usage tie-break缺重复有效同分错选证据，STOPPED / NOT_RUN。历史与本轮FAIL/PARTIAL
全部保留，85条质量样本未扩大，无有损质量保持结论。全部GPU同guard串行、GPU0/1、
Worker主线程CPU20/21，显存未达44GiB停止线。分阶段本地提交、最终证据hash及用户
原有patch保护收入B0 completion manifest；未push或创建PR，下方历史正文原样保留。

<!-- ENGINEERING-PROTECTED-NAV-END -->

<!-- ENGINEERING-REVISION-NAV-BEGIN -->

## 2026-11-01 CB1 / TP 疑似退化复核交付

当前以[复核收尾](engineering-revision-closeout.md)为准。CB1补充实际prefill重算审计，
实现显式prefill_fixed阶段预算，54项CPU与真实TP2/有限A/B通过运行验证；正式三轮
零timed抢占，但吞吐仍较fixed低2.005%，15/24 SLO_FAIL，fixed保持默认。
TP反向两边18/24、dynamic E名义+0.200%；未复现原正向−0.740%/16/24，
保留消除多余E特化的修正，仍不声称稳定Serving收益，原负结果不改写。
源码机制、复算证据与停止依据分别见[Controller](engineering-revision-controller.md)、
[TP](engineering-revision-tp.md)。本轮5次GPU guard PASS，本地分阶段提交；
未push、未改用户原有正文、未扩旧矩阵或85条质量样本。DP/PL2既有结论保留。

<!-- ENGINEERING-REVISION-NAV-END -->

<!-- ENGINEERING-REFINEMENT-NAV-BEGIN -->

## 2026-11-01 本轮修正交付

当前交付见 [重试预算、状态覆盖与TP冷路径收尾](engineering-refinement-closeout.md)。
已实现CB有界重试预算、DP4未确认ADD补计/确认水位，以及TP动态长度特化修正。
CPU72项、GPU reference21项、真实8B TP2与19次资源guard验证通过，阶段提交已落地。

Serving单独判定：固定W1的DP新方案正反向各24/24，相对requests吞吐约+20.7%/
+33.9%，仅本机固定负载的有限证据；W0名义约3%落在波动内。CB仍明显负收益，
TP正式无收益且16/24 SLO_FAIL，已合理停止继续调优；fixed预算与requests路由仍默认。
分项：[Controller](engineering-refinement-controller.md)、[DP](engineering-refinement-dp.md)、
[TP](engineering-refinement-tp.md)。PL2等既有能力保留，85条质量样本不扩，历史FAIL/
PARTIAL不改写。用户原有未提交正文保留，未push或创建上游PR。

下方prompt3的“当前交付”为前轮历史，旧DP4 PARTIAL与CB2判断按当时轨迹理解；
本轮新结果及停止理由以本轮总报告为准。

<!-- ENGINEERING-REFINEMENT-NAV-END -->

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

下一阶段的DP负载链、TP评分成本、Controller成本校准与可选H2O扩展，统一见
[详细工程实验计划](../plans/engineering-optimization-experiment-plan.md)。本次仅交付计划；
尚未实施这些Runtime修改或运行新GPU实验，不改变下列既有结论。

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
| 下一阶段工程优化（待执行） | [路线图](../plans/vllm_physical_kv_reclamation_engineering_roadmap.md) | [详细实验计划](../plans/engineering-optimization-experiment-plan.md) |

本页只维护当前结论和导航；[status](../plans/status.md) 维护当前状态与未决项，
reports保存实验事实与历史负结果，notes保存协议，source-analysis解释当前调用链、
状态所有者和invariant。历史plan只指向结果，不重复维护实验数表。

本次必要CPU回归28项通过，未执行GPU continuation；先前扩大选择时的上游fixture/
async插件失败及适用静态检查均在 [closeout-fixes](closeout-fixes.md) 单独记录。
旧154项、GPU轨迹、microbenchmark和完整profile仍按原报告范围使用，不并入本轮测试数。

</details>

</details>
