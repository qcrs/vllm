# P1 v8 当前状态与未决项

<!-- ENGINEERING-REVISION-NAV-BEGIN -->

## 2026-11-01 CB1 / TP 退化复核：当前状态

[本轮收尾](../reports/engineering-revision-closeout.md)已完成，代码提交`9af8339c25`。
CB1发现首次重试成功掩盖的后方prefill抢占/重算放大，新增显式prefill_fixed；
54项CPU、真实TP2控制链和有限正式A/B完成。正式三轮零timed抢占，但相对fixed
吞吐−2.005%、SLO15/24对18/24，仍无Serving收益，fixed默认，不扩调参。
TP仅补一次反向：dynamic E相对baseline+0.200%，两边18/24，未复现旧正向退化；
保留冷变体修正，不宣称稳定Serving收益或零回归。prefill/排队TTFT缺口仍在。
分项见[Controller](../reports/engineering-revision-controller.md)、
[TP](../reports/engineering-revision-tp.md)。下方各轮结论按当时证据保留；
未重跑DP/PL2、旧矩阵或质量样本，用户原有patch保留，未push。

<!-- ENGINEERING-REVISION-NAV-END -->

<!-- ENGINEERING-REFINEMENT-NAV-BEGIN -->

## 2026-11-01 重试预算与状态覆盖修正：当前状态

本轮三项修正、有限验证及停止决定已完成，见
[本轮总报告](../reports/engineering-refinement-closeout.md)。

| 方向 | 当前结果与保留边界 |
| --- | --- |
| CB1 retry / retry_pass | 尾槽与真实调度顺序预算、Pool/TP2生命周期通过；正式仍0/24、60次timed抢占，fixed默认；停止扩前瞻/调参 |
| CB2 | 新轨迹4/148次确有触发；基于收益优先级和有限投入STOPPED / NOT_RUN，未证明gate收益 |
| DP4 pending ADD | Engine确认水位精确撤销补计；固定W1正反向各24/24，吞吐相对requests约+20.7%/+33.9%；有限SERVING_BENEFIT，requests仍默认 |
| TP动态E | 消除无必要对齐特化；21项GPU reference和真实TP2通过；正式18/24→16/24、吞吐-0.740%，SLO_FAIL，停止继续优化 |
| 原策略与micro | PL0/PL1/PL2已验证能力保留；历史micro不改写为Serving收益，有损质量仍未证明 |
| 本轮收尾 | CPU72 passed、pre-commit/manual mypy-3.12通过；19次GPU guard PASS；用户原有patch保留，本地分阶段提交，未push |

分项见 [Controller](../reports/engineering-refinement-controller.md)、
[DP](../reports/engineering-refinement-dp.md)、[TP](../reports/engineering-refinement-tp.md)。
下方prompt3区块中的“当前”属于前轮交付；原FAIL/PARTIAL和用户正文保留，
不覆盖本轮新结论。85条质量样本继续冻结。

<!-- ENGINEERING-REFINEMENT-NAV-END -->

<!-- ENGINEERING-NEXT-X0-NAV-BEGIN -->

## 2026-11-01 prompt3实施完成：当前状态

本轮按 [prompt3](../prompts/prompt3.md) 完成必做阶段和条件判断。
总体证据、提交与限制见 [下一阶段收尾](../reports/engineering-next-closeout.md)。

| 范围 | 当前结果与未决项 |
| --- | --- |
| B0、CB0/CB1 | 已实施并验证预算/真实生命周期；CB1 demand Serving 0/24、468次timed抢占，保留实验选项及负结果 |
| CB2 | STOPPED / NOT_RUN；新demand轨迹无重复“无单候选覆盖D”进入证据 |
| TP2a / TP2b | 正确性PASS；完整reclaim micro分别相对前增量约8.5–13% / 40–52%缩短；Serving无稳定收益，grouped正反向均17/24 SLO_FAIL |
| PL0/PL1 / PL2 | 三策略/full逐请求绑定及单engine TP1/TP2安全更新、资源退役已验证；DP更新明确拒绝 |
| DP3 / DP4 | 工作量控制链与显式known_work路由已实施；6次有限A/B后Serving PARTIAL、W1 SLO_FAIL，requests仍默认 |
| X0 | 组合CPU 56 passed / 227 deselected；静态/复算/归档与分阶段本地提交完成，最终HEAD见收尾manifest |

分项见 [Controller](../reports/engineering-next-controller.md)、
[TP](../reports/engineering-next-tp.md)、[Policy](../reports/engineering-next-policy.md)、
[DP](../reports/engineering-next-dp.md)。85条质量样本冻结，有损质量未证明；历史
FAIL/PARTIAL均保留，不将micro当Serving、不将静态registry当安全更新。
本轮未扩旧矩阵、依赖、模型或未测支持域；用户原有待提交正文保留。

以下为本轮实施前的计划及历史结果，其中“尚未实施/待执行”不覆盖上面的当前状态。

<details>
<summary>prompt3实施前的计划及历史记录（原文保留）</summary>

<!-- ENGINEERING-NEXT-X0-HISTORY -->

## 2026-11-01下一阶段：详细计划已形成，尚未实施

后续执行入口为[下一阶段详细计划](engineering-next-stage-plan.md)：
B0 → CB0/CB1缺口预算 → TP2a及条件TP2b → PL0/PL1策略绑定 → PL2安全更新 →
DP3 shadow及条件DP4。CB2门控独立按证据触发，不与预算改动混合。
本次只更新计划与导航，没有新增Runtime、测试或GPU结果；下面的已完成状态保持原义。

## 2026-11-01工程深化：当前状态

本轮按[prompt2](../prompts/prompt2.md)和
[工程实验计划](engineering-optimization-experiment-plan.md)完成，未重启M0→DP。
总体审计、提交与简历叙事见[工程收尾](../reports/engineering-closeout.md)，
当前交付导航见[delivery](../reports/delivery.md)。

| 范围 | 状态 | 当前结论 |
| --- | --- | --- |
| 既有物理回收Runtime/R1/ownership | 保留 | 仍按既有配置与Gate使用，不扩大支持域或重跑旧矩阵 |
| D1 dense DP负载链 | PASS | 发布/独立汇总/归零/本地在途下界完成；CPU与真实DP2通过；未测得可分辨性能收益 |
| D2工作量/压力Router | NOT_RUN | 离线回放仅末次建议变化，未证明重复错选；[停止依据](../reports/engineering-dp-load-chain.md) |
| T0/T1评分融合 | PASS（正确性及局部成本） | 三个真实shape、合成KV的完整reclaim缩短约33–38%；[reference/生命周期/测量](../reports/engineering-tp-scoring.md) |
| T1 pressure Serving | FAIL（SLO） | 正反向无稳定收益；各run15/24，TTFT超标；timed零抢占，warmup另计 |
| T1 short无压力 | PASS（本配置预算） | 各6/6、≤5%无退化预算；不触发scorer，不能归因为融合 |
| TP通信/裁剪Runtime改动 | NOT_RUN | Reduce-root仅benchmark，完整链无稳定收益；保护区比例不足以支持继续裁剪 |
| C0 Controller成本 | PARTIAL | 29次关联，24次首次retry成功，5次preempt后重算admission；缺增量成本/反事实校准 |
| C1/C2 | NOT_RUN | 无重复时间错选或匹配动作成本证据；[Controller报告](../reports/engineering-controller-cost.md) |
| H0源码/H1 | 源码审查完成；GPU probe/H1 NOT_RUN | 当前FA2无per-key统计，未得到满足约束的实现依据；[H2O报告](../reports/engineering-h2o-feasibility.md) |

仍未解决：旧SinkRecent TP pressure为17/24 FAIL，T1不修复其TTFT；历史性能FAIL
及旧profile PARTIAL保留。C0的旧重NSYS轨迹非正式成本基线，不能用inclusive时间
建立已验证预测器。H0停止不表示测得性能FAIL，也不表示H2O不可实现。

85条质量样本冻结，有损质量保持未证明。未扩TP2×DP2、async/spec/connector等
支持域；未新建backend/dispatcher、controller policy或跨池机制。阶段所需实现、
受影响验证与继续/停止判断已完成；源码事实、协议和实验归档均见各报告。

以下折叠内容保留前一阶段正文和用户原有待提交说明，均为2026-10-31历史记录；
其中“未实施/待执行”不覆盖上面的当前状态。

<details>
<summary>2026-10-31功能收尾与计划记录（历史）</summary>

2026-10-31，限定收尾基于 `252c747ad6` 的独立Review执行。
原始环境和工程约束见 [prompt1](../prompts/prompt1.md)；本次范围以用户限定修复要求
和Review为准，未重启M0→DP。当前结论导航为 [delivery](../reports/delivery.md)，
逐项修改和验证为 [closeout-fixes](../reports/closeout-fixes.md)。
功能收尾后，按用户新授权完成一次O1补采；当前性能诊断结果见
[tp-ttft-diagnostic](../reports/tp-ttft-diagnostic.md)。

用户已接受下一阶段方向，本次形成
[详细工程实验计划](engineering-optimization-experiment-plan.md) 与收敛后的
[路线图](vllm_physical_kv_reclamation_engineering_roadmap.md)，尚未开始实现或新GPU实验。

## 当前状态

| 范围 | 状态 | 当前结论 |
| --- | --- | --- |
| R1请求生命周期 | PASS | feature-on在入口拒绝resumable/streaming-input，feature-off的pause→append→admission控制链通过；preempt清L/E后recompute保留 |
| R2、D1–D4 | PASS | Graph有效范围解释、生命周期声明、M2阶段快照、源码导航与文档职责已同步 |
| 必要行为回归 | PASS | 28项CPU测试通过；不代表GPU continuation或payload验证 |
| M0→DP既有实现与Gate | 已执行 | 只按各阶段报告的配置/正确性范围交付；历史记录见delivery导航，不重跑矩阵 |
| O1 | DIAGNOSTIC_DONE | 已审计旧trace并完成稳定ID/step/两rank设备关联；采样出现抢占且偏离原正式轨迹，不足以选择优化；未改Runtime，原SLO FAIL保留 |
| O2/O3 | NOT_RUN | 未测Tri真实Runtime完整评分或证明metadata净收益；未融合、增旁路或扩展实验 |
| 新阶段DP | NOT_RUN | 先修dense负载上报与接收链，再按证据判断工作量/KV压力策略；Prefix Affinity暂缓 |
| 新阶段TP | NOT_RUN | 先测真实shape完整评分成本，再择一优化；与SinkRecent压力TTFT分开归因 |
| Controller深化 | NOT_RUN | 现有coverage/zero-copy/byte-cost排序已完成；保留成本校准方向，有错选证据再改决策 |
| H2O扩展 | NOT_RUN | 可选；先查真实Attention Importance来源与成本，不扩大质量评测或默认支持域 |

feature-on拒绝范围含retention=1。fresh/APC及preempt后重算可按computed length初始化E；
保留有损KV的streaming重入不在契约内。MRV2/BF16/FA2/NHD/B16、单group、单在途与
实际Prefix/Graph/TP/DP组合的边界见delivery。现有ownership、hash保护、source row/E、
结果身份、全rank completion、AsyncOutput/maintenance等待与mutation后fail-stop均保留。

## 明确保留的缺口

- TP pressure仍因TTFT不满足既定SLO；O1已补齐一轮请求级关联，但profiling轨迹有
  6次抢占及重复prefill，不能代表原零抢占正式样本。尚缺原场景代表性成本和优化
  对照收益证据，本轮停止，不追加采样或盲扫batch。
- TP eager只在本机显式CPU放置和对应short/eager配置下通过预算；默认放置及CPU1/49
  微观慢因未普遍解决。DP skewed只在固定6/2路由和对应配置下通过。
- M2/Prefix/TP/DP历史性能FAIL及Graph/TP旧profile PARTIAL保留，详见各report；
  SLO仍为TTFT≤2s且TPOT≤80ms，不因完成路线到DP改为全PASS。
- TriAttention完整Runtime评分/normalization/selection/launch收益与decode metadata构造
  的代表性净收益未量化；未增加融合、noop旁路、backend、dispatcher或DP跨池机制。
- 85条质量样本冻结，有损质量保持未证明；token/payload正确不替代质量保证。
  TP2×DP2、async/spec/connector及其余未测组合不扩张。
- 完整上游suite受既有离线资源和fixture/插件条件限制；具体失败分列于历史final-review和
  本次closeout-fixes，不安装升级依赖或下载替代模型。

R1已封口、必要回归通过，限定功能收尾完成；随后O1诊断也已按范围收尾。
当前未追加Runtime优化；新阶段已形成计划，实施与实验结论后续按独立增量记录。

</details>

</details>
