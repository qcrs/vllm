# P1 v8 当前状态与未决项

2026-10-31，限定收尾基于 `252c747ad6` 的独立Review执行。
原始环境和工程约束见 [prompt1](../prompts/prompt1.md)；本次范围以用户限定修复要求
和Review为准，未重启M0→DP。当前结论导航为 [delivery](../reports/delivery.md)，
逐项修改和验证为 [closeout-fixes](../reports/closeout-fixes.md)。
功能收尾后，按用户新授权完成一次O1补采；当前性能诊断结果见
[tp-ttft-diagnostic](../reports/tp-ttft-diagnostic.md)。

## 当前状态

| 范围 | 状态 | 当前结论 |
| --- | --- | --- |
| R1请求生命周期 | PASS | feature-on在入口拒绝resumable/streaming-input，feature-off的pause→append→admission控制链通过；preempt清L/E后recompute保留 |
| R2、D1–D4 | PASS | Graph有效范围解释、生命周期声明、M2阶段快照、源码导航与文档职责已同步 |
| 必要行为回归 | PASS | 28项CPU测试通过；不代表GPU continuation或payload验证 |
| M0→DP既有实现与Gate | 已执行 | 只按各阶段报告的配置/正确性范围交付；历史记录见delivery导航，不重跑矩阵 |
| O1 | DIAGNOSTIC_DONE | 已审计旧trace并完成稳定ID/step/两rank设备关联；采样出现抢占且偏离原正式轨迹，不足以选择优化；未改Runtime，原SLO FAIL保留 |
| O2/O3 | NOT_RUN | 未测Tri真实Runtime完整评分或证明metadata净收益；未融合、增旁路或扩展实验 |

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
当前未追加Runtime优化，后续只在明确新目标或新证据下重开。
