# Prefill 预算公平性收尾

2026-11-08，起点`eef5da27d1`。已完成同GPU0/CPU20的新基线、容量边界修正和
0/512公平性对照。**短请求排队问题有直接修复证据；本轮不宣称TP加速或整体吞吐
超过fixed。** 完整协议、长短拆分与原始run见[统一评估](engineering-prefill-evaluation.md)，
范围见[计划](../plans/engineering-prefill-plan.md)。

## 实际交付

Serving/DP wrapper显式透传已有`long_prefill_token_threshold`；增加请求ID关联和
CPU monotonic首token分段，复算客户端→Scheduler→首次调度→prefill结果消费→
首sample→客户端。protected full-prefill admission/retry为确实还需输出的请求
计入首个decode token，修复页对齐prompt入场后立即decode缺页的边界。
CPU fixture同时修复未完成prefill就模拟产token及错误max_model_len。

保留最多两轮、真实decode完成门控、完整prefill预留、20%/50%及最多双victim。
未改Worker/TP协议，没有prefill中删KV或新DP算法；默认参数不变。
混合长短负载建议显式512，fixed/protected均应使用相同预算配置。

## 分项结果

| 范围 | 本轮结果 |
| --- | --- |
| Correctness | CPU92 passed /224 deselected；DP本地fixture13 passed /16 deselected；适用pre-commit和manual mypy-3.12通过 |
| 单卡GPU | 八组各24个正式请求均完成、零抢占/重算，FULL_DECODE_ONLY；实际最多单victim |
| Micro | NOT_RUN；未改搬移/TP执行，不重跑旧矩阵，不新增micro加速结论 |
| protected短TTFT | 正向526.6→132.7ms、反向531.1→139.5ms，改善74.80%/73.73% |
| protected长8192 TTFT | 943.4→1168.0ms、947.9→1164.6ms，增加23.81%/22.86%，仍<2s |
| protected完成速率 | 123.108→120.841tok/s、123.004→121.692tok/s，下降1.84%/1.07% |
| 相同512下与fixed比较 | protected吞吐低0.28%/0.02%，没有超过fixed；短TTFT仍高于fixed |
| Serving SLO | 八组均24/24；每请求TTFT≤2s且TPOT≤80ms，最慢逐请求TPOT24.85ms以内 |
| 真实TP2/DP2 | NOT_RUN；本轮仅GPU0可用，历史多卡验证不冒充本轮新组合测试 |

protected0的12个短请求全部等到同波8192 prefill结果消费之后才开始计算，512则
全部提前开始，正反两序一致。Scheduler内部排队中位约409/411ms降至29/29ms；
短prefill本身约40ms增至68ms，改善来自预算分配，不是kernel变快。
fixed也受益，故这是共有调度问题；当前无证据将其归因TP collective错误。

本轮旧配置也24/24，不能声称把历史TP2 SLO_FAIL修为PASS。合成输出序列0/512
一致仅为机械记录，不是质量评测。85条质量样本冻结、历史FAIL/PARTIAL保留。
TP2真实慢路径尚未在本轮资源下复现；当前停止扩大矩阵/参数，DP usage tie-break
仍STOPPED / NOT_RUN，prefill期间删KV继续暂缓。

## 归档、资源与Git

B0：`/data_ssd/qcrs/experiments/p1-v8/Engineering-Prefill-B0/20261108-134721-initial`。
最终：`/data_ssd/qcrs/experiments/p1-v8/Engineering-Prefill-Final/20261108-141656-closeout`。
含源码/hash、测试和静态日志、官方PR核查、正反向8组命令与原始产物索引、
独立逐请求复算、背景采样、用户改动审计及`completion-manifest.json`。
8次GPU guard全PASS，合计峰值17.564453GiB，未达44/45GiB边界。
仅GPU0/CPU20主线程；GPU1占用任务和GPU2均未动，未安装升级环境。

两个protected0的`EngineDeadError`均发生在完成所有请求后的主动shutdown期间；
原日志保留，正式请求无失败。JIT均在完整负载预热内。开发CPU/fixture/lint失败
和首次提交hook失败均保留，不清理失败证据。

本地阶段提交：实现`0246f3c4bc`、分析复算`c1e134a033`；评估/收尾提交见Git及
最终manifest。独立index精确纳入本轮文件，Scheduler原中文注释、notes用户改动、
staged plan/prompt4及plan的unstaged差异保留；status/delivery仅前置本轮导航，
历史正文逐字保留。未push、未创建PR。
