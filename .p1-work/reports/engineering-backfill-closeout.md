# Protected Admission Backfill 收尾

2026-11-08，起点 `6b06d384b5`。已实现有界 admission backfill，并完成单卡 GPU1 和
TP2 GPU1/2 对照：**短请求不再被装不下的长队首连带阻塞，长请求容量问题尚未解决。**
统一结果及原始目录见[评估](engineering-backfill-evaluation.md)，范围见
[计划](../plans/engineering-backfill-plan.md)。

## 实现

实现提交 `0ca2e5c80d`：新增默认0的`admission_bypass_limit`，仅
pressure/protected + FCFS显式开启；队首分配失败且无合法reclaim plan时让位，
最多两个后方请求成功入场，额度耗尽后停止后方新admission，队首入场/abort清状态。
补齐跳过队首后DP capacity-bound状态的保留，abort清理用O(1) request查询。

复用原Scheduler队列，不改Worker/TP commit contract、完整prefill准入、
reservation、单次20%/逻辑50%保护或最多双victim。参考
[vLLM #33499](https://github.com/vllm-project/vllm/pull/33499)与
[SGLang #45685](https://github.com/sgl-project/sglang/pull/45685)，未创建上游PR。

## 分项结果

| 范围 | 结果 |
| --- | --- |
| Correctness | 本地Qwen配置fixture107 passed /194 deselected；额度、恢复/abort、DP guard及既有回收行为通过；适用pre-commit/manual mypy-3.12通过 |
| GPU correctness | 本轮六次TP1/TP2 Serving完成；真实多victim、延迟completion、Graph/页复用及DP2生命周期沿用上一轮证据，未重跑 |
| Micro | 本轮NOT_RUN，不声称决策/执行链加速 |
| TP2 Serving | strict→backfill：6/24→18/24 SLO；107.40962→107.13192tok/s；fixed24/24、124.39958tok/s |
| 单卡 Serving | strict→backfill：6/24→18/24 SLO；101.05466→100.65318tok/s；fixed24/24、124.92321tok/s |
| 请求进展 | 两种部署128-token均0/12→12/12，8192-token仍0/6；SLO失败均来自TTFT，所有组零抢占/额外重算 |
| DP | 仅修backfill与capacity guard交互，无新DP2性能实验；usage tie-break仍STOPPED / NOT_RUN |

SLO仍TTFT≤2s且TPOT≤80ms。新mixed是合成调度诊断负载，不代替质量数据集；
85条冻结样本未扩大，未新增质量保持结论。TP2 backfill零回收，改善归因admission
顺序。backfill限制绕过次数，不保证无限输出下墙钟无饥饿；fixed/backfill0仍默认。

## 结论与停止

单卡匹配逻辑容量后复现TP2模式，说明容量准入和队头阻塞足以解释主要退化。
TP2通信或执行不是这一退化的必要条件，当前也未发现collective正确性故障。
fixed能释放更多页，protected合法可释放容量不足以接纳长请求；绕行修复短请求
被连带拖慢，不能消除物理容量缺口。停止扩大参数搜索，不把局部SLO改善包装为
总体吞吐成功。prefill中删除KV按用户要求暂缓，NOT_RUN。

## 归档与Git

B0：`/data_ssd/qcrs/experiments/p1-v8/Engineering-Backfill-B0/20261108-091924-initial`。
最终归档：`/data_ssd/qcrs/experiments/p1-v8/Engineering-Backfill-Final/20261108-112114-closeout`。
包含测试/静态日志、Serving独立复算、源码provenance差异、提交日志和
`completion-manifest.json`。开发测试失败、离线配置失败和复算脚本初次失败保留。

每种部署三组同源码、三轮测量；最后abort查询/DP guard修正及格式化晚于GPU测量，
最终CPU检查通过，未作最终字节一致的GPU复验。GPU0未用，六次guard PASS，项目
合计NVML峰值18.5625GiB。实验后GPU2已有他人任务，未使用或终止该任务。

只提交本轮4个实现/测试/测量文件及5个计划/报告/导航文件。独立index提交，保留
用户staged plan/prompt4、未暂存notes/plan及Scheduler原注释，历史报告未改写。
最终commit IDs、文件hash和原用户patch对比见completion manifest。
未push、创建PR、重跑旧矩阵或扩大质量样本。
