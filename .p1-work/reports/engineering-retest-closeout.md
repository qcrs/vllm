# GPU1/2 空闲基线与压力复测收尾

2026-11-08，使用用户指定的物理GPU1单卡和GPU1/2的TP2、DP2重建当前源码基线。
本轮没有改Runtime实现；验证了已有两轮protected、完整prefill及首decode容量预留、
backfill2和512-token prefill限额。完整参数、逐请求拆分和正反向数据见[统一评估](engineering-retest-evaluation.md)，
执行范围见[复测计划](../plans/engineering-retest-plan.md)。

## 正确性

CPU回归92项、DP本地fixture 13项通过；适用pre-commit和manual mypy-3.12通过。
真实GPU1/2 TP2双victim与延迟completion、保护边界、全rank同步、Graph/eager/FP32
attention对照、13页释放后改写复用通过。DP2等待者在abort前启动并复用本地释放页，
peer在abort后继续112次已验证forward。Correctness PASS表示被测执行与生命周期正确，
不表示Serving更快或质量已验证。

## Micro

88页缺口的真实TP2 Worker代表形状中，protected第二轮释放41页、复制0字节，Worker
中位0.341504ms、compaction 0.014336ms；CPU决策中位111.3685µs。它不含两轮间真实
decode和正常模型forward，且与fixed释放页数不同，不能据此宣称完整策略提速。

## Serving

项目SLO保持每个请求同时TTFT≤2s、TPOT≤80ms。关键结论：

- TP1与TP2 baseline档fixed/protected均24/24；完成速率差分别约−0.37%和−0.26%，未证明整体加速。
- TP1 burst protected比fixed低42.59% tok/s，联合SLO由21/24降至9/24。
- TP2 burst protected的SLO由9/24升至15/24，但吞吐正反向分别下降27.62%和28.22%；改善来自更多短请求达标，8192-token请求仍未达标。
- DP2 W1混长短负载中，known_work相对requests在4 GiB与1.5 GiB固定路由对照均正反向提高吞吐，SLO也改善。该收益来自既有路由，不归因protected Controller。
- DP2 W1/1.5下protected相对fixed正反向吞吐均下降约31%至44%；requests路由SLO为19/24或21/24降至18/24，known_work由24/24降至18/24。
- DP2 W0正向known_work只高4.85%，两边24/24；反向requests组没有正式样本，因此保留PARTIAL，不声称稳定收益。

综合来看，当前protected Controller满足实现正确性，但没有证明Serving整体收益；小池压力
下它会以TTFT等待和吞吐下降为代价。known_work对W1混长短负载有可重复的路由收益，
与物理KV回收分开归因。DP usage tie-break为STOPPED / NOT_RUN：缺少新鲜同分错选后
本地miss且peer有容量的重复证据。prefill期间删除KV仍未实施。

## 资源与归档

空闲批次归档在
`/data_ssd/qcrs/experiments/p1-v8/Engineering-Retest12Clean-B0/20261108-170517-idle-gpu12`，
逐run原始产物分别位于`Retest12-Correctness`、`Retest12-Micro`、
`Retest12-Serving`和`Retest12-DP`。29次guard为28 PASS、1 FAIL；项目显存峰值
39.888672 GiB，低于44 GiB停止线。该批正式窗口背景审计未发现外部GPU计算。唯一失败
是DP W0反向requests组在Serving开始前，TCPStore绑定`127.0.0.1:37217`遇到
`EADDRINUSE`。其后GPU2出现另一用户的大显存进程；没有触碰该任务，也没有在该背景
下追加GPU运行。逐请求Serving均按正式结果保留；主动shutdown后的EngineDeadError
及更早fixture失败也未从原始审计中删除。

此前GPU0/1共享背景批次、历史FAIL/PARTIAL均保留并与本次空闲批次分列。未重跑旧矩阵，
未扩大85条质量样本，不作质量保持结论。本轮只形成评估/收尾文档，没有Runtime代码变更。
本地按评估与收尾导航分阶段提交；仅暂存本轮文档，用户已有staged/unstaged改动保留，
未push、未创建PR。
