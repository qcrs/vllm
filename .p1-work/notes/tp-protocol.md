# TP2 验证协议（实施前固定）

## 目标与不变量

一个 Scheduler/KVCacheManager/BlockPool 对两个本地 GPU Worker；物理 GPU0,1，
Qwen3-8B BF16/MRV2/FLASH_ATTN/B16。只放开 TP2，PP/DCP/PCP、async/spec/connector
等已有门槛保持不变。每次运行由 run_guard 串行监测两卡项目显存之和，44 GiB
终止阈值、45 GiB硬上限；不操作 GPU0 的其他任务。

每 rank 消费同一 source row/E/P/K；本地 backing 地址不同。Tri 的每个 distinct
KV head 使用完整 GQA calibration group，先对 eligible members 做 query-head
z-score，再 GQA MAX；本地贡献按全局 KV-head/layer 数归一，一次 FP32[E] SUM。
rank0 stable top-K 后广播 int32[K]；Random同样只在rank0生成计划。SinkRecent
整数区间可重复生成，诊断仍逐元素核对。复制KV group只指定一个scoring owner，
owner使用完整group stats，不能只加载本地Q shard。

最后本地payload/row/E写后，在相同main stream记录local event，再执行专用
标量completion collective。AsyncOutput copy event或q0 maintenance event必须
覆盖该collective；Scheduler收到成功结果后才canonical commit/free。
`unique_reply_rank=output_rank`本身不提供all-rank完成证据。

## 测试设计四问

- 模块用途：将全局unique-head评分转换为一份shared KeepPlan，并在所有本地写完成后发布成功结果。
- I/O：rank-local K/stats、全局head mapping、同一intent；输出各rank相同plan/row/E，以及覆盖全组的device completion。
- 防止的失败：遗漏其他head或复制计权；遗漏复制group的非本地Q stats；浮点top-K分歧；output rank先完成导致early-free；rank故障后仍提交释放。
- 最低成本层级：扩展已有compaction/scoring suite，用独立CPU复数reference检验sharded/replicated评分和稳定ties；两卡真实collective fixture验证统一plan；8B MultiprocExecutor验证延迟rank、失败、真实canonical release/rewrite/reuse和下一forward。

## 运行顺序与证据

先 CPU/GPU scorer contract；再TP2 Eager的whole/token、Random/Tri两次回收，
记录全36层source snapshot/retained bit-exact、其他请求不变、各rank plan/E/row。
释放后才poison，再由C的真实模型写入证明全rank复用；A继续decode并检查logical
positions与physical slots。单进程FP32 attention检验本地shard；跨TP拓扑不要求
生成逐bit相同。

慢rank诊断在非output rank的本地compaction前注入受控延迟，先完成rank记录
local event；主控在该事件已完成、另一rank尚未完成时观测free增量=0和旧canonical
row不变，最后all-rank ACK才free。失败注入在非outputrank进入破坏性写入前抛错，
要求executor fail-stop、无canonical commit或reuse；不能把异常当普通fallback。

必要组合依次为Prefix+TP Eager、Graph+TP（含慢rank）、Prefix+Graph+TP token smoke。
Graph诊断检查每rank持久地址/graph对象/capture count、padding/reorder、replay与
matched-bound eager参考。组合不重复全量有损质量矩阵。

Serving以TP2 native为基线，同Pool/同负载/同预热，报告每轮/聚合TTFT、TPOT、ITL、
tokens/s、Goodput、preemptions及两卡NVML合计。使用现有TTFT≤2s、TPOT≤80ms、
无压力noop开销≤5%预算；有损结果与质量区分。另以NSYS诊断score SUM、plan广播、
completion三类通信和rank等待，不将profile时间混入正式Serving。
