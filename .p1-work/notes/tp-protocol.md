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

## T0/T1工程深化协议（2026-11-01）

以上是既有TP Gate协议；本轮只验证normalization/GQA/head reduction融合的新增风险，
没有重复慢rank、故障或Graph组合矩阵。单卡评分reference在物理GPU1，真实TP2
在GPU0,1；Worker主线程分别CPU20/21。沿用guard串行和项目合计显存限额。

micro复用`benchmark_kv_scoring.py --runtime-shapes`，输入来自旧Tri Runtime
intent与commit，E/keep/P/recent_start/L分别为2051/1025/16/2035/4100、
4100/2050/16/4084/4100和8194/4097/16/8178/8194。L必须取本次
`reclaim_commit.logical_tokens`，不能取上次settled值4099/8193或用E代替。
36层、每rank4个KV heads/16个statistics heads，BF16 stride
`[16384,256,1024,1]`；校准真实、KV合成，不称在线真实模型耗时。

每阶段3次预热、10次CUDA Events，按最慢rank中位数解释；restore在计时外。
独立测phase、raw+phase、normalization/reduction、score collective、stable selection、
plan broadcast、complete plan/reclaim；独立阶段不能相加。complete reclaim包含
row/E和completion，不含正常forward、Scheduler commit或allocation retry。
只在benchmark比较Reduce-to-Root，目标从group映射到global rank；Runtime继续AllReduce。

真实TP2轨迹采用TriAttention、两次回收及继续decode，检查全层retained payload、
其他请求不变、commit前未free、poison后真实rewrite/reuse与最终free queue。
脚本字段`early_free_observed=true`表示提前检查已执行且通过，不表示发现early free。

Serving使用固定TriAttention calibration/retention与相同prompt/arrival，pressure
long/full、batch1024、.75GiB/rank、seqs8、完整负载预热后3轮，正向A/B及一次
反向B/A全部保留。no-pressure只一组short/eager、8GiB/rank、batch2048对照。
正式trace off；warmup与timed的preemption分别报告。token一致不替代质量保证。

参数、hash、复算和结果见 [工程TP报告](../reports/engineering-tp-scoring.md)。首组A
缺dirty scorer即时快照，只能引用事后恢复来源；反向A/B与micro保存实际源码。
最后benchmark设备API/workspace统计收尾未重跑GPU，实测harness以原归档为准。

## 2026-11-01 Refinement：TTFT 与动态长度

用 `profile_tp_ttft.py --nsys-child` 可仅采同主机 host 时间戳；helper 即使绕过
Nsight 也归档。`analyze_tp_ttft.py --host-output` 划分入口、初始排队、own prefill
step、段间等待与输出送达，校验总和及原请求 TTFT；inclusive step 含其他请求和
maintenance，host 调用含提交/等待，nested phase 与两个 TP rank 不能相加。
prefill 的 pure GPU/通信归因仍需设备证据，不由 host 剩余量推断。

一次 host-only 零抢占轨迹发现 E4160 触发分组评分的长度对齐特化，故只取消两
个 grouped kernel 的 E 特化。先独立参考/同 scorer E 跨对齐 cache audit，随后
原8B TP2 continuation/free/reuse；有限正式 A/B 保留原 long/full/pool/batch/
affinity/预热及3轮条件，不清 Triton 缓存、不扫 chunk 或将诊断数值当正式收益。
`benchmark_kv_scoring.py --length-specialization-audit` 只观察进程 cache miss，
不能用两次不同磁盘缓存状态的首调用时间计算优化百分比。

## 2026-11-01 Revision：反向与阈值复核

用户对正式−0.740%/16/24疑似回归提出复核，本轮只补一次相同协议反向B/A，
每次完整预热+3轮，不修改内核或清磁盘缓存。反向dynamic E相对baseline+0.200%，
两边18/24且24条token IDs一致；不能选其中一方向报告收益，也不足以宣称稳定回归。
保留冷变体修正和原正向负结果，prefill/排队SLO缺口独立保留；具体范围及hash见
[反向报告](../reports/engineering-revision-tp.md)。
