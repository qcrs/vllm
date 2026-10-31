# M2 压力控制阶段记录

## 动机、问题与假设

M1 已证明确定触发的 payload/ownership/reuse。此处验证真实 allocation miss 是否在
原生 preempt/break 之前发出唯一 intent，保留 partial batch，在完成/free 后只延迟
一次正常 schedule；下一次仍不足则回到原生处理。L 不因回收回退，E 决定新需求。

## 环境与命令

vLLM v0.26.0 pin `568afb3a13806beb53bb2e6bd518269357b237c0`，M1提交
`bdc401bcab` 加各 run 的 invocation.diff/source 快照。Python3.10/Torch2.11.0+cu129/
Triton3.6.0，MRV2、BF16、FLASH_ATTN、B16、TP1、Eager、APC/spec/async/connector off，
单在途，full-ISL reserve off。模型 `/data/models/Qwen3-8B`，物理GPU1 A10080GB。
全部GPU实验由 `.p1-work/scripts/run_guard.py` 串行记录NVML，44GiB停止自己的进程组。

基础命令：`source .p1-work/scripts/env.sh` 后执行
`P1_RECLAIM=1 P1_CONTROLLER=1 P1_RETENTION=0.5 P1_PRESSURE=1 P1_TRACE=1
.venv/bin/python .p1-work/scripts/run_guard.py --stage M2 --name NAME -- @python
.p1-work/scripts/native_baseline.py`。每次完整命令、环境与源码在对应 invocation.json。

## OBSERVED

实验根 `/data_ssd/qcrs/experiments/p1-v8/M2`。各目录均有 stdout、KV trace、outputs、
NVML、结果及源码快照；脚本要求 A/B 都实际生成并完成。

| 目录后缀（日期前缀见原目录） | 场景和观察 | 状态 |
| --- | --- | --- |
| `20261030-235255-pressure-sink-recent` | WAITING B缺1页；A本轮q1，E261→130，释放8页；B下一轮获得全部8个released IDs | PASS |
| `20261030-235827-running-partial-sink-recent` | RUNNING缺1页/free0；A q0维护，E256→128，整页释放8页；B后续获得released ID8 | PASS |
| `20261031-000012-running-successful-partial-batch` | A已成功排q1，B RUNNING miss；保留A本轮forward，再回收A整页；B后续获得released ID8 | PASS |
| `20261030-235911-pressure-sink-recent-reserve4` | 与首个WAITING轨迹相同，G从1变2；固定rho仍释放8页 | PASS |
| `20261031-000107-bounded-retry-native-preempt` | 5页池，A31/B32 prompt；释放1页后下一轮A先占回该页，B仍缺页而原生preempt；两请求最终完成 | PASS |

前4项峰值17,098,080,256 bytes（15.92GiB）；最后一项17,028,874,240 bytes。
未触发阈值。ID观测来自真实allocation_done，不修改free queue。
小池最后一项可反复形成新的压力事件，并出现多次preempt；仅保证每次回收之后那一轮
不能再插回收，不将“最终完成”解释为无抖动或性能改善。

CPU命令：`.venv/bin/python -m pytest tests/v1/core/test_scheduler.py
-k 'pressure or allocator_trace' tests/v1/core/test_kv_cache_utils.py
-k 'pressure or allocator_trace or reclamation' -q`：16 passed /199 deselected。
真实Pool测试验证 partial成功列表、maintenance空batch、waiting重试、无收益与
未完成prefill原生处理、R0与成本选择。修改文件的全部pre-commit hooks通过，首次ruff
format更改一处格式后复跑，不跳过hook。

## 分析与边界

真实短缺已贯通M1的完成/提交/free链，下一轮确实重新计算allocation。R按物理页，
候选自身下一步页增长可能吃掉R；保留native fallback是必需语义。reserve对照的预算
未改变选择/R，不能声称headroom带来收益。小trace与首次Triton编译不用于Serving速度比较。

本记录不重做M1全层字节/FP32 Attention证据，不以CPU凭证测试代替GPU completion。
TriAttention共享评分与质量/Serving结果分别记录在`m2-scoring.md`、
`m2-evaluation.md`；其边界见对应报告。本报告只验收M2；后续Prefix/Graph/TP/DP
的独立证据见对应报告与`../plans/status.md`，不从M2结果继承兼容性结论。
