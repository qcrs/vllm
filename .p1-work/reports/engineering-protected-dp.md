# Protected Controller 与 DP usage 诊断

本轮 DP pressure tie-break **STOPPED**：固定 W1 轨迹没有本地 allocation miss，
不满足 prompt4 要求的重复同分错选及 peer 可服务余量证据。requests 继续默认，
known_work 保留 opt-in；没有新增 usage 权重、路由模式或快照超时规则。

## 源码核查与进入条件

`Scheduler.get_dp_load_snapshot` 的 U 为
`sum(max(0, num_tokens - (num_computed_tokens - num_in_flight_tokens)))`，只含已知
未完成工作。`KVCacheManager.usage` 来自 BlockPool 的
`1 - free_blocks/(num_gpu_blocks-1)`；free 包含可驱逐缓存页，不是 admission 保证。
本轮 APC off、同池大小，usage 可以作压力近似，但不能代替完整分配预算。

Engine counts 变化立即发布；仅 U/usage/watermark 变化按100ms合并，空闲不发
heartbeat。sequence、step、生成/接收时间和每 client ADD watermark 一起传播，
Frontend 仅移除自己被确认的 pending prompt。Runtime 没有 wall-age expiry，不能
把诊断的250ms筛选伪称为已有新鲜度合同。选路仍为
`S=max(4W+R,F)+(snapshot.U+unacked_pending_prompt)/batch`，同分按原有轮转。

诊断复用真实发布/Coordinator/Frontend，选路一条记录；Scheduler 仅每100ms或
counts变化记录 scalar snapshot/free，allocation miss 和 pressure action 才额外写行。
没有 payload clone、GPU同步或逐请求逐step全状态 trace；正式 Serving 关闭该诊断。

## 真实 DP2 W1

原始目录：
`/data_ssd/qcrs/experiments/p1-v8/Engineering-Protected-DP-Diagnostic/20261107-211754-w1-known-work-usage`。
命令由该目录 `invocation.json` 保存：`dp_serving.py --scenario w1 --routing known_work
--load-diagnostic --work-counters --policy sink_recent --pool-gib 4
--max-num-batched-tokens 512 --max-num-seqs 8 --worker-cpus 20 21 --repeats 2`，
外层沿用 GPU0/1 与 run_guard。完整 warmup 后两轮；该受扰动运行不作正式性能。

复算入口 `scripts/analyze_dp_pressure.py <run>`；结果 `pressure-analysis.json`：

- 25次路由（含small/full warmup），22次有完整有效的时间链；全部实际选路符合
  含 pending ADD 的 work score 与轮转顺序。
- 3次 score 同分，仅1次有效且诊断年龄≤250ms时选中较高 usage；0次本地 miss、
  0次 reclaim/preempt action，因此没有重复 pressure-causing misroute。
- 两副本最终 counts=[0,0]、U=0；不推断改变那一次同分选择后的服务结果。
- guard PASS、85.46秒、NVML项目合计峰42,830,135,296 bytes，无阈值触发。
- 最终源码的 DP CPU route/watermark/dense/MoE 选择15 passed / 36 deselected。
  只替换两个无缓存顶层 tokenizer fixture 为本地 Qwen3，不下载依赖或模型。

新报告不覆盖历史 DP4 PARTIAL/SLO_FAIL、pending ADD 修复的有限正反收益或 W0
噪声范围结论。后续 W1/W0 正式复测及 known_work+protected 组合另见
[统一评估](engineering-protected-evaluation.md)；85条质量样本继续冻结。
