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

## 最终组合 lifecycle 与一次超时

最终源码的 `known_work + protected` 组合在独立33页池、DP2×TP1 eager 下验证，
真实 GPU0/1、Worker主线程CPU20/21，所有 GPU 经同一 guard 串行。
`Engineering-Protected-Final-Correctness/20261107-213442-dp2-known-work-local-pressure`
原 guard FAIL/exit1、87.71秒，峰34,154,217,472 bytes，无资源阈值触发。
旧 harness 等待 C 首次输出才启动 B 和 abort A；protected 每次精确释放 D1，
前方 A 的下一次 decode 先重新使用该页。实际100次 commit、199次 miss 后仍未
满足 C 的启动等待条件。这是等待 admission 的真实限制，不归因为超时噪声。

只调整 correctness harness 的事件先后：protected 观察本地 commit 后启动 peer B，
B 产生16 token 后按原 affinity abort A，再验证 C 完成；fixed 的原等待条件保留。
没有修改 Runtime、正式 Serving workload、预算或超时阈值。
`20261107-213909-dp2-known-work-pressure-abort-reuse` PASS/exit0、35.20秒，
峰34,183,577,600 bytes：

- replica0 首次 miss D1 时 peer 有32 free，独立池和 foreign mutation 检查通过。
- 本地两次 reclaim，A 与 C 均真实改写释放页17，全36层 payload 验证通过。
- A 生成4 token 后 abort；C 在 abort 前未开始，随后完成64 token，B 完成128 token。
  peer 在 abort 后继续112次已验证 forward。
- 四个后续请求走 known_work，两个副本均处理请求；最终 free=[32,32]。

该 PASS 验证池隔离、abort affinity、continuation 和真实页 reuse；不覆盖“等待
请求在回收后立即 admission”的主张，也不属于 Serving 性能或有损质量证据。
原 FAIL 和修订后的 PASS 分开保存在 B0 `final-correctness-addendum.json` 及原目录。
