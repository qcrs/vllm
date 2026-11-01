# M2 压力控制源码与状态

## 2026-11-01 CB0 / CB1 与请求绑定增量

当前 [Controller报告](../reports/engineering-next-controller.md) 保留预算正确性及
Serving负收益；下文为原M2历史。`Scheduler._select_reclaim_candidate`一次扫描
构造合法候选快照，决策和显式trace消费同一份E/L/N/q/P/row/K/释放页/复制字节。
allocator的required/available/shortfall来自同次allocation；没有GPU回读或重复Pool扫描。
正式trace-off不写逐候选JSON。离线 `analyze_tp_ttft.py --candidate-output` 关联
intent→commit/release→首次retry与后续native动作，不将preempt后admission算retry成功。

`demand_reclaim_candidates` / `retarget_reclaim_candidate`在CPU上以fixed结果作为
K_floor，显式 `budget_mode=demand`依次求G和D对应最大合法K：
`min(E-1,B*(N-T))`，且K≥K_floor、实际可释放≥T；无单候选覆盖D保持原最大释放行为。
exact whole-block检查只在该K进行，不降低K换零拷贝。fixed仍默认；G的reserve是
软目标，D是当前硬缺口，均不能保证下一轮decode/admission状态不变。
320次新demand候选覆盖D，故CB2门控STOPPED/NOT_RUN。大量后续retry miss说明
覆盖当次D不等于后续成功；未通过临时reserve修补掩盖负结果。

PL0/PL1后 `EngineCore.add_request` 绑定host不可变 `PolicyBinding`；Scheduler
预算/保护区/zero-copy从请求绑定读取，`ReclaimRequest`和`ReclaimResult`传同一版本。
Worker slot只持有binding值，policy/stats/workspace按engine-local版本共享，full
请求不成为victim。preempt/recompute保版；全rank完成、source row/E身份及canonical
commit/free不变。retention为当前E域单次保留下限，不是历史累计质量保证。
PL2更新/资源退役边界见 [TP记录](tp-runtime.md) 和
[策略报告](../reports/engineering-next-policy.md)。

## 原 M2 阶段记录

本页保留压力控制提交 `d0b3ca3e91` 的阶段假设；`5f9f96d9d0` 补入评测导航。
下文“全私有页”和“仅SinkRecent”描述该阶段，不是最终支持契约。最终调用链以
[`Scheduler._select_reclaim_candidate` / `make_reclaim_request`](../../vllm/v1/core/sched/scheduler.py)
为准：P保护到最后shared整页末，含此前的exclusive gap；K按受保护prefix加私有suffix
预算推导。完整推导与取消hash发布见 [prefix-runtime](prefix-runtime.md)，
多policy的Worker评分/selection见 [m2-scoring](m2-scoring.md) 和
[`KVReclamationWorker.reclaim`](../../vllm/v1/worker/gpu/kv_reclamation.py)。

## 调用链与所有权

`KVCacheManager.allocate_slots` 在原有 `required_blocks / available_blocks`
计算点保存当次 `last_allocation_shortfall`，包含 waiting watermark 和 reserved。
该字段只是最新 allocation 的临时结果，不预测下一轮需求。Core 配置拒绝
full-sequence reservation，因此不会把该路径的 None 混为当前缺页。

`Scheduler.schedule` 在 RUNNING miss 的原生 preempt 之前及 WAITING miss 的
原生 break 之前调用 `_select_reclaim_candidate`。已有成功 allocation 和
`num_scheduled_tokens` 留在原公共收尾流程中；停止继续 admission。
`_update_after_schedule` 完成后才附加 intent，此时 q/in-flight 与 M1 contract 一致。
Worker 仍走 M1 的 post-forward/maintenance、GPU completion、canonical commit/free。

候选只来自进入本轮前已完成 prefill、无旧 in-flight 的 RUNNING request，q 为0/1。
M2初始阶段要求页全私有且非 null；最终Prefix路径允许只读shared保护域，mutable
suffix仍须私有。E+q与P决定 K，实际 canonical row 长度减 ceil(K/B) 决定 R。
R<=0 不触发回收。SinkRecent 用区间推导零拷贝和 changed members，不扫描 KV。
同一个 miss 的 G=shortfall+min(reserve_cap,live_decode_count)。
`choose_reclaim_candidate` 先取覆盖 G 的候选，按零拷贝、C/R、R降序、到达时间、ID
排序；无覆盖时先取最大 R。C/R 用整数交叉乘，不依赖浮点排序。

`reclaim_retry_pending` 是 Scheduler 唯一新增控制标记。发 intent 后置真，下次
schedule 消费并清除；该轮仍 miss 时直接执行原生 preempt/break。再后续轮次允许
新的压力事件，所以它不是生命周期内只允许一次回收，也不保证消除所有 preempt。
下一轮重新调用 allocator，尤其 K 对齐时候选自身通常立即需要增长一页。

## 验证层次

`test_scheduler.py` 复用已有 create_scheduler/create_requests，使用本地 Qwen3Config，
真实 Pool 与 Scheduler；只在输出边界提供完成凭证。GPU 完成与真实模型仍由8B实验验证。
覆盖已有 partial batch、q0 maintenance、waiting admission、R0、未完成prefill和
一次重试仍不足时的原生 preempt。`test_kv_cache_utils.py` 覆盖覆盖量与成本的选择。

初始小池验证只使用SinkRecent；最终intent传递source E/row、K和P，Worker按
`ReclamationConfig.policy` 执行SinkRecent、Random或TriAttention-inspired shared
scoring。评分链见 [m2-scoring](m2-scoring.md)，历史质量与HTTP Serving事实见
[m2-evaluation](../reports/m2-evaluation.md)；小池控制轨迹不证明Serving goodput或质量收益。
