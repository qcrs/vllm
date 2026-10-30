# M2 压力控制源码与状态

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

候选只来自进入本轮前已完成 prefill、无旧 in-flight 的 RUNNING request，q 为0/1，
页为私有非 null。E+q 决定 K，实际 canonical row 长度减 ceil(K/B) 决定 R。
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

当前实现仅 SinkRecent。TriAttention 的评分适配与正式质量/Serving仍未通过，不从
这些小池实验推断 goodput 或质量收益。
