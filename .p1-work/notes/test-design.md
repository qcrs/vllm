# P1 v8 测试设计

## M0 / M1 Payload

- 模块用途：将 current-member 有序 KeepPlan 映射到 retained KV；全层/Head 使用同一集合。
- I/O：真实 packed 4D Layer views、source row、升序 current-member indices；输出 valid retained bytes 与合法 retained row。无效 tail 无零值契约。
- 防止的失败：非连续 page ID、padding stride、partial tail、同 tile/跨 tile overlap、stale released-page read、K/V channel 混淆。
- 最低成本层级：小型 CPU scalar Oracle 独立逐元素寻址；GPU kernel fixture 与该 Oracle bit-exact 比较；真实 release/reuse、下一 decode 和 completion 在 Engine 验证。

附近 `test_gpu_block_table.py` 只覆盖 metadata staging，不能检验 Payload 搬移，因此新增 `test_kv_compaction.py`，后续复用同一 Oracle/fixture。Oracle 不 clone Pool、不进入 Serving。

## M0 allocator

沿已有 `tests/v1/core/test_scheduler.py` / `test_single_type_kv_cache_manager.py` suites 扩展真实 requirement/Pool 行为；Engine native/instrumentation 比较 token IDs。ID trace 不触发 GPU sync，Serving 默认只计数。

## M1 canonical ownership

- 模块用途：消费已完成的 Worker row/E，保留 canonical Block 对象并释放其余页。
- I/O：原 Request、源 row/长度、完成结果；输出新 ownership/E 和真实 free queue。
- 防止的失败：重复、null、外部、过期 ACK 导致错 free；按 logical L 重新申请已丢弃页；EOS/abort 双重释放。
- 最低成本层级：复用 `test_kv_cache_utils.py` 的真实 Pool 和 Request helper，验证 commit 后按 E 增长并被 B 复用；GPU完成、EOS/abort 和 preemption 在 8B Engine 实验验证。

## M2 pressure controller

- 模块用途：在真实allocation miss时选择一个可回收decode请求，保留partial batch并仅延迟一次重试。
- I/O：allocator同一计算点的required-available、当前ownership/E、已排定q；输出唯一intent或原生调度行为。
- 防止的失败：以token数量冒充真实释放页、排序不确定、丢掉partial allocation、反复回收而不执行native fallback。
- 最低成本层级：pure候选排序在既有KV suite检验；真实8B Engine小池验证miss→完成→release→下一step重新allocation，另用Scheduler真实Pool构造running miss与一次重试。

## M2 shared scoring

- 模块用途：按实际paged K_rot与校准Q频域统计产生唯一current-member有序集合。
- I/O：packed4D/真实stride、nonmonotonic row、GPU logical L_after、各层/统计head数据，输出FP32 shared scores与int32[K]计划。
- 防止的失败：用E作为时间、Qwen half pairing错误、query-head统计提前平均、GQA计权重复、保护域影响z-score、同分选择不稳定。
- 最低成本层级：在现有GPU compaction suite内用独立CPU FP32 dense复数计算比较raw/shared score与计划；包含query-head和KV-head stats、非连续页、padding、保护窗口及零方差。之后8B真实K与GPU logical状态验证。
