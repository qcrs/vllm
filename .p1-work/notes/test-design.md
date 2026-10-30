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
