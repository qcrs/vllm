# P1 v8 测试设计

## R1 resumable请求入口

- 模块用途：在reclamation支持域之外拒绝保留KV的streaming生命周期。
- I/O：`Scheduler.add_request` / `AsyncLLM.add_request` 接收resumable请求或AsyncGenerator；feature-on抛清晰错误且不改变请求、队列、Pool或前端状态，feature-off按原生session控制链执行。
- 防止的失败：allocator按retained E分配，而session再次admission把Scheduler/Worker E设回L。
- 最低成本层级：复用现有Scheduler/Pool fixture及完成结果helper，用64-token prompt、3个输出、16-token追加验证feature-on拒绝与feature-off暂停/重入；既有AsyncLLM suite只检查消费/注册之前的入口失败，用 `asyncio.run`，不依赖新增异步插件。CPU完成helper不证明GPU completion或payload。

## Graph capture / replay

- 模块用途：让原生模型 graph 读取持久 physical attention lengths，保留 logical RoPE/sampling 输入。
- I/O：dummy capture 与实际 replay 的 `InputBatch`、独立 physical buffer；输出 metadata 必须绑定同一 allocation，dummy refresh 清除真实请求的残留 lengths。
- 防止的失败：capture 绑定 logical L、运行时只换 Python metadata、dummy replay 读取旧 E、padding 未清零，以及回收后旧 row/tail 仍可读。
- 最低成本层级：在已有 compaction suite 用 CPU dummy metadata 验证地址/长度分离与 dummy refresh；真实 Qwen3-8B FULL replay 检验重复回收、同 bucket、padding/reorder、立即 release/rewrite/reuse、retained attention reference 和 capture count。Graph/Eager 使用相同 KeepPlan 对照，性能另跑无逐步同步的负载。

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

## Prefix ownership / APC

- 模块用途：保留 canonical shared prefix，只对已撤销全部 lookup 身份的私有 suffix 回收。
- I/O：真实 APC lookup/touch、当前 row/refcount、forward 后 E、临时 P/K；输出完成后的 row/E、缓存连续计数、free queue 与 KV events。
- 防止的失败：APC admission 的 Scheduler/Worker E 不一致；漏撤 suffix alias 后新增共享；deindex 当 free；非连续 shared 页被写；lossy 后沿原 logical chain 重新发布；正常 in-flight 被误判 lossy；最后引用前重用与 EOS/abort double-free。
- 最低成本层级：扩展现有 scheduler 与 prefix-caching suite，用真实 sha256 Request/BlockPool 复现 admission、lookup/event/ownership；仅 worker completion 用 CPU result。GPU Gate 使用 Qwen3-8B 的真实 A/B/C 请求、全层 payload 比较、FP32 attention、真实 release/rewrite/reuse，另在同配置 native 与 allocation pressure 下运行。CPU 不代替 GPU completion 或 payload 证据。

## TP shared plan / completion

四问与完整协议见`tp-protocol.md`。在现有compaction suite加入Hkv=1/4、TP2
贡献与独立CPU复数全query-head reference对照，覆盖复制owner保留完整GQA stats、
非owner零贡献及全局distinct-head等权。真实两GPU通信/统一plan与payload留给
独立小fixture和Qwen3-8B MultiprocExecutor轨迹。

扩展既有executor suite验证非output rank在sample_tokens失败会退出RPC loop，
以及AsyncOutput设备完成异常不转成可继续执行的普通FAILURE；feature关闭维持
原生行为。真实failure注入额外证明monitor终止TP组且canonical零commit/free。

## DP replica-local ownership / routing

四问见`dp-protocol.md`。复用真实AsyncLLM/internal dense DP2，不新增模拟allocator：
副本0的A/C制造一页真实shortfall，副本1空闲页不得计入available；本地回收后C
必须取得released ID并全层重写，A的retained KV保持不变。B在副本1持续decode，
abort A沿原affinity发送，逐forward核对B的128-token前缀KV不变。native/Sink的B
输出逐token对照，两个Pool最后均free32；四个额外请求验证原生router真实分布。
正式HTTP去掉payload同步hook，逐replica报告延迟/Goodput/preemption，独立trace
记录queue/free/重试，不用静态源码检查替代双GPU证据。
