# Prefix：APC 身份、共享保护与私有回收

基线和 Core 约束沿用 M1/M2；本 Gate 只开放完整页 APC，要求单 KV group、
`hash_block_size == block_size == 16`。Graph、TP、DP 和必要组合仍须各自验证。

## Admission 与两个长度

原生 `Scheduler.schedule` 对 waiting 请求先执行 `get_computed_blocks`，再把真实
cached Block 对象交给 `allocate_slots`；后者的 `allocate_new_computed_blocks` 经
BlockPool.touch 获得 active 引用。只有 allocation 成功，Scheduler 才把已命中长度
采纳为 `num_computed_tokens`，并推进本轮 q。

Prefix 扩展必须在这一成功边界同步初始化 E。Worker 的 `add_requests` 原本已经以
`new_req_data.num_computed_tokens` 初始化 CPU/GPU E；只打开 APC 配置会导致两端
不一致。真实 APC scheduler 测试曾复现 B 命中48但 SchedulerOutput E=0，而 Worker
应为48。修复后 before-forward 为 L_settled=E=48，q=16 完成后两端均为64。
不能在 allocation 失败前提前采纳命中，也不能把 lossy 请求的 E 随意补成 L。

## 候选与取消发布

`_select_reclaim_candidate` 遍历 canonical row，P 为最后一个 `ref_cnt>1` 页的末端；
允许保护域内部存在 ref=1 的 gap。预算为 P 加私有 suffix 的 retention/min_keep，
并覆盖 sink/recent 保护并集。K>=E_src 或释放不足一页时跳过；只计算真实可 free 页。
候选携带临时 P，与 `make_reclaim_request` 最终边界重算的 P 一致。

`make_reclaim_request` 对 P 后全部页调用原生 `KVCacheManager.evict_blocks`，不能只
清 primary hash。BlockPool 的 `_remove_cached_block_hashes` 同时清理 primary、
`cached_block_hashes_by_block` aliases 和反向 lookup，按每个 cache key 发出带 group
身份的 BlockRemoved。此时 row、refcount、free queue、cached count 都不提前提交。
仅 ref=1 也能被后来的 lookup 命中，故私有 suffix 仍必须先 deindex。

真实 miss 后停止 admission，单在途保证 deindex 后、Worker mutation 前没有新 lookup
把 suffix 再共享。原 forward 继续读取未变的源 row/payload。

## Worker 与完成提交

现有 Worker 已接受 ReclaimRequest 的临时 P。SinkRecent 的 prefix 为 max(sink,P)，
Random/TriAttention 同样只在 eligible suffix 内选取；所有 KeepPlan 使用当前 member
下标并严格升序。Whole 路径保留原保护 row，Token 路径沿用唯一 Stable CTA kernel，
identity prefix 无 store；无需新增 COW、持久 P map 或 payload backend。

GPU completion 仍由 MRV2 的 AsyncOutput copy_event 覆盖，maintenance 用既有独立
Event。Scheduler `update_from_output` 先按 q 推进 E，再消费成功回收结果。
`commit_reclamation` 验证 protected row 原样保留、suffix 独占；保留原 Block 对象，
提交 row/E，把 `num_cached_block[req_id]` 精确降到 P/B，再 free removed 私有页。
key 保留以维持原生 running allocation fast path，不能删除 count map 来重造生命周期。

EOS 在上述 commit 后释放新 row；abort 在 ACK 前已移除请求时仍跳过 ACK，由原生
释放路径处理旧 canonical ownership。共享页直到最后引用消失才进入 free queue。

## Lossy 后禁止原 chain 发布

唯一派生条件为 `E < num_computed_tokens - num_in_flight_tokens`。正常 schedule 的
乐观 L 比 E 提前 q，不等于发生 lossy。公共 `KVCacheManager.cache_blocks` 统一处理
此条件；`allocate_slots` 的内部 publication 也走该入口，独立 caller 不能绕过。
Request 的 logical tokens/block_hashes 不截断，不新增 `apc_disabled` 状态。

后续新 decode KV 依赖删减上下文，即使 token 相同也不能用原完整 chain 发布。
保护区原 hash 仍可被 C 真实命中。若下次 P 缩小，先 deindex 新 mutable 区再回收。
原生 preempt 把 L/E 清零，再通过 canonical APC 命中与真实重算恢复原生发布。

## 验证分工

`test_prefix_caching.py` 使用真实 sha256 Request、pool、lookup、alias 和 KV events，
验证撤销/commit/cache guard/reuse/最后引用；alias 由 pool 原生插入接口构造，用于
验证全 key 撤销，不宣称支持 fine-grained partial APC/COW。
`test_scheduler.py` 检验真实 admission 和候选 P，shared gap 用原生 touch/free 构造。

`.p1-work/scripts/prefix_reclamation.py` 用真实 Qwen3-8B Engine 验证 whole/token、
A/B 不同计划、C 命中和复用、全部36层 payload、后续 FP32 Attention 与完成时序。
CPU snapshot、Event.query 和 controlled lifecycle 注入只存在实验脚本，不进入 Runtime。
运行结果、限制和性能口径见 `reports/prefix.md`。
