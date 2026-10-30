# M0：固定源码与地址、所有权、完成边界

固定基线 `568afb3a13806beb53bb2e6bd518269357b237c0`，实现分支
`p1/v8-physical-kv-reclamation`。Python 源码使用本 worktree，编译扩展只复用现有环境二进制。

## 原生调用链

`EngineCore.step → Scheduler.schedule → KVCacheManager.allocate_slots → coordinator →
SingleTypeKVCacheManager.allocate_new_blocks → BlockPool.get_new_blocks`。
`allocate_slots` 的真实检查是 `required = num_blocks_to_allocate + watermark`，
`available = free - reserved`；full-sequence reservation 是另一种 miss。
本实验关闭 full-sequence reservation，dense FullAttention 不淘汰 skipped-window blocks。

Schedule 公共收尾乐观推进 L 与 in-flight；完成后 Scheduler 才减 in-flight。
`EngineCore.step` 等 Worker 结果、处理 abort queue，再调用 `update_from_output`。
`_preempt_request` free canonical row 后清 L，Worker 下次收到 preempt/re-add 后重算。
M1 的 E 必须在相同生命周期清零。

## Owner 与观测

Scheduler/KVCacheManager 是 canonical owner；`single_type_manager.req_to_blocks` 保存
原生 Block 对象；Worker row 只是执行镜像。`coordinator.get_blocks` 使用 `.get()`，
trace 的 free-after 读取不会重建 defaultdict key。测试同时比较 trace on/off 的 schedule、
preempt 和最终 pool，确认观测不改变状态。

`BlockPool.free_blocks` 将 refcount 归零且无 hash 的页 prepend 到 free queue；
有 hash 的页 append。Core APC off 时 released ID 自然优先复用，不修改队列制造结论。

## MRV2 地址消费者

`init_kv_cache` 创建 layer dict 并 bind 到 Attention 模块；runner.kv_caches 是保活 backing
list，不是 layer view dict。实际 8B 36 个 view：`[2730,8,16,256]` BF16，stride
`[32768,256,2048,1]`。K/V 在最后维拼接，V 从128开始；真实 element offset 是
`row[i//B]*s_block + h*s_head + (i%B)*s_token + c*s_channel`。

`prepare_inputs → prepare_pos_seq_lens` 用 logical L 生成模型位置和 InputBatch.seq_lens；
combine_sampled_and_draft_tokens、prompt logprobs 等仍消费 logical metadata。
`prepare_attn → BlockTables.compute_slot_mappings` 及 DefaultModelState.prepare_attn 才应
分别使用 E 派生的 cache positions 与 KV seq lengths。不能全局替换 logical seq_lens。

## 完成边界

原 `sample_tokens` 先创建 AsyncOutput，再 postprocess；AsyncOutput 的 copy stream
只等待此前 main stream 工作。因此回收 step 必须先 postprocess/E+=q、搬移、发布 row/E，
再创建 AsyncOutput。get_output 的 copy_event 才能覆盖全部旧页访问和新布局写入。
无 forward 的 maintenance 必须独立 Event，不能 fake sampler 输出。

Persistent row overwrite 不清 tail；安全条件是下一次 allocation 覆盖 ceil((E+q)/B)，
KV write 用 E+offset，Attention 只读 E+q，logical RoPE 继续 L+offset。
GPU 完成后 Scheduler 先提交 row/E/free，再处理 EOS；abort 已按旧 row 清理则跳过结果。

这些是当前源码事实与 M1 接入约束，M0 的 Engine 实验没有进行任何回收。
