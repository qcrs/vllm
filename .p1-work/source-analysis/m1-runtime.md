# M1 物理 KV 回收实现与源码学习

基线为 vLLM v0.26.0 `568afb3a13806beb53bb2e6bd518269357b237c0`，运行 MRV2。
本阶段支持 Core：单在途 batch、TP1、Eager、FullAttention/B16/BF16、FLASH_ATTN，
APC/spec/connector/async/cascade off。扩展必须独立验证后再开放配置。

## Logical 与 retained KV 的两个长度

Request 增加唯一持久语义 `effective_kv_len`（E）；`num_computed_tokens`（L）保持原逻辑。
`KVCacheManager.allocate_slots` 的 requirement 与实际 allocation 共同使用 E。
Scheduler 在正常输出完成后 E 加本轮 q。Worker 由 add/preempt/reclaim/completion
生命周期维护镜像；普通 step 不再重复传输 settled E 字典做一致性断言。

Worker 的 E 为 indexed CPU/GPU mirror，复用 `StagedWriteTensor`。
`prepare_inputs` 仍以 L 构造 input token、RoPE positions、sampling/prompt-logprob 输入。
仅传给 `prepare_attn` / `model_state.prepare_attn` 的局部 `InputBatch` view 使用
physical positions E..E+q、physical seq_lens E+q 及对应 CPU upper bound。
因此 slot_mapping、max_seq_len、FA metadata 与 allocator 对物理长度的理解一致。
收尾 review 将 physical positions/seq_lens 合入原生 `prepare_pos_seq_lens` 一次
launch；GPU E+=q 合入原生 `post_update`，CPU mirror 在同一 postprocess 边界推进。
feature-off 的可选 tensor 参数为 None，由 Triton specialization 删除物理分支。

`finish_requests` 同时移除 preempted 的 Worker state，恢复的 Request 经 `add_requests`
重新建立 E；Scheduler `_preempt_request` 将 L/E 一起清零，再由原生 prefill 重算。

收尾的配置检查要求 generation runner；pooling不经过上述postprocess，不能继承
E生命周期。`FullAttentionSpec`还可能含sliding/chunk/non-causal语义，需显式拒绝；
Worker初始化检查实际FA impl的ALiBi，因为其bias按物理距离计算，不能靠旧K中的
RoPE保留logical距离。这些是初始化支持边界，不在普通decode中重复检查。

## 回收与完成边界

M1 的实验脚本在正常 `schedule` 收尾之后构造确定触发 intent；生产代码提供
`make_reclaim_request`，M2 才接真实 allocation miss。
source E 为 settled E+本轮 q，source row 复制 canonical ID 列表，q 只能为0或1。

正常回收 step 在 `sample_tokens` 中先完成 sampling/prompt-logprobs，再执行一次
`postprocess_sampled`、E advance、retention、payload copy、row overwrite/E staging。
随后才构造 `AsyncOutput`：其 copy stream 的 `wait_stream(main_stream)` 与 copy_event
覆盖全部回收写入。无回收 step 保留原先的 async-output/postprocess 顺序。
不能将 kernel launch 返回当成 GPU 完成。

`total_tokens==0` 的 maintenance 分支不运行 forward 或 sampling；回收后单独 record
并 synchronize 一个 CUDA Event，返回空 sampled output 与非空 reclaim result。
EngineCore 继续经过 abort queue 和 `update_from_output`。

Scheduler 在普通 sampled/EOS 循环之前消费结果。活 Request 要求源 row/E 和 target K
匹配，retained IDs 唯一、非 null、属于源 ownership。保留原 canonical Block 对象，
先换 row/E，再 `BlockPool.free_blocks(reversed(removed))`。不删除 manager 原有的
running/cache bookkeeping。EOS 随后释放新 row；GPU 执行中 abort 在结果消费前按旧
canonical row 清理，回收 ACK 随后跳过，避免 double-free。
payload 已改写后的异常直接传播，不采用继续使用旧 KV 的恢复路径。

## Whole-block 与 Stable CTA

SinkRecent plan 在当前 member 域构造，重复回收无需额外 logical-position map。
当两段边界满足 page/offset 可表示性时，仅重建 row；否则保留旧 row 的前 ceil(K/B)
页，由唯一 Stable CTA kernel 压紧。无效 tail 没有清零契约。

每 Layer/Head 一个 CTA，grid=(8,36)。同 CTA 顺序处理 token tiles，完整 tile 的
load 后执行无条件 `tl.debug_barrier()` 再 store。严格升序 KeepPlan 保证 s[j]>=j，
下一 tile 来源不会被已处理目的覆盖；不同 Layer/Head 的地址互斥。
所有 offsets 为 int64，使用 packed4D 的真实 stride，支持非单调 row、padding、
storage offset 和 partial tail。初始及优化后 TTIR/TTGIR/LLVM/PTX 均保留在 SSD。

初始 runtime stride 标量阻碍 compiler 连续性分析，TTGIR 把 warp 排在 token 维，
生成大量 b16 scalar memory instructions。依据 NCU 后增加由真实 tensor stride 与
data_ptr 推导的 alignment hint，以及一致 channel stride 的编译期常量。
四维地址语义不变；不同 channel stride 仍由同一 kernel 读取真实值。
优化后 warp 沿 channel 维排布。T16/W4 保留为默认：T32 仅约1%差异，寄存器明显更多。

实验同步和 CPU KV snapshot 仅存在 `.p1-work/scripts/live_reclamation.py`，不进入
Serving Runtime。该脚本用真实 Engine A/B、原生 Pool 与真实内部 request ID；
不修改 free queue，不用外部 ID 直接查询随机化后的 Scheduler ID。
