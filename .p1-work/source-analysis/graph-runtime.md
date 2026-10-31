# Graph：持久物理长度与 replay 消费链

## 所有权和生命周期

模型图由 `gpu/cudagraph_utils.py::ModelCudaGraphManager` 持有，
`BatchExecutionDescriptor` 只包含执行模式、token/request bucket、uniform q 和 LoRA case。
E、L、Block IDs 都不是 graph key。`capture` 启动时分配的 graph pool、output buffer
和 input tensor 引用持续到 runner 销毁，不能靠换 Python metadata 改变已捕获的地址。

`gpu/input_batch.py::InputBuffers` 由 runner 构造并持有，positions 为
`int64[max_tokens]`、seq_lens 为 `int32[max_reqs]`、query starts 为
`int32[max_reqs+1]`，均在同 CUDA device。每轮 prepare_inputs 更新内容；positions
和 seq_lens 继续分别表示 logical RoPE positions 和 logical total lengths。
非 attention consumer（sampler/prompt logprob/chunked prefill）继续消费这些原生值。

`KVReclamationWorker` 另外持有同shape的 `cache_positions`、`kv_seq_lens` 以及
resident-request-index域的 staged effective lengths。原生 `prepare_pos_seq_lens`
按当前 mapping 一次生成 logical 与 physical positions/lengths；`prepare_attn`
返回临时 physical InputBatch view，CPU upper bound 按 E+q 构造。view 生命周期
只有本轮，backing allocation 贯穿 runner。原生 `post_update` 在 postprocess
边界同时推进 GPU L/E；`advance` 只推进实际请求的 CPU E mirror，padding不参与。

`BlockTables` 原生持有 resident rows、gather后的 input rows、slot mappings。
[`BlockTables.append_block_ids`](../../vllm/v1/worker/gpu/block_table.py) 在
`overwrite=True` 时只覆盖新 row 的有效 prefix，并缩小 `num_blocks`；不清除 resident
row 的旧 tail。`_gather_block_tables_kernel` 也只复制当前 `num_blocks` 个 ID 到
持久 input row，真实请求的无效 tail 可以保留旧值；只有 padded request row 整行清零。

旧 tail 不可寻址依赖有效范围：下轮 allocator 先保证至少 `ceil((E+q)/B)` 页，
[`prepare_pos_seq_lens`](../../vllm/v1/worker/gpu/input_batch.py) 生成
`cache_positions=E+offset` 和 `kv_seq_lens=E+q`。slot kernel 按真实 query 范围写入，
attention 按 physical seq_lens 读取，不能访问有效页外的旧 ID；slot kernel 本身不另查
`num_blocks`，所以 allocator/E 一致性是必要前提。padding lengths 置0、padding slots
置无效值，与真实 row tail 的保留是两种语义。无需清零旧 ID 或 payload。
已有 released-page poison、C全层重写和A下一 replay 的证据见
[Graph报告](../reports/graph.md)；本次R2仅静态核对这些 bounds，没有重跑GPU。

## 已捕获引用缺陷及修复

原 `prepare_inputs_to_capture` 调 `InputBatch.make_dummy`，其 seq_lens 指向
logical InputBuffers。实际 Runtime 虽把 metadata 换成 physical view，FULL replay
仍读取 capture 时的 logical tensor。回收后 L>E，这种只换 Python object 的办法无效。

`ModelCudaGraphManager.capture` 新增可选 `attn_seq_lens`。feature开启时 runner
传入唯一持久 physical buffer；dummy batch 建立后将 dummy长度复制进去，并仅替换
attention batch 的 seq_lens view。model input positions仍用原logical InputBuffers。
feature关闭传None，完整沿原生capture路径。

dummy execution（profile/DP native padding）通过 `prepare_dummy_attn` 清零持久
physical lengths，再填有效 dummy 范围；避免真实 E 或旧大batch padding残留。
`InputBatch.make_dummy` 自身清 logical seq_lens tail，capture copy因此也覆盖完整tail。
CPU测试在原 compaction suite 验证两次dummy内容刷新、相同地址、logical/physical分离。

## metadata与上界

[`DefaultModelState.prepare_attn`](../../vllm/v1/worker/gpu/model_states/default.py)
在FULL/breakable PIECEWISE使用 padded范围，
Runtime max_seq_len 来自physical CPU upper bounds；`attn_utils.build_attn_metadata`
继续走原backend builder。FA2在本A100使用普通varlen接口，FA3的persistent scheduler
metadata与tail clearing虽沿原生链，但本硬件没有验证FA3。

FULL capture 原生 `for_capture=True` 将 scalar max_seq_len 固定 max_model_len，
因为 launch需要覆盖将来的合法E范围；实际seqused_k依然读取每轮物理长度，
不是让attention读取512个有效成员。回收E不改变合法上界，不需要recapture。

最初eager参考使用actual-E scalar，BF16最终hidden个别元素超过既定容差。
`20261031-094927-full-token-numerics-diagnostic` 仅改参考的max_seq_len=512，
46次forward的hidden/logits/全部36层追加KV逐bit相同。动态scalar路径有15步超容差、
argmax均相同。因此主验证固定相同launch上界并采用零容差，保留动态对照负结果。
该证据只定位到scalar选择造成数值路径差异，未profile到FA2内部具体reduction分支。

## publication point与completion

`GPUModelRunner.execute_model` 仍按原生descriptor dispatch，FULL调用run_fullgraph；
prefill或超bucket走native fallback，单纯回收不强制eager。PIECEWISE另用原生
breakable graph；诊断实际观察到61次PIECEWISE，而不是只检查配置开关。

回收step的采样和postprocess/E+=q后，在同main stream上依次score/plan、compaction、
resident row/E publication。`AsyncOutput` 的copy stream wait main stream，copy_event
完成后才给Scheduler成功结果；q0 maintenance用已有Event同步。Scheduler只在此后
commit canonical row/E并free。下一step输入更新和replay排在同一串行执行边界。
本阶段没有把compaction放进graph，也没有每step新增device synchronize。

诊断检查原始GPU completion event.query发生在任何payload读取之前，canonical free
后才poison released页；C真实覆盖后A继续decode。所有36层有效payload用实际KeepPlan
和原source snapshot逐bit比较，同时检查其他请求/prefix内容不变。
ptr/shape/stride/dtype与capture计数每步验证；没有把free pool等同CUDA显存归还系统。

## 证明边界

独立FULL token/whole、Random/TriAttention token、breakable PIECEWISE token和
Prefix+FULL token/whole均有Qwen3-8B证据。有限buckets `[2,4]` 的padding/reorder/
多次回收不代表任意backend、FA3/4、spec/async/connector或TP已验证。
真实压力和无hook Serving另见Graph报告；TP completion需要下一阶段独立all-rank门槛。
