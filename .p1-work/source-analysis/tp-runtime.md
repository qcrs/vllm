# TP：head ownership 与 all-rank 发布边界

## 已核对的 pinned 调用链

`Qwen3Attention` / `QKVParallelLinear` 对 Hkv>=TP 使用连续 head shard，
local Hkv=Hkv/TP；Hkv<TP 时每 rank 一个 KV head，复制因子 TP/Hkv，
global head=rank/复制因子（整数除法）。Q heads 仍按 TP 分片。
离线 scorer 不依赖本轮 Q，因此复制 group 的第一个 rank 可使用完整校准
GQA group，不能把自己的 Q shard 误当完整 group。模型权重和 cache 不迁移。

`MultiprocExecutor.execute_model` 与 `sample_tokens` 只读取 output rank 的
RPC response。每 rank 都会执行 MRV2 `sample_tokens`，但非 output rank
不会执行 `AsyncOutput.get_output()`；单 rank 的 copy event 因而不足以证明
远端 compaction/row/E 写已完成。

`AsyncOutput` 在 output-copy stream 上 wait main stream，记录 copy event；
host `get_output()` 同步该 event 后才把成功结果发回 Scheduler。
q=0 maintenance 在 `execute_model` 内记录并同步 main-stream event。
回收专用 completion 必须在上述两个事件之前，并排在每个 rank 的最后写之后。

`GroupCoordinator.broadcast` 使用 TP device group 的 torch.distributed broadcast。
原生 all_reduce 分发可能使用 custom allreduce 或 NCCL。回收 score 与 completion
使用同一 TP device group 的显式 NCCL SUM，以便明确观测 FP32[E] 和标量
两种通信；PyTorch ProcessGroup 在调用 stream 与 NCCL stream 之间建立依赖。
它们在图外执行，不改变模型图内原生通信路径。

## 异常路径与已实现的完成边界

`WorkerProc.worker_busy_loop` 捕获普通 RPC 异常后只向选定 output rank 发 FAILURE；
非 output rank 的普通异常会被记录后继续读下一条 RPC。若它在 destructive copy
之前退出本次回收、output rank 已进入 completion，则仅依靠原生正常响应路径
会等待超时；该非 output rank 不能继续接受旧布局的工作。
已在启用回收时让execute/sample与AsyncOutput异常离开RPC busy loop，使原生
worker monitor终止整个执行组。未启用回收时沿用原生异常处理。

`TriAttentionScorer`按distinct KV head/layer权重生成local贡献；复制KV的唯一
owner装载完整GQA query-head stats，非owner贡献0。`KVReclamationWorker.reclaim`
执行score SUM、rank0 stable top-K和plan broadcast；每rank更新payload/row/E后
记录local event并调用标量completion SUM，随后AsyncOutput/maintenance的event
覆盖该collective。canonical提交/free始终只在Scheduler。

11条8B轨迹覆盖慢rank、非output rank失败、q=0/maintenance、Prefix/Graph组合与
released-ID实际重写；独立NCCL用例覆盖Hkv4/Hkv1和stable ties。故障注入零commit、
free不变、两worker退出；慢rank未完成时canonical/free不变。细节及限定见
`../reports/tp.md`。三类collective实测次数/bytes/time与rank等待、eager性能FAIL见
`../reports/tp-serving.md`，不将逻辑payload当实际链路流量。

## 请求级TTFT诊断的状态owner与关联

当前源码基于 `2c3379cc883b`。HTTP入口
[OpenAIServingCompletion._create_completion](../../vllm/entrypoints/openai/completion/serving.py)
使用外部request_id生成completion id；
[InputProcessor.assign_request_id](../../vllm/v1/engine/input_processor.py)
保存external id并随机化内部id。诊断保存SSE response id与Scheduler内部id，
以唯一的 `response_id-0-<8hex>` 映射连接，不把到达顺序当身份。

[EngineCore.step](../../vllm/v1/engine/core.py) 依次调用
`Scheduler.schedule`、`execute_model(non_block=True)`、future result、必要的
`sample_tokens`，再 `Scheduler.update_from_output`。本契约async off、单在途；
诊断step tag只在工具子类里传递，不成为Runtime第二份状态。
`Scheduler._update_after_schedule`已推进computed，记录prefill起点需减q。
同id preempt清KV后重算会使logical起点回退，与保留KV的streaming重入不同。

[GPUModelRunner.execute_model/sample_tokens](../../vllm/v1/worker/gpu/model_runner.py)
与 [AsyncOutput.get_output](../../vllm/v1/worker/gpu/async_utils.py) 的host返回含义不同。
execute主要发射forward，sample可发射post-forward reclaim；host reclaim返回
只说明已排队。回收的两rank标量completion、output-copy event或q0 event、Core
收到成功结果、canonical commit/free依次构成发布边界。普通无回收step的非output
rank postprocess不受output rank copy event直接约束，可在Core处理开始后结束；
不能把“全rank全部普通kernel先结束”额外套成该路径的原生契约。

工具 [tp_ttft_hooks.py](../scripts/tp_ttft_hooks.py) 记录原生方法与实际等待，不改
metadata、payload或event；[analyze_tp_ttft.py](../scripts/analyze_tp_ttft.py) 按
process+CUDA correlation连接设备活动，Graph保持opaque。host、GPU与通信范围
可重叠，不能相减得出可消除成本。协议见 [notes](../notes/tp-ttft-diagnostic.md)，
本次采样事实、代表性限制及停止判断见 [report](../reports/tp-ttft-diagnostic.md)。
