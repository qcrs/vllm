# Graph 验证与 HTTP Serving 协议

## 问题与预先固定的判定

验证原生 FULL_DECODE_ONLY 的持久 physical lengths 绑定，以及图外回收之后的
replay、实际 allocator pressure 和 Serving 代价。正确性诊断逐步同步，不用于性能。
正式 Serving 复用现有脚本，关闭 KV trace 和 payload hook，保留 45 GiB 硬上限、
44 GiB guard 阈值、串行 GPU1、Qwen3-8B BF16/MRV2/FA2/B16。

Graph 正确性先覆盖 whole/token、两次回收、batch reorder、padding、released IDs
被 C 实际写新 KV、Prefix+Graph；Random/TriAttention 只增加 token 路径。
每次保留所有36层 valid KV 的 CPU source snapshot，completion 后逐bit比较 KeepPlan，
released pages 只在 canonical free 之后写 NaN，以暴露 stale read。
每次 replay 的 row/E/slot、logical positions、持久地址、原 graph 对象/capture count 均检查。

FULL capture 使用原生 max_model_len 上界，eager 重跑主对照也固定同一 scalar，
其余输入、retained KV、实际 KeepPlan 相同，要求 hidden 和全36层 append KV 逐bit一致，
logit argmax 一致。第一层 attention 另与独立 CPU FP32 dense softmax 比较
atol/rtol=0.016。此对照不改变 Runtime 的 actual-E metadata builder。

保留两条早期失败：只读 ExecuteModelState 字段赋值错误；动态 E 上界 eager
与 capture512 的 BF16 数值差异。独立诊断 46 次回收后 forward，仅把 eager
max_seq_len 从 E 改512，hidden/logits/追加KV全部逐bit一致；动态 E 的15步超原
atol0.0625/rtol0.03，argmax均未变。原容差未放宽，失败产物保持原样。
不把它称作 graph 与任意 eager 数值执行拓扑逐bit相同。

## 正式压力矩阵

- `serving_reclamation.py --graph-mode full`，CompilationMode.NONE，
  FULL_DECODE_ONLY buckets `[1,2,4,8]`；真实 prefill 沿原生 eager，decode 必须有 replay。
- APC off、TP1/DP1、async/spec/connector/cascade/reserve-full-ISL off，单在途。
- Pool 8、6、1.5 GiB 作为低/中/高压力探针；压力由实际 miss/preempt/reclaim 判断。
- Native Full、retention1 noop、SinkRecent retention0.5；另外6/1.5 GiB运行
  Random/TriAttention，校准 stats 沿用M2固定文件。headroom=0。
- long workload：8请求，prompt `[4096,8192]*4`，output128，
  arrival `[0,.3,.6,.9,5,5.3,5.6,5.9]`，temperature0/seed0/ignore_eos。
- 每server先32/8小预热，再完整同负载预热；正式3轮。APC off无跨轮cache复用。
- 沿用M2/Prefix SLO：TTFT≤2s，TPOT≤80ms；无压力noop吞吐开销≤5%。
  只有同Pool Goodput提高且延迟满足才称该配置性能通过，保留负收益与超SLO结果。
- TTFT/TPOT/ITL分位数、token/request throughput、SLO Goodput、preemptions、
  实测 NVML peak、完整输出和原生 metrics 全量保存。小样本P99不称生产tail估计。
  Full/noop token IDs必须相同；有损组质量不由合成负载推断。

## 短上下文与 profile

额外固定 short workload：两请求128/256 prompt、各128 output、arrival0/.3，
8 GiB pool。FULL/eager × Full/noop 共4组，均完整预热和3轮正式；用于判断
低batch短E时额外 CPU staging 是否抵消图收益，不复用无完整预热的旧 eager 数据。

NSYS另行诊断 FULL压力轨迹，保留capture/warmup与稳态边界、cudaGraphLaunch、
图外compaction、H2D/stream/Event等待和GPU gap；不把profile延迟混入正式Serving。
MRV2当前未给 ModelRunnerOutput 填 cudagraph_stats，原生日志开关不提供可靠hit率；
诊断 descriptor计数与NSYS实际launch作为证据，不额外扩张Runtime计数器。

阶段授权来自 prompt1 和当前 active Goal：本阶段完成报告、回归、提交后自动进入TP，
不重新引入旧Prefix后暂停条件。
