# TP pressure TTFT 单轮诊断协议

2026-10-31，用户授权在既有trace不足时补一次当前配置diagnostic，再按成本证据决定
是否优化。该协议只约束本轮O1；不重启M0→DP，不扩展85条质量样本，不扫batch。
实验结果与停止判断见 [诊断报告](../reports/tp-ttft-diagnostic.md)。

## 固定配置与归档

使用现有 `.venv/bin/python`、Qwen3-8B BF16、物理GPU0/1、Worker CPU20/21、
TP2、0.75GiB/rank、SinkRecent retention=0.5、reserve=0、batch1024、max seqs=8、
FULL_DECODE_ONLY、APC/async off。输入、arrivals、128输出及完整负载预热复用
`serving_reclamation.py`。SLO仍为TTFT≤2s且TPOT≤80ms。

```bash
export P1_CUDA_VISIBLE_DEVICES=0,1
source .p1-work/scripts/env.sh
.venv/bin/python .p1-work/scripts/run_guard.py \
  --stage TTFT --name tp-batch1024-timeline -- \
  @python .p1-work/scripts/profile_tp_ttft.py \
  --policy sink_recent --pool-gib 0.75 --workload long --tp 2 \
  --graph-mode full --full-warmup --worker-cpus 20 21 \
  --max-num-batched-tokens 1024 --repeats 1 --trace
```

沿用 `run_guard.py` 串行、44GiB停止阈值和45GiB硬上限；不操作其他用户任务。
新产物只写 `/data_ssd/qcrs/experiments/p1-v8/TTFT`。guard的invocation保存基线和
直接脚本；wrapper额外复制间接helper。真实server命令由wrapper写入
`diagnostic-command.json`；原helper的 `protocol.json.command` 是拦截前命令。

NSYS使用cuda/nvtx/osrt、graph级CUDA Graph span、flush=0、sample/cpuctxsw=none，
以保留完整尾部。没有额外device synchronize或更改原生完成等待。

## 身份、时间与完成语义

- client显式提交 `request_id=r0-N`，保存首个带token_ids的SSE返回id；Scheduler
  保存原生随机化后的内部id。分析要求 `response_id + -0-<8hex>` 唯一匹配，
  不用到达顺序推断身份，不关闭原生id随机化。
- 诊断Scheduler为每个真实 `SchedulerOutput` 添加可pickle的step tag；两rank
  Worker据此记录同step的NVTX和monotonic/thread CPU时间。原生模型预热尚未进入
  请求生命周期，完成 `compile_or_warm_up_model` 后才要求真实step tag。
- Scheduler在原生schedule之后已推进logical computed；记录该段起点时减q。
  同一id的logical起点回退意味着preempt/recompute，不能误认作retained KV重入。
- 各进程首尾NVTX clock mark用于NSYS与monotonic对齐；采用末次offset，另外保存
  首尾差值。CUDA launch与activity按process identity和correlationId连接，再按
  该线程内最内层phase归属step。Graph span保持opaque，不推算图内通信成本。
- prefill段完成以Core收到Worker结果并执行 `update_from_output` 的边界记录；
  `AsyncOutput.get_output` 保留真实copy event等待。q=0 maintenance沿用原生event。
  回收必须有两rank completion collective及canonical commit，不以host reclaim
  return充当GPU完成。关联不等于重新验证payload或任意异常组合。

## 分项与判读

TTFT拆为client→Scheduler enqueue、enqueue→first schedule、包含本请求的prefill
step wall、段间等待、Core首个token输出→client接收。各项必须求和回到原始TTFT。
step wall可包含同batch其他请求，不把各请求的step wall再相加作为总成本。

GPU使用逐rank区间union；host wall/thread包含嵌套和与GPU重叠，不能与GPU相加，
也不能用二者相减估计可消除开销。collective device time包含rank等待、依赖、
调度与profiler影响，不等于纯链路传输。其他用户的活动不属于项目activity union。

诊断吞吐、SLO和抢占单列，先与旧正式负载轨迹比较。只有成本占比及代表性足以
支持收益才改实现，并在同配置无profiler对照TTFT、TPOT和吞吐；证据不足则保留
限制并停止。本协议不授权后续重复采样、质量扩评或新backend。
