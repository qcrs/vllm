# Graph HTTP Serving 与执行成本

## 结论与判定边界

**PASS（限定配置）**：FULL_DECODE_ONLY 下1.5 GiB压力Pool，SinkRecent、Random、
TriAttention各24请求均满足TTFT≤2s、TPOT≤80ms，Goodput从native的0提升到
0.809–0.814 requests/s。8 GiB无压力noop平均吞吐开销0.63%，通过≤5%预算。
6 GiB实际无preemption，不能把它叫回收收益。短上下文FULL的noop吞吐117.04，
eager为63.42 tokens/s，图收益没有被新增执行边界抵消。

本实验是合成压力的服务评测，不证明有损策略任务质量；质量沿用M2报告的独立结果。
每组long只有24请求、short只有6请求，分位数是本次样本描述，不是生产tail估计。
正确性、Prefix组合、逐bit参考与数值边界见`graph.md`。

## 固定协议与复核

协议在运行前保存于`../notes/graph-serving-protocol.md`。Qwen3-8B BF16，GPU1，
MRV2/FA2/NHD/B16，TP1/DP1，APC/async/spec/cascade/connector/reserve-full-ISL off。
CompilationMode.NONE，FULL_DECODE_ONLY buckets `[1,2,4,8]`，prefill走原生eager。
max_model_len9216，batch tokens2048，max seqs8，retention0.5、headroom0。
Tri使用M2 `20261031-000856-normalized-q-calibration/frequency.pt`。

long为8请求，prompt `[4096,8192]*4`、output128，arrival
`[0,.3,.6,.9,5,5.3,5.6,5.9]`；short为128/256 prompt、各128 output，arrival0/.3。
均temperature0、seed0、ignore_eos。每个server先32/8预热，再完整负载预热，
然后正式3轮，正式轮关闭trace/payload诊断。所有正式轮内无Triton JIT warning，
预热warning保留。所有17组exit0、无guard触发；最大NVML项目峰值24.135 GiB。

```bash
source .p1-work/scripts/env.sh
.venv/bin/python .p1-work/scripts/run_guard.py --stage Graph \
  --name http-full-1.5-sink_recent -- \
  @python .p1-work/scripts/serving_reclamation.py \
  --graph-mode full --policy sink_recent --pool-gib 1.5
.venv/bin/python .p1-work/scripts/summarize_graph_serving.py
```

结果根目录`/data_ssd/qcrs/experiments/p1-v8/Graph`，`serving-summary.json`保存
17组完整目录名、每轮/聚合指标、guard、prompt hash及诊断。validator核对原始
request数据、token usage/数量、分位数、SLO计数、吞吐分母、预热和资源。
Native/noop共84个输出逐token相同。每个run保存invocation、source hash/snapshot、
protocol/prompts、server log、3份requests、metrics、serving、NVML与result。

## OBSERVED 三轮聚合

TTFT/TPOT列为pooled P99（ms），preemptions为三轮增量之和；完整P50/P95、ITL
在summary和各轮原始JSON中，不将每轮分位数平均成全样本分位数。

| 负载/模式 | Pool GiB | 策略 | tokens/s | Goodput req/s | TTFT P99 ms | TPOT P99 ms | preemptions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| long/FULL | 8 | Native | 110.24 | 0.861 | 1265.1 | 29.8 | 0 |
| long/FULL | 8 | noop | 109.55 | 0.856 | 1279.1 | 30.2 | 0 |
| long/FULL | 8 | SinkRecent | 109.81 | 0.858 | 1279.9 | 30.1 | 0 |
| long/FULL | 6 | Native | 110.03 | 0.860 | 1284.5 | 29.9 | 0 |
| long/FULL | 6 | noop | 109.51 | 0.856 | 1298.1 | 30.2 | 0 |
| long/FULL | 6 | SinkRecent | 108.81 | 0.850 | 1305.8 | 30.7 | 0 |
| long/FULL | 6 | Random | 110.25 | 0.861 | 1252.6 | 29.9 | 0 |
| long/FULL | 6 | TriAttention | 109.91 | 0.859 | 1262.2 | 30.1 | 0 |
| long/FULL | 1.5 | Native | 9.67 | 0 | 98318.5 | 134.1 | 948 |
| long/FULL | 1.5 | noop | 9.61 | 0 | 98991.4 | 135.0 | 948 |
| long/FULL | 1.5 | SinkRecent | 104.24 | 0.814 | 1923.6 | 32.3 | 6 |
| long/FULL | 1.5 | Random | 103.58 | 0.809 | 1933.0 | 33.0 | 6 |
| long/FULL | 1.5 | TriAttention | 103.60 | 0.809 | 1959.7 | 32.8 | 6 |
| short/FULL | 8 | Native | 116.39 | 0.909 | 63.3 | 14.7 | 0 |
| short/FULL | 8 | noop | 117.04 | 0.914 | 60.1 | 14.7 | 0 |
| short/eager | 8 | Native | 63.99 | 0.500 | 62.2 | 28.8 | 0 |
| short/eager | 8 | noop | 63.42 | 0.495 | 60.3 | 29.1 | 0 |

1.5 GiB的Sink/Random/Tri吞吐相对native分别+978.08%、+971.22%、+971.46%，
主要避免大量native重算；不将其外推为正常容量下普遍加速。8/6 GiB Sink分别
-0.39%/-1.11%，6 GiB Tri -0.11%；这些无压力数据不支持回收加速声明，
Random的+0.20%也不能归因为算法。short FULL noop较native +0.56%、eager -0.90%，
只用于预算及图收益判断，不声称微小差异显著。

独立trace `20261031-095530-http-full-1.5-sink-pressure-diagnostic` 每轮22个
allocation miss、18次commit、累计释放1796页、2次native preempt；每次释放32–256页。
累计释放可含同页重复周转，不是Pool容量。正式性能不包含JSON trace成本。

## NSYS profile 与资源监控修复

NSYS诊断使用同配置、1轮预热/1轮计时，`profile_graph_serving.py`以NVTX metrics
marks分隔阶段；`analyze_graph_profile.py`只统计metrics-2-after到metrics-3-before。
profile不并入正式17组吞吐。

早期NSYS运行的guard峰值为0，不能作为资源合格证据。实际读取NVML PID的
`/proc/PID/environ`确认，profile下EngineCore保留`P1_VLLM_ROOT`/`P1_RUN_DIR`，
不保留`P1_WORKTREE`，且NSYS另建process group。guard原识别条件漏计。
现同时识别上述明确项目标识，PGID读取不再被environ权限异常跳过；阈值终止
覆盖已经识别的项目进程组，不操作无项目标识的其他GPU任务。早期产物保留。

有效诊断为`20261031-104758-nsys-full-sink-pressure-graceful-diagnostic`，
显式`--shutdown-timeout 20`让EngineCore正常退出，并以
`--cuda-flush-interval=500`周期刷新。NVML峰值19,006,488,576 bytes（17.702 GiB），
exit0、未触发guard。计时窗口44.681–54.676s，末次CUDA活动距结束6.179ms；
EngineCore无“Not all CUDA events”警告，原始SQLite/rep/summary均保存。

| 计时窗口观测 | 次数 / 成本 |
| --- | --- |
| cudaGraphLaunch | 274次，CPU API合计56.359ms |
| GPU graph spans | 273条，合计3297.104ms |
| 全部forward / eager dispatch | 308 / 34；graph API占88.96% |
| 图外Stable CTA compaction | 16次，合计7.164ms |
| physical positions/seq_lens准备 | 616次，合计1.391ms |
| effective length advance | 308次，合计0.689ms |
| staged physical/原生metadata写 | 166次，合计0.906ms |
| H2D | 782次、300,488 bytes、GPU合计0.952ms |
| D2H | 616次、12,624 bytes、GPU合计1.072ms |
| EventSynchronize | 308次、host累计6793.631ms |
| 观测GPU activity union | 8543.899ms / 9995.285ms窗口 |
| 相邻活动gap P50/P95/P99/max | 0.0014 / 0.2222 / 1.9809 / 64.4977ms |

EventSynchronize包含本轮forward和output等待，不能全部归因回收。
图外model prefill kernels是原生eager prefill，不是decode隐式fallback。
H2D包含全部原生输入staging，不全是回收metadata。
这组profile与正式Serving分开；周期刷新与trace有额外开销。

**profile边界：PARTIAL_GPU_SPANS。** 虽然活动已覆盖全轮，274个graph API中
correlationId618949没有对应GPU span（其余273个均匹配）。因此GPU union是
观测下界，窗口剩余1451.386ms是观测活动之外的上界，包含arrival/CPU/HTTP等待，
不能叫纯scheduler idle；不以它精确核算GPU利用率或全部graph成本。
NSYS2024.7.1还提示driver12.7使用12.6 trace libraries；不升级环境来消除该限制。
早期103100的尾部截断、104208的cudaProfiler接口方案仅采到3个runtime事件均保留，
不用于稳态成本。正确性和正式Serving PASS不依赖这些不完整profile。

`analyze_graph_profile.py`强制kernel/graph/runtime/memcpy表存在、检查尾部覆盖，
并逐correlation检查graph span；缺span明确标PARTIAL，不能静默当PASS。
该证据已足以判断本配置的copy/physical metadata kernel不是主要GPU成本，
不支持新增graph内compaction或额外metadata backend。
