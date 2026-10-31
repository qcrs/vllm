# TP HTTP Serving 协议（运行前固定）

Qwen3-8B BF16，GPU0,1、TP2/DP1/PP1，原生mp Executor；MRV2/FA2/B16，
APC/async/spec/cascade/connector/reserve-full-ISL关闭。其余复用Graph正式协议：
long 8请求 `[4096,8192]*4`、每请求128输出，arrival
`[0,.3,.6,.9,5,5.3,5.6,5.9]`；short128/256两请求、各128输出、arrival0/.3。
小预热后完整负载预热，每组正式3轮，正式关闭trace和逐步检查。

Pool数值是每rank的KV backing bytes；TP2的等效全模型KV容量为其2倍，
native/noop/有损各策略必须同TP2/同每rank容量比较。固定8 GiB每rank为宽松容量，
0.75 GiB每rank为对照Graph1.5 GiB的全模型等效压力容量。
双卡项目NVML之和不超过45 GiB，44 GiB停止。

long/FULL两种Pool均测native、noop、SinkRecent，压力Pool补Random、Tri；
short/FULL与short/eager测native/noop、8 GiB每rank。共12组。
TTFT≤2s、TPOT≤80ms、无压力noop平均吞吐开销≤5%预算不随结果调整。
对每轮/聚合TTFT/TPOT/ITL、tokens/s、Goodput、preempt及NVML保存原始证据；
native/noop输出逐token对照。有损质量不从合成Serving推断。

采集修正：首个0.75GiB noop在warmup完成后触发原客户端180秒读取超时，
该目录保留FAIL；后续transport timeout固定600秒，只防止长排队截断数据，
不改变上述SLO或重复次数。已成功的180秒组未触发该限制，保持原结果。

另跑压力Tri NSYS诊断，标记正式轮窗口和score SUM/plan broadcast/completion。
计通信次数、逻辑payload、GPU NCCL活动及host等待；逻辑bytes不是总线实测traffic。
诊断开销不并入正式Serving。另用慢rank轨迹分离local copy完成与group ACK等待。
