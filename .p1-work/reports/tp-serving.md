# TP HTTP Serving 与通信成本

Qwen3-8B BF16，物理GPU0,1、TP2/DP1、MRV2/FA2/B16，APC/async/spec/cascade/
connector/reserve-full-ISL off。源HEAD `d5f1029450` 加TP diff，run的invocation/source
固定实际源码。每组小预热、完整负载预热、3正式轮；协议见`../notes/tp-serving-protocol.md`。
long为8请求4096/8192交替、各128输出、固定arrival；short为128/256两请求、各128输出。
容量是每rank backing bytes，0.75GiB×2等效全模型KV1.5GiB；不能和TP1容量数值直接比较。
所有实验由44GiB停止阈值guard监测，未触发45GiB项目上限。

## 正式结果

12个配置均完整采集、guard exit0。native/noop共60个输出逐token相同。
固定SLO为TTFT≤2s且TPOT≤80ms；分位数从3轮合并请求计算，ITL来自客户端stream
chunk接收时间，同一chunk内token时间相同。吞吐含排队时间；预热不并入正式数据。

宽松long/FULL noop平均吞吐开销0.145%，short/FULL为1.034%，均通过≤5%预算。
short/eager为14.330%，性能FAIL。压力Full/noop每轮316次preemption，三种有损策略
降至每轮约11–12次、吞吐约48–51tokens/s，但全部压力配置Goodput为0，SLO FAIL。
宽松long每轮6/8达到SLO，仍有2个请求TTFT约2.44s，不能称全部延迟通过。

| workload/graph | GiB/rank | policy | tokens/s（三轮） | TTFT P50/P95/P99(s) | TPOT P50/P95/P99(ms) | ITL P50/P95/P99(ms) | SLO合计 | preempt合计 |
| --- | ---: | --- | --- | --- | --- | --- | ---: | ---: |
| long/full | 8 | full | 104.34/104.42/104.43 | 1.73/2.44/2.44 | 20.28/32.63/32.66 | 11.74/12.73/277.73 | 18/24 | 0 |
| long/full | 8 | noop | 104.28/104.27/104.19 | 1.73/2.43/2.44 | 20.40/32.73/32.80 | 11.89/13.10/277.91 | 18/24 | 0 |
| long/full | 8 | sink_recent | 104.10/104.07/104.38 | 1.73/2.43/2.44 | 20.40/32.81/32.83 | 11.92/12.95/277.67 | 18/24 | 0 |
| long/full | 0.75 | full | 6.28/6.28/6.25 | 78.70/155.98/156.46 | 172.59/206.89/211.24 | 265.15/276.27/281.38 | 0/24 | 948 |
| long/full | 0.75 | noop | 5.41/5.38/5.40 | 91.78/182.30/182.74 | 198.69/241.36/241.45 | 305.84/323.82/325.66 | 0/24 | 948 |
| long/full | 0.75 | sink_recent | 47.92/48.81/48.91 | 3.27/13.43/13.74 | 116.86/149.79/152.27 | 14.24/327.65/333.34 | 0/24 | 36 |
| long/full | 0.75 | random | 50.00/51.37/50.44 | 3.20/12.67/12.74 | 108.99/143.88/145.39 | 14.18/323.94/332.04 | 0/24 | 33 |
| long/full | 0.75 | triattention | 50.66/47.52/48.90 | 3.25/13.30/13.87 | 114.88/148.86/153.40 | 14.32/324.04/330.88 | 0/24 | 35 |
| short/full | 8 | full | 140.28/140.89/142.24 | 0.05/0.07/0.07 | 11.52/11.75/11.76 | 11.37/11.93/14.05 | 6/6 | 0 |
| short/full | 8 | noop | 140.05/138.12/140.87 | 0.05/0.07/0.07 | 11.72/11.94/11.99 | 11.50/11.97/14.08 | 6/6 | 0 |
| short/eager | 8 | full | 62.79/62.78/63.06 | 0.06/0.08/0.08 | 29.18/29.31/29.32 | 28.98/30.03/33.53 | 6/6 | 0 |
| short/eager | 8 | noop | 53.81/53.78/54.01 | 0.05/0.06/0.06 | 34.61/34.75/34.76 | 34.35/35.87/40.10 | 6/6 | 0 |

正式各组未出现计时窗口内Triton JIT警告，相关警告仅在预热。单组NVML峰值最高约
33.164GiB（宽松long），压力有损18.715–18.738GiB。Pool内释放页不会使预分配
NVML下降；tokens/s也不是等量FLOPs比较，有损后E和输出内容都可能变化。
该合成负载不证明任务质量，质量限制沿用M2报告。

## 通信和等待诊断

`20261031-121544-nsys-http-tri-diagnostic`正式诊断窗口19.458s，每rank分别31次
score SUM、plan broadcast与completion SUM。每rank累计逻辑payload分别313148、
156544、124 bytes，不能当作链路实际bus traffic。GPU NCCL累计时间如下，包含
rank等待，不能解释为纯传输延迟。31次commit累计释放2446页，单次9–256页，
10次preemption；该频次属于独立trace轮，不替代正式三轮。

| rank | score(ms) | plan(ms) | completion(ms) |
| ---: | ---: | ---: | ---: |
| 0 | 0.4715 | 0.2485 | 0.2829 |
| 1 | 238.4258 | 15.8534 | 27.2935 |

全部collective span均匹配；缺1个graph GPU span，整体标为`PARTIAL_GPU_SPANS`。
末尾GPU工作距窗口结束约5.52ms，没有窗口截断证据；graph成本仍只能报告已匹配
部分。profile原始sqlite、逐次payload/host/GPU时间和缺失ID均保留。

`20261031-121731-nsys-small-tri-diagnostic`为两次回收，每rank三种通信各2次，
逻辑payload为1660/756/8 bytes，span完整。rank0 GPU时间0.0252/0.0153/0.0485ms，
rank1为42.619/2.973/2.632ms。该轨迹含payload同步hook与首次冷启动，不能冒充
稳态小E通信成本。`20261031-121826-nsys-slow-rank-diagnostic`两次completion
每rank合计8 bytes，rank0 host累计3079.3ms、GPU仅0.0215ms，rank1分别69.16/
45.82ms。人工3s慢rank主要体现在host等待，GPU kernel时间不覆盖完整group等待。

short/eager超预算追加相同负载native/noop完整预热profile，目录分别为
`20261031-121928-nsys-short-eager-full-profile`和
`20261031-122045-nsys-short-eager-noop-profile`。窗口5221.0→5874.3ms；
rank0 kernel总和1172.9→1146.5ms，rank1为4535.0→5382.4ms，其中原生
`cross_device_reduce_1stage`在rank1累计3512.5→4363.4ms。新增physical prepare/
advance GPU执行每rank合计不足1ms，末尾GPU工作距窗口结束4–5ms。证据支持
差异主要表现在rank等待/host launch节奏，尚未定位CPU具体热函数，不声称已确诊
或修复。未据此添加猜测优化。所有诊断不并入正式吞吐，保留14.33%预算FAIL。

## Evidence

大产物根目录`/data_ssd/qcrs/experiments/p1-v8/TP/`，机器汇总`serving-summary.json`。
每组含protocol/prompts/SHA、requests/usage/token IDs、metrics前后快照、server/stdout、
invocation/source、NVML及result。失败`20261031-114106-http-long-full-0.75-noop`
仅预热完成后触发180秒transport timeout；保留FAIL，不合并。后续固定600秒transport
仅用于完整采集，不改变SLO；已成功组未触发原180秒限制，保留其结果。

| 配置 | 目录 |
| --- | --- |
| long/full/8.0/full | `20261031-112520-http-long-full-8-full` |
| long/full/8.0/noop | `20261031-112644-http-long-full-8-noop` |
| long/full/8.0/sink_recent | `20261031-112806-http-long-full-8-sink_recent` |
| long/full/0.75/full | `20261031-112929-http-long-full-0.75-full` |
| long/full/0.75/noop | `20261031-114906-http-long-full-0.75-noop` |
| long/full/0.75/sink_recent | `20261031-120229-http-long-full-0.75-sink_recent` |
| long/full/0.75/random | `20261031-120436-http-long-full-0.75-random` |
| long/full/0.75/triattention | `20261031-120641-http-long-full-0.75-triattention` |
| short/full/8.0/full | `20261031-120847-http-short-full-8-full` |
| short/full/8.0/noop | `20261031-120939-http-short-full-8-noop` |
| short/eager/8.0/full | `20261031-121030-http-short-eager-8-full` |
| short/eager/8.0/noop | `20261031-121131-http-short-eager-8-noop` |

## 功能收尾复核（2026-10-31）

普通decode的physical prepare / GPU E advance已合入原生prepare/post-update
kernel；删除每步只为断言发送的E字典。三条新的Prefix+FULL+TP、q0慢rank、
maintenance慢rank轨迹通过，见 [final-review.md](final-review.md)。

相同short/eager协议、完整预热、每server3正式轮，修改前native/noop为
59.874/61.910tokens/s，未稳定复现历史14.33%。修改后按native/noop/noop/native
次序运行，A对61.541/53.426（13.187%开销），B对62.047/61.670（0.607%），
两对全部轮合并为61.794/57.548（6.871%），仍为预算FAIL。12对输出逐token
相同，短负载SLO均12/12；不挑选单组宣布已解决。独立host诊断观察到慢轮rank0
CPU执行增加，rank1等待增加；跨server的完整原因尚未确定，不归因于两个小kernel。

旧profile的missing span继续标PARTIAL；新
`Review/20261031-144831-tp-full-tri-nsys-flush0`使用flush interval0，不改环境版本。
窗口17.484s，424个graph spans完整；每rank三类通信各27次，162个collective spans
全匹配，无incomplete/drop诊断，tail约5.70ms。每rank逻辑payload为
score307356、plan153652、completion108 bytes；rank0 GPU累计
0.429/0.218/5.026ms，rank1为217.364/17.219/56.698ms，仍包含等待。
该诊断不合入正式吞吐。原始数据和重算入口见final-review。
