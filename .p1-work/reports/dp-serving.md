# DP2 HTTP Serving、逐副本压力与路由成本

10组正式配置已全部完成。native两档noop开销预算PASS，压力三种有损策略均
24/24达SLO；人工skewed的SinkRecent仍只有18/24达标，性能FAIL。五条独立trace
均已完成，包含宽松池两副本零回收；其结果不混入正式性能。
局部ownership/abort正确性见[dp.md](dp.md)。

## 固定协议

Qwen3-8B BF16/internal dense DP2，每replica TP1/PP1、独立Engine/Scheduler/
BlockPool和一张物理GPU，使用GPU0,1。MRV2/FA2/NHD/B16，FULL_DECODE_ONLY、
capture sizes `[1,2,4,8]`、Inductor off；APC/async/spec/cascade/connector/
reserve-full-ISL off。源HEAD `ff0788e1f0`加DP脚本；各run的invocation/source及
hook副本保存实际运行版本，正式性能不启用payload或逐step allocator hook。
沿用Python3.10、PyTorch2.11.0+cu129、Triton3.6.0与CUDA Toolkit12.1/GCC12；
激活入口为`../scripts/env.sh`，完整环境审计见[m0.md](m0.md)。

native为8请求4096/8192交替、各128输出、arrival
`[0,.3,.6,.9,5,5.3,5.6,5.9]`，温度0、seed0、ignore_eos。
每replica4GiB为宽松组、1.5GiB为压力组；容量不能跨pool相加作为admission预算。
skewed前6个long显式送replica0，最后2个prompt128/256显式送replica1；这是人工
容量隔离对照，不能据此归因native算法。协议见[dp-protocol.md](../notes/dp-protocol.md)。

每配置先32/8小预热、再完整负载预热，均排除；随后3轮正式测量。transport timeout
600s与固定SLO（TTFT≤2s且TPOT≤80ms）独立。Goodput为同时满足两个条件的
请求数/整轮时间；TPOT是单请求首末输出间平均值，ITL分位数单列，不将平均TPOT
达标解释为每次token gap达标。ITL来自客户端stream chunk接收时间，同chunk内
tokens记同一时间。分位数合并三轮请求/ITL计算，吞吐使用三轮总token/总窗口。
逐replica吞吐与Goodput均用公共窗口；每轮另保留本地active_span，避免只用本地
活动时间掩盖另一副本的长尾。noop预算使用三轮吞吐均值的相对开销，阈值5%。

## 正式结果

10组各3轮、共240个请求均完成128输出，guard exit0。native/noop在4GiB与
1.5GiB各24个输出逐token相同，共48个。4GiB的Full/Sink输出也24/24相同。
宽松noop平均吞吐开销−0.0109%，只是样本内接近0；压力noop为0.0739%，均在
5%预算内。每组正式计时窗口均无Triton JIT警告；相关warning只在预热阶段。

native压力Full每轮190次preemption，三轮570次全部来自replica0；该副本0/12
达SLO，replica1为12/12。Sink/Random/Tri均无preemption，24/24达SLO，
聚合吞吐约109tokens/s，Goodput约0.85req/s。Sink相对Full吞吐约4.88倍、
Goodput约9.75倍，仅描述相同到达/池预算的合成负载，不代表等量计算或质量保持。

skewed的Full每轮221次preemption，三轮663次；Sink三轮分别3/4/3次，
吞吐11.23→101.19tokens/s、Goodput0.0219→0.5929req/s。Sink的replica0
仍有6/18请求TTFT超过2s、P99为2.91s，所以该配置性能FAIL；replica1的6个
短请求全部达标。不能只按总吞吐或preemption下降判通过。

| scenario/GiB/policy | tokens/s（三轮） | Goodput(req/s) | TTFT P50/P95/P99(s) | TPOT P50/P95/P99(ms) | ITL P50/P95/P99(ms) | SLO合计 | preempt合计 |
| --- | --- | ---: | --- | --- | --- | ---: | ---: |
| native/4/full | 109.22/109.05/109.26 | 0.8529 | 0.60/1.13/1.14 | 17.57/24.69/24.75 | 16.76/18.93/152.96 | 24/24 | 0 |
| native/4/noop | 109.70/108.23/109.63 | 0.8530 | 0.61/1.13/1.13 | 17.89/24.82/25.30 | 16.32/19.10/153.76 | 24/24 | 0 |
| native/4/sink_recent | 109.92/108.51/109.82 | 0.8548 | 0.61/1.12/1.14 | 17.82/25.07/25.13 | 16.71/19.40/154.14 | 24/24 | 0 |
| native/1.5/full | 22.43/22.40/22.37 | 0.0875 | 0.62/37.65/37.71 | 17.14/106.42/106.82 | 16.06/193.05/194.47 | 12/24 | 570 |
| native/1.5/noop | 22.40/22.42/22.33 | 0.0874 | 0.61/37.62/37.77 | 17.35/106.48/107.07 | 16.31/193.18/194.68 | 12/24 | 570 |
| native/1.5/sink_recent | 108.59/110.42/108.77 | 0.8535 | 0.61/1.18/1.19 | 17.46/24.58/24.78 | 15.69/18.77/153.82 | 24/24 | 0 |
| native/1.5/random | 109.09/109.65/109.23 | 0.8541 | 0.55/1.18/1.18 | 17.47/24.40/24.58 | 15.78/19.21/153.81 | 24/24 | 0 |
| native/1.5/triattention | 110.49/108.56/110.49 | 0.8581 | 0.61/1.17/1.18 | 17.67/24.22/24.71 | 16.27/18.58/157.10 | 24/24 | 0 |
| skewed/1.5/full | 11.28/11.23/11.17 | 0.0219 | 27.26/83.70/84.13 | 103.37/159.23/160.30 | 18.00/220.02/224.96 | 6/24 | 663 |
| skewed/1.5/sink_recent | 105.06/98.17/100.58 | 0.5929 | 0.91/2.67/2.89 | 35.64/51.25/52.16 | 17.92/196.81/220.06 | 18/24 | 10 |

## 逐replica指标与原生路由

以下请求/prompt tokens为三轮输入总数，不把重复prefill计算量计入该输入账本。
native每轮原生路由均为4个8192-token请求到replica0、4个4096-token请求到
replica1，即请求数4/4但prompt tokens32768/16384。该偏斜在全部native配置
重复出现；4GiB时仍全部达SLO，1.5GiB时Full的本地容量不足导致replica0长尾。
这是本固定交替到达序列的实测，不能推断所有native流量都有同样偏斜。

| scenario/GiB/policy | replica | 请求/prompt tokens | tokens/s | Goodput(req/s) | TTFT P50/P95/P99(s) | TPOT P50/P95/P99(ms) | ITL P50/P95/P99(ms) | SLO | preempt |
| --- | ---: | --- | ---: | ---: | --- | --- | --- | ---: | ---: |
| native/4/full | 0 | 12/98304 | 54.59 | 0.4265 | 0.97/1.14/1.14 | 21.33/24.73/24.76 | 18.31/19.19/197.98 | 12/12 | 0 |
| native/4/full | 1 | 12/49152 | 54.59 | 0.4265 | 0.35/0.38/0.38 | 15.93/17.16/17.19 | 14.71/15.46/42.45 | 12/12 | 0 |
| native/4/noop | 0 | 12/98304 | 54.59 | 0.4265 | 0.97/1.13/1.13 | 21.74/25.11/25.37 | 18.40/19.33/199.57 | 12/12 | 0 |
| native/4/noop | 1 | 12/49152 | 54.59 | 0.4265 | 0.35/0.38/0.38 | 16.17/17.51/17.61 | 14.94/15.72/37.59 | 12/12 | 0 |
| native/4/sink_recent | 0 | 12/98304 | 54.71 | 0.4274 | 0.98/1.14/1.15 | 21.65/25.12/25.14 | 18.58/19.79/199.30 | 12/12 | 0 |
| native/4/sink_recent | 1 | 12/49152 | 54.71 | 0.4274 | 0.35/0.39/0.39 | 16.16/17.45/17.46 | 14.95/15.80/44.42 | 12/12 | 0 |
| native/1.5/full | 0 | 12/98304 | 11.20 | 0.0000 | 19.23/37.70/37.72 | 104.07/106.68/106.87 | 18.45/193.78/195.26 | 0/12 | 570 |
| native/1.5/full | 1 | 12/49152 | 11.20 | 0.0875 | 0.36/0.38/0.38 | 16.13/17.43/17.70 | 14.71/15.79/43.28 | 12/12 | 0 |
| native/1.5/noop | 0 | 12/98304 | 11.19 | 0.0000 | 19.25/37.71/37.79 | 104.27/106.87/107.15 | 18.67/193.91/195.30 | 0/12 | 570 |
| native/1.5/noop | 1 | 12/49152 | 11.19 | 0.0874 | 0.35/0.38/0.38 | 16.07/17.36/17.36 | 14.95/15.43/35.65 | 12/12 | 0 |
| native/1.5/sink_recent | 0 | 12/98304 | 54.63 | 0.4268 | 1.00/1.19/1.19 | 21.06/24.73/24.79 | 17.82/20.93/195.23 | 12/12 | 0 |
| native/1.5/sink_recent | 1 | 12/49152 | 54.63 | 0.4268 | 0.36/0.39/0.39 | 16.12/17.38/17.41 | 14.92/15.69/42.78 | 12/12 | 0 |
| native/1.5/random | 0 | 12/98304 | 54.66 | 0.4270 | 0.88/1.18/1.19 | 20.72/24.50/24.60 | 18.03/20.94/192.31 | 12/12 | 0 |
| native/1.5/random | 1 | 12/49152 | 54.66 | 0.4270 | 0.36/0.39/0.39 | 16.16/17.43/17.45 | 14.93/15.87/42.53 | 12/12 | 0 |
| native/1.5/triattention | 0 | 12/98304 | 54.92 | 0.4291 | 1.00/1.18/1.19 | 20.98/24.53/24.77 | 17.73/19.03/192.56 | 12/12 | 0 |
| native/1.5/triattention | 1 | 12/49152 | 54.92 | 0.4291 | 0.35/0.39/0.39 | 16.25/17.89/17.94 | 15.09/16.16/45.50 | 12/12 | 0 |
| skewed/1.5/full | 0 | 18/110592 | 8.42 | 0.0000 | 44.03/83.84/84.16 | 131.06/159.45/160.39 | 185.78/221.64/225.21 | 0/18 | 663 |
| skewed/1.5/full | 1 | 6/1152 | 2.81 | 0.0219 | 0.05/0.06/0.06 | 13.88/14.13/14.16 | 13.82/14.22/14.95 | 6/6 | 0 |
| skewed/1.5/sink_recent | 0 | 18/110592 | 75.89 | 0.3953 | 1.15/2.74/2.91 | 41.96/51.52/52.22 | 18.44/200.51/221.35 | 12/18 | 10 |
| skewed/1.5/sink_recent | 1 | 6/1152 | 25.30 | 0.1976 | 0.06/0.06/0.06 | 14.67/14.80/14.82 | 14.56/15.06/16.36 | 6/6 | 0 |

公共窗口使native两副本tokens/s相同，但压力Full的replica1每轮只活动约7.9s、
replica0约45s；每轮active_span保留在serving.json。skewed显式分配6/2请求，
不能用它证明native路由缺陷。首个DP Gate保持原生算法，本地回收已使native压力
组全部达SLO；未实现pressure-aware hint的独立A/B，因此不声明routing收益。

## Router CPU与观测成本

每ADD仅调用一次原生选择函数并记录其返回值、当前waiting/running与request ID。
route计时包括原生选择和affinity登记，wrapper包括route、counts复制、JSON写入与
记时观测；两者不是额外吞吐开销的因果估计。下面只取3正式轮共24次ADD；
机器汇总另保留含小预热/完整预热的33次样本，避免将冷启动观测混作稳态路由。

| scenario/GiB/policy | route P50/P95/P99(μs) | wrapper含route/write P50/P95/P99(μs) |
| --- | --- | --- |
| native/4/full | 15.96/16.70/17.10 | 84.95/88.58/96.07 |
| native/4/noop | 15.19/15.81/15.89 | 85.33/94.48/4109.13 |
| native/4/sink_recent | 15.32/16.59/19.01 | 83.33/92.87/104.62 |
| native/1.5/full | 15.25/17.74/19.54 | 86.66/119.75/4170.12 |
| native/1.5/noop | 15.13/16.99/18.94 | 84.06/104.02/112.28 |
| native/1.5/sink_recent | 15.24/16.28/16.56 | 83.27/96.13/98.06 |
| native/1.5/random | 14.99/16.26/16.99 | 83.84/95.84/4002.17 |
| native/1.5/triattention | 15.25/16.19/16.47 | 84.85/103.08/105.68 |
| skewed/1.5/full | 4.72/5.20/6.67 | 73.19/109.40/642.11 |
| skewed/1.5/sink_recent | 3.59/4.98/9.16 | 73.25/4639.86/5436.05 |

native选择本身P50约15μs，wrapper通常约84–87μs，但部分配置P99有4–5ms
outlier；skewed显式rank跳过负载扫描，选择计时更短。这些是含OS调度的墙钟样本，
不足以给outlier定性为I/O或算法瓶颈。所有策略使用相同观测；正式总延迟已包含它，
不从结果中扣除，也不以平均route时间隐藏P99。

## 独立诊断

五条trace均PASS：native/skewed各full/SinkRecent的四条1.5GiB压力诊断，
加一条4GiB/native/SinkRecent宽松诊断，均完整负载预热后1轮带trace。
下表只取正式诊断窗口；预热单独保存在机器汇总。free/occupied、waiting/running为
schedule时刻的样本；压力pool总682页含null，宽松pool总1820页含null。
occupied峰值按
`total−1−min(free)`计算。累计R可能含相同页重复释放，不能当作唯一页数或同时
新增容量。事件、频次和R只归属于这些trace轮，不复制为无trace正式组的精确计数。

| scenario/GiB/policy | replica | miss/preempt | commit/累计R | R/次范围 | free范围/occupied峰值 | max waiting/running | retry steps |
| --- | ---: | --- | --- | --- | --- | --- | ---: |
| native/1.5/full | 0 | 190/190 | 0/0 | — | 33–681/648 | 3/2 | 0 |
| native/1.5/full | 1 | 0/0 | 0/0 | — | 154–681/527 | 0/2 | 0 |
| native/1.5/sink_recent | 0 | 4/0 | 4/768 | 128–256 | 25–681/656 | 0/2 | 4 |
| native/1.5/sink_recent | 1 | 0/0 | 0/0 | — | 154–681/527 | 0/2 | 0 |
| skewed/1.5/full | 0 | 221/221 | 0/0 | — | 33–681/648 | 5/2 | 0 |
| skewed/1.5/full | 1 | 0/0 | 0/0 | — | 642–681/39 | 0/2 | 0 |
| skewed/1.5/sink_recent | 0 | 21/4 | 16/1611 | 32–259 | 0–681/681 | 2/6 | 16 |
| skewed/1.5/sink_recent | 1 | 0/0 | 0/0 | — | 642–681/39 | 0/2 | 0 |
| native/4/sink_recent | 0 | 0/0 | 0/0 | — | 779–1819/1040 | 1/2 | 0 |
| native/4/sink_recent | 1 | 0/0 | 0/0 | — | 1292–1819/527 | 0/2 | 0 |

Sink的native诊断在replica0有4次回收、4次deferred retry，无native preemption；
skewed诊断仍有4次preemption。replica1在四条轨迹中均无miss/select/commit，
其free不会进入replica0本地available。压力Full在peer空闲时仍发生本地preemption，
不能以聚合free掩盖这一差异。诊断时间含逐step JSON观测，不纳入正式吞吐。

宽松诊断在完整预热与正式窗口的两个replica中，allocation miss、reclaim_select、
schedule intent、reclaim_commit和retry均为0。结合worker只消费已发布intent的
调用链，证明本轨迹未进入评分/压紧链；不能仅从无preemption推断零回收。

## 资源与结论边界

正式10组的项目NVML合计峰值最高43,220,205,568 bytes（40.25GiB），未触发
44GiB guard停止阈值或45GiB硬上限。4GiB×2使用较多显存是两个完整8B副本加
各自KV/Graph资源；页归还预分配Pool不会使NVML同步下降。1.5GiB native组峰值
37,952,159,744 bytes（35.35GiB），skewed约35.12GiB。这里统计带项目标记的
进程；同机有非项目GPU任务，未停止它们，因此不作独占设备的性能声明。

每个配置只有3轮、逐replica只有6–18个请求，P99是该有限样本的插值，不能当作
生产尾延迟估计。合成重复文本的输出固定128 tokens，不证明实际任务质量；有损
质量边界沿用[m2-evaluation.md](m2-evaluation.md)的85样本配对结果，未证明质量
保持。native压力下三种有损策略的微小吞吐差异不足以排序策略优劣。

本阶段没有新增Runtime分支；dense DP的既有拓扑与TP1本地回收契约已满足隔离
需求。正式Full/noop压力配置与skewed Sink仍有SLO失败；root汇总PASS不会覆盖
这些负结果。DP2×TP2、MoE DP/EP、跨副本APC共享与active KV迁移不在本Gate内。

## 复现与产物

```bash
bash .p1-work/scripts/run_dp_serving.sh
.venv/bin/python .p1-work/scripts/summarize_dp_serving.py
```

大产物根目录`/data_ssd/qcrs/experiments/p1-v8/DP/`。每组保留protocol/prompts/
SHA、requests/usage/token IDs、metrics前后快照、routing、server/stdout、invocation/
source、NVML与退出码；汇总为`serving-summary.json`。根status只表示证据完整性
与native/noop输出一致性，具体SLO、noop预算及质量边界分别报告。
汇总器重新读取原始请求、路由与trace，核对10组正式配置、5条诊断、两条正确性
轨迹及48个native/noop输出；`summary-verification.log`保留完整解析结果。

| 配置 | 目录 |
| --- | --- |
| native/4/full | `20261031-123148-http-native-4-full` |
| native/4/noop | `20261031-123316-http-native-4-noop` |
| native/4/sink_recent | `20261031-123445-http-native-4-sink_recent` |
| native/1.5/full | `20261031-123615-http-native-1.5-full` |
| native/1.5/noop | `20261031-124008-http-native-1.5-noop` |
| native/1.5/sink_recent | `20261031-124401-http-native-1.5-sink_recent` |
| native/1.5/random | `20261031-124530-http-native-1.5-random` |
| native/1.5/triattention | `20261031-124659-http-native-1.5-triattention` |
| skewed/1.5/full | `20261031-124827-http-skewed-1.5-full` |
| skewed/1.5/sink_recent | `20261031-125521-http-skewed-1.5-sink_recent` |

| 独立诊断 | 目录 |
| --- | --- |
| native/1.5/full | `20261031-125653-http-native-1.5-full-diagnostic` |
| native/1.5/sink_recent | `20261031-125914-http-native-1.5-sink_recent-diagnostic` |
| skewed/1.5/full | `20261031-130024-http-skewed-1.5-full-diagnostic` |
| skewed/1.5/sink_recent | `20261031-130417-http-skewed-1.5-sink_recent-diagnostic` |
| native/4/sink_recent | `20261031-130544-http-native-4-sink_recent-diagnostic` |
