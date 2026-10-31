# Prefix HTTP Serving：固定共享前缀下的压力与成本

Prefix Serving 的配置结论为：**8 GiB noop 开销预算 PASS；6 GiB SinkRecent
延迟/Goodput预算 PASS；3 GiB和1.5 GiB SinkRecent性能 FAIL**。
后两档虽然吞吐提高，TTFT仍超过预先固定的2秒预算；不能仅据回收或吞吐判通过。
Prefix正确性沿用 [prefix.md](prefix.md) 的限定范围，本轮未修改Runtime或pytest tests。
后续Graph、TP、DP及组合另有独立报告，当前总状态见`../plans/status.md`。

## 协议、环境与复现

预先记录的协议为 [prefix-serving-protocol.md](../notes/prefix-serving-protocol.md)。
Runtime为`bf31e422344815839e210209a869082edf73a31d`；每次invocation另保存实验脚本
snapshot、SHA256与当时diff，不能仅以提交号替代脚本版本。
Qwen3-8B BF16位于`/data/models/Qwen3-8B`，物理GPU1/A100 80GB；
现有Python3.10、torch2.11.0+cu129、Triton3.6.0、vLLM v0.26.0/MRV2。
FLASH_ATTN/NHD/B16、单full-attention group、TP1/DP1、Eager、单在途；
APC on，async/cascade/reserve-full-ISL off，无connector/speculation。
max_model_len=9216、max_num_batched_tokens=2048、max_num_seqs=8。

每轮8请求，prompt长度`[4096,8192]*4`，共享最前256 tokens；紧接共享域的单token
为8个不同数字，使各私有suffix的APC hash chain分离。各组保存相同prompt IDs与SHA256，
temperature=0、seed=0、ignore_eos=true，每请求128输出。
arrival秒数固定为`[0,.3,.6,.9,5,5.3,5.6,5.9]`。
输入是合成英文重复片段，仅测Serving；不构成有损回答质量证据。

三组为Native Full、`sink_recent/retention=1`的noop，以及SinkRecent retention=0.5；
后两组pressure on、reserve_cap=0、sink/recent=16/16、private min_keep=128。
原定6/3/1.5 GiB共9组；6 GiB实测仍有preemption后，在首条8 GiB运行前追加三组
无压力基线，原数据及SLO均保留。每server先32/8预热，再完整负载预热1轮，正式3轮。
完整预热及每正式轮前调用原生`/reset_prefix_cache`并要求`success=true`，不跨轮积累APC。
dev router只在loopback server启用。正式12组均无host trace、payload hook或每步GPU同步；
两条独立trace诊断各1轮正式，诊断延迟不进入性能表。

SLO为**TTFT≤2秒且TPOT≤80毫秒**，按每请求同时满足计算。
TTFT从客户端实际发起到首token_ids chunk；TPOT为首末token接收间隔/127；
ITL为相邻token接收间隔，同chunk tokens共享时刻。
wall time从本轮epoch到全部响应完成，含固定arrival间隔、排队、回收和同步。
goodput为SLO-good请求数/wall time；request/s=output token/s÷128。
每组聚合throughput/goodput用3轮总tokens或总good请求数除总wall time；
noop的≤5%开销预算按预定的三轮吞吐算术均值相对Native计算。

全部GPU实验经run_guard串行执行；NVML每0.1秒采样项目总显存，44 GiB终止阈值、
45 GiB硬上限。14条运行均exit0/guard PASS，未触发限制，未操作其他用户任务。
Pool是显式KV预算，不靠gpu_memory_utilization估计；原生日志容量与实测峰值如下。
每Pool三策略峰值相同，最大25,841,106,944 bytes，约24.066 GiB。

| Pool/GiB | 原生日志KV tokens | 总页数（含1个null页） | 项目NVML峰值/bytes |
| --- | --- | --- | --- |
| 8 | 58,240 | 3,640 | 25,841,106,944 |
| 6 | 43,680 | 2,730 | 23,727,177,728 |
| 3 | 21,840 | 1,365 | 20,491,272,192 |
| 1.5 | 10,912 | 682 | 18,907,922,432 |

```bash
source .p1-work/scripts/env.sh
.venv/bin/python .p1-work/scripts/run_guard.py \
  --stage Prefix --name http-apc-6-sink_recent -- \
  @python .p1-work/scripts/serving_reclamation.py \
  --prefix-caching --policy sink_recent --pool-gib 6 --repeats 3
.venv/bin/python .p1-work/scripts/summarize_prefix_serving.py
```

其余组合只改policy/pool；诊断增加`--trace --repeats 1`。
未升级依赖、下载模型/数据或修改学习基线。

## 正式12组结果

每格斜线分隔正式第1/2/3轮，SLO-good分母均8。
12×3×8=288个正式请求；每轮49,152 logical prompt tokens、1,024输出tokens。

| Pool/GiB | 策略 | output token/s | TTFT P99/s | TPOT P99/ms | SLO-good/8 | goodput req/s | preempt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | Full | 89.47/91.41/91.70 | 2.207/2.013/2.017 | 55.44/53.64/53.76 | 7/7/7 | .6116/.6249/.6269 | 2/2/2 |
| 6 | noop | 90.90/90.68/90.46 | 2.039/2.026/2.019 | 53.79/53.58/53.62 | 7/7/7 | .6214/.6199/.6184 | 2/2/2 |
| 6 | SinkRecent | 99.05/98.74/99.39 | 1.272/1.259/1.264 | 38.78/37.42/36.88 | 8/8/8 | .7738/.7714/.7765 | 0/0/0 |
| 3 | Full | 77.01/76.81/76.93 | 4.288/4.308/4.250 | 34.66/34.76/34.38 | 5/5/5 | .3760/.3750/.3756 | 2/2/2 |
| 3 | noop | 70.83/70.63/70.79 | 5.297/5.394/5.268 | 37.79/38.48/37.69 | 5/5/5 | .3459/.3449/.3457 | 3/3/3 |
| 3 | SinkRecent | 83.86/86.63/89.50 | 2.927/2.622/2.230 | 60.21/61.04/57.51 | 7/6/7 | .5732/.5076/.6118 | 0/0/0 |
| 1.5 | Full | 33.98/34.00/33.70 | 20.863/20.838/21.052 | 30.00/29.73/29.98 | 1/1/1 | .0332/.0332/.0329 | 7/7/7 |
| 1.5 | noop | 33.38/33.52/33.32 | 21.340/21.199/21.376 | 30.17/30.58/30.72 | 1/1/1 | .0326/.0327/.0325 | 7/7/7 |
| 1.5 | SinkRecent | 86.21/86.52/85.32 | 2.353/2.406/2.804 | 59.89/60.37/53.43 | 7/7/7 | .5893/.5914/.5832 | 6/6/5 |
| 8 | Full | 93.88/94.56/94.46 | 1.663/1.656/1.665 | 57.03/57.11/57.07 | 8/8/8 | .7335/.7387/.7379 | 0/0/0 |
| 8 | noop | 93.93/92.94/94.10 | 1.698/1.721/1.718 | 57.15/57.89/57.63 | 8/8/8 | .7338/.7261/.7352 | 0/0/0 |
| 8 | SinkRecent | 95.18/94.73/94.55 | 1.699/1.711/1.701 | 57.69/57.61/57.76 | 8/8/8 | .7436/.7401/.7387 | 0/0/0 |

聚合req/s、goodput和相对结果如下。8 GiB可容纳本轮全部请求的最大页需求，三策略
均无preemption；6 GiB三轮Native均有2次preemption，不能当无压力基线。

| Pool/GiB | req/s：Full/noop/Sink | goodput：Full/noop/Sink | noop均值吞吐开销 | Sink聚合吞吐相对Full | 判定 |
| --- | --- | --- | --- | --- | --- |
| 8 | .7367/.7317/.7408 | .7367/.7317/.7408 | 0.68% | +0.55% | noop开销PASS；Sink无压力收益未证明 |
| 6 | .7098/.7084/.7739 | .6210/.6199/.7739 | 0.20% | +9.04% | Sink延迟/Goodput预算PASS |
| 3 | .6009/.5527/.6766 | .3756/.3455/.5638 | 8.01% | +12.59% | Sink性能FAIL：TTFT超预算 |
| 1.5 | .2648/.2610/.6720 | .0331/.0326/.5880 | 1.45% | +153.77% | Sink性能FAIL：TTFT超预算 |

6 GiB Sink的24个请求全部满足SLO，goodput提高24.61%，preempt总数6→0。
8 GiB的0.55%吞吐差异不作为回收加速证据。3 GiB noop每轮比Native多1次preemption，
吞吐低8.01%；保留该负结果，不能把8 GiB的0.68%外推为所有压力下的开销。
本轮没有profile定位3 GiB noop调度差异，不把相关性写成单一CPU开销因果结论。

以下为每组24请求合并后的描述性分位数；ITL样本为24×127个间隔。
原始`serving.json`和机器汇总还保留每轮P50/P95/P99，不能用小样本P99推断生产尾延迟。

| Pool/GiB | 策略 | TTFT P50/P95/P99 (s) | TPOT P50/P95/P99 (ms) | ITL P50/P95/P99 (ms) |
| --- | --- | --- | --- | --- |
| 6 | Full | 1.060/2.048/2.208 | 42.349/53.940/55.240 | 25.611/199.945/242.784 |
| 6 | noop | 1.063/2.056/2.064 | 42.699/53.800/53.941 | 25.757/204.902/241.877 |
| 6 | SinkRecent | .814/1.266/1.275 | 35.273/38.443/38.726 | 24.801/159.556/194.670 |
| 3 | Full | .846/4.292/4.314 | 31.783/34.656/34.758 | 24.679/30.111/183.597 |
| 3 | noop | 1.461/5.323/5.402 | 34.968/37.789/38.360 | 25.663/37.770/227.198 |
| 3 | SinkRecent | 1.120/2.607/2.933 | 46.747/60.212/61.026 | 25.810/206.348/228.497 |
| 1.5 | Full | 10.835/21.113/21.263 | 27.214/29.960/30.031 | 24.974/27.055/180.443 |
| 1.5 | noop | 11.029/21.573/21.621 | 27.842/30.592/30.737 | 25.492/27.211/173.874 |
| 1.5 | SinkRecent | 1.250/2.433/2.768 | 45.599/59.656/60.547 | 29.449/200.390/224.854 |
| 8 | Full | 1.017/1.668/1.673 | 42.043/57.476/57.520 | 25.509/201.667/238.664 |
| 8 | noop | 1.056/1.723/1.729 | 42.567/57.998/58.253 | 25.440/206.465/242.054 |
| 8 | SinkRecent | 1.051/1.708/1.717 | 42.070/58.096/58.168 | 25.247/204.682/241.214 |

TPOT是整段decode平均间隔，ITL的高分位仍可超过80毫秒；协议未设置ITL SLO。
不能把TPOT通过表述为每个token间隔都通过。
全部正式server日志的Triton JIT警告都在预热：`_advance_effective`在32/8预热，
3 GiB及1.5 GiB Sink的`_compact_stable`在完整负载预热。正式轮未发现JIT警告；
因此不能用首次编译解释3 GiB、1.5 GiB的正式TTFT失败。

## APC与forward工作量的统计口径

所有36个正式轮的每请求usage.cached_tokens均为`[0,256,256,256,256,256,256,256]`，
合计1,792 tokens，占49,152 logical prompt tokens的3.646%。
各请求共享域占4K/8K prompt的6.25%/3.125%；实际P在回收时由canonical引用决定，
两者不混为同一共享率。四个Pool三轮共**96个Native/noop配对回答token IDs全相同**；
该identity证明noop的输出契约，不要求有损Sink与Full相同。

`PrefixCacheStats.record`将preempted queries/hits单独记账，
`PrometheusStatLogger.record`只累加普通queries/hits。waiting admission失败重试仍可
重复lookup，所以公开prefix counter既不包含全部preempt重入，也不等于成功请求数。
例如3 GiB Full每轮公开query/hit=57,344/2,304；1.5 GiB Sink为
90,112/3,328、90,112/3,328、61,440/2,560；均不能替代上述首prefill usage。
所有正式轮首次prefill local_compute指标均为47,360=49,152−1,792；
`Scheduler.schedule`只在num_preemptions≤0时设置PrefillStats，
`Request.take_prefill_stats`只交付一次，因此该指标不是实际总forward或总重算tokens。
源码位置分别为`vllm/v1/metrics/stats.py`、`vllm/v1/metrics/loggers.py`、
`vllm/v1/core/sched/scheduler.py`与`vllm/v1/request.py`。

两条1.5 GiB诊断以monotonic边界截取正式轮，沿每请求ownership周期重建工作量。
在本配置每次WAITING/PREEMPTED allocation_done时，L=0、B16且无lookahead，
由`hit=16*(owned_blocks−ceil(q/16))`恢复此次完整页命中量；
到free_before的`settled_logical_tokens−hit`是该周期实际完成的forward token数。
累加后减`49,152−1,792+8×127=48,376`，得到额外forward工作；
包括prompt/decode重算，不单称prefill重算。compaction搬移和Attention FLOPs不在此token计数内。
分析器核对周期闭合、8个首admission、首hit合计、最终L、周期数=8+preempt。
逐周期的request ID、hit、终点L和forward数保存在机器汇总中，可对照原始trace复核。

| 诊断正式轮 | allocation miss | reclaim select/commit | P | 累计R | preempt/ownership周期 | 峰值持有KV请求数 | 完成forward/额外forward tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Native | 878 | 0/0 | 无 | 0 | 7/15 | 2 | 48,445/69 |
| SinkRecent | 31 | 24/24 | 全256 | 2,223 | 5/13 | 7 | 48,419/43 |

Sink单次commit释放16–251页；每页为全36层合计2.25 MiB，累计2,223页是churn，
不能称同时多出约4.88 GiB容量。预分配Pool的NVML也不会随释放页数同步下降。
Native的7次preempt后再次admission命中量为6,384/2,528/6,864/976/5,312/1,472/5,808，
说明这条APC-on轨迹重用了大部分已计算完整页，额外工作主要是未满页部分。
由此不能把正式吞吐差异主要归因为减少重算；ownership峰值2→7、miss878→31更支持
容量释放改善并发/等待这一解释，但本轮没有NSYS归因，不能量化各环节贡献。
Sink更短的有效Attention长度及有损输出也可能影响吞吐，tokens/s并非等量FLOPs对比。

诊断预热必须分开：Sink为29次回收、累计2,327页、6次preempt，Native为7次preempt。
正式无trace矩阵不具备reclaim频次/P/R或forward账本，不能把诊断数复制到正式轮。
早期Sink诊断protocol曾误写cached_tokens“包含re-admissions”；raw保留，
本报告和汇总`counter_scope_correction`明确修正其解释，不改动原始响应。
后续取消私有suffix发布造成的APC损失，仍以正确性报告C多计算176 tokens的独立对照为证据；
本HTTP负载的八个不同私有suffix不测重复私有suffix的命中收益。

## 验证、限制与Evidence Index

汇总脚本校验12个唯一配置、共享边界/全部prompt SHA、每轮8请求/128输出、usage、
分位数、goodput、无trace、原生reset日志、96个identity与两条诊断周期账本。
`serving-summary-validation.log`保存汇总成功输出；最终pre-commit与diff检查通过，
日志为`serving-pre-commit-final.log`。mypy hook不覆盖`.p1-work`脚本，不能据hook状态
声称做过实质mypy检查。本轮没有重跑已提交的23项针对性/92项Prefix正确性tests；
它们的结果与14项离线模型配置加载失败仍完整保留在正确性报告。

本轮只有三轮、单卡顺序对照，未随机化顺序或估计置信区间；输出128 tokens，
共享比例固定且较小，没有生产负载、长输出、高P矩阵或Prefix有损质量验收。
SinkRecent的6 GiB PASS仅指该配置延迟/Goodput预算，不等于默认启用或完整质量通过。
Random/TriAttention、Graph/TP/DP及组合没有新增Serving证据。

大产物根目录为`/data_ssd/qcrs/experiments/p1-v8/Prefix/`，不纳入Git：

| Pool/GiB | Full | noop | SinkRecent |
| --- | --- | --- | --- |
| 6 | `20261031-024846-http-apc-6-full` | `20261031-025009-http-apc-6-noop` | `20261031-025129-http-apc-6-sink_recent` |
| 3 | `20261031-025246-http-apc-3-full` | `20261031-025415-http-apc-3-noop` | `20261031-025547-http-apc-3-sink_recent` |
| 1.5 | `20261031-025712-http-apc-1.5-full` | `20261031-025950-http-apc-1.5-noop` | `20261031-030231-http-apc-1.5-sink_recent` |
| 8 | `20261031-030430-http-apc-8-full` | `20261031-030550-http-apc-8-noop` | `20261031-030710-http-apc-8-sink_recent` |

诊断为`20261031-024718-http-diagnostic-1p5-sink`、
`20261031-030830-http-diagnostic-1p5-native`。每目录含`invocation.json`、`source/`、
`protocol.json`、`prompts.json`、`server.log`、`stdout.log`、`requests-*.json`、
`metrics-before-*.txt`/`metrics-*.txt`、`warmup-summary.json`、`serving.json`、
`nvml.jsonl`与`result.json`；诊断另有`kv-trace.jsonl`。
根目录机器汇总为`serving-summary.json`，便览为`serving-table.md`；
原始protocol与每轮数据保留，分析器可从这些文件重新生成汇总。
