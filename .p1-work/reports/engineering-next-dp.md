# 下一阶段 DP 工作量反馈与路由验证

## DP3：实现与状态来源

DP3 已实现；默认仍是 D1 的 `C=max(4W+R,F)`。Scheduler 从唯一的
`requests.values()` 统计未完成请求：`U=sum(max(0,num_tokens-
(num_computed_tokens-num_in_flight_tokens)))`。waiting、chunk 余量、preempt
后的已知历史和 decode 未计算项都在该集合内；不预测 max_tokens、不提前扣除
在途 forward，也不重复查询 Prefix Cache。usage 来自原 KVCacheManager/BlockPool。

`Scheduler.get_dp_load_snapshot` 返回 U、usage、source sequence、真实
`current_step` 和生成 monotonic ns。`processed_step_seq` 是可选 deferred-free
fence，不能当通用迭代计数，本次诊断已据源码修正。EngineCore 拥有发布序列和
上次采样状态：counts 改变立即发送，counts 不变时每100ms采样 U/usage，仅变化
才发送，最后归零立即发布；即使无变化也更新时间，避免退化为逐 token 扫描。
不增加 idle 心跳、GPU读取或前端请求生命周期表。

原 output queue/client=-1 经 Msgpack/ZMQ 到 Coordinator。后者继续按各 dense rank
latest 独立汇总，添加 `coordinator_received_ns`；前端协议为
`(counts,wave,running,loads,published_ns)`，counts 编码仍是 `[waiting,running]`。
`DPAsyncMPClient._apply_load_stats` 类型化解码并按 managed ranks 切片，保存实际
接收时间。MoE 保留自己的 step/wave publisher，未以 dense 实验扩大 MoE 支持域。
状态年龄、Engine→Coordinator、合并等待、publish→Frontend 分开计算；同机
monotonic 关联只适用于本机，不能宣称跨机器时钟已校准。

已有 F 下界、乐观 waiting、轮转 tie、显式 rank、late-interaction sticky、
abort/finished affinity 均保持。local F 只覆盖本 client；usage 仅观察，既不承诺
admission 容量，也不把活跃可回收页算 free。

## 必要验证与诊断

CPU 最终 DP3 选择器 `dplb or dense_dp or moe_dp`：10 passed / 33 deselected。
包含真实 Request 的在途/preempt/decode/finished 工作量、受控时钟的 load-only
合并及立即归零、真实 ZMQ Coordinator→实际前端解码、旧路由/affinity 与 MoE 字段。
日志在 [CPU归档](../experiments/Engineering-Next-DP-Diagnostic/cpu-validation)。
runner 只将两个无关顶层 tokenizer fixture 指向已缓存 Qwen3-8B；不代表完整上游
suite 通过。初次静态失败与修复日志均保留。

两个真实 DP2×TP1 诊断均 Qwen3-8B BF16、GPU0,1、Worker 主线程 CPU20/21、
batch512/seqs8、4GiB/replica、SinkRecent/.5/reserve0、APC/async off、
FULL_DECODE_ONLY `[1,2,4,8]`。输入 token IDs/hash、arrival、源码、affinity、
模型参数和实际 argv 均在各 run 中；一次 guard 包含32/8小预热、完整负载预热、
两个相同 timed rounds，未更改预注册输入。

| 场景与原始结果 | guard / 项目NVML峰 | timed观察（含诊断开销） |
| --- | --- | --- |
| [W1](../experiments/Engineering-Next-DP-Diagnostic/20261101-111851-w1-known-work-shadow-fixed) `[8192,512]*4` | PASS / 42,830,135,296 bytes | 每轮6/8 SLO，约173.49/172.98 output tok/s；每轮prefill为2048/32768，重算0 |
| [W0](../experiments/Engineering-Next-DP-Diagnostic/20261101-112110-w0-known-work-shadow) `[4096]*8` | PASS / 42,830,135,296 bytes | 每轮8/8 SLO，约271.74/239.30 tok/s；prefill为16384/16384、20480/12288，重算0 |

arrivals 均 `[0,.05,.10,.15,.20,.25,.30,.35]` 秒，输出各128。
这些是诊断结果，不能充作正式 A/B 收益。两个副本最终 counts/U/usage 都归零，
sequence/step 前进；观察到 counts 不变的首次送达分别156/132次，证明 load-only
变化实际通过全链路。W1 Engine→Coordinator P50/P95约.87/1.10ms，合并等待
18.45/96.09ms，publish→Frontend .62/1.58ms；W0约.91/1.09、45.74/93.99、
.57/1.18ms。W1 route状态年龄最大3.92s来自完成后不发idle心跳，不是网络延迟。

[首次W1 FAIL](../experiments/Engineering-Next-DP-Diagnostic/20261101-111640-w1-known-work-shadow)
在完整负载预热遇回收时，旧 `dp_hooks` 的 `vars(intent)` 无法JSON序列化嵌套
PolicyBinding；已改 `dataclasses.asdict`，同配置重跑。该失败不是ownership错误，
也未形成完整 timed 样本，不能用于 DP4 门槛或性能汇总；原始失败、日志与显存保留。

复算入口为 [analyze_dp_work.py](../scripts/analyze_dp_work.py)：

```bash
export P1_CUDA_VISIBLE_DEVICES=1
source .p1-work/scripts/env.sh
.venv/bin/python .p1-work/scripts/analyze_dp_work.py \
  .p1-work/experiments/Engineering-Next-DP-Diagnostic/20261101-111851-w1-known-work-shadow-fixed \
  .p1-work/experiments/Engineering-Next-DP-Diagnostic/20261101-112110-w0-known-work-shadow \
  --output /data_ssd/qcrs/experiments/p1-v8/Engineering-Next-DP-Diagnostic/dp3-analysis.json
```

输出保留每个请求的 C/U、source sequences、状态年龄、选择/shadow、实际 post-output
状态及route→first schedule。该区间含入口传输，不等于纯排队时间；脚本只按首达
sequence 计算传播分布，另算调度实际prefill/重复prefill与全部输入hash。

## DP4 进入判断

预注册门槛为两个快照年龄均≤250ms、C差≤1、U差≥512，且跨不同请求/新序列重复，
并关联真实工作量或排队不利。W1两个 timed 的 `r0-6/r1-6` 均满足：C为`[6,5]`、
U为`[2,14848]`，序列分别`[35,52]`/`[63,101]`，年龄约100/160ms；旧路由选择
replica1，shadow选择0。真实post-output中replica1 U=21504、两个waiting长请求，
replica0 U=4、以decode为主；两个请求约2411/2405ms后首次schedule，TTFT约
3.45/3.46s。warmup也有不同新鲜序列的同类观察。不是仅凭shadow不同下结论。

因此条件成立，进入一次显式 `S=C+U/batch` 的 DP4 实现与 W1/W0 A/B。
不修改pool/负载找正结果，不引入usage惩罚、迁移或跨池借页。正式结果随后按独立
源码状态和关闭trace/load-diagnostic的测量记录，不用本节诊断数据代替。

## DP4 实现与有限正式对照

`DPLBAsyncMPClient`读取additional_config的`dp_load_balancing`：默认`requests`
仍使用C，显式`known_work`使用`S=C+U/max_num_batched_tokens`，缺快照的U为0。
非法模式和MoE使用known_work均明确报错。只影响新请求选择；显式rank、sticky、
本地F下界、乐观waiting、轮转tie以及abort/finished affinity保持原语义。
没有增加过期预测、usage惩罚或活跃请求迁移，也没有修改DP策略版本的支持范围。

CPU路由回归16 passed / 33 deselected，覆盖两模式的sticky、F与完成释放、
显式rank和abort、相同counts不同U及缺快照轮转。pre-commit及manual mypy-3.12
通过，初次静态失败日志也在
[CPU归档](../experiments/Engineering-Next-DP-Formal/cpu-validation)。

两边均沿上节W1/W0固定输入、硬件和Runtime配置，完整负载warmup后3轮×8。
`--work-counters`只在CPU统计实际scheduled/prefill/recomputed token，每请求完成
写一次；ADD各写一次route快照。两边使用相同wrapper，不启用trace/load-diagnostic、
payload检查、逐步同步或NSYS。复算检查三轮全部128输出、公式/轮转、工作恒等式、
Prometheus交叉核对、源码/间接helper与输入hash一致；不是拿旧native结果作A。

| 场景/顺序 | 模式与原始归档 | 三轮output tok/s | timed SLO | warmup tok/s；SLO |
| --- | --- | --- | --- | --- |
| W1正向A | [requests](../experiments/Engineering-Next-DP-Formal/20261101-112848-requests-w1) | 176.561 / 175.590 / 175.447 | 18/24，FAIL | 175.516；6/8 |
| W1正向B | [known_work](../experiments/Engineering-Next-DP-Formal/20261101-113054-known-work-w1) | 209.874 / 209.064 / 209.326 | 21/24，FAIL | 208.711；7/8 |
| W0正向A | [requests](../experiments/Engineering-Next-DP-Formal/20261101-113322-requests-w0) | 280.558 / 279.765 / 278.723 | 24/24 | 278.734；8/8 |
| W0正向B | [known_work](../experiments/Engineering-Next-DP-Formal/20261101-113455-known-work-w0) | 276.120 / 274.649 / 275.159 | 24/24 | 274.789；8/8 |
| W1反向B | [known_work](../experiments/Engineering-Next-DP-Formal/20261101-113639-known-work-w1-reverse) | 213.184 / 210.882 / 253.729 | 22/24，FAIL | 258.249；8/8 |
| W1反向A | [requests](../experiments/Engineering-Next-DP-Formal/20261101-113933-requests-w1-reverse) | 216.297 / 215.662 / 215.832 | 21/24，FAIL | 178.644；6/8 |

以下聚合使用全部24条timed请求和三轮elapsed之和；分位不平均，小样本P99只描述
本次请求，warmup不参与聚合。

| 场景/顺序/模式 | output tok/s；goodput req/s | TTFT P50/P99/max，秒 | TPOT P50/P99/max，ms |
| --- | --- | --- | --- |
| W1正向A | 175.864；1.03046 | .4754 / 3.4391 / 3.4396 | 15.865 / 32.901 / 32.904 |
| W1正向B | 209.421；1.43159 | .8375 / 2.5205 / 2.5217 | 18.437 / 27.638 / 27.643 |
| W0正向A | 279.680；2.18500 | .8843 / 1.3798 / 1.3826 | 18.938 / 22.782 / 22.813 |
| W0正向B | 275.308；2.15084 | .8588 / 1.4656 / 1.4809 | 18.890 / 22.830 / 22.855 |
| W1反向B | 224.316；1.60643 | .8796 / 2.4667 / 2.4768 | 18.485 / 27.366 / 27.453 |
| W1反向A | 215.930；1.47608 | .8708 / 2.4730 / 2.4731 | 21.462 / 26.807 / 26.816 |

W1正向吞吐+19.081%、goodput+38.928%，同时TTFT/TPOT P50变差；反向聚合
仅+3.884%/+8.831%，B前两轮均慢于A，第三轮显著更快。A两次启动的聚合吞吐
从175.864到215.930，且下表真实路由/工作分配改变。**不满足稳定超出重复波动的
Serving收益判据**，不能只用正向或反向第三轮写提速。W0吞吐/goodput均-1.563%，
两边24/24 SLO，落在既定≤5%退化预算；没有正向收益，不补W0反向，不继续扫参。

### 实际工作、状态年龄与测量成本

| run | warmup prefill，replica0/1 | 三轮timed prefill，replica0/1 |
| --- | --- | --- |
| W1正向A | 2048 / 32768 | 每轮2048 / 32768 |
| W1正向B | 9728 / 25088 | 每轮9728 / 25088 |
| W0正向A/B | 16384 / 16384 | 两模式每轮均16384 / 16384 |
| W1反向B | 17408 / 17408 | 9728 / 25088；9728 / 25088；17920 / 16896 |
| W1反向A | 2048 / 32768 | 每轮10240 / 24576 |

全部六run的warmup与timed实际重算和Prom抢占均为0。每请求实际scheduled=
prompt+127+recomputed，既统计decode又核对prefill counter；不将prompt counter
单独当作重算证据。正式回收和首次retry次数未采集，明确为UNKNOWN，不复制诊断值。
W1正向A各轮route为`[1,0,1,0,1,0,1,0]`，反向A为
`[1,0,1,0,1,0,0,0]`；反向B第三轮也改变route。这是异步反馈下的实际行为，
不是固定6/2实验，也不能由工作更均匀推导稳定用户收益。

| run | timed快照年龄P50/P95/max，ms | 两副本均≤250ms的ADD | 路由CPU P50/P95，µs | 观测wrapper含写盘P50/P95，µs |
| --- | --- | --- | --- | --- |
| W1正向A | 105.36 / 3705.18 / 3750.98 | 15/24 | 25.61 / 29.40 | 235.24 / 1394.58 |
| W1正向B | 101.64 / 2018.78 / 2066.07 | 15/24 | 27.52 / 29.87 | 232.30 / 261.41 |
| W0正向A | 71.15 / 126.21 / 148.43 | 24/24 | 24.86 / 26.29 | 234.45 / 358.30 |
| W0正向B | 108.12 / 207.94 / 260.02 | 22/24 | 26.43 / 28.68 | 229.06 / 330.01 |
| W1反向B | 103.05 / 1931.10 / 2020.57 | 17/24 | 27.52 / 29.03 | 239.47 / 258.54 |
| W1反向A | 78.48 / 2849.82 / 3625.16 | 15/24 | 25.83 / 29.78 | 237.84 / 325.21 |

年龄包含完成后idle快照，不能视为网络耗时；DP4本身没有250ms过期剔除规则。
CPU成本覆盖每run全部33次ADD（小预热1、完整warmup8、timed24），是同配置
观测下的分布，不宣称无观测时精确开销。逐请求C/U/source sequence及生成、接收、
使用时间保留在`routing.jsonl`，各轮route/工作和ITL分位在复算JSON中。

三对全部24条timed输出token IDs分别一致；不证明有损质量保持。六次guard均
PASS/exit0、无资源触发，NVML项目合计峰：W0两次42,828,038,144 bytes，W1反向B
42,832,232,448 bytes，其余W1均42,830,135,296 bytes，低于44GiB停止阈值。
每run的两个Worker主线程affinity均为CPU20/21；不声称全部helper线程绑定。

### 复算与最终判断

完整可执行argv、exit、输入/源码/helper/派生结果hash见
[DP4分析manifest](../experiments/Engineering-Next-DP-Formal/dp4-analysis-manifest.json)。
[最终复算](../experiments/Engineering-Next-DP-Formal/dp4-final-analysis.json)由
`analyze_dp_work.py --formal <上述六个run> --output <目标JSON>`产生；只读明确传入
的run，不重启GPU或扫描旧矩阵。首次四run的复算和反向结果均保留，未覆盖原始样本。

DP3为IMPLEMENTED/CORRECTNESS_PASS；DP4门槛成立后已实施并完成有限A/B，
正确性PASS，性能PARTIAL、W1 SLO_FAIL。保留显式实验模式，requests仍默认；
停止扩大实验，不宣称SERVING_BENEFIT。旧D1无可分辨收益、旧DP/TP/M2/Prefix
FAIL/PARTIAL和85条质量样本冻结保持原义。
