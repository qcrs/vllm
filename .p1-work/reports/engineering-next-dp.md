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
