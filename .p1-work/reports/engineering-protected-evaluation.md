# Protected Controller：最终统一评估

2026-11-07，按 prompt4 在同一最终 Runtime 源码完成 correctness、micro 和固定
Serving 对照。protected 为 opt-in，fixed 预算与 requests 路由继续默认。单次最多
删除私有 E 的20%，至少保留已计算私有 L 的50%；两者不是有损质量承诺。
DP usage tie-break 的进入条件不成立，实施停止，详见
[DP 诊断](engineering-protected-dp.md)。

## 源码、配置与证据口径

最终 Runtime 的 SHA256：

| 文件 | SHA256 |
| --- | --- |
| `vllm/v1/kv_reclamation.py` | `6b35ffc4722f103ffcd82aa32a5711223f089e750e5187cde35f716ad4cc8ccb` |
| `vllm/v1/core/sched/scheduler.py` | `094a20d75be085d6158a4741a91b9c8a056654e61e6a65d107a9312124f5870b` |
| `vllm/v1/core/kv_cache_manager.py` | `7d6c955dcfe8127457a09b204feac27c729b9392fda8fc627260bd4fae92a494` |
| `vllm/v1/worker/gpu/kv_reclamation.py` | `a0d63011bfca3b994ed9a082640451ce2571aeb315d77b849f7c3fd7b0f45918` |
| `vllm/v1/engine/core_client.py` | `eb860f61a17738d6fc451f16cb6c1f17fcf796ce6c8d5d3c71bc7d9cbd737547` |

Scheduler 工作树 hash 含用户原有中文注释；提交时仅排除该注释，所有实际实验使用
同一源码。HEAD 随文档/阶段提交前进，不能只由 HEAD 推断源码相同；完整 hash、
工作树补丁、准确 argv、退出码、input/config、affinity 和 NVML 保存在每个 run。
`worker-affinity-*.json` 只记录 PID/rank/CPU；micro 的 rank 结果直接记录 GPU UUID。
GPU0/1 的 UUID 分别为 `GPU-3ecfad62-035b-2626-e769-79c785e7665d`、
`GPU-a53fa601-8471-d782-2971-46e5a8e5d328`。B0 `hardware-observation.json`
是 native 完成后、fixed 运行期间的主机观测，不能冒称 native 的即时 Worker UUID。
protected 及后续正式运行另以 `hardware-observations.jsonl` 关联运行中 PID 与 UUID。

固定环境为现有 `.venv`、Qwen3-8B BF16、MRV2、FA2/NHD/B16，APC/async off；
GPU0/1 两张 A100 80GB PCIe，Worker 主线程分别 CPU20/21。全部 GPU 由
`run_guard.py` 同锁串行，项目进程显存合计44GiB停止、45GiB硬限；无模型/依赖下载
或升级，没有占用 GPU2。micro 的 KV 为合成 Tensor，不能称为模型级性能。

以下所有原始目录相对于 `/data_ssd/qcrs/experiments/p1-v8`。B0 为
`Engineering-Protected-B0/20261107-204106-initial`，统一复算入口为
`scripts/analyze_protected_evaluation.py`，复算输出与验证日志保存在 B0。
本轮没有重跑旧矩阵，也没有扩大冻结的85条质量样本。

## Correctness

| 最终检查 | 实际结果与边界 |
| --- | --- |
| CPU 预算/Pool/Scheduler | 三个既有 suite 选择 `reclamation or pressure`，77 passed / 289 deselected；含单次 cap、逻辑 floor、尾页、单/双 victim、不可覆盖、坏第二 intent/ACK 无部分 mutation、abort/finished |
| 分担优化 | 四组 E/L/P 的独立 Fraction 穷举 oracle 与交点相邻整数页选择一致；不以实现自身作 reference |
| DP pending ADD | 既有本地 tokenizer fixture runner，15 passed / 36 deselected；核对 watermark/route/F/dense/MoE 边界，不称完整上游 suite |
| TP2 双 victim | `Engineering-Protected-Final-Correctness/20261107-213244-pair-tp2-graph-delay-reuse` PASS；真实 miss D4、首批两 victim，整批 completion 后 commit/free，全36层 payload、两 rank 同计划、Graph replay，实际改写2/3/4/17/25/26，最终free32 |
| TP2 单 victim q0 | `.../20261107-213340-single-q0-tp2-graph-reuse` PASS；两次 token/whole-block，Graph/continuation/真实页面复用，最终free32 |
| DP2 组合 lifecycle | `.../20261107-213909-dp2-known-work-pressure-abort-reuse` PASS；独立池、本地两次回收，A/C真实改写17；abort后peer继续112次验证forward，C/B分别完成64/128token，最终free=[32,32] |
| 故障边界 | C2已有 mutation 后第二 victim/rank1故障注入，修复 CUDA teardown 阻塞后 fail-stop PASS，0 canonical commit/free；未重跑该阶段矩阵 |
| 静态检查 | 受影响 pre-commit、mypy-3.10及manual mypy-3.12 PASS；准确命令、日志、hash在 B0 `final-validation/` |

TP2双 victim 最终两 rank 各61 FULL、46 matched-eager、72 FP32 attention reference；
延迟3秒期间 canonical row/free 不变。单 victim 各62 FULL、47 matched-eager、48 FP32
attention reference。TP2峰值分别18,152,947,712和18,144,559,104 bytes，DP组合
34,183,577,600 bytes，均未触发资源阈值。

原 DP harness 在 `.../20261107-213442-dp2-known-work-local-pressure` 超时 FAIL：
protected 释放 D1 后，前方 A decode先占用该页，C还未开始；旧脚本等待C首次输出
才触发abort。实际100次commit、199次miss，不能宣称立即admission。只修订
correctness harness 的事件等待顺序后完成后续验证，Runtime与正式输入不变；
`c_started_before_abort=false`。旧FAIL和后续PASS均保留，详见DP报告。

## Micro

原始 run：`Engineering-Protected-Final-Micro/20261107-213419-controller-tp2-route`，
guard PASS，23.09秒、项目NVML峰17,179,869,184 bytes。入口
`benchmarks/kernels/benchmark_protected_reclamation.py --runtime-shapes <既有shape文件>
--worker-cpus 20 21`；完整参数与shape/hash随run归档。

CPU调用真实 Scheduler 候选扫描/选择，3次预热、30次计时；2或8候选。下表为
中位μs，每格为“2候选 / 8候选”。pair只在无单候选可覆盖时进入，D取单候选
容量+1。所有结果独立复算，不把不同释放量称为等动作加速。

| E / L / P | fixed single | protected D1 | protected pair |
| --- | --- | --- | --- |
| 2051 / 4100 / 16 | 65.949 / 253.143 | 35.793 / 128.700，无合法计划 | 33.124 / 126.673，无合法计划 |
| 4100 / 4100 / 16 | 107.330 / 418.076 | 131.575 / 510.064 | 167.913 / 1792.937，D52 |
| 8194 / 8194 / 16 | 189.223 / 760.759 | 209.534 / 830.357 | 248.331 / 2184.499，D104 |
| 272 / 272 / 0 | 22.331 / 75.757 | 38.787 / 141.162 | 75.652 / 1382.265，D4 |

两真实 NCCL rank、36层合成KV、实际stride `[16384,256,1024,1]`，3次预热+10次
样本；先取每个样本的最慢rank，再取中位。`worker_batch` 覆盖实际
`KVReclamationWorker.reclaim_batch` 的plan、通信、搬移、device metadata与全rank
completion；**不含模型forward、Scheduler决策或canonical commit/free**。
compaction单独计时；两者的wall差不能当作纯可消除调度开销。

| E / L / P | 动作 | victim / 页 | 估计copy bytes，每rank | Worker完成wall ms | compaction设备区间 ms |
| --- | --- | --- | --- | --- | --- |
| 2051 / 4100 / 16 | fixed | 1 / 64 | 149,962,752 | 8.187 | 0.340 |
| 2051 / 4100 / 16 | protected | 无合法计划 | 0 | 不执行 | 不执行 |
| 4100 / 4100 / 16 | fixed | 1 / 128 | 301,105,152 | 8.641 | 0.713 |
| 4100 / 4100 / 16 | protected single | 1 / 1 | 601,620,480 | 8.669 | 0.697 |
| 4100 / 4100 / 16 | protected pair | 2 / 52 | 1,085,276,160 | 12.945 | 1.210 |
| 8194 / 8194 / 16 | fixed | 1 / 256 | 602,947,584 | 8.802 | 1.160 |
| 8194 / 8194 / 16 | protected single | 1 / 1 | 1,205,600,256 | 9.553 | 1.340 |
| 8194 / 8194 / 16 | protected pair | 2 / 104 | 2,170,552,320 | 15.511 | 5.064 |
| 272 / 272 / 0 | fixed | 1 / 8 | 17,694,720 | 0.537 | 0.080 |
| 272 / 272 / 0 | protected single | 1 / 1，zero-copy | 0 | 0.360 | 0.015，空操作 |
| 272 / 272 / 0 | protected pair | 2 / 4，两zero-copy | 0 | 13.503 | 0.014，空操作 |

copy bytes为策略估计的读写量，不是显存峰值或实测带宽。device事件区间可包含
launch/通信空隙，不能当作纯kernel耗时。每sample的host调用返回wall、设备区间、
完成wall以及compaction完成wall均在 B0 `final-micro-analysis.json`，原10样本在run。
小shape双zero-copy仍出现13.503ms完成wall，存在collective/排队波动；不由单项
0.360ms宣称稳定优化，也不追加无依据的micro扫描。

不带观测wrapper的真实route方法1000样本中位：requests 3.012μs、known_work
4.154μs。它只度量CPU方法，不包含跨进程消息或完整HTTP路径。

微观结论是成本已量化：protected拒绝低于逻辑floor的动作；大K的小额释放可能复制
更多数据，pair有额外候选比较和两次执行成本。未证明本轮protected带来micro加速；
既有T1/T2的历史加速百分比不在本轮重新归因。

## Serving

每组先排除32/8小请求和一整轮 workload 预热，再采3轮×8请求；每请求输出128。
只在 TTFT≤2s 且 TPOT≤80ms 时计入通过数，goodput 为通过请求数除以三轮总耗时。
分位数由24个请求合并计算，ITL按原 token 间隔样本口径；小样本只作描述。
独立复算逐请求 token 数、工作量恒等式、Prometheus抢占、回收commit和源/input/helper
hash；不以资源guard PASS替代SLO结论。

### TP2 固定压力

FULL_DECODE_ONLY `[1,2,4,8]`，batch1024、maxseq8、0.75GiB/rank，APC off、reserve0。
输入 `[4096,8192]*4`，到达 `[0,.3,.6,.9,5,5.3,5.6,5.9]`秒。fixed/protected均
SinkRecent、retention=.5；native禁用回收。逐步trace、payload验证和profiler关闭，
只保留请求级工作量与累计commit计数。所有以下汇总排除warmup。

| 模式 | 联合SLO | goodput req/s | 输出tok/s | TTFT p50/p95/p99 秒 | TPOT p50/p95/p99 ms |
| --- | --- | --- | --- | --- | --- |
| native | 0/24 FAIL | 0 | 5.882 | 84.220 / 171.448 / 178.288 | 174.551 / 236.637 / 239.824 |
| fixed | 0/24 FAIL | 0 | 43.037 | 4.528 / 15.718 / 15.895 | 121.160 / 163.021 / 168.080 |
| protected | 0/24 FAIL | 0 | 4.905 | 100.698 / 199.107 / 203.235 | 212.428 / 264.884 / 267.770 |

native 原始目录 `Engineering-Protected-Final-TP-Formal/20261107-214010-native-forward`，
fixed 为 `.../20261107-215244-fixed-forward`，protected 为
`.../20261107-215509-protected-forward`；均guard PASS。native三轮输出吞吐
6.768/5.475/5.567，fixed为43.607/43.609/41.938，protected为4.927/4.811/4.981tok/s。
native轮次有漂移，且共享机器有其他用户任务；不把相对native的差额包装为稳定收益。

| 模式 | scheduled / prefill / recompute tokens | miss / victim回收 / preempt | 释放页 / zero-copy / 最大victim | 估计copy bytes，每rank |
| --- | --- | --- | --- | --- |
| native | 2,249,685 / 2,246,637 / 2,099,181 | 594 / 0 / 594 | 0 / 0 / 0 | 0 |
| fixed | 254,235 / 251,187 / 103,731 | 111 / 90 / 15 | 8,058 / 6 / 1 | 16,879,435,776 |
| protected | 1,844,547 / 1,841,499 / 1,694,043 | 948 / 480 / 468 | 963 / 0 / 1 | 436,677,378,048 |

ITL p50/p95/p99 分别为 native 185.931/339.133/377.561ms、fixed
26.451/305.268/314.814ms、protected 199.077/436.667/452.298ms。
工作量满足每请求 `scheduled=prompt+127+recompute`，抢占计数与Prometheus一致，
victim回收数与canonical commit增量一致。释放页为累计归还次数，不是唯一页面或
同时新增容量；copy bytes为策略估计，不是实测传输量。正式最大victim=1，不能由
正确性里的pair证明本负载触发pair。三组NVML峰为19,876,806,656 / 19,931,332,608 /
19,931,332,608 bytes。

protected相对fixed的观测吞吐下降88.60%，相对native下降16.60%。每轮protected
均156次抢占、160次回收；fixed均5次抢占、30次回收。保护floor限制了连续压缩，
精确小额释放又没有为排在前方的decode预留容量，反复搬移和后续重算抵消了释放。
这是本负载下的负结果：容量约束通过，SLO与性能目标FAIL。没有正向protected收益
需要复核，且不就native漂移主张因果收益，因此不追加昂贵TP反向或参数扫描。

### DP2 W1/W0 与组合

固定 SinkRecent、retention=.5、reserve0、每replica独立4GiB池，batch512、maxseq8、
FULL_DECODE_ONLY `[1,2,4,8]`，APC off、Worker主线程CPU20/21。单项均使用 fixed
预算；组合仅改为 protected。W1输入 `[8192,512]*4`，W0输入 `[4096]*8`，
到达均为 `[i*.05 for i in range(8)]`秒，128输出。完整预热后各3轮，关闭逐步
trace、load-diagnostic、payload验证和profiler；保留每次ADD选路与请求完成的标量
工作量计数。正向先requests后known_work，反向交换顺序，两种路由所有轮次均纳入。

| 场景 / 顺序 / 路由 | 联合SLO | goodput req/s | 输出tok/s | TTFT p50/p95/p99 秒 | TPOT p50/p95/p99 ms |
| --- | --- | --- | --- | --- | --- |
| W1 正向 requests | 19/24 | 0.977803 | 158.095 | 0.858 / 2.529 / 3.137 | 29.638 / 47.343 / 48.162 |
| W1 正向 known_work | 17/24 | 0.685843 | 123.936 | 1.491 / 4.274 / 4.491 | 24.691 / 56.193 / 57.526 |
| W0 正向 requests | 16/24 | 0.635413 | 121.999 | 1.193 / 3.668 / 4.348 | 31.285 / 53.675 / 58.356 |
| W0 正向 known_work | 20/24 | 1.157634 | 177.813 | 1.029 / 2.235 / 2.667 | 22.201 / 37.858 / 42.068 |
| W1 反向 known_work | 18/24 | 0.952901 | 162.628 | 1.431 / 2.969 / 3.031 | 25.843 / 41.594 / 46.181 |
| W1 反向 requests | 17/24 | 0.896984 | 162.090 | 1.710 / 2.518 / 3.170 | 31.416 / 41.136 / 46.031 |
| W0 反向 known_work | 17/24 | 0.727473 | 131.459 | 1.134 / 3.338 / 3.438 | 27.755 / 48.094 / 49.506 |
| W0 反向 requests | 18/24 | 0.775212 | 132.303 | 1.186 / 3.132 / 3.746 | 25.027 / 48.512 / 54.195 |
| 组合 W1 known_work+protected | 24/24 | 2.022092 | 258.828 | 0.874 / 1.718 / 1.731 | 21.177 / 25.918 / 26.197 |

八组DP单项均 SLO_FAIL，主要缺口是TTFT；全部逐请求结果保存在各目录
`requests-*.json`，并由 B0 `final-unified-analysis.json` 的 `request_slo` 独立判定。
组合的24/24只代表该次固定运行满足项目SLO，不表示消除了本地压力或获得可归因收益。

| 场景 / 顺序 / 路由 | 三轮联合SLO | 三轮输出tok/s | ITL p50/p95/p99 ms |
| --- | --- | --- | --- |
| W1 正向 requests | 6/8 / 7/8 / 6/8 | 177.477 / 149.287 / 150.537 | 25.737 / 58.602 / 118.718 |
| W1 正向 known_work | 5/8 / 7/8 / 5/8 | 124.959 / 124.568 / 122.314 | 16.703 / 111.730 / 158.748 |
| W0 正向 requests | 6/8 / 5/8 / 5/8 | 124.151 / 144.175 / 104.171 | 18.678 / 72.108 / 136.408 |
| W0 正向 known_work | 7/8 / 6/8 / 7/8 | 209.521 / 146.706 / 189.303 | 16.113 / 51.859 / 102.231 |
| W1 反向 known_work | 7/8 / 5/8 / 6/8 | 170.650 / 150.509 / 168.268 | 16.933 / 60.258 / 133.867 |
| W1 反向 requests | 6/8 / 4/8 / 7/8 | 176.903 / 147.568 / 164.504 | 24.832 / 58.522 / 103.501 |
| W0 反向 known_work | 5/8 / 6/8 / 6/8 | 129.870 / 129.928 / 134.694 | 18.016 / 57.466 / 132.305 |
| W0 反向 requests | 6/8 / 6/8 / 6/8 | 149.064 / 116.694 / 135.185 | 17.182 / 54.057 / 138.124 |
| 组合 W1 known_work+protected | 8/8 / 8/8 / 8/8 | 259.840 / 259.235 / 257.421 | 16.287 / 55.002 / 61.539 |

known_work相对同顺序requests的汇总差额如下。W1正向退化、W0正向增益
在反向未稳定复现，轮内与轮间也有波动；不挑选最快轮次、不以小样本分位数宣称
显著性。因此本轮为 **NO_STABLE_ROUTING_BENEFIT / SLO_FAIL**，停止扩大实验。

| 负载 | 顺序 | 输出吞吐变化 | goodput变化 | 相同timed token IDs |
| --- | --- | --- | --- | --- |
| W1 | 正向 | -21.607% | -29.859% | 24/24 |
| W0 | 正向 | +45.749% | +82.186% | 24/24 |
| W1 | 反向 | +0.332% | +6.234% | 24/24 |
| W0 | 反向 | -0.638% | -6.158% | 24/24 |

四对各24条输出token IDs相同，是这些样本的可观察结果，不证明有损策略
质量保持。历史pending ADD修复的有限收益、原DP4 PARTIAL与SLO_FAIL保留各自配置
和时间范围，不由本轮覆盖。

### DP 工作分配、usage 与观测成本

下表是三轮timed的合计，rank顺序为replica0/replica1。usage峰值取请求工作量
计数覆盖区间内的最大值；不能解释为选路时已知容量或NVML显存利用率。

| 场景 / 顺序 / 路由 | 请求数 r0:r1 | prefill tokens r0/r1 | scheduled tokens r0/r1 | 请求计数区间usage峰值 r0/r1 |
| --- | --- | --- | --- | --- |
| W1 正向 requests | 13 : 11 | 22,016 / 82,432 | 23,667 / 83,829 | 37.273% / 99.945% |
| W1 正向 known_work | 12 : 12 | 52,224 / 52,224 | 53,748 / 53,748 | 61.352% / 61.462% |
| W0 正向 requests | 14 : 10 | 57,344 / 40,960 | 59,122 / 42,230 | 72.347% / 57.944% |
| W0 正向 known_work | 12 : 12 | 49,152 / 49,152 | 50,676 / 50,676 | 57.944% / 57.944% |
| W1 反向 known_work | 12 : 12 | 52,224 / 52,224 | 53,748 / 53,748 | 61.352% / 61.517% |
| W1 反向 requests | 13 : 11 | 22,016 / 82,432 | 23,667 / 83,829 | 37.218% / 99.945% |
| W0 反向 known_work | 12 : 12 | 49,152 / 49,152 | 50,676 / 50,676 | 57.944% / 57.944% |
| W0 反向 requests | 12 : 12 | 49,152 / 49,152 | 50,676 / 50,676 | 57.944% / 57.944% |
| 组合 W1 known_work+protected | 12 : 12 | 52,224 / 52,224 | 53,748 / 53,748 | 61.297% / 61.517% |

W1 requests正反向各在replica1出现1次allocation miss、1次fixed回收，释放
257页、估计copy bytes为1,206,779,904；两次都无preemption/recompute。其余DP单项
及组合的miss/reclaim/preempt/recompute均为0。全部工作量满足
`scheduled=prompt+127+recompute`，请求/副本映射与Prometheus prompt累计相符。
known_work使本轮W1按已知token量分配均衡，仍不足以证明Serving提速。

以下是每组24次timed选路、每rank一个latest snapshot的usage，以及合并48个
snapshot的年龄。年龄含空闲时保留的旧快照；counts变化即时发布，纯U/usage变化
按100ms合并，空闲无heartbeat，Runtime没有wall-age expiry。这里不把250ms诊断
筛选变成正式路由合同，也不把旧快照的低usage当成admission保证。

| 场景 / 顺序 / 路由 | 选路usage r0 中位/最大 | 选路usage r1 中位/最大 | snapshot年龄 min/中位/max ms |
| --- | --- | --- | --- |
| W1 正向 requests | 0.880% / 5.388% | 2.639% / 12.314% | 9.591 / 83.999 / 2624.559 |
| W1 正向 known_work | 0.880% / 5.333% | 1.759% / 8.796% | 5.854 / 91.434 / 4780.308 |
| W0 正向 requests | 0.000% / 5.278% | 1.759% / 12.314% | 18.577 / 98.732 / 4703.058 |
| W0 正向 known_work | 0.880% / 7.037% | 1.759% / 12.314% | 19.281 / 97.875 / 3452.105 |
| W1 反向 known_work | 1.759% / 7.092% | 1.759% / 10.555% | 42.175 / 108.411 / 2499.203 |
| W1 反向 requests | 1.759% / 5.388% | 1.759% / 12.314% | 27.446 / 104.656 / 2150.580 |
| W0 反向 known_work | 0.000% / 5.278% | 1.759% / 12.314% | 20.672 / 97.420 / 4425.657 |
| W0 反向 requests | 1.759% / 8.796% | 1.759% / 12.314% | 16.261 / 91.906 / 5218.190 |
| 组合 W1 known_work+protected | 1.979% / 30.786% | 1.759% / 10.555% | 31.611 / 100.506 / 2047.228 |

下表route方法与观测wrapper分别计时；每组33次，包含small/full warmup，
不能和上面的24请求timed汇总混作同一采样域。wrapper含JSON写入成本；本次正式
观测存在该固定扰动，不能把wrapper耗时当成生产router成本。无wrapper的1000样本
纯方法micro另见前节。pending列是选路时存在未确认ADD的次数，以及单replica补计
token上界；复算逐事件核对水位确认与撤销，未新增另一份Runtime状态。

| 场景 / 顺序 / 路由 | route p50/p95 μs | 含写入wrapper p50/p95 μs | 含pending的选路次数 | 最大pending tokens |
| --- | --- | --- | --- | --- |
| W1 正向 requests | 27.112 / 37.225 | 290.539 / 442.011 | 0/33 | 0 |
| W1 正向 known_work | 41.011 / 54.124 | 328.188 / 1309.589 | 19/33 | 8,704 |
| W0 正向 requests | 26.813 / 33.253 | 282.650 / 493.154 | 0/33 | 0 |
| W0 正向 known_work | 39.489 / 53.218 | 322.820 / 461.581 | 20/33 | 8,192 |
| W1 反向 known_work | 39.267 / 52.751 | 323.431 / 395.139 | 23/33 | 8,704 |
| W1 反向 requests | 28.548 / 36.120 | 291.646 / 397.984 | 0/33 | 0 |
| W0 反向 known_work | 29.718 / 45.860 | 242.527 / 359.314 | 19/33 | 8,192 |
| W0 反向 requests | 27.949 / 29.555 | 289.763 / 326.114 | 0/33 | 0 |
| 组合 W1 known_work+protected | 38.520 / 43.150 | 320.999 / 367.316 | 22/33 | 8,704 |

原始DP目录均位于 `Engineering-Protected-Final-DP-Formal/`，组合位于
`Engineering-Protected-Final-Combo-Formal/`：

| 场景 / 顺序 / 路由 | 原始run目录 |
| --- | --- |
| W1 正向 requests | `20261107-220952-w1-requests-forward` |
| W1 正向 known_work | `20261107-221114-w1-known_work-forward` |
| W0 正向 requests | `20261107-221243-w0-requests-forward` |
| W0 正向 known_work | `20261107-221410-w0-known_work-forward` |
| W1 反向 known_work | `20261107-221856-w1-known_work-reverse` |
| W1 反向 requests | `20261107-222021-w1-requests-reverse` |
| W0 反向 known_work | `20261107-222147-w0-known_work-reverse` |
| W0 反向 requests | `20261107-222413-w0-requests-reverse` |
| 组合 W1 known_work+protected | `20261107-221526-w1-known-work-protected` |

以上九组guard均PASS/exit0，项目NVML合计峰值最高42,830,135,296 bytes，低于
44GiB停止线。八组DP正反向及组合均有运行中PID→GPU UUID、Worker CPU20/21观测；
GPU0存在其他用户负载，不能宣称独占或由运行顺序差异推出因果收益。

组合三轮均8/8、输出吞吐259.840/259.235/257.421tok/s，但0次回收，不能把
258.828tok/s归因于protected。真正本地压力下的known_work+protected能力由前述
独立小池correctness运行支持，且等待请求在abort前尚未开始。DP usage tie-break
仍 **STOPPED / NOT_RUN**：诊断未得到重复新鲜同分错选后的miss/抢占证据；正式
W1 requests的两次本地miss也不自动证明该进入条件。

## 完成判断与复核入口

protected单/双victim保护及TP2/Graph/真实页复用完成；micro成本完成量化，本轮
没有已证明的micro加速。TP2为负性能结果，DP没有稳定路由收益，固定SLO保持
TTFT≤2s且TPOT≤80ms。组合满足该次SLO，但正式运行未触发回收。

统一复算文件为 B0 `final-unified-analysis.json`，`analysis-manifest.json`保存准确
argv、exit0、日志与analyzer hash；不依赖本文四舍五入表格做原始计算。
`final-correctness-analysis.json`、`final-correctness-addendum.json`、
`final-micro-analysis.json`分别保留生命周期、旧FAIL及修订PASS、micro全部样本。
`audit.json`和`completion-manifest.json`给出最终源码、提交、资源及用户改动审计。

本轮故障注入首轮FAIL、缺参数命令FAIL、DP旧harness超时FAIL和优化前native中断
PARTIAL原样保留，不计入正式性能聚合。历史CB1 `0/24`、60次抢占、旧DP4 PARTIAL
及其他FAIL/PARTIAL继续保留。没有重跑旧矩阵或扩大85条质量样本，不声称有损
质量保持；fixed预算与requests路由继续默认。收尾及提交见
[本轮closeout](engineering-protected-closeout.md)。
