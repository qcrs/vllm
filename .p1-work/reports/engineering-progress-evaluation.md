# Protected Controller 进展修正：统一评估

2026-11-08，起点 `e3550ac916`。完成已知 prefill admission/净容量修正、同净页目标
优先整页重映射，以及仅在合法原计划内合并有限 completion headroom。**抢占与重算
风暴已修复，但最终 TP2 吞吐仍较同源码 fixed 低27.76%，未达到 TP 性能目标。**
DP 已有 known_work 在 W1 正反向分别观察到+16.09%/+21.52%吞吐、均24/24 SLO；
本轮无新增 DP Runtime，收益不归因于 Controller。usage tie-break 缺证据，停止实施。

## 协议、来源与比较边界

用户指定只用物理 GPU1/2，覆盖 prompt3/4 的旧 GPU0/1 配置，GPU0未使用。GPU
实验全部由既有 `run_guard.py` 共用锁串行；Worker 主线程 CPU20/21，其他 helper
线程不声称统一绑核。模型为现有 Qwen3-8B BF16，MRV2、FA2/NHD/B16、离线缓存；
未下载模型或安装升级依赖。项目显存合计44GiB停止、45GiB硬限；29次 guard 的
NVML 最高39.888672GiB，均未触线，实际项目进程仅观察到 GPU1/2。

大产物根为 `/data_ssd/qcrs/experiments/p1-v8`，下文目录相对该根。最终证据中心为
`Engineering-Progress-Final-B0/20261108-021957-resume`（下文“最终B0”）：

| 入口 | 内容 |
| --- | --- |
| `final-unified-analysis.json` | 最终 TP、DP、组合与 micro 的独立严格复算，含逐请求 SLO、分布、每副本工作量 |
| `final-unified-command.json` / `.log` | 完整分析 argv 与执行日志 |
| `capacity-analysis.json` | 独立 trace 的合法净容量与拒绝原因复算 |
| `command-index.json` | 本轮全部29次 guard 的准确子命令、环境与目录 |
| `resource-audit.json` | 每个 guard 的退出状态、NVML peak、实际物理 GPU |
| `audit-progress.py` / `audit.json` | 最终文件、29次运行、源码/input/artifact hash、Git保护复核 |
| `completion-manifest.json` | 最终本地提交、文件 hash、用户差异与审计入口 |

SLO 仍为每请求 **TTFT≤2s 且 TPOT≤80ms**，同一请求两项同时通过才计数。
goodput 沿既有“合格请求数/计时区间”口径；输出吞吐为输出 token/计时区间。三轮
×8请求，预热全部剔除；同一 stream chunk 中 token 共享客户端收到时间。小样本
p95/p99只描述本批数据，不声称统计显著性。guard PASS 仅代表运行和资源验证通过。

最终 Runtime 为 `533fa38ed5` 的实现；其后 `deb3d9e9d6` 仅更新分析/测量入口。
TP 两组同源码、同仪表，DP 各组相关 Runtime hash一致；TP/DP 不是同一种部署，
不互相做吞吐倍数比较。诊断 trace 与正式 Serving 分开。原 native 与旧矩阵不重跑，
历史报告不改写。85条质量样本冻结，保留比例与 payload PASS 均不是质量保证。

## Correctness：PASS

| 层次 | 最终证据与结果 |
| --- | --- |
| CPU | 扩展既有 KV/Pool/Scheduler suite，99 passed / 289 deselected；覆盖净尾页需求、完整 prefill、reservation、单/双 victim、页分担 oracle、有界合并及不可行无 mutation |
| DP CPU | 本地 fixture 13+2=15 passed，覆盖路由/watermark/engine state；非完整上游 suite |
| 静态检查 | 受影响 pre-commit 与 manual mypy-3.12 通过；Runtime与测量入口分别留日志 |
| TP2 单 victim | 两 rank 计划/执行一致、continuation、Graph replay、FP32 attention/Graph-eager reference、真实 release/rewrite/reuse，最终 free32 |
| TP2 双 victim | 首批两个 victim；rank1 第二 victim 延迟≥2.5s时 canonical不变，全 rank完成才ACK/commit/free；Graph及页改写复用通过，最终free32 |
| DP2 压力 | C 在 abort A 前输出并改写释放页；peer无回收、abort后112次payload验证forward，两个独立Pool最终均free32 |

CPU/静态日志及 `audit-correctness.py` / `correctness-analysis.json` 位于
`Engineering-Amortized-B0/20261108-014007-initial`；GPU最终目录分别为
`Engineering-Amortized-Correctness/20261108-015330-single-q0-tp2-graph-reuse`、
`20261108-015400-pair-tp2-graph-delay-reuse`、`20261108-015435-dp2-progress-abort-reuse`。
CPU日志为 `cpu-final.log`、`dp-cpu-final.log`、`dp-state-final.log`、`lint-final.log`、
`mypy312-final.log`；DP engine-state 使用已归档 `run-dp-state.py` 本地 tokenizer
fixture。测量入口 lint/type 日志另在最终 B0。

已运行 GPU 子命令如下；它们均由 guard 执行，不能直接绕过锁运行：

```bash
.venv/bin/python .p1-work/scripts/tp_reclamation.py --mode full --lifecycle q0 --protected --demand-pages 1 --worker-cpus 20 21
.venv/bin/python .p1-work/scripts/tp_reclamation.py --mode full --protected --victims 2 --delay --worker-cpus 20 21
.venv/bin/python .p1-work/scripts/dp_reclamation.py --policy sink_recent --budget-mode protected --routing known_work --worker-cpus 20 21
```

## Micro：小额复制修复成立，不等于 Serving 加速

最终 micro 为 `Engineering-Amortized-Micro/20261108-015655-protected-amortized-cost`，
复用固定 Runtime shapes，全36层、TP2。每个样本取两 rank 最大耗时再取中位数；
`worker_batch` 覆盖 Worker 回收至 completion，**不含模型 forward 或 Scheduler
commit/free**。compaction独立计时，不与Worker总量相加。

| E/L/P | 动作 | 释放页 | 估计copy bytes | Worker wall ms | compaction wall ms |
| --- | --- | ---: | ---: | ---: | ---: |
| 4100/4100/16 | fixed单victim | 128 | 301,105,152 | 0.90335 | 0.74368 |
| 同上 | protected单victim | 1 | 0 | 0.37323 | 0.04620 |
| 同上 | protected双victim | 52 | 0 | 0.67323 | 0.04737 |
| 8194/8194/16 | fixed单victim | 256 | 602,947,584 | 1.35811 | 1.19808 |
| 同上 | protected单victim | 1 | 0 | 0.38700 | 0.04854 |

这些动作释放量不同，不能声称同动作加速。E2051/L4100/P16的protected计划因逻辑
floor无合法空间，明确为 infeasible，未虚构Worker耗时。小额页回收现在能合法整页
重映射，解决了“为多保留不足一页而复制大段KV”的选择缺口；不能把本轮GPU1/2
数字直接与历史GPU0/1的约9ms数字算倍数。

E4100、八候选的决策CPU中位数：fixed scan406.48µs；protected单victim scan527.16µs、
完整forecast572.33µs，prefill reservation1.86µs；双victim scan4910.15µs、完整
forecast5001.57µs。双victim路径有真实CPU成本，但最终TP Serving最大victim为1，
不能把主要退化归因于双victim扫描。

隔离路由 micro：requests2.838µs、known_work4.061µs；正式含instrumentation的route
p50约25–27µs /36–38µs，二者是不同测量范围，不混用。

```bash
.venv/bin/python benchmarks/kernels/benchmark_protected_reclamation.py --runtime-shapes /data_ssd/qcrs/experiments/p1-v8/Engineering/20261031-232434-initial-state/tp-runtime-shapes-corrected.json --worker-cpus 20 21
```

上述为guard子命令，全部形状、device/host/wall与复制量见统一JSON。

## TP2 Serving：仍为负结果，瓶颈已收敛到容量与排队

最终正式目录 `Engineering-Amortized-TP2-Serving/`：
`20261108-015742-fixed-gpu12-forward` 对 `20261108-015940-protected-gpu12-forward`。
SinkRecent、retention0.5、reserve0、每rank0.75GiB、batch1024、seq8、maxlen9216；
APC/async关闭，FULL_DECODE_ONLY `[1,2,4,8]`，完整 workload 预热后正式三轮。
输入 `[4096,8192]*4`，到达 `[0,.3,.6,.9,5,5.3,5.6,5.9]`，每请求输出128。

| 指标 | fixed | protected |
| --- | ---: | ---: |
| 输出吞吐 tok/s | 101.547505 | 73.358786 |
| SLO合格请求 | 18/24 | 6/24 |
| goodput req/s | 0.595005 | 0.143279 |
| TTFT p50/p99 s | 1.725745 / 2.784649 | 3.615811 / 6.748967 |
| TPOT p50/p99 ms | 21.4173 / 32.7957 | 10.9956 / 15.4441 |
| 回收次数 / zero-copy次数 | 56 / 6 | 18 / 9 |
| 释放页 | 5902 | 954 |
| 估计copy bytes | 13,359,513,600 | 8,683,241,472 |
| allocation miss | 56 | 1551 |
| 抢占 / 重算token | 0 / 0 | 0 / 0 |
| 最大单批victim | 1 | 1 |

protected吞吐为fixed的72.24%，下降27.7591%。它的TPOT更低，但首token前等待更长，
使总体吞吐与SLO更差；不能单挑decode指标宣称整体提速。两边均未全部通过SLO。
miss包含重复拒绝检查，并非请求数，也不等同执行过1551次回收。

旧protected在本轮GPU1/2的历史参考
`Engineering-Progress-Baseline/20261107-231707-tp2-old-protected-gpu12` 为11.19195tok/s、
0/24、480次回收、468次抢占、1,694,043重算token，TTFT p50/p99为42.4583/84.4915s。
当前有明显恢复，但该旧运行 `reclamation_counts=false`、最终为true，且跨源码，只作
历史参考，不声称严格因果6.55倍。用户此前4.9tok/s还来自GPU0/1，也不作本轮A/B。

最终诊断目录
`Engineering-Progress-Final-Diagnostic/20261108-022023-tp2-protected-capacity-gpu12`
独立于正式数据：514次计划、508空计划，508次全因合法净容量不足。首例free424页，
8192-token请求需512页，缺88页；唯一4097-token候选floor3278，最多净释放52页，
仍差36页。其他running下一步需求为0，排除了该例decode抢页。长请求各被拒绝
127–128次，首次失败检查到allocation约1.40–1.60s；这不是总排队时间。

诊断六次回收中，三个小额动作各释放3页且zero-copy；三个大动作source8193、
floor6555、K6559、释放103页，zero-copy K6545低于floor。已修复小额复制，但保护
限制仍使请求无法及时并行入场。不能通过反复未推动进展的部分回收绕过单次cap。
回收变少但仍落后，按有界合并计划停止继续调参，不补反向、不扩矩阵。

```bash
.venv/bin/python .p1-work/scripts/serving_reclamation.py --policy sink_recent --pool-gib 0.75 --budget-mode fixed --repeats 3 --reclamation-counts --work-counters --graph-mode full --full-warmup --tp 2 --max-num-batched-tokens 1024 --max-num-seqs 8 --worker-cpus 20 21
.venv/bin/python .p1-work/scripts/serving_reclamation.py --policy sink_recent --pool-gib 0.75 --budget-mode protected --repeats 3 --reclamation-counts --work-counters --graph-mode full --full-warmup --tp 2 --max-num-batched-tokens 1024 --max-num-seqs 8 --worker-cpus 20 21
```

这些为guard子命令；诊断只增加`--trace`且一轮，不能将diagnostic时延当正式结果。

## DP2 Serving：已有路由在混长短负载中有有限收益

最终DP目录为 `Engineering-Progress-Final-DP-Serving/`，每副本4GiB、batch512、seq8，
W1 `[8192,512]*4`，W0 `[4096]*8`；到达0–0.35s、间隔0.05s，输出128，三轮。
单项Controller均fixed，正向requests→known_work，反向顺序相反。

| 负载/方向 | requests→known_work tok/s | 吞吐变化 | SLO | goodput req/s |
| --- | ---: | ---: | ---: | ---: |
| W1正向 | 215.988→250.750 | +16.09% | 21/24→24/24 | 1.476482→1.958985 |
| W1反向 | 211.802→257.392 | +21.52% | 21/24→24/24 | 1.447866→2.010875 |
| W0正向 | 267.386→278.388 | +4.11% | 24/24→24/24 | 2.088955→2.174907 |
| W0反向 | 278.814→276.349 | −0.88% | 24/24→24/24 | 2.178232→2.158977 |

W1 TTFT p99正向2.47268→1.70247s、反向2.55803→1.67955s，TPOT p99约26–27ms。
每副本prefill由requests的30,720/73,728改善为known_work正向53,248/51,200、反向
52,736/51,712；这是已有工作量路由改善分配与排队的有限证据。每组对应24条output
token IDs一致，miss/reclaim/preempt/recompute全0，不能归因本轮Controller。

W0反向无提升，正向requests首轮偏低；不声称稳定收益。组合
`20261108-024206-w1-known-work-protected-gpu12-combo` 为252.130tok/s、24/24，
goodput1.969767req/s，TTFT p50/p99为0.880571/1.692756s，TPOT p50/p99为
21.1967/27.0867ms；压力动作全0，不能证明protected的Serving收益。

usage诊断为25次路由、23有效快照、3同分、0次高usage同分错选、0 miss，进入条件
不成立，**STOPPED / NOT_RUN**。准确运行入口、目录和局限见[DP报告](engineering-progress-dp.md)。

## 失败保留与最终判断

29次guard中27 PASS、2 FAIL：fixed首次因GPU2背景占用、启动free8.58GiB低于
utilization0.4所需31.7GiB，未进入测量；pair首次延迟probe漏注入，修probe后验证通过。
没有终止背景任务，没有覆盖失败目录。CPU开发FAIL、离线tokenizer collection FAIL、
初版分析AssertionError和开发lint FAIL均保留，详见[收尾](engineering-progress-closeout.md)。

已修正能证明的进展预算与复制缺口；保留20%/50%/两victim和SLO。TP仍无Serving
收益，剩余容量不足有直接证据；DP W1有限正向证据来自已有路由，W0不支持稳定提速。
不将保留比例解释成质量收益，不以未验证收益为继续调参理由。fixed/requests继续默认，
历史FAIL/PARTIAL、旧矩阵和85条质量样本原样保留。
