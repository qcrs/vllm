# DP2：工作量路由复核与 usage 停止结论

2026-11-08，本轮使用物理 GPU1/2、Worker 主线程 CPU20/21；覆盖旧 prompt3/4 的
GPU0/1 放置，其他环境、资源与归档规则继承。本轮没有增加 DP 路由 Runtime，已有
`known_work` / pending ADD watermark 的语义保持；`requests` 仍默认。

## usage 诊断与条件阶段

现有信号链为 Scheduler/BlockPool → `DPLoadSnapshot` → coordinator → frontend
路由。`kv_cache_usage = 1 - free / (num_gpu_blocks - 1)`，free 包括可驱逐的缓存页，
所以它是池占用近似量，不能直接当成请求已知剩余工作量或精确可接纳请求数。
序列与 watermark 用来识别有效快照，Runtime 没有基于墙钟年龄的 expiry；250ms
只是本次 trace 的诊断筛选，未增加路由 contract。

证据目录为
`/data_ssd/qcrs/experiments/p1-v8/Engineering-Progress-DP-Diagnostic/20261108-000347-w1-known-work-usage`。
`pressure-analysis.json` 保存25次路由（含预热）、23次有效快照、3次 work score
同分，选中较高 usage 的同分为0次，随后 allocation miss 和压力动作也均为0。
原始 `dp-load-*.jsonl`、准确 argv 和 hash 均保留。

缺少 prompt4 要求的重复“有效新鲜同分 → 选高 usage → 本地 miss/抢占，同时 peer
仍有余量”证据。因此 usage tie-break **STOPPED / NOT_RUN**。不增加 usage 权重、
迁移、跨池借页或无依据的新鲜度超时；该停止不阻塞已有路由的正式复核。

## 正式协议与结果

最终正式目录为
`/data_ssd/qcrs/experiments/p1-v8/Engineering-Progress-Final-DP-Serving`。
Qwen3-8B BF16，每副本 TP1、独立4GiB KV pool，batch512、seq8，APC/async 关闭，
三轮×8请求，每请求输出128 token。W1 prompt 为 `[8192,512]*4`，W0 为 `[4096]*8`；
到达时间0–0.35s、间隔0.05s。单项比较均为 `budget_mode=fixed`，只改变 routing。
预热和计时区间分离。正式关闭高扰动逐步 trace，保留同口径 work counters。

“正向/反向”指实验先后顺序：正向 requests→known_work，反向 known_work→requests；
两方向各三轮，各24个计时请求。SLO 仍为每请求 TTFT≤2s 且 TPOT≤80ms。

| 负载/方向 | requests tok/s | known_work tok/s | 吞吐变化 | SLO requests→known_work | goodput req/s requests→known_work |
| --- | ---: | ---: | ---: | ---: | ---: |
| W1 正向 | 215.988 | 250.750 | +16.09% | 21/24→24/24 | 1.476482→1.958985 |
| W1 反向 | 211.802 | 257.392 | +21.52% | 21/24→24/24 | 1.447866→2.010875 |
| W0 正向 | 267.386 | 278.388 | +4.11% | 24/24→24/24 | 2.088955→2.174907 |
| W0 反向 | 278.814 | 276.349 | −0.88% | 24/24→24/24 | 2.178232→2.158977 |

W1 TTFT p99 正向为2.47268→1.70247s，反向为2.55803→1.67955s；TPOT p99约
26–27ms。W1 goodput 分别提高32.68%和38.89%。各方向对应24条 output token IDs
一致；所有单项的 miss/reclaim/preempt/recompute 均为0。

W1 的每副本计时 prefill token 总量解释了收益：requests 为30,720/73,728，known_work
正向为53,248/51,200、反向为52,736/51,712。只看请求数量会把长短请求的工作量混在
一起；已有 known_work 在此固定混长短负载中分配更均衡，减少长请求排队。收益来自
已有路由，**不能归因于本轮 Controller 或 usage tie-break**，也不是本轮首次发明。

W0 等长负载不支持稳定提速。requests 正向三轮为248.569/279.834/276.003tok/s，
存在波动；反向两组接近，变化符号反转。保留这组负控制，不挑正向+4.11%当作普遍
收益，不增加轮次。W1 仅有本机固定负载的有限重复证据，不声称统计显著或普遍改善。

## 组合与真实压力生命周期

组合 `20261108-024206-w1-known-work-protected-gpu12-combo` 为252.130tok/s、24/24，
goodput1.969767req/s；TTFT p50/p99为0.880571/1.692756s，TPOT p50/p99为
21.1967/27.0867ms。但两个池的 miss/reclaim/preempt/recompute 全为0，只能说明
组合可运行，不能说明 protected 提升 Serving，也不能据此验收真实压力回收。

真实压力能力来自
`Engineering-Amortized-Correctness/20261108-015435-dp2-progress-abort-reuse`：本地池
释放页由等待者 C 在 abort A 前实际改写复用，peer 不回收；abort 后 peer 又有112次
payload 核对的 forward，两个独立池最终均 free32。这是行为正确性证据，不是吞吐
证据。CPU route/watermark/engine-state 本地 fixture 13+2项 PASS，不称完整上游 suite。

## 命令与原始入口

下面均是已通过 `run_guard.py` 执行的子命令，不直接绕过 guard 运行；每次准确命令、
环境、源码与资源记录见最终 B0 的 `command-index.json`、各 run 的 `invocation.json`、
`protocol.json`、`runtime-source-hashes.json`、`requests-*.json`、`work.jsonl` 和
`worker-affinity-*.json`。

```bash
.venv/bin/python .p1-work/scripts/dp_serving.py --policy sink_recent --pool-gib 4 --scenario w1 --routing requests --repeats 3 --work-counters --max-num-batched-tokens 512 --max-num-seqs 8 --worker-cpus 20 21
.venv/bin/python .p1-work/scripts/dp_serving.py --policy sink_recent --pool-gib 4 --scenario w1 --routing known_work --repeats 3 --work-counters --max-num-batched-tokens 512 --max-num-seqs 8 --worker-cpus 20 21
.venv/bin/python .p1-work/scripts/dp_serving.py --policy sink_recent --pool-gib 4 --scenario w1 --routing known_work --budget-mode protected --repeats 3 --work-counters --max-num-batched-tokens 512 --max-num-seqs 8 --worker-cpus 20 21
```

W0 用同入口的 `--scenario w0`；反向复核顺序及全部目录见 command index。最终严格
复算为 `Engineering-Progress-Final-B0/20261108-021957-resume/final-unified-analysis.json`，
准确分析 argv 为同目录 `final-unified-command.json`。完整延迟分布、逐请求 SLO 和
每副本记录均在该 JSON；[统一评估](engineering-progress-evaluation.md)汇总。

历史 DP4 PARTIAL、前轮 W1/W0 FAIL 和旧报告完全保留。有限新结果不回写历史；
85条质量样本未扩大，token IDs 一致不能推广成有损回收质量保持。
