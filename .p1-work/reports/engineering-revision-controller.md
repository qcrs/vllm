# CB1 重算放大审计与 prefill 阶段预算修正

2026-11-01，承接用户对“首次重试成功但 Serving 明显退化”的复核要求。
实现提交 `9af8339c25`，起点 `e0fe252fd6`；环境、输入、CPU20/21、GPU0,1、
合计显存44/45GiB与本地 Git 规则沿用 prompt3。fixed 继续默认。
本轮修正缓解了实验模式的重算放大，**仍未取得相对 fixed 的 Serving 收益**。

## 问题：局部可行不等于后续工作可承受

页数计算与实际 commit release 一致，未发现新的 ownership/completion 错误。
旧 `retry_pass` 优先覆盖当前已知的一轮分配；整轮不可行时退回 requester 前缀。
因此，requester 首次重试成功后，后方 prefill 仍可能沿原生路径抢占、重算，
再挤压容量。首次重试成功率不是端到端有效性的充分指标。

本轮只读复算[旧 SinkRecent 诊断](../experiments/Engineering-Refinement-CB-Diagnostic/20261101-140148-retry-pass-sink)，
按 `timeline-scheduler.jsonl` 的 `logical_start/q/prompt_tokens` 统计实际执行工作。
一个 timed round 的49,152个输入 prompt tokens 实际执行180,804个，额外重算131,652个。
74次回收中72次预测 requester 覆盖、72次首次重试成功，却有20次抢占。
全诊断40次抢占的最近回收均未覆盖完整 pass，其中36次 requester 首次重试成功。
这支持“局部预算目标过短导致容量压力反复出现”的机制解释，不是释放页算少了。

原始 `vllm:prompt_tokens_total` 统计输入，不能拿它证明没有重算。
可复算结果见[B0](../experiments/Engineering-Revision-B0/20261101-151241-cb-tp-audit)
的 `old-cb-work-audit.json` / `analyze-cb-revision.py`；后者保存输入与自身 SHA。

## 本轮有限修正

新增显式 `budget_mode=prefill_fixed`：

```text
若存在 unfinished 请求，且扣除本次 scheduled/APC 进度后，已知未计算历史 > 1 token：
    用原 fixed retention 批量释放
否则：
    用已有尾页感知 retry_pass 最大合法 K
```

`Scheduler._select_reclaim_candidate` 读取 `self.requests`，覆盖 running、waiting
及 preempt 后的 output-history replay。不能只读 `is_prefill_chunk`，因为尚未入场
的请求初始也是 False。已调度完的最后一个 prefill chunk 和已结束请求不算积压。
此判定只使用已知工作，不预测未来到达或未知输出，不引入新前瞻长度/reserve。

fixed/retry_pass 均沿用请求 pin 的 retention/min_keep/保护区；没有降低保留下限，
也没有改变 Worker plan/compactor、全 rank completion、canonical commit/free/reuse。
`effective_budget_mode` 仅在显式 trace 中记录实际分支，离线分析使用同一分支。
调用链与状态解释见 [m2-pressure](../source-analysis/m2-pressure.md)。

## 正确性与低扰动诊断

- CPU：现有 Scheduler/真实 Pool 回归54 passed / 200 deselected，覆盖 decode、
  following/waiting prefill、last chunk、abort 与已知 output history 重算。
  初始 fixture 失败和后续通过分别保留在 B0 的 `cb-phase-cpu-first.log` / `cb-phase-cpu.log`。
- 受影响 pre-commit、manual mypy-3.12 通过；初次格式修正记录保留。
- [真实 Qwen3-8B TP2 诊断](../experiments/Engineering-Revision-CB-Diagnostic/20261101-153141-prefill-fixed-sink)：
  guard PASS，峰19,931,332,608 bytes；全部87次选中候选的预测 release 与实际 commit 一致，
  80次预测 requester 覆盖均首次重试成功；6次硬缺口无候选覆盖。

诊断使用 `profile_tp_ttft.py --nsys-child --trace`，只采 host hooks，未运行 NSYS。
以下各为一轮 timed，完整预热另计；两次独立诊断存在采集扰动，不当正式性能 A/B：

| 指标 | 旧 retry_pass | prefill_fixed |
| --- | ---: | ---: |
| 实际执行 prompt tokens | 180,804 | 56,279 |
| 额外 prompt 重算 | 131,652 | 7,127 |
| 抢占 | 20 | 1 |
| 回收次数 | 74 | 41，fixed 24 / retry_pass 17 |
| 预测 requester 覆盖 / 首次重试成功 | 72 / 72 | 38 / 38 |
| 预计复制字节 | 18,242,519,040 | 5,508,366,336 |
| SLO | 0/8 | 5/8 |

复制量是候选成本估计，非硬件带宽实测。新诊断仍有1次 timed 抢占，warmup另有3次。
该次 decision3825 已走 fixed：requester需要64页、可用20页，最大合法候选释放34页，
考虑下一轮增长后仍差11页，随后 r0-7 重算。不能声称该规则保证零抢占。
纯 decode 时保留更多 KV，也不会为尚未到达的下一波请求承诺容量。

本轮没有新增 K 算术或执行路径：K 来自已验证的 fixed/retry_pass；不重跑旧 payload
矩阵。原 q0/q1 全36层 retained payload、TP2 continuation/free/rewrite/reuse 证据仍见
[前轮正确性报告](engineering-refinement-controller.md)。本轮诊断验证新增分支的真实
allocation→intent→completion→commit/release→retry 控制链，不冒充新的逐层 payload 检查。

## 正式有限 A/B 与停止

固定8B BF16、FA2/NHD/B16、TP2、FULL_DECODE_ONLY[1,2,4,8]、0.75GiB/rank、
batch1024、seqs8、APC/async off、retention=.5、reserve0，完整负载预热后各3轮×8。
两边 Runtime 源码/hash 和实际输入相同，仅 budget_mode 不同；trace/profiler off。

| timed 指标 | [fixed](../experiments/Engineering-Revision-CB-Formal/20261101-153329-fixed-sink) | [prefill_fixed](../experiments/Engineering-Revision-CB-Formal/20261101-153708-prefill-fixed-sink) |
| --- | ---: | ---: |
| 三轮 output tok/s | 101.251 / 101.119 / 101.149 | 98.123 / 99.633 / 99.693 |
| 合并 output tok/s | 101.173 | 99.144（−2.005%） |
| SLO / goodput req/s | 18/24 / 0.593 | 15/24 / 0.484 |
| timed / warmup 抢占 | 0 / 0 | 0 / 1 |
| TTFT max | 2.824s | 3.008s |
| 逐请求平均 TPOT max | 32.985ms | 33.045ms |
| TTFT / TPOT 失败请求数 | 6 / 0 | 9 / 0 |

24条输出 token IDs 相同，两次 guard PASS、峰均19,931,332,608 bytes。
新增失败 r0-6/r1-6/r2-6 的 TTFT 为2.178/2.035/2.026s；全部失败仍由 TTFT 决定。
正式关闭逐步 trace，实际 prefill/复制量未测，不能把输入 counter 当成实际计算量；
也不能把2%差异全部归因于复制。保留更多 KV 增加后续 attention/重复回收成本是
合理机制假设，本轮没有正式细分证据量化各项贡献。

与[前轮正式 retry_pass](engineering-refinement-controller.md)的0/24、60次timed抢占
相比，本轮候选已接近 fixed 的时间量级，但这不是同一次启动配对的因果提速证明。
相对本轮 fixed 仍无收益，因此不补反向寻找正结果，不扩前瞻/参数搜索，不提升默认。
CB2沿前轮按收益优先级停止的决定，本轮不加 gate、多 victim 或新的调度策略。
85条质量样本冻结；输出一致不等于有损质量保证。历史 demand468次抢占与全部 FAIL
保留，未将诊断改善写成 Serving 达标。

复算命令（每个 shell 先按 prompt3 激活环境）：

```bash
.venv/bin/python .p1-work/experiments/Engineering-Revision-CB-Formal/analyze-formal.py
```

同目录 `summary.json` / `analysis.log` 保存逐请求 SLO、warmup/timed counters、
输入与源码一致性及 SHA。诊断复算命令与全部 GPU argv 见
[本轮收尾](engineering-revision-closeout.md)及各 run 的 `invocation.json`。
