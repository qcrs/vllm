# Controller 重试容量修正：实现、验证与停止

2026-11-01，本轮承接用户的尾页诊断，基线 `8cc5e15891`。环境、输入、GPU0/1、
CPU20/21、资源与 Git 规则继续使用 prompt3；初始用户 patch 和源码保存在
[Refinement B0](../experiments/Engineering-Refinement-B0/20261101-134008-initial)。
本报告补充[原 CB1](engineering-next-controller.md)，不覆盖 demand 的历史负结果。

## 实现与源码路径

`ReclaimAllocation` / `ReclaimRetry` 保存同一调度状态的一次同步重试预测。
`Scheduler._reclaim_retry` 沿真实 running 顺序、token budget 和 long-prefill 阈值，
使用 post-forward E/q 与原 `get_num_blocks_to_allocate` 计算页需求；waiting requester
复用原 APC lookup，不再查询缓存、不修改 Pool。尚未到达的请求不进入预测。

显式 `budget_mode=retry` 覆盖截至 requester 的分配；`retry_pass` 还优先覆盖同轮
后方已知 running 分配，完整 pass 不可行则退回 requester 前缀。两者均实验性，
`fixed` 默认保持。对 victim 的净容量贡献为：

```text
release(K) - victim_growth(K) = N - ceil((K + q_next) / B)
free + N - ceil((K + q_next) / B) >= other_required + soft_reserve
```

`retry_reclaim_candidates` 在该约束、至少释放一页与原 K_floor 中求最大合法 K；
软 reserve 不可满足则退回硬目标，不能突破 retention/min_keep/保护区。
尾槽通过 K 决定，不写死 K-1，不无条件额外预留一页。不为零拷贝降低最大 K。
不同 victim 仍沿既有 coverage/zero-copy/bytes 排序，不比较绝对 K 推断质量。

执行继续复用 `ReclaimRequest`→Worker plan/compact→全 rank completion→实际等待→
Scheduler canonical commit/free；未改 KV payload、版本 pin、source row/E 或 hash
边界。预测不是 reservation：EOS、abort、后续 admission 或合法 floor 可能使整轮
容量仍不足，一次 deferred retry 后继续原生分支。

## 正确性与诊断

- CPU：48 passed / 200 deselected，见 B0 的 `cb-full-pass-cpu-first.log`。
  独立枚举合法 K 验证最大性；真实 Pool 覆盖 victim 先 decode、running/waiting
  requester、requester 自救、q=0/1、完整 pass 软目标与 floor 退回。
- 真实 Qwen3-8B TP2：[q0](../experiments/Engineering-Refinement-CB-Check/20261101-134852-retry-tail-q0)
  / [q1](../experiments/Engineering-Refinement-CB-Check/20261101-134935-retry-tail-q1)
  验证新增尾槽 K 的全36层 retained payload、继续生成、free/rewrite/reuse；两次
  guard PASS，项目 NVML 峰值分别18,123,587,584 / 18,144,559,104 bytes。
- [retry 诊断](../experiments/Engineering-Refinement-CB-Diagnostic/20261101-135027-retry-sink)：
  `candidates-complete.json` 中206/206预测与首次 retry 成功，但 timed46次抢占、
  0/8 SLO、24.14 tok/s。前方 decode 自救成功后，后方 prefill 仍可触发抢占。
- [retry_pass 诊断](../experiments/Engineering-Refinement-CB-Diagnostic/20261101-140148-retry-pass-sink)：
  148次决策，144次预测 requester 覆盖且首次 retry 成功，108次预测完整 pass 覆盖；
  4次硬缺口无单候选覆盖。所有选中候选预测 release 与实际 commit 一致。
  timed20次抢占、0/8 SLO、35.80 tok/s。该轨迹有 host hooks/trace，不能当正式性能。

## 正式有限 A/B

TP2 SinkRecent、FULL、0.75GiB/rank、batch1024、seqs8、CPU20/21，完整负载预热后
各3轮×8请求，trace/profiler off，只改变预算；未重跑旧矩阵。
[fixed A](../experiments/Engineering-Refinement-CB-Formal/20261101-140554-fixed-sink)与
[retry_pass B](../experiments/Engineering-Refinement-CB-Formal/20261101-140733-retry-pass-sink)
保存实际命令、源码、输入、所有请求 token IDs、日志、NVML、affinity与结果；
逐请求复算为同目录根下 `summary.json`。

| timed 指标 | fixed | retry_pass |
| --- | --- | --- |
| 三轮 output tok/s | 101.290 / 101.339 / 101.398 | 36.070 / 36.059 / 36.111 |
| SLO | 18/24 | 0/24 |
| 抢占 | 0 | 60，每轮20 |
| TTFT max | 2.800s | 21.148s |
| 平均 TPOT max（逐请求） | 32.896ms | 109.058ms |
| TTFT / TPOT 超标请求数 | 6 / 0 | 15 / 18，存在重叠 |

两次 guard PASS、NVML合计峰值均19,931,332,608 bytes。新预算比旧 demand 的468次
抢占明显减少，但仍远差于固定预算；不能把这种跨运行描述成稳定 Serving 收益。
本轮未改 scorer，不重复旧 micro；新增预算引起输出变化不要求与 fixed 相同，
85条质量样本冻结，有损质量仍未证明。

## 停止判断

尾页竞争和单轮需求预测已实现并验证；Serving 仍 FAIL，`fixed` 继续默认。
当前最大 K 仍容易形成少量释放→再次耗尽→再次处理较大 retained KV 的循环。
有限一轮前瞻不足以摊销整个请求剩余生命周期；继续扩大前瞻/扫 reserve 会混入
未建模未来进度和质量预算，超出本轮合理工程投入，停止继续调参，不补反向寻找收益。

CB2 重新审计后，确有4次无单候选覆盖D（warmup/timed各2次）且随后原生抢占，
原“无进入证据”不适用于新轨迹。但它只影响148次动作中的4次，部分回收还可能
降低后续重算；没有净收益依据支持再扩一个门控。CB2 本轮 STOPPED / NOT_RUN，
理由是收益优先级与有限投入，不谎称仍无触发事件。原历史结论保留日期和作用域。

简历可讲“基于物理页边界与真实调度顺序构建有界重试容量模型，并用 Pool、TP2
生命周期和有限 A/B 识别其摊销边界”；不可写“动态预算提升 Serving 吞吐/质量”。
