# 下一阶段 Controller：候选证据与缺口预算

## B0 / CB0

本轮从 `04a89d6037` 开始，分支与环境保持 prompt3 约束。初始用户对 status/delivery
的修改单独备份，未纳入本阶段提交。B0 的即时源码、环境、输入 hash 和用户 patch 在
[初始归档](../experiments/Engineering-Next-B0/20261101-093618-initial)。

Scheduler 一次扫描产生合法候选快照，记录 E/L/q/P/row、预算、复制字节和预计释放，
原分配器保存 required/available；trace 增加 pid/sequence。正式 trace-off 不写候选
JSON、不增加设备同步。离线 `analyze_tp_ttft.py --candidate-output` 关联 commit、
首次 retry 和后续原生动作，预测释放与实际 commit 必须一致。

[CB0 低扰动诊断](../experiments/Engineering-Next-Controller-Diagnostic/20261101-093841-cb0-all-candidates)
使用 TP2 SinkRecent、FULL、0.75GiB/rank、CPU20/21、完整预热及一个 timed round，
保留 host timeline，未启用 NSYS。45 次决策全部匹配 commit，43 次首次 retry 成功，
2 次无单候选覆盖硬缺口；两例均在 warmup，其中 running 一例后续抢占，waiting
一例继续等待后 admission。timed 为 5/8 SLO，warmup 3/8，不能视为正式性能对照。
诊断仍有额外 host/JSON 工作，不能由 select→commit inclusive 时间拟合回收增量成本。
项目 NVML 合计峰值 19,914,555,392 bytes，guard PASS。

CPU 受影响回归为 25 passed / 200 deselected，日志在 B0 `cb0-cpu.log`。
旧 C0 的 29/24/5、时间 PARTIAL 与旧 Serving FAIL 均保持原义。

CB1 和条件 CB2 的实现、正式对照及结论见下文。

## CB1：最大合法保留预算

新增显式 `budget_mode=demand`，`fixed` 保持默认。复用同次候选快照，以固定
retention 结果为 K_floor，对 G、D 依次求 `min(E-1,B*(N-T))`；无覆盖保留原候选。
只在最大合法 K 检查 exact whole-block，不减少 K 换零拷贝。Scheduler、Worker
共享 target_keep，原 source row/E、ownership、completion 与 canonical free 不变。

恢复执行前另存[即时快照](../experiments/Engineering-Next-B0/20261101-100701-resumed-current)，
未重置 HEAD 或覆盖 dirty 文档。CPU 18 项 KV、11 项 pressure 测试通过，覆盖
partial tail、保护域、只满足 D、无覆盖、真实 Pool 释放及 q=0/1 的同源快照。
日志及[候选关联复算](../experiments/Engineering-Next-Controller/20261101-cb1-validation)
完整保留。真实 TP2 Qwen3-8B 的
[q0](../experiments/Engineering-Next-Controller/20261101-100750-cb1-demand-page-q0)和
[q1](../experiments/Engineering-Next-Controller/20261101-100842-cb1-demand-page-q1)
均通过 continuation、全层 retained payload、free/rewrite/reuse；每次仅回收一页，
不以 CPU 模拟代替设备完成。两次峰值约18.124GB，guard PASS。

### 正式固定预算 / demand 对照

TP2、SinkRecent、FULL、0.75GiB/rank、batch1024、seqs8、CPU20/21，
完整 warmup 后各3轮×8请求，仅改变预算。实际 argv、源码/输入 hash、token IDs、
请求级延迟、指标及资源在
[fixed](../experiments/Engineering-Next-Controller-Formal/20261101-100913-cb1-fixed)和
[demand](../experiments/Engineering-Next-Controller-Formal/20261101-101036-cb1-demand)。
`cb1-summary.json`保存全部轮次与逐请求复算。

| 结果 | fixed | demand |
| --- | --- | --- |
| 三轮 output tok/s | 101.052 / 101.140 / 101.130 | 11.124 / 11.134 / 11.127 |
| timed SLO | 18/24，FAIL | 0/24，FAIL |
| timed TTFT max | 2.821s | 84.830s |
| timed TPOT max | 33.035ms | 112.180ms |
| timed preemptions | 0 | 468（每轮156） |
| 三轮 goodput requests/s | .5921 / .5926 / .5926 | 0 / 0 / 0 |
| warmup SLO / 抢占 | 4/8 / 0 | 0/8 / 156 |
| NVML项目合计峰值 | 19,910,361,088 bytes | 相同 |

两次 guard PASS 只说明运行/资源通过，不能覆盖 Serving FAIL。24对输出 token IDs
完全相同，但这不是质量保证。Prometheus prompt counter 每轮均49152，按该指标
定义不包含全部重算，不能把它当实际 forward 工作量；正式重算细分 UNKNOWN。

[另一次 TriAttention demand shadow](../experiments/Engineering-Next-Controller-Diagnostic/20261101-095250-cb1-demand-shadow)
中320次 commit 的预测/实际 release 一致、共归还642页，仅8次首次 retry成功，
312次 allocation_miss。首次 D=24 已满足，但后续 decode 先消费1页，使 requester
再次少1页并进入抢占；之后的小额反复回收重现该问题。该诊断与正式 SinkRecent
不是同组。**覆盖当前 D 不等于后续 admission 成功**，删除更少也不等于更好性能。
本轮停止扩大 demand 实验，保持显式实验选项，不添加未经对照的 reserve 修补。
负收益稳定且巨大，不追加反向对照寻找正结果。

## CB2：STOPPED / NOT_RUN

新 demand 轨迹320次合法候选均可覆盖 D，未满足“重复无单候选覆盖 D，随后进入
原生动作”的进入门槛。312次 retry miss 与旧 CB0 的两例不能替代该门槛。
因此不实现 skip_insufficient，不发额外 intent、不改变 hash 撤销边界，继续 TP/PL/DP。
旧 C0 29/24/5 及时间 PARTIAL、旧所有性能 FAIL 均保留。
