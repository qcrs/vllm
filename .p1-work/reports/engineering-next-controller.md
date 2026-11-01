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

CB1 和条件 CB2 的实现、正式对照及结论在后续增量追加。
