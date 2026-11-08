# Decode 进展驱动的两轮 protected 回收

2026-11-08。实现提交 `2439253387`，GPU harness 提交 `380b5480ea`。
本轮接在已有 bounded backfill 后，修复“一轮不能满足缺口便放弃所有回收路径”的
限制。只对已完成 prefill 的 decode 请求回收；不在 prefill 中删除 KV。
结果见[统一评估](engineering-two-round-evaluation.md)，交付范围见
[计划](../plans/engineering-two-round-plan.md)。

## 预算与启用

`ReclamationConfig.decode_reclaim_rounds` 默认 1，只接受 1/2；2 要求显式
`pressure=True, budget_mode="protected"`。fixed 及旧单轮策略仍可复现。
示例实验配置：

```python
ReclamationConfig(
    pressure=True,
    budget_mode="protected",
    decode_reclaim_rounds=2,
    admission_bypass_limit=2,
)
```

每次私有物理 KV 删除不超过 20%，私有物理 KV 相对已计算逻辑历史至少保留 50%，
共享 prefix 不删除，每轮最多两个 victim。这些是容量/删除约束，尚不是质量保证。
输出预算最多再生成两个 token 就必定完成的候选优先自然释放，不预测实际 EOS。

`protected_reclaim_plan()` 先尝试单轮。不能覆盖完整已知 prefill 的净缺口时，
`protected_two_round_plan()` 才枚举一个或两个 victim 的可行路径。预测包含两轮
之间的两个 decode token、其他 running 的页增长、最终 admission retry 及其下一
token 预留；同时确认首轮释放后已有请求能继续执行。单轮/两轮均不可行时不修改页表，
继续已有 bounded backfill 或等待。

88 页案例的 micro 算术：E=L=4097，单轮最大合法释放 52 页不足；第一轮保留
3278 token，两个 decode 后 E=3280/L=4099，第二轮可再释放 41 页。这里的长度
用于预测可行性，不能直接作为第二轮执行页表。micro 的两轮不包含模型 forward，
真实 decode 语义另由模型 correctness 证明。

## Scheduler 状态与执行边界

Scheduler 仅持有一个 `DecodeReclaimWindow`：受益等待者 ID、首轮 victim 的逻辑
完成水位、已尝试轮数。启动窗口要求 FCFS、等待者尚未 admission、所有 running
已完成 prefill。窗口内已有 decode 正常执行，暂停后方新 admission，防止释放页
被新请求消费。单轮可行路径仍沿原实现，不创建两轮窗口。

```text
真实 admission miss
  → 单轮不足、两轮可行
  → 第一次 reclaim + 全 rank completion + canonical commit/free
  → victim 实际完成 decode，逻辑水位前进且无 in-flight token
  → 按真实 Pool/E/L 重算第二轮
  → 第二次 completion/commit
  → 原等待者实际 admission，清窗口
```

只调用 `schedule()` 增加乐观 L 不能解锁第二轮。第二轮仅使用首轮选定的 victim
集合，按当时真实缺口重算；可能只需其中一个。第二轮没有可行计划或已用尽两轮，
则等待自然释放，不为该窗口发第三轮。受益者 abort/移除或成功 admission 清窗口。

“最多两轮”针对同一等待者的一次 admission 窗口。后续请求的 admission 或已有
decode 的新压力可以发生新的合法回收，仍受单次 20% 和逻辑累计 50% 底线限制，
不能解释为每个 victim 生命周期仅回收两次。

Worker、compactor、TP 通信和 Scheduler commit/free 协议保持原契约：所有 TP rank
完成操作后才提交 canonical 页表、释放 suffix；Graph 继续读取更新后的页表。
本轮没有新增 DP 路由算法；DP 各副本独立持有窗口和 Pool，没有跨池迁移。

## 行为验证与边界

现有 planner/Scheduler suite 覆盖可行/不可行缺口、至多双 victim、peer 增长、
真实完成与乐观 schedule 的区别、后方 admission 阻挡、第三轮抑制、abort 清理、
近完成自然释放，以及原 protected/backfill 语义。CPU 118 项、DP 本地 fixture
15 项通过。测试合同和最便宜验证层级的四问见计划，不修改用户已有测试笔记。

Qwen3-8B BF16 单卡真实两轮，以及 TP2 单/双 victim、延迟 completion、FULL
Graph、旧页面改写后真实复用均通过。DP2 受益等待者在 abort 前实际入场并复用本地
释放页；peer 未回收，abort 后继续完成 forward。首次两次单卡 fixture 失败保留，
不能将所有 correctness 启动都写成 PASS。

性能边界也明确：两轮能扩大可准入容量，但会增加一次回收和至少一次真实 decode
等待；长 prefill 更早入场也会占用计算时间、延迟后方短请求。容量可行不等于满足
2 秒 TTFT，更不等于超过删除更多 KV 的 fixed。正式 Serving 不开 trace，另做
一次独立诊断解释状态；具体结果及停止判断以统一评估为准。
