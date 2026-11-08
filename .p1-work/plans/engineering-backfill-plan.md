# Protected Admission Backfill

2026-11-08，起点 `6b06d384b5`。本轮只使用 GPU1 单卡和 GPU1/2 TP2；GPU0 禁用。
复用现有 `.venv`、Qwen3-8B、`run_guard.py`、NVML 44/45 GiB 限制、CPU20/21、
离线缓存、85 条冻结质量样本与本地分阶段 Git 规则。B0 位于
`Engineering-Backfill-B0/20261108-091924-initial`。

## 问题与改动

此前 TP2 证据显示，protected 的完整 prefill 容量检查会让 8192-token 请求因合法
回收不足而留在队首。当前失败后立即 `break`，导致可入场的后方短请求也不调度。
这轮不放宽完整 admission，不做 prefill 中回收。只在显式启用时允许 FCFS 队首因
容量不足且无合法 reclaim plan 而暂时让位；最多让有限数量的后方请求成功入场。
额度耗尽后暂停新 admission，已有运行请求释放容量时重新尝试队首。

上游参考：vLLM PR #33499 展示了同一 KV 队头阻塞与有界绕过；PR 讨论也记录仅重试
队首并不能阻止已入场请求持续占用 KV。故本轮保留小的绕过上限和显式 opt-in，
不宣称严格墙钟公平或对无限输出请求的无饥饿保证。

DP known-work 路由此前只在混长短 W1 有收益，等长 W0 不稳定；usage tie-break 仍
缺进入证据，本轮不新增路由策略。补齐 backfill 与 DP prefill capacity guard 的交互，
避免跳过唯一队首后错误清除容量满状态。物理回收保护比例、候选与 TP completion 不变。

## 验证与测量

- 用现有 Scheduler/真实 BlockPool 验证无合法回收时后方短请求可入场、额度按成功
  admission 计算、耗尽后新请求停止、队首容量恢复后先入场。
- 对 fixed、protected strict 和 protected bounded-backfill 使用同一新混合到达负载；
  在单 GPU1 与 TP2 GPU1/2 各测一次部署形态，单卡/TP2 KV 池按逻辑页数匹配。
- 汇报正确性、micro/决策开销和 serving 分开；旧 long 负结果、历史 FAIL/PARTIAL、
  85 条质量样本与 SLO TTFT≤2s 且 TPOT≤80ms 全部保留。
- 不重跑旧矩阵，不扩质量样本，不 push 或创建 PR；阶段提交只含本轮文件。

## 完成与停止结论

实现提交 `0ca2e5c80d`。结果见[统一评估](../reports/engineering-backfill-evaluation.md)
和[收尾](../reports/engineering-backfill-closeout.md)。

| 阶段 | 结果 |
| --- | --- |
| 定位与上游参考 | 完成；容量不足叠加 FCFS 队头阻塞，等效容量单卡复现 TP2 主退化 |
| bounded backfill | 完成；仅 pressure/protected + FCFS opt-in，成功入场最多 2 次，默认 0 |
| Correctness | 本地 Qwen 配置 fixture 107 passed；新增 DP guard 回归、lint/type checks 通过 |
| 单卡与 TP2 Serving | 六次 guard PASS，各配置三轮；两种部署 SLO 均 6/24→18/24，fixed 24/24 |
| Micro | 本轮 NOT_RUN；不改 Worker/kernel，不宣称 micro 或总吞吐加速 |
| 长请求/吞吐 | 未解决；8192-token 仍 0/6，backfill 吞吐低于 fixed，停止扩大参数搜索 |
| DP usage tie-break | STOPPED / NOT_RUN；进入证据仍不满足 |
| prefill 中删除 KV | 按用户要求暂缓，NOT_RUN |

新 mixed 是定位长短请求队头阻塞的合成 Serving 负载，未代替冻结质量样本，未重跑
旧矩阵。最后 abort 清理/DP guard 修正及格式化晚于 GPU 测量，差异在评估中单列。
