# Protected KV 回收：稳定版结果

本页汇总当前稳定版的 Controller 入口、已有 correctness、micro 与 Serving 证据。
Serving 数字来自[2026-11-08 GPU1/2 评估](engineering-retest-evaluation.md)，
本次整理没有重跑 GPU 或扩大质量样本。

## Controller

不配置 `kv_reclamation` 时使用 vLLM native KV 管理；配置后只启用 protected Controller。
`budget_mode`、demand/retry 候选选择等旧 Controller 分支和实验入口已移除。
`policy` 仍用于选择 KV 保留/压缩方式，包括 `full`、`sink_recent`、`random` 和
`triattention`，不再选择 Controller。

Protected 每轮删除不超过 victim 私有物理 KV 的 20%，并至少保留已计算私有逻辑上下文的
50%；一次最多两个 victim。配置两轮时，第二轮须等待真实 decode 完成后重新规划。项目
SLO 保持逐请求 TTFT≤2s 且 TPOT≤80ms。

## Serving

同一小 KV 池下，native 与 protected 的已有正式 baseline 结果如下。每组24个请求，模型为
Qwen3-8B BF16；GPU1 单卡与 GPU1/2 TP2 使用相同逻辑容量，每池 10912 tokens。输入为
4096、8192、128、128 tokens 的混合请求，按固定到达间隔发送。

| 部署 | Native 完成速率 | Protected 完成速率 | 联合 SLO | Native 抢占 / 重算 token | Protected 抢占 / 重算 token |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPU1 单卡 | 34.291 tok/s | 120.746 tok/s | 7/24 → 24/24 | 386 / 692,697 | 0 / 0 |
| GPU1/2 TP2 | 25.286 tok/s | 123.078 tok/s | 0/24 → 24/24 | 399 / 713,597 | 0 / 0 |

这些是指定小池和到达负载下的完成速率，不是整卡饱和吞吐或通用加速承诺。KV 回收会改变
请求可见的上下文；当前没有 protected 的模型质量评估，因此不能声称质量无损。单独的高突发
测试中，长请求仍有 TTFT 超过 2s 的情况，容量保护不保证长请求立即入场。

## Correctness 与成本

已有回归记录为 CPU 92 项、DP 本地 fixture 13 项通过，适用 lint 和 mypy 检查通过。真实
TP2 测试覆盖双 victim、延迟 completion、全 rank 同步、Graph/eager 路径和释放页改写复用；
DP2 测试覆盖等待者进展、abort 与 peer continuation。Correctness PASS 只说明被测执行和
页面生命周期符合约定，不代表性能或模型质量。

已有 TP2 micro 中，第二轮释放 41 页且复制量为 0；Worker 中位数 0.341504ms、compaction
中位数 0.014336ms，Controller CPU 决策中位数 111.3685µs。该 micro 不含正常模型 forward
和两轮间的 decode，不能单独证明端到端加速。

DP 的 `known_work` 路由表现属于路由结果，不能归因于 protected Controller；现有数据没有
建立独立、稳定的 DP 回收性能收益。85 条质量样本维持原范围，历史 FAIL/PARTIAL 结果仍在
各自归档报告中。

完整条件、分组结果、资源记录与失败审计见[统一评估](engineering-retest-evaluation.md)
和[收尾报告](engineering-retest-closeout.md)。
