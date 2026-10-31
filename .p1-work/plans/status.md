# P1 v8 执行状态

授权入口是 `.p1-work/prompts/prompt1.md` 与本次用户 Long-Running Goal，覆盖原reference的review-only身份。
不升级核心环境、不修改学习基线、不移植旧Runtime、不创建上游PR。

| 阶段 | 状态 | 证据与限制 |
| --- | --- | --- |
| M0 | PASS | Native/trace identity、真实 allocator miss/free、CPU oracle、完整本地数据审计完成；28 tests passed |
| M1 | PASS | Whole/token live reuse、重复回收、FP32 Attention、EOS/abort/maintenance/preempt/q0/fail-stop、42 tests、优化及NCU/NSYS完成；见reports/m1.md |
| M2 | PASS | 压力控制16tests/5条8B轨迹；Tri校准/3评分tests/重复reuse及NCU优化；四策略85样本质量评测；五策略在6 GiB与1.5 GiB下的真实HTTP Serving对照均完成，详见reports/m2-pressure.md、reports/m2-scoring.md、reports/m2-evaluation.md |
| Prefix | PASS | 完整页APC正确性、23项针对性tests及prefix全92项通过；12组HTTP对照/两条诊断完成：8 GiB noop开销0.68%满足预算，6 GiB Sink延迟/Goodput预算PASS，3/1.5 GiB因TTFT超预算性能FAIL；96个Native/noop输出相同；见reports/prefix.md、reports/prefix-serving.md |
| Graph | PASS | 7条8B轨迹覆盖whole/token、FULL/PIECEWISE与Prefix；17组HTTP、84个native/noop输出相同；8 GiB noop开销0.63%、1.5 GiB有损SLO通过；旧profile缺1个span保留PARTIAL，收尾新profile274个graph spans全匹配；见reports/graph.md、graph-serving.md、final-review.md |
| TP | PASS（正确性） | 11条原8B轨迹及收尾组合/q0/maintenance慢rank回归PASS；FULL noop开销≤5%；short/eager历史14.33%、改后ABBA合并6.871%，仍超5%预算，压力SLO仍FAIL；新profile424个graph spans及162个collective spans完整；见reports/tp.md、tp-serving.md、final-review.md |
| DP | PASS（正确性） | 两条8B双Engine局部压力/abort轨迹PASS；8页由C全层重写，peer128输出相同、abort后112次验证forward；10组正式HTTP/5条诊断完成；48个native/noop输出相同，两档noop开销≤5%；native压力三种有损策略24/24达SLO，skewed Sink仅18/24，性能FAIL；见reports/dp.md、dp-serving.md |

必要组合：Prefix+Graph、Prefix+TP、Graph+TP、Prefix+Graph+TP smoke；不以独立Gate推断组合通过。
所有模型实验统一Qwen3-8B BF16；单卡只用物理GPU1，TP2/DP2才用0,1，项目显存合计≤45GiB。
大产物在 `/data_ssd/qcrs/experiments/p1-v8`。保留负结果，状态只在实际对应验证后更新。

完整M0→M1→M2→Prefix→Graph→TP→DP实施与验证已执行到末阶段，
交付入口为`../reports/delivery.md`。各阶段PASS只适用于对应报告的实现/正确性
范围；Serving按配置分别判定。M2/Prefix部分压力SLO、TP short/eager开销与压力
SLO、DP skewed Sink SLO仍为FAIL；Graph/TP旧profile仍PARTIAL，新代表性补采完整。
有损质量未证明保持，按用户最新要求不再扩大质量评测；不将独立Gate推断为全部
组合或所有配置性能通过。功能缺陷、过度校验和Triton review详见
`../reports/final-review.md`，包含本轮支持范围修复和普通decode热路径精简。
