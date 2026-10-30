# P1 v8 执行状态

授权入口是 `.p1-work/prompts/prompt1.md` 与本次用户 Long-Running Goal，覆盖原reference的review-only身份。
不升级核心环境、不修改学习基线、不移植旧Runtime、不创建上游PR。

| 阶段 | 状态 | 尚需证据 |
| --- | --- | --- |
| M0 | PASS | Native/trace identity、真实 allocator miss/free、CPU oracle、完整本地数据审计完成；28 tests passed |
| M1 | PASS | Whole/token live reuse、重复回收、FP32 Attention、EOS/abort/maintenance/preempt/q0/fail-stop、42 tests、优化及NCU/NSYS完成；见reports/m1.md |
| M2 | PASS | 压力控制16tests/5条8B轨迹；Tri校准/3评分tests/重复reuse及NCU优化；四策略85样本质量评测；五策略在6 GiB与1.5 GiB下的真实HTTP Serving对照均完成，详见reports/m2-pressure.md、reports/m2-scoring.md、reports/m2-evaluation.md |
| Prefix | PASS | 完整页APC正确性：shared/gap/P缩小、全部aliases撤销、lossy发布guard、全36层payload、三策略重复reuse、真实miss与EOS/abort/preempt通过；23项针对性tests、prefix全92项通过；诊断成本见reports/prefix.md，新增Serving性能收益仍TO_VERIFY |
| Graph | NOT_RUN | FULL/PIECEWISE真实replay、静态地址、E/row/slot刷新、recapture计数 |
| TP | NOT_RUN | 全局distinct KV/GQA评分/统一plan、all-rank completion、延迟rank/reuse |
| DP | NOT_RUN | native dense双replica/pool/affinity，局部压力隔离、aggregate/per-replica指标 |

必要组合：Prefix+Graph、Prefix+TP、Graph+TP、Prefix+Graph+TP smoke；不以独立Gate推断组合通过。
所有模型实验统一Qwen3-8B BF16；单卡只用物理GPU1，TP2/DP2才用0,1，项目显存合计≤45GiB。
大产物在 `/data_ssd/qcrs/experiments/p1-v8`。保留负结果，状态只在实际对应验证后更新。

当前阶段边界：完成并提交 Prefix 后暂停推进，未经阶段转换授权不进入 Graph；
Goal tracking 保持 active。Prefix PASS仅指报告限定的正确性，不将独立Gate推断为组合或Serving性能通过。
