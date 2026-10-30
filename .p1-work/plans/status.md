# P1 v8 执行状态

授权入口是 `.p1-work/prompts/prompt1.md` 与本次用户 Long-Running Goal，覆盖原reference的review-only身份。
不升级核心环境、不修改学习基线、不移植旧Runtime、不创建上游PR。

| 阶段 | 状态 | 尚需证据 |
| --- | --- | --- |
| M0 | PASS | Native/trace identity、真实 allocator miss/free、CPU oracle、完整本地数据审计完成；28 tests passed |
| M1 | PASS | Whole/token live reuse、重复回收、FP32 Attention、EOS/abort/maintenance/preempt/q0/fail-stop、42 tests、优化及NCU/NSYS完成；见reports/m1.md |
| M2 | NOT_RUN | 真实miss/candidate/headroom/一次retry/native fallback、Tri shared/calibration/score reference、质量与Serving |
| Prefix | NOT_RUN | shared保护、所有suffix aliases撤销、lossy后停止发布、真实APC命中与reuse |
| Graph | NOT_RUN | FULL/PIECEWISE真实replay、静态地址、E/row/slot刷新、recapture计数 |
| TP | NOT_RUN | 全局distinct KV/GQA评分/统一plan、all-rank completion、延迟rank/reuse |
| DP | NOT_RUN | native dense双replica/pool/affinity，局部压力隔离、aggregate/per-replica指标 |

必要组合：Prefix+Graph、Prefix+TP、Graph+TP、Prefix+Graph+TP smoke；不以独立Gate推断组合通过。
所有模型实验统一Qwen3-8B BF16；单卡只用物理GPU1，TP2/DP2才用0,1，项目显存合计≤45GiB。
大产物在 `/data_ssd/qcrs/experiments/p1-v8`。保留负结果，状态只在实际对应验证后更新。
