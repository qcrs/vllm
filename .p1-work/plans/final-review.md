# P1 v8 历史功能收尾计划

本计划对应功能审查提交 `4ad1a03e1c`，已执行完成。范围为实现完整性、功能缺陷、
冗余检查、Triton落地及TP/DP验证；未扩大质量样本、放宽SLO或创建上游PR。

## 结果归属

[final-review报告](../reports/final-review.md) 保存当时的finding、154项测试、
真实GPU轨迹、profile、命令、性能FAIL和离线资源限制。该报告是历史结果来源，
本plan不再复制数表或维护后续状态。

后续CPU placement与pressure配置实验见
[performance-followup](../reports/performance-followup.md)。独立Review在
`252c747ad6` 发现的R1及本次限定修复另见
[独立Review基准](../reports/independent-closeout-review.md) 和
[closeout-fixes](../reports/closeout-fixes.md)。
当前支持契约与导航由 [delivery](../reports/delivery.md) 维护，未决项由
[status](status.md) 维护。历史功能PASS不等于所有性能预算通过，也不覆盖保留KV的
streaming session重入；feature-on现明确拒绝该生命周期。
