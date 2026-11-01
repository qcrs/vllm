# 下一阶段策略绑定与安全更新

## PL0 / PL1

`RetentionPolicy` 的受控 factory 统一 SinkRecent、Random、TriAttention 的 score/plan
接口。公共 Worker 保持 SUM、TP root selection/broadcast、compactor、row/E 和完成协议；
策略不修改 Pool/hash。Random 仍仅 TP root 消耗 RNG，TriAttention 校准文件一次读取并
对实际读取的 bytes 求 SHA256，资源按版本持有，不按请求复制。

`EngineCore.add_request` 在处理 ADD 时绑定不可变 `PolicyBinding`，经
`NewRequestData` 传到 Worker slot；Scheduler 的保护区、retention/min_keep、zero-copy、
intent 和 commit result 使用相同绑定。full 请求禁止回收，feature-on 入口约束不变。
preempt/recompute 保持版本，Worker remove 清理 slot 绑定。没有引入 Codec/offload。

CPU 34 passed / 200 deselected，覆盖 full、预算、preempt 保版、真实 Pool 及序列化。
日志见 [PL01 CPU](../experiments/Engineering-Next-Policy/pl01-validation)。Worker 实际
传输使用 pickle；初次 MsgpackDecoder(NewRequestData) 遇到既有 TYPE_CHECKING torch
解析限制，保留 FAIL；CPU验证 PolicyBinding 的实际 msgpack 编码，真实TP2验证实际
Worker pickle传输。不在测试里新增仓库禁止的pickle导入。

[Random TP2 FULL 生命周期](../experiments/Engineering-Next-Policy/20261101-105334-pl01-random-bound-lifecycle-fixed)
通过两次回收、两rank计划一致、36层 retained payload、旁观请求不变、继续生成及
释放页 poison/rewrite/reuse。guard PASS，NVML项目合计峰18,167,627,776 bytes。
[初次启动 FAIL](../experiments/Engineering-Next-Policy/20261101-105120-pl01-random-bound-lifecycle)
揭示内部预热不经过 EngineCore；已在预热输入显式设置启动版本，业务请求仍严格校验。

这是绑定阶段的验证。混合版本、全rank prepare/publish、资源安全退役在 PL2 完成后
追加；静态 registry 不代表已完成在线更新。正确性场景带同步/clone，不作为性能对照。
85条质量样本继续冻结，payload或token正确不代表有损质量保持。
