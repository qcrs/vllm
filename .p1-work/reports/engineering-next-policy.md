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

## PL2 安全更新

`EngineCore.update_reclamation_policy(dict)` 经既有结构化 utility 和 executor collective
RPC执行。允许更新policy、stats_path、retention、sink/recent/min_keep；pressure、
budget_mode、reserve不能在线混改。单调分配engine-local版本，所有TP rank完成资源
准备、有限score/selection预热和设备同步，且version/config/实际校准bytes的SHA256
一致后才发布默认。CPU注入异常、缺rank、重复rank、config/hash不一致均零发布，
失败版本不复用；原utility错误通道明确返回失败，不增加猜测回滚或通用重试。

调用方式为已有client的`call_utility[_async]("update_reclamation_policy", options)`；
此后EngineCore处理的ADD绑定新默认，也可通过SamplingParams.extra_args中的
`kv_reclamation_version`选择尚存的已准备版本。既有请求和intent保留旧绑定。
本版只支持单engine TP1/TP2。DPLB广播入口和带Coordinator的EngineCoreProc明确
拒绝更新；dense DP会重写本地DP size=1，不能仅靠该字段判断。没有新增HTTP管理服务。

正常step先等待真实输出，再由Scheduler commit/free，随后`post_step`检查逻辑引用。
旧版本非默认且Scheduler.requests无引用时，所有Worker在设备全部stream同步后删除
对应adapter/stats/workspace，核对全rank退役ACK后EngineCore删除版本。preempt仍在
requests中，继续pin；EOS/abort沿既有完成与step收尾协议。Worker残留的finished slot
只持有不可变binding值，不持有版本资源，slot remove清理它。回收自己的completion
仍必需，prepare的同步不替代它。失败prepare可能留下未发布的Worker资源，随engine
关闭释放；不宣称失败后跨rank事务回滚。mutation后异常维持原fail-stop。

| 验证 | 结果与边界 |
| --- | --- |
| CPU utility/失败ACK/单engine拒绝 | 7 passed / 34 deselected；真实结构化utility dispatch、旧新请求保版、无引用才退役、拒绝DP广播 |
| [真实TP2](../experiments/Engineering-Next-Policy/20261101-110234-pl2-mixed-delayed-tp2) | FULL；一个rank prepare延迟1s，观察默认仍旧版；首次回收另延迟3s，canonical未提前free；三策略各回收一次，新full不回收 |
| [真实TP1](../experiments/Engineering-Next-Policy/20261101-110649-pl2-mixed-tp1) | GPU1/CPU21，同样混合版本；三次回收、全36层payload、旁观请求、attention reference、继续生成、poison/rewrite/reuse通过 |
| 退役 | 两种TP规模最终仅版本3/full，旧版0/1/2退役均晚于对应local-done；Tri资源每rank减少610,304 bytes，CUDA allocator保留缓存不等于NVML立即下降 |
| NVML合计峰值 | TP2 18,167,627,776 bytes；TP1 17,179,869,184 bytes，均guard PASS/exit0 |

TP2的Random/Tri/full更新时间约1.005/3.879/0.00081s；TP1约1.005/3.402/0.00084s。
Random含人为1s prepare延迟，Tri含首次JIT/有限warmup，不是稳定态热更新性能保证。
首个新请求step walltime：TP2 B/C/D约49/904/790ms，TP1约49/904/813ms；这些step
还包含其他请求、payload检查和同步，仅记录本次受扰动冷开销，不能当TTFT或Serving
A/B。完整证据、逐rankmemory、markers和复算结果见
[PL2验证归档](../experiments/Engineering-Next-Policy/pl2-validation)。

PL0/PL1/PL2均IMPLEMENTED/CORRECTNESS_PASS。没有新增Serving收益或质量保证；
切默认full仅影响新请求，不恢复旧请求已删除KV。85条质量样本继续冻结。
