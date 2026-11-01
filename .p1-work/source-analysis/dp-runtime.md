# DP2 dense 源码与局部 ownership

## Refinement：快照覆盖与 ADD 确认

`DPLBAsyncMPClient` 仅在 `known_work` 下维护短暂 pending：每个 replica 的
`request_id -> (dp_add_sequence, prompt_tokens)`。原 `reqs_in_flight` 仍拥有
sticky/abort/finished affinity，pending 不复制完整生命周期。路由使用
`max(4W+R,F)+(snapshot.U+pending_prompt_tokens)/batch`；requests 默认不登记补计。

序号是 frontend/replica-local，通过 `EngineCoreRequest` 编码及 `Request` 预处理
传递。现有同源 ZMQ 输入、预处理队列和主线程按顺序消费 ADD；可靠主线程
`EngineCoreProc._handle_client_request` 在 `add_request` 返回后才推进对应
client 水位。`_maybe_publish_request_counts` 将水位 copy 与同次 U 打包，水位变化
也参与状态去重，沿原100ms合并与队列转换立即发布。preprocessing 线程不读取
全局确认状态；尚未 admission 的 ADD 不能被提前确认。

Coordinator 继续透传每 replica latest。Frontend `_apply_load_stats` 在应用快照后
删除本 client `seq <= watermark` 的 pending；其余 client 的 ACK 不删除本地记录，
旧/重复/部分 ACK 不抹掉后续 ADD。finished/error 可先于快照删除其 pending；
abort 发出不等于 Engine 已停止工作，因此不提前删除。新鲜时间戳只说明快照年龄，
watermark 才界定它覆盖哪些已派发 ADD。多 frontend 只能补计各自尚未确认的工作，
不声称掌握其他 frontend 的全部新工作。

CPU 编码/主线程/ZMQ 与真实 DP2 HTTP 的证据见
[修正报告](../reports/engineering-refinement-dp.md)；不改 MoE step/wave、Pool
ownership 或 PL2 支持域，不将异步快照视为 admission 容量保证。

## 2026-11-01 DP3 / DP4 当前增量

下文 D1 和旧局部压力轨迹保留历史语义。本轮实现、进入证据及有限 W1/W0
正式对照见 [下一阶段DP报告](../reports/engineering-next-dp.md)；不重跑旧矩阵。

`Scheduler.get_dp_load_snapshot` 从 canonical `requests.values()` 计算
`U=sum(max(0,num_tokens-(num_computed_tokens-num_in_flight_tokens)))`；同一请求
移入 waiting 不增加第二份记录。在途 forward 不提前扣减，未知未来输出不计入。
`DPLoadSnapshot` 是 CPU 控制消息：U、usage、sequence、`current_step`、生成及
Coordinator 接收 monotonic ns；没有 GPU score、Block ownership 或容量承诺。
usage 的 owner 仍是本 replica 的 KVCacheManager/BlockPool。

EngineCore 持有 last counts、上次采样时间/状态和 source sequence。counts 改变或
最终归零立即发布；其余变化按100ms采样，U/usage 不变也更新时间，避免逐 token
扫描。Scheduler 只生成当前快照，不持有另一份 frontend 生命周期。普通 output
queue 的 client=-1 编码后发到 Coordinator；dense 继续按每 rank latest 汇总，
在现有节奏发布 `(counts,wave,running,loads,published_ns)`。Frontend 的
`DPAsyncMPClient._apply_load_stats` 解码、切 managed slice，并记录接收时间。
同机 monotonic 可关联首达延迟；状态年龄还含合并等待/idle，不等于网络延迟。

`DPLBAsyncMPClient` 的持久 `known_work_batch_size` 由 additional_config 的
`dp_load_balancing` 选择：默认 `requests` 继续 `C=max(4W+R,F)`；显式
`known_work` 仅 dense 使用 `S=C+U/max_num_batched_tokens`。缺快照的 U 为0，
不新增过期预测或usage惩罚。只有新 ADD 选择engine；显式rank、sticky、轮转tie、
本地F下界、optimistic waiting、abort/finished affinity仍由原client拥有。
DP4 是可选实验模式，未将其设为全局默认；MoE保留原step/wave，拒绝此模式。
两种路由均不跨Pool free/reclaim，不迁移活跃请求。

CPU控制链、真实DP2诊断及正式A/B分别验证消息/路由、实际工作与用户时延。
`P1DPWorkScheduler` 仅实验脚本统计scheduled/prefill/重算，每请求完成写一次；
不进入生产Scheduler的ownership状态，不把诊断频数复制到正式结果。

## 既有 D1 与局部 ownership 记录

真实双Engine/GPU局部压力、释放/重写与abort轨迹已PASS，详见`../reports/dp.md`。
Serving性能独立报告，不以源码推导或诊断同步hook代替性能实测。

`EngineCoreClient.make_async_mp_client`在internal DP选择`DPLBAsyncMPClient`。
原生frontend启动两个EngineCoreProc；`EngineCoreProc.run_engine_core`保存
`parallel_config.data_parallel_index=dp_rank`，dense分支把worker配置的DP size/local
重置1、rank重置0，分别构造`EngineCoreProc(engine_index=dp_rank)`。每个进程各有
Scheduler/KVCacheManager/BlockPool，MRV2 `dp_size=1`，不进入MoE DP padding同步。
TP group每replica size1，回收代码的TP score/plan/completion通信分支均不执行。
源码入口为
[`EngineCoreClient.make_async_mp_client`](../../vllm/v1/engine/core_client.py)、
[`EngineCoreProc.run_engine_core`](../../vllm/v1/engine/core.py) 的dense拆分和
[`KVReclamationWorker.reclaim`](../../vllm/v1/worker/gpu/kv_reclamation.py)。
以symbol定位；旧 `kv_reclamation.py:122` 已落在内部计划分支，不能作为reclaim入口。

`DPLBAsyncMPClient.get_core_engine_for_request`只在新请求ADD时选择replica：
显式`request.data_parallel_rank`优先，其后保留late-interaction sticky选择；否则按
`max(waiting*4+running, local_inflight)`最小值，轮换tie起点，
选中后本地waiting增量弥补约100ms coordinator统计间隔。`reqs_in_flight`保存affinity，
abort按该映射分组发向原engine，finished后删除映射。原生内部控制和stats协调保留，
不能将它误称为cross-DP reclamation barrier。

本轮D1修复见 [工程报告](../reports/engineering-dp-load-chain.md)。普通dense
`EngineCoreProc.run_busy_loop`在处理输入后和engine step后调用基类
`_maybe_publish_request_counts`。Scheduler的`get_request_counts()`返回
`(running, waiting)`；发布使用具名字段构造`SchedulerStats`，含最后一次归零。
`last_counts`由EngineCore拥有，只抑制重复计数，不是请求生命周期表。
既有output线程以client index -1经Msgpack/ZMQ送到Coordinator。

`DPCoordinatorProc`只在`enable_wave_coordination`时保留MoE的step/wave聚合和
最少等待；dense每rank独立推进，以各rank最新计数汇总。内部和frontend编码均为
`[waiting, running]`，原有约100ms发布节奏不变。Frontend stats task替换
`lb_engines`；这会覆盖乐观waiting增量，因此选择时从已有`reqs_in_flight.values()`
派生Counter作本client下界F。该扫描是O(本client在途请求数)，未新增重复的完成/
abort账本。F不覆盖其他client、不是token工作量，也不保证远端admission成功。

MoE仍走`DPEngineCoreProc`自己的busy loop和publisher，保留step/wave。
本轮只做CPU消息字段回归，未运行MoE模型；不能由dense结果扩张MoE支持声明。

HTTP completion从`X-data-parallel-rank` header取得显式rank，原生默认不返回replica
编号。逐replica测量需要在诊断server入口记录原生router的返回值并关联request ID；
该观测不改选择算法，也不把free摘要加入安全预算。原生Prometheus已有engine标签。
HTTP调用入口为
[`OpenAIServingCompletion._create_completion`](../../vllm/entrypoints/openai/completion/serving.py)，
header解析由
[`GenerateBaseServing._get_data_parallel_rank`](../../vllm/entrypoints/generate/base/serving.py)
拥有；路由与abort分别为同一
[`DPLBAsyncMPClient`](../../vllm/v1/engine/core_client.py) 的
`get_core_engine_for_request` 和 `abort_requests_async`。诊断server为
[dp_server.py](../scripts/dp_server.py)，独立CLI不改生产入口。

`KVCacheTrace`只有time/request/Pool字段，未含replica身份。DP诊断按保留的
`data_parallel_index`写独立replica目录，包含GPU UUID/PID、scheduler与payload
trace；不把两个pool中同数字Block ID混为一页。
同进程CPU pool ID只用于诊断身份，不作为跨进程可传递ownership。

实现沿用native router。本地required16/available15且peer free32的真实miss没有
借用peer页；Sink一次source E258→K129释放8页，C全层重写，abort A后B继续
112次已验证forward，native/Sink的B输出128tokens完全相同。实际worker DP/TP
均1，两个GPU UUID不同。无需新增Runtime跨DP逻辑。

只有可重复的native偏斜与独立A/B收益支持时才保留routing hint；人工定向skewed
负载是本地容量边界，不能归因native算法。DP2×TP2需要4GPU，不在本机Gate内。
