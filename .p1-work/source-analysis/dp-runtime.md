# DP2 dense 源码与局部 ownership

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
