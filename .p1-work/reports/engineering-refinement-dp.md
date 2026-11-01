# DP4 待确认工作量修正与有限 Serving 验证

2026-11-01，本轮修正“快照很新，但没有覆盖刚派发 ADD”的状态缺口。新增逻辑只
作用于显式 `known_work`，`requests` 仍为默认；沿用 prompt3 的 W1/W0、GPU0/1、
CPU20/21、4GiB/replica、batch512/seqs8、SinkRecent/fixed、FULL_DECODE_ONLY、
APC/async off，不改变权重、发布周期或质量样本。
本报告补充[原 DP4 报告](engineering-next-dp.md)，保留其 PARTIAL / SLO_FAIL。

## 状态覆盖与实现

原评分 `C+U/batch` 中，C 有 optimistic waiting 与本地 in-flight F 下界，U 却只取
Engine 快照。新请求已经派发而快照尚未覆盖它时，后续请求会低估该 replica。
时间戳新鲜度不能回答“是否包含这次 ADD”。本轮采用：

```text
effective_U = snapshot.U + sum(prompt_tokens of unacknowledged local ADDs)
score = max(4W+R,F) + effective_U / max_num_batched_tokens
```

`DPLBAsyncMPClient` 为每 frontend/replica 分配单调 `dp_add_sequence`，只保留
短暂的 `request_id -> (sequence,prompt_tokens)`；tokens 兼容 token IDs/embeds
既有长度入口，不预测未知输出、不按 client_count 放大。本地 pending 不是远端
精确队列，也不包含其他 frontend 未确认工作。

序号经 `EngineCoreRequest` 编码与 `Request.from_engine_core_request` 进入原处理链。
同源 ZMQ→预处理→input_queue→Engine 主线程顺序消费；`_handle_client_request`
在 `add_request` 返回后才更新 `dp_add_watermarks[client_index]`。它与同次 U
快照一起 copy/发布，水位变化参与去重；counts 变化仍立即发布，其他变化继续
100ms 合并，不加 idle 心跳或逐 token 同步。

Coordinator 沿原每 replica latest 透传。Frontend 应用快照后，仅删除本 client
`seq <= watermark` 的 pending，既不漏掉后续 ADD，也不双计已覆盖 prompt。
abort 发出不等于 Engine 已结束，因此继续补计，finished/error 经原 affinity
删除。显式 rank、sticky、轮转 tie、F、原 abort 分组及 requests 路径保持；MoE
继续拒绝 known_work，PL2 不扩为跨 DP 热更新。

源码路径见 [DP 调用链](../source-analysis/dp-runtime.md)，稳定测量契约见
[DP 协议](../notes/dp-protocol.md)。此处改变 CPU 控制状态，不改 KV payload、
Pool ownership 或跨副本迁移。

## 正确性证据

组合 CPU 回归 **72 passed / 227 deselected**，覆盖 CB、DP 及原策略相关契约；
命令和完整日志见 [Refinement B0](../experiments/Engineering-Refinement-B0) 中
`run-cpu.py` / `combined-cpu-first.log`。沿用 runner 对两个未缓存、无关 tokenizer
的本地 Qwen3 替换；不称完整上游 suite 通过。

新增行为复用现有 `test_engine_core.py` / `test_engine_core_client.py`：真实编码
保留 sequence，主线程 admission 后才确认；同 counts/U 的 ACK 仍发布、快照字典
不随后续水位 mutation 改变；部分/重复/其他 client ACK、finished 先到与 abort
语义通过；真实 Coordinator/ZMQ 传播后准确撤销补计。pre-commit 和 manual
`mypy-3.12` 最终通过，首次格式检查失败日志保留。

`analyze_dp_work.py --formal` 用实际 ADD 日志按本 client watermark 重建每次
pending 集合，断言既无遗漏也无重复，再复算公式与轮转选择。每轮八条 arrivals
先于完成（由 F 验证），因此这段重建不猜测完整生命周期。正式计时不启用
load-diagnostic/逐 step trace/NSYS；两模式的源码及 helper hash 一致。

## W1：正向与反向

每次完整负载预热后3轮×8请求，全部原始结果保留：

| 运行 | 三轮 output tok/s | SLO | TTFT max | 逐请求 TPOT max |
| --- | --- | --- | --- | --- |
| [requests 正向](../experiments/Engineering-Refinement-DP-Formal/20261101-141625-requests-w1) | 210.371 / 207.798 / 212.198 | 20/24 | 2.541s | 31.769ms |
| [known_work+pending 正向](../experiments/Engineering-Refinement-DP-Formal/20261101-141747-pending-work-w1) | 249.449 / 253.901 / 257.565 | 24/24 | 1.739s | 27.094ms |
| [known_work+pending 反向先运行](../experiments/Engineering-Refinement-DP-Formal/20261101-142019-pending-work-w1-reverse) | 249.252 / 249.342 / 249.634 | 24/24 | 1.698s | 27.094ms |
| [requests 反向后运行](../experiments/Engineering-Refinement-DP-Formal/20261101-142233-requests-w1-reverse) | 211.968 / 176.194 / 175.201 | 19/24 | 3.446s | 32.957ms |

按三轮公共时间窗合并，正向吞吐210.107→253.595 tok/s（+20.698%），goodput
1.368→1.981 req/s（+44.838%）；反向186.324→249.409 tok/s（+33.858%），
goodput1.152→1.949 req/s（+69.083%）。基线两次启动本身有漂移，故不把区间
端点当精确通用收益；新方案两次启动的六个 timed round 全部8/8，本 W1 有限
证据支持 `SERVING_BENEFIT / SLO_PASS`。这组 A/B 比较默认 requests 与修正后的
known_work 整体，没有隔离旧 known_work 与 pending 单项的性能百分比。

requests 正向每轮两 replica 的 prefill 分别9728/25088、9728/25088、10240/24576；
反向为10240/24576、2048/32768、2048/32768。修正后正向17920/16896、
17408/17408、17408/17408，反向三轮16896/17920。全部 timed 零抢占、零重算，
每对24条输出 token IDs 全部一致；这里不验证有损质量。

正向 r0-6 决策可直接核对：replica1 快照水位6/U6656，r0-4 的 sequence7、
8192-token prompt 尚未确认；pending 将它纳入。replica0 pending 为512tokens，
最终新长请求选 replica0，避免重复堆积到 replica1。两个新路由 run 均有21/33
次 ADD 决策使用非零 pending，最大补计8704/8192tokens，包含小预热及完整预热。

W1 原 router CPU 中位约25.7–26.9μs，新约36.7–37.9μs；包括观测写文件的 wrapper
中位约260–304μs，最大9.54ms。观测不是零成本，两边相同；不把该 wrapper 成本
当 Runtime 路由函数成本。资源 guard 全 PASS，项目 NVML 合计峰值均
42,830,135,296 bytes，低于44GiB停止线；实际 affinity 文件确认两个 Worker
主线程分别 CPU20/21，不声称整个服务所有线程绑核。

## W0：均匀对照与停止

| 运行 | 三轮 output tok/s | SLO | TTFT max | 逐请求 TPOT max |
| --- | --- | --- | --- | --- |
| [requests](../experiments/Engineering-Refinement-DP-Formal/20261101-142609-requests-w0) | 277.881 / 248.388 / 279.693 | 24/24 | 1.703s | 25.098ms |
| [known_work+pending](../experiments/Engineering-Refinement-DP-Formal/20261101-142744-pending-work-w0) | 275.331 / 276.049 / 276.528 | 24/24 | 1.380s | 22.735ms |

合并吞吐267.858→275.968 tok/s，名义+3.028%；goodput2.093→2.156 req/s。
requests 本次三轮跨度约12.6%，收益未超过其重复波动；W0 只判定本次无超过5%
退化、SLO均通过，不声称可分辨提速，不为3%的名义正值继续追加反向或扫参。
requests prefill 为16384/16384、20480/12288、16384/16384；新路由三轮均
16384/16384。两边零抢占/重算，24条 timed 输出全部一致；guard PASS，峰值
42,830,135,296 / 42,828,038,144 bytes。评分和完整 pending 集合复算通过，结果
见实验 stage 根目录 `w0-paired.json`。

## 结论边界

工作量覆盖正确性已验证，W1 的本轮有限正反向观察取得 Serving 收益。此结果
不改写原 DP4 的 PARTIAL，不证明任意到达分布、输出长度或多 frontend 均稳定
提速；默认保持 requests。没有继续扫 C/U 权重、刷新周期或新增压力惩罚。
W0 未测得可分辨提速；本轮共六次正式 run 后停止，不扩旧矩阵。正式复算输出
见实验 stage 根目录 `w1-paired.json` / `w0-paired.json`，所有 run 的 protocol、输入
token IDs/hash、源码/hash、routing、请求、工作量、日志和 NVML 可独立审计。
