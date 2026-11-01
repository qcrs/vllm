# P1 v8 测试设计

## Refinement：TP 动态长度与冷路径

- 用途：分组评分在物理 E 跨页对齐状态变化时复用同一动态 kernel，避免为不参与地址对齐的长度重编译。
- I/O：相同层、row、logical L、eligible 域，改变动态 E；shared scores/KeepPlan 保持原数值契约。
- 风险：取消特化后 partial tile/页边界掩码错误，或只消除数值特化却仍为16对齐生成新版本。
- 最低有效验证：扩展现有独立 complex reference 为 E=95/96，另在同一 scorer 顺序调用跨对齐 E，通过 Triton cache hook 确认第二次不新增 grouped 进程缓存变体；真实8B TP2 continuation/free/reuse验证执行链。冷路径诊断与正式 A/B 分开，不将 host inclusive span 当纯 GPU 时间。

## Refinement：DP4 未确认 ADD 工作量

- 用途：仅补计已派发但未被 Engine 快照覆盖的 prompt，维持原 C+U/batch。
- I/O：每 frontend/replica ADD 序号经真实序列化进入主线程；快照携带处理水位，前端按本 client 水位撤销临时补计。
- 风险：旧快照抹掉刚派发的长请求、部分/重复/其他 client ACK 错删、完成先于 ACK、abort 提前删除、counts/U 不变漏发布，以及显式 rank/sticky 改变。
- 最低有效验证：现有 engine/client suite 验证序列化与主线程确认、真实 Coordinator/ZMQ 传播、路由和生命周期；再用固定 W1/W0 的真实 DP2 HTTP 有限 A/B 验证指标。沿原 100ms 合并，不新增逐 token 同步。

## Refinement：首次 retry 容量预算

- 用途：将回收后下一次同步调度中、截至 requester 的分配纳入最大合法 K。
- I/O：Scheduler 同次 running 顺序、post-forward E/q、原 allocator 页需求、free；输出满足预测重试的最大 K，软 reserve 不压低原 retention/保护底线。
- 风险：victim 尾页恰满后抢走一页、victim 在 requester 后却被多收费、requester 自身被回收、q=0/1、前方 prefill 改变 token budget、APC 再查询引入副作用。
- 最低有效验证：纯预算用枚举 K 的独立容量 oracle；现有 Scheduler/真实 Pool 验证 running/waiting 首次重试、页增长与下限退回；TP2 helper 验证新尾槽 K 的实际 payload/continuation/free/reuse，再按固定正式配置验证。
- 预测只覆盖已知的一次同步 pass，不锁定未来新到达或 EOS；无法在该 pass 排到 requester 时保留 fixed floor/原单次重试，不声称预留容量。
- 有限前瞻：`retry_pass` 将同一 token budget 内 requester 后面的已知 running 分配作为软目标；全部覆盖不可行则退回 requester 前缀。真实 Pool 验证前方 decode 自救后不挤掉后方 prefill；纯预算 oracle 验证软目标不越过 floor，不加入未知到达或完整未来生命周期。

## CB0/CB1 缺口预算

- 用途：同一候选快照产生固定预算或满足硬缺口/软 reserve 的最大合法 K。
- I/O：source E/row/q/P、原分配器 D 与配置底线；输出合法 K、预计释放页及 exact zero-copy。
- 风险：partial tail 或共享前缀导致页数错误、降低 K_floor、将 reserve 当硬缺口、未完成 forward 冒充已完成、预测与 commit/free 不一致。
- 最低有效验证：现有 KV/Scheduler suite 的参数化预算与真实 Pool 行为，再用既有 TP2 helper 验证 q=0/1、payload/continuation/free/reuse；一次无 NSYS 的 trace 关联全部合法候选与首次 retry，正式 fixed/demand 各三轮。

## R1 resumable请求入口

- 模块用途：在reclamation支持域之外拒绝保留KV的streaming生命周期。
- I/O：`Scheduler.add_request` / `AsyncLLM.add_request` 接收resumable请求或AsyncGenerator；feature-on抛清晰错误且不改变请求、队列、Pool或前端状态，feature-off按原生session控制链执行。
- 防止的失败：allocator按retained E分配，而session再次admission把Scheduler/Worker E设回L。
- 最低成本层级：复用现有Scheduler/Pool fixture及完成结果helper，用64-token prompt、3个输出、16-token追加验证feature-on拒绝与feature-off暂停/重入；既有AsyncLLM suite只检查消费/注册之前的入口失败，用 `asyncio.run`，不依赖新增异步插件。CPU完成helper不证明GPU completion或payload。

## Graph capture / replay

- 模块用途：让原生模型 graph 读取持久 physical attention lengths，保留 logical RoPE/sampling 输入。
- I/O：dummy capture 与实际 replay 的 `InputBatch`、独立 physical buffer；输出 metadata 必须绑定同一 allocation，dummy refresh 清除真实请求的残留 lengths。
- 防止的失败：capture 绑定 logical L、运行时只换 Python metadata、dummy replay 读取旧 E、padding 未清零，以及回收后旧 row/tail 仍可读。
- 最低成本层级：在已有 compaction suite 用 CPU dummy metadata 验证地址/长度分离与 dummy refresh；真实 Qwen3-8B FULL replay 检验重复回收、同 bucket、padding/reorder、立即 release/rewrite/reuse、retained attention reference 和 capture count。Graph/Eager 使用相同 KeepPlan 对照，性能另跑无逐步同步的负载。

## M0 / M1 Payload

- 模块用途：将 current-member 有序 KeepPlan 映射到 retained KV；全层/Head 使用同一集合。
- I/O：真实 packed 4D Layer views、source row、升序 current-member indices；输出 valid retained bytes 与合法 retained row。无效 tail 无零值契约。
- 防止的失败：非连续 page ID、padding stride、partial tail、同 tile/跨 tile overlap、stale released-page read、K/V channel 混淆。
- 最低成本层级：小型 CPU scalar Oracle 独立逐元素寻址；GPU kernel fixture 与该 Oracle bit-exact 比较；真实 release/reuse、下一 decode 和 completion 在 Engine 验证。

附近 `test_gpu_block_table.py` 只覆盖 metadata staging，不能检验 Payload 搬移，因此新增 `test_kv_compaction.py`，后续复用同一 Oracle/fixture。Oracle 不 clone Pool、不进入 Serving。

## M0 allocator

沿已有 `tests/v1/core/test_scheduler.py` / `test_single_type_kv_cache_manager.py` suites 扩展真实 requirement/Pool 行为；Engine native/instrumentation 比较 token IDs。ID trace 不触发 GPU sync，Serving 默认只计数。

## M1 canonical ownership

- 模块用途：消费已完成的 Worker row/E，保留 canonical Block 对象并释放其余页。
- I/O：原 Request、源 row/长度、完成结果；输出新 ownership/E 和真实 free queue。
- 防止的失败：重复、null、外部、过期 ACK 导致错 free；按 logical L 重新申请已丢弃页；EOS/abort 双重释放。
- 最低成本层级：复用 `test_kv_cache_utils.py` 的真实 Pool 和 Request helper，验证 commit 后按 E 增长并被 B 复用；GPU完成、EOS/abort 和 preemption 在 8B Engine 实验验证。

## M2 pressure controller

- 模块用途：在真实allocation miss时选择一个可回收decode请求，保留partial batch并仅延迟一次重试。
- I/O：allocator同一计算点的required-available、当前ownership/E、已排定q；输出唯一intent或原生调度行为。
- 防止的失败：以token数量冒充真实释放页、排序不确定、丢掉partial allocation、反复回收而不执行native fallback。
- 最低成本层级：pure候选排序在既有KV suite检验；真实8B Engine小池验证miss→完成→release→下一step重新allocation，另用Scheduler真实Pool构造running miss与一次重试。

## M2 shared scoring

- 模块用途：按实际paged K_rot与校准Q频域统计产生唯一current-member有序集合。
- I/O：packed4D/真实stride、nonmonotonic row、GPU logical L_after、各层/统计head数据，输出FP32 shared scores与int32[K]计划。
- 防止的失败：用E作为时间、Qwen half pairing错误、query-head统计提前平均、GQA计权重复、保护域影响z-score、同分选择不稳定。
- 最低成本层级：在现有GPU compaction suite内用独立CPU FP32 dense复数计算比较raw/shared score与计划；包含query-head和KV-head stats、非连续页、padding、保护窗口及零方差。之后8B真实K与GPU logical状态验证。

## Prefix ownership / APC

- 模块用途：保留 canonical shared prefix，只对已撤销全部 lookup 身份的私有 suffix 回收。
- I/O：真实 APC lookup/touch、当前 row/refcount、forward 后 E、临时 P/K；输出完成后的 row/E、缓存连续计数、free queue 与 KV events。
- 防止的失败：APC admission 的 Scheduler/Worker E 不一致；漏撤 suffix alias 后新增共享；deindex 当 free；非连续 shared 页被写；lossy 后沿原 logical chain 重新发布；正常 in-flight 被误判 lossy；最后引用前重用与 EOS/abort double-free。
- 最低成本层级：扩展现有 scheduler 与 prefix-caching suite，用真实 sha256 Request/BlockPool 复现 admission、lookup/event/ownership；仅 worker completion 用 CPU result。GPU Gate 使用 Qwen3-8B 的真实 A/B/C 请求、全层 payload 比较、FP32 attention、真实 release/rewrite/reuse，另在同配置 native 与 allocation pressure 下运行。CPU 不代替 GPU completion 或 payload 证据。

## TP shared plan / completion

四问与完整协议见`tp-protocol.md`。在现有compaction suite加入Hkv=1/4、TP2
贡献与独立CPU复数全query-head reference对照，覆盖复制owner保留完整GQA stats、
非owner零贡献及全局distinct-head等权。真实两GPU通信/统一plan与payload留给
独立小fixture和Qwen3-8B MultiprocExecutor轨迹。

扩展既有executor suite验证非output rank在sample_tokens失败会退出RPC loop，
以及AsyncOutput设备完成异常不转成可继续执行的普通FAILURE；feature关闭维持
原生行为。真实failure注入额外证明monitor终止TP组且canonical零commit/free。

## DP replica-local ownership / routing

四问见`dp-protocol.md`。复用真实AsyncLLM/internal dense DP2，不新增模拟allocator：
副本0的A/C制造一页真实shortfall，副本1空闲页不得计入available；本地回收后C
必须取得released ID并全层重写，A的retained KV保持不变。B在副本1持续decode，
abort A沿原affinity发送，逐forward核对B的128-token前缀KV不变。native/Sink的B
输出逐token对照，两个Pool最后均free32；四个额外请求验证原生router真实分布。
正式HTTP去掉payload同步hook，逐replica报告延迟/Goodput/preemption，独立trace
记录queue/free/重试，不用静态源码检查替代双GPU证据。

## D1 dense DP 状态闭环

- 模块用途：EngineCore 发布 Scheduler 队列变化，Coordinator 汇总独立副本，Frontend 据快照和本地 affinity 表选择 engine。
- I/O：`(running, waiting)` 经真实 Msgpack 进入 `[waiting, running]`；完成后归零，未变化不重复发布；MoE 保留 step/wave。
- 防止的失败：dense 不发布、字段颠倒、完成后旧负载残留、把独立 step 当同步边界，以及快照覆盖乐观 waiting 后忽略本地在途请求。
- 最低成本层级：复用两个 engine suite，CPU 检查 busy loop 发布与序列化、真实 Coordinator ZMQ 的独立更新、路由下界/完成/显式 rank/abort；DP2 HTTP 异构与均匀对照验证真实进程链。既有 suite 顶层 tokenizer 未缓存，选中 CPU 测试用已存在的 Qwen3 tokenizer 替换该无关模型资源，不运行其模型测试，原 collection 失败单独归档。

## T1 normalization / GQA / head reduction融合

- 模块用途：减少逐层评分归一化和归约的发射成本，保持current-member shared score和KeepPlan语义。
- I/O：FP32 raw scores、eligible域population variance/mean、local distinct-head/GQA映射，输出FP32 layer contribution；沿原顺序和权重累加。
- 防止的失败：非2幂head/group的padding参与MAX/mean、partial token tile越界、零方差分母变化、同分或相邻cutoff的计划变化，以及TP贡献计权错误。
- 最低成本层级：扩展现有compaction suite的独立dense复数reference到group=3，再用受控raw scores隔离归约的partial tile/零方差/ties；真实TP2 collective reference检查所有rank计划，一条Qwen3-8B重复回收→continuation→free/rewrite/reuse验证真实路径。未改变completion/Graph协议，不复制旧故障注入矩阵；性能只在独立benchmark/Serving测量。

## TP2a 有序层累计融合

- 用途：在既有 normalization/GQA 归约中累加 FP32 层贡献，消除独立 add 和 layer_total。
- I/O：raw score、eligible population variance/mean、distinct-head 权重与旧 total；输出同顺序 shared score/KeepPlan。
- 风险：Torch alpha-add 与 FMA 舍入差异在 cutoff 改变排名，零方差/partial tile/padding 或 TP 权重出错。
- 最低有效验证：既有独立 complex reference 与 near-cutoff suite、真实 TP2 SUM/plan、8B continuation/reuse；三个冻结形状的当前 T1 基线与候选 micro，只有完整链受益才反向及 Serving。

## TP2b 四层探针

用途是批量 raw/eligible population variance/有序累计；I/O仍为相同shared score/KeepPlan。
风险是跨层stride、统计域、GQA和舍入改变排名；先复用独立复数reference、零方差与真实TP2完整heads测试，
再比较冻结三形状计划hash与完整链。probe不把逐层normalization隔离计时冒充分组阶段。

## PL0/PL1 与 PL2 验证设计（2026-11-01）

- 模块用途：在 ADD 边界 pin 不可变策略，Controller/Worker/intent/commit 同版；更新先全rank准备再发布，引用清空和设备完成后退役。
- I/O contract：已准备version通过extra_args选择，结构化utility准备/发布/退役；未知version、full回收、ACK失败/不一致及仍被引用的退役明确拒绝。
- 主要失败：预算与scorer串版、preempt/slot复用串请求、某rank未准备提前发布、旧intent未完成提前卸载。
- 最低有效层次：既有Scheduler/Pool/engine suites验证CPU状态/序列化/拒绝；复用tp_reclamation与worker hook，在8B TP1/TP2验证延迟prepare、混合策略、全层payload、continuation与free/reuse。

## DP3 已知工作量快照

- 用途：Scheduler产生去重unfinished工作量，经原Coordinator通道到实际Frontend shadow。
- I/O：`num_tokens - (num_computed_tokens - num_in_flight_tokens)`与Pool usage；输出带engine/sequence/current_step/生成与接收时刻的快照，不预测max_tokens。
- 风险：提前扣除在途、preempt重复/漏算、counts不变压住U更新、完成未归零、把状态年龄冒充网络延迟、改变affinity/F/tie语义。
- 最低有效验证：现有engine/client suite用真实Request计算在途/重算/结束，受控clock验证100ms合并及立即归零，真实ZMQ验证独立副本快照与Frontend接收；再固定W1/W0各完整预热+两轮真实DP2 HTTP诊断，不改Router。
- DP4进入判据预注册：不同请求、不同source sequence上重复出现相近C（差≤1）、U差至少512且两副本状态年龄≤250ms；还须结合实际排队/工作/忙闲判断，shadow分歧本身不是错选证据。

## DP4 显式工作量路由

- 用途与I/O：仅在dense internal DP新ADD上，用已收到的U及固定batch计算`S=C+U/batch`；默认requests保留D1。
- 风险：U代替F下界、改变sticky/显式rank/abort、相同分数失去轮转、把未知输出当工作量、路由更均匀冒充性能收益。
- 最低有效验证：在现有client suite扩展两模式行为与tie，复用affinity/F用例；W1/W0各完整预热+3轮A/B，关闭trace/load-diagnostic，仅每完成请求写CPU工作汇总、每ADD写路由快照，正收益再一次反向。
