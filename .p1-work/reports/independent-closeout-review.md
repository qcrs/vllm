# P1-v8 Physical KV Cache Reclamation 独立收尾 Review

审查日期：2026-10-31。审查基准 HEAD：
`252c747ad6127d4c17540e5fb08704336fd756be`；分支：
`p1/v8-physical-kv-reclamation`；vLLM 基线：
`568afb3a13806beb53bb2e6bd518269357b237c0`（v0.26.0）。
审查开始时工作树干净。本报告记录该 HEAD 的发现，尚未实施下文修复。
后续修复请引用问题编号，并记录修复提交、验证命令、结果和剩余边界。

## 1. 结论与修复顺序

**当前不能无条件宣布功能收尾：先修复 R1 的 streaming-input session 重入误放行。**
有损回收后，保留 KV 的 session 再次进入 admission 时会把 physical E 重置为
logical L，破坏 allocator 容量与 Worker 访问长度的一致性。
该问题已由真实 Scheduler/BlockPool 的 CPU 最小复现确认；未新测 GPU 后果。

除 R1 外，已审查的普通请求链、重复回收、EOS/abort、preempt/recompute、q=0、
maintenance、Prefix、Graph、TP completion 和 dense DP 隔离未发现新的实质缺陷。
当前结构沿 canonical owner 与执行镜像组织，不需要大规模重构或另建事务框架。

性能结论可以保留，但必须带配置边界：

| 场景 | 独立重算结果 | 判定与边界 |
| --- | --- | --- |
| TP2 short/eager，8GiB/rank，CPU20/21，batch2048 | native/noop 为 62.712686 / 62.168299 tok/s；开销 0.868066%；12 对请求逐 token 相同 | 本机显式绑核配置的样本内预算 PASS；不是默认任意 CPU 放置通过 |
| TP2 pressure，0.75GiB/rank，CPU20/21，batch1024，SinkRecent | 平均 100.917 tok/s；17/24 达 SLO；零抢占；Goodput 0.5584 req/s | SLO FAIL，七个失败请求均为 TTFT 超标 |
| DP2 skewed，1.5GiB/replica，CPU20/21，batch512，SinkRecent，固定 6/2 路由 | 平均 125.392 tok/s；24/24 达 SLO；零抢占；Goodput 0.9796 req/s | 该限定配置 PASS；不能归因于 native router 改进或跨池容量共享 |

SLO 固定为 **TTFT ≤ 2s 且 TPOT ≤ 80ms**。历史 TP eager 14.33%/6.871% FAIL、
M2/Prefix 压力 FAIL、旧 DP skewed 18/24 和旧 profile PARTIAL 都应保留。
原实验矩阵允许“正确但该配置性能 FAIL”，不要求所有负载统一加速。
因此 TP 压力 FAIL 是尚未达到的性能目标，不自动否决整个 Core。

| 顺序 | 编号 | 性质 | 建议行动 |
| --- | --- | --- | --- |
| 1 | R1，P1 | 已证实功能缺陷，收尾阻塞项 | 明确拒绝 feature-on 的 resumable/streaming-input，或完整实现 retained E 的 session 传递；补一条现有 CPU suite 回归 |
| 2 | R2，P2 | 已证实学习文档技术错误 | 修正 Graph tail 的安全性解释；不为文档增加清零代码 |
| 3 | D1–D4 | 文档同步与导航 | 将 R1 及支持边界写入交付入口，标清阶段快照，修复失效 symbol/行号 |
| 条件性继续 | O1–O3 | 性能分析或优化机会 | 只有继续追求对应性能目标时执行最小判别实验，不作为功能收尾前置条件 |

未确认 P0。质量保持尚未证明；按用户范围冻结已有质量评测，不扩样。

## 2. 审查方法、证据等级与环境

本轮完整阅读原始任务、四份 reference、六个交付入口及八份 source-analysis，
按需追踪 M0/M1/M2、Prefix、Graph、TP、DP 的报告、协议、脚本和测试。
审查覆盖原始设计对应的最终实现，未将最近一次提交或已有 PASS 当作免审依据。

本轮实际执行：

- 静态追踪真实请求、分配、forward、评分/选页、搬移、mapping 更新、完成、commit、free 和后续复用调用链。
- 只读读取原始 requests、serving、protocol、metrics、affinity、token IDs 和 allocator trace，独立重算正式性能结果。
- 以只读模式查询 NSYS SQLite；Graph API/GPU span 关联使用 correlationId 和进程身份，不仅按 kernel 名称推断。
- 使用真实 Scheduler/BlockPool 和既有完成结果 helper，最小复现 R1；临时脚本未另存实验目录，参数与观察输出记录在第 4 节。

既有真实 GPU 轨迹经过审查，本轮未重跑 GPU、完整实验矩阵或质量评测，未升级依赖、
下载模型、修改 reference 或学习基线。编写本报告前未修改 Runtime、脚本或测试。
报告落盘时仅新增此文档并在 delivery 增加导航。

证据分类如下，后续修复不得混写：

| 类型 | 含义 |
| --- | --- |
| 已证实 | 源码存在明确 contract 破坏，并有最小复现；或原始数据可独立重算 |
| 已有 GPU 证据支持 | 本轮检查源码与归档轨迹；本轮未重新执行该 GPU 场景 |
| 待验证假设 | 尚不能从现有证据拆出因果或收益，只提出最小判别方式 |
| 支持范围外/未测 | 已明确排除或缺少 Gate；不自动算作实现遗漏 |
| 性能负结果 | 功能正确，但对应配置未满足预算；不能改口径、删样本或升级为 PASS |

后续如需运行，沿用 `source .p1-work/scripts/env.sh`、`.venv/bin/python` 与 `uv`。
真实模型仅 `/data/models/Qwen3-8B` BF16；单卡 GPU1，TP2/DP2 使用 GPU0,1，
禁止 GPU2。GPU 实验经 `run_guard.py` 串行运行，44GiB 停止阈值及 45GiB 硬上限不变。
大型产物存放 `/data_ssd/qcrs/experiments/p1-v8`，不创建上游 PR。

## 3. 设计要求 → 实现 → 证据 → 缺口

下表源码链接指向工作树文件，symbol 是主要定位依据；行号只以本报告 HEAD 为准。

| 设计要求 | 最终实现位置 | 证据与判断 | 缺口或范围 |
| --- | --- | --- | --- |
| L 用于 logical position/RoPE/sampling，E 用于实际 allocation/slot/attention | [KVCacheManager.allocate_slots](../../vllm/v1/core/kv_cache_manager.py)、[prepare_pos_seq_lens / post_update](../../vllm/v1/worker/gpu/input_batch.py)、[KVReclamationWorker](../../vllm/v1/worker/gpu/kv_reclamation.py) | 普通 prefill/decode 与 post-forward E 推进完整；preempt 清 L/E 后走 recompute/APC | R1：保留 KV 的 streaming 重入错误沿用 fresh 初始化 |
| Whole-block row-only 与稳定 token compaction | [whole_block_row](../../vllm/v1/kv_reclamation.py)、[StableKVCompactor.compact](../../vllm/v1/worker/gpu/kv_compaction.py) | current-member 严格升序计划、真实 stride/int64 地址、payload bit-exact、释放页被新请求全层重写后旧请求继续 decode | 未发现新增缺陷；不外推任意 layout/dtype/backend |
| GPU 完成后才 canonical commit/free，并被后续请求真实复用 | [GPUModelRunner.sample_tokens / execute_model](../../vllm/v1/worker/gpu/model_runner.py)、[AsyncOutput.get_output](../../vllm/v1/worker/gpu/async_utils.py)、[commit_reclamation](../../vllm/v1/core/kv_cache_manager.py) | reclaim step 的 postprocess/copy/row/E 排在输出完成事件之前；maintenance 独立等待；Scheduler 持原 Block 对象并 free | CPU helper 只证明控制链；GPU 完成由既有真实轨迹和 stream/event 依赖共同证明 |
| allocation miss、候选、真实页收益、bounded headroom、一次 deferred retry | [Scheduler._select_reclaim_candidate / make_reclaim_request](../../vllm/v1/core/sched/scheduler.py) | 真 miss 接入；保留已有 partial batch；下一轮重试失败后 native fallback；R 按实际页差计算 | 不降低首次 full-prefill peak；不保证消除等待或所有 preemption |
| Prefix 共享保护、私有 suffix hash 失效、有损内容不重新发布 | `make_reclaim_request`、`KVCacheManager.cache_blocks / commit_reclamation` | mutation 前撤销私有 suffix 全 aliases；P 到最后 shared 整页末，覆盖其前 exclusive gap；真实 hit/reuse | fine-grained shared partial page 不在 Gate 内 |
| Graph 真 capture/replay，storage 稳定，下一 replay 消费新 E/row/slot | [ModelCudaGraphManager.capture](../../vllm/v1/worker/gpu/cudagraph_utils.py)、`KVReclamationWorker.prepare_attn`、[BlockTables](../../vllm/v1/worker/gpu/block_table.py) | FULL/PIECEWISE 实际执行；buffer 地址/capture count 与 released-page poison/rewrite/next-decode 有证据 | R2 文档错误；旧缺 span profile 保持 PARTIAL；不外推 FA3/4 |
| TP distinct-KV/GQA 评分、统一计划、所有 rank 完成后一次 free | `KVReclamationWorker.reclaim`、原生 multiprocess executor/monitor | score SUM → rank0 stable plan/broadcast → 本地 copy/row/E → completion collective → host ACK → canonical free；慢 rank/fail-stop 轨迹支持 | host sleep 注入不等于所有异步 GPU 慢 kernel 场景均已测试 |
| dense DP replica 状态和页池隔离，沿用 native routing/abort affinity | [EngineCore](../../vllm/v1/engine/core.py)、[DPLBAsyncMPClient](../../vllm/v1/engine/core_client.py)、本地 reclamation | 每 replica 独立 Engine/Scheduler/Pool/Worker；peer 不参与回收；真实 reuse/abort 后 peer 输出对照 | 无跨池借页、active KV migration；人工 skewed 不证明 native router 缺陷 |
| shared TriAttention 策略适配与质量/成本边界 | [TriAttentionScorer / protected_topk](../../vllm/v1/worker/gpu/kv_scoring.py)、[M2 评分报告](m2-scoring.md)、[质量报告](m2-evaluation.md) | 离线完整 GQA stats、logical round_start、稳定 ties 和共享计划已实现；85 条探索性质量结果保留 | 不是原 per-head 论文直接复现；质量保持未证明，不扩样 |

能力入口已拒绝 MRV1、async、多在途、spec、KV connector、PP/DCP/PCP、pooling、
restricted/non-causal attention、ALiBi 等不符合当前 contract 的配置。
`KVCacheManager.__init__` 另明确要求 `hash_block_size == block_size == 16`，
不能因 `validate_cache` 未单独检查 hash granularity 就判为误放行。
TP2×DP2、SWA/hybrid、prefill-time reclaim 等不从当前 Gate 推断支持。
本轮未发现需要重新选择核心技术路线的 DESIGN_CONFLICT。

## 4. 已证实问题与最低成本修复

### R1 / P1：streaming-input session 重入重置 retained E

**状态：待修复；功能收尾阻塞。**

触发条件：请求 `resumable=True`，已发生有损回收，暂停后保留 KV，再以同一
request ID 追加输入。该组合未在请求入口拒绝，原交付也未明确将其排除。
“不支持 prefill-time reclamation”不能豁免该问题：错误发生于已有有损布局的重入，
并不需要在 prefill 中执行回收。

源码链：

1. `vllm/v1/engine/async_llm.py:472` 有实际 `resumable=True` 入口。
2. `Scheduler.add_request`（`scheduler.py:2319`）接受请求和 session update，无 reclamation guard。
3. `_handle_stopped_request` 保留 KV 并转为 `WAITING_FOR_STREAMING_REQ`；`_update_request_as_session`（`:1470`）追加 prompt，保留 logical computed length。
4. `KVCacheManager.allocate_slots`（`kv_cache_manager.py:439`）按 retained E 和新增 tokens 算物理需求。
5. `Scheduler.schedule` admission（`scheduler.py:1069`）无条件执行 `request.effective_kv_len = num_computed_tokens`，将 E 恢复为 L。
6. `GPUModelRunner.add_requests`（`model_runner.py:815`）删除旧请求执行状态，`:838` 用 logical `new_req_data.num_computed_tokens` 初始化 Worker E。

本轮 CPU 最小复现使用现有
`tests/v1/core/test_scheduler.py::complete_pressure_step` 和
`tests/v1/core/utils.py::create_scheduler/create_requests`。
步骤可直接迁入现有 `pressure_scheduler` fixture：

1. 本地临时 `Qwen3Config(architectures=["Qwen3ForCausalLM"])`，MRV2；32 blocks、B16、batch256、APC off。
2. reclamation 设置 `pressure=True, min_keep_tokens=16, sink_tokens=0, recent_tokens=0`。
3. A：64 prompt tokens、`max_tokens=3`，在 admission 前令 `resumable=True`。
4. 完成 prefill；下一 decode 调 `make_reclaim_request(A, output, 32)`，交由既有 helper 完成真实 Scheduler/Pool commit。
5. 再完成一个 decode，自然达到 max_tokens，A 暂停。
6. 同 ID A 再提交 16 prompt tokens、`max_tokens=8`、`resumable=True`；只运行下一次 schedule 并检查 metadata。

观察输出：

```text
paused:
  status=WAITING_FOR_STREAMING_REQ, L=66, E=33, blocks=3
continued:
  status=WAITING, L=66, E=33, prompt_len=82
admitted:
  L_optimistic=82, E_reset=66, q={"A":16}
  row=[1,2,5,4], capacity=64
  worker_initial_E=66, worker_attention_length=82
```

原因是分配按 `33+16=49` 取得四页，而发布给执行侧的 E 已变成 66，attention
长度将按 82 使用。control contract 已破坏，可能使用 stale/reused row tail。
**本轮未执行该 GPU forward，不能把风险后果写成已观测的 GPU 越界或污染。**
helper 提供的是已完成 Worker result，不替代 GPU payload/completion 证据。

推荐收尾修法：在 feature-on 的可靠请求入口明确拒绝 resumable/streaming-input，
错误信息说明当前支持边界，在任何入队/分配/mutation 前失败；feature-off 保留原生能力。
该方案封住误放行，不新增 session 状态维护。

如果决定完整支持，则必须同时解决 canonical E 的跨 session 保留、new-request
metadata 传递和 Worker 初始化，区分 fresh/APC/recompute 与保留 KV 的重入。
只删 Scheduler 的赋值、只保留旧 Worker 状态，或无条件用 L 补足分配，都不足以证明语义正确。

验收条件：

- 拒绝方案：现有 Scheduler suite 一条行为回归，确认 feature-on 在改变 Pool/队列前清晰拒绝，feature-off 仍走原生路径；不新增 GPU 实验。
- 支持方案：上述 CPU 复现中 E 不丢失，allocation/row/slot/attention 长度一致；再补一条 Qwen3-8B continuation/reuse GPU 轨迹。
- 更新 delivery/status/final-review 的支持声明，明确区别 preemption 后 recompute 与 streaming session 重入。

### R2 / P2：Graph 学习文档错误描述清旧 tail

**状态：待修订；文档技术错误，不是已证实 Runtime 缺陷。**

位置：`source-analysis/graph-runtime.md:24` 写“回收覆盖 resident row 后清旧 active tail”。
实际 `vllm/v1/worker/gpu/block_table.py:107::append_block_ids` 只 stage 新 prefix，
并更新有效 block 数，没有清除 resident row 旧 tail。

误述会使读者把安全性归因于不存在的清零，并可能诱导增加无必要写入。
应改为：覆盖有效 prefix、缩小有效 block 范围，下一步 gather/slot/attention
受 E 与有效范围限制，旧 tail 不可寻址；padding row 的处理另行说明。

最低成本验收：静态对齐 `append_block_ids`、gather、physical slot/seq_lens 的 bounds，
引用现有 poison/reuse/next-replay 轨迹；不增加 payload/tail 清零，不重跑 GPU。

## 5. 性能证据核验与剩余归因

### 5.1 TP eager：0.868% 的限定结论足够成立

四个独立 server 均先小预热、再完整负载预热，各三轮正式计时；顺序为
full/noop/noop/full。8GiB/rank、short prompt128/256、arrival0/0.3s、各128输出、
eager、batch2048、maxseq8、APC off、CPU20/21。双方使用相同 AffinityWorker，
正式计时没有逐步 profiler/allocator trace。规范化启动参数后的差异符合策略/端口用途。

| 服务顺序 | 策略 | 三轮 tok/s | 均值 tok/s |
| --- | --- | --- | ---: |
| A | full | 63.262336 / 63.238094 / 61.029103 | 62.509844 |
| B | noop | 61.636810 / 62.827739 / 63.134194 | 62.532914 |
| B | noop | 62.610650 / 62.210189 / 60.590212 | 61.803684 |
| A | full | 63.875696 / 62.318695 / 62.552193 | 62.915528 |

按既定轮吞吐均值公式重算：

```text
native = 62.71268629764694 tok/s
noop   = 62.16829899758390 tok/s
overhead = 100 * (1 - noop / native) = 0.8680656693276778%
```

全部轮次保留；12 对请求 token IDs 相同、preemption=0、guard PASS。
实际 client arrival 偏差最大约 2.11ms，流式 chunk 为单 token。
这支持当前配置的样本内预算通过，没有挑最好组的证据。
只有四个 server、每策略六轮；请求数不能当独立服务级重复数。
不能宣称统计保证、生产尾延迟保证或默认无绑核普遍通过。

[AffinityWorker](../scripts/worker_affinity.py) 继承原生 Worker，
`super().init_device()` 后仅 `sched_setaffinity(0, {cpu})`；CPU 由显式参数按 local rank
选择并记录。CPU20/21 是机器配置，未成为 Runtime 默认值。
已有 helper 线程不变，之后新建线程可能继承；未隔离 SMT sibling、IRQ 或整物理核。
作为实验工具合理，不需要为此建设资源分配框架。

同服务交叉干预支持慢档随 CPU1 放置改变，不能进一步确定 driver、IRQ、SMT、
频率等微观机制。CPU1/49 底层慢因和默认放置波动可以保留，非功能收尾阻塞项。
本轮性能 followup 没有修改 Runtime，不能把新结果归因为新 Triton kernel。

### 5.2 TP 压力：剩余是 TTFT，尚不能只归因某个 kernel

三轮吞吐 101.408765 / 101.431062 / 99.911147 tok/s，达标数 6/6/5。
七个失败请求 TTFT 约 2.130–2.952s，TPOT 全低于 80ms。
平均吞吐 100.916991 tok/s；总输出除总时长约 100.911953 tok/s。
Goodput 是达标请求数除全部正式轮总时长，不是达标请求的 tokens/s。
两个吞吐聚合口径接近，但不能在报告中混用公式。

同 affinity 的 batch2048 control 每轮七次抢占，batch4096 pilot 十三次，
batch1024 formal 为零；batch512 pilot 也零抢占，却仅4/8达 SLO。
较小 chunk 的收益符合调度/容量冲突变化，不支持“batch 越小越好”或“越大越快”。
batch 改变同时改变回收时机和实际 retained 成员，不能推断质量等价。

独立 diagnostic 正式窗内有 19 次 miss/select/commit，累计归还 2054 blocks。
按首次 allocation 顺序和固定输入序列对照，后到达的长请求首次 allocation 前
约有 1.49/1.54s 等待，此后到首 token 又约 1.26/1.27s；长 prefill 约九个 chunk，
每步 token 预算被 running decode 分走。
内部随机 request ID 未完整保存 client-name 映射，这种对应依赖已知到达/输入顺序，
不能当作完整请求级因果 span。

已排除该轮 preemption 和毫秒级 client 到达偏差解释；已有证据支持入场等待与
chunked-prefill 链共同影响 TTFT。现有 allocator 时间戳不能进一步拆出 GPU prefill、
host 调度、通信、maintenance 等份额。O1 的最小实验用于解决这个缺口。

### 5.3 DP：局部池效果成立，人工偏斜不归因 router

新 skewed formal 三轮24/24，max TTFT 约1.74468s，max TPOT 约27.09ms。
replica0 的18个长请求和 replica1 的六个短请求分别达标，避免总吞吐掩盖 peer 尾延迟。
client 到达偏差最大约1.65ms。
固定6/2路由未改变，不能写成 native router 学会压力感知或 peer 空闲页支援本地池。

独立 diagnostic 正式窗内 replica0 有10次 miss/select/commit、累计归还1025 blocks；
replica1 没有回收。两池隔离证据成立。
历史 batch2048 的18/24继续 FAIL；新结果不能外推 Random、TriAttention、APC 或其他模型。

### 5.4 公平性与统计边界

- 新正式结果与 diagnostic/profile 分离，profiler 扰动数据未并入正式吞吐。
- pilot、失败轮和正式结果保留；没有发现以删除慢轮获取新 PASS 的证据。
- CPU affinity 是机器配置收益，batch/chunk 是调度参数收益，prepare/E-advance 融合是 Runtime 收益，compactor/scorer 调优是 kernel/评分实现收益；四者不能混称同一优化。
- 新压力配置的绝对 SLO 可判断；若要新增“同配置相对 native 的净收益”主张，需补相同 batch/affinity 的配对对照。目前不为未提出的主张补跑实验。
- 累计归还 blocks 可含同一页反复循环使用，不是唯一物理页数或同时新增容量；Pool free/reuse 也不等于 NVML 显存归还系统。
- 24 请求的小样本 P95/P99 是该样本描述，不是生产尾延迟承诺。

## 6. Triton、CUDA Graph 与优化优先级

### 6.1 Triton 已落地的部分

`KVReclamationWorker.reclaim → StableKVCompactor.compact` 使用 T16/W4/stages1，
真实 packed view、stride 与 int64 页地址参与计算。
严格升序 KeepPlan 保证 `keep[j] >= j`；一个 layer/head CTA 独占 payload，
tile 内先 load、`tl.debug_barrier` 再 store，跨 tile 顺序推进。
未发现当前执行映射的跨 CTA 读写覆盖。
不能为了增加 CTA 数直接把 token tile 任意并行化，否则原位依赖证明失效。

M1 micro 的32K half-retention，T16约3.669ms、T32约3.634ms，但寄存器56对122；
4K两者约0.458ms，W8反而更慢。保留 T16 有依据，未见值得新增 shape dispatcher 的强反向取舍。
compiler spill 为0；NCU 观察到288 CTA、56 registers、occupancy约16.77%、DRAM约63%。
低 occupancy 本身不是缺陷，NCU duration 也不与 CUDA Event median 混算。
TP 每 rank 只有144 CTA，不能从单卡288 CTA 推断所有实际 shape 都最优。

M2 完整 shared-scoring plan micro：4K约6.149ms，32K约8.436ms。
phase 已提到每计划一次，但36层循环仍逐层执行 `var_mean/std/sub/div/GQA amax/mean/add`，
有临时 tensor 和多 launch。
micro 的 workspace max_len 与真实 Runtime max_model_len stride 不同，TP local head 数也不同；
优化方法确实落地，不代表真实全部 shape 的参数已最优。
`protected_topk` 的稳定 tie 和成员升序是 contract，不能直接换成不稳定 top-k。

### 6.2 CUDA Graph 与 TP profile 核验

模型 graph 引用持久 physical seq_lens；row/E 变动更新原 storage 内容，
回收位于图外 post-forward/maintenance 边界。下一次 replay 前走真实 metadata builder，
不是仅替换 Python object。prepare 不推进 E，postprocess 才推进；padding 不算真实 q。
已审查的重复回收、released-page 重写与 next-replay 轨迹支持该生命周期。

以 `metrics-2-after` 到 `metrics-3-before` 为计时窗，独立查询 SQLite：

| 窗口 | 重算结果 | 含义 |
| --- | --- | --- |
| Graph Sink，9.893113s | 274 API / 274 graph spans，0 missing；16次 compact合计7.080359ms | 实际 replay/压紧存在；仅消除 compact 的窗口占比上限约0.072% |
| 同 Graph 窗口 | prepare308次共0.804062ms；post_update308次共1.885468ms；活动 union 8540.978045ms | 融合进入真实 Serving，旧独立 physical prepare/E advance kernel 未出现；空隙不能全部当可优化 Runtime 成本 |
| TP Tri，17.483842s | 424 API / 424 graph spans，0 missing；每 rank score/plan/completion各27次，共162个collective span | 该窗口 graph/collective 关联完整，不追溯升级旧 PARTIAL |
| 同 TP 窗口 | 每 rank compact共8.687/8.678ms，56 registers、localmem0；972个raw-score kernel约10.978/11.007ms | raw kernel 时长不足以代表完整评分/回收成本 |

TP rank0 的 score/plan/completion NCCL 累计约0.429/0.218/5.026ms，
rank1 约217.364/17.219/56.698ms，包含 rank 等待，不能当纯链路传输时间。
outer reclaim host NVTX 累计 rank0/rank1约1.527/1.352s，包含发射、同步等待和 profiler
扰动，不能直接减掉 raw kernel 时间就声称剩余全可优化。
该 Serving profile 分支没有安装 payload snapshot hook；trajectory 诊断分支另行处理。

FULL Graph 与 eager reference 的比较需要相同 captured max_seq_len scalar；
actual-E scalar 引起的 BF16数值路径差异已保留为负对照，不是放宽容差。
当前证据没有表明需要因 E 改变 recapture，也不支持把 compaction 纳入模型 graph 重写。

### 6.3 后续优化机会排序

下面是优化机会，不是额外确认的功能 findings。无法估计收益的项目明确不填百分比。

| 编号/顺序 | 方向 | 证据可信度与预计收益 | 成本与最低判别方式 | 停止条件 |
| --- | --- | --- | --- | --- |
| O1，优先但有条件 | TP pressure TTFT 的 admission/prefill/reclaim 分项 | 已确认TTFT失败及入场等待，细分因果未确定；收益不能估计 | 只做当前配置一轮，关联 client→enqueue→first schedule→prefill completion→reclaim completion，区分host/GPU/通信；不先扫batch | 若接受该配置性能FAIL，不阻塞功能收尾，可停止 |
| O2，其次 | TriAttention 的 normalization/reduction/launch 局部融合 | micro和Runtime源码支持多launch成本；Serving净收益未量化，不能用于解释当前Sink压力 | 先测一次真实Tri回收shape的完整host/GPU分项，复用工作区或局部融合需保持数值/tie contract | 占比不足或成本高于收益时保留现状，不重写整个scorer |
| O3，低优先级 | Graph/普通decode的 `prepare_attn` host metadata构造 | 每步 `np.zeros`、`dataclasses.replace`、upper-bound tensor确实存在；诊断约数十μs，当前TP预算已过 | 先确认稳态CPU占比；若值得，复用CPU上界buffer，保持有效范围与padding语义 | 无代表性收益则停止，不新增noop旁路 |
| O4，可停止 | compactor继续调tile/warp或增加backend | 当前Graph窗口compact仅约0.072%；参数已有实测，无收益放大依据 | 只有新workload证明其占比变大才重开，不能仅因occupancy低启动重写 | 当前无需继续 |
| O5，可停止 | DP routing hint、跨池借页或迁移 | 固定6/2路由不证明native router缺陷；现有局部隔离正确 | 仅当native routing反复出现可复现偏斜且有独立对照收益时重开 | 当前无需新增Runtime逻辑 |

默认无绑核慢档的 CPU1/49 微观解释仍是待验证事项，但现有证据足够限定报告，
不应机械重跑旧6.871%或继续无限 profile 来争取“所有配置通过”。

## 7. 工程结构、校验与测试取舍

Scheduler 保持 canonical Request E 和 Block 对象，Worker CPU/GPU E 是执行镜像，
不是无依据重复状态；普通每步重复 E 字典传输已删除。
回收专用 source/ownership 检查位于破坏性状态交接处，不在每次 decode 额外通信。
GPU mutation 之后失败采用 fail-stop，没有用 fallback/retry 隐藏半提交状态。
未发现需要大重构、通用 transaction/version 框架或额外 DP barrier 的依据。

| 校验/事件 | 保护对象或成本 | 建议 |
| --- | --- | --- |
| 初始化的支持范围、layout、ALiBi、stats/RoPE检查 | 明确能力边界，阻止语义不兼容；低频 | 保留 |
| mutation前私有suffix与hash保护 | 防止共享页被写、hash重新命中有损内容 | 保留，不能移到mutation之后 |
| source E、source row、结果身份、retained唯一性/子集、P与refcount | Worker→canonical commit的关键contract | 保留；现有稀疏回收频率不足以支持为性能删除 |
| TP completion collective + AsyncOutput/maintenance等待 | 全rankGPU写完成后才free/reuse | 必须保留，不能用rank0返回或仅enqueue替代 |
| Graph active bounds与padding | 旧tail、padding和复用页不可观察 | 保留；R2修文档而非新增清零 |
| Worker initialize_cache与compactor构造中的部分BF16/ndim重复检查 | 同一packed-view contract在初始化重复表达 | 可统一owner，但收益很小，不作为专门优化任务 |
| `local_reclaim_done` | 生产只record，诊断读取 | 可移到诊断wrapper；保留真正TP/host完成边界。收益未量化 |

测试判断：

- CPU oracle、kernel fixture、真实 Scheduler/Pool、GPU completion、真实复用与对照 attention 证明不同层次，不因数量多就删除。
- identity/shift-one/alternating/recent 和 stride1/2 覆盖不同重叠/地址风险，没有发现应立即批量删除的重复矩阵。
- `complete_pressure_step` 的模拟完成结果不能验证异步；现有GPU证据对此有补充，R1用它验证canonical控制缺陷是合适层级。
- TP慢rank用host sleep延迟compact enqueue，证明提前完成rank不能free；不冒充真实长异步kernel的普遍压力验证。
- 单卡copy enqueue后故障与TP非output rank在copy前故障覆盖不同边界；当前无需为了形式完整重新排列所有组合。
- 最值得补的是R1的一条现有CPU suite行为测试。只有选择完整支持streaming，才增加对应真实GPU continuation轨迹。
- 不因完整上游suite离线模型配置资源缺失而安装依赖或下载模型；不以新测试数量替代核心失效模式覆盖。

85条质量样本仅探索性；中文QA四策略0/2，不能证明质量保持。
token一致性和给定KeepPlan的payload正确性也不能替代有损质量保证。
保留冻结状态，不新增质量实验。

## 8. 中文学习文档与追踪文档修订

| 编号 | 位置/问题 | 具体修订与验收 |
| --- | --- | --- |
| D1 | `reports/final-review.md`称prefill/resume沿原生初始化，未区分两种resume | 链接R1；区分preempt/recompute与保留KV的streaming重入；修复后再写实际支持结论 |
| D2 | `source-analysis/m2-pressure.md`仍写候选页全私有、intent使用SinkRecent，属于阶段快照 | 标注阶段/HEAD，链接最终Prefix的P推导与多policy路径，避免读者当成最终通用实现 |
| D3 | `source-analysis/dp-runtime.md`等硬编码行号漂移，如kv_reclamation.py:122 | 以symbol为主，行号绑定固定HEAD；修实际追踪断点，不批量润色全篇 |
| D4 | delivery/status及plan/report存在内容复制，容易漏同步 | delivery保留当前结论和导航，status保留当前未决项，reports保留历史证据/负结果，notes保留协议，source-analysis解释当前因果链；plan不再复制数表 |

R2优先于一般文风整理。已有14.33%/6.871%历史FAIL与0.868%限定PASS的解释基本清楚，
不是互相矛盾；不能删掉旧失败。旧缺span的PARTIAL不因新窗口完整而改写。
阶段PASS与配置性能FAIL应分别说明，不能把执行路线到DP等同于所有预算通过。
不批量改写历史英文产物，不另建一套重复学习文档。

## 9. 修复验收清单与停止范围

以下复选框只跟踪后续工作，不表示本报告已实施修复：

- [ ] R1：在明确入口拒绝或正确支持streaming重入；记录采取的契约及修复提交。
- [ ] R1：现有Scheduler suite最小行为回归通过；若完整支持，再补单条真实GPU continuation/reuse。
- [ ] R2：修正Graph tail说明，与源码及现有poison/reuse证据一致，不增加清零。
- [ ] D1：同步delivery/status/final-review的支持声明，区分两种resume。
- [ ] D2–D3：标明M2阶段快照并修复关键symbol/行号导航。
- [ ] 若继续要求TP pressure达SLO：先完成O1的一轮分项诊断，再按证据选修法；否则明确冻结17/24 FAIL。

R1封口、相关文档准确后，可以按已经验证的MRV2/BF16/FA2/NHD/B16、单group、
单在途和实际Gate组合做功能交付。无需以全部性能负结果消失或质量保持证明作为
本轮额外扩张任务，但交付必须继续明确这些限制。

当前可以停止：完整矩阵重跑、质量扩样、默认CPU慢核微观机制追到底、compactor无依据重写、
DP跨池架构、新backend和未测组合扩张，以及以删除关键校验换取性能数字。
后续只在新证据或明确新目标出现时重开。

## 10. 证据索引

原始产物优先于汇总。下列绝对路径为本机SSD归档；迁移报告时需一并保留归档定位。
本轮未执行会覆写归档summary的汇总器main。

| 证据 | 原始入口 |
| --- | --- |
| 原始任务与环境约束 | [prompt1](../prompts/prompt1.md) |
| 主设计 | [Physical_KV_Cache_Reclamation_Design_v8](../reference/Physical_KV_Cache_Reclamation_Design_v8.md) |
| 技术选型与实验矩阵 | [技术选型](../reference/P1-v8-实验设计与技术选型方案.md)、[实验矩阵](../reference/P1-v8-实验矩阵与兼容性扩展方案.md)、[项目评审](../reference/Physical_KV_Cache_Reclamation_Design_v8-简历项目评审.md) |
| 原有阶段报告与交付导航 | [delivery](delivery.md)、[final-review](final-review.md)、[performance-followup](performance-followup.md)、[status](../plans/status.md) |
| TP eager ABBA四服务 | [full A](/data_ssd/qcrs/experiments/p1-v8/Perf/20261031-153455-tp-eager-affinity-a-full/serving.json)、[noop A](/data_ssd/qcrs/experiments/p1-v8/Perf/20261031-153746-tp-eager-affinity-a-noop/serving.json)、[noop B](/data_ssd/qcrs/experiments/p1-v8/Perf/20261031-154104-tp-eager-affinity-b-noop/serving.json)、[full B](/data_ssd/qcrs/experiments/p1-v8/Perf/20261031-154349-tp-eager-affinity-b-full/serving.json)；各目录另有requests-0/1/2、protocol、invocation、metrics和affinity |
| TP压力正式三轮 | [serving](/data_ssd/qcrs/experiments/p1-v8/Perf/20261031-155620-tp-pressure-batch1024-formal/serving.json)及同目录requests/protocol/metrics |
| DP skewed正式三轮 | [serving](/data_ssd/qcrs/experiments/p1-v8/Perf/20261031-155803-dp-skewed-batch512-formal/serving.json)及同目录requests/protocol/metrics/routing |
| TP压力独立diagnostic | [目录](/data_ssd/qcrs/experiments/p1-v8/Perf/20261031-155951-tp-batch1024-diagnostic)；只截正式HTTP时间窗，不混入预热 |
| DP压力独立diagnostic | [目录](/data_ssd/qcrs/experiments/p1-v8/Perf/20261031-160109-dp-batch512-diagnostic)；分别读取replica事件 |
| 新Graph完整profile | [trace.sqlite](/data_ssd/qcrs/experiments/p1-v8/Review/20261031-145135-graph-sink-nsys-flush0/trace.sqlite) |
| 新TP完整profile | [trace.sqlite](/data_ssd/qcrs/experiments/p1-v8/Review/20261031-144831-tp-full-tri-nsys-flush0/trace.sqlite) |
| compactor参数micro | [benchmark.json](/data_ssd/qcrs/experiments/p1-v8/M1/20261030-234426-stable-stride-specialized/benchmark.json)、[NCU CSV](/data_ssd/qcrs/experiments/p1-v8/M1/stable-ncu-optimized.csv) |
| shared scorer micro | [scoring.json](/data_ssd/qcrs/experiments/p1-v8/M2/20261031-002705-shared-scoring-profile-phase-hoist/scoring.json) |
| R1 CPU复现 | 第4节记录完整fixture参数、步骤和输出；未归档独立run，不属于GPU证据 |

Perf正式实验source_head为`4ad1a03e1c`；截至本报告HEAD的后续差异为实验工具/配置交付，
没有把未在该正式run执行的Runtime优化补记为性能收益。
将来修改实现后，修复结果应另记新提交和实际验证范围，本报告作为发现时的基准保留。
