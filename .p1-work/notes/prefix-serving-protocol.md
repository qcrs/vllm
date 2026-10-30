# Prefix HTTP Serving 预先固定协议

## 问题与范围

Prefix 正确性已在 bf31e42234 通过，本实验补齐 APC on 的真实 HTTP 压力与成本证据。
验证共享前缀保护之后，私有 suffix 回收能否减少 preemption 并兑现为请求延迟/吞吐收益。
仍为 Eager、TP1/DP1、单在途，不进入 Graph；没有诊断 payload hook 或每步 GPU 同步。

复用 `.p1-work/scripts/serving_reclamation.py`，增加 `--prefix-caching` 模式；
保留 M2 的 APC off 模式。模块输入是固定 token IDs、arrival、Pool 与策略，
输出是流式 token 收取时刻、原生 usage/counters 与原始响应。
主要防止跨轮 APC 累积、不同策略请求不一致、冷编译当稳态、trace 开销混入正式计时。
最便宜的有效验证为一条真实 HTTP diagnostic，随后运行相同无 trace 矩阵；
不为脚本 wiring 新增重复 unit suite。

## 环境与固定参数

- Qwen3-8B BF16，`/data/models/Qwen3-8B`，物理 GPU1。
- vLLM v0.26.0/MRV2，Python3.10、torch2.11.0+cu129、Triton3.6.0；现有环境与模型。
- FLASH_ATTN，B16，APC on，async/cascade/reserve-full-ISL off，无 connector/spec。
- max_model_len=9216，max_num_batched_tokens=2048，max_num_seqs=8。
- KV Pool 请求值固定为 6、3、1.5 GiB，分别作为低/中/高压力探针；
  实际是否发生压力只以 miss/preempt/reclaim 证据判断，不能由标签推断。
- Native Full、retention=1 noop、SinkRecent retention=0.5 三组；
  pressure on，reserve_cap=0，sink/recent=16/16，private min_keep=128。
- 所有 GPU 运行经现有 run_guard 串行执行，44 GiB 终止阈值、45 GiB 项目硬上限。

## 输入、预热与计时

每轮8请求，prompt 长度 `[4096,8192]*4`，共享最前256 tokens。
紧接共享域的单 token 使用8个不同数字，确保每个私有 suffix 的 APC hash chain 分离。
输入为已有合成英文片段重复，并非质量评测；保存完整 token IDs 与各自 SHA256。
每请求 temperature=0、seed=0、ignore_eos=true，固定生成128 tokens。
arrival 秒数 `[0,.3,.6,.9,5,5.3,5.6,5.9]`，每个 server 正式重复3轮。

先做32-token prompt/8-token output 预热，再做一整轮同负载预热，均排除正式统计。
在完整预热与每轮正式请求前，通过 loopback-only 的原生 `/reset_prefix_cache` 清空 APC；
要求 success=true，不自行重试或重置活请求。Dev router 只为该实验启用。
全负载预热不能保证所有 K/P 变体均已编译；server log 若仍出现 JIT，按实际标注。

TTFT 从请求发起到首 token_ids chunk；TPOT 为首/末 token 间隔除127；
ITL 来自客户端流式收取，相同 chunk 的 tokens 使用同一时刻。
wall time 从本轮 epoch 到全部响应完成，包含固定 arrival 间隔、等待与回收。
原生 streaming usage 记录每请求 cached_tokens；原生 metrics 每轮前/后保存并求差。
per-request usage 来自首个输出；prefix query/hit counters 不包含 preempt 后的查询，
也不等价于成功 admission 次数。源码 `PrefixCacheStats.record` 将 preempted 查询单独记账，
`PrometheusStatLogger.record` 仅向公开 counter 加 queries/hits。
computed prompt 指标解释依 pinned source，不能以 logical prompt 减一次 hit 冒充全部重算。

## 判定与证据边界

沿用 M2 预先固定 SLO：TTFT≤2秒且TPOT≤80毫秒。
报告 TTFT/TPOT/ITL P50/P95/P99、output token/s、request/s、SLO goodput，
每轮分别报告，3轮共24请求的 P99 只作为小样本描述，不称生产 tail 估计。
低压力 noop 的吞吐开销预算为相对 Native≤5%；需3轮均值比较，同时保留每轮值。
有损组只有同 Pool 的 SLO-good 请求/秒提高且延迟预算满足，才可称该配置性能通过；
只改善吞吐或 preempt 但仍超过 SLO 时，明确记录性能未通过。
这里不评估有损回答质量，不能据合成负载建议默认启用；M2 质量结论不扩展为 Prefix 质量通过。

先以1.5 GiB/SinkRecent 进行 trace diagnostic（1轮正式 + 1轮预热），
验证真实 miss、protected P、commit/R；若不触发，保留负结果再解释。
正式9组关闭 `VLLM_KV_TRACE_FILE`，不使用 diagnostic 延迟作正式结果。
诊断里的 reclaim 次数/R 只能归属于该条运行，正式组以原生 counters 为准；
没有生产 reclaim counter，不为实验扩张 Runtime instrumentation。

大产物保存于 `/data_ssd/qcrs/experiments/p1-v8/Prefix/`，每次包含命令、commit/diff、
脚本 snapshot、protocol、prompt IDs、server log、requests、metrics、NVML与退出码。
协议在首条新 GPU 实验前固定；后续修正必须保留旧失败与说明，不能事后放宽 SLO。

下一步只在已有 Prefix 授权内补性能证据；阶段转换问题仍 pending，Graph/TP/DP保持 NOT_RUN。

## 运行中补充：无压力基线

`20261031-024846-http-apc-6-full` 的3轮均出现2次 native preemption，
因此6 GiB实际是有压力区间，不能用来判断无压力 noop 开销。
保留原9组和固定SLO/输入/arrival；额外固定8 GiB下的Full/noop/SinkRecent三组，
仍每组1轮完整预热和3轮正式测量。8 GiB理论可容纳约58K tokens，
大于全部8请求49,152 prompt + 1,024 output，最终无压力仍以counter实测判断。
此项在首条8 GiB GPU运行前记录，不更改或丢弃6 GiB负结果。

统计口径补充：`Scheduler.schedule` 只在 `num_preemptions<=0` 时初始化
PrefillStats，`take_prefill_stats` 仅向首个prefill输出交付一次；
因此 `prompt_tokens_by_source_total{source="local_compute"}` 不能衡量重算总量。
本矩阵正式组只报告该原生首次prefill指标与preemption次数，实际重算token数量保持未测；
既有Prefix正确性轨迹中第三请求多算176 tokens的对照仍独立有效。
早期diagnostic的protocol中cached_tokens注释误写“包含re-admissions”，
该原始artifact保留，以上源码核对和汇总里的correction字段覆盖其文字解释，不改raw数据。

## 运行中补充：诊断重算账本

现有host trace足以在本配置重建forward token工作量，无需修改Runtime计数器。
对每次 WAITING/PREEMPTED allocation_done，B16/单full group/no lookahead保证
`hit = 16*owned_blocks - 16*ceil(q/16)`；APC命中只含完整页。
直到下次free_before，`settled_logical_tokens - hit` 为该ownership周期完成的forward
token数。单在途、先完成再canonical free，使此边界不包含尚未完成的计划q；
同一步已allocation但被取消的q不会被加进实际L。
对每请求所有周期求和，再减首次prompt工作和127个decode输入，得到额外forward工作；
这包含preempt后重新计算的prompt和decode tokens，不冒充单独prefill重算量。

已有SinkRecent trace正式轮重建13个周期（8请求+5次preempt），
共48,419 forward tokens；无重算参考为49,152−1,792+8×127=48,376，额外43 tokens。
另补1.5 GiB/Native诊断：相同协议、1轮完整预热+1轮正式，trace on；
只用于preempt/recompute账本，不混入正式12组性能统计。先前SinkRecent trace保持不变。
账本脚本需校验周期闭合、首admission hit与usage合计一致、preempt数与周期数一致、
完整输出长度和L终值一致。该结果限B16/单group/单在途的本实验，不能扩展为通用trace分析器。
