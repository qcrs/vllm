# Protected Controller：进展与复制成本修正

本轮从 `e3550ac916` 继续。只用用户指定的物理 GPU1/2，Worker 主线程 CPU20/21；
其余 prompt3/4 环境、资源、归档与 Git 约束继承。历史 FAIL/PARTIAL 保留。

## 根因与实现

旧实现覆盖当前 allocation miss，却不覆盖完整已知 prefill，也不扣除 victim 自己的
下一步尾页分配。前方 decode 可拿走释放页；仅使第一次 retry 成功，仍可能接纳
无法完成 prefill 的请求，随后反复抢占重算。

新逻辑仅在 opt-in protected pressure 路径启用。等待 admission 检查完整已知 prefill，
给已接纳 prefill 保留尚未分配的容量；预算同时覆盖 post-forward 后运行请求和等待者
的已知 prefill/下一 decode。物理需求使用 E，逻辑保护 L 仍只包含已计算上下文。
复用真实 allocator 的 full-sequence gate/reserved-block 参数，不新增页所有权状态。

对 N 个源页、保留 K、下一步 t 个 token、原分配需求 A，净容量贡献为
`N - ceil((K+t)/B) + A`。选择单/双 victim 先覆盖净缺口，保护单次删除≤20%、逻辑
保留≥50%；无合法覆盖就等待/原生抢占，不新增第三个 victim，不预测未知输出。
这避免把“释放页数”误当成“等待请求可消费的净页数”。

SinkRecent 在同净容量目标下寻找邻近 exact whole-page remapping，最多额外删除
B−1 个 token，仍受原保护 floor 约束。例如 E4100、D1，K4096 需复制，K4084 可以
重映射。尾页场景可能物理多释放一页并由 victim 自身消费，净贡献不变。
双 victim 的逻辑保留率变为分段单调，用二分交点相邻 split 选择并与独立 token
枚举 oracle 比较。这一取舍不等于质量无损，85条质量样本未扩大。

fixed/requests 仍默认。未更改 TP completion、canonical commit/free 或 DP 路由契约。
DP correctness harness 恢复严格的 abort 前进展与等待者改写复用要求。延迟 probe
移至 reclaim 入口以覆盖 token/whole 两条路径；首次 pair 的 probe 漏注入 FAIL 保留。

## 正确性证据

大产物根目录为 `/data_ssd/qcrs/experiments/p1-v8`。

- B0：`Engineering-Progress-B0/20261107-231330-initial`。
  相关 KV/Pool/Scheduler CPU suite **86 passed / 289 deselected**；本地 tokenizer
  fixture 的 DP suite **15 passed / 36 deselected**。开发 FAIL/setup FAIL 保留。
  适用 pre-commit 与 manual mypy-3.12 通过，日志见 B0。
- `Engineering-Progress-Correctness/20261107-234132-single-q0-tp2-graph-reuse`：
  PASS，两个 rank 均 continuation、Graph/eager reference、FP32 attention 和
  release/rewrite/reuse；回收两次，最终 free32。
- `Engineering-Progress-Correctness/20261107-234635-pair-tp2-graph-delay-reuse-fixed-probe`：
  PASS，首批两 victim（K223 copy、K126 whole），第二 rank 第二 victim 延迟≥2.5s
  时 canonical 不变；全 rank 完成后 ACK、commit/free，随后 Graph 与真实改写复用。
  共7批、8个 victim，最终 free32。前次 probe FAIL 目录保留。
- `Engineering-Progress-Correctness/20261107-234729-dp2-protected-progress-reuse`：
  PASS，独立 Pool33；C 在 abort A 前输出，并改写本地释放页 `[2,18]`；peer 没有
  回收，abort 后仍有112次经 payload 核对的 forward；最终两个 Pool 均 free32。

B0 的 `audit-correctness.py` 与 `correctness-analysis.json` 独立核对上述源码 hash、
GPU映射、guard、affinity、cap、rank计划一致、completion时序、Graph与改写复用。
q0 极短窗口不要求 observer 采样成功；延迟批次必须实际观测到 canonical 保持不变。
correctness PASS 不能解释为速度、SLO 或模型质量 PASS。

micro 与 Serving 见后续统一评估；不能用 correctness 代替性能结论。

## 有界合并与最终实现

`533fa38ed5` 在上述修正后补充有限 completion headroom。只在原硬缺口已经存在合法
单/双 victim 计划后，才在同一 victim 集合中合并近期小额回收；不增加 victim，不对
不可行缺口先做部分删除。每个请求的容量上界为
`min(prompt_tokens + max_tokens - 1, max_model_len - 1)`，减去 post-forward 已计算量
及硬预算已覆盖量，再向上取页。最终目标不超过选中 victim 的合法净容量。

该上界仅用于容量预算，EOS 可能提前终止；它不进入已计算逻辑 L，也不改变 DP
known_work 中的已知工作量 U。没有新增权重、搜索参数或页所有权状态。20% 单次
cap、50% 逻辑 floor、prefix/min_keep/sink/recent 和全 TP completion 后才 commit/free
继续成立。最终源码入口为 `Scheduler._reclaim_retry`、
`Scheduler._protected_prefill_reserved_blocks`、`Scheduler._select_reclaim_plan`、
`KVCacheManager.allocate_slots` 和 `protected_reclaim_plan`。

## 最终正确性

最终 Runtime 的 B0 为 `Engineering-Amortized-B0/20261108-014007-initial`：相关 CPU
suite **99 passed / 289 deselected**，DP 本地 fixture 为 **13+2=15 passed**；后者不是
完整上游 DP suite。适用 pre-commit 和 manual mypy-3.12 通过。最终测量入口的静态
检查另见 `Engineering-Progress-Final-B0/20261108-021957-resume`。

最终真实 GPU 验证位于 `Engineering-Amortized-Correctness/`：

- `20261108-015330-single-q0-tp2-graph-reuse`：两 rank 计划一致、continuation、
  FP32 attention/Graph-eager reference、释放页改写复用 PASS，最终 free32。
- `20261108-015400-pair-tp2-graph-delay-reuse`：单批两个 victim，rank1 第二 victim
  延迟至少2.5s时 canonical 不变，全部完成后 ACK/commit/free；Graph 和真实复用
  PASS。最终5批、6个 victim，最终 free32。
- `20261108-015435-dp2-progress-abort-reuse`：C 在 abort A 前已输出并改写本地
  释放页 `[14,15,16,17]`；peer 无回收，abort 后还有112次 payload 验证 forward；
  两个池最终均 free32。严格的等待者进展要求通过，没有再靠 abort 放行来验收。

独立审计为上述 B0 的 `audit-correctness.py` / `correctness-analysis.json`。开发阶段
失败和旧报告中的失败保留，见[收尾报告](engineering-progress-closeout.md)。

## 最终性能与剩余容量瓶颈

同源码、同仪表、同 GPU1/2 的三轮正式 TP2 对照：fixed 为101.548tok/s、18/24 SLO，
protected 为73.359tok/s、6/24，吞吐仍 **−27.76%**。两边均零抢占、零重算；protected
回收18次、fixed56次，但 protected 有1551次 allocation miss。miss 统计包含同一等待
请求反复重试，不能解释成1551个请求失败。正式最大 victim 都为1。

独立 trace 位于
`Engineering-Progress-Final-Diagnostic/20261108-022023-tp2-protected-capacity-gpu12`，
复算在最终 B0 `capacity-analysis.json`。514次 protected 计划中508次为空，独立按
候选净容量复算后，**508次全部是合法容量不足**。首例池681页，8192-token 入站请求
需512页、当前 free424页，缺88页；唯一4097-token victim 的 floor3278，只能净释放
52页，仍缺36页。其他 running 的下一步需求为0，排除了该例前方 decode 抢页或预算
重复计算。四个长请求各被拒绝127–128次；首次失败检查至实际分配约1.40–1.60s，
这不是请求总排队时间。

trace 的三次大回收均 source8193、floor6555、K6559、释放103页；邻近 zero-copy
K6545 低于 floor，不能合法选用。三次小回收各释放3页，已全部 zero-copy。因此既
不能继续把剩余退化归因于每次都复制，也不能把它归因于双 victim 扫描。

中间版本 trace 曾有48次 timed commit、45次只释放1页；合并后最终正式仅18次回收。
这说明小额维护频率已被修正，但没有解决保护约束下长请求的入场容量和排队问题。
已触发[有界合并计划](../plans/engineering-amortized-plan.md)的停止条件：停止继续
调参，保留 fixed 默认及本轮 TP 负结果。继续研究应先明确容量/排队或质量与删除比例
的取舍；反复空回收、放宽 SLO、换 workload 或增大 pool 均不能作为本轮收益。

完整 micro、Serving 配置和归因见[统一评估](engineering-progress-evaluation.md)。
