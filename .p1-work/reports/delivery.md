# P1 v8 工程交付索引

工作树与分支固定为`worktrees/p1-v8-physical-kv-reclamation`、
`p1/v8-physical-kv-reclamation`，基线vLLM v0.26.0 `568afb3a13`。
所有真实模型实验为本地Qwen3-8B BF16；单卡使用物理GPU1，TP2/DP2使用0,1。
现有Python3.10/Torch2.11+cu129/Triton3.6与工具链保持不变。大型产物只在
`/data_ssd/qcrs/experiments/p1-v8`，各run保留命令、源快照/提交、参数、日志、
NVML和退出码；失败产物未覆盖。

## 已实现的运行语义

logical L用于RoPE、sampling和原token/hash chain；physical E用于attention
长度、KV slot和allocator实际需求。回收只选择已完成prefill的decode请求，
统一current-member升序KeepPlan，所有层/head使用同一保留集合。整页计划只改
row，token计划由Stable CTA Triton原地压紧。Worker完成后Scheduler才提交
canonical row/E并将removed Block对象归还原生Pool；不会以NVML下降代替释放证据。

allocation miss连接到确定性候选、真实页收益、bounded headroom与一次deferred
retry；无可行收益后沿用native fallback。SinkRecent、Random与TriAttention-inspired
shared scoring均集成真实Runtime；Tri使用离线完整GQA query-head统计，不能称为
原论文per-head方案复现或无损压缩。

Prefix保护完整共享页，mutation前撤销私有suffix全部hash aliases；Graph使用
持久row/E缓冲与图外回收。TP2先全局distinct-KV评分、rank0统一计划，各rank本地
压紧后经过全组完成边界才释放。dense DP2保留两个独立Pool与原生请求affinity，
空闲peer页不计入本地预算，不迁移active请求。

## 证据入口

| 阶段 | 证据与结论 | 报告 |
| --- | --- | --- |
| M0 | 环境/数据审计、native baseline、真实allocator与CPU scalar oracle，28 tests | `m0.md` |
| M1 | 真实整页/token释放与重写、重复回收、EOS/abort/q0/maintenance/preempt/fail-stop，42 tests | `m1.md` |
| M2 | 真实miss/一次重试/回收、离线校准与shared scorer、85样本质量与HTTP压力评测 | `m2-pressure.md`、`m2-scoring.md`、`m2-evaluation.md` |
| Prefix | 完整页APC保护/取消发布/真实hit及reuse，独立正确性PASS；12组Serving按配置判定 | `prefix.md`、`prefix-serving.md` |
| Graph | 七条8B轨迹、FULL/PIECEWISE与Prefix组合，独立正确性PASS；17组Serving | `graph.md`、`graph-serving.md` |
| TP | 11条8B轨迹、真实NCCL distinct-GQA/stable ties与23项回归PASS；12组Serving | `tp.md`、`tp-serving.md` |
| DP | 双Engine局部压力/重写/abort与peer逐token对照PASS；10组正式Serving与5条独立诊断完成，含宽松池零评分/回收链 | `dp.md`、`dp-serving.md` |

性能优化使用CUDA Events、NSYS与NCU。固定36层/8 KV heads/D128/B16的32K
half-retention compaction约18.39→3.66ms，寄存器128→56，无compiler spill；
32K完整shared scoring plan约184.14→8.44ms。仅描述相同shape的microbenchmark，
不外推为Serving加速倍数。资源/occupancy/grid与短shape限制见上述原始报告。

## 未通过的预算与支持边界

质量评测为85条探索性配对样本。Full/Sink/Random/Tri target-contained分别
45/41/39/44，不足以证明质量保持；中文QA四策略均0/2。正式HTTP是合成压力
探针，不把有损后tokens/s提升当作等量计算或生产质量保证。

Prefix的3/1.5GiB、M2的1.5GiB尾延迟未满足既定SLO。TP的short/eager noop
吞吐开销14.33%，超过5%预算；压力配置虽然降低preemption，Goodput仍为0。
TP宽松long配置也只有18/24请求达SLO，不能将noop吞吐预算PASS当作全部延迟通过。
Graph/TP各有一条NSYS缺少graph GPU span，时间窗完整但相关成本仅为PARTIAL；
TP eager差异主要表现为rank等待，CPU具体热函数尚未定位，不声称已修复。
DP native压力下三种有损策略均24/24达SLO，Sink的preemption三轮570→0、
吞吐22.40→109.25tokens/s；人工skewed Sink仍18/24达标，TTFT超预算，性能FAIL。
DP native/noop两档共48个请求输出逐token相同，宽松noop开销约0；未新增routing
hint或把两个Pool的free合并为全局预算。各结论仅适用于对应报告中的固定负载。

实现与Gate只覆盖既定MRV2、BF16、FA2/NHD/B16、单full-attention group与单在途
配置。Async/spec/connector、PP/DCP/PCP、SWA/hybrid、fine-grained shared partial
page、prefill-time回收及非标准位置偏置不从当前Gate推断支持。TP2×DP2与全量
组合笛卡尔矩阵未执行。必要Prefix+Graph、Prefix+TP、Graph+TP及三项smoke的
具体证据见对应报告。各独立Gate不自动改变v8设计的治理状态或生产默认配置。

## 最终核验与可复现入口

DP汇总器`../scripts/summarize_dp_serving.py`直接读取原始请求、原生路由、metrics、
scheduler/allocator trace及guard结果，重算三轮聚合/逐副本指标、输出一致性与
无压力零回收。`serving-summary.json`的PASS仅表示证据完整性和native/noop
输出一致，不能覆盖上述性能FAIL；`correctness-summary.json`保留两条通过轨迹
及一次已定位并重跑的诊断断言失败。大型原始产物均未加入Git。

DP复现使用`../scripts/run_dp_serving.sh`，局部ownership轨迹命令见`dp.md`。
质量评测入口`../scripts/quality_reclamation.py`随最终交付补入版本管理，沿用
原M2实验保存的85样本配对协议。原脚本SHA与四组实验快照一致；提交钩子要求
将`import re`改为`import regex as re`，对原始340条答案回放核对归一化与三个
质量指标保持一致，未重新下载数据或改变既有结果。

最终离线核验重算DP汇总后，两份summary的SHA保持不变；报告的10行聚合、20行
逐副本、10行路由与10行诊断表逐格核对通过。记录位于DP产物根目录的
`delivery-summary-verification.log`和`delivery-verification.json`。
提交检查命令为`.venv/bin/python -m pre_commit run`；ruff、Markdown、mypy3.10、
shell及其余适用钩子的通过结果保存在`delivery-pre-commit-rerun.log`；首次失败
日志`delivery-pre-commit.log`保留了上述`re`导入问题。

剩余TO_VERIFY限于上述未通过预算的进一步优化、质量保持及未测组合；本次固定
实施路线已执行到DP，当前不扩大实验矩阵。项目GPU进程已退出；同机其他用户
任务保留。下一步可从各负结果对应的raw/profile继续定位，不需要重新建立基线。
