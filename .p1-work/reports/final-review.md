# P1 v8 功能收尾审查

本轮依据 `../prompts/prompt1.md`、四份 v8 reference 与用户最新要求，聚焦
Runtime 是否完整实现既定语义、功能遗漏、冗余检查、Triton 落地及 TP/DP。
不扩大质量评测，也不将已有 85 条探索性样本升级为质量保持证明。
基础提交为 `829b07dd62`；各新 run 的 invocation/source 保存本轮实际 diff。
大产物统一在 `/data_ssd/qcrs/experiments/p1-v8/Review/`，历史负结果未覆盖。

## 已修复的问题

| 问题 | 原因与影响 | 修复 |
| --- | --- | --- |
| 局部/非因果 attention 可能误放行 | `FullAttentionSpec` 也能携带 sliding window、chunk 或 non-causal 标志；仅检查类型不能证明 dense causal 语义 | 初始化拒绝这三种 restricted spec；CPU 参数化测试覆盖 |
| pooling runner 可能误放行 | pooling 不经过 generation postprocess，物理 E 不会按当前契约推进 | `ReclamationConfig.from_vllm_config` 要求 generate runner；测试验证拒绝 |
| ALiBi 可能静默使用压紧后的距离 | FA 实现按物理位置计算 ALiBi bias；保留旧 K 的 RoPE 不能补偿此类相对位置偏置 | Worker 检查实际 attention impl，初始化明确拒绝 ALiBi；测试确保在访问/修改 cache 前失败 |
| 普通 decode 多两个 Triton launch | physical prepare 与 E advance 原先各单独发射一次，二者工作可与原生长度处理合并 | 合入 `prepare_pos_seq_lens` / `post_update`；不改变 post-forward 推进时点 |
| 每步跨进程重复发送 E 字典 | Scheduler 发送 canonical E，仅供 Worker 逐项断言；Worker 本已有完整 add/preempt/advance/reclaim 生命周期 | 删除该字段、构造和 Worker 扫描；保留回收 intent 的 source-E 校验 |
| commit 重复校验同一 ownership 事实 | 已验证 retained 包含完整 protected prefix、整个其余 suffix 均私有，再检查 removed refcount 无新信息；retained set 也构造了三次 | 删除被覆盖的检查，set 只构造一次；保留提交边界的源 row、结果身份、唯一性、子集和 sharing 检查 |

pooling 测试同时暴露 fixture 的 `Qwen3Config()` 缺少 architectures，被上游推断为
pooling。fixture 现明确使用 `Qwen3ForCausalLM`，模拟本项目真实生成模型。
没有通过放宽 guard 来迁就测试。

## 复核后保留的边界

- **L/E 分离**：模型位置、sampling、原始 token/hash chain 使用 L；physical
  attention、slot、allocation 使用 E。新 kernel 一次生成两套输入，prepare 不推进 E。
  postprocess 才推进 GPU E，CPU mirror 同步推进，prefill/resume 使用原生初始化。
- **完成与释放**：回收先完成 sampling/postprocess，再更新 payload/row/E；
  AsyncOutput 的 event 或 maintenance event 覆盖写入。Scheduler 收到成功 ACK
  后才提交 canonical Block 对象与 free。source/ownership 检查位于破坏性状态交接处，
  有实际 invariant，保留这些检查。
- **Prefix**：保护直到最后一个 shared 整页；mutation 前撤销整个私有 suffix 的
  hash aliases。压紧后的 suffix 不重新按原 token identity 发布。保留完整页共享契约。
- **Graph**：logical 与 physical buffers 分配一次，capture 与 replay 使用同一物理
  seq_lens 地址；每步更新有效范围和 padding，回收不进入 model graph。
- **TP**：distinct-KV/GQA 贡献 SUM、rank0 stable plan、broadcast、本地 copy/row/E、
  completion SUM、host ACK、canonical free 的顺序保持。非 output rank 失败仍由原生
  monitor fail-stop，不用成功 rank 的本地 event 提前释放。
- **DP**：每个 Engine 保持独立 Pool、native router 与 abort affinity；不借 peer
  空闲页，不迁移 active request，不新增跨 DP 回收 barrier。

本轮未发现上述已验证配置中的新增 ownership 或 completion 缺陷。结论来自源码链、
现有失败/成功轨迹和本轮回归，不能外推到未测组合。当前支持仍限 MRV2、单在途、
BF16、FA2/NHD/B16、单 dense full-attention group、生成 runner；TP1/2、dense DP2
与 Prefix/Graph 的具体组合以各报告实际用例为准。TP2×DP2、async/spec/connector、
PP/DCP/PCP、prefill-time 回收、fine-grained shared partial page 均未新增支持。

## Triton 已调优并进入 Runtime

`KVReclamationWorker.reclaim → StableKVCompactor.compact` 实际采用 T16/W4/stages1，
真实 strides、int64 offsets、对齐提示、identity skip、CTA 内 load/barrier/store。
36 层/8 KV heads/D128、32K half-retention 的 CUDA Event median 从约 18.39ms 降到
3.66ms；registers 128→56，无 compiler spill。NCU 的 DRAM 约 201.71GB/s→1.22TB/s。
只有 288 个 CTA，achieved occupancy 仍约 16.77%，不能称所有 shape 已最优。
T32 与 T16 延迟接近但寄存器更多，因此维持 T16。证据见 `m1.md`。

Tri shared scorer 把未来 query phase 从逐 tile 重算提到每次计划只算一次。
32K 完整 plan 约 184.14→8.44ms，raw kernel registers 255→72；4K 完整 plan
仍约 6.15ms，Torch reduction/launch 是已知限制。没有无依据新增 backend。
评分路径和原始 NCU/NSYS 见 `m2-scoring.md`。

本轮新 trace 验证实际 launch：Graph 正式诊断窗口有 308 次原生 prepare、308 次
post-update；TP 窗口两 rank 合计各 534 次。旧的独立 physical prepare / E advance
kernel 均未再出现。Graph 窗口实际执行 16 次 `_compact_stable`。
这证明优化进入真实 Serving，不仅存在于 benchmark 参数。

## TP eager：功能收尾，吞吐预算仍 FAIL

这里的 5% 是**无压力时启用回收框架允许损失的吞吐比例**，不是显存上限。
口径保持原协议：`100 × (1 - noop平均tokens/s ÷ native平均tokens/s)`。
旧 native 62.874、noop 53.864，即吞吐少 14.330%；预算要求至少约 59.731 tok/s。
noop 保留全部 KV，用来量框架成本，并不是通过省略功能的方式获取通过。

本轮保持 8GiB/rank、short/eager、相同两请求及 arrival、完整负载预热、每 server
3 个正式轮。改动前先复测，再按 native/noop/noop/native 顺序复测改动后源码。
所有 round 全量保留，不挑最快 server。

| 对照 | native mean tok/s | noop mean tok/s | 吞吐开销 | ≤5% |
| --- | ---: | ---: | ---: | --- |
| 历史 TP 正式结果 | 62.874 | 53.864 | 14.330% | FAIL |
| 本轮融合前 | 59.874 | 61.910 | -3.400% | PASS，仅该次 |
| 融合后 ABBA 的 A 对 | 61.541 | 53.426 | 13.187% | FAIL |
| 融合后 ABBA 的 B 对 | 62.047 | 61.670 | 0.607% | PASS，仅该次 |
| 融合后全部 6+6 轮合并 | 61.794 | 57.548 | 6.871% | FAIL |

同一源码的两个 noop server 相差明显，融合前也不能稳定复现旧 14.33%。因此不能
把新 kernel 变化归因为「已消除 14.33%」或把负开销解释为稳定加速。
改动后 12 对请求输出逐 token 相同，native/noop 均 12/12 达现有短负载 SLO，
这与吞吐预算是不同结论。峰值 35,454,451,712 bytes，未触发 44GiB guard。

旧 eager NSYS 已观察到 rank1 原生 `cross_device_reduce_1stage` 累计等待增加，
两个新增小 kernel 的 GPU 时间不足以解释整个差距。本轮 host-phase/cProfile
确认 physical prepare/advance 原先存在独立 launch，已予融合；host 执行时间包含
Python、driver 与调度等待，诊断本身也有开销，不并入上述正式吞吐。
跨 server 的 launch 节奏/等待差异仍是未解决的性能问题，不归咎于未证实的具体
硬件或调度原因，也不扩大为 TP 结果错误。

隔离重采的host诊断为 `20261031-145413-tp-eager-thread-host-full` 与
`20261031-145509-tp-eager-thread-host-noop`。最后100步的rank0 execute_model
wall P50为27.584→33.452ms，thread CPU P50为27.579→33.453ms；rank1约
27.838→27.972ms。慢轮的大差距没有表现为rank0主线程被挂起的墙钟空洞，
但这些计数不能区分Python执行、driver忙等、CPU频率或SMT影响。每rank允许
CPU集合均为0–95，没有按这些诊断新增CPU绑核策略。原始phase/cProfile与
`Review/host-review-summary.json`保留，不能据此宣称完整根因已定位。

正式比较由 `../scripts/summarize_review.py` 从原始 requests/serving/guard 重算，
生成 `Review/serving-review-summary.json`，同时逐 token 核对全部配对输出。

## Profile 缺失证据已补采

旧 Graph 与 TP 的 sqlite 各有一条 incomplete CUPTI event dropped，导致各缺
一个 graph GPU span；不是把 join 条件放宽就能补回的数据。原 PARTIAL 记录保留。
本轮不升级驱动/NSYS，仅把 flush interval 作为诊断参数，补采时设为 0。

| 新目录 | window | 完整性与成本边界 |
| --- | --- | --- |
| `20261031-144831-tp-full-tri-nsys-flush0` | 首个计时轮 17.484s | PASS；424 graph spans；每 rank score/plan/completion 各27次，162个collective span全匹配；无 incomplete/drop 记录，GPU tail距结束约5.70ms |
| `20261031-145135-graph-sink-nsys-flush0` | 首个计时轮 9.893s | PASS；274 graph launch API与274 GPU spans全部匹配；无 incomplete/drop 记录，GPU tail距结束约5.71ms |

TP 每 rank score/plan/completion 的逻辑 payload 分别为 307356/153652/108 bytes；
rank0 NCCL 累计 0.429/0.218/5.026ms，rank1 为 217.364/17.219/56.698ms。
device time 包含 rank 等待，不能当纯通信传输延迟或链路实测 traffic。
Graph 已匹配活动 union 为 8540.978ms；余下时间包含 arrival、HTTP 和 CPU 等待。
工具仍提示 12.7 driver 使用 12.6 tracing libraries，完整性 PASS 仅指上述计时窗口
和匹配项目，不代表工具所有能力都已验证。诊断吞吐不替代正式 Serving。

## 本轮真实模型回归

所有模型均为本地 Qwen3-8B BF16；以下目录在 Review，guard 均 PASS。

| 目录 | 检查与结果 |
| --- | --- |
| `20261031-143734-tp-prefix-full-token` | Prefix+Graph+TP，两 rank各61 FULL/3 NONE、42 padded、47 FP32 attention、46 matched-eager；10页真实重写 |
| `20261031-143830-tp-full-token-q0-delay` | q0且非output rank慢3s；全组完成前canonical/free不变，retained/reuse正确 |
| `20261031-143924-tp-full-token-maintenance-delay` | maintenance且非output rank慢3s；空batch ACK仍等待全组完成 |
| `20261031-144010-dp-local-pressure-reuse-abort` | C取得10–17共8页并在36层实际覆盖；peer无回收；abort后112次peer forward验证，最终free32/32 |

DP 新回归的 peer 128 个输出与原 native
`DP/20261031-122856-local-pressure-full` 完全相同；比较记录保存在新 run 的
`peer-output-comparison.json`。本轮没有修改 native DP 路由来迁就人工 skew。

最终相关测试分两次执行：`20261031-145939-final-core-reclamation` 为24 passed、
199 deselected；`20261031-150141-final-local-prefix-compaction-executor` 为
130 passed、5 deselected。后者完整覆盖single-type manager、Prefix与GPU
compaction/scoring文件，加executor不依赖模型下载的用例；共154项通过。
前者覆盖scheduler pressure/APC/E生命周期与allocator ownership/config拒绝。
14条既有Torch弃用warning保留。命令如下：

```bash
export P1_CUDA_VISIBLE_DEVICES=1
source .p1-work/scripts/env.sh
.venv/bin/python .p1-work/scripts/run_guard.py --stage Review --name core -- \
  @python -m pytest tests/v1/core/test_scheduler.py \
  tests/v1/core/test_kv_cache_utils.py \
  -k 'reclamation or pressure or allocator_trace' -q
.venv/bin/python .p1-work/scripts/run_guard.py --stage Review --name local -- \
  @python -m pytest tests/v1/core/test_single_type_kv_cache_manager.py \
  tests/v1/core/test_prefix_caching.py tests/v1/worker/test_kv_compaction.py \
  tests/v1/executor/test_executor.py -k 'not custom_executor' -q
```

没有把整个上游suite写成PASS。`20261031-145719-final-full-reclamation-suites`
在12项通过后，上游默认OPT配置加载产生16个失败，随后中断重复失败；离线环境
无 `facebook/opt-125m` 配置，且proxy路径缺socksio。随后完整四文件运行
`20261031-150002-final-prefix-compaction-executor` 为130 passed/5 failed，
5个custom-executor用例依赖未缓存的 `Qwen/Qwen3-0.6B`，停在离线资源加载，
没有运行该模型。它们也不符合本项目8B真实模型门槛，明确排除后得到上述130项
PASS。未安装额外依赖、下载替代模型或修改上游测试默认模型来消除失败。
本项目真实8B executor/TP/DP运行证据见前表。

最终 staged 文件执行 `.venv/bin/python -m pre_commit run`，所有适用钩子通过，
包括 ruff、Markdown、mypy 3.10、SPDX 与 forbidden-import 检查；无适用文件的
钩子保持 Skipped。日志保存在 `Review/final-review-pre-commit.log`。
`git diff --check` 通过。GPU0/1 已无项目 compute 进程；GPU2 的其他任务未改动。

## 收尾结论与保留项

功能验收回答「是否按设计真实回收并安全复用」；现有及本轮证据支持已测配置。
质量评测按用户最新范围冻结，不作为本轮阻塞。没有宣称生产质量保持、所有性能
预算通过或全组合兼容。

保留负结果：TP eager 吞吐预算仍 FAIL；TP 压力 Goodput仍为0；M2/Prefix部分压力
TTFT/TPOT超 SLO；DP人工skewed Sink仅18/24达 SLO。这些不是通过放宽配置检查
或删除 ownership 验证来修复的问题。新 profile 补齐了证据完整性，不抹掉旧负结果。
