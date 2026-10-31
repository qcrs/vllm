# P1 v8 独立Review限定修复记录

2026-10-31，基准为 `252c747ad6127d4c17540e5fb08704336fd756be`，分支为
`p1/v8-physical-kv-reclamation`。本次完成R1、R2、D1–D4的限定功能收尾。
[独立Review](independent-closeout-review.md) 保持发现时原文；其“待修复”未被改写。
开始时已有delivery导航改动与新增Review均保留，原diff和Review SHA256见归档baseline。
本地修复提交和最终文件hash写入归档 `final-state.json`，不改写历史提交或创建上游PR。

固定环境按 [prompt1](../prompts/prompt1.md) 和现有AGENTS执行；全部Python命令使用
现有 `.venv/bin/python`，本次未升级依赖、下载模型、修改reference或学习基线。
未新跑GPU、模型质量评测或M0→DP矩阵，已有85条样本保持冻结。

## 逐项处理

| 编号 | 复核结论与处理 | 修改位置 |
| --- | --- | --- |
| R1 | 当前HEAD可复现retained E被L覆盖；选择feature-on在入口拒绝resumable/streaming-input，feature-off保留原生能力 | [Scheduler.add_request](../../vllm/v1/core/sched/scheduler.py)、[AsyncLLM.add_request](../../vllm/v1/engine/async_llm.py) |
| R1回归 | 拒绝前不改变队列/Pool/请求；feature-off真实pause→append→admission；前端不消费输入、不注册输出或发请求 | [test_reclamation_resumable_admission](../../tests/v1/core/test_scheduler.py)、[test_reclamation_rejects_streaming_before_frontend_state](../../tests/v1/streaming_input/test_async_llm_streaming.py) |
| R2 | append/gather只更新有效prefix，resident/input真实row旧tail未清零；安全性来自E、容量及访问bounds，padding单列 | [graph-runtime](../source-analysis/graph-runtime.md) |
| D1 | 区分preempt释放并清L/E后recompute与保留KV的session重入，同步当前拒绝契约 | [m1-runtime](../source-analysis/m1-runtime.md)、[delivery](delivery.md)、[status](../plans/status.md)、[final-review](final-review.md) |
| D2 | 标明M2初始阶段快照，链接当前shared-prefix P推导和多policy路径 | [m2-pressure](../source-analysis/m2-pressure.md)、[prefix-runtime](../source-analysis/prefix-runtime.md)、[m2-scoring](../source-analysis/m2-scoring.md) |
| D3 | 以当前symbol替换漂移行号；修正HTTP header owner和源码/报告导航 | [dp-runtime](../source-analysis/dp-runtime.md)、graph-runtime、m2-pressure；源码symbol静态核对见检查归档 |
| D4 | delivery只保留当前结论与导航，status维护未决项；历史plan不再复制report实验数表 | delivery、status、[final-review计划](../plans/final-review.md)、[performance-followup计划](../plans/performance-followup.md) |

reports继续保存实验事实和历史负结果，notes保存协议，source-analysis解释当前调用链、
状态所有者与invariant；测试四问补充在 [test-design](../notes/test-design.md)。
未新增一套重复学习文档，也未为了R2增加tail清零或payload写入。

D4将旧delivery中的交付核验事实归档于此，未重跑这些历史检查：DP原核验的两份summary
SHA保持不变，10行聚合、20行逐副本、10行路由、10行诊断表逐格通过；证据为
`/data_ssd/qcrs/experiments/p1-v8/DP/delivery-verification.json` 和同目录
`delivery-summary-verification.log`。原质量脚本改用 `regex as re` 后回放340条既有答案，
指标不变，不是新增样本；原hook失败/通过日志为 `delivery-pre-commit.log` /
`delivery-pre-commit-rerun.log`。复现入口继续为 `summarize_dp_serving.py`、
`run_dp_serving.sh`、`quality_reclamation.py`，位于 `.p1-work/scripts/`；原报告保留其协议。

## R1复现、选型与支持契约

已沿AsyncLLM入口、Scheduler session暂停/追加、admission、allocator和Worker初始化
复核真实生命周期。按独立Review第4节参数，32 blocks/B16/batch256，A为64 prompt、
max_tokens=3、resumable；prefill后回收到K32，再自然暂停并追加16 tokens：

```text
paused: L=66, E=33, blocks=3
continued: L=66, E=33, prompt_len=82
admitted: E_reset=66, q=16, row=[1,2,5,4], capacity=64
worker_initial_E=66, worker_attention_length=82
```

真实Scheduler/BlockPool的分配按33+16=49得到四页，发布的E却恢复为66。
这确认控制契约破坏；没有执行该GPU forward，不能声称观测到GPU越界或payload污染。
复现脚本、原命令、输出和exit0保存在 `20261031-171803-bounded-fixes`。

原设计要求preempt后释放KV、清零L/E再recompute，没有要求支持保留有损KV的session。
当前没有跨session传递retained E的metadata契约，因此明确拒绝该组合；retention=1
也统一遵守feature-on支持范围，不引入策略特例或静默重算。

- `Scheduler.add_request` 是canonical admission的contract owner，在existing-session
  分支、streaming_queue创建和入队之前拒绝 `request.resumable`，不改变allocation或
  Worker E初始化来掩盖不一致。
- `AsyncLLM.add_request` 是必要公共前端边界：对AsyncGenerator和直接传入的resumable
  EngineCoreRequest，按与Core相同的feature开启语义（additional_config中
  `kv_reclamation` 非None）提前抛出同一清晰 `ValueError`。仅在Scheduler拒绝会使
  异步ADD异常离开EngineCore busy loop，前端只能得到EngineDead，且已注册输出状态。
- 未在InputProcessor、EngineCore、allocator、session私有方法或Worker重复加guard。
  feature-off路径保留，CPU回归实际经过原生暂停、追加和再次admission。

Worker仍只需对fresh/APC或preempt后的重算从computed length初始化E。运行中普通请求
始终以canonical E和执行镜像驱动allocation/slot/attention；本轮未改动GPU路径。
继续保留ownership与Prefix hash/shared-page保护、source E/row及结果身份校验、TP全rank
completion、AsyncOutput/maintenance实际完成等待、Graph active bounds/padding及
mutation后失败的fail-stop契约。

## 实际验证与证据入口

归档根目录为 `/data_ssd/qcrs/experiments/p1-v8/Closeout`，本轮只使用CPU。

| 目录 | 实际结果 | 证明边界 |
| --- | --- | --- |
| `20261031-171803-bounded-fixes` | 旧HEAD复现exit0，R1与Review一致；保存baseline/initial.diff/原Review hash | 控制链缺陷，未执行GPU forward |
| `20261031-172010-bounded-regression` | 12 failed、26 passed、199 deselected，exit1；保留完整日志 | 扩大选择暴露既有fixture配置和缺少async插件，不能写成全suite通过 |
| `20261031-172147-targeted-regression` | 28 passed、201 deselected，exit0，guard PASS，项目NVML峰值0 | 必要控制、ownership、Prefix/config和两处拒绝行为；非GPU payload/completion证明 |

旧HEAD复现的实际命令（当时尚未加入拒绝修复；修复后该脚本应在admission被拒绝）：

```bash
source .p1-work/scripts/env.sh
.venv/bin/python /data_ssd/qcrs/experiments/p1-v8/Closeout/20261031-171803-bounded-fixes/reproduce_r1.py
```

首次扩大选择的实际命令：

```bash
.venv/bin/python .p1-work/scripts/run_guard.py \
  --stage Closeout --name bounded-regression -- @python -m pytest \
  tests/v1/core/test_scheduler.py tests/v1/core/test_kv_cache_utils.py \
  tests/v1/streaming_input/test_scheduler_streaming.py \
  tests/v1/streaming_input/test_async_llm_streaming.py \
  -k 'reclamation or pressure or allocator_trace or streaming' -q
```

其中8条原生Scheduler streaming测试在构造阶段因MagicMock model_config未设置
`is_encoder_decoder=False`，进入encoder-decoder断言后失败，尚未走到add_request。
另4条async函数因当前环境没有pytest-asyncio失败（含当时新加的2条）；未安装插件。
新增前端拒绝测试改为同步测试内 `asyncio.run`，已有async测试和fixture未改动。
这两类完整suite限制保留，不用mock掩盖allocator或GPU异步问题。

必要回归实际命令：

```bash
source .p1-work/scripts/env.sh
.venv/bin/python .p1-work/scripts/run_guard.py \
  --stage Closeout --name targeted-regression -- @python -m pytest \
  tests/v1/core/test_scheduler.py tests/v1/core/test_kv_cache_utils.py \
  tests/v1/streaming_input/test_async_llm_streaming.py \
  -k 'reclamation or pressure or allocator_trace' -q
```

28项含新增Scheduler on/off两例、前端AsyncGenerator/EngineCoreRequest两例与24项已有
核心回归。16条warning为14条既有Torch弃用和2条既有asyncio marker警告。
Core使用真实Pool和现有完成helper；helper不证明GPU completion。前端沿用既有
mock_async_llm，仅检查拒绝发生在任何方法调用和generator消费前，不模拟成功执行。
feature-off回归只证明控制链保留，未宣称完成GPU streaming continuation验证。

静态检查结果：适用pre-commit（含ruff、Markdown和mypy3.10）PASS；12份变更Markdown
的124个本地链接及24个关键源码symbol通过；`git diff --check`通过。
首轮pre-commit仅E501发现前端错误字符串超过88列，拆行后复跑通过；行为与通过28项
回归的源码AST一致。原Review SHA256与baseline相同。

实际检查命令如下；pre-commit使用本次修改文件列表，展开参数和exit code见归档
`checks.json`，原Review保持原内容：

```bash
.venv/bin/python -m pre_commit run --files \
  .p1-work/notes/test-design.md .p1-work/plans/final-review.md \
  .p1-work/plans/performance-followup.md .p1-work/plans/status.md \
  .p1-work/reports/closeout-fixes.md .p1-work/reports/delivery.md \
  .p1-work/reports/final-review.md .p1-work/source-analysis/dp-runtime.md \
  .p1-work/source-analysis/graph-runtime.md .p1-work/source-analysis/m1-runtime.md \
  .p1-work/source-analysis/m2-pressure.md tests/v1/core/test_scheduler.py \
  tests/v1/streaming_input/test_async_llm_streaming.py \
  vllm/v1/core/sched/scheduler.py vllm/v1/engine/async_llm.py
.venv/bin/python /data_ssd/qcrs/experiments/p1-v8/Closeout/20261031-171803-bounded-fixes/check-links-symbols.py
.venv/bin/python /data_ssd/qcrs/experiments/p1-v8/Closeout/20261031-171803-bounded-fixes/performance-recheck.py
git diff --check
```

链接检查限本地文件目标，symbol以源码AST核对；不包含外部URL和Markdown anchor。
归档 `final-state.json` 关联本地修复提交、文件hash与原Review hash。性能旧数据只读
重算脚本、输入hash和结果另存 `performance-recheck.py` / `performance-recheck.json`，
不调用会覆盖旧summary的main，也未改写原始输入。

## O1–O3判断与停止理由

| 编号 | 已有证据与本轮决定 | 再开启时的最小判别步骤 |
| --- | --- | --- |
| O1 | TP pressure零抢占但7个TTFT失败；现有trace支持排队与chunked-prefill共同影响，缺完整请求级关联，不能选择有因果收益的修法。本次保留FAIL，未扫batch或新做GPU实验 | 只对当前配置一轮关联client→enqueue→first schedule→prefill completion→reclaim completion，区分排队、host、GPU和通信 |
| O2 | 真实代码存在逐层normalization/reduction/selection多launch；micro E4096完整plan约6.15ms，不等于真实Runtime shape的Serving净收益。本次未融合，也不拿Tri成本解释SinkRecent TTFT | 先测真实shape/stride的完整评分、归一化、selection与发射成本，再按数值和stable-tie契约选择局部融合 |
| O3 | `prepare_attn` 每步构造CPU上界buffer和metadata；旧稳态诊断约43.57/41.66μs，当前指定TP eager预算已过，无代表性净收益证据。本次不改 | 先确认稳态占比及预期收益，足够时才复用buffer，保留active/padding语义，不加noop旁路 |

host NVTX总时间包含发射、等待和profiler扰动，不能减去raw kernel时间后全部算成
可消除开销。低occupancy不支持重写compactor；不新增backend、dispatcher、通用框架、
DP跨池借页、active KV migration或无证据routing hint。可选优化不阻塞本次R1收尾。

## 保留的性能负结果与未验证范围

只读重算现有正式结果，未重跑旧6.871%或任何完整矩阵：

- TP eager native/noop均值62.712686/62.168299 tok/s，开销0.868066%。只支持本机
  CPU20/21、short/eager、8GiB/rank、batch2048的有限样本预算通过；历史14.33%和
  6.871%仍FAIL，默认CPU放置与CPU1/49微观机制未普遍解决。
- TP pressure CPU20/21、0.75GiB/rank、batch1024的三轮为6/6/5，共17/24、
  零抢占，平均100.916991 tok/s、Goodput0.558432 req/s；仍因TTFT判SLO FAIL。
- DP skewed CPU20/21、1.5GiB/replica、batch512三轮24/24，平均125.392280 tok/s、
  Goodput0.979627 req/s；只覆盖固定6/2路由和当前配置，历史batch2048的18/24仍FAIL。

SLO始终为 **TTFT ≤ 2s 且 TPOT ≤ 80ms**。M2/Prefix/TP/DP历史配置FAIL与
Graph/TP旧profile PARTIAL不覆盖，原始事实保存在对应reports；新完整profile不改变旧窗。
累计归还blocks不等于唯一物理页数或同时新增容量。新配置收益不能归因为本轮入口拒绝
改动，也不把diagnostic/profile吞吐混入正式性能。

85条样本未扩评，有损质量保持未证明；token一致与payload搬移正确不能替代质量。
本次没有GPU completion或continuation新证据，不外推retained session、多在途、
TP2×DP2、其他模型/backend/layout或未测组合。必要回归通过且文档准确后停止限定收尾。
