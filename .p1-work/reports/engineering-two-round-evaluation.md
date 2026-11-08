# 两轮 protected 回收统一评估

2026-11-08，基线 HEAD `1d2b991e53`，Runtime 实现 `2439253387`，harness
`380b5480ea`。单卡、TP2、DP2 分开报告；本轮只使用物理 GPU0/1，GPU2 未使用。
机制见[Controller](engineering-two-round-controller.md)，范围见
[计划](../plans/engineering-two-round-plan.md)。

## 协议与可复核性

- 沿用已有 `.venv`、Qwen3-8B BF16、离线模型/缓存和 V2 runner。没有依赖升级、模型
  下载或更换 attention backend。单卡 GPU1/Worker 主线程 CPU21，TP2/DP2 GPU0/1、
  Worker 主线程 CPU20/21；不将该主线程放置写成所有线程的 affinity。
- GPU 经 `run_guard.py` 和同一 `gpu.lock` 串行；项目显存合计 44GiB 停止、45GiB
  硬上限。外部 GPU 任务保留，背景 CSV 在 B0。正、反运行顺序用于观察本机波动，
  不能将本次绝对性能与前轮 GPU1/2 的数字直接拼接比较。
- SLO 每请求同时要求 TTFT≤2s 且 TPOT≤80ms；HTTP 完成/guard PASS 与 SLO PASS
  分开。吞吐为 timed 总输出 token / timed 总耗时；分位数直接合并24条原始请求，
  不平均各轮分位数。预热、模型加载及 Graph capture 不计入 timed 区间。
- 每组小预热后完整负载预热，再三轮24请求；temperature0、seed0、ignore_eos、每请求
  输出128 token。正式不开 trace；包含真实回收、等待和排队耗时。已有 work counters
  对照 Prometheus 抢占与 canonical commit，scheduled=prompt+127+recompute。
- 每次保存输入 token IDs/hash、配置、逐请求流式时刻与输出、源码副本/hash、GPU UUID、
  affinity、NVML、日志。`analyze_two_round_evaluation.py` 独立复算每组指标，验证最终
  Runtime 字节一致、工作量计数及协议；跨组源码/输入 hash 再交叉比较。

B0：`/data_ssd/qcrs/experiments/p1-v8/Engineering-TwoRound-B0/20261108-120031-initial`。
该目录保留早期失败、CPU/静态日志、背景采样、汇总脚本和
`evaluation-summary.json`；各 Serving run 的 `audited.json` 为独立复算。

## Correctness：PASS，失败启动保留

| 验证 | 最终结果与依据 |
| --- | --- |
| CPU planner/BlockPool/Scheduler 本地 fixture | 118 passed /194 deselected；`cpu-full.log` |
| DP state + route 本地 fixture | 2+13=15 passed；`dp-state.log`、`dp-route-all.log`，不重复计子集 |
| lint/type | 适用 pre-commit 与 manual mypy-3.12 通过；`lint-harness-final.log`、`mypy312-harness.log` |
| 单卡真实两轮 | `TwoRound-Correctness/20261108-122140-tp1-eager-window-check` PASS |
| TP2 FULL、延迟 completion、单 victim | `20261108-122246-tp2-full-delayed-one-victim` PASS；每 rank FULL replay61、改写复用10页，最终free27 |
| TP2 FULL、延迟 completion、双 victim | `20261108-122359-tp2-full-delayed-two-victims` PASS；每 rank FULL59、改写复用13页，最终free42 |
| DP2 独立池/窗口、abort/continuation | `20261108-122621-dp2-two-round` PASS；C在abort前入场、复用本地页，peer未回收、abort后112次已验证forward，最终free各27 |

以上 GPU 相对目录均在 `/data_ssd/qcrs/experiments/p1-v8/`。CPU结果是相关本地fixture，
不表示整个 vLLM suite 全通过。真实单 victim 首轮step1→完成decode step2→第二轮
step3→C step4入场；TP2双 victim 首轮step3回收两个、step4完成decode、step5第二轮
只需一个，随后C实际入场。延迟 rank completion 时 canonical 页表未提前修改。
不是靠 abort 清空容量才使受益等待者前进。

保护检查逐次验证 E 删除≤20%、相对已计算逻辑 L 私有保留≥50%。单 victim 示例为
E257→206，完成decode后E208→174、L259。双 victim 首轮E259→211及E257→206，
第二轮其中一个E213→174。两轮之间必须真实完成 forward，乐观 scheduled L 不算。

两次早期 GPU FAIL 原样保留：`121627-tp1-eager-one-victim` 的28页小池不能支持
max_model_len512，fixture改448；`121947-tp1-eager-one-victim-fixed-fixture`
把后续C的回收错误映射到首轮A的逻辑水位，harness出现KeyError，改为检查同一
admission窗口。没有把失败目录覆盖成PASS。最终四个成功 correctness run 的Runtime
与最终代码一致，harness修正不改 Worker/TP 协议。

## Micro：容量路径成立，未证明执行加速

`TwoRound-Micro/20261108-122712-tp2-88-page`：36层、TP2、每rank4 KV heads，
真实Runtime layout的合成KV。E=L4097、P0、B16、缺88页；单轮protected无计划，
两轮可行。30次CPU决策中位109.94µs；GPU每部分3次预热、10次测量，raw samples
保存在`micro-rank-{0,1}.json`，决策在`two-round-decision.json`。

| 动作 | 释放页 | 估计复制字节/rank | Worker wall中位 rank0/rank1 ms | 纯compaction CUDA Event中位 rank0/rank1 ms |
| --- | ---: | ---: | ---: | ---: |
| fixed | 129 | 299630592 | 3.196 / 3.197 | 0.334848 / 0.333824 |
| 两轮第一轮 | 52 | 481001472 | 1.729 / 1.729 | 0.560128 / 0.561664 |
| 两轮第二轮 | 41 | 0 | 2.062 / 2.067 | 0.002048 / 0.002048 |

第一轮keep3278，预测两个decode后E3280/L4099，第二轮释放41页且zero-copy。
micro只验证该形状及执行成本，真实decode由correctness证明。两轮Worker wall要
相加，约3.79/3.80ms，另有模型decode和调度等待；不同策略删除量不同，不能宣称
比fixed的约3.20ms更快。完整Worker Event含协调等待，不解释为纯kernel耗时。

## Serving：按部署和运行顺序报告

单卡/TP2使用同一mixed输入：prompt长度`[4096,8192,128,128]×2`，到达时刻
`[0,.3,.6,.9,5,5.3,5.6,5.9]`。单卡pool1.5GiB，TP2每rank0.75GiB，等效逻辑
容量；batch1024、max_seqs8、FULL_DECODE_ONLY Graph `[1,2,4,8]`。fixed默认预算，
旧protected为rounds1+backfill2，新策略为rounds2+backfill2。正向fixed→r1→r2，
反向r2→r1→fixed，仅一次反向，无额外参数搜索。

### 单卡：相对旧 protected 改善，未超过 fixed

| 顺序/组 | 输出 tok/s | 联合SLO /24 | SLO goodput req/s | 8192 TTFT p50 s | 回收 / miss |
| --- | ---: | ---: | ---: | ---: | ---: |
| 正向 fixed | 108.525 | 24 | 0.8479 | 1.328 | 12 / 12 |
| 正向 protected r1 | 80.032 | 18 | 0.4689 | 4.439 | 6 / 771 |
| 正向 protected r2 | 93.466 | 23 | 0.6998 | 1.595 | 20 / 155 |
| 反向 protected r2 | 107.663 | 24 | 0.8411 | 1.387 | 24 / 30 |
| 反向 protected r1 | 85.021 | 18 | 0.4982 | 4.100 | 6 / 771 |
| 反向 fixed | 108.201 | 24 | 0.8453 | 1.425 | 12 / 12 |

r2相对同顺序r1吞吐分别+16.79%、+26.63%，8192请求由0/6变为5/6、6/6；
128与4096请求均全部达标。相对fixed吞吐分别−13.88%、−0.50%，不能宣称胜出。
正向r2三轮吞吐101.940/79.867/102.387 tok/s、SLO8/7/8；中间一轮第二波8192
TTFT4.7098s，回收4次/miss135，另两轮均回收8次/miss10。原始慢轮保留，不只挑
反向或最好一轮。该偶发路径的精确触发未用trace复现，不推断为已解决。

### TP2：容量进展改善，长短请求仍争用 prefill 时间

| 顺序/组 | 输出 tok/s | 联合SLO /24 | SLO goodput req/s | 8192 TTFT p50 s | 回收 / miss |
| --- | ---: | ---: | ---: | ---: | ---: |
| 正向 fixed | 110.094 | 22 | 0.7884 | 1.952 | 12 / 12 |
| 正向 protected r1 | 84.246 | 18 | 0.4936 | 4.678 | 6 / 771 |
| 正向 protected r2 | 90.682 | 14 | 0.4133 | 2.417 | 20 / 282 |
| 反向 protected r2 | 106.605 | 12 | 0.4164 | 2.325 | 24 / 30 |
| 反向 protected r1 | 89.628 | 18 | 0.5252 | 4.249 | 6 / 771 |
| 反向 fixed | 106.924 | 13 | 0.4525 | 2.390 | 12 / 12 |

r2相对同顺序r1吞吐分别+7.64%、+18.94%，但SLO18/24降至14/24、12/24。
8192请求仍均0/6；128请求从r1的12/12降为8/12、6/12，4096均6/6。相对fixed
吞吐−17.63%、−0.30%，没有整体Serving胜出。fixed自身SLO22/24→13/24，说明
运行顺序/外部负载对2秒边界有影响；不把新旧绝对数字跨时段当作纯TP实现差异。

上述12组全部零抢占、零额外重算，每请求TPOT均≤80ms；SLO失败来自TTFT。
TTFT/TPOT/ITL完整分布、每轮与每长度明细在各`audited.json`，不以p50代替逐请求
联合SLO。r2的回收数包含不同受益者的单轮和两轮窗口，不能全部当作同一个窗口
反复两次以上。TP的copy bytes按单rank记录，不能与单卡字节直接等同。

### DP2：运行回归通过，本负载没有触发 Controller

沿用W1混长短输入`[8192,512]×4`、到达间隔0.05s，两个独立TP1副本，
每副本pool4GiB、batch512、max_seqs8、known_work路由，FULL_DECODE_ONLY Graph。
只比较r1→r2，不重跑requests/known_work旧路由矩阵。

| 组 | 输出 tok/s | 联合SLO /24 | SLO goodput req/s | 8192 TTFT p50 s | 回收 / miss / 抢占 / 重算 |
| --- | ---: | ---: | ---: | ---: | --- |
| protected r1 | 170.765 | 21 | 1.1673 | 1.572 | 0 / 0 / 0 / 0 |
| protected r2 | 205.273 | 21 | 1.4032 | 1.413 | 0 / 0 / 0 / 0 |

两组512-token请求均12/12、8192-token均9/12；每请求TPOT均≤80ms，失败来自TTFT。
两组路由分配相同：rank0/rank1分别15/9请求、prefill53760/50688 token，合计
scheduled107496 token、额外重算0。两组均无allocation miss，因此两轮分支没有执行；
名义吞吐+20.21%不能归因Controller，也没有理由据此继续反向扫参。DP2真实压力下
的两轮执行和等待者进展由独立correctness小池验证。usage tie-break仍STOPPED / NOT_RUN。

DP首次启动`20261108-125137-dp2-protected-r1-forward`误传脚本不支持的
`--graph-mode full`，argparse退出2、无GPU分配；失败原样保留。DP脚本自身固定
FULL_DECODE_ONLY，移除该参数后两组完成，未修改DP Runtime或改动Graph协议。

### TP2 独立诊断：入场路径成立，正式慢路径未复现

`TwoRound-Diagnostic/20261108-125858-tp2-r2-mixed-trace` 使用相同TP2 r2配置，
额外开启trace、只测一轮，不并入上面的正式24请求结果。独立复算见
`diagnostic-audited.json`；server request按首次容量事件排序并用完成work记录的
prompt长度核验，不把client的`r0-i`直接当server ID。

| 两轮窗口 | 第一轮/第二轮释放页 | 真实完成逻辑水位 | 首个plan到实际admission | admission到prefill尾chunk分配 |
| --- | --- | --- | ---: | ---: |
| 第一波8192 | 52 / 41 | 4096→4098 | 35.121ms | 1.176884s |
| 第二波8192 | 52 / 41 | 4096→4098 | 35.056ms | 1.180895s |

第一波0.578416s计划补88页，0.590615s首轮commit，完成真实decode后0.602277s
按真实状态补37页，0.613122s第二轮commit，0.613537s长请求实际入场。第二轮为
zero-copy；不是连续删两次却没有计算进展。timed共8次commit/10次miss、零抢占/
额外重算，两组两轮窗口以外是其他入场或continuation的合法单轮，不是窗口第三轮。

8192请求随后按1023-token chunk消耗batch预算，第一波prefill尾chunk于1.790421s
分配，短请求的首次容量检查于1.790511s出现，再一次单轮回收后1.837987s入场。
第二波同样在长prefill尾chunk时才检查后方短请求。容量更早解决后，长prefill仍与
短请求争用调度和计算时间；这些时间是allocation事件区间，包含forward、调度和
trace开销，不能解释为纯GPU kernel时间，也未量化为正式SLO退化的全部来源。

诊断吞吐122.344tok/s、8/8联合SLO，TTFT最大约1.556s，**未复现正式TP2慢路径**。
其启动背景GPU0/1仅126/12MiB、利用率均0%；正式测量期间存在其他Python负载，
背景快照有约14–15GiB占用。外部任务均保留；诊断明显不同的资源背景与trace扰动
使其不能覆盖正式14/24、12/24负结果。没有证据断言TP collective有错误，也不能
将全部波动归因外部任务。单卡正向r2慢轮及TP2正式TTFT波动的精确触发仍未复现，
本轮停止扩实验。

### 原始Serving目录

以下均相对`/data_ssd/qcrs/experiments/p1-v8/TwoRound-Serving/`：

```text
20261108-122804-tp1-fixed-forward
20261108-122959-tp1-protected-r1-forward
20261108-123127-tp1-protected-r2-forward
20261108-123512-tp2-fixed-forward
20261108-123636-tp2-protected-r1-forward
20261108-123814-tp2-protected-r2-forward
20261108-124129-tp1-protected-r2-reverse
20261108-124245-tp1-protected-r1-reverse
20261108-124411-tp1-fixed-reverse
20261108-124605-tp2-protected-r2-reverse
20261108-124730-tp2-protected-r1-reverse
20261108-124902-tp2-fixed-reverse
```

DP两组成功目录同在上述Serving根目录：

```text
20261108-125239-dp2-protected-r1-forward
20261108-125553-dp2-protected-r2-forward
```

## 复现命令

每个新shell先设置设备，再激活；不使用系统Python或手工启动旁路服务。

```bash
export P1_CUDA_VISIBLE_DEVICES=0,1
source .p1-work/scripts/env.sh
.venv/bin/python .p1-work/scripts/run_guard.py \
  --stage TwoRound-Serving --name tp2-protected-r2-forward -- \
  @python .p1-work/scripts/serving_reclamation.py \
  --policy sink_recent --budget-mode protected \
  --decode-reclaim-rounds 2 --admission-bypass-limit 2 \
  --pool-gib 0.75 --tp 2 --workload mixed \
  --graph-mode full --full-warmup --repeats 3 \
  --max-num-batched-tokens 1024 --max-num-seqs 8 \
  --reclamation-counts --work-counters --worker-cpus 20 21
```

单卡改设备1、pool1.5、tp1、worker-cpus21；r1改rounds1；fixed改budget-mode
fixed，并省略rounds/backfill参数。每次实际命令在对应`invocation.json`，CPU
fixture日志及本地Qwen tokenizer wrapper在B0，不将原环境网络失败写成Runtime失败。
CPU118项日志未保存原始argv，结果和本地fixture范围以该日志为准，不补造历史命令。
两轮TP2 correctness命令：

```bash
.venv/bin/python .p1-work/scripts/run_guard.py \
  --stage TwoRound-Correctness --name tp2-full-delayed-two-victims -- \
  @python .p1-work/scripts/tp_reclamation.py \
  --mode full --tp 2 --protected --decode-reclaim-rounds 2 \
  --victims 2 --delay --lifecycle repeat --worker-cpus 20 21
.venv/bin/python .p1-work/scripts/run_guard.py \
  --stage TwoRound-Micro --name tp2-88-page -- \
  @python benchmarks/kernels/benchmark_protected_reclamation.py \
  --two-round-case --worker-cpus 20 21
```

DP2同样先激活设备0,1，然后：

```bash
.venv/bin/python .p1-work/scripts/run_guard.py \
  --stage TwoRound-Serving --name dp2-protected-r2-forward -- \
  @python .p1-work/scripts/dp_serving.py \
  --scenario w1 --routing known_work --policy sink_recent \
  --budget-mode protected --decode-reclaim-rounds 2 --admission-bypass-limit 2 \
  --pool-gib 4 --repeats 3 --max-num-batched-tokens 512 --max-num-seqs 8 \
  --work-counters --worker-cpus 20 21
```

正式结果可用以下CPU命令复算；`RUN_DIRECTORY`替换为上文成功Serving目录，
不启动GPU、不重跑负载：

```bash
.venv/bin/python .p1-work/scripts/analyze_two_round_evaluation.py RUN_DIRECTORY
.venv/bin/python .p1-work/scripts/analyze_two_round_evaluation.py --diagnostic \
  /data_ssd/qcrs/experiments/p1-v8/TwoRound-Diagnostic/20261108-125858-tp2-r2-mixed-trace
```

## 质量、归因和停止边界

冻结85条质量样本未扩、旧质量矩阵未重跑；合成Serving输出token序列只作机械差异
记录，不代表质量评估。保护更多KV没有自动取得质量收益证据。两轮间真实decode
也不等于删除无损。

本轮不实现prefill中删除、DP usage tie-break、offload或跨池迁移；没有其进入证据。
固定预算和requests路由仍默认。能够分两轮补足缺口修正了上一轮“一轮不足便只能
等自然释放”的限制，不承诺所有长请求可入场或满足SLO。后续新方向不能通过删除
本轮失败、只挑最快轮或降低项目SLO来成立。

本轮结论为IMPLEMENTED / CORRECTNESS_PASS；单卡相对r1有有限Serving改善，
micro不判加速、TP2联合SLO退化、相对fixed未证明Serving胜出，DP2性能归因NOT_RUN。
`decode_reclaim_rounds=1`保持默认，2只作为显式选项。prefill中删除KV按用户要求
暂缓，DP usage tie-break缺进入证据而停止，不在本轮增加复杂调度策略或参数扫描。

全部GPU共23次guard：20 PASS、3 FAIL，项目NVML合计峰值42830135296 bytes
（39.888672GiB），未触及44/45GiB资源线；失败见上文。最终源码、静态日志、命令、
用户patch与历史正文保护审计、提交索引见
`/data_ssd/qcrs/experiments/p1-v8/Engineering-TwoRound-Final/20261108-130116-closeout`
的`completion-manifest.json`及[收尾](engineering-two-round-closeout.md)。
