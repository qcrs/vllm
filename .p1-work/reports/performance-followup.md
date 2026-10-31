# TP eager 与压力 Serving 性能收尾

2026-10-31，基于功能收尾提交 `4ad1a03e1c`。本轮优化启动配置和实验工具，
没有修改 `vllm/` Runtime、回收策略或 Triton kernel；质量评测按用户要求冻结。
Qwen3-8B BF16、物理 GPU0/1、现有依赖及 44GiB 停止阈值保持不变。
原始证据根目录为 `/data_ssd/qcrs/experiments/p1-v8/Perf/`。

本页保留该轮正式性能和历史事实。后续O1请求级关联、独立诊断及代表性限制见
[TP TTFT诊断](tp-ttft-diagnostic.md)，其profile数据不并入本页正式汇总。

## 结论与适用范围

TP2 short/eager 在本机显式 `--worker-cpus 20 21` 配置下，原生 Full/noop
ABBA 四次独立服务、各三轮，合并吞吐开销 **0.868% ≤ 5%，PASS**；
12 对请求输出逐 token 相同。历史无绑核的 14.33% 和上一轮合并 6.871%
仍保留 FAIL。新结果证明这项启动配置通过预算，不证明任意 CPU 放置均通过。

这里的“5% 预算”是相同输入、容量和运行配置下，允许回收框架带来的吞吐损失：
`100 × (1 − noop 吞吐 / native 吞吐)`。noop 保留框架、retention=1、不丢 KV；
full 为原生关闭框架。该预算与有损策略的任务质量损失是两回事。

## TP 两档执行速度的因果证据

预热后 cProfile 对齐中，native/noop 的 rank0 总 self CPU 时间为
4.6057/4.6656s，rank1 为 4.5804/4.6837s；两边调用数接近。
noop 的 physical `prepare_attn` 各 140 次，含 metadata replacement 的平均
cumulative 时间约 43.57/41.66μs。这个量级不足以解释慢档多出的约 6ms/step。
cProfile 会改变主机发射节奏，其吞吐不并入正式 ABBA。

同一服务空闲时切换 Worker 框架的独立诊断中，关闭它后仍约 52–53 tok/s；
将两个 Worker 主线程放到 CPU20/21 后，打开/关闭分别为 62.747/62.374 tok/s。
该诊断限定 eager、retention=1、E=L，Scheduler 框架始终存在，只用于归因，
不能当作 native 对照或生产旁路。源码见 `profile_tp_toggle.py`。

另一条同服务 CPU 交叉干预保留完整 noop 路径，仅改变主线程 affinity。
下面 execute P50 由原始逐步事件按正式 HTTP 时间窗口重算，包含 prefill 步：

| phase | rank0/rank1 CPU | tok/s | execute P50 ms，rank0/rank1 |
| --- | --- | ---: | --- |
| unbound | 实测 43 / 1 | 54.081 | 28.218 / 33.050 |
| slow-remote-a | 1 / 0 | 53.118 | 33.345 / 28.927 |
| slow-local | 0 / 1 | 54.333 | 28.770 / 32.975 |
| fast-local | 20 / 21 | 63.086 | 27.336 / 27.658 |
| fast-remote | 21 / 20 | 62.397 | 27.656 / 27.515 |
| slow-remote-b | 1 / 0 | 51.895 | 34.085 / 28.395 |
| fast-local-b | 20 / 21 | 60.298 | 27.501 / 28.414 |

慢执行随 CPU1 移到任一 rank；将 20/21 对调后仍快，排除了“仅远端 NUMA”
的解释。本机 CPU1/49 为同一物理核的 SMT siblings，20/68、21/69 同理。
GPU0/1 属于不同 NUMA node，拓扑为 SYS，无 NVLink。已确认可干预的原因是
主机线程放置；CPU1/49 的底层慢因究竟属于 driver、IRQ、SMT 争用或其他因素，
本轮没有进一步精确拆分，不将未证实机制写成结论。

落地为可选 `worker_affinity.AffinityWorker`：在原生 `init_device` 后对当前
主线程调用 `sched_setaffinity`，`execute_model`、sampling 和回收路径直接继承。
已存在的 helper 线程 affinity 不变；之后新建线程可能继承当前 affinity，
不能宣称进程所有 helper 线程都保持原设置。TP/DP 都记录实际 local rank 与
affinity。CPU20/21 是本机实验配置，不是通用硬编码的 Runtime 默认值。

## 正式 TP eager ABBA

short 输入 128/256 tokens、arrival=0/0.3s，各输出 128 tokens；8GiB/rank，
eager、APC off、max batch=2048、max seqs=8。先 32/8 小预热，再完整负载预热，
各服务三轮计时；双方都使用同一 affinity Worker，没有逐步 profiler/trace。

| 服务顺序 | 策略 | 吞吐三轮 tok/s | 均值 tok/s | SLO |
| --- | --- | --- | ---: | ---: |
| A | full | 63.262 / 63.238 / 61.029 | 62.510 | 6/6 |
| B | noop | 61.637 / 62.828 / 63.134 | 62.533 | 6/6 |
| B | noop | 62.611 / 62.210 / 60.590 | 61.804 | 6/6 |
| A | full | 63.876 / 62.319 / 62.552 | 62.916 | 6/6 |

两对开销为 −0.037%、1.767%，全 6+6 轮合并 native/noop 为
62.712686/62.168299 tok/s，开销 **0.868066%**。没有舍弃较慢轮次，负开销
按有限样本波动解释。双方各 12/12 达 SLO，12 对输出相同，所有 guard PASS；
项目 NVML 峰值 35,454,451,712 bytes（约 33.02GiB）。

## 压力配置与不变的验收口径

使用原 long 输入 `[4096,8192]×4`，arrivals 为
`[0,0.3,0.6,0.9,5,5.3,5.6,5.9]`，各 128 输出。
DP skewed 的最后两个输入仍为 128/256，前六个固定 replica0、后两个 replica1。
SLO 始终为 **TTFT≤2s 且 TPOT≤80ms**；APC/async off，FULL_DECODE_ONLY，
max seqs=8、SinkRecent retention=0.5、reserve=0。TP 为 0.75GiB/rank，
DP 为 1.5GiB/replica。只调整主线程放置和原生 chunked-prefill batch 配置。

旧脚本为 SinkRecent 传了 TriAttention 的 frequency 文件路径，新脚本省略它；
`KVReclamationWorker.__init__` 只在 policy=triattention 时读取该文件，故本组
策略不消费此参数。汇总器验证输入 SHA、arrival、容量、策略与 SLO 没有改变。
batch 改变可改变回收时机，不能由相同 retention 参数推断质量等价。

所有 pilot 也完整保留。下面均排除预热，pilot/control 为一轮，formal 为三轮；
goodput 为满足两项 SLO 的请求数除以全部正式轮总耗时：

| 类型 / batch | tok/s 各轮 | SLO | Goodput req/s | preempt 合计 |
| --- | --- | ---: | ---: | ---: |
| TP 512 pilot | 95.539 | 4/8 | 0.3732 | 0 |
| TP 1024 pilot | 99.772 | 5/8 | 0.4872 | 0 |
| TP 2048 affinity control | 67.501 | 1/8 | 0.0659 | 7 |
| TP 4096 pilot | 61.417 | 0/8 | 0 | 13 |
| TP 1024 formal | 101.409 / 101.431 / 99.911 | **17/24** | **0.5584** | **0** |
| DP skewed 512 pilot | 125.168 | 8/8 | 0.9779 | 0 |
| DP skewed 512 formal | 125.462 / 125.299 / 125.416 | **24/24** | **0.9796** | **0** |

TP formal 平均 100.917 tok/s；TTFT P50/P95/P99 为 1.817/2.798/2.917s，
TPOT 为 21.44/32.82/32.87ms。每轮达标数为 6/6/5，失败请求是每轮的第4、8个，
以及第三轮的第7个，全部因 TTFT；它们的 TTFT 为 2.130–2.952s，TPOT 均低于
80ms。该配置 SLO 继续 **FAIL**，不能因吞吐翻倍而宣称全部达标。

历史 TP Sink 的 batch2048、无绑核三轮为 48.548 tok/s、0/24、36 次 preempt。
本轮同源码、同 affinity 的 2048 control 到 1024 正式结果，抢占 7/轮降至 0；
放大到 4096 反而出现 13 次抢占与更长 TPOT。因此支持选择较小 chunk 来减少
长 prefill 阻塞和容量反复冲突，不支持“批量越大越快”。旧报告的 48.548→100.917
同时包含源码收尾、主机放置和 batch 配置变化，只作为整体参考，不归因于单一改动。

DP skewed formal 平均 125.392 tok/s，TTFT P50/P95/P99 为
0.675/1.734/1.743s，TPOT 为 17.49/27.04/27.08ms。
replica0 处理18个长请求，18/18达标，TTFT P99=1.743s；replica1六个短请求
6/6达标，TTFT P99=56.72ms。历史 batch2048 结果为 101.271 tok/s、18/24，
10 次 preempt、Goodput=0.5929 req/s；新配置三轮全部达标，判 **PASS**。
没有更改人工 6/2 路由，也没有把 peer Pool 并入本地容量。

TP/DP 新正式峰值分别为 19,910,361,088 / 37,515,952,128 bytes，所有 guard PASS。
压力策略只复测 SinkRecent；不将新结果外推到 Random、TriAttention、APC 或其他
模型/硬件，不替换原 M2/Prefix 的压力 FAIL。有限的24请求不是生产尾延迟保证。

两条独立 trace 使用相同新配置，完整预热后一轮；仅核查回收，不并入正式吞吐。
从 HTTP 正式时间窗口内的 allocator 原始事件重算如下：

| 诊断 | miss / select / commit | 累计归还 blocks |
| --- | --- | ---: |
| TP scheduler | 19 / 19 / 19 | 2054 |
| DP replica0 | 10 / 10 / 10 | 1025 |
| DP replica1 | 0 / 0 / 0 | 0 |

每次 commit 都有 `released_blocks>0` 和真实 E 缩减；DP peer 不参与回收。
累计归还可包含同一页多次循环使用，不能当作同时新增容量或唯一页数。
底层 canonical ownership、TP completion 与全层重写正确性沿用功能收尾证据，
本轮没有更改这些调用链。压力三轮正式实验关闭 allocator trace。

## 已完成代码的 review 判断

本轮证据不支持继续删除 ownership/completion 检查来解决 TP 慢档，也不支持
为 noop 增加绕过 E/metadata 的特殊执行分支；未增加这类 Runtime 复杂度。
新增参数在已有两个 Serving 脚本共用，既有默认 batch=2048、max seqs=8 保留。
性能汇总中的校验位于实验数据入口，用于阻止负载或口径变化被误报为收益，
不进入 decode 热路径。Triton 的真实调优与功能缺陷修复沿用
[final-review.md](final-review.md)，本轮没有以启动优化冒充 kernel 优化。

## 复现与证据索引

CPU20/21 只适用于本机，换主机应先确认拓扑和可用核心。以下命令串行运行，
每次 guard 都创建新目录，不覆盖旧结果；需要完整 ABBA 时按 full/noop/noop/full
顺序运行 short/eager 命令，每个服务三轮：

```bash
export P1_CUDA_VISIBLE_DEVICES=0,1
source .p1-work/scripts/env.sh

# 将 policy 按 full / noop / noop / full 次序运行。
.venv/bin/python .p1-work/scripts/run_guard.py --stage Perf --name tp-eager-rerun \
  -- @python .p1-work/scripts/serving_reclamation.py \
  --policy noop --pool-gib 8 --workload short --tp 2 --graph-mode eager \
  --full-warmup --worker-cpus 20 21 --repeats 3

.venv/bin/python .p1-work/scripts/run_guard.py --stage Perf --name tp-pressure-rerun \
  -- @python .p1-work/scripts/serving_reclamation.py \
  --policy sink_recent --pool-gib 0.75 --workload long --tp 2 --graph-mode full \
  --full-warmup --worker-cpus 20 21 --max-num-batched-tokens 1024 --repeats 3

.venv/bin/python .p1-work/scripts/run_guard.py --stage Perf --name dp-skewed-rerun \
  -- @python .p1-work/scripts/dp_serving.py \
  --policy sink_recent --pool-gib 1.5 --scenario skewed \
  --worker-cpus 20 21 --max-num-batched-tokens 512 --repeats 3

.venv/bin/python .p1-work/scripts/summarize_performance_followup.py
```

汇总器重算本次归档的四个 ABBA 服务、全部 pressure pilot/control/formal，核对
输入 SHA、到达时间、策略/容量/SLO、真实 affinity、raw token IDs、路由与前后
metrics，分别输出开销和 SLO 状态；证据完整性 PASS 不覆盖性能 FAIL。
汇总脚本按本轮唯一目录后缀查找归档；额外复跑可另存目录，作为新一轮分析。

| 证据 | Perf 下目录 |
| --- | --- |
| 预热后 native cProfile | `20261031-151844-tp-eager-native-steady-cprofile` |
| 预热后 noop cProfile | `20261031-152351-tp-eager-noop-steady-cprofile` |
| 同服务框架切换 | `20261031-152752-tp-eager-same-server-toggle` |
| CPU 交叉干预 | `20261031-153044-tp-eager-core-placement-crossover` |
| ABBA full A | `20261031-153455-tp-eager-affinity-a-full` |
| ABBA noop A | `20261031-153746-tp-eager-affinity-a-noop` |
| ABBA noop B | `20261031-154104-tp-eager-affinity-b-noop` |
| ABBA full B | `20261031-154349-tp-eager-affinity-b-full` |
| TP 512 pilot | `20261031-154520-tp-pressure-batch512-pilot` |
| TP 1024 pilot | `20261031-154706-tp-pressure-batch1024-pilot` |
| DP 512 pilot | `20261031-155051-dp-skewed-batch512-pilot` |
| TP 2048 control | `20261031-155241-tp-pressure-batch2048-affinity-control` |
| TP 4096 pilot | `20261031-155439-tp-pressure-batch4096-pilot` |
| TP 1024 formal | `20261031-155620-tp-pressure-batch1024-formal` |
| DP 512 formal | `20261031-155803-dp-skewed-batch512-formal` |
| TP 独立回收诊断 | `20261031-155951-tp-batch1024-diagnostic` |
| DP 独立回收诊断 | `20261031-160109-dp-batch512-diagnostic` |

还保留早期 `20261031-151421-tp-noop-placement` 和
`20261031-151641-tp-eager-noop-forward-cprofile` 探索；后者含冷路径，不能用作
稳态成本证据。完整 guard 清单在 `performance-followup-summary.json`。
各 run 的 invocation 保存 HEAD、tracked diff 和直接执行脚本的源快照；
affinity Worker 与诊断所导入的 Serving helper 另外复制到 run 根目录。
不声称 guard 会自动捕获任意 ignored 的间接 import。

机器汇总为 `performance-followup-summary.json`，包括19个本轮guard记录，最高
项目显存37,515,952,128 bytes（34.94GiB），均未触发44GiB停止阈值。
运行结束后GPU0/1项目进程已退出，GPU2其他任务保留。提交前脚本ruff
check/format、全部适用pre-commit（含mypy3.10）及`git diff --check`通过；
日志为Perf下`performance-followup-pre-commit-rerun.log`和`summary-verification.log`。
首次hook发现诊断切换脚本使用旧`torch.cuda.synchronize`，现改为同当前设备的
`torch.accelerator.synchronize`；本地已核对该API语义，原实验源快照继续保留。
本轮没有Runtime变更，不重复此前154项功能回归；主张由真实Serving、12对输出
一致性、受控CPU干预及两条allocator诊断验证，没有新增低价值单测。
