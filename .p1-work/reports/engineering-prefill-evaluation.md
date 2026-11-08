# Prefill 预算公平性：同卡基线与修复评估

2026-11-08，起点 `eef5da27d1`。本轮只用物理 GPU0，Worker 主线程 CPU20。
**明确修复的是短请求的计算排队及首步 decode 容量边界；没有证据要求重写 TP 通信。**
在本轮混合负载中，protected 开启已有 512-token prefill 限额后，短请求 TTFT
正反向均改善约74%，吞吐较自身不限额配置下降约1–2%；相同限额下距 fixed
吞吐不足0.3%。所有八组各24/24 SLO，但历史 TP2 负结果保留。

## 1. 问题、改动与源码边界

此前的两轮回收只保证长请求有空间入场。Scheduler 先遍历 running，再处理
waiting；未启用 `long_prefill_token_threshold` 时，running 长 prefill 可以占用
1024-token batch 的几乎全部剩余预算，短请求即使有入场路径也得不到计算机会。
这是容量准入与计算预算公平性的两个问题，不能都叫作“TP 实现退化”。

本地已有每请求 prefill 限额。本轮在 Serving/DP wrapper 显式透传该参数，
比较0与512；fixed/protected 同样设置，未把512改成全局默认。新增低扰动 CPU
时间戳和请求ID关联，将 TTFT 拆为入口、首次调度排队、prefill结果消费和回包。
不增加逐步 trace、KV payload clone 或 GPU 同步。实现见
[Scheduler](../../vllm/v1/core/sched/scheduler.py)、
[Serving wrapper](../scripts/serving_reclamation.py)、[work hook](../scripts/dp_hooks.py)。

扩展行为回归还发现了独立的容量边界：满页 prompt 只证明 prefill 能放下，紧接的
第一个 decode token 需要新页，可能立即抢占。protected full-prefill admission
和 reclaim retry 现计入首个 decode token，仅在请求确实还需输出时多留必要的一页；
不预留全部未来输出，不改变 fixed 分配行为。测试同时修正中间 prefill chunk
不应模拟产 token、`max_model_len` 不应误设成 batch 大小的 fixture 状态。

保留单次私有KV删除≤20%、相对已计算逻辑私有历史保留≥50%、每轮最多两个
victim、最多两轮及真实 decode 进展门控、完整 prefill 容量预留和 backfill2。
没有 prefill 中删除KV、usage tie-break、新评分器或 Worker/TP completion 改动。

## 2. 同卡协议与口径

环境继承 prompt3/4：现有 `.venv/bin/python`、Python3.10.20 / Torch2.11.0+cu129 /
Triton3.6.0、Qwen3-8B BF16、MRV2、FA2/NHD/B16，未安装升级依赖或下载模型。
`P1_CUDA_VISIBLE_DEVICES=0` 后重新 source env；HTTP子进程启用multiprocessing，
单卡TP1。GPU1有其他任务，GPU2未使用；本轮真实 TP2、DP2 为 **NOT_RUN**。

每组由 `run_guard.py` 同锁串行启动：1.5GiB KV pool、reserve0、APC关闭、
batch1024/maxseq8、`sink_recent`、FULL_DECODE_ONLY、capture `[1,2,4,8]`。
每组先32-token/8-output小预热，再完整负载预热，最后正式3轮；不含模型启动与
预热时间。protected均rounds2/backfill2，其余参数相同。

每轮原负载 `[4096,8192,128,128]×2`，到达秒数
`[0,.3,.6,.9,5,5.3,5.6,5.9]`，每请求128输出、temperature0/seed0/ignore_eos。
没有换数据集、扫描更多threshold、扩大85条质量样本或重跑旧矩阵。

正向 fixed0→protected0→fixed512→protected512；反向完全逆序。每组汇总全部
24个正式请求，不挑最好一轮。SLO仍逐请求同时 **TTFT≤2s且TPOT≤80ms**。
tok/s = 全部输出token / 三轮实际负载耗时总和，包含到达间隔和等待，
表示该固定到达负载的完成速率，不是最大饱和吞吐。

独立[分析器](../scripts/analyze_prefill_evaluation.py)验证Runtime/helper/input
归档hash一致、实际GPU/affinity/Graph配置、预热剔除、请求ID一一对应、
`scheduled_tokens=prompt+127+recomputed`、counter/commit计数一致，重新计算SLO、
分长度指标和时间拆分。每组75264个prompt token、78312个scheduled token。

## 3. Correctness：受影响范围 PASS

```bash
export P1_CUDA_VISIBLE_DEVICES=0
source .p1-work/scripts/env.sh
.venv/bin/python -m pytest tests/v1/core/test_scheduler.py \
  tests/v1/core/test_kv_cache_utils.py \
  -k 'reclamation or pressure_protected or admission_bypass' -q
```

92 passed /224 deselected，覆盖现有回收契约、两轮门控/保护边界、0/512短请求
实际进展、页对齐首decode预留及单token输出无需预留。DP原本地 `-k dplb`
fixture为13 passed /16 deselected；没有把历史15项计为本轮结果。
适用pre-commit和manual mypy-3.12通过。日志在下述B0。

八组真实单卡Graph Serving共192个正式请求均完成128输出，零抢占、零额外重算，
最多一个victim实际被用到。每种策略0/512逐请求合成输出序列均24/24一致；
这只是该合成负载的机械记录，不能替代质量评估或证明无损。

本轮没有重新跑多卡双victim、延迟completion、物理页改写复用或DP abort套件。
历史已通过的证据见[两轮收尾](engineering-two-round-closeout.md)，其覆盖范围
不自动等于本轮新容量预留组合已通过真实TP2/DP2。

## 4. Micro：NOT_RUN

未改Worker/compactor/TP collective，本轮没有重跑旧micro矩阵，也没有新增
CPU扫描或GPU搬移加速结论。正式work hook仅计动作/估算字节，不测kernel耗时。
历史两轮Worker约3.79/3.80ms、fixed约3.20ms仍见两轮评估，不能拿本轮TTFT改善
覆盖该负结果，亦不能用总搬移字节推导耗时。

## 5. Serving：短请求延迟改善，整体吞吐无加速

表中TTFT为分长度中位数，单位ms；每组短请求12个，4096和8192各6个。

| 顺序 | 策略/限额 | 输出tok/s | 短128 TTFT | 4096 TTFT | 8192 TTFT | 联合SLO | 回收 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 正向 | fixed /0 | 119.320 | 572.5 | 428.5 | 1123.9 | 24/24 | 12 |
| 正向 | protected /0 | 123.108 | 526.6 | 370.6 | 943.4 | 24/24 | 24 |
| 正向 | fixed /512 | 121.180 | 98.1 | 438.2 | 1139.9 | 24/24 | 12 |
| 正向 | protected /512 | 120.841 | 132.7 | 394.8 | 1168.0 | 24/24 | 24 |
| 反向 | protected /512 | 121.692 | 139.5 | 392.5 | 1164.6 | 24/24 | 24 |
| 反向 | fixed /512 | 121.717 | 100.2 | 434.2 | 1130.0 | 24/24 | 12 |
| 反向 | protected /0 | 123.004 | 531.1 | 368.6 | 947.9 | 24/24 | 24 |
| 反向 | fixed /0 | 124.684 | 484.5 | 370.5 | 931.3 | 24/24 | 12 |

protected 0→512：短TTFT下降74.80%/73.73%，8192 TTFT增加23.81%/22.86%，
吞吐下降1.84%/1.07%。fixed同样受益，短TTFT下降82.86%/79.33%；
其吞吐+1.56%/−2.38%，未形成稳定吞吐收益。
相同512限额下protected相对fixed吞吐−0.28%/−0.02%。这支持“本配置短延迟
明显改善、接近fixed完成速率”，不支持“protected吞吐超过fixed”。

八组每个请求TPOT均≤24.85ms，SLO全部达标；goodput为0.932–0.974req/s。
本轮旧0限额也达标，故不能声称SLO由FAIL修到PASS。fixed0正反向自身吞吐相差
约4.5%，长TTFT有波动；保留全部轮次和背景，不把不同日期/卡位的旧TP2数字
混入本轮同卡A/B，也不推断本轮解决了所有旧TP2慢路径。

## 6. 机制核对：等在首次调度之前

每组12个短请求都在同波8192 prefill完成前到达Scheduler。
protected0中，它们全部在长prefill结果消费后才首次调度；protected512中，
全部12个在此前已开始计算，正反顺序相同。这是实际请求进展，非仅剩余预算。
fixed0也有12个在长prefill最后forward完成前开始，不能把fixed0描述成严格串行。

| protected短请求，中位ms | 正向0→512 | 反向0→512 |
| --- | ---: | ---: |
| Scheduler接收→首次调度 | 408.8→29.1 | 410.9→29.4 |
| 首次调度→prefill结果消费 | 39.8→68.3 | 39.8→68.5 |
| 客户端TTFT | 526.6→132.7 | 531.1→139.5 |

改善主要来自缩短排队；短prefill自身并未变快。512同时增加长prefill的chunk数，
付出更多调度/小批次成本，因此长请求稍慢。这是可解释的公平性取舍。
分段分别取中位数不会相加恰好等于总中位数；逐请求各段之和已复算与TTFT误差
<1ms。`prefill结果消费`是Scheduler在CPU消费forward输出的边界，不是纯CUDA
kernel时间；入口段包含HTTP、engine传输和调度忙碌，不能全归因网络。

首decode预留的CPU反例和GPU组合回归均通过，但本轮没有单独关闭该修复的GPU
反事实，不能把上述74%全部归因于该容量修复；0/512对照测的是预算公平性。

## 7. 上游依据、推荐与停止

已核查官方PR正文/状态并归档 `upstream-pr-review.json`：

- [vLLM #57951](https://github.com/vllm-project/vllm/pull/57951)，已合并：
  说明长prefill限额避免预算被耗尽，并增加独占豁免。本地已有基础限额，先启用验证。
- [vLLM #49075](https://github.com/vllm-project/vllm/pull/49075)，open：
  V1部分prefill并发参数执行问题；本轮不依赖只调这些参数来修复。
- [SGLang #45685](https://github.com/sgl-project/sglang/pull/45685)，open：
  chunk与物理容量约束相关，但不是本地代码可直接套用的补丁。

本轮推荐混合负载显式使用 `--long-prefill-token-threshold 512`；如需protected，
继续rounds2/backfill2和20%/50%约束。该设置用约1–2%本负载完成速率代价换取
约74%短TTFT改善，长请求仍低于2s。它利用vLLM已有能力，不能包装成新调度算法。
源码默认0及fixed/requests默认保留，没有宣称512适合所有负载。

停止扫描threshold或添加独占豁免：本轮短请求进展目标已实现，总体代价已量化。
独占场景的进一步优化可以独立研究，不在本轮叠加以追求最好数字。
DP usage tie-break仍STOPPED / NOT_RUN；GPU1占用使TP2/DP2本轮NOT_RUN，
没有新增多卡性能或全局质量结论，prefill中删KV继续暂缓。

## 8. 原始产物、资源与日志异常

B0：`/data_ssd/qcrs/experiments/p1-v8/Engineering-Prefill-B0/20261108-134721-initial`。
最终：`/data_ssd/qcrs/experiments/p1-v8/Engineering-Prefill-Final/20261108-141656-closeout`。
`serving-manifest-{forward,reverse}.json`含精确命令/退出码；`analysis-all.json`
含8组、每轮、逐请求分段及源码hash；`run-audit.json`核对日志和背景。
全部正式run位于 `/data_ssd/qcrs/experiments/p1-v8/Prefill-Serving/`：

| run目录 | 组 |
| --- | --- |
| `20261108-140238-gpu0-forward-fixed0` | 正向fixed0 |
| `20261108-140442-gpu0-forward-protected0` | 正向protected0 |
| `20261108-140552-gpu0-forward-fixed512` | 正向fixed512 |
| `20261108-140704-gpu0-forward-protected512` | 正向protected512 |
| `20261108-140846-gpu0-reverse-protected512` | 反向protected512 |
| `20261108-140959-gpu0-reverse-fixed512` | 反向fixed512 |
| `20261108-141111-gpu0-reverse-protected0` | 反向protected0 |
| `20261108-141223-gpu0-reverse-fixed0` | 反向fixed0 |

8次GPU guard全部PASS，项目NVML合计峰值17.564453GiB，未触及44GiB停止或45GiB
硬限。前后进程、GPU UUID、topology、1秒背景util/memory/power及CPU affinity
已归档；GPU1其他任务始终保留。GPU0早期背景约703MiB、后期约126MiB，
没有据此假定两序系统背景完全一致。仅约束Worker主线程，未全局重绑helper。

两个protected0的server.log在所有请求和末次counter读取之后、wrapper主动SIGTERM
关闭engine期间出现`EngineDeadError`；错误行均在`shutdown triggered`之后，
不是请求中途失败，原日志完整保留。八组`_compact_stable`JIT均出现在完整负载
预热结束之前，没有把首次JIT时间混入正式结果。
开发CPU/fixture/lint失败、首次提交hook失败及其修正均保留B0；不删除或改写历史
FAIL/PARTIAL。本地阶段提交与用户patch保护审计见[收尾](engineering-prefill-closeout.md)。
