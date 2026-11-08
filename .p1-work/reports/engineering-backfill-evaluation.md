# Protected Admission Backfill：TP 瓶颈定位与评估

2026-11-08，起点 `6b06d384b5`，实现提交 `0ca2e5c80d`。**已修复 protected 中长队首
连带阻塞短请求的问题；长请求容量不足仍在，总吞吐仍低于 fixed。** 匹配逻辑 KV
容量后，单卡与 TP2 复现同一 SLO 模式，主要瓶颈定位于容量准入和 FCFS 队头阻塞。
当前无证据把主退化归因 TP collective；这不等于排除所有 TP 性能问题。

## 根因与实现

[上一轮容量诊断](engineering-progress-controller.md)有直接轨迹：池可用681页，
8192-token队首需512页，free424，缺88；唯一4097-token victim在20%/50%边界内
最多净释放52页，仍差36页。完整prefill admission不能启动它，FCFS的`break`又
挡住本可入场的短请求。fixed按retention=0.5可释放更多页，protected单次最多删20%、
逻辑已计算私有上下文至少保留50%；更严格的保护本身不保证更快。

本轮新增默认0的`admission_bypass_limit`，仅pressure/protected + FCFS显式生效。
分配失败且没有合法reclaim plan时，队首暂时让位，后方每成功入场一次消耗额度，
上限2；失败候选不消耗。额度耗尽后停止后方新admission，原队首仍优先重试，
入场/abort后清状态。不改完整prefill准入、已接纳请求reservation、20%/50%、
最多两个victim及Worker/compactor/TP completion。

另修复DP交互：被移到skipped队列的KV-blocked head仍属于capacity-bound，不能因
waiting为空而重新触发prefill节流。abort清理用`self.requests`做O(1)查询。
有界backfill限制新增入场次数，不提供无限输出场景下的墙钟无饥饿保证。

上游参考（2026-11-08均为open PR）：

- [vLLM #33499](https://github.com/vllm-project/vllm/pull/33499)：同类KV队首阻塞，
  参考有界绕过、按成功入场计数及DP容量满状态处理。本地仅protected opt-in，cap2，
  未照搬上游cap8。
- [SGLang #45685](https://github.com/sgl-project/sglang/pull/45685)：物理池无空闲页时
  暂挂不能继续的chunked prefill，让其他工作推进；没有在prefill中删除KV，仅作容量
  感知调度参考。prefill中删除KV按用户要求暂缓，本轮NOT_RUN。

## Correctness：本轮CPU / 静态检查PASS

最终归档：`/data_ssd/qcrs/experiments/p1-v8/Engineering-Backfill-Final/20261108-112114-closeout`。

| 检查 | 结果与范围 |
| --- | --- |
| CPU Scheduler/BlockPool/预算/DP节流 | 107 passed /194 deselected；成功入场额度、跨step耗尽、队首恢复/abort、DP容量guard及既有回收contract |
| 本地fixture | 只把旧测试离线facebook/opt-125m配置替换为/data/models/Qwen3-8B，不加载权重；非完整上游suite |
| pre-commit | 本轮4个源码/测试/测量文件通过，含Ruff、format、mypy等适用检查 |
| manual mypy-3.12 / diff whitespace | PASS |
| Serving运行 | TP1/TP2六次guard PASS，均完成三轮，零抢占/额外prefill重算 |

CPU精确入口见归档`run-cpu-local.py`，测试范围为：

```text
tests/v1/core/test_scheduler.py tests/v1/core/test_kv_cache_utils.py
-k 'reclamation or reclaim or pressure or admission_bypass or backfill or throttle or prefills_gating'
```

最终日志：`cpu-local-final.log`、`lint-final.log`、`mypy312-final.log`。
保留原配置不可用的`cpu-final.log`（50 passed /6 failed）、DP guard修正前真实
回归失败`cpu-regression-before-fix.log`、`format-development.log`和独立复算脚本
初次错误`serving-audit-development-fail.log`，不以最终PASS覆盖开发FAIL。

真实TP2单/双victim、延迟completion、Graph replay、页面改写复用和DP2
abort/continuation沿用[上一轮正确性证据](engineering-progress-evaluation.md)，本轮
未重跑这些场景。85条冻结质量样本未扩大，未新增质量评估，不声称质量保持。

## Micro：NOT_RUN

本轮修改admission顺序，没有修改kernel/Worker回收路径，未新增microbenchmark。
不把CPU行为或Serving短请求改善描述为决策、复制或回收链加速。此前zero-copy与
Worker成本按[原测量范围](engineering-progress-evaluation.md)保留。

## Serving：短请求进展改善，吞吐仍低于fixed

新mixed是定位长短请求队头阻塞的合成Serving负载，不是质量数据集：

```text
prompt tokens: [4096,8192,128,128,4096,8192,128,128]
arrival seconds: [0,0.3,0.6,0.9,5,5.3,5.6,5.9]
output tokens/request: 128
```

每种配置完整warmup后测三轮×8请求=24。每请求仍须同时满足TTFT≤2s和TPOT≤80ms；
吞吐/goodput均为三轮算术平均，goodput只计联合SLO通过请求。统一Qwen3-8B BF16、
MRV2、FA2/NHD/B16、APC/async off、FULL_DECODE_ONLY Graph `[1,2,4,8]`、
batch-token1024、seq8。

TP1用物理GPU1、Worker主线程CPU20、KV1.5GiB；TP2用物理GPU1/2、Worker主线程
CPU20/21、KV0.75GiB/rank。两者日志均为`GPU KV cache size:10,912 tokens`，逻辑
容量匹配。其他helper threads不声称全部绑核。GPU0未用；六次均经共享guard串行，
项目合计NVML峰值18.5625GiB，低于44/45GiB限制。

| 指标 | fixed | protected strict | protected + backfill2 |
| --- | ---: | ---: | ---: |
| TP2联合SLO | 24/24 | 6/24 | 18/24 |
| TP2输出吞吐tok/s | 124.39958 | 107.40962 | 107.13192 |
| TP2 goodput req/s | 0.971872 | 0.209784 | 0.627726 |
| TP1联合SLO | 24/24 | 6/24 | 18/24 |
| TP1输出吞吐tok/s | 124.92321 | 101.05466 | 100.65318 |
| TP1 goodput req/s | 0.975963 | 0.197372 | 0.589765 |

全部请求TPOT<80ms，SLO失败都来自TTFT。分长度结果两种部署相同：

| prompt长度 | 请求数 | strict SLO | backfill SLO |
| --- | ---: | ---: | ---: |
| 4096 | 6 | 6/6 | 6/6 |
| 8192 | 6 | 0/6 | 0/6 |
| 128 | 12 | 0/12 | 12/12 |

backfill下TP2短请求TTFT0.0516–0.0612s，长请求约2.9219–2.9413s；TP1短请求
0.0355–0.0628s，长请求约2.8691–2.9720s。短请求从被连带阻塞恢复达标，长请求仍
等待容量，backfill总吞吐也未高于strict。fixed在本负载仍更好。

所有组抢占及额外prefill/recompute tokens均0。canonical reclaim commits：TP2
strict/backfill均0，fixed每轮4；TP1 strict0、backfill每轮1、fixed每轮4。故TP2
改善来自admission顺序，不能归因物理回收；TP1也不能单独证明那一次回收的收益。
SinkRecent的scorer_calls可为0，不能用它代替canonical commit计数。

正式目录（相对于`/data_ssd/qcrs/experiments/p1-v8/`）：

- `AdmissionBackfill-TP2/20261108-095106-mixed-fixed-gpu12`
- `AdmissionBackfill-TP2/20261108-095234-mixed-protected-strict-gpu12`
- `AdmissionBackfill-TP2/20261108-095409-mixed-protected-backfill2-gpu12`
- `AdmissionBackfill-GPU1/20261108-110415-mixed-fixed-matched-capacity`
- `AdmissionBackfill-GPU1/20261108-110535-mixed-protected-strict-matched-capacity`
- `AdmissionBackfill-GPU1/20261108-110709-mixed-protected-backfill2-matched-capacity`

各run保存protocol/invocation、源码hash、逐请求结果、服务日志、Worker affinity及
资源轨迹。最终归档`audit-serving.py` / `serving-audit.json` / `serving-audit.log`
独立复算输入hash、同部署三组源码、SLO/计数、GPU及资源，PASS。

## 源码版本与停止结论

每种部署内部三组Runtime源码相同；TP2与TP1之间新增pressure/protected配置合法性
检查，有效实验配置执行分支不变。六次GPU测量后做了O(1) abort清理、DP capacity
guard修正和格式化，差异保存在`final-source-vs-serving.patch`，最终源码hash在
`serving-audit.json`。没有把六次测量描述为最终字节相同源码GPU复验。六次TP1/TP2
不启用DP的`throttle_prefills`，该交互修正由CPU回归验证，本轮无新DP2 Serving。

DP known_work在W1的正反向有限收益及W0不稳定结果沿用[上一轮评估](engineering-progress-evaluation.md)。
本轮未新增DP路由；usage tie-break缺重复新鲜同分错选后发生miss/抢占的证据，仍为
**STOPPED / NOT_RUN**。

本轮交付容量不足时的有界短请求进展，未达到protected超过fixed的总体吞吐目标。
backfill默认0；长请求容量/吞吐标记未解决，停止扩大参数搜索。prefill中删除KV按
用户要求暂缓。历史FAIL/PARTIAL、旧矩阵及冻结质量样本保留。
