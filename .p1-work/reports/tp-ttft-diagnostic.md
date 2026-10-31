# TP pressure TTFT 请求级诊断与停止判断

2026-10-31，基于 `2c3379cc883b40acc6b9cbb9a84d58c85fa38259`。
本轮完成O1的既有trace审计和一轮有效diagnostic，补齐稳定请求ID、chunk、两rank
设备活动与canonical commit的关联。**未修改Runtime，未实施性能优化**：采样轨迹
明显偏离原正式零抢占负载，尚不足以选择能改善原TTFT的具体修法。

原正式TP pressure batch1024仍为 **17/24、零抢占、SLO FAIL**；本诊断不替代该
结论。TP eager 0.868%与DP skewed 24/24仍仅适用于原配置，历史FAIL/PARTIAL和
85条冻结质量样本保留。SLO始终是TTFT≤2s且TPOT≤80ms。

## 既有证据能补齐什么

审计仅只读旧原始数据；记录为
`TTFT/20261031-175743-existing-trace-audit/audit.json`，包含基线、输入hash及缺项。
旧 `Perf/20261031-155951-tp-batch1024-diagnostic` 的10.136193s计时窗内有1071组
allocation check/done、19组miss/select/commit，能按engine id还原q/L/E及提交。
client只保存 `r0-N` 和相对时间，未保存SSE返回id，engine使用随机 `cmpl-...`。
因此不能可靠连接client与engine；按到达顺序猜测的请求映射未用作归因证据。
enqueue、各段实际完成和host/GPU/通信也不在旧trace中，故执行本次最小补采。

## 补采是否代表原负载

固定配置和命令见 [协议](../notes/tp-ttft-diagnostic.md)。源文件、命令、环境、
日志、退出码均在 `TTFT/20261031-180022-tp-batch1024-timeline`；有效run的guard
PASS、exit=0，项目显存峰值19,969,081,344 bytes，未触停止阈值。
两rank实际affinity为CPU20/21，CUDA_VISIBLE_DEVICES为0,1；项目进程结束，其他
用户任务未终止。运行前存在其他任务，未对其资源占用做受控干预。

首次启动归档 `TTFT/20261031-175833-tp-batch1024-timeline` 在模型预热时失败：
诊断Worker误要求原生warmup batch携带真实请求step tag。尚未发送HTTP测量请求，
exit=1，guard日志和错误钩子源码保留。修正预热/真实请求边界后只采到下面一轮
有效数据，没有追加矩阵或第二轮测量。

输入hash、arrivals、容量、策略及batch与旧诊断一致；原helper命令除动态port外
相同。新增wrapper实际替换诊断Scheduler/Worker，启用NSYS和稳定HTTP id，原生
随机内部id保留。真实命令以 `diagnostic-command.json` 为准。

| 证据 | 请求数 | 吞吐tok/s | 抢占 | SLO | 用途 |
| --- | ---: | ---: | ---: | ---: | --- |
| 原正式三轮 | 24 | 均值100.917 | 0 | 17/24 | 原性能结论 |
| 原allocator诊断 | 8 | 101.024 | 0 | 6/8 | 旧回收轨迹 |
| 本次NSYS诊断 | 8 | 52.599 | 6 | 0/8 | 仅诊断，不能当正式回退或优化对照 |

本诊断耗时19.468s，TTFT P50/P95为3.259/9.164s，TPOT P50/P95为98.389/118.648ms。
明显偏差包含额外重算，不能把所有变化都归因于profiler；尚未分离profiler、
instrumentation、shape冷路径和运行时背景争用的影响。一次完整负载预热也未保证
后续所有compactor K特化均热：计时窗中仍有两次约188/190ms的host compact范围。
其中module-load API仅约0.26/0.24ms，其余未精确拆分为编译、cache或其他host工作；
不能把整段写成GPU搬移或直接声称全可消除。

## 请求级分解

8/8请求通过SSE response id唯一匹配Scheduler内部id；全程537个schedule均有
complete和两rank execute，计时窗269个step。CPU/nsys三进程末次clock offset
对齐，首尾offset最大差165.461μs，保留该对齐误差，未据此讨论微秒级因果。
352个计时Graph launch均有设备span，按process+correlation连接无遗漏activity。
派生结果为 `ttft-analysis.json`，原timeline与SQLite保留。

单位ms，四舍五入；prefill列是包含该请求的完整step wall，可能与同batch其他
请求重叠。段间包含decode/reclaim/重新排队，不与GPU或host inclusive再相加。

| client id | TTFT | client→enqueue | 首次排队 | prefill step | 段间 | 输出送达 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| r0-0 | 741.19 | 8.47 | 0.06 | 724.89 | 0.15 | 7.63 |
| r0-1 | 2015.49 | 66.24 | 367.06 | 1549.25 | 19.51 | 13.43 |
| r0-2 | 2681.75 | 133.30 | 1568.82 | 934.16 | 37.66 | 7.82 |
| r0-3 | 3834.87 | 16.63 | 2169.64 | 1559.08 | 75.44 | 14.08 |
| r0-4 | 1135.01 | 24.74 | 199.02 | 902.06 | 0.25 | 8.93 |
| r0-5 | 4388.35 | 185.23 | 640.04 | 3347.64 | 199.99 | 15.45 |
| r0-6 | 4870.56 | 62.72 | 4027.67 | 770.30 | 0.15 | 9.72 |
| r0-7 | 11473.63 | 173.08 | 4387.70 | 6499.41 | 398.55 | 14.89 |

r0-3的首次排队中2112.48ms由前面请求prefill step占用，56.58ms由decode占用；
其自身九段prefill约1559.08ms。r0-7首次等待4387.70ms，其中4170.47ms为其他
prefill step，206.50ms为decode，8.26ms为q=0 maintenance，其余为step间隙。
这证明该采样中的排队与前序prefill相连，不是空闲时allocator单独等待。
client→enqueue包含HTTP、入口处理和Core可接收请求的时机，未细拆成网络开销。

r0-5有两次logical起点回退，19段实际计算18,382 prompt tokens；r0-7有四次回退，
36段计算35,651 tokens。全8条输入49,152 tokens，首token前实际prefill合计86,801。
这与6次preempt对应，不能把重复计算当成原正式零抢占场景的固有成本，也不能
归因于retained KV session重入。其余请求段数为4/9/5/9/5/5，逐段q、起点和完成
时间均在机器结果中。

## host、GPU与通信

91个含prefill的step累计wall为16,099.38ms；下表逐rank统计，host范围嵌套且与
GPU重叠，**不得把各行相加或用host减kernel求“可消除成本”**。

| 计时窗prefill分项 | rank0 ms | rank1 ms |
| --- | ---: | ---: |
| GPU activity union | 15247.31 | 15632.24 |
| 模型collective activity union | 10096.72 | 10598.71 |
| 其他kernel activity union | 5147.59 | 5029.97 |
| execute_model host inclusive | 4402.16 | 3271.48 |
| prepare_inputs host inclusive | 119.27 | 54.35 |
| prepare_attn host inclusive | 29.11 | 12.43 |
| sample_tokens host inclusive | 126.66 | 112.54 |
| AsyncOutput真实等待所在host范围 | 11419.76 | 不执行该output-rank方法 |

85个q≥1000的大prefill step的wall P50为185.43ms；GPU busy P50为177.16/180.46ms，
模型collective为118.05/120.05ms。prefill模型通信主要观测为NCCL BF16 AllReduce；
collective占对应prefill wall约62.7%/65.8%，包含rank等待、发射节奏及采样扰动，
不是纯传输，也不能据此选择新的通信backend或估计可移除收益。普通decode的
176个Graph/rank按完整span计时，图内kernel/通信未展开，不外推上述通信比例。

Scheduler schedule/complete的计时窗wall累计108.10/26.21ms；schedule完成到
Worker execute入口P50为0.265/0.151ms。prefill的output等待thread CPU只有13.89ms，
wall为11.420s，主要表现为真实完成等待，不能通过去掉event等待来优化。

## 回收完成与发布

计时窗29次intent均匹配两rank回收、completion collective、真实输出等待和canonical
commit，含一次q=0 maintenance；累计归还2424 blocks，**不等于唯一页数或同时
新增容量**。它与旧19次/2054 blocks不同，属于本次轨迹差异。

29次host reclaim return都早于同rank最后设备完成；真实完成等待没有被钩子替换。
例step279：rank0 host return至其GPU结束相隔8.005ms，两rank GPU结束后约0.746ms
才canonical commit。该step归还128 blocks，source E=4103→2051，结果身份与E匹配。
所有回收的两rank设备活动结束均早于commit；不是只拿host return证明完成。

回收phase设备busy累计13.12/43.31ms，其中completion collective为2.43/33.77ms，
compactor kernel约10.08/8.93ms；host reclaim inclusive为509.86/479.58ms，包含
前述cold shape与发射等待。不能用差值全算优化空间，也不能因CTA occupancy重写
compactor。本轮关联不替代既有全层payload、慢rank/fail-stop及reuse正确性证据。

## 决策、验证与真实缺口

O1实施了trace审计、一次有效采样及请求/step/CUDA分析；不改Runtime、不扫batch，
也不再追加采样。主要原因是尚未取得代表原零抢占负载的成本证据，无法据此选择
预期能改善原TTFT的修法。没有候选改动，所以未执行同配置优化对照，没有声称
TTFT、TPOT或吞吐改善。原17/24 FAIL仍是当前正式结果。

O2未测TriAttention真实Runtime完整评分/normalization/selection，不拿SinkRecent
数据替代它；O3未改metadata对象/CPU buffer，当前诊断没有证明代表性净收益。
不新增noop旁路、backend/dispatcher、DP借页/migration/routing hint，不移除任何
ownership、hash、source E/row、结果身份、全rank完成或mutation后fail-stop边界。

实际运行命令包括协议内guard命令，以及：

```bash
/opt/nvidia/nsight-systems/2024.7.1/bin/nsys export --type sqlite \
  --output "$run/trace.sqlite" "$run/ttft.nsys-rep"
.venv/bin/python .p1-work/scripts/analyze_tp_ttft.py "$run"
.venv/bin/python -m ruff check .p1-work/scripts/analyze_tp_ttft.py \
  .p1-work/scripts/profile_tp_ttft.py .p1-work/scripts/tp_ttft_hooks.py
.venv/bin/python -m ruff format --check .p1-work/scripts/analyze_tp_ttft.py \
  .p1-work/scripts/profile_tp_ttft.py .p1-work/scripts/tp_ttft_hooks.py
```

这里的 `run` 是上述18:00:22归档绝对路径。export/analysis均exit=0，identity/step
及GPU span join PASS。初版分析错误地要求非output rank普通postprocess必须先于
Core update；源码表明该要求只对本轮观察的回收完成链成立，故改为记录普通尾部
而在reclaim commit核对全rank完成。初版失败保留 `analysis-initial.log`；未改Runtime。
最终适用pre-commit、Markdown本地链接/源码symbol、原Review hash及diff检查结果
见该run的 `validation/`；最终提交与文件hash见 `final-state.json`。

适用pre-commit全部PASS（含mypy3.10、Markdown）；首轮仅分析脚本的 `import re`
违反项目规则，改用已有 `regex` 后通过。7份变更Markdown的57个本地链接、19个
关键源码/工具symbol通过，独立Review及审计记录中旧输入hash不变，`git diff --check`
通过。分析脚本对本次真实原始数据重算通过；没有Runtime变更，不重复既有28项
功能回归或追加GPU/质量矩阵。

仍缺少：原零抢占正式轨迹的逐阶段代表性成本、纯链路传输与rank等待的因果分离、
图内通信、cold shape完整host拆分以及任何优化后的无profiler收益证据。本轮未新增
GPU payload/continuation、质量评测或组合Gate；这些限制保留，不以诊断PASS代替
性能或质量PASS。
