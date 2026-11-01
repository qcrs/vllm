# TP 首 token 关键路径与动态长度修正

2026-11-01，本轮只修复已定位的无必要长度特化，未重排 prefill 或盲扫 chunk。
正确性 PASS；消除跨 E 对齐的新 kernel 变体；正式 Serving 无收益、SLO 仍 FAIL。
这与[历史 TP2a/TP2b micro](engineering-next-tp.md)分开判定，不重跑旧矩阵。

## TTFT 分解与剩余瓶颈

[host-only 诊断](../experiments/Engineering-Refinement-TP-Diagnostic/20261101-143006-grouped-host-ttft)
沿用 TP2、CPU20/21、long/FULL、batch1024、seqs8、0.75GiB/rank、fixed，完整预热后
一轮。`profile_tp_ttft.py --nsys-child` 绕过 NSYS，只采同主机 host 时间戳与
Scheduler/Worker hooks；`analyze_tp_ttft.py --host-output` 按请求稳定 ID 关联。

| 第一波最后请求 r0-3 | 时间 |
| --- | ---: |
| 入口到 Engine enqueue | 93.029ms |
| enqueue 到首次 schedule | 1406.855ms |
| 自身参与的 prefill steps | 1196.899ms |
| prefill 段间等待 | 60.232ms |
| 最后 step 完成到客户端首 token | 14.010ms |
| 总 TTFT | 2771.025ms |

分项之和与原请求 TTFT 校验一致；本轮52个 prefill steps、23次回收、零抢占，
每请求 prefill 总量等于输入且无重算起点。第一波24576 prompt tokens 的最后
首 token 约在3.658s，约2.9s的 deadline 仍差约0.758s。必要窗口吞吐约8.47k
prompt tokens/s，尚不计 decode 等工作；本轮未证明设备的独立 prefill 上限。
初始排队与 prefill 均有显著贡献，不能据此认定公平重排足以全部达标。

这里的 own prefill 是 inclusive step wall time，可能包含其他请求、提交与等待，
不等于该请求的纯 GPU forward。host 的 execute_model 与 AsyncOutput 含提交和
等待，阶段有嵌套，两 TP rank 不能相加。缺少设备级分解的计算/通信细因保留
UNKNOWN；不把相减余量全部算成可优化开销。

## 找到并修复一个冷路径

第二波 step416 在约5.896s开始，wall3465.987ms：r0-3 的 E4160→K2080，两个
rank reclaim 均约3457ms，thread CPU 约2825/2835ms，compact仅约1.52/10.81ms。
该 step 是 decode/reclaim，未混入上表第一波 prefill；其后请求的入口和排队会
受同一阻塞影响。本次诊断因此只有4/8 SLO、75.437 tok/s，不能当正式基线。

本地 Triton3.6 JIT 对普通整数也按16整除性特化。预热遇到非16对齐 E，而 timed
遇到4160。在该 step 对应估算 wall 窗口，缓存生成9个 `_grouped_raw` 和1个
`_grouped_normalize`；与另一对齐状态比较，TTIR 仅 `%E` 的
`tt.divisibility=16` 属性不同，去掉这一个属性后逐字相同。

20份原 TTIR/metadata、hash、mtime、成对比较见诊断目录
`jit-cache-evidence/index.json`。wall 窗口用读取时同主机 wall/monotonic offset
换算，属于估算，非原生精确 wall 事件；不把3466ms全归为已精确测定的编译时间。

E 只作 mask bound，不决定指针地址对齐，故仅在
`vllm/v1/worker/gpu/kv_scoring.py` 两个 grouped 入口声明
`@triton.jit(do_not_specialize=["E"])`。保持 FIRST=0,4,…,32、stride/head/group
常量、指针对齐、原 FP32 运算、SUM/selection/broadcast 和 completion/commit。
不通过枚举所有 E 预热隐藏问题。调用链见[scoring 源码记录](../source-analysis/m2-scoring.md)，
测量口径见[TP 协议](../notes/tp-protocol.md)。

## 正确性与 cache micro

- [单卡独立 complex reference](../experiments/Engineering-Refinement-TP-Check/20261101-143905-dynamic-length-reference)：
  扩原 suite 为 E95/96，覆盖 partial tile、GQA、跨四层组、near-cutoff，
  21 passed / 14 deselected；guard PASS，峰515,899,392 bytes。
- [真实8B TP2 FULL continuation/free/reuse](../experiments/Engineering-Refinement-TP-Check/20261101-144114-dynamic-length-real-continuation-reuse)：
  E272→K127、E143→K62，两次回收全36层 retained payload，47次/rank FP32
  attention reference，继续生成、完成后释放与真实 rewrite/reuse；无 early free，
  最终32页归还，两 rank 计划一致。guard PASS，峰18,165,530,624 bytes。
- pre-commit 与 manual mypy-3.12 PASS；初次长行/格式失败保留在
  [B0](../experiments/Engineering-Refinement-B0) 的 `tp-dynamic-precommit-first.log`。

同一 scorer 合成36层 local4heads KV，真实 calibration，按 E4159→4160→4161→4159
连续调用；Triton `jit_cache_hook` 记录分组 kernel 的进程 cache miss 与特化数据。

| E 序列 | 修正前 cache misses | 修正后 cache misses |
| --- | ---: | ---: |
| 4159，首次 | 10 | 10 |
| 4160，跨16对齐 | 10 | 0 |
| 4161 | 0 | 0 |
| 4159，再次 | 0 | 0 |

[before](../experiments/Engineering-Refinement-TP-Micro/20261101-143836-scalar-specialization-before)
/ [after](../experiments/Engineering-Refinement-TP-Micro/20261101-144026-scalar-specialization-after)
四次 score 和 KeepPlan SHA 全部逐bit一致，峰均895,483,904 bytes。cache miss
可能加载已有磁盘产物，不等于每次实际编译；首次 wall994ms/4563ms的缓存状态
不同，不能计算冷编译提速百分比。这里只证明取消 E 对齐的新变体，不新报
完整 reclaim micro 提速；历史 TP2a8.5–13% / TP2b40–52% 仍限原受测形状。

## 正式 Serving 与停止

相同8B BF16、FA2/NHD/B16、TP2、FULL_DECODE_ONLY[1,2,4,8]、CPU20/21、
APC/async off、fixed retention=.5、reserve0、0.75GiB/rank、batch1024、seqs8。
两次完整预热后各3轮×8，trace/profiler/counts hooks off；正式仅 scorer 的上述
两个 decorator 不同。没有清缓存，baseline 已经过诊断产生的 E 对齐磁盘编译，
因此本组对照不复现3.46秒冷路径，也不检验其修复百分比。

| 正式 run | 三轮 output tok/s | SLO | TTFT max | 逐请求平均 TPOT max |
| --- | --- | --- | --- | --- |
| [baseline](../experiments/Engineering-Refinement-TP-Formal/20261101-143644-aligned-cache-baseline) | 101.350 / 101.536 / 101.424 | 18/24 | 2.784s | 32.768ms |
| [dynamic E](../experiments/Engineering-Refinement-TP-Formal/20261101-144305-dynamic-length-candidate) | 100.580 / 100.662 / 100.816 | 16/24 | 2.847s | 33.142ms |

合并吞吐101.436→100.686 tok/s（-0.740%）；goodput0.594→0.524 req/s。
没有 Serving 收益，SLO 退化原样记录；新增失败 r0-6/r1-6 的 TTFT 为
2.01582/2.01069s，不放宽2秒门槛，也不以吞吐下降不足5%称 SLO PASS。
24条输出 token IDs 一致，timed均零抢占、prompt counter准确为49152/轮，
无额外重算计数证据；formal没有逐请求trace，重算细因不作额外推断。
warmup各1次抢占另计。ITL>80ms为243/3048与252/3048，仅辅助观察，不改 SLO。

两次 guard PASS，峰均19,935,526,912 bytes。逐请求失败、counter、goodput、
输入/源码/helper hash及可复算脚本见 stage 根目录 `summary.json` /
`analyze-formal.py`，初次 protocol 比较误含动态端口的 FAIL 也保留。

本轮保留动态长度的合理修正，不将它包装成 Serving 提速。prefill 排队/执行
仍为主缺口；没有单项调度改动可填平约0.76秒的证据，停止追加反向、chunk/
权重搜索和设备优化。本轮未证明纯 prefill 硬件上限或最佳调度；历史 TP2b
正反向17/24 FAIL、旧 profile PARTIAL 继续保留。质量85条样本冻结，payload/
输出一致不等于有损质量保证。
