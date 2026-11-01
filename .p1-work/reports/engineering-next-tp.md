# 下一阶段 TP 评分执行

2026-11-01。TP2a 保留，正确性与完整回收 micro 通过；Serving 仍为
SLO_FAIL，未证明稳定吞吐收益。TP2b 的独立探针结论在下文追加。

## TP2a 实现及数值契约

`TriAttentionScorer.shared_scores` 仍按层执行 raw score、eligible 域
`torch.var_mean(correction=0)`，将层权重累加放入
`_normalize_reduce_scores`。删除 `layer_total` 与独立 `total.add_`。
归一化仍用 FP32、`sqrt`、`div_rn` 与零方差分母1；关闭隐式 fusion，末尾显式
`tl.fma(layer_score, FP32_weight, previous)` 复现 Torch alpha-add 的舍入。
层序、独占 head owner、全局 SUM、root stable selection/broadcast、compactor 和
device completion 均保留。无跨层 atomic。

`test_shared_frequency_accumulation_matches_torch_alpha_add` 对2/36层验证逐bit
累加与临界计划；已有独立复数 reference 覆盖非连续页、partial tile、非2幂 GQA、
零方差和 L≠E。新增测试四问见 [test-design](../notes/test-design.md)。

## 正确性证据

- [单卡 GPU1 reference](../experiments/Engineering-Next-TP-Check/20261101-101959-fused-layer-reference)：
  9 passed / 14 deselected，NVML峰515,899,392 bytes。
- [真实 TP2 完整 heads reference](../experiments/Engineering-Next-TP-Check/20261101-102043-layer-accumulation-collectives)：
  3 cases × 2 ranks，SUM及广播计划一致，峰1,329,594,368 bytes。
- 真实8B FULL_DECODE_ONLY continuation/free/reuse：
  [运行归档](../experiments/Engineering-Next-TP-Check/20261101-103148-fused-layer-real-continuation-reuse)。
  以该目录的 `result.json`、`evidence.json` 和 rank validation 为实际验收证据。

## 完整链 micro

基线为当前 T1，KV为合成，形状来自已冻结的真实 Runtime shape。3次预热、10次
Events；每个配对样本先取两rank最大值再取median，不将隔离阶段相加。

| 顺序/实现 | E2051 ms | E4100 ms | E8194 ms | 实测 scorer kernels/rank |
| --- | ---: | ---: | ---: | ---: |
| 正向 T1 | 4.76094 | 5.00539 | 5.67794 | 146 |
| 正向 TP2a | 4.12160 | 4.33842 | 5.00250 | 110 |
| 反向 TP2a | 4.2949 | 4.4738 | 5.0977 | 110 |
| 反向 T1 | 4.7570 | 4.9560 | 5.5705 | 146 |

相对T1约8.5–13%完整回收改善，三形状没有>5%退化；全部A/B、两rank、
all-reduce/reduce-root计划hash一致。源文件、hash、host submit、全部样本及计划见
[micro汇总](../experiments/Engineering-Next-TP-Micro/tp2a-summary.json)。
每run峰4,173,332,480 bytes。删除36KiB/rank的layer_total；其余workspace保持。
launch audit 在计时前预热过scorer，因此 `first_reclaim_in_process` 不是全冷scorer。
reduce-root仅benchmark，不改Runtime collective。

## Serving 及失败记录

TP2、CPU20/21、Qwen3-8B BF16、batch1024、seqs8、0.75GiB/rank、固定retention=.5、
reserve0、APC/async off、FULL_DECODE_ONLY[1,2,4,8]。完整warmup后3×8，
trace/profiler/payload clone off。`--reclamation-counts` 只在计时区间外查询诊断RPC，
计数不增加GPU同步；helper源码随run保存。

| 实现 | 3轮 output tok/s | timed SLO | scorer/rank及commit每轮 |
| --- | --- | --- | --- |
| T1 | 98.5111 / 100.0113 / 100.2122 | 15/24 FAIL | 24 / 23 / 23 |
| TP2a | 99.9403 / 100.1566 / 100.4054 | 15/24 FAIL | 24 / 24 / 23 |

两run timed零抢占，prompt counter每轮49152；无法可靠细分重算，记UNKNOWN。
warmup各25次scorer/commit、2次抢占，另列而不并入timed。
吞吐差异落在轮次波动内，停止扩大Serving对照，不重启旧TTFT矩阵。
NVML峰两者均19,935,526,912 bytes。TTFT/TPOT各分位、counter及全部轮次见
[Serving汇总](../experiments/Engineering-Next-TP-Formal/tp2a-serving-summary.json)。

原始A为[当前T1](../experiments/Engineering-Next-TP-Formal/20261101-102630-current-t1-counted-pressure)，
B为[TP2a](../experiments/Engineering-Next-TP-Formal/20261101-103008-fused-layer-counted-pressure)。
初次A的[FAIL](../experiments/Engineering-Next-TP-Formal/20261101-102444-current-t1-pressure)
来自计数hook未保留RPC签名；已用`functools.wraps`修复，失败归档保留。
旧T1 pressure15/24、旧SinkRecent17/24及历史profile PARTIAL继续有效。
token/payload正确不代表有损质量保持，85条样本未扩评。

## TP2b 进入判断

TP2a后 normalization/统计隔离约2.3–2.45ms，raw约1.3ms，剩余逐层发射仍是
主要成本，满足四层分组探针进入条件。固定四层、复用compactor指针/stride、
预分配raw/stat workspace，先验证数值/计划及完整链，按证据决定保留或停止。
