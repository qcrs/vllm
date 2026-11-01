# TP 动态 E 修正的反向复核

2026-11-01，本轮回应“micro 修正为何让正式 SLO 18/24变16/24，是否有 bug”。
只补一次反向启动，不修改内核、不扫 chunk，不重跑已通过的正确性矩阵。
**反向未复现退化；仍没有稳定 Serving 收益，也不能认定稳定算法回归。**

## 修复目标与剩余瓶颈

原修改仅在两个 grouped kernel 上增加 `@triton.jit(do_not_specialize=["E"])`。
E只作 mask bound；取消其整数16整除性特化，不改 stride/指针对齐、浮点计算、
GQA、score SUM、stable selection、broadcast 或 completion。
局部目标是避免遇到另一种 E 对齐状态时生成不必要的 kernel 变体。

[前轮证据](engineering-refinement-tp.md)已验证跨 E 对齐 cache miss 10→0、
21项独立 GPU reference、真实8B TP2全层 retained payload 和 continuation/free/reuse。
这些仍成立；cache miss不等于每次实际编译，不能用不同磁盘缓存状态的首次调用
时间计算冷路径提速百分比。本轮没有新的完整 reclaim micro 数字。

原正式 A/B 完整预热、沿用已有磁盘缓存，没有重现诊断的约3.46秒冷调用。
因此，它衡量的是预热后 Serving，不能要求取消冷变体带来可见吞吐提升。
更早的 TP2a/TP2b micro 只减少每次毫秒级回收成本，不能直接填平 prefill/排队
约0.76秒的缺口。两个 TP rank 耗时不能相加；正常 forward 不在 reclaim micro 内。

## 一次反向结果

固定 Qwen3-8B BF16、TP2/GPU0,1、Worker主线程CPU20/21、TriAttention、fixed预算、
0.75GiB/rank、batch1024、seqs8、FULL_DECODE_ONLY[1,2,4,8]、APC/async off。
每个独立服务完整负载预热后3轮×8，trace/profiler off、不清缓存。
本次先运行 dynamic E，随后 baseline；源码快照确认仅上述两个 decorator 不同。

| 反向 timed 指标 | [baseline，后运行](../experiments/Engineering-Revision-TP-Formal/20261101-152015-aligned-e-reverse) | [dynamic E，先运行](../experiments/Engineering-Revision-TP-Formal/20261101-151732-dynamic-e-reverse) |
| --- | ---: | ---: |
| 三轮 output tok/s | 100.858 / 100.915 / 100.896 | 100.867 / 101.192 / 101.214 |
| 合并 output tok/s | 100.890 | 101.091（+0.200%） |
| SLO / goodput req/s | 18/24 / 0.591 | 18/24 / 0.592 |
| TTFT max | 2.829s | 2.825s |
| 逐请求平均 TPOT max | 33.019ms | 32.984ms |
| timed抢占 | 0 | 0 |

两次 guard PASS、项目 NVML 峰均19,935,526,912 bytes；24条输出 token IDs 一致，
全部失败请求均因 TTFT>2s。`summary.json` 保留所有请求、counter、输入/hash与源码比较；
实际重算量未采逐请求trace，不由输入 counter 推断计算次数。

前轮正向吞吐−0.740%、SLO18/24→16/24原样保留；额外失败 TTFT2.01582/2.01069s。
正反方向不同、反向两边均18/24，支持“部分请求处于2秒阈值边缘，运行变化可改变
成功数”的解释。本样本不足以建立统计显著性，也不能宣称零回归或 Serving 加速。
保留消除不必要特化的冷路径修正，停止继续内核/权重/形状搜索。

## 仍未解决的首 token 窗口

已有零抢占 host 轨迹中，第一波最后请求约93ms入口处理、1.407s初始排队、
1.197s参与prefill steps、60ms段间等待、14ms输出送达，TTFT合计2.771s。
自身参与的 step 是 inclusive wall time，不等于独占 GPU forward；计算、通信与
rank等待的设备级贡献仍 UNKNOWN。该证据没有证明一个调度调整能使全部请求达标。

应将后续性能课题定义为“固定输入的 prefill 可达窗口与排队公平性”，单独验证，
不能继续把小幅 scorer 加速当成解决 TTFT 的路线。本轮按简历项目合理投入结束；
保留原 SLO_FAIL、profile PARTIAL 与85条质量样本边界，不扩旧矩阵。

复算命令（按 prompt3 激活环境后）：

```bash
.venv/bin/python .p1-work/experiments/Engineering-Revision-TP-Formal/analyze-formal.py
```

同目录 `summary.json` / `analysis.log` 及各 run 的 `invocation.json` 保存全部证据。
