# GPU0/1 当前版本基线与压力复测

2026-11-08，起点 `9f62c70be5`。用户授权重新建立基线并实测单卡、TP2、DP2。
B0：`/data_ssd/qcrs/experiments/p1-v8/Engineering-Retest-B0/20261108-143448-initial`。
环境、资源、归档与Git继承prompt3/4；本轮单卡使用GPU0/CPU20，双卡GPU0/1、
Worker主线程CPU20/21。GPU2有他人任务，不使用、不清理。

## 固定比较条件

本轮验证当前两轮protected与prefill公平性配置，不扩大旧预算扫描。
固定Qwen3-8B BF16、FA2/NHD/B16、APC/async off、FULL_DECODE_ONLY
capture `[1,2,4,8]`、maxseq8、threshold512、output128、seed0、完整负载预热、
正式3轮。保留20%单次删除/50%逻辑累计保留下限、最多双victim、最多两轮且
中间有真实decode、full-prefill与首decode容量预留、backfill2。

单卡与TP2均用mixed `[4096,8192,128,128]×2`、batch1024，逻辑KV池容量相同：
TP1为1.5GiB，TP2为每rank0.75GiB。分别测原到达时刻
`[0,.3,.6,.9,5,5.3,5.6,5.9]`及其乘0.1的burst档；各档顺序
native full、fixed sink_recent、protected sink_recent。固定池是人为压力条件，
并非显卡全部容量。仅增加wrapper `arrival_scale`，默认1保持原行为。

DP2分开归因：W1 `[8192,512]×4`，原每50ms一个请求；batch512、threshold512。
4GiB/replica对比requests+fixed与known_work+fixed，验证路由；
1.5GiB/replica完成requests/known_work × fixed/protected四组，验证小池与组合。
W0 `[4096]×8`、4GiB/replica、相同到达时刻做requests/known_work两组均匀控制。
DP两池独立，不能将DP总容量与TP容量直接比较。

每项有正向收益或明显启动漂移，仅补一次反向启动复核对应A/B；
不继续扫描arrival、阈值、batch或pool来追求正结果。

## 验证与停止

先运行当前CPU行为回归和DP本地fixture；GPU串行补TP1 full、TP2 full延迟
completion单/双victim、DP2 abort/continuation和本地页面改写复用。
再复用两轮88页代表形状micro，分列CPU决策、TP Worker/compaction及路由；
不重跑旧micro矩阵。通过后执行上述正式Serving。

每项GPU均通过run_guard共用锁，合计44GiB停止、45GiB硬限。每次guard最长
15分钟；超时只终止该run的自有进程组，保留原始日志并停止该项，不自动重试。
真实correctness失败先定位，不继续同类性能主张。请求错误、资源停止或超时
保留FAIL/PARTIAL，不挑最好轮、不覆盖历史数据。正式关闭trace、payload检查、
额外设备同步和重profiler。

独立复算逐请求TTFT/TPOT与联合SLO、按长短拆分首token排队、output tok/s、
goodput、抢占/重算、实际回收页和复制字节。吞吐是有限到达负载的完成速率，
本轮压力档不是寻找绝对最大QPS。DP增加每副本工作/路由及usage可用性说明；
零回收不归因Controller。SLO始终TTFT≤2s且TPOT≤80ms。

85条质量样本冻结，不增加质量实验；有损策略无质量保持保证。usage tie-break
仍STOPPED，无新进入证据不实施；prefill期间删KV继续暂缓。
按harness/分析/收尾阶段精确本地提交，用户改动保留，不push、不创建PR。
