# 当前版本：单卡、TP2、DP2 基线与突发压力复测

2026-11-08，起点 `9f62c70be5`。本轮验证已有两轮 protected、完整 prefill/首步
decode 容量预留、backfill2 和 512-token prefill 限额；没有新增 Runtime 算法，
没有修改 Worker/TP 协议。执行范围见[计划](../plans/engineering-retest-plan.md)。
用户17:05明确告知GPU1/2已空并要求重建基线。本报告主表改为该空闲批次，
此前共享批次单列保留。正反向可比组已完成；DP W0 requests反向组在正式采样前
启动失败，单列为PARTIAL，不在占用中的GPU2上重试。

## 比较条件与口径

最初按用户要求在 GPU0 单卡、GPU0/1 双卡建立基线；其后 GPU1 出现外部训练，
保留已完成与受污染的结果。用户随后明确授权改用 GPU1/2，覆盖旧 prompt 的
GPU2 禁用约束。新设备重新建立同配置基线，不跨设备计算策略收益。

单卡为 GPU1/CPU20；TP2、DP2 为 GPU1/2、Worker 主线程 CPU20/21。
GPU UUID 分别为 `GPU-a53fa601-8471-d782-2971-46e5a8e5d328`、
`GPU-25019de8-09aa-328a-be23-4bec986badad`；GPU1/2 为 NODE 链路、无 NVLink。
此前批次GPU1有外部CUDA进程，GPU2有持续计算背景，该批多卡性能为共享背景/PARTIAL。
本批每组启动前检查目标GPU无外部CUDA进程，并持续保存每秒逐PID采样；发现外部
计算则标记对应组PARTIAL。逐PID pmon的`-`表示未取得数值，不解释为测得零占用。

环境固定 Qwen3-8B BF16、现有 Python3.10.20/Torch2.11.0+cu129/Triton3.6.0、
MRV2、FA2/NHD/B16；不安装升级依赖或下载模型。GPU 全部由 `run_guard.py`
共用锁串行运行，合计项目显存 44GiB 停止、45GiB 硬限；不操作他人任务。

TP1 与 TP2 使用相同逻辑 KV 容量：单卡 1.5GiB、TP2 每 rank 0.75GiB，实际均
10912 tokens/682 blocks。该人为小池用于暴露容量压力，不是使用整张 A100 的容量
测最大吞吐，也不让 TP2 获得额外逻辑容量。每轮输入 `[4096,8192,128,128]×2`，
baseline 到达秒数 `[0,.3,.6,.9,5,5.3,5.6,5.9]`，burst 将到达时间乘 0.1。
batch1024/maxseq8、prefill threshold512、输出128、seed0、APC/async off、
FULL_DECODE_ONLY capture `[1,2,4,8]`。先小预热、完整负载预热，再正式三轮。

native/full 不设置回收 additional-config；fixed 和 protected 均用 sink_recent。
fixed retention=.5；protected 另启用 rounds2/backfill2，单次私有 KV 删除≤20%、
逻辑已计算私有上下文保留≥50%，每轮最多两个 victim，两轮之间必须完成真实
decode。三者保留上下文的语义不同；与 native 比较性能不能当作无损提速。

DP2 每 replica 独立 TP1 和 KV 池。W1 `[8192,512]×4`，W0 `[4096]×8`，每50ms
到达一个请求；batch512/threshold512、maxseq8、输出128及三轮协议相同。
4GiB/replica 对比 requests/known_work 路由（均 fixed，实际29120 tokens/池）；
1.5GiB/replica 对比 routing×budget 四组（10912 tokens/池）。DP 总容量不与 TP
直接混比。DP 零回收时，性能差异不能归因 Controller。

SLO 始终逐请求同时 **TTFT≤2s 且 TPOT≤80ms**；goodput 为达标请求数/正式总耗时。
tok/s 为三轮全部输出 token/正式总耗时，包含到达间隔和等待，是有限到达负载的
完成速率，不是最大饱和吞吐。每组24请求，不挑最好轮、不扩大85条质量样本。
只对正向收益或明显漂移的相关 A/B 做一次反向复核，不扫 threshold/batch/pool。

独立[分析器](../scripts/analyze_retest_evaluation.py)检查源码/helper/input hash、
NVML PID→实际物理卡、CPU affinity、Graph capture、池容量、请求 ID、128输出/
127 ITL、`scheduled=prompt+127+recomputed`，重新计算每轮及逐请求 SLO、goodput、
长度分组与首 token 时间拆分。DP 同时核查路由公式、pending ADD watermark、
每副本工作量与 snapshot。正式关闭逐步 trace、KV clone 和额外设备同步。

## Correctness

CPU 回归92 passed /224 deselected，DP 本地 fixture13 passed /16 deselected；
适用 pre-commit 和 manual mypy-3.12 通过。CPU suite 未扩大为完整上游 suite。
实际命令与日志保存于旧设备 B0；没有因切卡重复 CPU 回归。

旧设备上真实 TP1、TP2 单/双 victim 延迟 completion、Graph、page rewrite/reuse，
及 DP abort/continuation 均通过。最初 TP1 fixture 仍使用28页，在新增首 decode
预留后两轮路径不再可行；该次 FAIL 保留。仅将对应 TP/DP probe fixture 调整为
29页（扣除 null page 后28可用页）；TP2双victim仍用43页。恢复要测的可行路径，
未修改 Runtime 绕过检查。

新 GPU1/2 补测：

| 检查 | 结果 |
| --- | --- |
| TP2 双 victim/延迟 completion | PASS；首轮两个 victim 释放7页，真实 decode 后第二轮释放2页；全 rank 结果一致，ACK 晚于两个 rank completion |
| 保护边界 | 每个 intent 独立复算20%/50%边界，最多两个 victim，延迟等待时 canonical 未提前 commit |
| TP2 Graph/数值/页面复用 | 每 rank FULL59、Graph/eager57、FP32 attention reference115，13个释放页真实改写；最终42页 free，零抢占 |
| DP2 生命周期 | 等待者在 abort 前开始；本地释放页由该等待者改写；peer 在 abort 后继续112次已验证 forward，最终两个池各28页 free |

这证明被测状态下的执行与生命周期正确，不证明模型质量、所有部署组合或性能收益。

## Micro

只复测一个88页缺口代表形状，真实 TP2 Worker、合成 KV，3次预热/每阶段10样本，
逐样本取慢 rank 后报告中位数。它不含正常模型 forward 或两轮之间的真实 decode。

| 动作 | 释放页 | 估算复制 bytes/rank | Worker device ms | compaction device ms |
| --- | ---: | ---: | ---: | ---: |
| fixed，E=L=4097 | 129 | 299630592 | 0.545792 | 0.380928 |
| protected 首轮 | 52 | 481001472 | 0.775680 | 0.609280 |
| protected 第二轮，E3280/L4099 | 41 | 0 | 0.341504 | 0.014336 |

protected 单轮不足以覆盖88页；计划/真实 Worker 两轮路径可行。Controller CPU 决策
30样本中位111.3685µs。第二轮可整页重映射，但 Worker 仍有元数据及 TP 协调成本，
零复制不等于零耗时。三种动作释放量不同，不将这些数字包装成同动作加速；
也不把两个阶段中位数之和当作完整两轮Serving时间。共享批次Worker为
1.453056/1.686016/1.110528ms，compaction为0.380416/0.605696/0.015360ms；
空闲后执行等待减少，compaction接近。两批源码未变，不能将其差值称为算法优化。
旧GPU0/1 micro另有归档，不跨卡计算策略收益。

## Serving：GPU1单卡

以下均为同GPU1、相同逻辑池和512限额下的正向三轮独立复算，每组24请求。
突发档改变到达密度，不改变prompt、输出量、池或保护预算。

| 档位/策略 | 输出 tok/s | 联合SLO | 抢占 | 重算token | 回收次数 | 释放页 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline native/full | 34.291 | 7/24 | 386 | 692697 | 0 | 0 |
| baseline fixed | 121.197 | 24/24 | 0 | 0 | 12 | 804 |
| baseline protected | 120.746 | 24/24 | 0 | 0 | 24 | 835 |
| burst fixed | 195.860 | 21/24 | 3 | 8400 | 36 | 2886 |
| burst protected | 112.439 | 9/24 | 0 | 0 | 6 | 420 |

baseline protected相对fixed完成速率−0.37%，两者均达标；没有总体提速证据。
burst protected相对fixed−42.59%，尽管抢占/重算为零，15个请求因TTFT失败；
fixed仅3个TTFT失败，两组TPOT均通过。零抢占不能单独作为策略更优的依据。

| 档位/策略 | 128 TTFT中位s | 4096 TTFT中位s | 8192 TTFT中位s | 8192首次调度排队中位s | 8192 prefill结果消费中位s |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline fixed | 0.112 | 0.435 | 1.127 | 0.00027 | 1.076 |
| baseline protected | 0.162 | 0.395 | 1.158 | 0.099 | 1.009 |
| burst fixed | 1.151 | 1.144 | 2.032 | 0.564 | 1.423 |
| burst protected | 1.062 | 2.031 | 5.087 | 4.133 | 0.919 |

突发protected的8192请求主要等在首次调度前，prefill本身反而短于fixed。
这与“小池、完整prompt容量预留及保护下限使入场受限”的判断一致，也在单卡
复现了问题；不能统称为TP通信缺陷。正式未启用逐步候选trace，因此867次
allocation miss只作为计数，不逐一推断为同一个排除原因。

本轮没有重跑native burst：旧设备该档已出现432次抢占、756207个重算token、
3/24 SLO，重复这一失败配置不会帮助区分fixed/protected。保留native baseline
作为lossless对照；其与有损回收的差异不能解释为保持质量不变的加速。

## Serving：GPU1/2 TP2

同一空闲设备批次、相同逻辑容量的正向结果如下。baseline 的 fixed/protected
均24/24，完成速率差−0.26%；此前共享背景下的明显差距没有在此复现。

| 档位/策略 | 输出 tok/s | 联合SLO | 抢占 | 重算token | 回收次数 | 释放页 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline native/full | 25.286 | 0/24 | 399 | 713597 | 0 | 0 |
| baseline fixed | 123.394 | 24/24 | 0 | 0 | 12 | 1170 |
| baseline protected | 123.078 | 24/24 | 0 | 0 | 24 | 840 |
| burst fixed | 177.183 | 9/24 | 3 | 8400 | 36 | 2886 |
| burst protected | 128.238 | 15/24 | 0 | 0 | 6 | 420 |

burst protected完成速率低27.62%，但短请求全部达标：fixed按128/4096/8192长度为
6/12、3/6、0/6，protected为12/12、3/6、0/6。各正式轮fixed均3/8、protected均5/8。
两组逐请求TPOT最大33.67/24.13ms，失败均集中TTFT。反向复核同样为9/24→15/24，
吞吐177.859→127.660tok/s（−28.22%）。SLO通过数改善来自短请求，不表示长请求
问题解决；吞吐下降在正反向重复出现。

| 档位/策略 | 128 TTFT中位s | 4096 TTFT中位s | 8192 TTFT中位s | 8192首次调度排队中位s | 8192 prefill结果消费中位s |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline fixed | 0.209 | 0.785 | 1.575 | 0.00015 | 1.507 |
| baseline protected | 0.183 | 0.576 | 1.607 | 0.247 | 1.288 |
| burst fixed | 1.903 | 1.855 | 3.029 | 0.944 | 1.995 |
| burst protected | 0.869 | 2.077 | 4.718 | 3.462 | 1.208 |

## 容量与调度归因

当前源码中，protected admission预留完整剩余prompt及首decode，victim必须已完成
prefill；两轮路径也要求当前running均完成prefill。512限额改善计算分享，不能创造
KV容量。已有8192请求的逻辑50%下限意味着至少4096个KV token、256页必须保留；
新8192请求连同首decode需要513页。即使忽略其余请求、decode增长，允许任意轮次
回收，最低仍需769页，大于扣除null page后的681页，差88页。

`capacity-audit.json`调用当前`protected_reclaim_floor`得到
`8192→6554→5244→4196→4096`。这是放宽条件后的示例容量下界，不是逐步trace，
不能把每次allocation miss都归为同一原因。正式长请求的等待阶段与该限制一致。
单卡同容量也复现突发退化，当前证据指向保护预算、入场预留和到达密度的组合，
没有发现TP rank一致性或页面提交错误。TP通信代价没有单独profile，不能从本表
推导其精确占比。

## Serving：GPU1/2 DP2

下表为同一空闲批次的正向三轮结果；各组均零抢占、零重算，TPOT全部通过，
失败来自TTFT。路由和Controller分开比较，不将DP独立池的总容量当作TP容量。

| 负载/每池GiB | 路由 | 预算 | 输出tok/s | 联合SLO | allocation miss | 回收次数 | 释放页 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| W1/4 | requests | fixed | 200.435 | 20/24 | 1 | 1 | 257 |
| W1/4 | known_work | fixed | 253.887 | 24/24 | 0 | 0 | 0 |
| W0/4 | requests | fixed | 266.360 | 24/24 | 0 | 0 | 0 |
| W0/4 | known_work | fixed | 279.288 | 24/24 | 0 | 0 | 0 |
| W1/1.5 | requests | fixed | 193.181 | 19/24 | 29 | 29 | 3937 |
| W1/1.5 | requests | protected | 123.667 | 18/24 | 1143 | 0 | 0 |
| W1/1.5 | known_work | fixed | 254.261 | 24/24 | 22 | 22 | 3385 |
| W1/1.5 | known_work | protected | 174.037 | 18/24 | 759 | 0 | 0 |

W1/4的requests每副本正式prefill为22528/81920 tokens，known_work为53248/51200；
两组请求数都14/10，说明只平衡请求数不能代表平衡计算。8192长请求达标8/12→12/12，
TTFT中位1.692→1.252s；512短请求中位0.083→0.785s，两组短请求仍12/12达标。
known_work没有回收，收益归因已有工作量路由，不归因protected Controller。

W0正向requests三轮279.353/278.345/244.465tok/s，known_work为
279.385/279.487/278.994，配对输出速率+4.85%，两边均24/24。requests副本工作
53248/45056、known_work49152/49152；正差集中在一轮的分配变化。反向requests组
在正式Serving前因TCPStore绑定`127.0.0.1:37217`遇到`EADDRINUSE`，没有请求样本；
该方向标记PARTIAL，不据此宣称W0路由有稳定收益。

小池known_work+fixed实际发生22次回收、归还3385页，三轮均8/8达标；这里同时
覆盖了路由和物理回收的执行，不等于证明protected优于fixed。protected两种路由
均零回收：存在大量miss，但当前已接纳请求/保护预算下没有启动可行回收，长请求
排队。known_work将其完成速率提高，但没有把18/24变为24/24。

路由CPU成本（W1/4，24次正式路由中位）requests26.339µs、known_work37.0555µs。
snapshot生成到路由使用年龄分别p50 101.409/101.204ms、最大3663.229/299.538ms；
小池requests+protected最大9033.789ms。usage是收到的快照，不是路由时瞬时真值，
不能把旧快照的低usage当作当时peer一定可接纳。正式数据未启用逐步miss关联trace，
不足以新增usage tie-break的进入证据；沿用STOPPED / NOT_RUN。

## 反向复核与综合结论

| 对照 | 正向变化 | 反向变化 | 结论 |
| --- | --- | --- | --- |
| TP2 burst fixed→protected | tok/s −27.62%；SLO 9→15/24 | tok/s −28.22%；SLO 9→15/24 | protected让更多短请求达标，但完成速率损失重复出现；8192请求仍未达标 |
| DP2 W1/4 requests→known_work，fixed | tok/s +26.67%；SLO 20→24/24 | tok/s +18.85%；SLO 21→24/24 | 混长短负载下路由收益正反向一致，不归因protected Controller |
| DP2 W1/1.5 requests→known_work，fixed | tok/s +31.62%；SLO 19→24/24 | tok/s +16.78%；SLO 21→24/24 | 小池下路由收益正反向一致，SLO也改善 |
| DP2 W1/1.5 requests→known_work，protected | tok/s +40.73%；SLO 18→18/24 | tok/s +43.52%；SLO 18→18/24 | 路由相对收益稳定，但protected两种路由均未改善SLO |
| DP2 W1/1.5 requests，fixed→protected | tok/s −35.98%；SLO 19→18/24 | tok/s −44.20%；SLO 21→18/24 | protected在requests路由下持续较慢 |
| DP2 W1/1.5 known_work，fixed→protected | tok/s −31.55%；SLO 24→18/24 | tok/s −31.43%；SLO 24→18/24 | protected在known_work路由下也持续较慢 |
| DP2 W0/4 requests→known_work，fixed | tok/s +4.85%；SLO 24→24/24 | requests组启动失败，无可比样本 | PARTIAL；不声称均匀负载下有稳定路由收益 |

单卡baseline档fixed/protected分别121.197/120.746tok/s、均24/24；TP2 baseline
档123.394/123.078tok/s、也均24/24，均未显示整体加速。burst档单卡protected比
fixed低42.59%、SLO从21/24降至9/24；TP2 protected的SLO计数改善可重复，但吞吐
下降约28%。当前性能结论应分开陈述：known_work对W1路由有效；protected有界回收
满足执行正确性，但未证明Serving整体收益，低池下还会显著减速。没有证据支持DP
usage tie-break，也没有扩展质量结论。

本轮没有改Runtime源码。项目SLO继续为逐请求同时TTFT≤2s且TPOT≤80ms；未重跑旧
矩阵、未扩大85条质量样本。干净批次正式测量窗口的背景审计均未发现外部GPU计算；
共29次guard为28 PASS/1 FAIL，项目显存峰值39.888672GiB，低于44GiB停止线。
失败是上述DP W0反向启动失败。其后GPU2出现另一用户的大显存进程，未触碰，也未
在该背景下追加运行。

## 原始产物

旧设备 B0：
`/data_ssd/qcrs/experiments/p1-v8/Engineering-Retest-B0/20261108-143448-initial`。
共享设备 B0：
`/data_ssd/qcrs/experiments/p1-v8/Engineering-Retest12-B0/20261108-151936-shared-gpu12`。
空闲重测B0：
`/data_ssd/qcrs/experiments/p1-v8/Engineering-Retest12Clean-B0/20261108-170517-idle-gpu12`。
新 correctness 在 `Retest12-Correctness`，micro 在 `Retest12-Micro`，正式 TP1/TP2
在 `Retest12-Serving`，DP 在 `Retest12-DP`，均位于同一大产物根目录下。
各 manifest 保留精确命令与退出码；各 run 保存实际源码/input/config、UUID/affinity、
NVML、原始日志及请求结果，派生 audit 与原始数据分开。
