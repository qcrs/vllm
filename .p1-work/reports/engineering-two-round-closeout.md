# Decode 两轮回收收尾

2026-11-08，起点`1d2b991e53`。已实现最多两轮、由真实decode进展驱动的protected
回收，并补齐物理GPU1单卡、GPU0/1 TP2及DP2验证。**单卡相对旧protected改善，
TP2吞吐改善但联合SLO退化，尚未证明超过fixed；保留显式选项，不改默认。**
完整数据见[统一评估](engineering-two-round-evaluation.md)，机制见
[Controller](engineering-two-round-controller.md)，范围见
[计划](../plans/engineering-two-round-plan.md)。

## 实现与修正的问题

旧planner要求一次回收补齐净缺口，一轮不够就拒绝启动。新增
`decode_reclaim_rounds=2`：先试单轮，必要时证明同组最多两victim的两轮路径可行；
计入已有decode增长、最终admission retry及下一token预留。Scheduler持有一个受益
等待者窗口，暂停后方新admission，等真实decode完成且无在途token才按实际状态
重算第二轮；最多两轮，用尽后等自然释放，admission或abort清窗口。

每轮仍满足当前私有KV删除≤20%、相对已计算逻辑私有历史保留≥50%、最多两个
victim。最多再输出两token必定结束的候选优先自然释放。没有prefill中删除，
Worker/TP completion、canonical commit/free和Graph页表更新仍复用原协议。
默认rounds1、fixed预算和requests路由保持不变，没有新增DP路由算法。

## 分项结果

| 范围 | 实测结论 |
| --- | --- |
| Correctness | CPU本地fixture118 passed /194 deselected；DP state/route共15 passed；适用pre-commit和manual mypy-3.12通过 |
| 单卡/TP2真实执行 | Qwen3-8B BF16两轮、TP2单/双victim、延迟completion、FULL Graph及旧页改写后复用通过；不只验证计划可行 |
| DP2真实执行 | 两副本独立窗口/池，受益等待者abort前实际入场并复用释放页，peer无回收，abort后continuation通过 |
| Micro | 缺88页案例首轮52页+第二轮41页，第二轮zero-copy；CPU决策中位109.94µs，两轮Worker合计约3.79/3.80ms，fixed约3.20ms，不能判执行加速 |
| 单卡Serving | r1→r2正/反向吞吐+16.79%/+26.63%，SLO18/24→23/24、24/24；8192请求0/6→5/6、6/6。相对fixed仍−13.88%/−0.50% |
| TP2 Serving | r1→r2正/反向吞吐+7.64%/+18.94%，SLO18/24→14/24、12/24；8192仍0/6，后方短请求部分超时。相对fixed仍−17.63%/−0.30% |
| DP2 Serving | 同known_work路由，r1/r2均21/24；170.765→205.273tok/s但两边均零回收/miss，不能归因Controller，性能增益未成立 |

所有14组正式Serving均零抢占、零额外重算、逐请求TPOT≤80ms；SLO失败来自TTFT。
SLO仍为每请求同时满足TTFT≤2s且TPOT≤80ms。正、反向各三轮，不用最好一轮
代替全样本。合成输出token IDs一致仅为机械差异记录，冻结85条质量样本未扩，
旧矩阵未重跑，没有新质量收益或无损保证。

## TP定位与停止结论

两轮修复了明确的容量进展限制：真实TP2诊断中缺88页，首轮释放52页，完成decode
后第二轮释放41页，约35.1ms内长请求实际入场。随后8192 prefill从入场到尾chunk
分配的调度区间约1.18s，后方短请求到尾chunk时才获容量检查，确认了计算排队取舍。
该区间包含forward、调度和trace，不是纯kernel耗时。

独立诊断8/8 SLO、122.344tok/s，没有复现正式TP2慢路径，不能替换正式负结果。
诊断时GPU0/1几乎空闲，正式期间另有Python负载；不能断言全部波动来自外部，
也没有证据把本轮问题定性为TP collective错误。单卡正向r2慢轮及TP2正式TTFT
波动的精确触发仍未复现。容量可行不等于计算调度有足够余量满足2秒TTFT。

本轮状态为IMPLEMENTED / CORRECTNESS_PASS；单卡对r1有有限Serving改善，
MICRO_BENEFIT未成立、TP2 SLO_FAIL、对fixed总体Serving收益未成立。停止扩大
参数/矩阵/样本，不把有限两轮继续堆成无限重试。DP usage tie-break仍为
STOPPED / NOT_RUN，prefill中删除KV按用户要求暂缓。

## 资源、失败与Git

B0：`/data_ssd/qcrs/experiments/p1-v8/Engineering-TwoRound-B0/20261108-120031-initial`。
最终归档：`/data_ssd/qcrs/experiments/p1-v8/Engineering-TwoRound-Final/20261108-130116-closeout`。
包含源码/hash、测试与静态日志、GPU实际命令/原始产物索引、正式/诊断独立复算及
`completion-manifest.json`。单卡Worker主线程CPU21，TP2/DP2主线程CPU20/21。

全部23次GPU guard共20 PASS、3 FAIL，峰值合计39.888672GiB，未达44GiB停止线
或45GiB硬上限。两次correctness fixture/harness失败及一次DP不支持参数启动失败
均原样保留；最终四个correctness成功run与全部正式对照Runtime字节一致。
GPU任务全部同锁串行、只用GPU0/1，GPU2任务和其他进程未被终止。

本地阶段提交：实现`2439253387`、测试/测量`380b5480ea`，评估与本收尾提交见最终
manifest和Git日志。独立index只纳入本轮精确文件；用户staged plan/prompt4、
未暂存notes/plan和Scheduler原注释保留。status/delivery只增加本轮导航，历史正文
逐字保留。未push、未创建PR，历史及本轮FAIL/PARTIAL均保留。
