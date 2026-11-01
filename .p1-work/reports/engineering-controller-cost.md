# C0：Controller完成边界与重试结果审计

C0补齐requester/victim、两rank completion、commit与首次retry的关联；29次回收中
24次首次retry成功，5次仍走native preemption后重算admission。**增量时间成本校准
仍为PARTIAL**：原trace包含正常forward和显著扰动，缺少未选候选及匹配的native
反事实成本。C1时间排序、C2动作门控均NOT_RUN，保留现有Controller。

## 数据与分析变化

只读复用`TTFT/20261031-180022-tp-batch1024-timeline`，没有新GPU实验。
这是SinkRecent/TP2/CPU20,21的重NSYS诊断，6次抢占、52.599tok/s、0/8 SLO，
偏离原正式零抢占轨迹；详见 [原诊断](tp-ttft-diagnostic.md)，不能作为正式成本基线。

扩展[`analyze_tp_ttft.py`](../scripts/analyze_tp_ttft.py)：按真实step的schedule时间
定位allocation miss与selection，再关联victim commit、requester首次retry/后续成功、
victim再次schedule。`outcome_events`包含区间内**所有**native preemptions，不能只看
requester事件而漏掉其他请求被抢占；后续成功单列`allocation_success`。
增加`released_covers_goal`、`useful_released_blocks=min(released,goal)`及zero-copy/
token-copy分组，中位数使用`statistics.median`，CLI允许输出到新路径。

Scheduler仍拥有候选过滤、shortfall/reserve与canonical commit；Worker完成plan/
compaction/row/E后执行completion collective，AsyncOutput实际等待后才free。
本次只改离线分析，不增加热路径统计、同步、EMA或状态表。

## 首次retry结果

29次intent中RUNNING miss24、WAITING admission miss5；requester与victim全部不同。
zero-copy2、token-copy27，累计释放2424 blocks；累计数不等于唯一物理页或新增容量。
覆盖goal24/29，与首次retry成功24/29一致。5例未成功如下，reserve均0：

| step | requester简称 | shortfall | 实际release | 候选数 | retry required / available |
| ---: | --- | ---: | ---: | ---: | --- |
| 343 | r0-5 | 47 | 32 | 5 | 64 / 49 |
| 371 | r0-7 | 62 | 33 | 7 | 63 / 34 |
| 381 | r0-7 | 63 | 32 | 7 | 64 / 33 |
| 390 | r0-7 | 36 | 18 | 7 | 64 / 45 |
| 399 | r0-7 | 23 | 19 | 7 | 64 / 60 |

这5例都是RUNNING miss，首次retry再次miss→native preempt requester自身→后续
以`logical_tokens=0`、`request_status=PREEMPTED`重新admission。不能把后一次
`allocation_done`计成“回收重试成功”。诊断全程6次抢占与此5例不矛盾：它们是
不同统计集合，不以intent数反推全程preemption总数。

[`choose_reclaim_candidate`](../../vllm/v1/kv_reclamation.py)先选能覆盖goal的候选，
否则优先最大release，再按zero-copy、单位release搬移成本和稳定tie-break排序。
结合源码，5例不足覆盖意味着当时过滤后的候选中无单个可覆盖者；这是**源码推断**，
trace没有保存全候选成本，不能升级成已观测的时间错选或更优动作证明。

## 时间边界，不能直接作为增量成本

单位ms；跨CPU/nsys时钟最大校准偏差165461ns，保留该误差：

| 观测区间 | min | median | max | 实际含义 |
| --- | ---: | ---: | ---: | --- |
| select→commit inclusive | 7.735783 | 18.724136 | 198.496177 | 包含同step正常forward及发射 |
| select→首个reclaim GPU activity | .866317 | 16.616039 | 51.325029 | 不能全归为Controller开销 |
| 两rank reclaim GPU span | .685226 | 1.435529 | 183.450093 | 包含launch gap/TP等待，不是kernel busy之和 |
| 最后GPU完成→commit | .284220 | .526004 | .793166 | output返回与canonical发布范围 |
| commit→首次retry | .251146 | .547390 | 1.220829 | 包含Scheduler下一步 |
| commit→后续allocation成功 | .251146 | .576638 | 15.989034 | 5例含preempt后的新admission |
| commit→victim再次schedule | .117429 | .135829 | 1.151556 | 不是相对不回收的victim stall |

zero-copy2例的E288–1056/keep144–528，GPU span中位.840197ms；token-copy27例的
E531–8211/keep265–4105，中位1.462943ms。它们是被选候选的观察分组，不是拟合
误差已验证的校准表，不能据此断言任意zero-copy更快，也不能和T1 TriAttention
micro独立阶段相加。冷shape、背景争用与采样扰动尚未分离。

## C1/C2停止理由与保留能力

已有coverage/zero-copy/byte-cost排序、安全候选过滤、bounded reserve、一次deferred
retry和全rank完成继续交付。没有反复、可解释且超过校准误差的真实时间错选，C1不实施。
5次release不足与native preempt共存说明“执行过回收”不保证避免重算，但缺少同输入/
到达时刻的native动作对照，不能证明C2动作门控会改善TTFT/goodput。

若未来有新证据，仍须区分running miss和waiting admission：后者不能升级为主动抢占。
动作决策必须早于`Scheduler.make_reclaim_request`的hash mutation；mutation后失败
依然fail-stop。native重算与有损retention的质量不等价；本轮不扩85条质量样本。

## 可复算证据

[最终分析归档](../experiments/Engineering/20261101-0832-closeout)保存
`analysis/controller-analysis.json`、`analysis/ttft-analysis-extended.json`、日志及
`analysis-manifest.json`。manifest记录实际argv、三个CPU分析exit0、源码/hash与
派生结果hash；原NSYS/SQLite/timeline未覆盖。

入口为`analyze_tp_ttft.py RUN --output NEW_JSON`，再用同脚本
`RUN --analysis NEW_JSON --controller-output CONTROLLER_JSON`。
最终结果与先前`Engineering-Controller/20261101-0832-outcome-audit`保留相同计数和
语义；旧分析产物也保留。适用pre-commit与mypy-3.12通过。本次没有Controller
Runtime改动或新的GPU正确性/性能声明。
