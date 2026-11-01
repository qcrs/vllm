# 重试预算、DP 状态覆盖与 TP 冷路径：本轮收尾

2026-11-01，承接 `8cc5e15891`，按用户最新授权实施三项修正并完成有限验证。
DP4 在固定 W1 上取得正反向收益；Controller 修复重试容量预测，但仍负收益；
TP 消除无必要的长度特化，稳态 Serving 无收益。按简历项目的合理投入边界收尾，
不以全 SLO PASS 为目标追加扫参。前轮全部阶段的交付见
[prompt3 收尾](engineering-next-closeout.md)，本报告只增加本轮证据。

## 实际实现与结果分层

| 方向 | 正确性与实现 | micro | Serving 与取舍 |
| --- | --- | --- | --- |
| CB 重试预算 | `retry` / `retry_pass` 根据尾槽、post-forward E/q、真实 running 顺序与原 allocator 求最大合法 K；Pool 和真实 TP2 生命周期通过 | 本轮不重复旧评分 micro | fixed 18/24；retry_pass 0/24、60次 timed 抢占，仍明显负收益；fixed 默认，实验模式保留 |
| DP4 pending ADD | 每 frontend/replica 序号，经 Engine 主线程 admission 后随 U 发布 watermark，前端仅补未覆盖 prompt；编码、Coordinator、abort/finished 测试通过 | 路由 CPU 中位增加约11μs，不作为 GPU micro 收益 | W1 两次启动各24/24；相对 requests 合并吞吐分别+20.7%/+33.9%，有限 `SERVING_BENEFIT / SLO_PASS`；requests 仍默认 |
| TP 动态 E | 两个 grouped Triton 入口取消 E 整除性特化，独立数值/KeepPlan 与真实 TP2 continuation/free/reuse通过 | 同 scorer E4159→4160的新增进程 cache miss由10变0；不计算冷编译提速百分比 | baseline18/24→候选16/24，吞吐-0.740%，`SLO_FAIL`；保留合理冷路径修正，停止追加调度优化 |

分项源码、原始实验、逐请求结果和限制分别见
[Controller](engineering-refinement-controller.md)、[DP](engineering-refinement-dp.md)、
[TP](engineering-refinement-tp.md)。旧 TP2a/TP2b 完整回收缩短约8.5–13% / 40–52%
只限[原 micro](engineering-next-tp.md)，本轮未重新测得完整回收加速或 TP Serving 收益。
本轮 guard 全 PASS 不表示全部性能 PASS；旧 FAIL/PARTIAL 与原始失败文件均保留。

## 技术判断与停止依据

CB 将“当前释放缺口”改为“下一次有界调度分配可行”，通过
`free + N - ceil((K+q_next)/B) >= other_required + soft_reserve`
求最大合法 K。未写死 K-1、盲加 reserve 或突破保护/floor。`retry` 诊断206/206
首次重试成功；`retry_pass` 的144次预测覆盖也全部首次成功，且108次覆盖完整
known running pass。这证明尾页竞争机制得到处理，但不构成资源 reservation，
也未解决整个请求生命周期的小额重复回收摊销问题。

正式 retry_pass 相对 fixed 吞吐约36.1对101.3 tok/s，仍0/24 SLO。比旧 demand
468次抢占减少不代表优于 fixed；停止扩大前瞻、扫 reserve 或增加多 victim。
CB2 在新轨迹确有4/148次无单候选覆盖D且随后抢占，前轮“无进入证据”不再适用。
本轮基于影响占比、部分释放的潜在重算价值和投入优先级停止，`STOPPED / NOT_RUN`；
未运行独立 gate A/B，不声称 gate 已被证明无收益。

DP 的关键是状态覆盖关系，而非快照年龄。有效 U 等于快照 U 加未确认本地 ADD；
watermark 与该次 U 同时复制发布，确认后精确撤销临时补计。只维护短暂 pending，
没有在前端复制完整生命周期、预测未知输出或调整 C/U 权重。显式 rank、sticky、
abort affinity、F 下界和 MoE 原路径保持；不扩大 PL2 为跨 DP 原子更新。

W1 六个 timed round 全部8/8，相同输入的输出 token IDs一致、零抢占/重算。
该 A/B 比较 requests 与修正后的 known_work 整体，不能把百分比单独归于 pending。
基线两次启动有漂移，+20.7%/+33.9%不是跨负载保证。W0 两边24/24，名义+3.03%
落在 baseline 轮间约12.6%跨度内，只交付本次无超过5%退化，不继续反向或扫权重。

TP host-only 零抢占轨迹中，第一波最后请求 TTFT2.771s：入口93ms、初始排队
1407ms、自身参与的 prefill step1197ms、段间60ms、送达14ms。第一波最后首 token
相对全通过窗口仍差约0.758s。own prefill 为 inclusive wall，不能当纯 GPU forward，
两个 rank 耗时不能相加；设备计算、通信与独立 prefill 上限仍 UNKNOWN。

同一轨迹的第二波 E4160 出现3.46秒回收冷路径，缓存证据支持无必要长度对齐特化。
仅取消两个 grouped kernel 的 E 特化，保持运算、SUM/plan、compactor/completion。
正式两边磁盘缓存已热，因此不能用诊断4/8→正式16/24声称修复收益；正式新增
两条约2.01s的边界失败照实记录，不放宽2s门槛。没有证据支持调度改动能填平
秒级缺口，停止追加 chunk、公平性和设备优化。既有 PL0/PL1/PL2 能力保留，
本轮未重跑前轮完整矩阵。

## 验证、资源与复核入口

| 检查 | 实际结果及边界 |
| --- | --- |
| 组合 CPU 回归 | 72 passed / 227 deselected / 23 warnings；复用四个现有 suite，runner仅替换两个未缓存无关 tokenizer，不称完整上游 suite |
| GPU reference | 21 passed / 14 deselected / 14 warnings；E95/96、partial tile/GQA/跨组/near-cutoff独立 reference |
| 真实8B TP2 | CB q0/q1与TP动态E分别验证全36层 payload、两rank同plan、continuation/free/rewrite/reuse；TP每rank47次FP32 attention reference |
| 静态检查 | 受影响 pre-commit、manual mypy-3.12通过；初次格式/静态/复算失败保留，最终不以初次失败冒称通过 |
| 有限正式对照 | CB两次、DP六次、TP两次，共10次；各完整负载预热后3轮×8，诊断与正式分目录；没有新增质量样本 |
| GPU guard | 本轮19次全部运行/资源 PASS；显存峰42,830,135,296 bytes，低于合计44GiB停止/45GiB硬限；GPU1单卡、GPU0/1双卡，未使用GPU2 |

环境继续使用 prompt3 的 `.venv/bin/python`、Qwen3-8B BF16、FA2/NHD/B16、
离线缓存、guard共用锁和CPU20/21 Worker主线程放置；不改OMP/NUMA，不安装升级
依赖或下载模型。guard、source/helper/input hash、affinity和退出码见各原始run。
原始CPU与静态日志见 [B0](../experiments/Engineering-Refinement-B0)，最终全量清单与
离线复核见 [本轮审计](../experiments/Engineering-Refinement-Closeout/20261101-145304-audit)。

审计中的 `audit-refinement.py` / `audit.json` 枚举全部本轮guard（含失败状态，
如存在），校验实际源码快照、输入、资源与affinity。`analysis-manifest.json`记录
离线复算命令/退出码及派生hash；`completion-manifest.json`记录最终HEAD与用户
patch逐字保护检查。收尾只复核已有产物，没有再次运行GPU。

## 分阶段本地提交与简历表述

- `434f6a9b54`：尾槽与有界重试/完整pass预算，含CB验证和负结果。
- `606688a35e`：pending ADD与Engine确认水位，含W1/W0有限Serving证据。
- `a5f27e8b0d`：动态E特化修正、host TTFT分析和cache审计，含无Serving收益结论。
- 本报告所在提交：总报告与status/delivery本轮导航；最终SHA见审计manifest。

初始用户仅有status/delivery待提交；已备份原文件和patch。本轮只向两文件新增
导航块，index基于原HEAD正文加本轮块构造，不纳入用户原有内容；移除本轮块后
与初始用户文件逐字一致。未reset/amend/rebase、push或创建上游PR。

简历可追加：

> 沿真实调度顺序建模物理页回收与重试容量；使用Engine确认水位补齐DP路由中
> 尚未被快照覆盖的prompt工作量，在固定双A100/Qwen3-8B混合负载W1的有限
> 正反向对照中，将SLO成功由20/24、19/24提高至各24/24，合并吞吐提高约
> 20.7%、33.9%；通过真实TP2页复用、独立数值reference与失败归档验证边界。

面试可展开页取整与尾槽、admission/快照因果覆盖、版本pin与设备完成、Triton
标量特化与冷热缓存归因。CB/TP无Serving收益也属于工程判断，不包装成全面提速。
85条质量样本冻结，有损质量未证明；payload/token一致不替代任务质量评测。
