# Controller 进展修正与 TP/DP 复核收尾

2026-11-08，承接 `e3550ac916`，完成实现、正确性、micro、有限正式 Serving 和剩余
容量诊断。**修复了抢占/重算风暴和小额回收不必要复制；TP仍未超过fixed，DP已有
known_work仅在固定W1混长短负载复现有限收益。** 不以局部修复代替整体性能目标。

详细机制见[Controller](engineering-progress-controller.md)，DP归因与停止证据见
[DP](engineering-progress-dp.md)，全部配置、命令与分项结果见[统一评估](engineering-progress-evaluation.md)。

## 实现与边界

protected admission 检查完整已知prefill，保留已接纳prefill尚未分配的容量；回收预算
覆盖post-forward运行请求和requester，并扣除victim自身下一步分配，避免释放页被
吞回后无法推动进展。同净页目标下优先合法exact whole-page remapping，最多额外
删B−1 token，保留原floor。仅原硬缺口有合法计划后，在同一victim集合合并有限
completion headroom，摊薄连续小额回收；该容量上界不进入逻辑已计算L或DP工作量U。

单次私有KV删除≤20%、逻辑已计算私有上下文保留≥50%、最多两个victim不变。
Scheduler拥有canonical，Worker执行镜像，全TP completion与实际等待后才整批
commit/free；没有新增rollback、迁移或跨池借页。fixed/requests仍默认。

## Correctness、micro、Serving分别验收

- Correctness **PASS**：CPU99项；DP本地fixture13+2项（非完整上游suite）；真实
  Qwen3-8B TP2单/双victim、延迟completion、Graph/eager/FP32 attention reference、
  release/rewrite/reuse通过。DP2等待者在abort前输出并改写本地释放页，peer在abort后
  继续112次payload验证forward，两池最终free32。适用lint/type checks通过。
- Micro **已测量，局部复制修正成立**：E4100的protected单victim释放1页、copy0、
  Worker0.373ms；fixed释放128页、copy301MB、0.903ms。释放量不同，不能说同动作
  提速。八候选protected单victim完整决策约572µs，pair约5ms；正式TP未使用pair。
  Worker测量不含forward或Scheduler commit/free，不代替Serving。
- TP2 Serving **负结果保留**：同源码同仪表三轮，fixed101.548tok/s、18/24，protected
  73.359tok/s、6/24，吞吐−27.76%。两边均零抢占/重算；protected18次回收，但1551次
  allocation miss（含重复等待重试）。剩余瓶颈有容量证据，停止继续调参。
- DP2 Serving **W1有限收益**：已有known_work正反向吞吐+16.09%/+21.52%，SLO均
  从21/24到24/24；副本prefill分配更均衡。W0为+4.11%/−0.88%，不支持稳定提速。
  每方向对应24条output IDs一致，但单项均零回收，收益不归因Controller。
- 组合 **24/24但无回收归因**：252.130tok/s，压力动作全0；真实压力能力另由
  correctness小池验证。usage tie-break **STOPPED / NOT_RUN**，无重复有效新鲜
  同分错选后本地miss且peer有余量的证据。

项目SLO仍TTFT≤2s且TPOT≤80ms；逐请求同时满足才算通过。小样本分位数不解释为
统计显著。有损质量未扩评，85条冻结样本不变，20%/50%无已验证的质量收益结论。

## 剩余瓶颈与停止依据

独立最终trace的514次protected计划中508次空计划，全为合法净容量不足。池681页，
8192-token请求需512页、free424，缺88；唯一4097-token victim最多净释放52，仍差
36页。该例其他running allocation为0，排除前方decode抢页。三次小额回收已全部
zero-copy；三次大回收的邻近zero-copy K低于floor。问题不能继续归咎逐页复制或
双victim扫描。停止条件已满足：回收变少但Serving仍落后，交付容量与排队证据。

后续若要继续，先明确在固定容量下如何取舍并发、排队和删除比例，并补质量收益
依据；不能反复部分回收绕过cap，也不能把调大pool、换输入或放宽SLO算成本轮收益。
当前不继续做参数扫描，不补TP反向，不重跑native或旧矩阵。

## 原始证据与失败保留

大产物根 `/data_ssd/qcrs/experiments/p1-v8`：

- 初始B0 `Engineering-Progress-B0/20261107-231330-initial` 保存起点HEAD/diff、用户
  staged.patch与旧protected参考；`Engineering-Amortized-B0/20261108-014007-initial`
  保存有界合并设计、CPU/static日志和最终correctness独立审计。
- 最终B0 `Engineering-Progress-Final-B0/20261108-021957-resume` 保存统一/容量复算、
  全29次command index/resource audit、`audit-progress.py` / `audit.json` 和
  `completion-manifest.json`。各run另存源码/input/config hash、argv、environment、
  affinity、NVML、exit code及原始payload/requests/work记录。

全部29次guard中27 PASS、2 FAIL。`Engineering-Progress-Baseline/20261107-232917-tp2-fixed-gpu12`
因GPU2启动free8.58GiB不足utilization0.4所需31.7GiB失败，未终止背景任务；稍后同配置
retry独立归档。`Engineering-Progress-Correctness/20261107-234244-pair-tp2-graph-delay-reuse`
因延迟probe未覆盖whole路径失败，修probe后另目录PASS。

开发失败也保留：Progress B0 `cpu-final.log`的hash_block_size fixture遗漏
（1 failed/85 passed，后86 passed）；Amortized B0 `cpu-development.log`的字段名/
旧pair期望（3 failed/96 passed，后99 passed）；`dp-state-offline-collection-fail.log`
的远程tokenizer离线collection失败，改用既有本地fixture后2 passed；初版micro分析
AssertionError、开发lint格式FAIL。旧轮原生PARTIAL、DP4 PARTIAL、CB1 0/24/60抢占、
旧protected 0/24/468抢占等报告原样保留，没有用新结果替换。

归档审计初版误读micro affinity文件位置的FAIL另存最终B0
`audit-development-1.log`；已按`micro-rank-*.json`的`cpu_affinity`修正审计，未重跑实验。

## 环境、资源与 Git

本轮只用用户指定物理GPU1/2（GPU0未用），现有Python3.10环境、Qwen3-8B BF16、
FA2/NHD/B16、MRV2与离线缓存；Worker主线程CPU20/21。所有GPU同guard锁串行，
NVML实际项目GPU仅1/2，合计峰值39.888672GiB，未触44GiB停止线或45GiB硬限。
未安装升级依赖、下载模型或改OMP/NUMA，背景任务保留，不声称独占硬件。

本轮阶段提交：

- `d6d4e7602b`：完整已知prefill admission/净回收预算、整页重映射优先及进展验证。
- `533fa38ed5`：原合法victim内有限completion headroom、有界回收合并与CPU验证。
- `deb3d9e9d6`：最终测量入口、严格统一分析及合法容量诊断。
- 最终报告/导航提交：准确HEAD与上述列表收入最终B0 `completion-manifest.json`。

独立index仅提交本轮精确文件，带attribution/sign-off。用户staged prompt4、计划的
staged/unstaged版本和Scheduler中文注释原样保留；初始staged.patch字节一致。
status/delivery只置顶本轮导航，移除新增导航后与起点e3550ac916逐字一致；旧报告
不变。未reset/amend/rebase、push或创建PR，大产物未入Git。最终dirty仅用户三文件。
