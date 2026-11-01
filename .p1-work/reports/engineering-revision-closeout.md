# CB1 与 TP 疑似退化复核收尾

2026-11-01，本轮以 `e0fe252fd6` 为起点，落实用户要求的 CB1/TP 深入复核与合理修正。
已完成源码追踪、一个有限预算增量、必要 CPU/真实 TP2验证及有限正式对照，
不重做已交付的 DP/PL2，不扩大样本、依赖、模型或旧矩阵。

| 范围 | 当前结论 | 报告 |
| --- | --- | --- |
| CB1机制 | 首次重试成功掩盖后方抢占重算；不是新发现的页释放/ownership错误 | [Controller](engineering-revision-controller.md) |
| CB1修正 | 显式 prefill_fixed：已知prefill积压用fixed，纯decode用retry_pass；54项CPU及真实TP2控制链通过 | 同上 |
| CB1 Serving | 三轮均零timed抢占，15/24；相对fixed18/24、吞吐−2.005%，无收益，fixed仍默认 | 同上 |
| TP反向 | dynamic E相对baseline+0.200%，两边18/24；未复现前轮−0.740%/16/24，仍无稳定Serving收益 | [TP](engineering-revision-tp.md) |
| micro | 本轮不新增完整回收micro；原cache miss10→0与历史TP2a/TP2b量级、支持范围保留 | 同上 |
| 停止 | 不扩大前瞻/调参、gate或prefill重排；未解决的TTFT与全部历史FAIL/PARTIAL保留 | 两份分项报告 |

## 验证与复算入口

全部 Python 通过 `.venv/bin/python`，沿用 prompt3 的 `env.sh`；GPU经共享guard
串行，GPU0,1/CPU20,21，未使用GPU2，项目合计峰最大19,935,526,912 bytes。
本轮5次GPU guard均PASS：TP反向2次、CB新诊断1次、CB正式2次。guard PASS
仅代表运行/资源通过，不代替SLO或逐层payload证明。

CPU与静态原始日志保存在
[Revision B0](../experiments/Engineering-Revision-B0/20261101-151241-cb-tp-audit)：

```bash
.venv/bin/python -m pytest tests/v1/core/test_scheduler.py tests/v1/core/test_kv_cache_utils.py -k 'pressure or reclamation' -q
```

结果54 passed / 200 deselected；pre-commit及manual mypy-3.12通过。初始CPU fixture
失败和首次format调整记录保留；没有通过删除失败日志或缩小SLO获得PASS。

真实模型诊断入口为 `profile_tp_ttft.py --nsys-child`，CB正式入口为
`serving_reclamation.py`；完整argv、运行HEAD/diff、源码/输入hash、GPU映射、
affinity、NVML、请求与退出码均在各 run 的 `invocation.json` 和相邻产物中。
旧诊断的实际工作量与新诊断同口径复算：

```bash
.venv/bin/python .p1-work/scripts/analyze_tp_ttft.py .p1-work/experiments/Engineering-Revision-CB-Diagnostic/20261101-153141-prefill-fixed-sink --candidate-output .p1-work/experiments/Engineering-Revision-CB-Diagnostic/20261101-153141-prefill-fixed-sink/candidates-complete.json
.venv/bin/python .p1-work/experiments/Engineering-Revision-B0/20261101-151241-cb-tp-audit/analyze-cb-revision.py .p1-work/experiments/Engineering-Revision-CB-Diagnostic/20261101-153141-prefill-fixed-sink --candidates .p1-work/experiments/Engineering-Revision-CB-Diagnostic/20261101-153141-prefill-fixed-sink/candidates-complete.json --output .p1-work/experiments/Engineering-Revision-CB-Diagnostic/20261101-153141-prefill-fixed-sink/work-audit.json
```

正式两组复算入口见分项报告；只读复算不再运行GPU。最终检查/命令与退出码、
源码/分析hash和本地提交列表保存在B0的 `completion-manifest.json`。

## 交付边界与简历表述

代码阶段提交 `9af8339c25`；文档/证据阶段提交见本文件所在commit及completion manifest。
初始仅 `status.md` / `delivery.md` 的用户修改完整保留，导航只暂存本轮增量；
未push、创建PR、reset/amend/rebase，未把大产物加入Git。

可讲述的技术结论是：构造尾页与真实分配顺序的有界预算，并识别首次重试成功率
遗漏的重算放大；以已知prefill阶段选择回收粒度，用CPU/TP2和正式A/B界定收益边界。
TP部分可讲长度特化、缓存状态与端到端占比，以及正反向验证对阈值指标的必要性。
不写“动态预算优于fixed”或“TP已解决TTFT”；输出一致与payload正确也不替代质量保证。
