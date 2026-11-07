# Protected Controller：有界回收与批次提交

本轮承接 `80564d1b66`，默认仍为 fixed。protected 显式启用，在原 retention、
min_keep、sink/recent 和共享 prefix 之外，单次最多删除私有 E 的20%，保留至少
已计算逻辑私有 L 的50%。这些是实验边界，不是模型质量保证。

## C1：单 victim

`E=effective_kv_len+q`，`L=num_computed_tokens+q` 来自选择时尚未执行
`_update_after_schedule` 的 Scheduler 状态。q 仅允许0/1，完整 prefill 且无在途请求
才参与；P 保护到最后一个共享页。retention 仍按当前私有 E 计算。
Request 的逻辑计算计数不因 reclaim 回退，preempt 则按既有路径归零并重算。

固定释放 D 页时取 `K=min(E-1, B*(N-D))`，仅在满足 floor 时选中；优先更高
回收后私有逻辑保留率，再比较 zero-copy、copy bytes 和 arrival/id。

- CPU：现有 KV/Pool/Scheduler suite 的 pressure/reclamation 选择，60 passed、
  200 deselected。包括逐页枚举最大合法 K、共享 prefix、连续真实 Pool commit
  直到逻辑下限，日志见 B0 的 `c1-cpu.log`。
- TP2：`Engineering-Protected-C1/20261107-204256-single-tp2-graph-reuse`，
  Qwen3-8B BF16，GPU0/1，Worker 主线程CPU20/21，guard PASS，峰值
  18,144,559,104 bytes。两次单 victim reclaim，token 与 whole-block 路径，
  两 rank 计划/payload一致，均实际改写释放页2/17；各47次 Graph/eager 和48次
  FP32 attention reference。该受扰动生命周期不作为 Serving 性能。
- 适用 pre-commit 通过；阶段测试与静态日志归档在
  `/data_ssd/qcrs/experiments/p1-v8/Engineering-Protected-B0/20261107-204106-initial`。

历史 retry_pass 0/24、60次 timed preemption、DP4 PARTIAL、TP SLO_FAIL 均保留。
后续批次和最终评估在本报告追加，不覆盖历史结果。

## C2：最多双 victim 与整批结果验证

没有单 victim 能覆盖真实 D 时，按稳定顺序枚举两个请求的合法页分担，取每个
页目标的最大 K；先比较超额释放、最低逻辑保留率及两者差距，再比较 zero-copy、
估计复制量和 arrival/id。无法覆盖时返回空计划，沿原生等待/preemption。

Scheduler 先 prepare 全部 intents 并核对 source E/row/binding，再一次 deindex 私有
suffix。SchedulerOutput、ExecuteModelState 和 ModelRunnerOutput 使用同序 tuple；
Worker 先验证整批 slot/binding/E，然后逐 victim 执行既有 KeepPlan、搬移和全 TP
completion。实际输出等待后，Manager 先验证全部存活结果，再 commit/free。
这不是可回滚设备事务。native abort 立即释放，晚到 ACK 跳过已结束 victim；同一步
正常结束则先提交，再释放。单在途支持域不增加异步/connector contract。

- CPU：77 passed / 300 deselected，包括单/双 victim、坏第二 plan 零部分 deindex、
  坏第二 ACK 零部分 commit/free、不可覆盖及结束请求边界。适用 pre-commit 和
  manual mypy-3.12 通过，日志见 B0 的 `c2-validation/`。
- 双 victim：`Engineering-Protected-C2/20261107-205519-pair-miss-tp2-graph-delay-reuse`，
  pool33、A256/B128、C160 在 step16 的真实 miss D4；首批 whole-block E272→224、
  token E142→128。延迟3秒期间 canonical/free 不变，累计14次 victim 回收；两rank
  各61 FULL、46 matched-eager、72 FP32 attention，36层 payload 验证；实际改写
  页2/3/4/17/25/26，最终free32。guard PASS，NVML峰18,152,947,712 bytes。
- 单 victim 最终协议：`Engineering-Protected-C2/20261107-211124-single-final-q0-tp2-graph-reuse`，
  q0、D1，两次 token/whole-block 回收；两rank各62 FULL、47 matched-eager、48 FP32
  attention，实际改写2/17、最终free32，guard PASS。前次命令遗漏脚本要求的
  `--demand-pages 1` 导致 harness assertion，原 FAIL 目录保留。
- mutation 后第二 victim/rank1 注入失败：首次 `20261107-205848-pair-second-fail-stop`
  因 CUDA teardown/peer collective 卡住而 FAIL，保留。Worker fatal 异常现在直接退出，
  由原 executor death monitor 终止 peer；修复后的
  `20261107-210303-pair-second-fail-stop-fixed` PASS，canonical commits0、free不变、
  executor_failed=true、Worker exit[-9,1]。没有恢复、重算或部分释放。

GPU均 Qwen3-8B BF16、GPU0/1、Worker主线程CPU20/21、同一guard串行。这些生命周期
运行包含 payload/同步/故障注入，不作为 Serving 吞吐或质量保持证据。最终 micro、
Serving 与 DP 条件结论另见后续统一评估报告。

## 最终源码与有限复核

双 victim 页分担现在只检查两条线性保留率曲线交点相邻的两个整数页目标，越界时
截到合法边界。候选对仍全部枚举；精确 Fraction 排序与原目标相同。现有 CPU suite
增加独立穷举 oracle，覆盖不同 E/L/P、尾页及边界，避免逐页扫描随 D 增长。

最终源码复核在 `Engineering-Protected-Final-Correctness/`：

- `20261107-213244-pair-tp2-graph-delay-reuse`：双 victim、延迟 completion、
  TP2 全36层 payload、Graph replay 和真实释放页改写 PASS，guard峰值
  18,152,947,712 bytes。
- `20261107-213340-single-q0-tp2-graph-reuse`：单 victim q0、token/whole-block、
  TP2/Graph 和真实页面复用 PASS，guard峰值18,144,559,104 bytes。
- 三个 CPU suite 选择 `reclamation or pressure`：77 passed / 289 deselected；
  DP route/watermark 选择15 passed / 36 deselected。适用 pre-commit 和
  manual mypy-3.12 PASS。准确命令、日志及 hash 收入 B0 最终验证归档。

protected 精确释放 D 页不提供后续 admission reservation。DP 最终 lifecycle 首次
运行显示，前方 decode 可以先重新占用刚释放的页，使等待请求尚未 admission；
原超时 FAIL 保留，不能把保护比例或预测页数正确写成等待请求必然立即开始。
组合的 abort/continuation/reuse 结果及该限制见
[DP 报告](engineering-protected-dp.md)。micro 与 Serving 分列于
[统一评估](engineering-protected-evaluation.md)，不由这些 correctness 运行推断性能。
