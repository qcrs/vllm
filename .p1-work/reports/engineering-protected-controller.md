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
