# P1 v8 性能继续优化

授权为用户在功能收尾提交 `4ad1a03e1c` 后要求继续优化 TP eager 与压力 SLO。
保持模型、环境、GPU guard、负载、SLO 与历史证据；不新增任务质量评测。

## 判断与验证

1. 定位跨 server 的约 54/62 tok/s 两档差异。优先在同一服务进程中干预 CPU
   placement，记录实际运行 CPU、主线程 wall/CPU 时间，区分环境放置与新增路径。
2. 只落地证据支持的实现或启动配置优化。正式 native/noop 采用相同 CPU 配置，
   保留完整预热、ABBA、每 server 三轮，不将 profiler 吞吐混入。
3. 用固定压力负载复测实际收益，必要时调优原生 chunked-prefill/batch 配置；
   变更配置明确单列，不替代旧协议的 FAIL，不放宽 TTFT/TPOT 或减少请求。
4. 验证 token identity、真实回收/页复用与 completion invariant，更新报告并提交。

数据根为 `/data_ssd/qcrs/experiments/p1-v8/Perf/`；运行均由 `run_guard.py`
串行监测。GPU2 其他用户任务保持不动。
