# P1 v8 工程收尾审查

本轮按用户最新要求聚焦实现完整性、功能缺陷、冗余检查与 TP/DP 收尾。
不扩大质量样本；已有质量结论保持探索性，不作为性能收益成立的证明。
依据为 prompt1 与四份 v8 reference；只修改本 worktree，不创建上游 PR。

## 验收边界

- 必须修复：违反 logical/physical、GPU completion、canonical ownership、
  Prefix identity、TP 全 rank 完成、DP 本地池隔离的缺陷。
- 必须核实：Triton 实测优化进入 Runtime；普通 decode 是否存在可删除的重复
  launch/校验；TP eager 开销定位到源码并验证改进，不能只旁路 retention=1。
- 保留负结果：既定压力负载的 TTFT/TPOT SLO 失败、质量保持尚未证明、未测组合。
  不为消除 FAIL 改 SLO，不用新路由或质量方法扩张此次范围。

## 最终核对与证据

| 审查项 | 现有证据 | 本轮状态 |
| --- | --- | --- |
| E 生命周期 / 原生重算 / APC hit | Core/Prefix 轨迹与本轮 kernel 重排、prefill、padding 测试；prepare 不推进 E，postprocess 同步推进 GPU L/E | PASS |
| partial batch / q0 / maintenance / EOS / abort | 原 Scheduler/8B 轨迹；本轮 q0、maintenance 慢 rank 与 DP abort 回归 | PASS |
| Prefix protected prefix / suffix 全 alias 取消 | 完整 Prefix suite 与第三请求真实 hit；本轮 Prefix+Graph+TP 重写 10 页 | PASS |
| Graph 静态地址 / 下一 replay metadata | 原 Graph/组合轨迹与本轮融合后的 FULL replay；新窗口 274 个 spans 完整 | PASS |
| TP 统一计划 / distinct GQA / 全组完成 / fail-stop | 原 ties/GQA/故障轨迹；本轮两条慢 rank 回归均等待全组 ACK 才释放 | PASS |
| DP 本地 allocator / affinity / peer 不变 | 本轮 C 取得 8 页并全层重写；peer 128 输出逐 token 相同，abort 后 112 次 forward，最终 free32/32 | PASS |
| 初始化支持范围 | 修复 restricted FullAttentionSpec、pooling 与 ALiBi 误放行，均有拒绝测试 | PASS |
| 冗余 launch / 传输 / 校验 | 两个 decode launch 合入原生 kernel；删除 E 字典传输和被 ownership 检查覆盖的重复断言 | PASS |
| Triton 压紧 / 评分调优及 Runtime 参数 | M1/M2 CUDA Events、NCU 与本轮真实 Serving NSYS；默认 T16/W4/stages1 实际使用 | PASS |
| Graph/TP profile 缺失证据 | 新窗口 274/424 个 graph spans 和 162 个 TP collective spans 全匹配；旧 PARTIAL 保留 | PASS |
| TP eager 吞吐预算 | 融合后 ABBA 全部 6+6 轮合并开销 6.871%，超过 5%；历史 14.33% 保留 | FAIL |
| TP eager 跨 server 差异根因 | host/thread CPU 诊断保留；独立 launch 已消除，仍不能确定全部等待/节奏差异的原因 | TO_VERIFY |
| 最终相关测试 | Core 24 passed；single-type/Prefix/compaction/scoring/本地 executor 130 passed，共 154 项 | PASS |
| 最终代码与文档检查 | 全部适用 pre-commit 钩子（含 mypy 3.10）及 git diff --check 通过；日志在 Review | PASS |
| 整个上游测试集 | OPT 配置及 custom-executor 的 0.6B 配置未缓存；保留失败日志，不下载替代模型或放宽验证 | BLOCKED |
| 扩大有损质量评测 | 按用户最新范围冻结；85 条探索性样本不能证明质量保持 | NOT_RUN |

上述功能 PASS 限于报告实际验证的配置，不覆盖 TP2×DP2、async/spec/connector、
PP/DCP/PCP 等未测组合。既有 M2/Prefix/TP/DP 压力 SLO 的 FAIL 不改口径。
本轮功能审查与回归已完成；性能未通过项作为明确负结果交付，不能写成预算通过。
详细 finding、修复、命令与原始证据目录见
[final-review.md](../reports/final-review.md)。
