# Protected Controller：进展与复制成本修正

本轮从 `e3550ac916` 继续。只用用户指定的物理 GPU1/2，Worker 主线程 CPU20/21；
其余 prompt3/4 环境、资源、归档与 Git 约束继承。历史 FAIL/PARTIAL 保留。

## 根因与实现

旧实现覆盖当前 allocation miss，却不覆盖完整已知 prefill，也不扣除 victim 自己的
下一步尾页分配。前方 decode 可拿走释放页；仅使第一次 retry 成功，仍可能接纳
无法完成 prefill 的请求，随后反复抢占重算。

新逻辑仅在 opt-in protected pressure 路径启用。等待 admission 检查完整已知 prefill，
给已接纳 prefill 保留尚未分配的容量；预算同时覆盖 post-forward 后运行请求和等待者
的已知 prefill/下一 decode。物理需求使用 E，逻辑保护 L 仍只包含已计算上下文。
复用真实 allocator 的 full-sequence gate/reserved-block 参数，不新增页所有权状态。

对 N 个源页、保留 K、下一步 t 个 token、原分配需求 A，净容量贡献为
`N - ceil((K+t)/B) + A`。选择单/双 victim 先覆盖净缺口，保护单次删除≤20%、逻辑
保留≥50%；无合法覆盖就等待/原生抢占，不新增第三个 victim，不预测未知输出。
这避免把“释放页数”误当成“等待请求可消费的净页数”。

SinkRecent 在同净容量目标下寻找邻近 exact whole-page remapping，最多额外删除
B−1 个 token，仍受原保护 floor 约束。例如 E4100、D1，K4096 需复制，K4084 可以
重映射。尾页场景可能物理多释放一页并由 victim 自身消费，净贡献不变。
双 victim 的逻辑保留率变为分段单调，用二分交点相邻 split 选择并与独立 token
枚举 oracle 比较。这一取舍不等于质量无损，85条质量样本未扩大。

fixed/requests 仍默认。未更改 TP completion、canonical commit/free 或 DP 路由契约。
DP correctness harness 恢复严格的 abort 前进展与等待者改写复用要求。延迟 probe
移至 reclaim 入口以覆盖 token/whole 两条路径；首次 pair 的 probe 漏注入 FAIL 保留。

## 正确性证据

大产物根目录为 `/data_ssd/qcrs/experiments/p1-v8`。

- B0：`Engineering-Progress-B0/20261107-231330-initial`。
  相关 KV/Pool/Scheduler CPU suite **86 passed / 289 deselected**；本地 tokenizer
  fixture 的 DP suite **15 passed / 36 deselected**。开发 FAIL/setup FAIL 保留。
  适用 pre-commit 与 manual mypy-3.12 通过，日志见 B0。
- `Engineering-Progress-Correctness/20261107-234132-single-q0-tp2-graph-reuse`：
  PASS，两个 rank 均 continuation、Graph/eager reference、FP32 attention 和
  release/rewrite/reuse；回收两次，最终 free32。
- `Engineering-Progress-Correctness/20261107-234635-pair-tp2-graph-delay-reuse-fixed-probe`：
  PASS，首批两 victim（K223 copy、K126 whole），第二 rank 第二 victim 延迟≥2.5s
  时 canonical 不变；全 rank 完成后 ACK、commit/free，随后 Graph 与真实改写复用。
  共7批、8个 victim，最终 free32。前次 probe FAIL 目录保留。
- `Engineering-Progress-Correctness/20261107-234729-dp2-protected-progress-reuse`：
  PASS，独立 Pool33；C 在 abort A 前输出，并改写本地释放页 `[2,18]`；peer 没有
  回收，abort 后仍有112次经 payload 核对的 forward；最终两个 Pool 均 free32。

B0 的 `audit-correctness.py` 与 `correctness-analysis.json` 独立核对上述源码 hash、
GPU映射、guard、affinity、cap、rank计划一致、completion时序、Graph与改写复用。
q0 极短窗口不要求 observer 采样成功；延迟批次必须实际观测到 canonical 保持不变。
correctness PASS 不能解释为速度、SLO 或模型质量 PASS。

micro 与 Serving 见后续统一评估；不能用 correctness 代替性能结论。
