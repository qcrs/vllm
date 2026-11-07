# Protected Controller 与 DP：本轮收尾

2026-11-07，承接 `80564d1b66`，完成 protected 预算、最多双 victim 的执行协议，
以及最终源码的 correctness、micro 和固定 Serving 复测。fixed 预算与 requests 路由
保持默认。DP usage tie-break 缺少重复有效同分错选证据，**STOPPED / NOT_RUN**。

实现与详细证据分别见 [Controller](engineering-protected-controller.md)、
[DP 诊断与组合生命周期](engineering-protected-dp.md)、
[统一评估](engineering-protected-evaluation.md)。下面的结论只覆盖实际验证范围。

## 实现、所有者与停止判断

选择时 `E=effective_kv_len+q`、`L=num_computed_tokens+q`，只接纳完整 prefill、无旧
在途且 q=0/1 的候选。既有 retention 保持当前 E 域语义；protected 额外要求私有 E
单次最多删除20%、已计算私有 L 至少保留50%，继续保护 prefix、sink/recent 和
min_keep。未计算的 prompt 与未知未来输出不进入 L；preempt 沿原生重算生命周期。

真实 allocation miss 下优先单 victim；无单个可覆盖时才比较候选对，精确释放 D 页，
优先最低逻辑保留率与平衡，再比较 zero-copy、复制量和稳定顺序。每对只检查保留率
交点相邻的至多两个整数页分担；候选对仍全部枚举。没有合法单/双计划就返回原生
等待/preemption，不降低保护边界、不增加第三个 victim。独立 Fraction 穷举 oracle
核对解析选择。

Scheduler 拥有 canonical row/E 和 Block；Worker 执行镜像。Scheduler 先完整 prepare
与 validate，再 deindex 私有 suffix；同序 tuple 贯穿输出、Worker 和 ACK。Worker
先检查整批 host contract，逐 victim 执行既有 KeepPlan、compaction 与全 TP completion。
实际输出等待后，Manager 验证全部存活结果才一起 commit/free。mutation 后失败
fail-stop；Worker fatal 路径直接退出，避免 CUDA teardown 阻塞 peer collective。
设备侧没有 rollback。abort 晚 ACK 与同一步正常结束仍走原生生命周期边界。

保护比例是容量 contract，不能解释为模型质量或 admission reservation。最终 DP
小池运行显示，前方 decode 可以先占用释放页，等待请求在 abort 前尚未开始；原超时
FAIL 保留。修订 correctness harness 的等待顺序后验证 abort、peer continuation 和
真实 rewrite/reuse，未修改 Runtime、正式输入或超时阈值。

DP usage 是 `1-free/(num_gpu_blocks-1)`，free 包含可驱逐缓存页；Runtime 没有 wall-age
expiry，诊断250ms筛选不是路由新鲜度合同。固定 W1 诊断25次路由、3次 work score
同分、1次新鲜筛选内选高 usage，但0次 miss/reclaim/preempt，不满足 prompt4 门槛。
因此不添加 usage 权重、tie-break、迁移或跨池借页，known_work/pending ADD 保持原义。

## 最终验证与性能

- Correctness：CPU预算/Pool/Scheduler 77 passed；DP watermark/route 15 passed，后者
  使用既有本地 tokenizer fixture runner，不称完整上游 suite。真实 Qwen3-8B TP2 单/双
  victim、延迟 completion、全36层 payload、Graph replay、commit/free/rewrite/reuse
  PASS；DP2 独立池、abort affinity、peer continuation 与真实页复用 PASS。
- Micro：Controller 决策、TP2 Worker 完整回收链、compaction 和路由成本已分别测量。
  protected 大 K 小额释放可能复制更多数据，双 victim 有额外选择与执行成本；未证明
  本轮 micro 加速。Worker micro 不含模型 forward 或 Scheduler commit/free。
- TP2 Serving：native/fixed/protected均0/24、goodput0；输出吞吐分别5.882/
  43.037/4.905tok/s。protected相对fixed观测下降88.60%，480次回收、468次抢占，
  正式最大victim=1；没有Serving收益，SLO_FAIL，停止反向扩展与调参。
- DP2 Serving：W1正向requests/known_work为19/24、17/24，反向17/24、18/24；
  known_work相对requests吞吐变化正向−21.607%、反向+0.332%。W0正向16/24、20/24，
  反向18/24、17/24；变化正向+45.749%、反向−0.638%。正向差异未稳定复现，
  NO_STABLE_ROUTING_BENEFIT / SLO_FAIL；不覆盖历史有限收益，不扩大固定轮次。
- 组合Serving：W1 known_work+protected为24/24，goodput2.022092req/s、
  258.828tok/s，但miss/reclaim/preempt/recompute均0；该次SLO_PASS不能归因protected。
  真实压力与abort/reuse以独立correctness小池证据为准。

故障注入首次 CUDA teardown 卡住 FAIL、单 victim 命令遗漏参数 FAIL、最终 DP 旧等待
条件超时 FAIL，以及优化前 native 被中断的 PARTIAL 全部保留。guard PASS 仅表示运行
与资源检查通过，不代表 SLO 或质量通过。历史 CB1 `0/24`/60次抢占、原 DP4 PARTIAL、
旧 TP/Serving FAIL/PARTIAL 不改写；85条质量样本不扩，有损质量没有保持性结论。

## 复核、资源与 Git

原始产物根目录 `/data_ssd/qcrs/experiments/p1-v8`；本轮初始状态与最终审计位于
`Engineering-Protected-B0/20261107-204106-initial`。每个 GPU run 保存准确 argv、
HEAD/diff、源码/helper/input/config hash、环境、CPU affinity、NVML 与退出码。
`audit-protected.py` / `audit.json` 包含本轮全部 guard（含失败与中断），
`final-correctness-analysis.json` 和 `final-unified-analysis.json` 分别核对行为与测量口径；
`final-validation/` 保留 CPU、适用 pre-commit、manual mypy-3.12 与复算日志。
最终提交、文件 hash 和用户改动保护见 `completion-manifest.json`。

沿用现有 `.venv`、Qwen3-8B BF16、FA2/NHD/B16、MRV2、离线缓存；GPU0/1 双卡，
Worker 主线程 CPU20/21，所有 GPU 经同一 `run_guard.py` 锁串行，项目显存合计
44GiB停止、45GiB硬限。未使用 GPU2、下载模型、安装升级依赖或调整 OMP/NUMA。
共享机器的背景 GPU 任务保持原状；不声称独占硬件。

本轮阶段提交：

- `5cb185008e`：protected floor/cap、单 victim 与 CPU/TP2 验证。
- `b149f7a0f2`：最多双 victim、整批验证/commit 与 failure fail-stop。
- `de34bcad87`：DP usage 诊断、停止依据与最终测量入口。
- `af174ca5bd`：双 victim 交点解析分担与独立穷举 oracle。
- `d363ca2b4a`：真实 TP2/DP2 最终 correctness 与成本 micro。
- 最终报告与导航提交：以 `completion-manifest.json` 的最终 HEAD 为准。

所有提交使用独立 index 精确纳入本轮文件，并带 attribution/sign-off。初始用户
staged prompt4、staged/unstaged 计划与 Scheduler 中文注释原样保留；status/delivery
仅置顶本轮导航，移除导航后与初始文件逐字一致。未 reset/amend/rebase、push 或
创建 PR，大产物未入 Git。
