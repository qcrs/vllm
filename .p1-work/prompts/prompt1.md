# P1 v8 Physical KV Cache Reclamation — Long-Running Goal

你是负责 LLM Inference Runtime、vLLM、CUDA、Triton 和 GPU 性能工程的高级工程师。

目标是在现有 Worktree 中完成 P1 v8 Physical KV Cache Reclamation 的完整工程实现、GPU 验证、性能优化、质量与 Serving 评测及文档交付。不要只输出分析和计划，持续实际修改、运行、调试、优化和提交。

## 1. 固定工作环境

环境激活：执行 p1，使用现有项目激活脚本。

- Worktree：`~/learning/llm-kv-lab/worktrees/p1-v8-physical-kv-reclamation`
- Branch：`p1/v8-physical-kv-reclamation`
- Baseline：vLLM `v0.26.0`，Commit `568afb3a13806beb53bb2e6bd518269357b237c0`
- Python 环境：`~/learning/llm-kv-lab/.venvs/vllm-v026-torch211-cu129-py310`
- Python 3.10 / PyTorch 2.11.0+cu129 / Triton 3.6.0
- CUDA Toolkit：`/usr/local/cuda-12.1`
- GCC/G++：12
- 模型：`/data/models/Qwen3-8B`
- 实验目录：`/data_ssd/qcrs/experiments/p1-v8`
- 数据集：`/data_ssd/qcrs/datasets`、`/data_ssd/qcrs/hf_cache/datasets`

**模型统一使用 Qwen3-8B BF16。** 不再使用 Qwen3-0.6B、4B，不做小模型替代实验。Kernel 单测允许使用合成 Tensor。

GPU：

- A100 80GB，默认物理 GPU 1。
- M0/M1/M2 单卡实验使用 `P1_CUDA_VISIBLE_DEVICES=1`。
- TP=2、DP=2 才使用 `P1_CUDA_VISIBLE_DEVICES=0,1`。
- 整个项目进程显存硬上限 45GiB，目标峰值不超过该值；通过 NVML、nvidia-smi 实际监控，不能只依赖 `gpu_memory_utilization`。
- 不得擅自终止其他 GPU 任务。

保持现有依赖和编译环境，不随意升级 CUDA、PyTorch、Triton 或 GCC。

## 2. 设计依据

完整阅读 `.p1-work/reference/` 中四份 v8 原始设计文档，以及本次提供的 Goal、简历评审和补充材料。

技术决策以 v8 设计为准，具体实现以实验设计与技术选型方案为准，验收以实验矩阵为准。

保留原文档的技术路线、调用链、参数和测试要求，不重新设计已经确定的架构。发现真实源码冲突时记录 `DESIGN_CONFLICT` 和证据。

## 3. 执行路线

按以下顺序持续推进，不需要逐阶段等待确认：

`M0 → M1 → M2 → Prefix → Graph → TP → DP`

- **M0**：固定源码、环境、Native Baseline、Instrumentation、Allocator 观测和 CPU Oracle。
- **M1**：Logical/Physical 分离、Whole-block Zero-copy、Stable CTA Triton In-place Compaction、Worker Completion、Canonical Ownership、真实 Block Release/Reuse。
- **M2**：Allocation Miss、Candidate Selection、Bounded Headroom、SinkRecent、TriAttention Shared Scoring、Deferred Retry、Native Fallback，以及真实 Serving 评测。
- **Prefix / Graph / TP / DP**：按照 v8 既定方案完成兼容性扩展和独立 Gate。

M0/M1 不得因 8B 质量评测尚未准备好而停滞，但真实模型级 Engine 验证统一使用 8B。

## 4. 工程要求

以完整工程交付和实际性能为目标，不要为了追求最小实现、最安全的简单实现而主动缩减功能。

优先复用 vLLM 原生数据结构和生命周期，完成真实 Runtime 集成，而不是停留在 CPU Oracle、Kernel Fixture 或 Demo。

Triton 正确性通过后，使用 CUDA Events、NSYS、NCU 进行 Profiling，依据实际瓶颈优化 Tile、Channel Tiling、Warp、寄存器、Occupancy 和内存访问。不要未经测量就接受初始 Kernel 配置，也不要无依据建设多套 Backend。

避免无意义的通用框架和过度抽象，但必须解决真实的 Ownership、GPU Completion、异步执行、Block Reuse 和兼容性问题。

## 5. 数据、实验与证据

先审计本地数据集，不重新下载、不复制原始数据。

使用 Qwen3-8B Tokenizer 和 Chat Template 建立 Manifest。质量评测以 BabiLong 8K/16K/32K 和审计通过的 SCBench 子集为主。

区分 Full KV、SinkRecent、Random、TriAttention-inspired 的质量与 Serving 结果。

记录真实显存、KV 容量、TTFT、TPOT、Throughput、SLO Goodput、Scoring Latency、Compaction Latency 和 GPU 资源使用。

所有实验保存命令、参数、Commit、模型、数据集、GPU 映射、日志、结果和退出码。状态使用 `PASS / FAIL / BLOCKED / TO_VERIFY / NOT_RUN`，不得用静态检查代替 GPU 或 Runtime 验证。

## 6. 文档与 Git

项目资料统一放在：

`.p1-work/`

沿用已有目录：

- `prompts/`
- `reference/`
- `plans/`
- `source-analysis/`
- `notes/`
- `reports/`
- `experiments/`

大型实验数据统一存放在 `/data_ssd/qcrs/experiments/p1-v8`，不污染源码目录。

每项核心实现同步记录 vLLM 原始调用链、修改位置、数据结构、Ownership、状态转换、GPU Stream/Event、Kernel 地址与性能分析，以及实际测试结果。文档必须与代码同步，供后续深入学习。

保留现有 Git 规则：

- 只在指定 Worktree 和分支开发。
- 不污染 `third_party/vllm` 学习基线。
- 不覆盖已有提交和实验。
- 按功能和阶段提交。
- `.p1-work/` 选择性跟踪，不提交外部软链接、实验软链接、模型和大型产物。
- 每个重要 Commit 记录测试结果和遗留问题。

## 7. 持续执行

先检查 Worktree、Git、参考文档软链接、环境、GPU、模型和实验目录；完成 M0 基础工作后持续进入 M1、M2 和各扩展 Gate。

每个阶段执行：源码追踪 → 实现 → 编译 → 测试 → 修复 → 回归 → Profiling → 优化 → 更新文档 → Commit → 下一阶段。

独立阻塞不得阻止其他工作。只有设计冲突、不可恢复风险或真实外部阻塞才暂停受影响部分。

**最终交付：完整 vLLM Physical KV Reclamation Runtime、优化后的 Triton Kernel、真实 Qwen3-8B GPU/Engine/Serving 实验、Block Reuse 证据、完整 Git 历史、同步的技术学习文档和可复核的性能报告。**
