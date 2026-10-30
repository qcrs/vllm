# M2 TriAttention-inspired Shared Scoring

## 范围与环境

完成paged frequency scoring、真实Qwen3-8B normalized-Q校准、shared计划、Random负对照，
以及profiling后的phase提取优化。仍是shared retention适配，不能称为原per-head
TriAttention复现或无损压缩。M2压力控制提交`d0b3ca3e91`之上的源码快照在各run的
invocation.json/source中。环境沿M1：BF16、FLASH_ATTN/B16、MRV2/Eager、TP1、物理GPU1，
APC/async/spec/connector off，单在途。所有GPU任务经run_guard串行并NVML监控。

## OBSERVED

产物根 `/data_ssd/qcrs/experiments/p1-v8/M2`。

| 目录 | 结论 |
| --- | --- |
| `20261031-000856-normalized-q-calibration` | PASS；16,384 tokens、36层各32 Q heads的pre-RoPE q_norm统计；峰值20,524,826,624 bytes |
| `20261031-000942-shared-scorer-correctness` | 3 GPU tests PASS；独立FP32复数reference、GQA与stable计划 |
| `20261031-001456-triattention-live-repeat-reuse` | 8B两次回收后36层valid KV bit-exact，FP32 reference Attention正确，B复用A释放IDs7–18 |
| `20261031-002637-shared-scorer-phase-hoist-correctness` | 优化后3 GPU tests PASS；峰值515,899,392 bytes |
| `20261031-002849-triattention-phase-hoist-live-repeat-reuse` | 优化后重复回收、逻辑位置/物理slot、FP32 Attention与真实覆写reuse PASS；峰值17,098,080,256 bytes |

完整命令与exit code在run的invocation/result中。校准命令为激活环境后以run_guard运行
`@python .p1-work/scripts/calibrate_frequency.py`。正确性命令为
`@python -m pytest tests/v1/worker/test_kv_compaction.py -k frequency -q`。
live命令设置`P1_RECLAIM=1 P1_POLICY=triattention P1_PATH=token P1_LIFECYCLE=repeat
P1_TRACE=1 P1_STATS=.../20261031-000856-normalized-q-calibration/frequency.pt`并运行
`@python .p1-work/scripts/live_reclamation.py`。

校准模型config SHA为`f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30`，
WikiText SHA为`4cd0f6876d07a413aa911261ff6d363c72d757d47f0fdd6015702014c89cb9c7`。
频域stats `[36,32,3,64]` FP32共884,736 bytes；metadata记录完整token SHA及来源。

## 性能与资源

`benchmarks/kernels/benchmark_kv_scoring.py` 使用真实36层/8KV/128D/B16形状的合成
BF16 packed NHD，随机physical row，logical L=39743，3次warmup、10次CUDA-event测量。
比较初始`20261031-001803-shared-scoring-profile-initial`和优化后
`20261031-002705-shared-scoring-profile-phase-hoist`，相同stats/参数。

| E | 部分 | 初始median ms | 优化median ms |
| --- | --- | --- | --- |
| 4096 | raw36层 | 23.739 | 1.662 |
| 4096 | 完整plan | 26.335 | 6.149 |
| 32768 | raw36层 | 181.234 | 5.636 |
| 32768 | shared含normalize | 183.899 | 8.319 |
| 32768 | stable selection | 0.233 | 0.234 |
| 32768 | 完整plan | 184.141 | 8.436 |

4K raw存在早期samples变动，完整plan约6ms仍受逐层Torch reduction/launch成本影响，
不以某一次最小值替代median。32K完整plan约21.8倍改善，不等价于Serving吞吐倍数。
32K常驻score/total/phase共4,325,888 bytes，normalize测到临时2,098,688 bytes，
selection测到1,070,592 bytes；这些峰值按各步骤测量，不相加冒充同时占用。
性能run NVML峰值5,370,806,272 bytes，含合成全层KV和CUDA runtime。

NCU reports `scoring-initial-20261031-0021.ncu-rep`、
`scoring-phase-hoist-20261031-0028.ncu-rep`；命令对`_paged_frequency_scores`的首个
32K launch用`--profile-from-start off --launch-count 1 --set full`。
寄存器255→72/thread，achieved occupancy12.47%→41.57%，profile duration6.70ms→202.66us。
新kernel memory throughput355.62GB/s、compute85.21%；不是饱和HBM带宽的声明。

## 负结果与剩余验证

保留`20261031-000726-normalized-q-calibration`：32K maxlen/3GiB pool容量初始化失败，
成功run改maxlen20480但仍统计16K tokens。保留`20261031-001942-shared-scorer-ncu-initial`：
ncu应用命令前多余`--`导致参数失败；修正后才取得profile。

质量与HTTP Serving另行记录；目前只证明数学、payload、completion与真实reuse正确。
8B normalized stats不自动适用于其他模型。Prefix、Graph、TP、DP仍未由本报告验证。

提交前scheduler/allocator/reclamation定向回归16 passed /199 deselected；全部staged
pre-commit hooks通过。首次hook纠正两处ruff格式与benchmark的accelerator API，未跳过hook。
