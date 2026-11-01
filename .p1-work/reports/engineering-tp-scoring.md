# T0/T1：TP评分链测量与有限融合

T1保留：融合逐层normalization/GQA/head reduction后，三个真实Runtime形状的
microbenchmark完整reclaim范围缩短约33–38%。正确性、TP2真实回收/继续生成/
释放复用通过。**Serving没有稳定可辨收益**：首组吞吐+1.185%，反向组−0.021%；
四次pressure均15/24 SLO、TTFT超标，仍为FAIL。未修改Runtime collective。

本轮基于`6dee942bbd`调查、在`3c9ee4117c`之后提交T1；运行时存在dirty源码，
以各run快照/hash为准。scorer的实测与最终源码SHA256均为
`88428bb1a655b8633762c0553e2ea6652aa76973f99f7997e8ec0c58576b5f5d`。
固定校准为`M2/20261031-000856-normalized-q-calibration/frequency.pt`，SHA256为
`13ae79032da6fd8039759a89f4194c91c9818034e2b0c8646e462ba74f63ca9b`。
它仍是TriAttention-inspired shared scoring，不是H2O。

## 调用链与实际变化

`GPUModelRunner.sample_tokens`先`postprocess_sampled`更新logical L，再调用
`KVReclamationWorker.reclaim`。TriAttention路径是`shared_scores`→FP32 score SUM→
TP root `protected_topk`→KeepPlan broadcast→本地compaction/row/E发布→专用
completion collective→AsyncOutput实际等待→Scheduler canonical commit/free。
scorer不拥有Block对象；Scheduler仍是canonical E与ownership的唯一owner。

[`TriAttentionScorer.shared_scores`](../../vllm/v1/worker/gpu/kv_scoring.py)保留
eligible `[prefix:recent]`的Torch population `var_mean`、logical phase与raw paged
scoring。新增`_normalize_reduce_scores`一次完成sqrt/零方差分母、FP32 z-score、
GQA max和本地distinct-head mean；逐层`total.add_(alpha=...)`保留原顺序/权重。
使用`tl.div_rn`及`enable_fp_fusion=False`，掩码支持非2幂head/group和partial tile。
`layer_total`增加`4*max_len`常驻bytes，max_len9216时为36KiB/rank；没有KV scratch、
第二种Runtime backend、host `.item()`或新的同步。统计head仍按完整GQA group归属，
非owner贡献零；stable selection与全部rank完成边界沿原实现。

## T0/T1测量口径

从旧TriAttention trace提取形状和频次，使用真实stride/校准，**KV内容为合成Tensor**。
36层，每rank4个KV heads/16个statistics heads，BF16四维stride
`[16384,256,1024,1]`。输入为
`Engineering/20261031-232434-initial-state/tp-runtime-shapes-corrected.json`：

| E | keep | P | recent_start | logical L | 旧trace次数（含warmup） |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2051 | 1025 | 16 | 2035 | 4100 | 4 |
| 4100 | 2050 | 16 | 4084 | 4100 | 6 |
| 8194 | 4097 | 16 | 8178 | 8194 | 4 |

L来自`reclaim_commit.logical_tokens`；不能使用上一settled值4099/8193，尤其不能
用E代替重复回收后的logical时间。先前未修正L的micro产物保留，不用于下表。
manifest包含旧trace路径、行、request与时间，不把这14例当正式负载频率分布。

物理GPU0,1、主线程CPU20/21；每阶段3次预热、10次CUDA Events，按两个rank中
较大的中位数报告。restore在计时外；保留首次process内reclaim的冷路径与host-submit
样本。Events测量流上的完整范围，可能含host发射间隙；不是纯kernel busy。
normalization测量保留`var_mean`，仅预先备好raw值；raw一列包含phase。
complete_reclaim包含score/plan/copy/row/E/completion，**不含正常模型forward**。
各阶段独立测量，不能相加还原完整链，也不是在线Scheduler miss→retry时延。

单位ms，主表固定原生score AllReduce：

| E | phase T0→T1 | raw+phase T0→T1 | norm/reduce T0→T1 | plan T0→T1 | reclaim T0→T1 |
| ---: | --- | --- | --- | --- | --- |
| 2051 | .049152→.050688 | 1.177088→1.211904 | 3.959808→2.304512 | 6.228480→3.759616 | 6.611456→4.117504 |
| 4100 | .048128→.049152 | 1.148928→1.183232 | 3.975680→2.417664 | 6.279680→3.769344 | 6.766592→4.413952 |
| 8194 | .048128→.048128 | 1.152512→1.198080 | 3.960832→2.348032 | 6.308352→3.922944 | 7.420416→4.962816 |

T0 normalization/reduction约3.96ms，是优先于raw裁剪的成本依据。T0/T1 stable
selection约.185–.253ms、broadcast约.063–.067ms。只有32个保护元素，raw裁剪机会
约.39–1.56%，没有继续实现。归一化仍有`var_mean`和逐层累加，不声称全链单kernel。

benchmark另外只替换FP32 score collective为映射global root的`dist.reduce`，
int32 completion不变；未将该实验分支接入Runtime。比较完整reclaim：

| E | T0 AllReduce / Reduce-root ms | T1 AllReduce / Reduce-root ms |
| ---: | --- | --- |
| 2051 | 6.611456 / 6.529536 | 4.117504 / 4.109824 |
| 4100 | 6.766592 / 6.749184 | 4.413952 / 4.339200 |
| 8194 | 7.420416 / 7.418880 | 4.962816 / 4.982272 |

score collective本身约.069–.077ms，Reduce-root没有稳定完整链收益，停止通信修改。
所有形状在两个rank、两种collective与T0/T1的plan SHA均一致。两次micro的guard
PASS/exit0，项目NVML合计峰值4,148,166,656 bytes。

主结果目录分别为
[T0](../experiments/Engineering-TP-T0/20261031-235838-corrected-logical-runtime-shapes-baseline)和
[T1](../experiments/Engineering-TP-T1/20261101-000312-corrected-logical-fused-normalization-runtime-shapes)。
`invocation.json`保留真实argv、环境、HEAD/diff；`runtime-rank-{0,1}.json`保留全部样本。
最终benchmark仅将设备选择/cache/synchronize改为仓库要求的`torch.accelerator`等价API，
并给旧单卡分支的workspace统计补入`layer_total`。这些收尾改动未重跑GPU；实测harness
仍在原归档，scorer SHA未变。

## 独立正确性与真实TP2生命周期

复用现有suite/reference/helper，新增非2幂GQA group、partial tile、零方差、exact tie
与相邻cutoff检查；测试四问见 [test-design](../notes/test-design.md)。

| 检查 | 实际结果 | 产物目录（Engineering-TP-Check下） |
| --- | --- | --- |
| 单卡独立FP32 complex reference | 7 passed、14 deselected；raw/shared/plan、logical时间、GQA、零方差与ties | `20261031-234629-scoring-reference` |
| 真实TP2 score SUM/plan broadcast | 3 cases×2 ranks PASS；最大score误差约5.364e-6，ties误差0；plan一致 | `20261031-234833-fused-collectives-reference` |
| Qwen3-8B TP2 eager continuation/reuse | 两次真实Tri回收、全36层retained bit-exact、其他请求不变、poison后释放页真实rewrite/reuse；free_final=32 | `20261031-234918-fused-real-continuation-reuse` |

最后一条每rank执行47次FP32 attention reference，真实Worker主线程CPU20/21。
`early_free_observed=true`是旧脚本“提前检查已执行”的字段，检查要求commit前
free增量为0、canonical row不变，**不表示发现提前释放**；本次没有注入慢rank或故障。
未改变completion/Graph地址协议，未重复旧组合Gate。通过范围是这些reference和
实测plan，浮点归约树改变不构成对所有输入score逐bit相同或有损质量保持的证明。

## 同配置TriAttention Serving

Qwen3-8B BF16、TP2、GPU0,1，Worker主线程CPU20/21，FA2/NHD/B16、MRV2、
APC/async off、retention=.5、reserve=0、max_len9216、seqs8。pressure为
long/full、batch1024、.75GiB/rank，FULL_DECODE_ONLY `[1,2,4,8]`，完整负载warmup
后3轮，每轮8请求、49152 prompt tokens、1024 output tokens。参数、prompts/hash、
arrival和每请求结果均在run中；正式trace off，不启用payload诊断同步。

| run | 三轮output tok/s | 合并tok/s | TTFT P99/max(s) | TPOT P99/max(ms) | goodput(req/s) |
| --- | --- | ---: | --- | --- | ---: |
| A baseline | 98.383 / 98.478 / 98.878 | 98.579248 | 3.063144 / 3.065806 | 33.722 / 33.786 | .481344 |
| B fused | 98.652 / 100.228 / 100.380 | 99.747173 | 3.009971 / 3.047417 | 33.509 / 33.585 | .487047 |
| B reverse | 97.777 / 100.111 / 100.693 | 99.510735 | 2.967375 / 2.990393 | 33.268 / 33.284 | .485892 |
| A reverse | 99.413 / 99.578 / 99.603 | 99.531435 | 2.952075 / 2.955464 | 33.699 / 33.709 | .485993 |

四run均15/24 SLO；每轮index3/6/7因TTFT失败，TPOT全部通过。timed均零抢占，
warmup A各2次、B各1次抢占，不能写成全run零抢占。每pair token IDs 24/24一致，
timed prefill工作未显示额外重算。首组+1.184757%、反向−.020797%，运行顺序/波动
足以解释差异；不宣称稳定吞吐改善。fused warmup的JIT日志证明运行过normalize
kernel；正式无逐step评分计数，不能捏造timed调用次数。

no-pressure为short/eager、8GiB/rank、batch2048，同策略初始化、每轮2请求：

| 版本 | 合并tok/s | TTFT P99/max(s) | TPOT P99/max(ms) | SLO / goodput(req/s) |
| --- | ---: | --- | --- | --- |
| A | 62.717246 | .066094 / .066107 | 29.3674 / 29.3676 | 6/6 / .489978 |
| B | 64.233290 | .061399 / .061610 | 28.8328 / 28.8407 | 6/6 / .501823 |

该负载不触发scorer，+2.417268%只支持本配置≤5%无退化预算，不能归因为评分融合。
各run guard PASS/exit0；项目合计峰值pressure19,935,526,912 bytes、short
35,454,451,712 bytes。小样本P99仅描述本轮，不是总体尾延迟保证。

完整目录为 [Engineering-TP-Formal](../experiments/Engineering-TP-Formal)：
`20261031-234418-triattention-batch1024-baseline`、
`20261031-235035-triattention-batch1024-fused`、
`20261031-235340-triattention-batch1024-fused-reverse`、
`20261031-235613-triattention-batch1024-baseline-reverse`、
`20261031-235739-triattention-short-baseline`、
`20261101-000213-triattention-short-fused`。

首A缺少dirty scorer即时快照，`baseline-source-recovery/kv_scoring.py`为**事后恢复**，
provenance单独记录；其SHA与T0归档的baseline一致，为
`061653e08fedba9784e59d9e84318605077391ab6b6304b3bef5214b7d4ed45a`。
恢复文件不能升级成运行当时完整快照。反向A/B及T0/T1保留实际scorer快照，
因此收益判断保留这一首组证据限制，也不靠首组正值下结论。

## 复算与停止边界

[本次收尾归档](../experiments/Engineering/20261101-0832-closeout)的
`analysis-manifest.json`保留三条实际CPU复算argv/exit0、源码与结果hash；
`analysis/tp-summary.json`保留每轮、每请求聚合、affinity与最慢rank全部阶段。
执行入口为`summarize_tp_serving.py --engineering --baseline-micro --fused-micro
--output`，各参数的完整路径在manifest中。GPU命令见各run的`invocation.json`，
协议见 [tp-protocol](../notes/tp-protocol.md)。

适用pre-commit与手动mypy-3.12通过；初次E501和设备API hook失败日志保留。
未新装依赖或重复GPU矩阵。原SinkRecent17/24 FAIL不执行score SUM，不能由本轮
融合解释或宣称修复。旧profile PARTIAL及85条冻结质量结论保留；token一致与
搬移正确都不是质量保证。没有剩余主要成本证据，T1止于本项融合。
