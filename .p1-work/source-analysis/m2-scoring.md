# M2 Shared Scoring 源码记录

## 数学与输入

`KVReclamationWorker.reclaim` 在 postprocess 后读取 GPU
`req_states.num_computed_tokens[idx]`，因此 phase 使用 logical 下一位置 L_after，
重复回收不以 E 代替时间。`kv_scoring.py` 直接按 row 和真实四维 stride 读取 paged
K_rot 的前 D channels；Qwen half-style 配对为前/后 D/2，不需要原 key position map。

每层/统计 query head 计算 `mu_q * conj(K_rot)` 的未来17个offset相位平均，
加 `(E|q|-|Eq|)*|K_rot|` correction。只在 eligible member 区间算 token-axis
population variance/mean，再 z-score；GQA group max → distinct KV heads mean →
all layers mean。保护 sink/recent 后 stable argsort，最后 member index 升序。
Random 用固定seed GPU generator生成同预算负对照，也走相同保护/排序/compaction。
SinkRecent 保留原整页可表示快路径；评分计划不 D2H 探测 zero-copy。

`TriAttentionScorer.from_file` 在初始化核对模型 config SHA、层数、所有 Q/KV heads、
head_dim、RoPE配置/half style、Q归一化来源和finite stats；不匹配直接报错。
非标准/scaled RoPE没有适配，不以字段缺省或Layer取模恢复。stats在GPU常驻，
逐层复用 `[Hq,max_len]` FP32 score buffer、长度max_len的`total`与`layer_total`。
TP按完整GQA group分配distinct KV-head owner，本地贡献按全局head/layer数归一；
非owner返回零贡献，也参与同一次score SUM。

## 校准与独立验证

`.p1-work/scripts/calibrate_frequency.py` 通过实际Qwen3 MRV2模型的36个`q_norm`
forward hook，统计pre-RoPE normalized Q的real/imag/abs。使用本地WikiText valid
前16,384 raw tokens，无chat template；每层实际计数相同。metadata记录数据、tokens、
模型config SHA。校准只收集Q统计，不复用待评估的BabiLong/SCBench任务。

`tests/v1/worker/test_kv_compaction.py` 的CPU参考独立scalar解码paged KV，再用
FP32 complex运算逐offset计算；验证raw/shared/plan、logical时间、query/KV粒度
stats、GQA、padding/非连续row、保护域、零方差和stable tie。参考不调用production
score/address helper，也不进入Runtime。

## Profiling 驱动的改动

初始kernel每个head/token tile重复17组future phase sin/cos，36层也重复。
NCU观察255 registers/thread、12.47% achieved occupancy，单层profile 6.70ms。
`_future_query_phase` 现在每次shared plan只从GPU L计算一次`[2,F]`系数；
`_paged_frequency_scores` 加载512-byte系数后只算K相关项。没有历史trig cache、
host `.item()`、第二种scoring backend或新的KV payload scratch。

同形状NCU降到72 registers/thread、41.57% occupancy、202.66us，memory throughput
355.62GB/s。profile锁频/replay结果仅比较profile资源，不代替非profile CUDA events。
这项phase优化是既有M2结果，不计为本轮T1收益。

2026-11-01的T1保留eligible域的Torch `var_mean(correction=0)`，新增
`_normalize_reduce_scores`融合sqrt/零方差分母、FP32 z-score、GQA MAX和本地
distinct-head mean；每层写入`layer_total`后，仍按原顺序执行`total.add_(alpha=...)`。
kernel掩码覆盖非2幂head/group及partial tile，使用`tl.div_rn`且关闭FP fusion。
新增常驻空间为`4*max_len` bytes；max_len9216时36KiB/rank，没有KV scratch。
selection仍用Torch stable argsort，Runtime score SUM/plan broadcast/completion未变。

三个真实Runtime shape、真实stride/校准与合成KV的micro完整reclaim缩短约33–38%；
同配置Serving及反向启动未显示稳定收益。完整测量、独立reference、真实TP2
continuation/free/reuse与证据限制见 [工程TP报告](../reports/engineering-tp-scoring.md)。
这是有限形状的数值/plan验证，不保证任意输入逐bit相同或有损质量保持。

## Completion 与 ownership

评分在既有main stream、同一候选的一次reclaim内；其后的compactor、row/E发布与
完成事件顺序继承M1。Scheduler仍是唯一canonical owner；scorer只产KeepPlan，不free。
M2 candidate成本对非SinkRecent使用保守kept-token copy估计，不能假定任意评分集合
具备SinkRecent的连续zero-copy结构。实际性能与质量由各自实验报告给出。

## 2026-11-01 当前分组实现与长度特化修正

上述 `layer_total` 是 T1 历史实现。当前 TP2a 在 normalization 内按原 FP32 顺序
alpha-add；TP2b 用四层组复用 raw/variance/mean workspace，36层仍按9组依次
累计；详细数值和历史 micro 见 [下一阶段TP](../reports/engineering-next-tp.md)。

`_grouped_raw` / `_grouped_normalize` 的 E 是 mask bound，未声明 constexpr，但
Triton 默认仍按标量整除性分派版本。一次完整预热只触发非16对齐 E，timed
E4160 则出现9个 raw（FIRST 为0..32）及1个 normalize 的新特化。成对缓存 TTIR
除 `%E` 的 `tt.divisibility=16` 属性外逐字一致；10个文件的生成时间落在由同机
wall/monotonic offset 估算的3.46秒 reclaim 窗口内，不将整个窗口当纯编译时间。
本轮只将这两个入口声明为 `do_not_specialize=["E"]`。
数据指针、stride、head/group/layer 常量与既有 FP32 运算不变，不枚举所有长度
预热，也不取消指针对齐或扩大 Runtime 支持域。

行为验证扩展原独立 complex reference 的 E95/96；benchmark 增加同一 scorer 的
E4159→4160→4161→4159 进程 cache-miss 审计。cache miss 可能从既有磁盘缓存
加载，不能一律称重新编译或作为冷编译提速百分比。真实 Serving 结果与 host
时间线的非可加口径见 [本轮TP报告](../reports/engineering-refinement-tp.md)。
