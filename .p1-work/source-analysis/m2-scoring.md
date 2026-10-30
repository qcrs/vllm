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
逐层复用 `[Hq,max_len]` FP32 score buffer与长度E的累积向量。

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
归一化与selection仍用Torch；初始实测selection约0.23ms，无依据重写top-k。

## Completion 与 ownership

评分在既有main stream、同一候选的一次reclaim内；其后的compactor、row/E发布与
完成事件顺序继承M1。Scheduler仍是唯一canonical owner；scorer只产KeepPlan，不free。
M2 candidate成本对非SinkRecent使用保守kept-token copy估计，不能假定任意评分集合
具备SinkRecent的连续zero-copy结构。实际性能与质量由各自实验报告给出。
