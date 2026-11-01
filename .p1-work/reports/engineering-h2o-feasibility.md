# H0：H2O统计来源可行性审查与停止判断

H0完成源码来源审查，GPU probe及H1均NOT_RUN。本地FA2当前接口没有per-key
attention importance；直接搬官方real-drop实现会物化prefill Q×K。能够避开该矩阵的
attention内归约仍需新的统计方案与成本证据，本轮没有得到可控实现依据，故在probe前
停止。**这不是已测性能FAIL，也不证明H2O无法实现**；代码幅度本身不是停止理由。

## 绑定的原始资料

[H2O论文](https://arxiv.org/abs/2306.14048)和
[官方real-drop源码](https://github.com/FMInference/H2O/blob/ac75c2a8a9e76832b2a4139b9363373b56336bfb/h2o_hf/utils_real_drop/modify_llama.py)
为语义依据，源码绑定commit`ac75c2a8a9e76832b2a4139b9363373b56336bfb`。
下载源码SHA256为`c20cfc046c89db86e21430c1efeb4a9c4a99328037d3cd66887399dba83c246d`。
[源码审查归档](../experiments/Engineering-H2O/20261101-source-feasibility)保存文件、
本地attention/runner源码、Qwen3 config、manifest/hash及复核命令；不依赖可变化的main。

## 官方实现到底累计什么

`H2OKVCache_LayerWise`（第242行）持有每layer/head的`hh_score`。
`__call__`先更新分数（260行）；271–285行按head选择heavy+recent，并对KV和score
一起重排。`_update_hh_score`（317–326行）在query轴累计实际attention probability，
然后将历史score加回旧key；prefill的多个query也参与初始化。394/616行的
`_clean_cache`在395/617行调用`H2OKVCache_LayerWise._clean_scores`清理统计，
不能只实现decode top-k而忽略重入和释放。

attention主体487–509行显式计算QK，做FP32 softmax、转回query dtype，
`detach().clone()`后交给cache。它依赖完整probability，不是只取最后一条Q，
也不是从LSE或离线Q统计恢复。官方per-head保留集合与本项目全层/head共享
KeepPlan不同；若聚合为一维token score，应称H2O-inspired shared retention。

## 本地FA2提供的张量

当前Qwen3配置为36层、32 query/8 KV heads、head_dim128、BF16。真实调用在
[`FlashAttentionImpl.forward`](../../vllm/v1/attention/backends/flash_attn.py)附近的
`flash_attn_varlen_func`，传入paged KV、q和out，仅消费output。
[`flash_attn_interface.py`](../../vllm/vllm_flash_attn/flash_attn_interface.py)第315行的
FA2 op只返回out/softmax_lse；第264行声明LSE为`[nheads,total_q_seqlen]`，417行
只返回out或out+LSE。其测试用probability选项的文档也不构成此FA2调用可获得
正确per-key统计的契约。

LSE是每query的softmax归一化因子的对数，缺少key轴；out是对V的加权和。这些输出不足以反演
各历史key的累计重要性。历史Q在现有Runtime中未作为请求长期状态保留，不能在
reclaim时仅凭KV+最后Q重建完整历史。TriAttention的校准频域估计也不能替代它。

## 统计方案评估与未测范围

| 方案 | 判断与边界 |
| --- | --- |
| 原样替换为官方attention | N×N矩阵违反本轮约束；以32头、N8192计，单层完整BF16 probability张量理论为4GiB，FP32为8GiB，还未计softmax/clone等；这是尺寸推导，不是实测峰值 |
| reclaim时重做最后QK | 缺prefill和历史query累计，语义不成立；不自动换成近似策略 |
| 每步额外重做完整QK | 重复attention计算，没有收益依据，不以此强行完成接入 |
| attention内累计列重要性 | 潜在路线；需每query最终softmax归一后跨query tiles/GQA累加，处理paged current members和实际layer/head聚合；不能在online softmax未定分母时直接把tile局部和作为最终概率 |

最后一条需要解决prefill初始化与decode更新、临时归约/写入竞争、额外launch/流量及
独立softmax reference。现在没有该实现或误差/成本数据，无法给出可接受的probe方案。
最终一维FP32 score虽然仅4E bytes，不能把它当包含中间状态的总开销。没有通过运行
官方物化矩阵来制造负结果，也没有以合成分数代替真实来源。

H0的GPU成本/显存/reference为NOT_RUN，H1的KeepPlan同步重排、whole-block
zero-copy、preempt/recompute、EOS/abort/free/reuse均未实现/未验证。TP/Graph/Prefix
支持和默认策略未扩张。若未来统计方案成立，先限定TP1/eager/APC off，在相同
Qwen3-8B上验证真实importance来源与成本，再决定接入。冻结85条质量样本不变，
本轮没有H2O质量或性能优越性结论。
