# Graph 正确性验证

## 背景、问题和假设

本阶段解决 FULL capture 对 attention lengths 的持久引用：只在 replay 前替换
Python metadata，不能让已捕获 kernel 改读另一块 tensor。logical L 与 physical E
必须保持分离；回收后的下一 forward 使用 retained row/E 和连续 logical RoPE。
模型图外执行回收，新布局 GPU 写完成后才 canonical free，released IDs 真实复用。

**正确性限定 PASS**：本报告的七条 Qwen3-8B 轨迹均通过，覆盖 FULL/原生
breakable PIECEWISE、whole/token、Random/Tri、Prefix+FULL；Serving/profile
另见 `graph-serving.md`。不把诊断时间用作稳态性能，也不由此推断 TP/DP 支持。

## 环境和实现

基线 vLLM v0.26.0 `568afb3a13806beb53bb2e6bd518269357b237c0`，
实验在 branch `p1/v8-physical-kv-reclamation`、HEAD `386bd624f4` 加本阶段 diff。
run_guard 的 invocation/source 保存每条实际源码及 hash，提交后可对应复核。
使用既有 Python3.10/PyTorch2.11.0+cu129/Triton3.6.0 环境；Qwen3-8B BF16、
A100 80GB物理GPU1、MRV2、FA2/NHD/B16、一full-attention group、TP1/DP1。
单在途、async/spec/cascade/connector/reserve-full-ISL off；APC仅两条组合开启。
max_model_len512、max_num_batched_tokens256、max_num_seqs4、33页含null页。
模型和数据全离线；所有GPU实验经run_guard串行监测，总显存45 GiB硬上限，
44 GiB停止阈值，未操作其他GPU任务。

`ModelCudaGraphManager.capture` 可绑定 `reclamation.kv_seq_lens`；capture dummy
和后续dummy execution写同一physical buffer，清除padding/tail。logical input
positions/lengths仍供原生sampler、chunked-prefill等消费者使用。resident/input
block tables、slots和query starts继续复用原生allocation。不为E/Block ID变化
增加graph key或recapture逻辑。ownership、持久/逐步状态与发布点详见
`../source-analysis/graph-runtime.md`；协议在运行前写入
`../notes/graph-serving-protocol.md`。

## 测试设计和执行

A/B/C真实请求，A初始256 tokens，B128、C192；Prefix组合共享前64 tokens。
step16与32强制对A回收，奇数step反转scheduled request顺序，capture buckets
为`[2,4]`。FULL decode实际replay，prefill沿native NONE。保留全36层source
snapshot，以实际current-member KeepPlan逐bit检验retained有效KV，其他请求
和保护prefix不变。先检查GPU completion event.query，再读payload；canonical
free后才poison removed pages，C真实forward的append slots须全层finite且写入
bit pattern变化，A随后继续decode。ptr/shape/stride/dtype、graph对象身份、
capture计数每步核对，旧active tail不可被metadata读取。

FULL每次回收后的forward另重跑同输入、同retained KV、同KeepPlan的eager参考，
采用与capture相同的合法scalar上界512；hidden及全36层append KV零容差，logit
argmax相同。eager写完恢复graph的append slots，避免参考路径改变后续轨迹。
第一层attention另以CPU FP32 dense softmax独立计算，atol/rtol均0.016。
PIECEWISE的动态attention同样接受该FP32检查。

```bash
source .p1-work/scripts/env.sh
.venv/bin/python .p1-work/scripts/run_guard.py --stage Graph \
  --name full-token-matched-launch -- \
  @python .p1-work/scripts/graph_reclamation.py --mode full --path token
```

其他轨迹仅改`--path whole`、`--mode piecewise`、`--prefix`或`--policy`。
Tri校准文件沿用M2的
`20261031-000856-normalized-q-calibration/frequency.pt`，通过`P1_STATS`传入。
诊断逐步同步/payload hook只用于正确性，不进入HTTP测量。

## OBSERVED 结果

以下目录均位于 `/data_ssd/qcrs/experiments/p1-v8/Graph/`，guard exit0/PASS。
每条有61次实际graph dispatch、3次NONE、42个padded batch、47次FP32 attention
核对；六条FULL各有46次matched eager对照。所有capture增量均0，全部请求结束后
free pool恢复32页。最大项目NVML峰值17,179,869,184 bytes（16 GiB）。

| 目录 | 配置 | 两次E→K / 释放页数 | C实际改写released IDs数 |
| --- | --- | --- | --- |
| `20261031-095114-full-token-matched-launch` | FULL token | 272→127/9；143→62/5 | 10 |
| `20261031-095154-full-whole` | FULL whole | 272→128/9；144→64/5 | 9 |
| `20261031-095220-piecewise-token` | PIECEWISE token | 272→127/9；143→62/5 | 10 |
| `20261031-095243-full-prefix-token` | Prefix+FULL token，P64 | 272→159/7；175→126/3 | 8 |
| `20261031-095307-full-prefix-whole` | Prefix+FULL whole，P64 | 272→160/7；176→128/3 | 7 |
| `20261031-095440-full-token-random` | FULL Random token | 272→127/9；143→62/5 | 10 |
| `20261031-095505-full-token-triattention` | FULL Tri token | 272→127/9；143→62/5 | 10 |

第二次token K刻意少2，避免recent起点对齐页边界而变成whole路径；whole两次均
实际零拷贝。C不必取得每一页removed ID，但取得的页确实在36层被新KV覆盖。
Prefix A/B/C均由canonical APC取得真实64-token hit，prefix始终只读。

## 早期失败和数值分析

`20261031-094257-full-token-replay-reuse` 因诊断给只读ExecuteModelState字段
赋值失败，删除该无效赋值后继续；未修改Runtime以迁就脚本。
`20261031-094355-full-token-replay-reuse-fixed` 在动态E上界eager参考下，
2/4096 hidden元素超过预定atol0.0625/rtol0.03，未通过。

独立 `20261031-094927-full-token-numerics-diagnostic` 保持其他条件不变，
仅将参考max_seq_len从实际E改为capture512。46次回收后forward在capture-bound
参考下hidden/logits/全部层append KV逐bit相同。动态上界15步超原容差，最大
hidden abs4.75、KV abs2.78125，argmax变化0。raw差异tensor/JSON保持原样。
此结果定位到scalar改变引起数值路径变化，未定位FA2内部具体reduction分支。

正式主对照因此固定相同launch上界，并收紧为零容差，没有放宽原容差。
Runtime metadata仍按actual E构造；FULL capture的scalar沿原生max_model_len
合法上界，实际seqused_k消费持久physical lengths。该处理符合设计§15.3，
不声明graph与任意不同launch拓扑的eager逐bit相同。

## 回归和证明边界

`20261031-095651-regression-cpu` 运行以下命令：17 passed、317 deselected，
guard峰值515,899,392 bytes。目录名虽含cpu，frequency tests实际执行GPU。

```bash
.venv/bin/python -m pytest tests/v1/core/test_scheduler.py \
  tests/v1/core/test_kv_cache_utils.py \
  tests/v1/core/test_single_type_kv_cache_manager.py \
  tests/v1/core/test_prefix_caching.py tests/v1/worker/test_kv_compaction.py \
  -k 'reclamation or reclaim or physical or frequency or graph_attention' -q
```

CPU graph contract检验capture绑定physical地址、logical输入独立，以及dummy
刷新/清尾；复用现有compaction suite，四问设计见`../notes/test-design.md`。
真实图验证限定FA2、已捕获buckets与单在途。FA3/4 scheduler metadata、任意bucket、
多在途/spec/async/connector、TP及DP仍需各自Gate。诊断reclaim时间含JIT、同步和
观测开销，例如Prefix第二次token 163ms，不能称为稳态copy成本。
Graph压力下Serving、短上下文开销和NSYS是整体阶段结论的独立证据。

## Evidence Index 和下一步

每条目录包含`invocation.json`/`source/`、`identity.json`、`steps.jsonl`、
`comparisons.jsonl`（FULL）、`evidence.json`、`kv_trace.jsonl`、`nvml.jsonl`、
`stdout.log`和`result.json`。失败与诊断目录不覆盖、不冒充PASS。
17组正式HTTP Serving已完成，native/noop共84个输出逐token相同；8 GiB noop开销
0.63%，1.5 GiB三有损策略各24/24满足预定SLO，详见`graph-serving.md`。
NSYS完整时间窗口已采集，但274个graph API缺1个GPU span，保留PARTIAL限制。
Graph正确性/限定配置Serving PASS；TP、DP及未覆盖组合不继承这一结论。
最终pre-commit后提交Graph，继续TP。
连续阶段授权来自prompt1与active Goal，无需逐阶段确认。
