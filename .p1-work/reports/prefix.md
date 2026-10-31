# Prefix Gate：完整页 APC 与私有 suffix 回收

Prefix 正确性 Gate 为 **PASS**：真实 Qwen3-8B Engine 完成共享保护、全层搬移、
真实 APC hit、取消私有 suffix 发布、完成后释放/重用及生命周期验证。
本结论只覆盖 Eager、TP1/DP1、单在途、单 full-attention group、
`hash_block_size == block_size == 16`；不将诊断轨迹判为 Serving 性能通过。
新增HTTP对照见 [prefix-serving.md](prefix-serving.md)：8 GiB noop开销预算PASS，
6 GiB SinkRecent延迟/Goodput预算PASS；3/1.5 GiB仍因TTFT超预算而性能FAIL。
后续Graph/TP/DP及组合证据见对应报告与`../plans/status.md`；本报告保留Prefix
阶段的配置范围，不从其他Gate推断全部组合通过。

## 实现与关键修复

调用链和状态语义见 [prefix-runtime.md](../source-analysis/prefix-runtime.md)。
Scheduler 从 canonical Block 引用推导最后共享页末端 P，按私有 suffix 计算 K/R。
Worker mutation 前经原生 `evict_blocks` 撤销 P 后的 primary hash 和全部 aliases，
此时不改 ownership、refcount、free queue 或 cached count。完成后保留 protected row
及原 Block 对象，clamp `num_cached_block` 至 P/B，再释放 removed 独占页。
公共 `cache_blocks` 与 allocation 内 publication 均用 settled L/E 派生 lossy guard；
Request 的 logical token/hash chain 保持原语义。

验证发现真实 APC admission 缺陷：Scheduler 已采纳 hit length，E 却仍为0；
Worker 已按 hit length 初始化 E。新增真实 hit 单测复现后，在 allocation 成功边界
将 Scheduler E 同步至 `num_computed_tokens`。不在 allocation 失败前提前采纳命中，
也不将已有 lossy 状态直接补成 L。

Worker 沿用 M1/M2 已有的临时 P、shared KeepPlan 和 Stable CTA compactor，
本次没有增加 kernel、COW 或持久 prefix map。

## 环境与复现

模型 `/data/models/Qwen3-8B`、BF16、物理 GPU1、MRV2、FLASH_ATTN；
async、cascade attention、connector、speculation 关闭。max_model_len=512，
max_num_batched_tokens=256，KV budget=256 MiB，pool=49页（48可用）；
真实压力轨迹 pool=33页（32可用）。A/B prompt 为256/160 tokens，共享前64 tokens；
C 通常重复 A prompt，pressure 中改为相同64-token prefix 加不同192-token suffix。
KV events 使用 null publisher，不向外部发送事件。

所有 GPU 运行由现有 guard 串行执行，45 GiB 为项目硬上限、44 GiB 为 guard 终止阈值，
未触发终止。峰值最高为17,146,314,752 bytes（约15.97 GiB）；未操作其他用户任务。

```bash
source .p1-work/scripts/env.sh
P1_PREFIX_MODE=token P1_LIFECYCLE=repeat \
  .venv/bin/python .p1-work/scripts/run_guard.py \
  --stage Prefix --name token-repeat -- \
  @python .p1-work/scripts/prefix_reclamation.py
```

其他 mode 为 `native|identity|whole|pressure|no_gain|gap`；
lifecycle 为 `repeat|eos|abort|preempt`。Random/TriAttention 采用 token/repeat，
设置 `P1_POLICY=random|triattention`；TriAttention 另指定
`P1_STATS=/data_ssd/qcrs/experiments/p1-v8/M2/20261031-000856-normalized-q-calibration/frequency.pt`。
各目录保留 invocation、源码快照、NVML、stdout、result、identity、evidence、KV trace。
实验基于 `5f9f96d9d0` 加本阶段修改，实际源码以各次快照为准；大产物不提交 Git。

## GPU 结果

下列目录统一位于 `/data_ssd/qcrs/experiments/p1-v8/Prefix/`。
每项均为 PASS；E→K 指 forward 完成后的有效长度→retained 长度，R 为真实释放页数，
不是取消 hash 的页数。表中 reused 为 C 已实际写入不同内容的 released ID。

| 目录 | P；E→K；R | reused IDs | 后续 FP32 Attention checks |
| --- | --- | --- | --- |
| `20261031-021924-whole-repeat` | 64；A272→160/R7、A176→128/R3、B192→128/R4 | 5,6,8,9 | 70 |
| `20261031-022410-token-repeat-final` | 64；A273→159/R8、B193→127/R5、A177→111/R5 | 12–17 | 68 |
| `20261031-022205-pressure` | 64；A262→163/R6 | 12–17 | 53 |
| `20261031-022436-token-eos-final` | 64；A273→159/R8 | 11,12,13,15,16,17,25 | 0，A在该步结束 |
| `20261031-022458-token-abort-final` | 64；A273→159/R8，ACK前移除请求 | 12–17 | 0，A已abort |
| `20261031-022521-token-preempt-final` | 64；A273→159/R8，随后原生preempt/recompute | 12–17 | 10 |
| `20261031-022544-shared-gap-shrinking-p` | A的P144→64；A273→191/R6、B193→127/R5、A209→111/R7 | 13–17 | 68 |
| `20261031-022625-token-random-repeat` | 64；A273→159/R8、B193→127/R5、A177→111/R5 | 12–17 | 68 |
| `20261031-022651-token-triattention-repeat` | 64；A273→159/R8、B193→127/R5、A177→111/R5 | 12–17 | 68 |

全36层的 valid retained payload 与源 snapshot 按实际 KeepPlan bit-exact；
每次回收还逐层比较 protected bytes 和其他存活请求的全部 valid KV。
C 重写释放页后，A/B retained bytes 继续通过；后续第一层真实 Q 与 retained K/V
的独立 FP32 Attention 以 atol=rtol=1.6e-2 比较通过。
Random/TriAttention oracle 使用捕获的实际 KeepPlan 验证搬移与保护；
它们的 score 公式独立参考验证属于已有 M2，不把本实验称为新的公式验证。

实验在进入 canonical update、读取验证 payload 之前先检查 compaction 后的
CUDA Event 已完成；Runtime 完成链仍由 MRV2 output copy_event 提供。
无 EOS/abort 的 commit 精确增加 R 个 free 页；共享页 refcount 始终等于 canonical owners，
最后引用消失前不进入 free queue，结束时 free 恢复至全部可用页。
EOS 是在真实 Engine output seam 受控注入 EOS；abort 位于 GPU 完成后、ACK前；
二者验证生命周期而非自然生成 EOS 的概率。

gap 场景明确为实验构造：经原生 `touch` 和 canonical fixture holder 持有 row index8，
形成 shared 0–3/8、exclusive gap 4–7，P=144。释放 holder 且 C 结束后，下一次
A 回收的 P 缩回64；原来只读区域先 deindex 后才可修改。不宣称新增 partial-hit COW。

真实压力中 C 需要12页、available=8，shortfall=4；选择 A，forward后 E262→163，
释放6页使 free8→14，C 下一 step 重新 allocation 并写入 IDs12–17。
这条运行通过真实 miss 分支；whole/token/gap 的指定触发用于隔离边界和搬移验证。

## APC、重算与成本边界

`20261031-022230-native` 与 `20261031-022252-identity` 的 A/B/C 全部 token IDs
及 finish states 完全相同（输出长度60/60/4）；identity 没有 deindex 或回收。
机械比较记录为 `native-identity-comparison.json`，全部运行汇总为 `summary.json`。

| 场景 | B实际hit | C实际hit | C prompt需计算tokens | 回收 |
| --- | --- | --- | --- | --- |
| native / retention=1 | 64 | 240 | 16 | 无 |
| token/whole重复回收 | 64 | 64 | 192 | 有 |
| shared gap首次P144 | 64 | 144 | 112 | 有 |
| P过大、private suffix不足 | 240 | 240 | 16 | 明确跳过 |

hit 均检查实际 Block IDs，不能只据计数推断。A/B 初始 prompt 共享比例为25%/40%；
normal 首次回收 P/E_src=64/273≈23.44%，gap 为144/273≈52.75%。
native C 的240-token hit 是完整页且保留最后 token 计算的正常 APC 行为。
有损后 C 重复 A prompt 多计算176 tokens，是 suffix 停止发布的真实代价。
preempt 场景先将 L/E 清零，随后真实 hit256并重新计算28 tokens，恢复 E=L=284、
cached count=17；没有以修改 E 冒充重新计算。

`20261031-022315-no_gain` 中 P240/E257，private_min_keep32，预算 K>=E，
候选明确为空，无搬移、无取消发布，C仍hit240。这是93.75% prompt共享时的无收益边界。
R=0/P=E、P=0及不对齐 K 的补充边界由 CPU suites/已有 M1 证据覆盖。

deindex 用 `perf_counter_ns` 包围原生 evict 调用；包含 hash bookkeeping 和
KV events 构建，不包含 snapshot。whole 的三次为51.073/9.927/41.870 µs；
final SinkRecent token 为47.578/56.082/8.577 µs；pressure 为48.262 µs；
gap 为32.425/51.586/47.188 µs；Random 为47.230/52.629/8.119 µs；
TriAttention 为46.767/56.793/7.586 µs。单条短 trace，不能当稳定 P99。

CUDA Event诊断区间另保留原始值：whole约0.63–0.92 ms，final SinkRecent token约
3.16–4.98 ms，Random约3.65–7.73 ms，TriAttention约9.63–73.97 ms。
这些值含实验 hook、GPU→CPU读取、计划获取及冷启动，部分早期/新P运行可达约
170–218 ms；**不能作为纯kernel latency、无观测开销的回收性能或 Serving速度**。
本阶段复用原 kernel，未据此新增优化或推翻 M1/M2 的专门 profiling。
真实页容量由36×8×16×256×BF16计算，为每页2.25 MiB；R6/8分别归还13.5/18 MiB
给 allocator，预分配 Pool 的 NVML 占用不会因此同步下降。
Prefix on的12组HTTP Serving和两条诊断已另记于
[prefix-serving.md](prefix-serving.md)，包含固定SLO、APC/重算口径和负收益。
本节成本仍仅为正确性诊断；有损质量没有新增验收，不据正确性或单配置性能扩大默认启用范围。

## CPU、检查与保留的失败

```bash
source .p1-work/scripts/env.sh
.venv/bin/python -m pytest tests/v1/core/test_scheduler.py \
  tests/v1/core/test_prefix_caching.py tests/v1/core/test_kv_cache_utils.py \
  -k 'reclamation or pressure_ or allocator_trace' -q
# 23 passed, 288 deselected
.venv/bin/python -m pytest tests/v1/core/test_prefix_caching.py -q
# 92 passed
```

最终输出和 JUnit 保存为 Prefix 根目录下的 `cpu-targeted.*` / `cpu-prefix.*`。
包括真实 APC admission E 初始化、canonical pool/lookup、全部 suffix aliases及完整
BlockRemoved身份、lossy公共发布入口、正常in-flight不误判、partial tail、真实reuse与最后ref。
alias 由原生 `_insert_block_hash` 构造，用于验证全 key 撤销，不据此声称支持fine-grained APC。
源码、测试、实验脚本与文档的 pre-commit（含 mypy3.10）和 `git diff --check`
均通过；hook输出保留在 Prefix 根目录的 `pre-commit.log`。

扩大运行完整 prefix+KV utils suite 的结果为 **156 passed, 14 failed**：14项均依赖
未缓存的上游默认 Qwen 模型配置，offline 环境加载失败，部分错误来自代理缺少socksio。
没有下载模型/数据、安装依赖或用0.6B模型替代实验；不声称完整 utils suite通过。

`20261031-022009-pressure` 为保留的 **FAIL**：C原为A的同prompt，hit240，只需1页，
没有触发预期miss，最终“发生回收”断言失败。改为不同私有suffix后，
`20261031-022205-pressure` 才是通过的真实压力证据；没有为此改Runtime策略。
早期 `20261031-021805-token-repeat` 也保留，后由 final版本补齐其他活请求全部payload比较。

Prefix之后的阶段已按用户恢复的连续授权执行；当前总状态见`../plans/status.md`。
