# TP2 shared plan 与全组完成

## 范围和实现

Qwen3-8B BF16、A100物理GPU0,1，TP2/DP1/PP1、原生MultiprocExecutor，
MRV2/FLASH_ATTN/NHD/B16、一个full-attention group、单在途。模型/环境不升级。
源HEAD为`d5f1029450`加本阶段diff，各run的invocation/source保存实际状态。
11条正确性轨迹、独立NCCL评分与最终回归均PASS。12组正式Serving完整采集，
short/eager noop开销及压力SLO为性能FAIL；详见`tp-serving.md`，不将正确性PASS
扩大为全部性能预算通过。

一个Scheduler/KVCacheManager/BlockPool持有canonical row，两个Worker只更新
本地payload和physical E。TP2不新增allocator或复制free操作。配置只开放TP1/2，
其余既有PP/DCP/PCP、async/spec/connector等限制保持。

Tri在每rank按连续KV-head shard加载完整GQA组stats，先query-head z-score、
再GQA MAX、distinct-KV/layer等权。Hkv<TP时每复制组唯一owner读取完整组的
query-head统计，其他rank贡献0；不能只读取该rank的Q shard。一次FP32[E] SUM
后rank0执行stable top-K/member升序排序，再broadcast int32[K]。Random也只在
rank0生成计划；SinkRecent整数计划逐rank重建并在诊断中逐元素核对。

本地payload/row/E最后写之后record local event，再在同一调用stream执行
标量int32 completion SUM。ProcessGroup建立调用stream与NCCL stream依赖；
AsyncOutput copy event或q=0 maintenance event覆盖collective，host等待该事件后
才向Scheduler成功ACK。非output rank异常必须逃出RPC loop，由原生monitor
停止TP组；不能只依靠unique_reply_rank响应，也不能将异常当普通fallback。
feature关闭时保持原生RPC异常处理。源码链见`../source-analysis/tp-runtime.md`。

## 真实8B轨迹

协议在`../notes/tp-protocol.md`。33blocks含null、每rank128MiB KV backing，
max_model_len512，A/B/C prompt256/128/192，输出64/40/20；step16/32两次回收，
step17加入C，奇数step反转batch request顺序。全36层每rank4个local KV heads
独立快照，retained bit-exact、其他请求不变；检查logical positions/physical slots、
resident有效row/num_blocks、padding。canonical释放后才poison removed页，C必须
实际取得并在全层覆盖，A继续decode，最终free恢复32。

以下目录均在`/data_ssd/qcrs/experiments/p1-v8/TP/`，guard PASS、exit0。

| 目录 | 配置 | 每rank有效证据 |
| --- | --- | --- |
| `20261031-112031-eager-whole` | eager整页零拷贝 | 64 NONE、47 FP32 attention、12个released ID实际重写 |
| `20261031-112059-eager-token-delay` | eager token、非output rank延迟3s | 相同payload/reuse检查；先完成rank时free和canonical不变 |
| `20261031-112131-eager-token-fail` | 非output rank写前抛错 | canonical commits0、free不变、executor failed、两worker退出 |
| `20261031-112203-eager-random` | Random两次token | 同一plan/row/E、全层bit-exact、12页重写 |
| `20261031-112231-eager-tri` | Tri两次token | 同一plan/row/E、全层bit-exact、12页重写 |
| `20261031-112259-prefix-eager-token` | Prefix+TP，P64 | A/B/C真实64-token APC hit、共享prefix不变、10页重写 |
| `20261031-112327-full-whole` | Graph+TP整页 | 61 FULL/3 NONE、42 padded、46 matched-eager、12页重写 |
| `20261031-112355-full-token-delay` | Graph+TP token、慢rank | 同上；local completion不能提前free |
| `20261031-112427-prefix-full-token` | Prefix+Graph+TP token smoke | P64、61 FULL、46 matched-eager、10页重写 |
| `20261031-121433-full-token-q0-delay` | q=0与非output rank延迟 | 62 FULL/3 NONE、47 matched-eager、12页重写；free等待全组完成 |
| `20261031-121508-full-token-maintenance-delay` | maintenance与非output rank延迟 | 全层retained/reuse及maintenance ACK等待全组完成 |

FULL在相同capture合法scalar上界512、相同输入/retained布局下，与eager的hidden
和全层append KV零容差，logit argmax相同；graph对象/地址/capture计数不变。
各拓扑不要求TP1/TP2生成逐bit一致，任意launch上界数值路径限制沿用Graph报告。
组合只声明这些轨迹，不推断全部策略/容量/graph backend的笛卡尔组合。

慢rank第一次local-done相差约3.01s，观察窗口canonical/free均保持旧值；
group ACK后释放。故障轨迹返回RuntimeError(cancelled)，worker exitcodes[-9,0]，
是原生monitor对等待collective的peer执行fail-stop，不是正常reuse或静默恢复。
以上最大项目NVML合计18,144,559,104 bytes，未触发44GiB guard。

## 独立评分与回归

`20261031-105824-scorer-and-failstop-regression`：现有scoring/compaction及executor
suite中frequency/reclamation子集9 passed、22 deselected；Hkv1/4与独立CPU FP32
复数全GQA reference比较。最终实现保持TP1的mean运算顺序。

`20261031-112503-complete-gqa-nccl`：两GPU真实NCCL的Hkv4分片与Hkv1复制用例
通过，最大score误差分别1.88e-6/5.37e-6。复制owner加载全部8个Q heads，非owner
0个。该次标为ties的fixture只清零stats，epsilon项仍随K变化，因此不算ties证据；
后续修正为常数K并显式核对零方差与member顺序，旧产物保留。

`20261031-121421-complete-gqa-stable-ties-nccl`：Hkv4分片、Hkv1复制与真正零方差
stable ties三项在两GPU真实NCCL下PASS。`20261031-122228-final-reclamation-regression`
覆盖Scheduler、KV manager、Prefix、compaction/scorer与executor：23 passed、
328 deselected，19.79s，guard PASS。命令如下；14条Torch弃用warning保留。

```bash
.venv/bin/python -m pytest tests/v1/core/test_scheduler.py \
  tests/v1/core/test_kv_cache_utils.py \
  tests/v1/core/test_single_type_kv_cache_manager.py \
  tests/v1/core/test_prefix_caching.py tests/v1/worker/test_kv_compaction.py \
  tests/v1/executor/test_executor.py \
  -k 'reclamation or reclaim or physical or frequency or graph_attention' -q
```

```bash
bash .p1-work/scripts/run_tp_matrix.sh
export P1_CUDA_VISIBLE_DEVICES=0,1
source .p1-work/scripts/env.sh
.venv/bin/python .p1-work/scripts/run_guard.py --stage TP \
  --name complete-gqa-nccl -- @python .p1-work/scripts/tp_collectives.py
```

## 保留的早期失败

`20261031-111013-eager-token-local-reuse` 的诊断schedule wrapper未接收新增参数；
修正为转发`*args, **kwargs`。`20261031-111513-eager-token-local-reuse-fixed`
错误要求resident row无效tail全零；原生overwrite只保证有效prefix/num_blocks。
改为核对真实contract，并独立核对input metadata有效row/padding，未修改Runtime
迎合脚本。随后`20261031-111920-eager-token-valid-row-contract`通过。

`20261031-121246-full-token-q0-delay` FAIL：q=0少执行一个token，fixture产生的K
恰好整页对齐，未进入token compactor的delay hook。改为按intent的实际source E
计算recent边界，在该边界整页对齐时将K减1，确保测试实际token压紧；随后q0及
maintenance均通过。该失败不作为Runtime故障，也不删除旧产物。

## Evidence 与边界

每条有invocation/source、identity、evidence、rank-local事件、KV trace、stdout、
NVML和result。逐步payload hook只用于正确性，正式HTTP无该hook；Serving和
通信成本独立记录于`tp-serving.md`。不将合成Serving当作有损任务质量。
TP完成后继续DP；本阶段未修改模型权重/校准，不新增跨拓扑任务质量通过声明。

## 收尾后的回归

`Review/20261031-143734-tp-prefix-full-token`在融合普通decode长度kernel后重跑
Prefix+FULL+TP，10页全层真实重写、每rank61 FULL/3 NONE、47次FP32 attention、
46次matched-eager检查通过。`143830-tp-full-token-q0-delay`与
`143924-tp-full-token-maintenance-delay`验证非output rank慢3s时全组ACK前不free，
均PASS。初始化另拒绝pooling、局部/非因果spec与ALiBi误用。
完整审查和测试限制见 [final-review.md](final-review.md)。吞吐预算仍有FAIL，
新通信profile完整性已补齐；功能PASS不扩大为所有性能通过。
