# DP2：独立 Pool、局部压力与原生 affinity

真实Qwen3-8B BF16/internal dense DP2诊断为PASS。每replica为TP1/PP1、一个
EngineCore/Scheduler/BlockPool与一个GPU Worker；物理GPU0,1，36层各8 KV heads。
GPU UUID与PID实测不同，worker DP size均为1。沿用pinned native router与abort
affinity，未新增Runtime分支、跨pool借页、KV migration或跨DP回收collective。
正式Serving另见`dp-serving.md`，不能从本轨迹推断性能预算通过。

源HEAD为`ff0788e1f0`加DP诊断脚本，invocation/source及run内hook副本固定实际版本。
协议见`../notes/dp-protocol.md`，源码链见`../source-analysis/dp-runtime.md`。
每replica设置`kv_cache_memory_bytes=256MiB`并override为33页含null；实际页数以
override为准。max_model_len512、Eager/FA2/B16，
APC/async/spec/cascade/connector/reserve-full-ISL off。所有实验由44GiB guard监测。
Python3.10、PyTorch2.11.0+cu129、Triton3.6.0、CUDA Toolkit12.1/GCC12沿用
[m0.md](m0.md)的环境，激活入口为`../scripts/env.sh`。

## GPU证据

大产物根目录`/data_ssd/qcrs/experiments/p1-v8/DP/`；两条成功轨迹如下。

| 目录 | 行为 | 实际结果 |
| --- | --- | --- |
| `20261031-122856-local-pressure-full` | native Full KV | C需要16页、本池available15而peer free32，C本地等待；A abort后B/C正常完成，两个Pool最终free32 |
| `20261031-123102-local-pressure-sink-reuse-lifetime` | 本地SinkRecent回收 | 同一shortfall1；A forward后source E258→K129，17→9页；释放10–17共8页，C取得并在全36层覆盖；A retained bit-exact，最终free32/32 |

A/C prompt均256，固定replica0；B prompt128、输出128，固定replica1。A最多输出
256，在B达到16输出时经原生`reqs_in_flight`确认affinity仍为0，再abort A；C正常
输出64。两个run中B的128个输出逐token相同，SHA256为
`2d7817979ba2834746f0cffc75d5129bc4c9c63feefb59bbad254f6a3c573ae7`。
每次B后续forward均检查最初128个KV在全36层保持不变，累计127次，abort后仍有
112次真实验证forward。peer没有回收或released页。四个128/16附加请求经原生
waiting*4+running/tie轮换分配为0/1/0/1；未迁移active请求。

Sink轨迹17次后续forward检查A retained payload，C对全部8个released ID执行
poison→finite且字节改变的首次覆盖。相同block数字只在各自replica目录解释，
未将跨进程pool对象ID或GPU地址相等作为ownership证据。两个轨迹最大项目NVML
合计34,198,257,664 bytes（31.85GiB），无上限触发。

`correctness-summary.json`保存native/Sink的完整evidence与peer逐token比较，
由`summarize_dp_serving.py`中的`summarize_local()`生成。每条另存identity、两份
KV/scheduler/worker trace、
outputs、NVML、stdout与guard result；正式Serving不启用此payload同步hook。

## 保留的诊断失败

`20261031-122928-local-pressure-sink_recent` FAIL。C已经重写8页并完成，但hook
对页ID终生要求每次写前/写后字节不同；之后balanced请求的相同prompt/位置可
合法写出完全相同KV。失败发生在该后续断言。将检测限定为每次release后的首次
poison覆盖，重复release会重新开启检测；Runtime未为诊断修改。上述新目录重跑
PASS，旧日志/退出码保留并列入correctness-summary的failed_runs。

## 复现与范围

```bash
export P1_CUDA_VISIBLE_DEVICES=0,1
source .p1-work/scripts/env.sh
.venv/bin/python .p1-work/scripts/run_guard.py --stage DP \
  --name local-pressure-full -- @python .p1-work/scripts/dp_reclamation.py --policy full
.venv/bin/python .p1-work/scripts/run_guard.py --stage DP \
  --name local-pressure-sink_recent -- @python .p1-work/scripts/dp_reclamation.py \
  --policy sink_recent
bash .p1-work/scripts/run_dp_serving.sh
```

本Gate只验证dense DP2/TP1，APC跨副本共享及DP2×TP2不在范围；已有Prefix/TP/
Graph独立与组合证据见对应报告。部分进程的原生DP coordinator控制协调不等于
跨DP回收barrier；局部ownership与worker TP1分支不执行TP回收通信。
Serving的10组正式配置和5条独立诊断均已完成；本Gate范围内没有待补的正确性
轨迹。剩余SLO失败、质量限制与未测组合在Serving报告和交付索引中单列。
