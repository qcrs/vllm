# D1 dense DP负载闭环与D2停止判断

D1完成：普通dense进程能够发布真实队列变化，Coordinator独立汇总，前端完成归零，
快照不能抹掉本client的在途下界。真实DP2对照未测得可分辨性能收益。D2未实施：
一轮最小诊断与离线回放不足以证明重复错选或持续忙闲错配。

起点`6dee942bbdbfcc13f4cf598f71a91002066a3d5a`，当前分支本地实现。
上游学习绑定#49204 head `bf3c0293a6cf87880f75cc01206d8327e248d0e5`，
merge `59a6b0411d1817c712728a17fa8c2e0cdb4cf1f9`；实际diff归档于
[初始记录](../experiments/Engineering/20261031-232434-initial-state)。只借鉴publisher/
non-lockstep设计，未照搬整套策略；不创建上游PR。

## 问题、所有者与变化

旧dense走普通`EngineCoreProc`，只有MoE的`DPEngineCoreProc`发布请求计数。
Frontend已有路由并不代表收到Scheduler状态。D0正式运行的running始终为0，
快照主要是本地waiting增量；D1可看到running=1/2及最终全零。

| 位置 | 本轮变化与契约 |
| --- | --- |
| `EngineCoreProc.run_busy_loop` / `_maybe_publish_request_counts` | 输入后与step后发布变化，包括idle前最后归零；`last_counts`迁入基类 |
| `DPCoordinatorProc.process_input_socket` | dense各rank最新值独立汇总；仅MoE保留同步step/wave与等待 |
| `DPLBAsyncMPClient.get_core_engine_for_request` | `C=max(4W+R,F)`；F从既有`reqs_in_flight`派生，无新增生命周期表 |

Scheduler输出是`(running, waiting)`，具名编码为`SchedulerStats`；Coordinator/
Frontend为`[waiting, running]`。消息复用output queue→Msgpack/ZMQ→stats task，
不新增RPC。显式rank、late-interaction sticky、轮转tie-break、完成和abort affinity
沿原接口。Pool、回收策略、completion与canonical commit/free均不在D1改动范围。
详见 [源码链](../source-analysis/dp-runtime.md) 和 [协议](../notes/dp-protocol.md)。

## 正式对照

Qwen3-8B BF16，DP2×TP1，物理GPU0,1；Worker主线程CPU20/21，FA2/NHD/B16、
MRV2、APC/async off，SinkRecent retention=.5、reserve=0、max_len9216、
batch512、seqs8、FULL_DECODE_ONLY `[1,2,4,8]`。每次独立启动、完整负载预热后
3轮；每轮8请求、128输出，arrival `[0,.3,.6,.9,5,5.3,5.6,5.9]`。
native为4096/8192交替、1.5GiB/replica；uniform为全部4096、4GiB/replica。

| 源码/负载 | 每轮output tok/s | 合并output tok/s | TTFT P99/max(s) | TPOT P99/max(ms) | SLO / goodput(req/s) |
| --- | --- | --- | --- | --- | --- |
| D0 native | 113.569 / 113.417 / 113.327 | 113.437 | 1.224 / 1.225 | 20.335 / 20.336 | 24/24 / .8862 |
| D1 native | 113.574 / 113.325 / 113.222 | 113.374 | 1.237 / 1.240 | 20.368 / 20.372 | 24/24 / .8857 |
| D0 uniform | 124.621 / 124.714 / 124.630 | 124.655 | .471 / .472 | 17.318 / 17.325 | 24/24 / .9739 |
| D1 uniform | 124.491 / 124.542 / 124.654 | 124.562 | .473 / .476 | 17.314 / 17.324 | 24/24 / .9731 |

四次guard均PASS/exit0，无抢占；每轮实际prompt工作native的rank0/1为32768/
16384，uniform为16384/16384，每rank均4请求。native两版本token IDs 24/24相同，
uniform同样24/24相同。吞吐差native约-.056%、uniform约-.074%，均不能宣称收益；
uniform无压力退化在5%预算内。小样本分位数仅描述这些请求。

每rank的吞吐与goodput使用同轮公共窗；3轮合计native两rank各12请求、prompt
98304/49152；D1 TTFT P99为1.239/.447s，TPOT P99为20.370/17.007ms。
uniform两rank各49152 prompt，D1 TTFT P99为.456/.475s，TPOT P99为17.041/
17.319ms。每请求及每轮分布保存在summary/requests文件，不以跨轮相对时钟
拼接active span。无抢占且prompt执行量等于输入，未见额外prefill重算。

峰值项目NVML合计native为37,541,117,952 bytes，uniform为42,825,940,992 bytes，
均小于44GiB停止阈值；不是各卡分别可用44GiB。绑核证据是各run的
`worker-affinity-{0,1}.json`，只声明Worker主线程放置。

正式产物位于 [Engineering-DP-Formal](../experiments/Engineering-DP-Formal)：
`20261031-232701-d0-native-sink-batch512`、`20261031-232826-d0-uniform-sink-batch512`、
`20261031-233439-d1-native-sink-batch512`、`20261031-233605-d1-uniform-sink-batch512`。

## 最小诊断与D2反事实

[单次诊断](../experiments/Engineering-DP-Diagnostic/20261031-233810-d1-native-load-chain)
guard PASS/exit0，113.303 tok/s、8/8 SLO、零抢占，与正式轨迹接近。只在该run
启用`dp_load_diagnostic.py`，归档Scheduler post-output、消息与route，正式不启用。

51个首次观察到的计数generation，生成→Frontend延迟P50/P95/max为
2.347/59.567/88.145ms；生成→Coordinator publish为.761/58.844/87.636ms。
publish→Frontend P50/P95为.727/2.107ms。计数不变不发新generation，route时
旧generation年龄可达约2s，不能把这种状态年龄算成网络延迟。最终Scheduler、
Coordinator、Frontend均为`[[0,0],[0,0]]`。这是同机monotonic时间上的观测，
只覆盖实际收到的generation，不能证明每个中间状态都曾对外发布。

timed rank0回收4次，释放256/128/256/128 blocks，各次缺页请求后续allocation
成功；rank1无回收。累计释放不是唯一页数。不能单凭2:1 prompt量断言路由错误。

按计划离线计算`S=(max(4W+R,F)+U/512)*(1+usage)`；U使用最后可见poststep的
`sum(max(0,L-C+inflight))`，usage直接取BlockPool。timed 8次仅最后一次D2建议
改变选择：rank0/1 U=2048/1031，D1选0、D2选1；较早同类选择U=2048/1542时
两公式都选0。完整warmup未重复出现该差异。这里没有重建另一条服务动态，
瞬时chunk进度即能改变排序，也未证明改变route会改善TTFT/goodput。

因此D2为NOT_RUN，保留工作量启发式的适用限制；未新增U/usage发布字段、
pressure router、Prefix affinity、跨池借页或migration。

## 验证与复算

复用两个现有engine suites，7 passed、27 deselected：dense busy loop真实Msgpack
计数序列、MoE字段、真实Coordinator子进程/ZMQ non-lockstep、已有sticky、
常规LB、本地F/完成、显式rank/abort。CPU只证明控制链；DP2 HTTP补真实进程证据。
原suite顶层加载未缓存Llama tokenizer，直接collection失败。已归档runner仅将两个
无关tokenizer加载改用本地真实Qwen3，不执行其模型测试；不称完整上游suite通过。

runner及日志为初始记录中的`run-d1-cpu-tests.py`、`d1-cpu-tests.log`。复算入口：

```bash
.venv/bin/python .p1-work/scripts/summarize_dp_serving.py \
  --engineering /data_ssd/qcrs/experiments/p1-v8/Engineering-DP-Formal \
  --diagnostic /data_ssd/qcrs/experiments/p1-v8/Engineering-DP-Diagnostic/20261031-233810-d1-native-load-chain \
  --output /data_ssd/qcrs/experiments/p1-v8/Engineering/20261031-232434-initial-state/dp-summary.json
```

D1是学习上游后完成的跨进程正确性修复；不是本负载性能突破。旧固定6/2 DP结果
仍按 [历史报告](dp-serving.md) 使用；本轮不扩张支持域，不改变冻结85条质量样本结论。
