# DP2 Gate 协议（运行前固定）

目标为native internal dense DP2、每replica TP1/一GPU/一个Scheduler与Pool。
Qwen3-8B BF16、GPU0,1，既有MRV2/FA2/B16、单在途、APC/async/spec/cascade/
connector/reserve-full-ISL off。先确认实际worker dp_size1、两个独立process/Pool；
沿用native路由与active affinity，不添加cross-pool free或reclaim collective。

诊断走真实AsyncLLM与EngineCoreProc，通过诊断scheduler/worker extension记录
本地canonical状态、全36层retained/新写KV和生命周期。小池构造A真实miss而B池
空闲，记录A required/available/shortfall与B free；B的free不能使A allocation成功。
A本地回收后released ID实际分给新请求并全层写入，A继续decode。另在B运行时
结束/abort A，核对affinity和B retained KV/输出继续。两个pool相同数字ID按replica
命名空间解释。诊断hook不进入正式性能。

小池设置每replica KV memory参数256MiB、override实际33页含null、max_model_len512。
A/C定向replica0，
prompt均256，最多输出256/64；B定向replica1，prompt128、输出128。B达到16输出后
abort A，保留其余两个请求至正常结束；四个额外128/16请求走原生router。全36层
快照检查retained与C实际重写；B最初128个KV在后续forward和abort后保持不变。
native/Sink两次轨迹的B输出逐token对照。worker GPU UUID必须不同，worker DP/TP
均1，每pool初始/最终free32。等待条件有deadline并传播生成task错误。

测试设计四问：模块是replica-local ownership/pressure边界；输入为native路由的
请求和各池真实allocation状态，输出本池row/E/free与对应replica输出；防止foreign
free、全局空闲误预算、abort错路由、局部异常造成隐式跨DP等待；最低判别层为真实
双Engine/GPU集成，沿用单replica既有CPU语义测试而不堆叠同构mock。

正式Serving以DP2 native比较DP2 noop/策略，每组完整负载预热后3轮；沿用8请求
4096/8192交替、输出128、arrival `[0,.3,.6,.9,5,5.3,5.6,5.9]`，FULL_DECODE_ONLY。
Pool固定每replica4GiB宽松、1.5GiB压力；全模型权重在两卡复制，4GiB×2的选择
必须受44GiB guard约束，所有项目进程合计≤45GiB。不能沿用TP的8GiB每rank。
宽松测native/noop/Sink，压力测native/noop/Sink/Random/Tri，原生路由共8组；
压力另测定向偏斜native/Sink两组：前6个long定向replica0，后2个prompt128/256
定向replica1。只将该偏斜用作本地容量对照，不归因native路由算法。

TTFT≤2s、TPOT≤80ms、宽松noop平均吞吐开销≤5%，保持既定预算。聚合与每replica
分别保存TTFT/TPOT/ITL P50/P95/P99、tokens/s、Goodput、preemptions；路由记录
仅在ADD写一次request→replica/CPU耗时，所有策略同观测，单列该成本。实际分配
可能不均，按各轮route归属汇总，不能只报总吞吐；每replica吞吐用同轮公共时间窗，
同时保留本地first/last活动范围。native/noop输出逐token核对，有损不推断任务质量。
HTTP transport timeout固定600秒，与SLO独立；超时run保留FAIL，不合并不完整轮。

额外压力trace用于free/occupancy/reclaim/fallback/queue偏斜解释。正式计时不启用
逐step payload/allocator trace，不把profile/诊断开销并入正式Serving。保留负结果；
没有重复偏斜及收益证据时不实现pressure-aware router。
压力诊断固定为native/skewed各full/Sink四组，每组完整预热加一轮，分别按replica
和正式时间窗统计事件、R与queue，不能将这些频次直接复制给无trace正式轮。

正式矩阵完成后补一条4GiB/native/Sink trace，完整预热加一正式轮：逐replica核对
allocation miss、reclaim_select/commit与schedule intent均为0，证明无压力时未
调用评分/压紧链。沿用既定输入、SLO与资源约束，该诊断不加入正式吞吐。
