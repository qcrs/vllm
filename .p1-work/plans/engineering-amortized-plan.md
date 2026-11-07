# Protected Controller：有界回收合并

2026-11-08，承接 `d6d4e7602b`。仅使用用户指定的物理 GPU1/2（单卡1），
CPU20/21、现有环境、guard/44GiB停止/45GiB硬限、离线模型和Git规则不变。
B0为 `Engineering-Amortized-B0/20261108-014007-initial`；保留全部用户差异，
独立index精确提交，不push、不创建PR。

## 已有证据与问题边界

此前修复已消除正式TP2抢占/重算，且同页目标优先zero-copy；不能重复归为本次成果。
固定GPU1/2正式3轮，fixed为101.72–102.05tok/s、18/24，protected为
71.89–72.09tok/s、6/24。独立trace的timed区间48次commit中45次只释放1页，
全部zero-copy；514次计划466次为空。不能将当前退化继续归因于KV复制。
物理池681页，4097token的victim单次最多释放52页，8192token新请求缺88页，
严格保护下单次无法入场。这部分容量/排队代价与单页maintenance须分开报告。

## 实施contract

只在真实miss且原硬缺口有合法1–2victim计划后，合并同一victim集合的近期回收。
软目标来自这些已入场请求及当前requester的有限输出上限：
`min(prompt_tokens + max_tokens - 1, max_model_len - 1)`减去post-forward已计算量
和当前硬预算已覆盖量。每请求将剩余KV增量向上取页作为容量余量；这只是内存
上界，EOS可能提前结束，绝不进入逻辑已计算L。无权重、扫描参数或额外状态。

软目标最多用原硬计划选中victim的合法容量，不增加victim，不把原本不可行的硬
缺口变成部分回收。用同一个纯计划函数重新分担，保留20%/50%、共享prefix、
min_keep、sink/recent、净尾页需求、页重映射和确定性。fixed与requests默认不变。
记录额外保留/删除取舍；不声称质量无损或一定达到SLO。

## 验证与停止

先CPU行为与独立页预算测试，再真实TP2单/双victim、延迟completion、Graph及
rewrite/reuse，DP2本地进展/abort/reuse。复用现有micro记录决策与Worker成本。
最终同源码fixed/protected各3轮，仅新鲜正式压力对照，完整预热、SLO不变；有正向
收益或漂移才一次反向。旧protected正式数据只作冻结历史对照。
DP沿用固定W1/W0及必要组合；无重复新鲜同分错选→miss且peer有余量，停止usage
tie-break。历史FAIL/PARTIAL和85条质量样本保留，不扩矩阵。

若回收次数明显降低但Serving仍落后，交付容量与队头等待的证据，停止继续调参。
分别报告correctness、micro、Serving及限制，不以局部修复包装全面提速。
