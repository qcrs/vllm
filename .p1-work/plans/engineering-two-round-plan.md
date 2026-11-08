# Decode 进展驱动的最多两轮回收

2026-11-08，起点 `1d2b991e53`。用户授权实施并补齐单卡、TP2、DP2；本轮只用
物理 GPU0/1，GPU2 留给他人任务。环境、资源、归档和 Git 继承 prompt3/4。

## 问题与设计

旧 planner 要求一次补齐完整 prefill 净缺口；一次不够不代表两次合法回收也不够。
新增默认 1、仅允许 1/2 的 `decode_reclaim_rounds`。值 2 仅用于 pressure/protected。
单轮能覆盖仍走旧路径；单轮不够时才估算同一组最多两 victim 的两轮路径。
首轮按合法最大释放量规划，预测之间两个 decode token（重试步及第二轮 forward），
另扣其他运行请求增长；不能覆盖则不启动。第二轮只用实际状态重算，不执行预测页表。
启动窗口要求所有 running 已完成 prefill，避免中间多 chunk 的增长超出两步预测。
新模式的准入重试预算包含 running 本次 decode 及随后准入检查的下一 token 预留；
CPU 真实入场测试曾发现漏算这项预留导致差一页，失败日志与修复后结果均保留。

Scheduler 持有一个等待者窗口：request ID、首轮 victim 的逻辑完成水位、轮数。
窗口内暂停后方新 admission，已有 decode 正常计算。第二轮前必须
已经完成超过首轮水位的 decode，并且没有在途 token；仅 schedule 的乐观 L 不算。
两轮用尽不再回收该窗口，等待原请求入场或取消后清理。没有启动窗口的不可行等待者
继续沿用已有 bounded backfill。最多再生成两 token 必然结束的候选优先自然释放。

每轮保持私有 E 删除≤20%、私有已计算 L 保留≥50%、最多两 victim；不在 prefill
中删除 KV，不改变 Worker→全 TP completion→Scheduler 整批 commit/free 的契约。
不新增 DP 路由算法，usage tie-break 仍无进入证据。

## 测试设计四问

- 模块用途：将可行的两轮回收转化为等待者实际进展，而非重复部分释放。
- I/O：真实 allocation miss + 已计算 decode 状态→合法 intent；只有 completion 才能
  进入第二轮，后续 full-prefill admission 必须实际成功。
- 失败模式：空转绕过单次上限、乐观 L 误当完成、第三轮、后方抢页、近 EOS 无谓搬移、
  abort 残留窗口、DP 独立池状态泄漏。
- 最便宜层级：现有 planner/Scheduler/真实 BlockPool CPU suite 检查状态与页数；真实
  Qwen3-8B 单卡/TP2 检查 completion、Graph、改写复用；DP2 检查副本与生命周期。

## 评估与停止

按 correctness→micro→Serving 执行。fixed、当前 protected+backfill2、新两轮策略
保持相同输入、容量、Graph 和 CPU 放置；单卡与 TP2 各三轮，正收益或漂移仅补一次
反向。DP 同路由条件比较 Controller，避免把 known_work 收益算给回收。SLO 不变：
每请求 TTFT≤2s 且 TPOT≤80ms。质量不扩冻结85条、不重启旧矩阵；本轮没有 prefill
删除，仍不自动取得质量保证。保留所有 FAIL/PARTIAL，不强行追求好数字。

B0：`/data_ssd/qcrs/experiments/p1-v8/Engineering-TwoRound-B0/20261108-120031-initial`。
按实现、GPU验证/测量、文档阶段精确本地提交，不 push/PR，不混用户已有差异。
