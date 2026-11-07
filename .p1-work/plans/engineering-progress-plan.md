# Protected Controller 进展与成本修正

2026-11-07，起点 e3550ac916；B0 保存于
`Engineering-Progress-B0/20261107-231330-initial`。用户指定本轮只用 GPU1/2，
覆盖旧 prompt3/4 的 GPU0/1 放置；其余环境、CPU20/21、guard、显存和 Git 规则继承。
保留用户 staged/unstaged 文件，只用独立 index 提交本轮差异，不 push、不创建 PR。

## 已确认的问题与设计

1. 旧 protected 只覆盖当前 allocation miss，释放页可被下一轮前方 decode 重新占用。
   历史 retry_pass 虽能覆盖第一次重试，仍过度接纳长 prefill，造成重算放大。
2. 同样释放页数时，最大 K 可能需要复制整段 KV，稍少保留不足一页即可整页重映射。

protected 的等待 admission 检查完整已知 prefill（包括已生成后重算的历史），扣除
运行请求未分配的已知 prefill 容量与下一次 decode 分配。沿用 allocator 的
full_sequence_must_fit/reserved_blocks；所有需求均在物理 E 域，逻辑 L 仍仅为已计算量。
回收预算覆盖 post-forward 后所有 running 与 requester 的完整已知 prefill/下一步
decode；候选的净容量贡献扣除其自身下一步尾页分配。无合法单/双计划就等待/原生抢占。
这是一段有界的已知工作量容量检查，不保证未来未知输出永不触发压力。

同净容量贡献下，SinkRecent 优先最大的 exact zero-copy K，且 K 不低于原
protected floor；最多额外删 B−1 个 token。尾页分配可能使物理释放多一页、随即由
victim 自身消费，净贡献不变。此取舍不能称质量无损。其他 policy 继续最大 K。
双 victim 的保留率仍反向单调，二分检查交点相邻 split，并用独立 token 枚举 oracle
验证分段映射下的选择。fixed/demand/retry 等旧模式行为保持，默认不变。

## 测试 contract（先设计后实施）

- 用途/I/O：纯预算函数接收真实候选与 post-forward 分配需求，输出最多两 victim，
  保证实际净释放覆盖需求、20%/50% 保护和确定性；独立枚举比较页分担。
- 失败模式：尾页吞回释放页、长 prefill 首块能入场后半途抢占、累计 floor 失守、
  不可行整页映射改变释放量、APC 缓存页被重复计数。
- 最低层：扩展现有 KV/Scheduler 真 Pool fixture；检验等待者在 abort 前入场、完整
  chunked prefill 无重算，以及不可行请求无 intent/hash mutation。
- GPU：既有真实 TP2 单/双 victim、延迟 completion、Graph、rewrite/reuse；DP 原小池
  要求 C 在 abort A 前得到输出并真实复用释放页。沿用已有 helper，失败原样保留。

## 执行与停止

先完成 GPU1/2 old protected 基线，再 CPU/Runtime 提交、GPU correctness、micro、
有限 fixed/new protected 正式 Serving 正反向对照。仍 TTFT≤2s 且 TPOT≤80ms，
不重跑废弃矩阵、不扩85条质量样本。性能收益取决于实测，不预设 SLO 通过。
DP 独立诊断，只有 prompt4 的重复新鲜同分错选→本地压力且 peer 有余量证据才加
usage tie-break；否则明确 STOPPED，继续组合验证。分别归档 correctness/micro/Serving。
