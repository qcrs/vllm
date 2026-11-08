# 长 prefill 计算预算公平性修复与同卡基线

2026-11-08，起点 `eef5da27d1`。用户授权实施；GPU1 被占用，GPU2 不使用，
本轮仅物理 GPU0、Worker 主线程 CPU20。固定环境、资源、归档、Git 继承 prompt3/4。
B0：`/data_ssd/qcrs/experiments/p1-v8/Engineering-Prefill-B0/20261108-134721-initial`。

## 问题与边界

两轮回收解决容量准入后，running 长 prefill 仍能吃完每批 1024 token，
使 waiting 短请求得不到计算和容量检查。先启用已有
`long_prefill_token_threshold=512`，fixed/protected 使用同样的设置，
不改回收约束、full-prefill 预留、TP completion 或 DP 路由。
只有固定限额的独占预算浪费得到证据，才追加简单的独占请求豁免。

扩展 CPU 回归还复现了独立的入场边界缺陷：满页 prompt 入场后没有第一步 decode
空间，短请求立即被抢占。protected 的 full-prefill 容量检查与 retry 估算统一加上
第一个 decode token（仅确实还需输出时）；不预留全部未来输出。CPU fixture 同时
修正中间 chunk 不能产 token、max_model_len 不能等于 batch 的错误状态。

CPU 验证四问：Scheduler 用于将容量及 token 预算转换为请求进展；
输入为真实请求/BlockPool 与 Worker completion，输出为实际调度及首 token；
防止只留下 token 预算却未实际接纳短请求、未完成 prefill 就模拟产 token；
最低成本为现有 pressure fixture 的 Scheduler 行为测试，GPU 再验证真实生成。

## 新对照与停止条件

原混合负载不变：`[4096,8192,128,128]×2`，原到达时间、每请求输出128、
batch1024/maxseq8、FULL_DECODE_ONLY、1.5GiB KV pool、完整负载预热及3轮。
正向固定顺序：fixed0、protected0、fixed512、protected512；protected 均为
rounds2/backfill2。必要时仅补一次反向，不扫描更多阈值、不重跑旧矩阵。
有新的独占豁免实现时只补对应必要比较，不重新扩大已有矩阵。

首 token 分解复用低扰动 work hook：客户端发送、Scheduler 接收、首次调度、
prefill forward 结果消费、首个 sampled token、客户端收到首 token。
使用本机 monotonic 时钟和请求 ID 关联；forward 完成边界是 CPU 消费结果，
不是 CUDA event kernel 时间。正式不开逐步 trace/payload clone/额外 GPU 同步。
记录 GPU 背景占用，按长短拆 TTFT、联合 SLO、吞吐、抢占/重算及回收。

SLO 仍为每请求 TTFT≤2s 且 TPOT≤80ms。保留历史和本轮 FAIL/PARTIAL，
85条质量样本不扩、不恢复旧质量矩阵；合成 Serving 输出仅记录，不作质量保证。
GPU correctness/Serving 验证本轮调度组合，未修改 Worker 无需重跑旧 compaction
micro 矩阵；调度 micro 单列，不能当 Serving 提速。真实 TP2/DP2 因资源不足
本轮 NOT_RUN，CPU 契约回归与历史多卡证据分别列出，不冒充本轮多卡测试。

按实现/验证/收尾做本地精确提交；保留原 staged/unstaged 内容，不 push/PR。

## 执行结论

已完成实现、CPU92项/DP本地13项及适用静态检查、同GPU0/CPU20正反向八组对照。
protected短TTFT改善74.80%/73.73%，完成速率下降1.84%/1.07%；相同512配置下
距fixed完成速率0.28%/0.02%。八组均24/24 SLO、零抢占/重算。保留显式512推荐，
停止阈值扫描和追加独占豁免；micro及本轮真实TP2/DP2为NOT_RUN。
详细取舍、历史负结果和归档见[评估](../reports/engineering-prefill-evaluation.md)与
[收尾](../reports/engineering-prefill-closeout.md)。
