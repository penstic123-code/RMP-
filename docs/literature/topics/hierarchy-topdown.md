# 层级、分块与自上而下反馈

> 与主线 RMP 相关：RMP 的 R2、R3 背景，见 [23.10](../../design/candidates/23-routed-pool.md)。

“已核实”= 代理打开原文或 arXiv 摘要页确认；“未核实”= 只看了搜索片段或二手来源。检索日期 2026-11-19。目录与总结论见 [README](../README.md)。

### 字节模型的动态分块与层级

- [H-Net](https://arxiv.org/abs/2507.07955)（Hwang、Wang、Gu 2025）：相邻编码状态余弦相似度低于 0.5 即切分，比例损失压住压缩率，EMA 平滑 + STE 使切分可微；平滑模块对稳定性必不可少；编解码层用 Mamba 最好、纯 Transformer 最差。主实验 760M–1.3B。
- H-Net 小规模复现（社区，[链接](https://github.com/avikds/h-net-from-scratch-dynamic-chunking-on-raw-bytes)）：TinyStories、0.1–0.4M 参数、3 种子、4000 步 bits/byte：纯字节 Mamba 1.451 < 空格规则 1.543 < 动态分块 1.585；学到的边界只有 14–30% 落在空格上。另一复现（[链接](https://github.com/KidIkaros/hnet-kaggle)）结论：效率优势是大规模效应，小规模复现不出来。
- [BLT](https://arxiv.org/abs/2412.09871)（Meta，ACL 2025）：用小字节模型的下一字节熵切 patch，FLOP 对齐到 8B；优势在把省下的算力挪给全局模型。
- [Scratchpad Patching](https://arxiv.org/abs/2605.09630)（Google，NeurIPS 2026）：指出 patch lag（patch 内字节只能用上一 patch 的过时表示），在高熵位置插入临时状态刷新上下文。
- 2026 下半年：[ATDC](https://arxiv.org/abs/2605.30080)（压缩率课程式逐步提高更稳）；[Breaking the Token Ceiling](https://arxiv.org/abs/2609.12303)（约 1B 时字节模型低算力弱、高算力反超）；[The Right Byte First](https://arxiv.org/abs/2610.22071)。
- 含义：与 E38、E40 一致——小规模下纯字节递归胜过空格分块与动态分块。若再试层级：字节层要强、EMA 平滑、分阶段学习率、压缩率由低到高，并在 FLOP/字节对齐下比较。

### 序列模型中的自上而下反馈

- [Full-bandwidth transformer](https://arxiv.org/abs/2608.08888)（微软，2026-08）：上一步顶层隐状态经 GLU 与下一 token 嵌入融合送回输入；预训练后期才引入并混入少量多遍训练，仍并行；1B 时约等于多训 1.5 倍 token。
- [Latent Lattice LM](https://arxiv.org/abs/2611.21608)（MIT/Adobe，2026-11）：每层加跨时间递归的水平潜流调制主干；406M 时 PPL 15.41→14.40，GDN、Mamba-2 上也有效。
- [Recirculation](https://arxiv.org/abs/2608.17981)（2026-08）：推理期加深层→浅层反馈，原权重冻结；prefill 须串行。
- [LLM-JEPA](https://arxiv.org/abs/2509.14252)（ICLR 2026）：next-token 之外加嵌入空间预测损失，可并行的潜空间预测辅助目标。
- 含义：自上而下的收益报告都在大规模；本项目 E38、E40 在 0.36M 上同参数变差。可试“后期引入、不占主干参数”的做法。
