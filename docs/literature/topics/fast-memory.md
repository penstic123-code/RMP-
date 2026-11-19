# 快记忆规则与线性注意力层

“已核实”= 代理打开原文或 arXiv 摘要页确认；“未核实”= 只看了搜索片段或二手来源。检索日期 2026-11-19。目录与总结论见 [README](../README.md)。

### 生产级混合线性注意力 / delta 规则

- [Kimi Linear（KDA）](https://arxiv.org/abs/2510.26692)：通道级遗忘门（每维一个衰减率，GDN 每头一个标量）；KDA:MLA 3:1 最好；q/k/v 加短卷积（去掉 PPL 5.65→5.70）；q、k L2 归一化；输出分头 RMSNorm 再乘低秩 sigmoid 门（换 Swish 门 PPL 升到 5.81）；头维 128。
- [Qwen3-Next](https://www.alibabacloud.com/blog/602580)：Gated DeltaNet 与门控注意力 3:1；注意力输出门消除 attention sink 与巨激活；零中心 RMSNorm 并对 norm 权重做 weight decay。
- [RWKV-7](https://arxiv.org/abs/2503.14456)：推广的 delta 规则（向量门控、上下文学习率、擦除键与写入键解耦）。
- [混合线性注意力系统分析](https://arxiv.org/abs/2507.06457)：语言建模对配比不敏感，召回依赖全注意力层占比；推荐 3:1–6:1。
- 2026 下半年：[Sparse Delta Memory](https://arxiv.org/abs/2607.07386)（稀疏读写大显式记忆，召回与上下文学习更好）；[Complex KDA](https://arxiv.org/abs/2609.24797)（门取 [-1,1]、β 取 [0,2]，状态追踪更强）；[混合模型中的巨激活](https://arxiv.org/abs/2608.12149)（输出门压低激活幅度）；[长程检索主要由全注意力承担](https://arxiv.org/abs/2606.15378)。
- 含义：快记忆层可照搬生产配方（短卷积、通道级衰减门、输出 RMSNorm + sigmoid 门、q/k 归一化），均有量化收益，下一次训练前采纳。

### 其他新序列架构

- [xLSTM Scaling Laws](https://arxiv.org/abs/2510.02228)（ICLR 2026，未核实）：80M–7B 同计算下 loss 帕累托优于 Transformer；mLSTM 是门控矩阵记忆，可作同参数对照。
- [Kronecker Delta Rule（KronLSTM）](https://arxiv.org/abs/2610.26834)（Hochreiter 组，2026-10，已核实）：只沿一个方向修正关联记忆，干扰更小、范数无需归一化即有界；称常胜过（门控）DeltaNet，规模未知。快记忆更新规则的直接替代候选。
- [Preconditioned DeltaNet](https://arxiv.org/abs/2604.21100)（MIT，2026-04，已核实）：delta 规则加对角曲率预条件，340M、1B 上提升，改动小。
- [Linear Attention Architectures: Cross-Layer Routing](https://arxiv.org/abs/2607.07953)（Schlag 组，2026-07，已核实）：350M、15B token 下 KDA + Muon 最好；Muon 一贯优于 AdamW；跨层传写入误差没赢、传写入值小幅有益——与中枢跨层通信相关的反面证据。
- [Diffusion beats AR in data-constrained settings](https://arxiv.org/abs/2507.15857)（CMU，NeurIPS 2025，未核实）：低算力下自回归仍占优，不适合本项目。
- [SpikingBrain](https://arxiv.org/abs/2509.05276)（7B / 76B，未核实）、[LFM2](https://arxiv.org/abs/2511.23404)（门控短卷积 + 少量 GQA，350M 起，未核实）：无小规模结论；Hopfield / 能量 Transformer 2025–2026 年无小规模 LM 正面结果。
