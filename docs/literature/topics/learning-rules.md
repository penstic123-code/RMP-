# 不用反向传播的学习与脑机制落地

“已核实”= 代理打开原文或 arXiv 摘要页确认；“未核实”= 只看了搜索片段或二手来源。检索日期 2026-11-19。目录与总结论见 [README](../README.md)。

### 不用反向传播的学习

- [SOLO](https://arxiv.org/abs/2609.35440)（2026-09）：模块共享最终模块的只读读出、模块间不传梯度，340M–2B 时零样本与 BP 相差不到 1 分（模块内是否用 BP 未说明）。
- [贪心局部学习做 LM 预训练](https://arxiv.org/abs/2610.04867)（NeurIPS 2026 workshop）：2 块→4 块与端到端差距翻倍以上。
- [Split Forward Gradients](https://arxiv.org/abs/2607.16612)（2026-07）：16M GPT 的 WikiText-103 PPL：BP 150、Split-FG 387、纯前向梯度 2885。
- [固定结构化反馈代替 BPTT](https://arxiv.org/abs/2503.23104)（2025）；[μPC](https://arxiv.org/abs/2505.13124) 只在简单分类上；预测编码训练的 LM 只在极小模型上接近 BP（[链接](https://arxiv.org/abs/2211.03481)）。GrAPE（DFA 用于 Transformer）未核实。
- 含义：最好结果仍是“局部损失 + 块内 BP”；纯局部规则或前向梯度与 BP 有 2 倍以上 PPL 差距。离线继续用 BP，局部规则只放在快权重内部。

### 神经科学机制落地成 ML 方法

- [Palimpsa: Learning to Remember, Learn, and Forget in Attention-Based Models](https://arxiv.org/abs/2602.09075)（Neftci 组，2026，已核实）：每个注意力状态配“重要性”状态，用贝叶斯元可塑性控制可塑性；称各种门控线性注意力是其特例，MQAR 上胜过基线。与本项目快记忆最贴近，可加元可塑性门。
- [SynGAP: Metaplasticity as Adaptive Gradient Preconditioning](https://arxiv.org/abs/2608.14634)（2026-07，已核实）：无需任务边界，数据流上维护 Fisher 的 EMA 生成有界乘性梯度掩码，即流式 EWC；只在视觉上做过。
- [Neo-Hebbian plasticity rules for RNNs](https://neurips.cc/virtual/2025/133744)（NeurIPS 2025，未核实）：元学习找三因子规则（资格迹 × 调制信号）训练 RNN；可用于设计“惊奇作第三因子”的写入规则。
- [Zenke & Laborieux 2024 综述](https://arxiv.org/abs/2405.16922)（未核实）：突触标记与捕获、元可塑性的现象学模型。2025–2026 年未找到把突触标记与捕获、海马索引（参数内）或树突计算做进 LM 的工作——是空白。[HippoRAG](https://arxiv.org/abs/2405.14831)（未核实）只在检索层实现海马索引。
