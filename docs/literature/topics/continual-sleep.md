# 持续学习、睡眠巩固与选择性写入

“已核实”= 代理打开原文或 arXiv 摘要页确认；“未核实”= 只看了搜索片段或二手来源。检索日期 2026-11-19。目录与总结论见 [README](../README.md)。

### 持续 / 在线学习

- [Forgetting in LMs: Capacity, Optimization, and Self-Generated Replay](https://arxiv.org/abs/2605.26097)（Marek、Izmailov、Wilson 等，2026-05，已核实）：用模型自生成样本回放几乎能消除遗忘；但模型预训练到接近饱和、没有余量时回放也救不了；调低学习率减少遗忘但要多花很多步。0.3–5M 模型易饱和，需先测余量。
- [When to Review: Spaced Repetition for Continual Pre-Training](https://arxiv.org/abs/2608.17530)（2026-08，已核实）：按每条样本困惑度用 SuperMemo-2 安排复习时间，旧知识准确率追回 5–37 个百分点；可直接做睡眠回放的采样器。
- [SuRe: Surprise-Driven Prioritised Replay](https://arxiv.org/abs/2511.22367)（华为诺亚 / UCL，2025-11，已核实）：回放池优先存 NLL 高的序列，再用 EMA 合并快、慢两个 LoRA。几乎就是“在线低秩 + 睡眠巩固”的现成基线。
- [Can Scale Save Us From Plasticity Loss in LLMs?](https://arxiv.org/abs/2606.24752)（2026-06，已核实）：5M–314M 的 GPT 全部出现可塑性丧失，越大越晚，分布不变也会出现；只诊断不修复。
- [FIRE: Frobenius-Isometry Reinitialization](https://arxiv.org/abs/2602.08040)（ICLR 2026 Oral，已核实）：在等距约束下找离旧权重最近的新权重（Newton-Schulz 近似），恢复可塑性，GPT-0.1B 上有效。
- [RL's Razor](https://arxiv.org/abs/2509.04259)（MIT，ICLR 2026，未核实）：新任务上相对原模型的 KL 能预测遗忘，on-policy 更新遗忘更少；支撑自蒸馏、on-policy 巩固。

### 学习过程中的记忆形成

- [Language Models Need Sleep](https://arxiv.org/abs/2606.03979)（Behrouz 等，Google，2026-06/07，已核实）：睡眠分 Knowledge Seeding（短期记忆经 on-policy 蒸馏 + RL 模仿蒸进长期参数）与 Dreaming（RL 生成合成数据复习）。与本项目睡眠设计几乎同构，须作为对照。
- [Do LMs Need Sleep? Offline Recurrence](https://arxiv.org/abs/2605.26099)（Lee 等，2026-05，已核实）：上下文满了就“睡”——对上下文离线循环 N 次，用学到的局部规则写进 SSM 快权重，再清空 KV cache，训练时端到端反传穿过睡眠；睡得越久深层推理收益越大。
- [In-Place Test-Time Training](https://arxiv.org/abs/2604.06169)（ICLR 2026 Oral，已核实）：MLP 最后一层投影当快权重按块更新，目标与下一 token 预测对齐而非通用重构。
- [LaCT: Test-Time Training Done Right](https://arxiv.org/abs/2505.23884)（MIT / Adobe，ICLR 2026，未核实）：2K–1M token 大块更新快权重，GPU 利用率高得多，可用非线性快权重。
- [Strategy Coopetition: emergence and transience of ICL](https://arxiv.org/abs/2503.05631)（DeepMind / UCL，ICML 2025，未核实）：ICL 先出现后被“受上下文约束的权重内学习”挤掉；L2 正则能保住 ICL。提醒：训练久了快记忆的作用可能被慢权重吞掉，要持续监测。
- 惊奇门控写入：[TRC²](https://arxiv.org/abs/2602.22479)（单作者，已核实）、[SGM](https://arxiv.org/abs/2606.31495)（单作者，视觉，已核实），均为弱证据。

### 持久记忆与选择性写入

- [Sparse Memory Finetuning](https://arxiv.org/abs/2510.15103)（Meta 2025；[复现](https://arxiv.org/abs/2605.03229)）：只更新本批访问频率明显高于背景语料的少数记忆槽（类 TF-IDF）；学新事实时 NQ 遗忘：全量微调 89%、LoRA 71%、本方法 11%。
- [Memory Layers at Scale](https://arxiv.org/abs/2412.09764)（Meta 2024）：可训练稀疏 KV 查表，加参数不加计算。
- [M+](https://arxiv.org/abs/2502.00592)（ICML 2025）；[Larimar](https://arxiv.org/abs/2403.11901)（ICML 2024，一次性编辑与选择性遗忘）。
- [Delayed Supervision for Test-Time LMs](https://arxiv.org/abs/2609.32312)（2026-09）：隔开很多无关事件后再对早先事实提问并监督，DeltaNet、RWKV-7 等在 BABILong、RULER 上提升（本项目跨会话记忆对话即属此类）。
- [TRC²](https://arxiv.org/abs/2602.22479)（2026）：延迟惊奇写入 + 回放巩固 + 按实测遗忘自适应回放。
- 含义：睡眠巩固可采用“稀疏槽 + 按访问频率选择写入”。
