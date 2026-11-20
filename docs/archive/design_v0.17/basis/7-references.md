## 7　研究依据

- 固定递归核心与泄漏单元可参照[储备池计算综述](https://www.sciencedirect.com/science/article/pii/S1574013709000173)。
- 选择性对角线性循环参照 [Mamba](https://arxiv.org/abs/2312.00752)；对角循环网络的精确在线学习参照 Zucchet 等，“Online learning of long-range dependencies”（NeurIPS 2023）。
- delta 规则快权重记忆参照 [Schlag 等，2021](https://arxiv.org/abs/2102.11174) 及 [Yang 等，2024](https://arxiv.org/abs/2406.06484) 的 DeltaNet 并行训练；中枢的门控 delta 头参照 [Gated DeltaNet](https://arxiv.org/abs/2412.06464)（Yang 等，ICLR 2025）。
- 门控线性循环与局部滑动窗口注意力的混合参照 [Griffin](https://arxiv.org/abs/2402.19427)（De 等，2024）；线性循环模型在联想召回上的差距及 MQAR 任务见 [Zoology](https://arxiv.org/abs/2312.04927)（Arora 等，2023）。
- 小模型的语言能力参照 [TinyStories](https://arxiv.org/abs/2305.07759)（Eldan 与 Li，2023）。
- 本地预测误差参照 Rao 与 Ballard（1999）的预测编码模型；快慢两套记忆与睡眠巩固参照 McClelland 等（1995）的互补学习系统理论。
- 动态稀疏训练中的新生参数稳定性见 [Xiao 等，ICML 2026](https://proceedings.mlr.press/v306/xiao26t.html)；跨步信用分配参照 [SnAp](https://openreview.net/pdf?id=q3KSThy2GwB)。
- 重放参照[小型情景记忆研究](https://arxiv.org/abs/1902.10486)。
- 这些工作各自支持一种候选机制，三项联合能力以本项目实验为准。
- v0.11 引用的两份先导报告存放在个人账户中，共享本文档前须换成随附文件。

**2026-11 文献调研（代理调研所报，2026 年条目未逐篇独立核对）。**

- 测试时学习：[TTT-E2E](https://arxiv.org/abs/2512.23675)（只在线更新高层 MLP，配冻结静态副本）、[Titans 复现](https://arxiv.org/abs/2510.09551)（分块会削弱效果）、[MIRAS](https://arxiv.org/abs/2504.13173)（遗忘即 L2 保留正则）、[Titans](https://research.google/pubs/titans-learning-to-memorize-at-test-time/)、[ATLAS](https://arxiv.org/abs/2505.23735)。
- delta 规则变体：[负特征值](https://arxiv.org/abs/2411.12537)（β∈(0,2)）、[Gated DeltaNet-2](https://arxiv.org/abs/2605.22791)（按通道擦除 / 写入门）；[DeltaProduct](https://arxiv.org/abs/2502.10297)、[Log-Linear](https://arxiv.org/abs/2506.04761) 对本规模不划算。
- 防遗忘：[带重放组合后向迁移近零](https://arxiv.org/abs/2603.18641)、[低重放比下 LoRA 退化更缓](https://arxiv.org/abs/2505.12512)、[SDFT 自蒸馏](https://arxiv.org/abs/2601.19897) 与 [自蒸馏致崩溃的反例](https://arxiv.org/abs/2607.01763)、[遗忘集中于低频 token 的输出嵌入](https://arxiv.org/abs/2610.09835)、[EWC 用于持续预训练](https://arxiv.org/abs/2505.05946)、[LoRA 少学少忘](https://arxiv.org/abs/2405.09673)。
- 在线信用分配：[Zucchet 等 2023](https://arxiv.org/abs/2305.15947)、[只用即时导数的近似](https://arxiv.org/abs/2603.28750)。
- 巩固与做梦：[睡眠框架](https://arxiv.org/abs/2606.03979)。
- 多时间尺度：[Nested Learning / Hope](https://arxiv.org/abs/2512.24695)。
- 采纳决定见 [plan.md](../../plan/README.md)。


