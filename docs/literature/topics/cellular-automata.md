# 元胞自动机与 Wolfram（只作参考与启发）

本页只作参考与启发，不对应任何采纳决定；采纳与排期只在 [plan/adoption.md](../../plan/adoption.md)。“已核实”= 打开了摘要页或原文页确认；“未核实”= 只看了搜索片段或二手来源。检索日期 2026-11-20。目录见 [README](../README.md)。

### 元胞自动机数据用于语言模型训练

- [Training Language Models via Neural Cellular Automata](https://arxiv.org/abs/2603.10055)（Lee、Han、Kumar、Agrawal，2026-03，[代码](https://github.com/danihyunlee/nca-pre-pretraining)，已核实摘要）：先在神经元胞自动机（NCA）生成的合成时空数据上做下一 token 预测（“预预训练”），再做语言预训练。作者自报：1.64 亿 NCA token 使下游语言建模最多好 6%、收敛最多快 1.6 倍，胜过用 16 亿 token 自然文本（C4）做预预训练；迁移最多的是注意力层；最佳复杂度随领域变（代码偏简单动力学，数学与网页文本偏复杂）。模型 4 亿–30 亿参数，子词 token；网格、规则与分词细节摘要页未给。
- [Intelligence at the Edge of Chaos](https://arxiv.org/abs/2410.02536)（Zhang 等，ICLR 2025，未核实，搜索摘要）：用 Wolfram 初等元胞自动机数据分别训练 GPT-2，下游测推理与象棋走子。规则越复杂（第 III、IV 类）下游越好，最佳在“有结构但难预测”的混沌边缘；打乱时间顺序伤推理；复杂数据训练的模型更多关注历史状态。

### 把 NCA 当作语言模型架构

- [TextNCA](https://aclanthology.org/2026.findings-emnlp.1413/)（Mittal 等，EMNLP 2026 Findings，已核实摘要页）：一维因果，token 位置即细胞，窗口注意力取邻居、GRU 式门更新；窗口逐级加宽 8 → 32 → 128，每级参数迭代多次。约 3000 万参数、SlimPajama 6 万步：WikiText-103 困惑度 60.3，同规模 6 层 Transformer 52.8。打乱或拉平窗口顺序使困惑度增加 16.7–70.8；不迭代、保留窄到宽窗口顺序的滑动窗口 Transformer 表现相当（3 种子）——收益来自窄到宽的顺序，没有证据显示 CA 式迭代共享权重有益。作者定位为分析工具而非性能方法。

### NCA 的其他方向（2025–2026，均未核实）

- [NCA for ARC-AGI](https://arxiv.org/abs/2506.15746)（Xu & Miikkulainen，ALIFE 2025）：梯度训练局部迭代规则，从示例学出网格变换并用于测试输入；目的在研究 NCA 作为计算模型的能力与边界。
- [Universal NCA](https://arxiv.org/abs/2505.13058v1)：梯度下降训练连续的“通用元胞自动机”。
- [Cells to Pixels](https://arxiv.org/abs/2506.22899v2)：高分辨率 NCA 输出（二手报道称 SIGGRAPH 2026）。
- [BraiNCA](https://arxiv.org/html/2604.01932v1)：类脑 NCA，形态发生与运动控制。
- 记忆：[EngramNCA](https://arxiv.org/pdf/2504.11855)（记忆转移）；[非平衡系统中的记忆](https://arxiv.org/abs/2508.15726v2)（Physical Review Letters 2026）。
- 综述：[统一框架与参考实现 NCAtorch](https://arxiv.org/abs/2604.24990)；[应用综述](https://arxiv.org/pdf/2609.24595)。
- 记得 Google 2025 年有“可微逻辑门元胞自动机”工作，本次未检索到，未核实。

### Wolfram

- 背景（经典，未逐条核实）：《一种新科学》（2002）用初等元胞自动机论证简单规则可产生任意复杂行为；Rule 30（看似随机）、Rule 110（图灵完备，Cook 证明）；“计算不可约性”——许多系统的行为没有捷径，只能一步步算出来；Wolfram 物理项目（2020，超图重写）与 ruliad。
- [What's Really Going On in Machine Learning? Some Minimal Models](https://writings.stephenwolfram.com/2024/08/whats-really-going-on-in-machine-learning-some-minimal-models/)（2024-08 文章，2025-10 出电子书，已核实原文前约 80%）：
  - 最小模型：每个神经元只接两个输入的“网格神经网络”仍能学会目标函数，缩得太小才失败；“规则阵列”——元胞自动机里每个时空位置可用不同规则（如 And/Xor），以及离散的极简“Transformer”。
  - 训练：离散规则阵列主要靠适应性进化（随机单点突变，损失不升就保留），不用梯度；另提出“变化图”与带链式法则的离散“布尔微积分”，其反向版本便宜、近似前向变化图。
  - 论点：训练并不构建有结构的机制，而是在计算宇宙中“采矿”，找到恰好给出正确结果的复杂计算块；计算不可约性提供丰富性，使突变搜索不易卡住；因此通常没有可讲成故事的解释，只有局部的“可约口袋”可被人理解；完全可理解的系统必然是可约的、能力受限。
  - 文中未讨论持续学习。
- [What Is ChatGPT Doing … and Why Does It Work?](https://writings.stephenwolfram.com/2023/02/what-is-chatgpt-doing-and-why-does-it-work/)（2023，未核实）：通俗解释大语言模型，认为语言的规律性比想象的“计算上更浅”，因此能被网络学到。

### 对本项目的启发（只作参考）

1. 不把主干做成 CA：TextNCA 在语言上不如同规模 Transformer，局部迭代本身无增益；与本项目网格 + 邻居消息的结果（E13、E30）一致。
2. 有效的是“局部到全局”的层级：窄窗口在下、宽窗口在上，与 [design/candidates/14-predictive-hierarchy.md](../../design/candidates/14-predictive-hierarchy.md) 的方向一致。
3. 便宜的可选预热：NCA 合成数据预预训练成本低，可作为两天训练前的对照组；对几千万参数的字节级模型是否有效没人测过。
4. Wolfram 的观点与苦涩教训同向：能力来自在大空间中搜索 / 训练出的、不必可解释的计算，而非手工设计的机制；他的最小模型也说明稀疏局部连接（每神经元 2 个输入）仍能学，但训练靠突变搜索，效率远低于反传，不适合本项目的语言规模。
5. 计算不可约性提示：完全按可解释模块搭的系统能力受限；评价涌现应靠测量（层级指标、探针），不靠模块命名。
