# 路由模块、稀疏专家与涌现分工

服务 [第 23 章 路由模块池](../../design/candidates/23-routed-pool.md)。本页只列文献与对项目的含义，采纳与排期只在 [plan/](../../plan/README.md)。“未核实”= 只看了搜索片段或二手来源。检索日期 2026-11-20。目录见 [README](../README.md)。

### 最近的先例：带状态的模块 + top-k

- [Recurrent Independent Mechanisms（RIMs）](https://arxiv.org/abs/1909.10893)（Goyal 等，ICLR 2021，未核实）：N 个独立的循环模块，按对输入的注意力选 top-k 更新，其余模块状态不变，模块间经注意力通信。**与第 23 章最像**：区别只在我们用时间尺度给路由定方向、地址醒时冻结、用 MoE 的稳态均衡。
- [Fast and Slow Learning of RIMs](https://arxiv.org/abs/2105.08710)（Madan 等，ICLR 2021，未核实）：模块参数学得快、注意力（路由）学得慢，迁移更好。支持“路由慢、模块快”，与 23.3 一致。

### 负载均衡与分工

- [Auxiliary-Loss-Free Load Balancing](https://arxiv.org/abs/2408.15664)（DeepSeek，2024，未核实）：每个专家一个偏置，按近期负载调；**偏置只影响选谁，混合权重用不带偏置的分数**；比辅助损失好，因为不引入干扰梯度。后续把它解释为原始-对偶法（[arXiv 2512.03915](https://arxiv.org/html/2512.03915v2)）。
- [OpenMoE](https://arxiv.org/abs/2402.01739)（Xue 等，2024，未核实）：专家主要按 token ID 分工（特殊字符、助动词），没看到语义层分工。
- [Probing Semantic Routing in Large MoE](https://arxiv.org/abs/2502.10928)（2025，未核实）：大模型（Mixtral、DeepSeek、Llama-4）的中间层有语义路由，随规模与深度增强。
- [The Expert Strikes Back](https://arxiv.org/abs/2604.02178)（2026，未核实）：专家多是细粒度“操作”专家（如闭合括号），既非领域专家也非单个 token。
- 含义：小规模下预计先出现字节 / token 级分工，所以 Q1 必须测“当前字节之外”的选择性，否则过关没有意义。

### 规则生成的连接

- [Hash Layers](https://arxiv.org/abs/2106.04426)（Roller 等，NeurIPS 2021，未核实）：按 token 固定哈希到专家，无路由参数、无均衡损失，与 Switch、BASE 等学出的路由持平或更好；平衡哈希、只看局部特征的最好。含义：第 23 章的学到地址必须赢过哈希对照才算有用。
- [Spatially embedded RNNs（seRNN）](https://www.nature.com/articles/s42256-023-00748-9)（Achterberg 等，Nature Machine Intelligence 2023，未核实）：连接代价正比于三维距离，网络收敛到模块化小世界、功能相近的单元在空间上聚集，类似灵长类皮层。后续：[arXiv 2409.17693](https://arxiv.org/abs/2409.17693)。含义：空间位置规则（23.8 的 D）有类脑依据，可测拓扑图涌现。

### 持续学习：路由漂移是主要遗忘来源

- [LLaVA-DyMoE](https://arxiv.org/abs/2603.27481)（2026，未核实）、[StaR-MoE](https://arxiv.org/pdf/2605.17571)（2026，未核实）、[LLaVA-CMoE](https://arxiv.org/html/2503.21227v1)（2025，未核实）：即使冻结旧专家，训练新部分时路由器的共享参数变了，旧输入被分去别的专家，照样遗忘。
- [Lifelong-MoE](https://arxiv.org/abs/2305.12281)（Chen 等，ICML 2023，未核实）：分布变化时加少量新专家，冻结旧专家与旧门控，加输出层正则，算力基本不变、旧知识保留。
- 含义：在线时不止地址要冻结，查询投影 W_r 也要冻结；“新内容分到空闲模块”（23.3 O2）有 Lifelong-MoE 式先例。

### 放大规律与粒度

- [Unified Scaling Laws for Routed LMs](https://arxiv.org/abs/2202.01169)（Clark 等，ICML 2022，未核实）：参数量与每输入算力是两条独立的轴，各自都改善；专家数的收益与模型规模是双线性关系，越大收益越小；给出“有效参数量”。
- [Scaling Laws for Fine-Grained MoE](https://arxiv.org/abs/2402.07871)（Krajewski 等，ICML 2024，未核实）：专家做得比 FFN 小（粒度高）几乎在所有预算下更好，预算越大越明显；MoE 与稠密的差距随规模扩大。
- [Mixture of A Million Experts（PEER）](https://arxiv.org/abs/2407.04153)（He，DeepMind 2024，未核实）：百万个单神经元专家，用 product key 检索 top-k，复杂度约 √N；同算力下困惑度优于稠密 FFN 与粗粒度 MoE。即“连接由规则算出、不存索引”在大 N 下的现成做法。
- 含义：“很多小模块”的方向有放大规律支持；N 到上千时换 product key 路由。

### 与循环 / SSM 主干结合

- [MoE-Mamba](https://arxiv.org/abs/2401.04081)（2024，未核实）：Mamba + MoE 前馈层，约 2.2 倍少的步数达到同等效果。
- [BlackMamba](https://arxiv.org/abs/2402.01771)（Zyphra，2024，未核实）：Mamba + 路由 MLP，340M/1.5B、630M/2.8B，训练算力更省。
- 含义：这两者的专家是**无状态**前馈层，循环部分仍稠密。第 23 章路由的是**带状态**的模块，风险更大；应加“只路由 MLP”的对照组，分清收益来自哪里。

### 层级与深度

- [Emergence of Functional Hierarchy in a Multiple Timescale Network（MTRNN）](https://www.oist.jp/sites/default/files/2024-03/cnru_PLoSCB_2008_corrected_0.pdf)（Yamashita、Tani，PLoS CB 2008，未核实）：只靠快、慢两类神经元的时间常数差，不分模块，就自组织出功能层级。支持 R2 用时间尺度定方向。
- [Mixture-of-Depths](https://arxiv.org/abs/2404.02258)（Raposo 等，DeepMind 2024，未核实）：每块 top-k 选 token 计算，其余走残差跳过；k 固定所以计算图静态；可与 MoE 叠加。含义：R2 的每跳加一个“跳过”选项，深度可变而计算图不变。

### 实现

- [MegaBlocks](https://arxiv.org/abs/2211.15841)（Gale 等，MLSys 2023，未核实）：把 MoE 写成块稀疏矩阵乘，不丢 token、不补齐；分组矩阵乘是推荐路径。第 23 章 P1–P4 过后的提效版照此写。
