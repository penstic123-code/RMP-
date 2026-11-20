# 路由模块、稀疏专家与涌现分工

服务 [第 23 章 路由模块池](../../design/candidates/23-routed-pool.md)。本页只列文献与对项目的含义，采纳与排期只在 [plan/](../../plan/README.md)。“未核实”= 只看了搜索片段或二手来源。检索日期 2026-11-20。目录见 [README](../README.md)。

### 最近的先例：带状态的模块 + top-k

- [MoM: Mixture-of-Memories](https://arxiv.org/abs/2502.13685)（Du、Sun 等，ICLR 2026，已核实全文）：**与第 23 章最像**。一层内多个独立记忆状态，每个 token 选 top-k，未选的记忆“不激活、不变”；主实验每个记忆用 Gated DeltaNet；训练时按路由重排 token、每个记忆打包成变长序列跑现成内核再散回加权。缺：只有 4 个记忆选 2 个、一层内；无慢通道与局部 MLP 组成的模块；线性路由器 + Switch 辅助损失；另有一个常开的共享记忆；无睡眠冻结、秩与层级、自上而下。含义：23.2 的“未选状态不变”和 23.7 的“变长打包精确扫描”是 MoM 已有的，引用它，不算新。
- [ThalNet](https://arxiv.org/abs/1706.05744)（Hafner 等，NIPS 2017，已核实摘要）：循环模块经路由中心读写，层级、跳连、反馈从学习中涌现；但稠密、逐步 RNN、无稀疏激活。对应 23.2 的 R2。
- [BRIMs](https://arxiv.org/abs/2006.16981)（Mittal 等，ICML 2020，未核实）：RIMs 加自上而下信号的注意力合并。对应 R3。
- [Routers Learn the Geometry of Their Experts](https://arxiv.org/abs/2605.12476)（2026，已核实摘要）：无参数路由，每个专家保留路由到它的隐状态的滑动平均作质心、按余弦分配。即 23.3 的地址竞争学习，用在无状态 FFN 专家上。
- [MoxE](https://arxiv.org/abs/2505.01459)（2025）：top-k 路由到 xLSTM 循环专家，未说明专家状态是否跨时间保留。[Swimba](https://arxiv.org/abs/2603.06938)（2026）：把“每个专家各有一条 SSM 状态”列为一类设计，自己选了反方向。

- [Recurrent Independent Mechanisms（RIMs）](https://arxiv.org/abs/1909.10893)（Goyal 等，ICLR 2021，未核实）：N 个独立的循环模块，按对输入的注意力选 top-k 更新，其余模块状态不变，模块间经注意力通信。**与第 23 章最像**：区别只在我们用时间尺度给路由定方向、地址醒时冻结、用 MoE 的稳态均衡。
- [Fast and Slow Learning of RIMs](https://arxiv.org/abs/2105.08710)（Madan 等，ICLR 2021，未核实）：模块参数学得快、注意力（路由）学得慢，迁移更好。支持“路由慢、模块快”，与 23.3 一致。

### 负载均衡与分工

- [Auxiliary-Loss-Free Load Balancing](https://arxiv.org/abs/2408.15664)（DeepSeek，2024，未核实）：每个专家一个偏置，按近期负载调；**偏置只影响选谁，混合权重用不带偏置的分数**；比辅助损失好，因为不引入干扰梯度。后续把它解释为原始-对偶法（[arXiv 2512.03915](https://arxiv.org/html/2512.03915v2)）。
- [OpenMoE](https://arxiv.org/abs/2402.01739)（Xue 等，2024，未核实）：专家主要按 token ID 分工（特殊字符、助动词），没看到语义层分工。
- [Probing Semantic Routing in Large MoE](https://arxiv.org/abs/2502.10928)（2025，未核实）：大模型（Mixtral、DeepSeek、Llama-4）的中间层有语义路由，随规模与深度增强。
- [The Expert Strikes Back](https://arxiv.org/abs/2604.02178)（2026，未核实）：专家多是细粒度“操作”专家（如闭合括号），既非领域专家也非单个 token。
- 含义：小规模下预计先出现字节 / token 级分工，所以 Q1 必须测“当前字节之外”的选择性，否则过关没有意义。

- 查重结论（2026-11-20）：没有找到与第 23 章完全相同的设计。看起来新的是组合：地址规则生成连接 + 疲劳项用在带状态模块池；醒时冻结路由、睡眠才更新、新内容导向空闲模块；学到的秩分波、深度从一波长起；按块延迟的自上而下反馈。“新”只是没搜到。
- 哪些已被证明能做大（2026-11-21）：带状态模块的稀疏选择已做到上亿参数（MoM）；状态层 + 稀疏专家到数 B（BlackMamba、Jamba，专家无状态）；百万小专家检索（PEER）。新内容加新专家、冻结旧门有持续学习先例（Lifelong-MoE，需任务边界）。没人在规模上证明的：地址规则连接、层级涌现、醒时冻结 + 睡眠整理、无边界的新奇分配。小模型只按字节分工（E67、E77）与“语义路由要到大模型中间层才出现”（Probing Semantic Routing）一致。

### 规则生成的连接

- [Hash Layers](https://arxiv.org/abs/2106.04426)（Roller 等，NeurIPS 2021，未核实）：按 token 固定哈希到专家，无路由参数、无均衡损失，与 Switch、BASE 等学出的路由持平或更好；平衡哈希、只看局部特征的最好。含义：第 23 章的学到地址必须赢过哈希对照才算有用。
- [Spatially embedded RNNs（seRNN）](https://www.nature.com/articles/s42256-023-00748-9)（Achterberg 等，Nature Machine Intelligence 2023，未核实）：连接代价正比于三维距离，网络收敛到模块化小世界、功能相近的单元在空间上聚集，类似灵长类皮层。后续：[arXiv 2409.17693](https://arxiv.org/abs/2409.17693)。含义：空间位置规则（23.8 的 D）有类脑依据，可测拓扑图涌现。

### 几何生成连接与分工条件（2026-11-21 用户提出，均未核实，只看了摘要）

- [Resonant Sparse Geometry Networks](https://arxiv.org/abs/2601.18064)（Hays，2026 预印本）：节点嵌在学到的双曲空间，连接强度随测地距离衰减，按输入动态稀疏（二手总结约 2% 节点激活）；快：梯度传激活，慢：类 Hebbian 改结构。规模很小（4 万参数、合成任务），无独立复现。含义：双曲空间天然编码层级（离中心的远近 = 层级），可试“双曲地址”替代 23.9 的秩 + 分波。
- [Dynamics of specialization in neural modules under resource constraints](https://arxiv.org/abs/2106.02626)（Béna、Goodman，Nature Communications）：结构模块化不保证功能分工；分工只在特征可分且资源受限（模块间连接稀疏、带宽窄）时出现，且随训练过程变化。含义：E67 的分工可能来自 k 选 N 与窄残差流；可扫 D、k 检验，并在训练中途的检查点上跑 rmp_emergence 看分工的动态。
- [Training Neural Networks by Optimizing Neuron Positions](https://arxiv.org/abs/2506.13410)（Erb 等，2025）：神经元坐标是训练参数，权重由距离导出；MNIST 与常规网络相当，脉冲网络也验证。后续 [Space as Time](https://arxiv.org/abs/2511.01632) 用距离定突触延迟。难点：动一个神经元会改它所有的距离（与 E69 改一个地址劫持路由同类）。含义：23.8 规则 D 的更彻底版本；可比“地址用梯度学”与现行竞争学习。

### 结构生长与生成式连接的经典方法（2026-11-21 用户提出；凭已有知识整理，未重新核对原文）

- **HyperNEAT**（Stanley 等 2009）：神经元有坐标，小网络 CPPN 由两端坐标生成连接权重（间接编码，权重不单独存）。“连接由规则生成”的源头；可作长期方向：由地址生成模块间连接乃至模块权重。
- **ES-HyperNEAT**（Risi、Stanley 2012）：还决定神经元放在哪——在 CPPN 输出变化大（信息密）的区域按四叉树细分放更多隐藏神经元。对应“新模块放哪”。
- **Growing Neural Gas**（Fritzke 1995）：节点在输入空间竞争学习，在累积误差最大的节点与其误差最大的邻居之间插入新节点、误差重分配，老边老节点删除。与 23.3 的地址竞争学习几乎同构；其插入规则启发了新奇分配第四版“分裂”（复制父模块、地址取中点），对症 E69、E71 的失败原因。
- **NISPA**（Gurbuz、Dovrolis，ICML 2022）：稀疏网络持续学习，按重要性把单元分为稳定（冻结）与可塑，可塑单元的连接删减再长。对应“在线只改部分模块”，可按重要性选可改模块。
- [SC-ML：Sparsely-Connected Recurrent Motif Layer](https://arxiv.org/abs/2108.01793)（循环脉冲网络，未核实）：一组组循环连接的神经元（基序）内部稠密、基序间稀疏连接，基序拓扑用风险缓解的架构搜索优化。即 RMP 起点讨论的“模块内稠密、模块间稀疏”；其“搜索基序拓扑”对应 RMP 的地址规则与深度生长。
- [Dynamos：Dynamically Modular and Sparse General Continual Learning](https://arxiv.org/abs/2301.00620)（2023，未核实）：动态模块化与稀疏用于带回放的持续学习。对应 P3。
- **RigL**（Evci 等，ICML 2020）：固定稀疏度的动态稀疏训练，定期删最小权重、在梯度最大处长新连接。可借“按梯度决定往哪长”指导移波与地址更新。

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

### 深度生长

- [On the Inductive Bias of Stacking（MIDAS）](https://arxiv.org/abs/2409.19044)（Saunshi 等，NeurIPS 2024，未核实）：逐步堆叠加深，复制小模型中间块初始化大模型；训练快约 40%，困惑度持平，推理类下游更好；1B–8B 验证。
- [Curriculum-Guided Layer Scaling](https://arxiv.org/abs/2506.11389)（2025，未核实）：加深与数据由易到难同步，泛化与零样本更好。
- 含义：第 23 章深度从 1 波长起，既为层级涌现，也让前期训练更快；区别是按地址匹配自己往后挪，而非按时间表复制层。

### 实现

- [MegaBlocks](https://arxiv.org/abs/2211.15841)（Gale 等，MLSys 2023，未核实）：把 MoE 写成块稀疏矩阵乘，不丢 token、不补齐；分组矩阵乘是推荐路径。第 23 章 P1–P4 过后的提效版照此写。
