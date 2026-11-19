# 新型序列架构

“已核实”= 代理打开原文或 arXiv 摘要页确认；“未核实”= 只看了搜索片段或二手来源。检索日期 2026-11-19。目录与总结论见 [README](../README.md)。

已在 [fast-memory.md](fast-memory.md)（KDA、GDN、RWKV-7、xLSTM Scaling Laws、LFM2）和 [hierarchy-topdown.md](hierarchy-topdown.md)（H-Net、BLT）写过的不重复。

### SSM

- [Mamba](https://arxiv.org/abs/2312.00752)（Gu、Dao，2023-12，已核实）：输入相关的选择性对角 SSM + 硬件感知扫描。字节级见下文 MambaByte。含义：我们的门控对角递归就是这一族。
- [Mamba-2 / SSD](https://arxiv.org/abs/2405.21060)（Dao、Gu，ICML 2024，已核实）：每头标量衰减，SSM 与线性注意力统一为半可分矩阵，比 Mamba 快 2–8 倍。含义：标量衰减换速度，表达力不升。
- [Mamba-3](https://arxiv.org/abs/2603.15569)（Lahoti、Li … Dao、Gu，ICLR 2026，已核实）：三处改动：指数-梯形离散化、复数状态转移（等价于输入相关 RoPE，奇偶、模运算等状态追踪任务近乎满分）、MIMO。1.5B 平均下游比 GDN 高 0.6 点，MIMO 再加 1.2 点；状态减半即可达到 Mamba-2 的困惑度。另有 180M、440M 档（未核实数字）。含义：复数旋转和梯形离散化能直接移植到对角递归，参数几乎不增加。
- [LRU](https://arxiv.org/abs/2303.06349)（Orvieto 等，DeepMind，2023-03，已核实）：线性化 + 对角化 + 稳定参数化与初始化 + 前向归一化的普通 RNN 即可匹配深层 SSM（LRA）。S5（Smith 等，ICLR 2023，未核实）是 MIMO 对角 SSM + 并行扫描。含义：初始化、归一化比 SSM 理论更重要。
- [Hyena](https://arxiv.org/abs/2302.10866)（Poli 等，2023-02，已核实）：隐式长卷积 + 数据门控，2K 长度下训练算力省 20%。[StripedHyena 7B](https://huggingface.co/togethercomputer/StripedHyena-Nous-7B)（Together，2023-12，未核实）：Hyena + GQA 混合。含义：纯卷积族召回弱（见下文 Zoology），不考虑。
- Liquid S4（Hasani 等，2022，未核实）属 2023 前；后继 LFM2 见 fast-memory。

### 线性注意力 / RNN 复兴

- [RetNet](https://arxiv.org/abs/2307.08621)（微软，2023-07，已核实）：固定衰减 retention，可并行 / 递归 / 分块三种算法。被 GLA、GDN 超越。
- [GLA](https://arxiv.org/abs/2312.06635)（Yang 等，2023-12，已核实）：数据相关门控线性注意力 + FlashLinearAttention；与 LLaMA、RetNet、Mamba 相当，训练长度 2K 外推到 20K+。
- [HGRN2](https://arxiv.org/abs/2404.07904)（Qin 等，COLM 2024，已核实）：外积状态扩展，不加参数而扩大状态，具线性注意力形式。
- [Griffin / Hawk](https://arxiv.org/abs/2402.19427)（DeepMind，2024-02，已核实）：RG-LRU 门控线性递归（Hawk）+ 局部注意力（Griffin）；Hawk 下游超过 Mamba，Griffin 用 1/6 token 追平 Llama-2，至 14B。RecurrentGemma 为其产品化（未核实）。含义：RG-LRU ≈ 我们的门控对角递归；局部注意力是最便宜的召回补丁。
- [minGRU / minLSTM](https://arxiv.org/abs/2410.01201)（Feng、…、Bengio，2024-10，已核实）：门只依赖输入，可并行扫描。Shakespeare 字符级、3 层 d=384：测试损失 Transformer 1.547、minGRU 1.548、minLSTM 1.555、Mamba 1.575；Transformer 多需约 2.5 倍步数且过拟合。含义：小规模字符级上各架构差距 ≤0.03 nats，结构不是瓶颈。
- [RWKV-5/6（Eagle/Finch）](https://arxiv.org/abs/2404.05892)（Peng 等，2024-04，已核实）：矩阵状态 + 动态递归，0.46B–7.5B。RWKV-7 见 fast-memory。
- [xLSTM](https://arxiv.org/abs/2405.04517)（Beck … Hochreiter，2024-05，已核实）：sLSTM 标量记忆 + 记忆混合（不可并行，可做状态追踪）；mLSTM 矩阵记忆 + 协方差更新（可并行）；均为指数门。含义：mLSTM ≈ 我们的快权重；sLSTM 的非线性递归是可选状态追踪补丁。
- [MesaNet](https://arxiv.org/abs/2506.05233)（von Oswald … Sacramento，ICLR 2026，已核实）：每步用共轭梯度把上下文回归损失解到最优，可分块并行；困惑度和长上下文好于以前的 RNN，推理 FLOPs 更高，至 1B 级。含义：delta 规则的“多步解”上限，对我们太贵。
- [Comba](https://arxiv.org/abs/2506.02475)（Hu … Sun，2025-06，已核实）：“双线性 RNN”（GDN、TTT、RWKV-7）加闭环控制：标量加低秩转移 + 状态反馈与输出反馈修正；340M、1.3B。摘要无数字。
- [PaTH](https://arxiv.org/abs/2505.16381)（Yang、…、Kim，NeurIPS 2025，已核实）：用累积的数据相关 Householder 变换代替 RoPE，优于 RoPE。只对注意力有意义。
- [Forgetting Transformer（FoX）](https://arxiv.org/abs/2503.02130)（Lin … Courville，ICLR 2025，已核实）：softmax 打分加数据相关遗忘门，无需位置编码，优于 Transformer 并保留长上下文优势；“Pro 块”（借递归模型的组件）对 FoX 和 Transformer 都显著有益。
- [TTT 层](https://arxiv.org/abs/2407.04620)（Sun 等，2024-07，已核实）：隐状态本身是线性模型或 MLP，用自监督梯度步更新；125M–1.3B，16K 后仍随上下文下降（Mamba 不再下降）。[Titans](https://arxiv.org/abs/2501.00663)（Google，2024-12，已核实）、[ATLAS](https://arxiv.org/abs/2505.23735)（2025-05，已核实）：神经长期记忆，ATLAS 按窗口内多 token 优化记忆，BABILong 10M 长度约 80%。含义：都是 delta 规则的推广；我们的序列太短，用不上。

### 混合架构

- [Jamba](https://arxiv.org/abs/2403.19887)（AI21，2024-03，已核实）：Transformer / Mamba 层交错 + MoE，256K 上下文。
- [Zamba](https://arxiv.org/abs/2405.16712)（Zyphra，2024-05，已核实）：Mamba 主干 + 单个全局共享注意力块，7B。含义：共享一个模块到处复用，和我们的全局中枢思路同构。
- [Samba](https://arxiv.org/abs/2406.07522)（微软，ICLR 2025，已核实）：逐层 Mamba + 滑窗注意力，3.8B；4K 训练外推到 256K 密钥检索。
- [Hymba](https://arxiv.org/abs/2411.13676)（NVIDIA，2024-11，已核实）：同层并行的注意力头与 SSM 头 + 可学习 meta tokens + 跨层 KV 共享；1.5B 超过 Llama-3.2-3B。meta tokens ≈ 我们的工作空间槽。
- [Nemotron-H](https://arxiv.org/abs/2504.03624)（NVIDIA，2025-04，已核实）：绝大多数注意力层换成 Mamba-2，8B / 56B，推理快 3 倍。
- 配比研究：[Mamba-Transformer 混合分析](https://arxiv.org/abs/2510.26912)（2025-10，已核实）：串行混合短上下文好，并行混合长上下文好，规模未写。3:1–6:1 推荐见 fast-memory。≤50M 的配比扫描未找到。

### 递归 / 深度自适应 / 推理

- [HRM](https://arxiv.org/abs/2506.21734)（Wang 等，Sapient，2025-06，已核实）：慢 H 模块 + 快 L 模块（L 跑几步 H 更新一次），27M、每任务约 1000 样本，无预训练；1 步梯度近似（不做 BPTT）+ 深监督（外层循环每段都算损失）+ ACT。数字（按 TRM 表 4）：Sudoku-Extreme 55.0%、Maze-Hard 74.5%、自报 ARC-AGI-1 40.3%。
- [ARC Prize 的 HRM 分析](https://arcprize.org/blog/hrm-analysis)（2025-08，已核实）：半私有集复现 32%（ARC-AGI-2 2%）。①同参数普通 Transformer 未调参也只差约 5 点，1 次外循环时二者持平，改变 H/L 步数都更差——**层级本身贡献很小**；②外层精炼循环是主因：1→2 次 +13 点，1→8 次约翻倍；用 16 次训练、1 次推理也比 1 次训练高 15 点以上（训练时的精炼比推理时重要）；③ACT 与固定 16 次只差几点；④只用 400 个评测任务训练仍有 31%（全量 41%），本质是“零预训练的测试时训练”；⑤增广 300 份已近饱和；puzzle_id 嵌入必不可少。
- [TRM](https://arxiv.org/abs/2510.04871)（Jolicoeur-Martineau，Samsung SAIL，2025-10，已核实）：单个 2 层网络递归更新潜状态 z 和答案 y；7M 在 ARC-AGI-1 45%、ARC-AGI-2 8%。Sudoku-Extreme 消融（表 1）：TRM 87.4%（5M）；1 步梯度 56.5%（**完整反传穿过一次递归是最大因素**）；分开 f_H、f_L 82.4%（10M）；去 EMA 79.9%；4 层 n=3 79.5%；自注意力代替 MLP 74.7%；T=2、n=2 73.7%；ACT 86.1%。作者：加层伤泛化，2 层 + 多次递归最好，猜测是小数据下过拟合，无理论。
- 后续检验：[TRM on ARC-AGI-1](https://arxiv.org/abs/2512.11847)（2025-12，已核实）：换掉 puzzle ID 正确率归零；多数正确率在第 1 次递归就达到，几次后饱和；1000 样本投票 +11 点——结论是任务条件化 + 测试时算力，而非深度推理。[Tiny Autoregressive Recursive Models](https://arxiv.org/abs/2603.08082)（van der Schaar 组，ICLR 2026 workshop，已核实）：从 Transformer 逐级变到自回归 TRM，字符级算法任务、算力对齐：完整 TRM 无可靠收益、常大幅变差，部分两级精炼基线很强。[HRM-LM](https://arxiv.org/abs/2604.14442)（单作者，2026-04，已核实）：OpenWebText、约 1.2B，快慢两速共享权重远胜扁平共享的 UniTF（CE 4.18 vs 约 7.6），对独立层堆叠则随调参而定（同超参输 0.20 nats，各自搜参赢 0.23 nats，单种子）。
- [Looped Transformer](https://arxiv.org/abs/2502.17416)（Saunshi 等，Google，ICLR 2025，已核实）：k 层循环 L 次在加法、p-hop 归纳上接近 kL 层，明显好于 k 层；LM 推理任务上有竞争力；循环正则可兼顾记忆与推理。
- [Mixture-of-Recursions](https://arxiv.org/abs/2507.10524)（2025-07，已核实）：共享层栈 + 路由器为每个 token 选递归深度；135M–1.7B 同 FLOPs 帕累托更优，167M 胜 315M 基线。
- [Huginn](https://arxiv.org/abs/2502.05171)（Geiping … Goldstein，2025-02，已核实）：3.5B、800B token，递归块推理时可加到等效 50B 参数的计算。只有大规模。
- [Energy-Based Transformers](https://arxiv.org/abs/2507.02092)（Gladstone … Du，2025-07，已核实）：按能量梯度下降做预测，自报扩展速率最高快 35%、推理期加算力收益多 29%；无小规模字节结果，训练代价高。
- 含义：HRM/TRM 起作用的是“同一小网络反复迭代 + 完整反传 + 每轮都监督 + 增广”，不是“慢模块管快模块”；TRM 里合并 f_H、f_L 反而更好。E38/E40 把参数分给新层级会失败，与此一致。可试的是不加参数的权重共享迭代，见文末结论。

### 小规模有用的注意力变体

- [Canon 层](https://arxiv.org/abs/2512.17351)（Allen-Zhu，Physics of LMs 4.1，NeurIPS 2025，已核实）：在块内插入对相邻 token 表示的加权和（即因果短卷积）；合成预训练任务上推理深度约 2 倍，把 NoPE 拉到 RoPE 水平、线性注意力拉近 Mamba2 / GDN；强调学术规模（1.3B / 100B token）下架构差异常被种子噪声（2–4%）淹没。含义：与 KDA 的 q/k/v 短卷积是同一结论，进模块。
- [Differential Transformer](https://arxiv.org/abs/2410.05258)（微软，ICLR 2025，已核实）：两张 softmax 图相减消噪，各规模优于 Transformer。[nGPT](https://arxiv.org/abs/2410.01131)（NVIDIA，2024-10，已核实）：全部向量单位范数、表示在超球面上移动，达到同精度步数少 4–20 倍。[Stick-breaking](https://arxiv.org/abs/2410.17980)（IBM / Mila，2024-10，已核实）：折棍分配权重，自带近因偏置、不需 RoPE，2^11 训练外推 2^14。[NSA](https://arxiv.org/abs/2502.11089)（DeepSeek，2025-02，已核实）：压缩 + 选择 + 滑窗三路稀疏注意力，长上下文提速。含义：我们无 softmax 注意力，只有 nGPT 的归一化思想可借，其收益数字来自大模型。
- [多 token 预测](https://arxiv.org/abs/2404.19737)（Meta，2024-04，已核实）：共享主干 + n 个输出头；摘要称“模型越大越有用”，小模型收益小（论文内小模型变差的具体数字未核实）。EvaByte 用 8 头多字节预测主要为提速。含义：0.36M 档不做，或只当小权重辅助损失。

### 字节级架构（H-Net / BLT 之外）

- [MambaByte](https://arxiv.org/abs/2401.13660)（Wang、…、Rush，COLM 2024，已核实）：纯字节 Mamba 不分块。bits/byte（PG19 / Stories / Books / Code）：MambaByte 353M 0.930 / 0.908 / 0.966 / 0.396；MegaByte 758M+262M 1.000 / 0.978 / 1.007 / 0.411；字节 Transformer 320M 1.057 / 1.064 / 1.097 / 0.575。最小档 248M，无 ≤50M 结果。
- [SpaceByte](https://arxiv.org/abs/2404.14408)（Slagle，NeurIPS 2024，摘要已核实、数字未核实）：在空格等边界字节后插入大 Transformer 块；同算力优于其他字节架构，追平子词模型。二手数字 PG19：SpaceByte 1.009、MegaByte 1.083、字节 Transformer 1.138 bpb。
- [EvaByte](https://hkunlp.github.io/blog/2025/evabyte)（HKU / SambaNova，2025-01，已核实）：6.5B，EVA 线性化注意力 + 8 头多字节预测，1.5T 字节；训练数据少 5 倍仍可比子词模型，解码比普通字节模型快 5–10 倍。
- [Bolmo](https://arxiv.org/abs/2512.15586)（Ai2，2025-12，已核实）：把 OLMo 子词模型两阶段“字节化”为 1B / 7B 字节模型（架构细节未核实）。只有大规模。
- 含义：字节级的正面结果都在 ≥250M；小规模见下节与 hierarchy-topdown 里的 H-Net 复现。

### 极小规模的头对头对比（≤10M，字符 / 字节级）

这一节最重要，但对照严格的研究很少：
- [Baseline Shape Decides the Verdict](https://arxiv.org/abs/2609.29397)（单作者，2026-09，已核实摘要）：字节级、60K 参数、3 种子、98 次运行。①16M 字节预算下，不同深宽形状的 Transformer 验证损失相差 22.6%，最佳形状追平原文“路由三值块”，原文 22% 优势部分来自基线形状；②130M 字节时路由块（卷积 + 对角 SSM + 稀疏注意力）胜三种 Transformer 22–24%，**单纯门控对角 SSM 块再好 9.1%**，路由器也主要选递归通路。注：Transformer 有占 11–22% 参数的位置嵌入，未调学习率。
- minGRU 论文 Shakespeare 字符级（见上文）：Transformer、minGRU、minLSTM、Mamba 只差 0.03 nats。
- H-Net 社区复现（hierarchy-topdown）：TinyStories、0.1–0.4M，纯字节 Mamba 1.451 bpb 胜空格分块和动态分块。
- [Zoology](https://arxiv.org/abs/2312.04927)（Arora … Ré，2023-12，已核实）：17 个模型，门控卷积与注意力的困惑度差（Pile 上最多 2.1）有 82% 来自关联召回；70M 注意力在召回上胜 1.4B 门控卷积；加输入相关稀疏注意力补回 97.4%。[MAD](https://arxiv.org/abs/2403.17844)（Poli 等，ICML 2024，未核实正文）：几分钟的小合成任务（召回、记忆、压缩）可预测 70M–7B 的扩展排名。
- [Dynamical Memory Transformer](https://arxiv.org/abs/2610.41850)（Bendi-Ouis、Hinaut，2026-10，已核实摘要）：复对角递归记忆 + 注意力读出；语言建模 Mamba 第一，规模摘要未写（搜索片段称约 10M、TinyStories，未核实）。
- 未找到 ≤10M 下 GDN / Mamba / RWKV / LSTM / Transformer 同预算、多种子、调参的系统对比。现有证据一致指向：极小规模下门控对角递归不输任何架构，差距多在噪声内，基线形状和超参影响大于架构。

### 对项目的结论

1. **保留模块主干，不换架构**（不用重训）。没有哪族在 ≤10M 字节级稳定胜过门控对角递归 + delta 快权重；60K 字节研究里门控对角 SSM 最好，字符级各架构差 ≤0.03 nats。Mamba-3、MesaNet、Comba、TTT 的收益都在 ≥180M。
2. **先把对照做严**（不用重训）。按 60K 研究：基线要扫深宽形状、调学习率、≥3 种子；加 MQAR 式关联召回和 MAD 合成探针，因为小规模困惑度差主要来自召回（Zoology）。
3. **借两处小改进模块**（要重训，S 档对照）：a) q/k/v 和递归输入前加因果短卷积（Canon 层、KDA，fast-memory 第 7 条已列）；b) 对角递归改用 Mamba-3 的复数旋转（输入相关 RoPE）+ 梯形离散化，几乎不加参数，专补状态追踪。
4. **层级换成权重共享的迭代**（要重训）。HRM/TRM 的消融说明有效的是同一小网络反复迭代、完整反传穿过迭代、每轮都监督，而“慢模块管快模块”贡献很小（TRM 合并两网络更好，ARC Prize 改 H/L 步数更差）。E38/E40 把参数挪给新层级，正好走反了。可试：同一模块网格每字节跑 2 遍（或每 T 字节多跑 1 遍），参数不变、算力约 2 倍，对比同算力加宽。
5. **对第 4 条不要期望太高**：TRM 的收益来自 1000 样本谜题加强增广（作者自认防过拟合），ARC 收益主要靠 puzzle ID 和测试时投票；自回归 TRM 在字符级任务上无可靠收益（2603.08082）；HRM-LM 只在 1.2B 对扁平共享明确胜出。语言建模数据充足，迭代多半只值“同算力加宽”的量，排在第 3 条之后。
6. **召回不足时再加局部注意力**（要重训，低优先）：Griffin / Samba 式一个小滑窗注意力头，或把工作空间槽当 Hymba 的 meta tokens。只在召回探针显示 delta 记忆不够时做。
7. **多 token 预测、nGPT、注意力变体、EBT 暂不做**（不用重训）：收益随规模增长或只针对 softmax 注意力，0.36M 档预期为零或负。
