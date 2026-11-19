# 认知理论与自我改进（哥德尔机、侯世达、丹尼特、全局工作空间）

“已核实”= 代理打开 arXiv 摘要页确认；“未核实”= 只看了搜索片段或二手来源。检索日期 2026-11-19。目录与总结论见 [README](../README.md)。

### 哥德尔机与自我改进

- 哥德尔机（Schmidhuber 2003，cs/0309048，未核实）：只有证明改写后期望收益更高才改写自己。后来的变体都把“证明”换成“实测”。睡眠后的接受 / 回滚就是这个思想的实用版。
- [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954)（Zhang、Clune 等，Sakana，ICLR 2026，已核实）：代理改写自己的代码，每次改动用基准测试检验；保留所有变体的档案，而不是只留当前最好的一个。SWE-bench 20.0%→50.0%。含义：保留历次被接受的检查点档案，不只留一个回滚点。
- [Huxley-Gödel Machine](https://arxiv.org/abs/2510.21614)（KAUST / Schmidhuber，ICLR 2026，已核实）：指出“当前得分不能预测后代表现”，改用后代表现给变体打分。含义：睡眠只看当次探针是短视的，应记录每次接受之后的睡眠是否持续过关。
- [Gödel Agent](https://arxiv.org/abs/2410.04444)（ACL 2025，未核实）、[AlphaEvolve](https://arxiv.org/abs/2506.13131)（DeepMind 2025，未核实）：靠实测反馈做自改 / 进化搜索；AlphaEvolve 的上限在自动评估器的质量。含义：我们的探针集就是评估器，探针好坏决定睡眠能学到什么。
- 自指权重矩阵 SRWM（Irie、Schlag、Csordás、Schmidhuber，ICML 2022，arXiv 2202.05780，代码 IDSIA/modern-srwm，未核实）：一个矩阵自己生成键、查询和学习率，并按 delta 规则改写自己。含义：快记忆的直接升级——由快权重自己产生写入键和 β。
- [Metalearning Continual Learning Algorithms](https://arxiv.org/abs/2312.00276)（Irie 等，TMLR 2025，已核实）：元训练 SRWM，使其上下文内学习不遗忘旧任务；不用回放在 Split-MNIST 上胜过手工方法。含义：训练时可加“上下文内连续写入后再考早先内容”的目标。
- [Nested Learning / Hope](https://arxiv.org/abs/2512.24695)（Behrouz 等，Google，NeurIPS 2025，已核实）：把模型看成更新速度不同的多级嵌套优化，含自修改模块与“连续谱记忆”；摘要无规模数字。与本项目三种速度最接近，支持在“每拍”和“睡眠”之间再加中间速度的一级。
- [SEAL](https://arxiv.org/abs/2506.10943)（MIT，NeurIPS 2025，已核实）：模型自己写微调数据（自我编辑），用更新后模型的表现作 RL 奖励；主要问题是灾难性遗忘。含义：睡眠时让模型自己改写 / 复述当天内容进回放池，由锚定和探针把关。
- Celo / Celo2 学习型优化器（2501.12670、2602.19142，未核实）：Celo2 元训练约 4.5 GPU 小时，可稳定用到 1.3B。含义：以后可试作睡眠阶段的优化器，不急。

#### 2026 年递归自我改进（RSI）新动态（检索日期 2026-11-20，均未核实，只看了搜索摘要；只作参考）

- [ICLR 2026 RSI 专题研讨会](https://recursive-workshop.github.io/)（[ICLR 页面](https://iclr.cc/virtual/2026/workshop/10000796)，2026-04-26，里约）：主题含经验学习、合成数据流水线、多模态代理、弱到强泛化、推理时扩展。
- AutoHarness（DeepMind 投稿该研讨会，[OpenReview](https://openreview.net/pdf?id=g9rEYVNn5T)）：Gemini-2.5-Flash 自己写代码策略后，在 16 个 TextArena 游戏中对 Gemini-2.5-Pro 赢 9 个（胜率 56.3%）。
- [Sakana 多代理自我监督 RSI](https://sakana.ai/mass/)（arXiv 2610.12176）：两轮循环后，四个研究基准上每输出 token 得分达基础模型的 1.2–1.6 倍。
- [RSI 综述](https://arxiv.org/abs/2607.07663)：归类 1250 篇论文，区分有界自我精炼与开放式递归自我改进；结论是目前多为人在环路上，完全自主闭环尚未实现。
- [Recursive Criticality of AI Self-Improvement](https://arxiv.org/pdf/2609.00137)：自我改进“临界性”的理论文章，未细读。
- 含义：本项目规模的模型不能改进自己的架构。可借的三层：模型内由外层训练学出内层学习规则（[design/candidates/14-predictive-hierarchy.md](../../design/candidates/14-predictive-hierarchy.md) 14.6）；睡眠的接受 / 回滚即有界自我改进（加检查点档案与自我复述，见上文 DGM、HGM、SEAL）；研究流程中由外部代理提出改动、训练、读结果，上限取决于评估指标（bpc、记忆探针、层级指标）的质量与每轮评估耗时。

### 侯世达

- [Lewis & Mitchell, Evaluating the Robustness of Analogical Reasoning in LLMs](https://arxiv.org/abs/2411.14215)（TMLR 2025，已核实）：在反事实字母串（如打乱字母表）上 GPT 准确率大跌，人类不受影响。含义：用反事实探针区分真抽象和死记。
- Copycat / FARG（slipnet、codelet、温度）：2024–2026 年没找到把它做进神经网络语言模型的工作。可借的只有“温度”：工作空间内容不稳定时提高探索 / 采样熵，稳定后降低。
- Unexpected Benefits of Self-Modeling（Premakumar、Graziano 等，2024，arXiv 2407.10188，未核实）：加一个预测自身隐藏激活的辅助损失，网络变简单（权重分布变窄）。含义：可加小的“自我模型”头预测下一拍中枢 / 工作空间状态，其误差兼作不确定度信号。

### 丹尼特与全局工作空间

- [Shared Global Workspace](https://arxiv.org/abs/2103.01197)（Goyal、Bengio 等，ICLR 2022，已核实摘要）：专家模块竞争写入容量有限的工作空间，再广播给所有模块，促进分工。含义：我们 4 槽工作空间方向对，写入应是竞争式（top-k），而不是平均。
- Identifying indicators of consciousness in AI systems（Butlin、Long、Bengio、Chalmers 等，Trends in Cognitive Sciences 2025-11，未核实）：从理论推出的内部指标，GWT-1 到 GWT-4 为并行专家、有限容量工作空间、全局广播、依状态的注意（按需查询模块）。可作工作空间设计的检查清单。
- [Can We Test Consciousness Theories on AI?](https://arxiv.org/abs/2512.19155)（Phua 2025-12，已核实）：去掉自我模型后元认知校准消失、任务表现不变；缩小工作空间则访问能力逐渐下降；广播会放大内部噪声，使代理变脆。含义：广播加门槛（“点燃”），并在注入噪声下测试。
- 多重草稿、大脑中的名声、乔伊斯式虚拟机、Pandemonium：没找到近期机器学习实现。最接近的做法是让多个候选竞争，“名声”= 持续影响工作空间的时长。只是思路，无证据。

### 其他

- Thousand Brains Project（Hawkins 等，[2507.04494](https://arxiv.org/abs/2507.04494)，已核实；代码 thousandbrainsproject/tbp.monty，MIT）：皮层柱模块之间投票、快速赫布绑定学习；只在物体识别上评测，摘要无数字。网格模块经中枢对下一字节投票是可能的借鉴，属推测。
- 主动推理：只有立场 / 代理类论文，没有训练序列模型的具体结果。心智社会：没有近期具体工作。
