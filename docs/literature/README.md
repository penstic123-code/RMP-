# 文献目录

两级：**topics/** 按主题列条目（每条一两句 + 链接 + 是否核实），**cards/** 是准备采纳条目的实现卡（公式、超参、官方代码位置、对应本项目哪段代码、风险、是否要重训）。先看下面的结论，需要细节再点进主题或卡片。采纳决定和排期只在 [plan/adoption.md](../plan/adoption.md)；设计依据的经典引用见 [design/basis/7-references.md](../design/basis/7-references.md)。

## 主题（topics/）

| 主题 | 文件 | 内容 |
|---|---|---|
| 新型序列架构 | [architectures.md](topics/architectures.md) | SSM（Mamba-1/2/3）、线性注意力与 RNN 复兴、混合架构、HRM/TRM 递归、小规模正面对比 |
| 快记忆与线性注意力层 | [fast-memory.md](topics/fast-memory.md) | 生产级 delta 层配方（KDA、Qwen3-Next、RWKV-7）、Kronecker delta、预条件 DeltaNet、xLSTM、Muon |
| 持续学习与睡眠 | [continual-sleep.md](topics/continual-sleep.md) | 自生成回放、间隔重复、SuRe、可塑性丧失与 FIRE、LMs Need Sleep、测试时训练、稀疏记忆微调 |
| 层级与自上而下 | [hierarchy-topdown.md](topics/hierarchy-topdown.md) | H-Net 及其小规模复现、BLT、自上而下反馈的大模型结果 |
| 学习规则 | [learning-rules.md](topics/learning-rules.md) | 非反传方法与 BP 的差距、元可塑性、三因子规则 |
| 对话数据 | [chat-data.md](topics/chat-data.md) | TinyDialogues、TinyChat15M、小模型指令数据 |
| 认知理论与自我改进 | [cognition-selfimprove.md](topics/cognition-selfimprove.md) | 哥德尔机系列、SRWM、Nested Learning、SEAL、侯世达与类比、丹尼特、全局工作空间指标 |
| 元胞自动机与 Wolfram（只作参考） | [cellular-automata.md](topics/cellular-automata.md) | NCA 数据预预训练、混沌边缘、TextNCA、NCA 其他方向、Wolfram 最小模型与计算不可约性 |

## 实现卡（cards/）

| 文件 | 卡片 |
|---|---|
| [fast-memory.md](cards/fast-memory.md) | 1 KDA / GDN 配方，2 Muon，3 Kronecker delta，4 SRWM |
| [memory-workspace.md](cards/memory-workspace.md) | Palimpsa、预条件 DeltaNet（PDN）、In-Place TTT 写入目标、共享工作空间 + 噪声脆弱性、自我模型头、Nested Learning 中间时间尺度、延迟提问监督、元学习抗遗忘 |
| [topdown-data.md](cards/topdown-data.md) | Full-bandwidth、Latent Lattice、LLM-JEPA、H-Net 动态分块、Scratchpad Patching、TinyChat15M / TinyDialogues 数据配方 |
| [sleep.md](cards/sleep.md) | 5 SuRe，6 稀疏记忆微调，7 SM-2 间隔重复，8 FIRE |
| [sleep-2.md](cards/sleep-2.md) | 9 自生成回放与饱和，10 LMs Need Sleep，11 离线循环睡眠，12 SEAL，13 DGM/HGM 检查点档案，14 反事实探针，15 可塑性诊断，16 RL's Razor 的 KL |

## 对项目的总结论

**架构层面：**

1. 不换模块。10M 以下没有严格的架构正面对比；现有证据（字节级 6 万参数、莎士比亚字符级、H-Net 复现）显示架构差别多在噪声内，形状与调参影响更大。先把基线做严：形状扫描、调学习率、3 种子、MQAR 类探针。
2. HRM/TRM 起作用的不是“快慢两级”，而是外层反复细化与全程反传；共享一个网络比分开的 f_H/f_L 更好，自回归字符任务上无可靠增益。层级想法若再试，改为“同参数下同一网格每字节跑两遍”，与加宽同算力对比，预期放低。
3. 自上而下最值得试的是 Full-bandwidth 式“上一拍顶层输出经 GLU 回送第 0 层输入”：约 7 千参数、不占主干，可在现有检查点上续训最后约 25% 步数时加入（多遍并行）；H-Net 动态分块、Scratchpad、Lattice 水平流在本规模不看好（社区复现中纯字节以 1/3.6 的参数仍胜）。

**不用重训（睡眠 / 在线 / 评测）：**

4. 自生成回放我们已有（_dream + 锚定），缺的是“余量 / 饱和”测量——饱和的模型回放救不了，先测。
5. 睡眠返回值加 KL(早晨模型‖当前) 作遗忘预测（RL's Razor），几行代码。
6. 探针加对照项：换名、字节置换、答案顺序两向；接受门质量取决于探针。
7. 被接受检查点存成档案树，按后续睡眠是否持续过关挑祖先（Darwin / Huxley-Gödel）；只在反复回滚时才切换。
8. SuRe 相对现有惊奇优先回放只新增“慢 EMA LoRA”；间隔重复证据弱，只作回放采样三臂对照之一。
9. FIRE 不放进睡眠（短睡眠后必回滚），只在离线训练阶段之间用；可塑性丧失按拟合在 5M 参数约 220 亿 token 才出现，属长训练监控。
10. 工作空间先在现有检查点上做噪声测试；跨会话可加一级按会话更新的 LoRA。
- 对话数据：TinyChat（745 MB，CC BY-NC）与 TinyDialogues（141 MB，MIT）可直接下载，词汇受限、更适合小模型；make_chat_data.py 加解析与混合比例，SODA 降为少数。

**要重训（v2 前在 S 档对照）：**

11. 快记忆层：q/k/v 短卷积、通道级衰减门、输出 RMSNorm + sigmoid 门（不用 Swish）、q/k 归一化；对角预条件 PDN（与分块 WY 兼容，三行改动，吸收 Palimpsa）；可选 Mamba-3 的复数旋转。优化器试 Muon（按参数名分组）。
12. 训练目标：延迟提问监督 + 抗上下文内遗忘目标，共用一处 make_chat_data.py 改动（答案字节损失掩码、事实与提问间插更多无关对话）；需要跨批次延续状态。
13. 排后：Kronecker delta、In-Place TTT 写入目标、自我模型头、稀疏记忆层（需加乘积键记忆）、SRWM（无语言建模结果、不能分块并行）。
