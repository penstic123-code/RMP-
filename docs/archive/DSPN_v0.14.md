# 动态稀疏可塑性神经网络 · 设计与实验 v0.14

2026-11-19｜目标：在固定内存预算下持续预测、在线学习，并保留旧规律。

**当前决策。** v0.14 改模块与学习策略，除一次离线文本原型外都尚未实验，v0.13 网络保留为基线。模块从 16×16 稠密 tanh 循环改为“对角选择性线性循环 + 快权重记忆 + 局部 MLP + 本地预测头”；模块间消息从同层逐拍传递改为同一拍内由下层传给上层，使时间循环全为线性、可并行扫描训练。依据是 v0.12–v0.13 的三项失败：单个 16 单元模块记不住 8 步以前的输入（时延 8 时固定组 MSE 0.316，永远预测 0 为 0.313）；固定时间常数的泄漏单元只改善 1%–3%；隐藏模块的滞后逐跳反馈只补上与截断 BPTT 之间 57% 的差距，最差种子比固定网络差 14.1%。学习分三种速度：快记忆每拍按 delta 规则写入，不用误差；在线阶段每个模块以本地预测误差为主、读出误差的逐跳反馈为辅，乘精确对角痕迹，只更新少量参数；离线阶段用反向传播训练慢权重，并定期用重放池做睡眠巩固。循环改为对角后，模块内痕迹就是精确的在线梯度，SnAp-1 的近似只剩跨模块部分。沿用 v0.12–v0.13 已验证的做法：读出权重反馈优于固定随机反馈，痕迹沿待核对队列滞后更新，16 候选流式换边，NLMS 读出。新模块单模块同口径约 1,950 个持久数值，是 v0.13（560）的约 3.5 倍，必须做同字节对照。

## 1　系统怎样运行

模块分层排列，共 L 层，每层是一张网格，第一版为 4 层、每层 32 个模块（4×8）。第 1 层中 16 个模块各接一条外部观测通道。每个模块有 16 维线性状态 c、8×8 快记忆 S 和一个局部 MLP；模块将 MLP 输出 h 投影为 4 维消息 z。模块间消息只在同一拍内由下层传给上层：第 ℓ 层模块读取第 ℓ−1 层中正下方模块及其四邻居的消息，四邻居由坐标推算，边界不环绕。时间方向的记忆只放在各模块自己的 c 和 S 里，不经消息逐拍传播。这样时间循环全是线性的，训练时可以沿整段序列做并行扫描；v0.13 及更早的同层逐拍消息版本保留为对照，它的模块间循环经过 MLP，只能逐拍计算。第一版允许每个预测端接收至多两条活动远程入边。读出先导实验中的“三个槽”是独立实验配置。

1. 在时刻 t 观察 x(t)，先用它核对上一时刻对 x(t) 的预测。记录误差，然后用发出预测时保存的特征更新读出。第一步仅初始化。
2. 模块更新，自下而上逐层计算。输入 u：第 1 层为 [本通道当前输入]，更高层为 [下层正下方模块本拍的消息，下层四邻居本拍消息之和除以 4]。门 a = sigmoid(G·u + b_a)，每个单元一个保持率；状态 c ← a⊙c + (1−a)⊙(B·u)，不过 tanh。快记忆由 c 投影出键 k、值 v、查询 q，k 归一化后先按 S ← (1−γ)S 衰减，再写入 S ← S + β(v − S·k)kᵀ，最后读出 r = S·q；写入强度 β 和遗忘率 γ 都由 c 经 sigmoid 给出，γ 初始偏置使平均保持约 64 步。输出 h = MLP([c, r])，24→32→16，GELU；消息 z = P·h。初始化时 b_a 使时间常数 1/(1−a) 在 1–1000 步间对数均匀分布，其余矩阵随机初始化并保存种子。0<a<1 保证模块内循环稳定，模块间没有回环，不再需要 W 的谱范数检查；仍监测各层激活的饱和率。
3. 预测。每个模块有本地预测头：接观测通道的模块预测本通道 x(t+1)，其余模块预测下层输入模块下一拍的消息 z。对外读出接在最上层。在线阶段 P 冻结，所以消息目标不会被学习拉向常数。预测端另有对外读出，基础读出采用 NLMS：以误差乘输入特征，按特征平方范数加常数归一化。远程边只进入对外读出。普通 LMS 是同等地位的低内存对照。
4. 误差到达后，按第 3 节更新在线参数。痕迹不按当前时刻更新，而是随待核对队列弹出的那条记录滞后更新，使痕迹与误差对应同一次预测。
5. 先结清旧预测的学习，再更换连边；新边权重和学习状态从零开始。反馈延迟 D 步时，队列每步保存 u、a 和 c，供滞后痕迹递推，约 M·D·(|u|+2d) 个数，计入内存。

正式接口只有观测流 step(x)。过去的读出先导实验在预测后由生成器给出目标；它们分别检验了改权重与换边的机制，尚未覆盖完整单流系统。

## 2　省内存的三个位置

| 对象 | 当前做法 | 代价或边界 |
|---|---|---|
| 不存在的跨模块连接 | 保存活动来源索引与权重。容量、活动数、预留空槽分别计数；非活动模块对不持有长期权重或优化器状态。 | 局部块内仍支付真实稠密参数的成本。 |
| 寻找新连接 | 每 16 步共享抽取 16 个候选来源。按来源累计平方和，每个预测端单独累计误差平方和、候选特征乘残差。周期结束打分并清空。 | 扫全来源更快找到新规律。追踪速度优先时试 32 个候选。128 模块版另测“邻居的邻居 + 均匀远程探索”，与均匀抽样匹配候选数。 |
| 学习活动痕迹 | 默认对角精确痕迹：B、G 每个参数一个数（第 3 节）。快记忆写入路径在线不求导；MLP 无循环，不存痕迹，只用队列里发出预测时的激活。v0.13 的 SnAp-1 保留为对照。 | 痕迹与 B、G 同样大，按 d·\|u\| 增长，不再按 d² 增长。跨层的时间影响仍被忽略，由逐层反馈部分弥补。 |

模块宽度 d、输入维数 |u|、快记忆维数 m、MLP 隐层 n 固定口径下，每个模块持久数值约为：门与输入投影 2d|u|，痕迹 2d|u|，MLP (d+m)n + nd，快记忆 m² + 3md，消息投影 4d，本地预测头 d（接通道）或 16d（预测邻居消息）。d=16、|u|=5、m=8、n=32 时，接通道模块约 2,200 个数，隐藏模块约 2,450 个，其中 MLP 约占六成；v0.13 同口径约 660 个。FP32 下 M=10,000 时合计约 84–93 MiB。内存紧张时先把 n 降到 16（每模块约 1,600 个数）。若 Y 个预测端同时给 q 个共享候选打分，统计仍需 O(Yq)；逐跳反馈只用已有矩阵，不新建反馈矩阵。

**可选压缩。** 大多模块学到相似的 B、G 或 MLP 时，可比较“共享基础权重 + 少量独立偏移”与独立权重。将偏移索引、重建计算及效果损失一并统计。仅在相同任务精度下实际更省内存时采用。

## 3　学习策略：三种速度

在线学习只改一小部分参数，大部分知识留在离线训练的慢权重里。这样在线适应快，也不容易冲掉旧规律。

| 速度 | 学什么 | 规则 | 误差来源 |
|---|---|---|---|
| 每拍 | 快记忆 S | 门控衰减后 delta 写入 | 不需要误差 |
| 在线 | 对外读出；B、G；MLP 第一层的秩 2 偏移 | 学习率 × 反馈 × 痕迹，滞后更新 | 本地预测误差为主，读出误差的逐跳反馈为辅 |
| 离线 | 全部慢权重（K、V、Q、P、MLP、本地预测头及 B、G 的基础值） | 截断 BPTT 预训练；睡眠巩固 | 预训练数据与重放池 |

**在线反馈。** 模块 i 的状态反馈 δc_i 由两部分相加。一部分是本地误差：本地预测头权重转置乘本地误差，再经 MLP 在模块内反传到 c。另一部分是 λ 倍的逐层反馈：读出误差从最上层出发，上层模块把自身 δc 乘 (1−a)，经上层 B 的转置、本模块 P 转置和 MLP 局部导数传给下层。消息在同一拍内传递，所以逐层反馈不再按跳数滞后，v0.13 的时间对齐问题随之消失。λ 起点 0.3，在开发种子上调整。一个模块有多个上层读者时，反馈按读者求和。

**对角精确痕迹。** 对 B 的元素 B_ij，痕迹 e_ij ← a_i·e_ij + (1−a_i)·u_j；对门参数 G_ij，痕迹 f_ij ← a_i·f_ij + a_i(1−a_i)(c_i(t−1) − (B·u)_i)·u_j。更新 ΔB_ij = η·δc_i·e_ij，ΔG_ij = η·δc_i·f_ij。只要 a 和 u 不经本模块状态依赖参数，这就是模块内的精确在线梯度。两处仍是近似：下层参数的改变会经上层 c 的时间累积影响未来的输出，这条跨层的时间影响被忽略；快记忆 S 依赖过去的 c，在线不对写入路径求导，K、V、Q 只在离线阶段训练。所有痕迹随队列弹出的记录滞后递推，导数取发出预测时的值。

**MLP 秩 2 偏移。** 在线阶段 MLP 第一层实际权重为 W₁ + A·Bᵀ，只更新 A 和 B，每模块 2×(24+32) = 112 个数。MLP 没有循环，用队列中发出预测时的激活即可算梯度，不需要痕迹。秩限制了在线漂移的幅度，睡眠巩固时偏移并入 W₁。

**防止消息坍缩。** 不接通道的模块以邻居消息为预测目标，而消息本身由可训练的 P 产生；若 P 和本地预测头同时朝“预测邻居消息”的误差下降，最省事的解是让所有消息变成常数，误差归零而网络什么也没学到。离线阶段因此作两条规定：消息目标取 stop-gradient，邻居消息误差只更新本地预测头和本模块的 B、G、MLP，不更新任何模块的 P；P 只由对外读出误差和观测通道的本地误差训练。另加一项方差约束：每个消息维度按滑动平均估计方差，低于初始方差的 10% 时，对该维度施加 (阈值 − 方差) 的惩罚。在线阶段 P 冻结，这条风险不存在，但仍记录各消息维度的方差，低于阈值时报警。

**睡眠巩固。** 每 4,096 步或会话结束时进入巩固。先复制现场状态；从重放池抽片段，从独立状态预热；以小学习率用截断 BPTT 微调慢权重，并把 MLP 偏移并入 W₁ 后清零。巩固后用后续近期片段和旧记忆片段验收，规则与第 4 节换边保护相同：近期误差不恶化，且旧记忆 NMSE 恶化不超过 0.02（开发起点）才提交，否则回滚并恢复现场状态。巩固期间的副本与 BPTT 激活计入峰值内存。

**沿用的结论。** 反馈用对称形式。v0.12 中固定随机 ±1 反馈在学习率 0.02 时误差升到固定组的 1.15 倍，再大就发散；v0.13 中不滞后的逐跳反馈和固定随机反馈在接力链上都与“隐藏模块不学”无差别。两者保留为对照。

**调参与稳定性规则。** 学习率、λ 和巩固学习率是每种方法的一部分，各组都在开发种子上单独调整，测试种子只用于最终比较；最佳值落在网格边缘时继续外扩。运行中监测误差、饱和率和单步更新范数，超过阈值时裁剪更新或暂停在线学习。结果表除平均值外报告中位数和最差种子。完整递归系统的时间信用分配仍需以小网络反传参照。先例见 [RFLO](https://elifesciences.org/articles/43299v2)、[e-prop](https://www.nature.com/articles/s41467-020-17236-y)、[SnAp](https://openreview.net/pdf?id=q3KSThy2GwB)、Zucchet 等（2023）关于对角循环网络精确在线学习的工作，以及 [局部 Hebb/STDP 模型](https://neuronaldynamics.epfl.ch/online/Ch19.S2.html)。

## 4　记忆与保护旧知识

**工作记忆**由线性状态 c（时间常数 1–1000 步）、快记忆 S 和待核对的预测承担。**短期找边统计**积累最近输入对误差的解释力。**长期记忆**由跨会话保存的慢权重和活动路由承担，在线学到的内容经睡眠巩固进入慢权重。抗遗忘组另外使用不超过 64 KiB 的原始序列重放池，容量包含元数据，巩固与换边保护共用。无重放组保留为正式对照。

**快记忆怎样遗忘。** delta 规则只在新旧键相近时覆盖旧内容，情境切换后旧内容会留在 S 里干扰读出。因此 S 每拍先按门控遗忘率 γ 衰减（第 1 节），由模型自己决定何时清理；会话结束时 S 清零。对照为“无衰减”和“固定 γ”。S 的内容不直接巩固进慢权重，巩固只用重放池中的原始片段；没进池的经历不会被巩固。

**重放池存哪些片段。** 以 64 步为一段，按该段平均预测误差（意外程度）决定优先级。池未满时直接入池；池满时，新段的误差高于池中最低优先级时替换该段，同分时替换最旧的。巩固时按优先级成比例抽样。对照为均匀蓄水池抽样（每段以相同概率留存）。

**记忆长度。** v0.12 中，单个 16 单元 tanh 模块记不住 8 步以前的输入；固定时间常数的泄漏单元只改善 1%–3%，内部可塑性也无法补足。v0.14 用两种手段应对：输入决定的门让模块按需保持或清空状态，快记忆 S 按内容存取。两者都未验证。若第 6 节第 2 项仍不过关，回到网络层面的候选：跨模块的延迟链路、更宽的模块或专门的慢时间常数模块。

**保护旧知识。** 旧序列可更新权重。若旧通路被删除，还需在换边前比较原路由与候选路由。两者用同一批近期数据拟合，再用后续近期片段以及抽出的旧记忆片段验收。近期误差改善且旧记忆的 NMSE 恶化不超过预注册容差才接纳，开发起点为绝对增量 0.02。复制受影响目标的临时状态计入峰值。独立旧任务探针不入池。重放从独立状态预热并在结束后恢复现场状态。

**暂缓机制。** 注意力式入边门控、上下文广播和两级模块层级（v0.11 第 4 节）保留设计，但在基础系统通过第 6 节第 2、7 项验收前不实现。启用条件不变：已接入来源随情境交替有用时才试门控；多个输出需辨认同一情境时才试广播；多个输出重复使用同一对 L1 来源时才试共享模块 H。新增一项：字符级文本中长程原样召回不足时，才试一两个注意力模块作全局中枢。四者分别消融。

## 5　实验记录与设计依据

以下各行属于不同任务或种子集，误差只在同行指定的对照间比较；除最后一行的 v0.14 文本原型外，全部来自 v0.13 及更早的模块。v0.12 起，单模块实验在开发种子 0–9 上调参，在测试种子 1000–1099 上配对比较；“相对固定”指与同一种子固定核心的误差差，括号内为 95% 区间。

| 实验 | 已运行结果 | 对设计的影响 |
|---|---|---|
| 8 输入、8 输出，映射整体切换，10 种子 | 切换末段 MSE：冻结 1.400，只改原权重 0.774，允许换边 0.125；噪声底 0.1225 | 保留换边能力。结果限定于单一独立来源映射。 |
| 96 来源、8 输出，联合适应与遗忘 | 输入分离时稠密、稀疏都零遗忘；完全冲突且无情境提示时稀疏组旧任务 NMSE 增加 1.723，冻结组无法学好新任务 | 重点测试“有重叠、有可推断上下文”的任务；区分保留与重新学习。 |
| 在线寻找新来源，10 个独立确认种子 | 完整 16 步窗口 6,880 B，切换初期/末期 MSE 0.2883/0.1315；共享 16 候选 896 B，0.4766/0.1309；共享 32 候选 1,536 B，0.3675/0.1309 | 16 候选作为默认。896 B 相对稠密读出 3,072 B 减少 70.8%，只含被核算的持久数值数组。 |
| 隔离读出学习规则，给定正确来源 | LMS、NLMS、RLS 末段 MSE 0.1311、0.1313、0.1267；分别 224、224、800 B | 保留简单读出。RLS 精度收益需对照额外协方差存储。 |
| 16 单元循环块，10 种子，固定同一读出更新 | 冻结内部、Oja 型、简化反馈痕迹的冻结测试 MSE 分别为 0.2040、0.2984、0.2043 | 已被下面各行取代。 |
| v0.11 四种痕迹配置复核：原设置扩到 200 个测试种子 | 原随机反馈、学习率 0.0003 下，共享痕迹相对固定 −0.69%（−1.19% ~ −0.20%），66% 种子更好；把 200 个种子分成 20 组各 10 个，3 组得出相反结论 | 撤回“固定核心平均最好”。原学习率偏小约 100 倍。 |
| 反馈形式，学习率扫描（开发种子） | 随机 ±1 反馈在学习率 0.02、0.03、0.05 时误差为固定组的 1.15、1.38、1.61 倍；读出权重反馈在 0.01–0.05 均为 0.92–0.93 倍 | 默认读出权重反馈。 |
| 痕迹方法，时延 3、无反馈延迟 | 相对固定：共享 −11.5%，逐突触 −9.0%，多时间尺度 −8.1%（学习率网格上限 0.05）；SnAp-1 外扩至学习率 0.2 后 −33.2%（−36.1% ~ −30.3%），99% 种子更好，最差种子 +4.1% | 默认 SnAp-1。多时间尺度在短时延任务中无收益。 |
| 反馈延迟 4、16 步，时延 3 | 普通痕迹全部失效，相对固定在 −1.0% ~ +1.6% 之间，延长衰减率也无效。滞后痕迹：延迟 4 步时共享 −11.4%、SnAp-1 −23.8%；延迟 16 步时共享 −10.4%、SnAp-1（学习率 0.1）−27.9%（−30.7% ~ −25.2%），99% 种子更好 | 痕迹改为沿读出队列滞后更新。 |
| 时延 8、16 | 固定组 MSE 0.316，永远预测 0 的 MSE 为 0.313；所有痕迹方法变化不超过 ±0.5%。泄漏单元（a_min 0.1、0.03）使固定组改善 2.7%、0.9%，其上 SnAp-1 再改善 1.8%、无改善 | 单模块记忆不足，转由网络层面解决（第 4 节）。 |
| 4 模块接力链，隐藏模块反馈（v0.13）：输入只进模块 0，读出只在模块 3，任务整体后移 3 步 | 相对固定网络：只学预测端 −1.0%，固定随机反馈 −0.9%，不滞后的逐跳反馈 −1.0%（三者区间都含 0）；按跳数滞后的逐跳反馈 −18.7%（−21.0% ~ −16.5%），90% 种子更好，最差种子 +14.1%；截断 BPTT（窗口 16）−32.1%，100% 种子更好 | 隐藏模块采用滞后逐跳反馈。它补上 57% 的差距，所需额外状态约 224 个数，BPTT 窗口需 1,024 个。 |
| 同等持久字节，时延 3（W、输入投影、读出、痕迹合计） | 固定 d=16（304 个数）为基准；固定 d=22（550）−5.1%，最差 +27.6%；共享 d=22（572）−24.3%，最差 +53.5%；SnAp-1 d=16（560）−33.2%，最差 +4.1% | 在 d=16，字节用于 SnAp-1 痕迹优于用于加宽。时延 8 时三者都只改善 0.1%–1.8%。 |
| v0.14 原型，字节级莎士比亚（约 1.1 MB，后 10% 验证），只做离线训练，单种子，各 2,352 步（批 32 × 64 字节），CPU | 验证 bpc：同层逐拍消息版（8 模块，15.2 万参数）2.535；分层并行扫描版（2 层 × 4 模块，14.3 万参数）2.598；同参数 GRU（15.1 万参数）2.463。三者都生成像剧本的英文：人名、换行格式和大部分单词正确，句子无意义。分层版的并行扫描与逐拍计算结果一致（最大差 1.7×10⁻⁶） | 新模块能学会第一档文本，但在这一训练量下略差于 GRU（3%–5%），单种子不足以定排名。训练明显不足（成熟的字符级模型约 1.5 bpc）。下一步在 GPU 上用分块并行快记忆训到收敛，并补多种子。 |

痕迹实验的任务为 y(t) = tanh(x₀(t−L))·tanh(x₁(t)) + 0.4·x₀(t−L+1)，L=3 即 v0.11 任务，数据逐位相同。训练 4,000 步后冻结，测最后 512 步。反馈延迟 D 时，时刻 t 的误差在 t+D 才用于学习。所有组共享初始化、输入和读出规则；固定组在每个种子上只算一次并复用。临时矩阵、峰值字节和遗忘未在这些实验中评估。

## 6　下一轮的验收和自动科研顺序

1. **新模块单元测试。** 在单模块上比较在线对角痕迹给出的梯度与完整 BPTT 梯度，关掉快记忆时余弦相似度应接近 1，打开后的偏差来自 S 写入路径。同时检查门的取值范围、队列滞后对齐和序列化后预测一致。另做坍缩负对照：去掉 stop-gradient、允许 P 被邻居消息误差训练，确认消息方差确实塌缩，且第 3 节的规定能阻止它。
2. **单模块记忆。** 时延 3、8、16、32，开发种子 0–9 调参，测试种子 1000–1099 配对比较。对照为 v0.13 固定组、v0.13 SnAp-1 和同字节加宽的 v0.13 模块。通过条件：时延 8、16 明显低于永远预测 0 的 0.313；时延 3 不劣于 v0.13 SnAp-1 的 −33.2%。消融去门、去快记忆、去 MLP 各一组。
3. **接力链。** 4 模块链改为 4 层、每层 1 个模块，任务与 v0.13 相同。另跑同层逐拍消息的 v0.14 模块作对照，并记录两者的训练墙钟时间。对照为 v0.13 滞后逐跳反馈（−18.7%）和截断 BPTT（−32.1%）。消融：仅本地预测头、仅逐跳反馈、两者都有，并扫描 λ。目标是接近 BPTT，且最差种子不差于固定网络。
4. **跑完整单层系统。** 先用 32 个模块，接入 step(x)，同时打开读出学习、换边和三种速度的学习，检验本地预测头与逐跳反馈在多下游的网格中是否仍然成立。对照为“在线参数固定”和小规模截断 BPTT。16 通道流中切换 A→B→C，再回 A。任务包括来源分离、可由历史线索识别的重叠，以及无法凭观测历史区分的冲突负对照；其中加入需要 8 步以上记忆的通道。训练、开发、测试图独立。各方法同一时刻只见相同信息。顺序对照为持续值预测、稠密在线读出、固定稀疏、随机换边、流式评分换边和同学习峰值预算的小型 GRU。给 GRU 同样的在线更新机会，将训练缓冲算入预算。
5. **睡眠巩固。** 在第 4 项的切换流上比较无巩固、仅重放、重放加巩固；重放池分别用意外优先与均匀蓄水池两种选择策略；快记忆分别用门控衰减、固定衰减与无衰减。报告各旧任务遗忘量（最终误差减刚学完该任务时的误差）、回到 A 后的重学速度和巩固的峰值内存。
6. **在目标宽度重测预算。** 新模块中 MLP 占大头。在部署宽度（例如 d=32）做同字节比较：MLP 隐层 16 与 32、快记忆维数 4 与 8，以及把同样字节用于增加模块数。
7. **三项指标同时验收。** 扫 32/64/128/256 模块，在预先指定的部署规模比较模型数组、检查点、CPU 峰值 RSS 或 GPU 分配/保留峰值以及步延迟。初始工程门槛：末段 NMSE 相对稠密组至多高 5%，完整在线峰值至少下降 20%；切换后前 1,024 步误差相对优先适应的在线对照至多高 5%，且优于冻结；对可识别情境，独立冻结探针的最差旧任务 NMSE 增量 ≤0.05。独立报告回到 A 后的重学速度。
8. **字符级文本。** 字节级词表（256 个符号）。先离线预训练慢权重，再在线切换语料风格。对照为同参数、同分词、同文本量和同更新机会的小 GRU 与小 Transformer。指标为每字符比特数、切换后适应速度和切回后的遗忘量。
9. **最后试附加机制。** 按第 4 节的启用条件逐项试暂缓机制。
10. **运行器自动保存证据。** 每轮只改一个因素，保存配置、代码版本、图与轨迹种子、原始分段误差、各数组的 dtype/容量、运行峰值、耗时与退出状态。对照组结果按种子缓存，缓存键包含种子、任务参数、共享部件参数与代码版本，任一项变化即重跑。开发阶段锁定门槛，独立评估器执行至少 10 张测试图（单模块机制实验至少 100 个种子）；报告逐图配对差、95% 区间、中位数与最差值，跨门槛记为未定。重点测试未来观测不改变既有预测、冻结探针不写入、换槽不沿用旧痕迹、巩固回滚后状态一致、序列化后预测一致。自动报告须追溯到原始日志。

## 7　研究依据

固定递归核心与泄漏单元可参照[储备池计算综述](https://www.sciencedirect.com/science/article/pii/S1574013709000173)。选择性对角线性循环参照 [Mamba](https://arxiv.org/abs/2312.00752)；对角循环网络的精确在线学习参照 Zucchet 等，“Online learning of long-range dependencies”（NeurIPS 2023）。delta 规则快权重记忆参照 [Schlag 等，2021](https://arxiv.org/abs/2102.11174) 及 Yang 等（2024）的 DeltaNet 并行训练。本地预测误差参照 Rao 与 Ballard（1999）的预测编码模型；快慢两套记忆与睡眠巩固参照 McClelland 等（1995）的互补学习系统理论。动态稀疏训练中的新生参数稳定性见 [Xiao 等，ICML 2026](https://proceedings.mlr.press/v306/xiao26t.html)；跨步信用分配参照 [SnAp](https://openreview.net/pdf?id=q3KSThy2GwB)。重放参照[小型情景记忆研究](https://arxiv.org/abs/1902.10486)。这些工作各自支持一种候选机制，三项联合能力以本项目实验为准。v0.11 引用的两份先导报告存放在个人账户中，共享本文档前须换成随附文件。

## 附录　可复现代码

各段程序分别保存为对应文件，安装 NumPy 后用 Python 运行。以下均为 v0.12–v0.13 旧模块的实现；v0.14 文本原型见随附的 dspn_text.py（需要 PyTorch）。第一段是 v0.12 的痕迹实验核心，后两段保留读出和搜索实验的实现。结果是 CPU 机制实验。v0.12–v0.13 各表格行的完整运行脚本（exp1.py、exp1b.py、exp2.py、exp_recheck.py、exp_chain.py）随文档另附。


### 痕迹实验核心（trace_lab3.py，依赖 trace_lab2.py 的任务生成）

```python
"""DSPN v0.12 follow-up: two design fixes found by experiment 1.

Fix A (delay): lagged traces. The readout already keeps a queue of the states it
  predicted from. Updating the trace from that queue, D steps behind, lines the
  credit up with the error that arrives now. Extra cost: one stored state (d numbers).
  shared_lag : per-source EMA, D steps behind, x slope at prediction time   (2d)
  snap_lag   : SnAp-1 trace, D steps behind                                 (d*d + d)
Fix B (memory horizon): leaky units with spread time constants
  h <- (1-a) h + a tanh(W h + U x + b),  a_i spread evenly over [a_min, 1].

With D=0 and leak off, shared_lag == shared and snap_lag == snap (same numbers).
"""
import numpy as np
from trace_lab2 import make_task, STEPS, FREEZE

TRACE_NUMBERS = {'fixed': lambda d: 0, 'shared': lambda d: d, 'snap': lambda d: d * d,
                 'shared_lag': lambda d: 2 * d, 'snap_lag': lambda d: d * d + d}


def run(seed, L=3, D=0, d=16, method='fixed', lr=0.0, lam=0.5, a_min=None):
    initial, inputs, x, y = make_task(seed, L, d)
    a = np.ones(d) if a_min is None else np.linspace(a_min, 1.0, d)
    w = initial.copy()
    readout = np.zeros(d)
    state = np.zeros(d)
    pre = np.zeros(d)
    trace = np.zeros((d, d))
    prev_feat = np.zeros(d)          # state one step before the popped prediction (lag methods)
    queue = []
    errors = []
    for t in range(L, STEPS):
        old = state
        state = (1 - a) * old + a * np.tanh(w @ old + inputs @ x[t] + .15)
        error = y[t] - readout @ state
        if t >= FREEZE:
            errors.append(error ** 2)
            continue
        gain = a * (1 - ((state - (1 - a) * old) / a) ** 2)   # d state_i / d (W h)_i
        if method == 'shared':
            pre = lam * pre + (1 - lam) * old
        elif method == 'snap':
            trace = (1 - a)[:, None] * trace + gain[:, None] * (old[None, :] + np.diag(w)[:, None] * trace)
        queue.append((error, state.copy(), gain.copy()))
        if len(queue) <= D:
            continue
        err, feat, g = queue.pop(0)
        readout += .04 * err * feat / (1 + feat @ feat)
        if method == 'fixed':
            prev_feat = feat
            continue
        if method == 'shared':
            update = gain[:, None] * pre
        elif method == 'snap':
            update = trace
        elif method == 'shared_lag':
            pre = lam * pre + (1 - lam) * prev_feat
            update = g[:, None] * pre
        elif method == 'snap_lag':
            trace = (1 - a)[:, None] * trace + g[:, None] * (prev_feat[None, :] + np.diag(w)[:, None] * trace)
            update = trace
        prev_feat = feat
        w += lr * err * readout[:, None] * update
    e = float(np.mean(errors))
    return e if np.isfinite(e) else 1e3
```

### 隐藏模块反馈：4 模块接力链（chain_lab.py）

```python
"""DSPN v0.13: hidden-module feedback on a 4-module chain.

Module 0 receives the 2 input channels; only module K-1 has a readout. Modules talk
with 4-dim messages z = P h to their chain neighbours, m_i = sum z_j / 4 (v0.12 rule).
Information must be relayed through the hidden modules 1..K-2.

Task (v0.12 task shifted by H = K-1 hops):
  y(t) = tanh(x0(t-H-3)) * tanh(x1(t-H)) + 0.4 * x0(t-H-2)

Only the internal W of each module learns (plus the NLMS readout). P, U, V fixed.
Methods
  fixed      readout only
  pred_only  SnAp-1 in the readout module only (hidden modules frozen)
  random     every module SnAp-1; hidden modules get err x fixed random +-1 vector
  symmetric  every module SnAp-1; hidden feedback relayed back one hop at a time
             along the message path: delta_j = (U_i P_j / 4)^T (slope_i * delta_i)
  symmetric_lag  as symmetric, but module j's trace (and the downstream slopes used in
             the relay) run k = K-1-j steps behind, matching the k steps its state needs
             to reach the readout. Extra memory: the last K+1 chain states.
  bptt       truncated BPTT over the whole chain (window 16), W blocks only: reference
"""
import numpy as np

STEPS, FREEZE = 4512, 4000
K, d, q = 4, 16, 4
H = K - 1


def make(seed):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, .75 / np.sqrt(d), (K, d, d))
    P = rng.normal(0, 1 / np.sqrt(d), (K, q, d))
    U = rng.normal(0, 1.0, (K, d, q))
    V = rng.normal(0, .35, (d, 2))
    B = rng.choice([-1., 1.], (K, d))
    x = rng.normal(size=(STEPS, 2))
    y = (np.tanh(np.roll(x[:, 0], H + 3)) * np.tanh(np.roll(x[:, 1], H))
         + .4 * np.roll(x[:, 0], H + 2))
    return W, P, U, V, B, x, y


def coupling(P, U):
    """A[i][j] = U_i P_j / 4 for chain neighbours, as a dict of blocks."""
    return {(i, j): U[i] @ P[j] / 4 for i in range(K) for j in (i - 1, i + 1) if 0 <= j < K}


def big_matrix(W, C):
    A = np.zeros((K * d, K * d))
    for i in range(K):
        A[i*d:(i+1)*d, i*d:(i+1)*d] = W[i]
    for (i, j), blk in C.items():
        A[i*d:(i+1)*d, j*d:(j+1)*d] = blk
    return A


def run(seed, method='fixed', lr=0.0, window=16, lr_h=None):
    lr_h = lr if lr_h is None else lr_h
    W, P, U, V, B, x, y = make(seed)
    C = coupling(P, U)
    Vbig = np.zeros((K * d, 2)); Vbig[:d] = V
    h = np.zeros(K * d)
    readout = np.zeros(d)
    trace = np.zeros((K, d, d))
    hist_h, hist_prev, hist_e, hist_r = [], [], [], []
    errors = []
    hs = []
    learn = [i for i in range(K)] if method in ('random', 'symmetric', 'symmetric_lag') else ([K - 1] if method == 'pred_only' else [])
    for t in range(H + 3, STEPS):
        A = big_matrix(W, C)
        prev = h
        h = np.tanh(A @ prev + Vbig @ x[t] + .15)
        hp = h[-d:]
        err = y[t] - readout @ hp
        if t >= FREEZE:
            errors.append(err ** 2)
            continue
        slope = (1 - h ** 2).reshape(K, d)
        hb, pb = h.reshape(K, d), prev.reshape(K, d)
        hs.append(hb.copy())
        if len(hs) > K + 2:
            hs.pop(0)
        if method == 'symmetric_lag':
            for i in learn:   # lagged SnAp-1: module i looks k_i steps back
                k = K - 1 - i
                if len(hs) >= k + 2:
                    cur, old = hs[-1 - k], hs[-2 - k]
                    trace[i] = (1 - cur[i] ** 2)[:, None] * (old[i][None, :] + np.diag(W[i])[:, None] * trace[i])
        elif learn:
            for i in learn:   # SnAp-1 per module
                trace[i] = slope[i][:, None] * (pb[i][None, :] + np.diag(W[i])[:, None] * trace[i])
        r_used = readout.copy()
        readout += .04 * err * hp / (1 + hp @ hp)
        if method in ('pred_only', 'random', 'symmetric', 'symmetric_lag'):
            delta = np.zeros((K, d))
            delta[K - 1] = err * readout
            for j in range(K - 2, -1, -1):
                if method == 'random':
                    delta[j] = err * B[j]
                elif method == 'symmetric':
                    delta[j] = C[(j + 1, j)].T @ (slope[j + 1] * delta[j + 1])
                elif method == 'symmetric_lag':
                    k1 = K - 1 - (j + 1)
                    s1 = 1 - hs[-1 - k1][j + 1] ** 2 if len(hs) > k1 else slope[j + 1]
                    delta[j] = C[(j + 1, j)].T @ (s1 * delta[j + 1])
            for i in learn:
                W[i] += (lr if i == K - 1 else lr_h) * delta[i][:, None] * trace[i]
        elif method == 'bptt':
            hist_h.append(h); hist_prev.append(prev); hist_e.append(err); hist_r.append(r_used)
            if len(hist_h) == window:
                g = np.zeros(K * d)
                gA = np.zeros((K, d, d))
                for k in range(window - 1, -1, -1):
                    out = np.zeros(K * d); out[-d:] = hist_e[k] * hist_r[k]
                    g = (out + g) * (1 - hist_h[k] ** 2)          # -dL/d(pre-activation)
                    gb, pb_ = g.reshape(K, d), hist_prev[k].reshape(K, d)
                    for i in range(K):
                        gA[i] += np.outer(gb[i], pb_[i])
                    g = A.T @ g
                W += lr * gA / window
                hist_h, hist_prev, hist_e, hist_r = [], [], [], []
    e = float(np.mean(errors))
    return e if np.isfinite(e) else 1e3
```

### 任务生成与 v0.12 实验 1 的方法（trace_lab2.py）

```python
"""DSPN v0.12 trace experiments: one 16-unit (or d-unit) recurrent block,
online NLMS readout, local plasticity of W with feedback = readout weights.

Task: y(t) = tanh(x0(t-L)) * tanh(x1(t)) + 0.4 * x0(t-L+1)   (L=3 is the v0.11 task)
Feedback delay D: the error of the prediction made at t is only known at t+D.
  - readout learns with the features saved at prediction time (queue of D states)
  - W learns at t+D with the trace as it stands at t+D (the trace must bridge D)
Train 4000 steps, then freeze everything and score the last 512 steps.

Methods and persistent trace numbers per block (d=16):
  fixed   0      only the readout learns
  shared  d      per-source EMA of activity, x current target slope (decay tuned)
  multi   3d     shared trace at three fixed decays 0.5/0.9/0.99, averaged
  full    d*d    per-synapse EMA of (target slope x source activity) (decay tuned)
  snap    d*d    SnAp-1: e_ij = slope_i * (h_j(t-1) + W_ii * e_ij)   (no decay knob)
"""
import numpy as np

STEPS, FREEZE, SCORE = 4512, 4000, 512
TRACE_NUMBERS = {'fixed': lambda d: 0, 'shared': lambda d: d, 'multi': lambda d: 3 * d,
                 'full': lambda d: d * d, 'snap': lambda d: d * d}


def make_task(seed, L, d):
    rng = np.random.default_rng(seed)
    initial = rng.normal(0, .75 / np.sqrt(d), (d, d))
    inputs = rng.normal(0, .35, (d, 2))
    rng.choice([-1., 1.], d)        # v0.11 drew a random feedback vector here; kept so data match
    x = rng.normal(size=(STEPS, 2))
    y = np.tanh(np.roll(x[:, 0], L)) * np.tanh(x[:, 1]) + .4 * np.roll(x[:, 0], L - 1)
    return initial, inputs, x, y


def run(seed, L=3, D=0, d=16, method='fixed', lr=0.0, lam=0.5):
    initial, inputs, x, y = make_task(seed, L, d)
    w = initial.copy()
    readout = np.zeros(d)
    state = np.zeros(d)
    pre = np.zeros(d)
    multi = np.zeros((3, d))
    lams = np.array([.5, .9, .99])[:, None]
    trace = np.zeros((d, d))
    queue = []                      # (error, features) waiting for their observation
    errors = []
    for t in range(L, STEPS):
        old = state
        state = np.tanh(w @ old + inputs @ x[t] + .15)
        error = y[t] - readout @ state
        if t >= FREEZE:
            errors.append(error ** 2)
            continue
        slope = 1 - state ** 2
        # update the traces every step (they only look at activity)
        if method == 'shared':
            pre = lam * pre + (1 - lam) * old
        elif method == 'multi':
            multi = lams * multi + (1 - lams) * old
        elif method == 'full':
            trace = lam * trace + (1 - lam) * slope[:, None] * old
        elif method == 'snap':
            trace = slope[:, None] * (old[None, :] + np.diag(w)[:, None] * trace)
        queue.append((error, state.copy()))
        if len(queue) <= D:
            continue
        err, feat = queue.pop(0)    # the observation for this old prediction arrives now
        readout += .04 * err * feat / (1 + feat @ feat)
        if method == 'fixed':
            continue
        if method == 'shared':
            update = slope[:, None] * pre
        elif method == 'multi':
            update = slope[:, None] * multi.mean(0)
        else:
            update = trace
        w += lr * err * readout[:, None] * update
    e = float(np.mean(errors))
    return e if np.isfinite(e) else 1e3
```

### 最小换边实验（pilot.py）

```python
import numpy as np

def run(seed, mode, width=8, period=128):
    rng = np.random.default_rng(seed)
    search = np.random.default_rng(seed + 1000)
    graphs = rng.permutation(width)
    parents = np.stack([graphs, np.roll(graphs, 1)])
    x = rng.normal(size=(8192, width)).astype('f')
    noise = rng.normal(0, .35, x.shape).astype('f')
    dst = np.arange(width, dtype=np.int32)
    src = search.permutation(width).astype(np.int32)
    w = np.zeros(width, dtype='f')
    buf = np.empty((2, period, width), dtype='f')
    errors = []
    for t, row in enumerate(x):
        pred = row[src] * w
        target = .8 * row[parents[t // 4096]] + noise[t]
        err = target - pred
        errors.append(float(np.mean(err**2)))
        if t < 4096 or mode != 'freeze':
            w += .025 * err * row[src]
        buf[:, t % period] = row, target
        if (t+1) % period or mode == 'static' or (mode in ('weight','freeze') and t >= 4096):
            continue
        for k in dst:
            candidates = np.unique([src[k], *search.choice(width, 2, replace=False)])
            scores = [abs(buf[0, :, j] @ buf[1, :, k]) /
                      max(np.linalg.norm(buf[0, :, j]), 1e-8) for j in candidates]
            best = candidates[np.argmax(scores)]
            if best != src[k]:
                src[k], w[k] = best, 0.
    return [np.mean(errors[a:b]) for a,b in [(3072,4096),(4096,4224),(7168,8192)]], np.mean(src==parents[1])

if __name__ == '__main__':
    for mode in ['static','freeze','weight','online']:
        res = [run(s,mode) for s in range(10)]
        print(mode, np.mean([v[0] for v in res],0), np.mean([v[1] for v in res]))
```

### 读出学习与流式搜索（mechanisms.py）

```python
import numpy as np
D,Y,T,L,Q=96,8,1024,16,4
F=np.float32
def run(seed, rule, search,Q=4):
    rg=np.random.default_rng(seed)
    pools=np.stack([rg.choice(D,3,False) for _ in range(Y)])
    dense=search=='dense'
    idx=pools.astype('i') if search=='known' else np.zeros((Y,3),'i')
    w=np.zeros((Y,3),F)
    used=np.full(Y,3 if search=='known' else 0,'i')
    st=[w,idx,used]
    if dense:
        w=np.zeros((Y,D),F);st=[w];del idx,used
    if rule=='rls':
        cov=np.tile(4*np.eye(3,dtype=np.float64),(Y,1,1));st.append(cov)
    rng=np.random.default_rng(seed+1337)
    shared=search=='shared'
    if search=='window':
        hist=np.zeros((L,D),F); residual=np.zeros((L,Y),F);st.extend([hist,residual])
    elif search not in ('known','dense'):
        cand=np.zeros((1 if shared else Y,Q),'i')
        xx=np.zeros_like(cand,dtype=F)
        xy=np.zeros((Y,Q),F)
        ee=np.zeros(Y,F)
        st.extend([cand,xx,xy,ee])
    byphase=[]
    for phase in range(3):
        losses=[]
        for t in range(T):
            x=rg.standard_normal(D).astype(F)
            y=.8*x[pools[:,phase]]+rg.normal(0,.35,Y)
            v=x[None,:] if dense else x[idx]*(np.arange(3)<used[:,None])
            e=(y-(w*v).sum(1)).astype(F)
            losses.append(np.mean(e*e))
            if rule=='lms': w+=.04*e[:,None]*v
            if rule=='nlms': w+=.15*e[:,None]*v/(1+(v*v).sum(1,keepdims=True))
            if rule=='rls':
                pv=(cov@v[:,:,None])[:,:,0]
                gain=pv/(.98+(v*pv).sum(1,keepdims=True))
                w+=gain*e[:,None]
                cov[:]=(cov-gain[:,:,None]*pv[:,None,:])/.98
                cov[:]=(cov+cov.transpose(0,2,1))*.5
            if search in ('known','dense'): continue
            if search=='window':
                hist[t%L]=x;residual[t%L]=e
                if (t+1)%L: continue
                cand=np.arange(D)[None,:]
                score=np.abs(residual.T@hist)/np.sqrt(.001+(residual**2).sum(0)[:,None]*(hist**2).sum(0))
            else:
                if t%L==0:
                    for row in cand: row[:]=rng.choice(D,Q,False)
                    xx.fill(0);xy.fill(0);ee.fill(0)
                vals=x[cand];xx+=vals*vals;xy+=vals*e[:,None];ee+=e*e
                if (t+1)%L: continue
                score=np.abs(xy)/np.sqrt(.001+xx*ee[:,None])
            for k in range(Y):
                j=np.argmax(score[k]);src=cand[0 if shared or search=="window" else k,j]
                if score[k,j]<.78 or src in idx[k,:used[k]]: continue
                slot=used[k] if used[k]<3 else np.argmin(abs(w[k]))
                idx[k,slot]=src;w[k,slot]=0;used[k]=min(3,used[k]+1)
        byphase.append([np.mean(losses[:256]),np.mean(losses[-256:])])
    return np.mean(byphase[1:],axis=0),sum(a.nbytes for a in st)
if __name__=='__main__':
    for split,seeds in [('dev',range(3)),('explore',range(30,40)),('confirm',range(100,110))]:
        for rule,search,q in [('lms','known',4),('nlms','known',4),('rls','known',4),('nlms','dense',4),('nlms','window',4),('nlms','independent',4),('nlms','shared',4),('nlms','shared',16),('nlms','shared',32)]:
            vals=[run(s,rule,search,q) for s in seeds]
            print(split,rule,search,q,np.round(np.mean([v[0] for v in vals],axis=0),4),vals[0][1],flush=True)
```
