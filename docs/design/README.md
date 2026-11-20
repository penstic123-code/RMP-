# 动态稀疏可塑性神经网络 · 设计 v0.18

2026-11-20｜目标：在固定内存预算下持续预测、在线学习，并保留旧规律；文本模式下以字节流为输入，检验能否学会语言。

各版改动见 [changelog.md](changelog.md)。排期见 [plan/](../plan/README.md)，实验见 [experiments/](../experiments/README.md)，文献见 [literature/](../literature/README.md)。

## 目录

**现行架构（current/）**——描述现行代码（v0.17 见 [archive/design_v0.17](../archive/design_v0.17/README.md)，v0.16 见 [archive/design_v0.16](../archive/design_v0.16/README.md)，v0.15 纸面设计见 [archive/design_v0.15](../archive/design_v0.15/README.md)）

- [0　一页看全：现行训练的模型（XL5）](current/0-overview.md)：结构、前向、切块、训练设置、学习方式、已知弱点
- [1　系统怎样运行](current/1-runtime.md)：模块、层间数据流、读出、中枢、会话边界、动态分块（惊讶跃升定边界）、滑窗注意力、融合内核
- [2　参数与显存预算](current/2-memory-budget.md)
- [3　学习策略：三种速度](current/3-learning.md)：离线训练、在线学习（低秩 / e-prop）、睡眠巩固、聊天工具；选择性学习缺口
- [4　记忆与保护旧知识](current/4-memory.md)

**依据与讨论（basis/）**

- [7　研究依据](basis/7-references.md)：经典引用（近期文献按主题见 [literature/](../literature/README.md)）
- [8　脑机制对照、层级与稀疏](basis/8-brain-hierarchy.md)：设计讨论结论，含全局工作空间（8.7）

**候选方案（candidates/）**——全部未实验；2026-11-20 起主线是现有 DSPN 逐步放大，可借鉴的机制已提炼进 [plan/v2.md](../plan/v2.md)“候选章节提炼”，各章只作素材

- [9　两级主干 + 三层记忆](candidates/9-redesign-candidate.md)：独立设计提案，部分已提前采纳（9.5）
- [10　双扫预测层级网络 TSPH](candidates/10-tsph-cleanroom.md)：净室设计，demo 后按 10.5 对比
- [11　潮汐网络 Tide](candidates/11-tide-cleanroom.md)：只看文献与实验记录的新起点提案，目标训练快
- [12　萤火网络 Firefly](candidates/12-firefly-brain.md)：类脑提案，以 Tide 为对照
- [13　可用模型路线](candidates/13-usable-model.md)：几千万参数、两天训练的候选主干与验证顺序
- [14　统一块 + 涌现层级 + 自上而下预测误差](candidates/14-predictive-hierarchy.md)：不替换第 13 章，demo 后与之对比
- [15　回响网络 Echo](candidates/15-echo.md)（[续](candidates/15-echo-plan.md)）：思考回路 + 海马槽表 + 自我模型
- [16　四条路线 A–D](candidates/16-paths.md)：嫁接 / 朴素 / 回响改 / 回响全（[A](candidates/16a-graft.md)、[B](candidates/16b-plain.md)、[C](candidates/16c-echo-s.md)、[D](candidates/16d-echo-x.md)）
- [17　十套从零设计](candidates/17-overview.md)：海马-皮层、神经调质可塑、育儿、投票柱、笔记本睡眠、活权重、自研、克隆选择、卡尔曼写入、即时编译
- [18　活柱网络 LiveColumns](candidates/18-live-columns.md)（[续](candidates/18-live-columns-plan.md)）：会长出新柱的神经调质可塑柱群
- [19　定容活柱](candidates/19-fixed-budget-columns.md)：固定内存预算下的活柱
- [20　DSPN v2：精简 + 各部件接通](candidates/20-dspn-v2.md)：**已选定、实施中**——删持久头与网格，工作空间并入中枢广播，新奇信号管海马写入与在线可塑，睡眠由海马教皮层
- [21　迹柱网络 TraceCols v2](candidates/21-tracecols.md)（[实现、迹的算法、提速与验证](candidates/21-tracecols-impl.md)）：净室候选，经一次独立审查——一种块（衰减谱矩阵记忆 + 一专家路由）、两级硬边界压缩、只给衰减参数带迹（迹显存 = 状态显存）、一个新奇信号、醒时冻结结构 + 睡眠防遗忘；v1 原型 [tracecols.py](../../tracecols.py) 作对照；按绝对能力验收
- [22　DSPN 精简（v0.18）](candidates/22-dspn-lean.md)：**C1–C4 已实施（v0.18），C5 把关未跑**——删广播、分块路由 4 件套改为编码层惊讶定边界、预测头族改为编码层局部头、删冷启动评估；中枢保留（E21、E34），删中枢另做消融；经一次独立审查
- [23　路由模块池 RMP](candidates/23-routed-pool.md)：**测试用、未实施**——很多小模块、每拍按地址选 k 个、地址醒时冻结睡时更新、先后顺序由学到的秩决定、跨级读、可选块延迟自上而下反馈、时间并行训练；测涌现、抗遗忘与在线学习、随规模变强、好不好训练
- [23.11　层级与分工从哪来](candidates/23b-hierarchy-signals.md)：第 23 章续，**测试用**——按语境路由、分层预测（各波预测不同时间范围）、结构先验“越深越慢”、四源混合数据测按任务分工

（原第 5、6 节已移出：实验记录 → experiments/，验收与排期 → plan/。）

## 附录　代码

v0.12–v0.13 的实验代码在 [legacy/](../../legacy)，v0.14 文本原型见 [dspn_text_v014.py](../../legacy/dspn_text_v014.py)，当前代码见 [dspn_text.py](../../dspn_text.py)。
