# 动态稀疏可塑性神经网络 · 设计 v0.15

2026-11-19｜目标：在固定内存预算下持续预测、在线学习，并保留旧规律；文本模式下以字节流为输入，检验能否学会语言。

各版改动见 [changelog.md](changelog.md)。排期见 [plan/](../plan/README.md)，实验见 [experiments/](../experiments/README.md)，文献见 [literature/](../literature/README.md)。

## 目录

**现行架构（current/）**——代码按这里实现

- [1　系统怎样运行](current/1-runtime.md)：模块、层、中枢、读出、数据流
- [2　省内存的三个位置](current/2-memory-budget.md)
- [3　学习策略：三种速度](current/3-learning.md)：快记忆、在线低秩、睡眠；选择性学习缺口
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
- [15　回响网络 Echo](candidates/15-echo.md)：综合第 9–14 章与文献的提案——思考回路、海马槽表、自我模型调质；依据、验证与风险见 [15-echo-plan.md](candidates/15-echo-plan.md)
- [16　四条路线：从最稳到最野](candidates/16-paths.md)：按五条要求给出的四套方案、对比表与推荐顺序——[A 嫁接](candidates/16a-graft.md)（现成小模型 + 生活层，含共用的睡眠与评测定义）、[B 朴素](candidates/16b-plain.md)（自训标准小模型 + 睡眠慢改稠密权重）、[C 回响改](candidates/16c-echo-s.md)（混合主干 + 持久头 + 槽表）、[D 回响全](candidates/16d-echo-x.md)（再加思考回路、自我模型、自主练习）
- [17　十套从零设计的方案](candidates/17-overview.md)：三个代理只读文献与实验、不看设计文档并行设计，含对比表与组合建议——务实 [P1 笔记本 + 夜间入脑](candidates/17p1-notebook-sleep.md)、[P2 活权重](candidates/17p2-live-weights.md)、[P3 自家苗](candidates/17p3-homegrown.md)；神经科学 [N1 海马—皮层](candidates/17n1-hippo-cortex.md)、[N2 神经调质可塑性](candidates/17n2-neuromod-plastic.md)、[N3 育儿式发育](candidates/17n3-nursery.md)、[N4 千脑投票柱](candidates/17n4-voting-columns.md)；跨领域 [X1 克隆选择](candidates/17x1-clonal.md)、[X2 卡尔曼写入](candidates/17x2-kalman.md)、[X3 分层即时编译](candidates/17x3-jit.md)
- [18　活柱网络 LiveColumns](candidates/18-live-columns.md)：用户选定的 N2 + N4 合成——冻结主干上的神经调质可塑柱群，聊天时按四因子局部规则改写、按预测误差分配责任、按需长新柱；依据、验证与风险见 [18-live-columns-plan.md](candidates/18-live-columns-plan.md)
- [19　定容活柱](candidates/19-fixed-budget-columns.md)：第 18 章 + DSPN 的固定内存前提——RWKV-7 递归主干冻结，柱只挂上层通道混合、单一状态路径，责任用块内写入误差估计，柱数定额、满额先合并或淘汰；阶段二主干换自研模型

（原第 5、6 节已移出：实验记录 → experiments/，验收与排期 → plan/。）

**实现与本页描述的差异（dspn_text.py，2026-11-19）。** 文本模式第 1 层各模块直接以完整嵌入为输入（B 为 e_dim×d），没有单独的 4×e_dim 投影 R；读出接所有层的 h 而非只接最上层，经瓶颈与 E 共享。

## 附录　代码

v0.12–v0.13 的实验代码在 [legacy/](../../legacy)，v0.14 文本原型见 [dspn_text_v014.py](../../legacy/dspn_text_v014.py)，当前代码见 [dspn_text.py](../../dspn_text.py)。
