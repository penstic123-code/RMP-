# 排期（唯一施工入口）

做什么、按什么顺序、做到哪了，只写在本文件夹；当前状态和下一步只在本页。总入口见 [docs/README.md](../README.md)。

| 文件 | 内容 |
|---|---|
| 本页 | 当前状态、下一步、快速路线、迭代规矩 |
| [stages.md](stages.md) | demo 阶段表与过关条件、各阶段状态 |
| [adoption.md](adoption.md) | 文献与设计讨论的采纳决定和排期 |
| [research-lines.md](research-lines.md) | demo 之后的研究线：形状扫描、模块粒度、与 GRU 对比、层级 |
| [acceptance.md](acceptance.md) | 验收项定义 1–10（原设计第 6 节，编号不变） |
| [v2.md](v2.md) | demo 之后：删什么、加什么、改什么，放大到 3000 万参数的路线与过关条件 |

## 当前状态（2026-11-19 晚，交接用）

- 环境：conda 环境 `dspn`（Python 3.11，PyTorch CUDA 12.8 版，RTX 5060 Laptop 8 GB）。2026-11-19 起用户的渲染程序（Rtxpt.exe，原占约 4 GB）一直关闭，训练可用约 7 GB 显存（批 8 × 1024 时峰值 2.6 GB）。
- 代码：[dspn_text.py](../../dspn_text.py)（DSPN：字节级网格 + 中枢 + 工作空间 + 可选词级层 `word_*`、自上而下层 `td_layers`、会话边界字节 SEP=30；训练 / 采样入口）、[hub.py](../../hub.py)、[online.py](../../online.py)、[evals.py](../../evals.py)、[tools/chat.py](../../tools/chat.py)（聊天 demo）、[tools/make_chat_data.py](../../tools/make_chat_data.py)（对话数据）、[tsph.py](../../tsph.py) + [tools/train_tsph.py](../../tools/train_tsph.py)（净室对照方案 TSPH，设计第 10 章；已过 CPU 测试，未在数据上训练）。
- 数据：TinyStoriesV2-GPT4（训练 2.23 GB，验证 22.5 MB）；data/chat-train.txt、chat-valid.txt（SODA 双人对话 + 25% 模板记忆对话，一半跨会话，对话之间是 SEP 字节）。
- **demo 现状（2026-11-20 早）**：最好的对话模型 runs/dspn-L_chat3.pt（E50）。试用：`python tools/chat.py runs/dspn-L_chat3.pt --size L --extra "<tools/queues/overnight.json 里的 extra>" --episodic`（--episodic 开海马式情景记忆，跨会话约 5/8；不开为 0）。命令 /new 新会话、/sleep 睡眠（上下文蒸馏 + 旧语料锚定，E50 已能提交）、/save 存权重与情景库、/quit。已知短板：名字复制常错几个字节，人名与宠物名会混；持久头未被用上（E48、E50）。
- **正在跑**：队列 tools/queues/chat3_seeds.json 的 S 档补种子（预测自身 / 下层 0.01 的第二种子），日志 runs/chat3_seeds.log。
- **下一步（按顺序）**：
  1. 看 runs/chat_ft.log 末尾与生成的 runs/*_dspn-L_chat.json，在 docs/experiments/e41-.md 登记；`python dspn_text.py sample dspn --size L --tag _base2 --extra "<overnight.json 里的 extra>"` 看生成样本。
  2. 对话微调：正在跑（见上“正在跑”）；完成后跑生成样本。
  3. 聊天 demo：`python tools/chat.py runs/dspn-L_chat.pt --size L --extra "<同上>"`（命令 /new 新会话只留持久记忆、/sleep 睡眠、/save 存权重、/reset 全清）；测试脚本已写好 [tools/chat_scripts/memory.txt](../../tools/chat_scripts/memory.txt)（会话内召回、/new 后跨会话召回、/sleep 后召回、模板外的“我哥哥叫…”），用 `--script tools/chat_scripts/memory.txt` 跑并把记录登记进实验记录，检验会话内与跨会话记忆。
  4. demo 之后：TSPH 在 S 档对照（`python tools/train_tsph.py --size S --data tiny --batch 16 --T 1024 --tag _x1`，比 E38 字节级 1.247），再按设计 10.5 做记忆与遗忘对比。
  5. 自动任务队列见 [v2.md](v2.md)“任务队列”（2026-11-20 用户授权全自动推进，主线为自己的架构放大）。其后按 v2.md 的放大路线做：S 档配方对照 → 装融合内核 → 1000 万参数同墙钟筛选（含第 13 章两级方案与标准混合）→ 胜者 3000 万参数两天。
  6. 目标已定为“可用模型”（2026-11-19 用户确认；参数可以大，单次训练可跑约两天）：按 [design/candidates/13-usable-model.md](../design/candidates/13-usable-model.md) 13.6 的顺序做——装 triton-windows 与 flash-linear-attention 并做一次数值对照 → 约 1000 万参数三组各 2 小时同墙钟筛选（单级 / 两级固定边界 / 两级动态分块，基线 dspn-L_base）→ 开关自上而下预测 → 胜者放大到 3000 万–6000 万参数跑两天，再接对话续训与同一套聊天脚本。与第 4 项的 TSPH 对照可合并进同一轮筛选。
- 已定：不复现 E13 的莎士比亚结果。
- 第 4、5 阶段第一版的在线学习用“块内截断反传、只更新在线参数”的工程简化；设计中的对角痕迹与逐层反馈在 demo 跑通后替换，替换前后做对照。顺序：先对 c 的对角线性递推换成资格痕迹（可精确等于梯度，在线不存历史），再对 MLP 与跨层部分试逐层反馈或本地预测误差（预测编码）；每步比新词 bpc、旧数据遗忘与内存（理由见 design/current/3-learning.md 3.1）。

## 快速路线（2026-11-19 用户决定：少跑实验，尽快做出能对话、有记忆、会学习的 demo；优先于下表的逐阶段验收）

1. 不再扫描，直接定配置：S 档最佳比例（2 个大模块、16 维消息，E32、E33）放大到约 3–5M；保留中枢；工作空间只留基础版（E34 持平，用户要求保留；稀疏更新会锁死，不开）；睡眠方法取第 10 轮最好者。跳过 M 档形状扫描与工作空间消融。
2. 修完代码审查（2026-11-19）指出的在线学习与睡眠缺陷：中枢偏移并入、睡眠不训低秩偏移、验收与报告探针分开、睡后刷新池内优先级。
3. 对话数据：现成对话集（SODA 或 DailyDialog）+ 模板生成的“记忆对话”（对话中给事实、稍后提问），格式“用户：… / 模型：…”，只在模型轮次算损失。
4. 训练只跑两次：L 档 TinyStories 预训练一个通宵；对话数据上继续训练 1–2 小时。
5. 聊天程序：命令行对话；只从用户的话在线学习；/sleep 巩固、/new 新会话；演示跨会话记住用户教的事。
6. 研究项（工作空间长上下文与层级统计、M 档扫描、资格痕迹、预测编码、选择性学习的其余信号）推到 demo 之后。
7. 据独立设计提案（design/candidates/9-redesign-candidate.md）提前采纳：中枢出口加宽到每模块 32 维、中枢 4 头 dk64 异质时间常数（含近似持久头）、T=1024、每次评估存检查点。对话微调阶段再加：记忆对话插入“会话边界”（边界处清会话状态、保留持久头）、聊天时以快记忆为主记事（低秩梯度留作对照）。demo 之后按 9.4 的实验 A、B 决定是否改两级主干。词级层（9.2）应用户要求在通宵前实现（dspn_text.py word_*），先做 S 档对照再定是否用于通宵。
8. 文献与讨论的采纳排期见 [adoption.md](adoption.md)。
9. demo 之后实现净室对照方案 TSPH（design/candidates/10-tsph-cleanroom.md），按 10.5 的 X1、X2 与本项目比较；其自上而下预测、惊讶门控更新的测量方法（10.4）同样用于本项目。

## 迭代规矩

- 一次只改一个因素，先在 0.3M 上试；1 个种子筛选，有希望的补到 3 个种子再下结论；胜出的才放大。
- 每次运行把 JSON 存进 runs/ 并在 docs/experiments/ 最新分段文件登记一行。
- 门槛在开始前写进上表，不按结果回头改。
- 每个阶段结束时请 Codex 独立审一次代码和结论，逐条核实后写“采纳 / 不采纳 + 理由”。
