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
| [fast-clean.md](fast-clean.md) | **现行**：训练提速与部件去重（重复部件清单、阶段 A / B、执行顺序） |

## 当前状态（2026-11-20 晚，交接用；新窗口先读 fast-clean.md 和第 20 章再实施）

- 工作方式：用户授权全自动推进（git 存档、结果为先、只和自己的上一版比，见 [v2.md](v2.md)“用户定的方向”）。仓库已是 git，每步本地提交。
- 环境：conda 环境 `dspn`（Python 3.11，PyTorch 2.11 + CUDA 12.8，RTX 5060 Laptop 8 GB，Rtxpt.exe 已关，可用约 7 GB）。已装 triton-windows 3.6、fla-core 0.6、einops（均 --no-deps）。
- 代码要点（v0.17，2026-11-20 重写；v0.16 在 git 标签 v0.16）：[dspn_text.py](../../dspn_text.py)（模型与训练；完整配置在 SIZES；选项 chunk（动态分块）、bcast（窗口级广播）、pred_self/pred_down、local、grad_ckpt；训练默认 --stateful、WSD、bf16，可选 --compile、--bot_loss）、[hub.py](../../hub.py)、[online.py](../../online.py)（在线 lora / eprop；睡眠：上下文蒸馏 + 旧语料锚定）、[episodic.py](../../episodic.py)（海马式情景记忆）、[tools/chat.py](../../tools/chat.py)（--episodic）、[tools/analyze_hierarchy.py](../../tools/analyze_hierarchy.py)。设计见 [design/README.md](../design/README.md)（v0.17）。
- 数据：TinyStoriesV2；对话 data/chat3-*（当前，记忆对话 40%、带重现行），chat2-*（无重现行），chat-*（旧，

 换行，别再用）。make_chat_data.py 生成。
- **demo**：最好的对话模型 runs/dspn-L_chat3.pt（E50；chat4 与之持平，E52），只能在 git 标签 v0.16 的代码下运行（`git worktree add ../dspn016 v0.16`）。试用：`python tools/chat.py runs/dspn-L_chat3.pt --size L --extra "<tools/queues/overnight.json 里的 extra>" --episodic`；/new /sleep /save /quit。跨会话约 5/8（靠情景记忆），名字复制常错、人名宠物名会混。
- **正在跑**：无（GPU 空闲）。
- **已采纳进下次预训练**（[v2.md](v2.md)“已确认采纳”）：pred_self=0.01（E51）、nb_count（E53）、--sched wsd（E53）。不采纳：out_gate、conv、pred_nb、local、surprise（E45、E49、E53）。
- **v2 架构已选定**（2026-11-20，[design/candidates/20-dspn-v2.md](../design/candidates/20-dspn-v2.md)）：删持久头与网格代码，工作空间并入中枢块末广播，新奇信号管海马写入与在线可塑，睡眠加事实蒸馏（海马教皮层）。
- **下一步（2026-11-20 用户定：先提速、再让部件配合好并去重）**：按 [fast-clean.md](fast-clean.md) 的执行顺序做。
  1. A0：打 git 标签 v0.16，用现行代码跑 L 档 1500 步基准，作为提速起点。
  2. A1 / B1：删死代码与重复部件（持久头、网络内惊讶门控、pred_nb、词级、td、卷积、输出门、网格邻接、GRU / Transformer 对照）。
  3. A2：删工作空间，去掉每窗 16 段的块串行；广播改为窗口级（下层读中枢上一窗口末的读出）。
  4. A3–A6：全部换 fla 融合内核（含慢通道扫描）→ bf16 → torch.compile → 激活检查点加大批。
  5. B6 S 档确认 → v2 预训练 → B3–B5（新奇信号、在线通道去重、事实蒸馏）→ 对话微调。
- 已定：不复现 E13 的莎士比亚结果。不在用户的话上做睡眠交叉熵（E44）。

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
