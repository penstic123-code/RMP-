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
- 代码要点（v0.18，2026-11-20；v0.16 在 git 标签 v0.16）：[dspn_text.py](../../dspn_text.py)（模型与训练；完整配置在 SIZES；选项 chunk（动态分块，惊讶跃升定边界）、attn_layer、local、grad_ckpt；训练默认 --stateful、WSD、bf16，只报流式 bpc，可选 --compile、--bot_loss）、[hub.py](../../hub.py)、[online.py](../../online.py)（在线 lora / eprop；睡眠：上下文蒸馏 + 旧语料锚定）、[episodic.py](../../episodic.py)（海马式情景记忆）、[tools/chat.py](../../tools/chat.py)（--episodic）、[tools/analyze_hierarchy.py](../../tools/analyze_hierarchy.py)。设计见 [design/README.md](../design/README.md)（v0.18）。
- 数据：TinyStoriesV2；对话 data/chat3-*（当前，记忆对话 40%、带重现行），chat2-*（无重现行），chat-*（旧，

 换行，别再用）。make_chat_data.py 生成。
- **demo**：最好的对话模型 runs/dspn-L_chat3.pt（E50；chat4 与之持平，E52），只能在 git 标签 v0.16 的代码下运行（`git worktree add ../dspn016 v0.16`）。试用：`python tools/chat.py runs/dspn-L_chat3.pt --size L --extra "<tools/queues/overnight.json 里的 extra>" --episodic`；/new /sleep /save /quit。跨会话约 5/8（靠情景记忆），名字复制常错、人名宠物名会混。
- **正在跑（2026-11-20 18:24 起，约 38–40 小时）**：XL5 预训练 `python dspn_text.py train dspn --size XL5 --data mix --iters 1000000 --batch 8 --T 1024 --lr 4e-4 --compile --tag _v2`（独立进程，日志 runs/xl5_v2.log，检查点 runs/dspn-XL5_v2.pt 每 500 步覆盖、最好的另存 _best.pt）。17:49 的首跑在第 2500 步发散，修复后重开（E61）。
- **已采纳进下次预训练**（[v2.md](v2.md)“已确认采纳”）：nb_count（E53）、--sched wsd（E53）；pred_self=0.01（E51）在 v0.18 删除，由 C5 把关决定是否恢复。不采纳：out_gate、conv、pred_nb、local、surprise（E45、E49、E53）。
- **v2 架构已选定**（2026-11-20，[design/candidates/20-dspn-v2.md](../design/candidates/20-dspn-v2.md)）：删持久头与网格代码，工作空间并入中枢块末广播，新奇信号管海马写入与在线可塑，睡眠加事实蒸馏（海马教皮层）。
- **已完成（2026-11-20，v0.17，E54、E55）**：fast-clean A0–A5 与 B1（删重复部件、窗口级广播、fla 全内核、bf16、torch.compile），动态分块（chunk=N，默认关），在线 e-prop 的代码与精确性测试（tests/test_eprop.py）。L 档稳态约 17 万字节/秒（E41 的约 2.4 倍），显存约 2 GB。动态分块快约 24% 但 bpc 差 0.065，暂不用。
- **2026-11-20 午**：用户决定直接上约 3000 万参数（XL 档，分块 + 滑窗注意力，第 20.8 节第 2、3 档都已融合）、英文混合数据（data/mix，48 亿字节，E57）。XL 首次试跑发散，已加 pre-norm 修好（E57）。用户出门，**预训练尚未启动**，GPU 空闲（另有一个会话在跑 tracecols 实验，runs/tc-*.json 不是本线的）。
- **2026-11-20 下午**：v0.18 精简方案定稿（[第 22 章](../design/candidates/22-dspn-lean.md)，经独立审查修订：中枢保留，删广播、分块路由 4 件套改惊讶定边界、预测头族改编码层局部头、删冷启动评估）。用户：先写文档，不测、不实施。
- **2026-11-20 傍晚**：用户让先做 v0.18。C1–C4 已实施（删广播、预测头族、可导路由、冷启动评估；惊讶跃升 + 最大块长分块），42 个测试全过；C 档冒烟 1000 步不发散、编译后约 13.6 万字节/秒、切词约 2.4 倍基准率（E58）。旧 v0.17 检查点不能加载（设计快照 archive/design_v0.17）。
- **C5 把关已跑（E59）**：v0.18 过关（bpc 差 0.015、快约 14%、涌现无明显退化）。删中枢（D 组）与 v0.18 持平，但本轮 MQAR 饱和、测不到长程，删不删等长间隔 MQAR。复制与 MQAR 本轮无区分力。
- **下一步（按顺序）**：
  0. XL5 预训练进行中（E59、E60 定的结构：v0.18、无中枢、K = 1.5T/N）。训完：涌现五项（tools/emergence.py）、与 XL / C 档比规模曲线；C6 新奇信号在新底座上做。
  1. （已被第 0 条取代）原 XL 预训练命令：`python dspn_text.py train dspn --size XL --data mix --batch 8 --T 1024 --lr 6e-4 --compile --iters <按实测速度定> --tag _v2`。先跑 1000 步看编译后的字节/秒，再定 iters 使总时长约 48 小时；每 500 步存检查点。
  2. 新底座上做 B3（新奇信号接入聊天）、B4（E31 协议比 lora 与 e-prop，输的删）、B5（事实蒸馏），然后对话微调（chat3，--bot_loss）与记忆脚本，与 dspn-L_chat3 比。
  3. 每个检查点跑涌现指标（时间尺度层级、自发切词、复制相变等，v2.md 第 14 条）。
  4. 路由模块池（[第 23 章](../design/candidates/23-routed-pool.md)，测试用）：XL5 训练期间写 P0 代码与 CPU 测试，训完按 [research-lines.md](research-lines.md)“路由模块池线”P1–P6 跑。
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
