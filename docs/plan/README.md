# 排期（唯一施工入口）

做什么、按什么顺序、做到哪了，只写在本文件夹；当前状态和下一步只在本页。总入口见 [docs/README.md](../README.md)。

| 文件 | 内容 |
|---|---|
| 本页 | 当前状态、下一步、快速路线、迭代规矩 |
| [stages.md](stages.md) | demo 阶段表与过关条件、各阶段状态 |
| [adoption.md](adoption.md) | 文献与设计讨论的采纳决定和排期 |
| [research-lines.md](research-lines.md) | 研究线：形状扫描、模块粒度、层级、TraceCols、**路由模块池 RMP（P0–P6 与门槛）** |
| [acceptance.md](acceptance.md) | 验收项定义 1–10（原设计第 6 节，编号不变） |
| [v2.md](v2.md) | demo 之后：删什么、加什么、改什么，放大到 3000 万参数的路线与过关条件 |
| [fast-clean.md](fast-clean.md) | **现行**：训练提速与部件去重（重复部件清单、阶段 A / B、执行顺序） |

## RMP 线当前状态（2026-11-24 更新；E129 完成，上云训练与 BPE 多阶段评测已全部就绪）

- **真正站得住的核心资产**：
  - **138M 独立放大（E110）**：零 Attention `WaveNet` 无痛放大至 138M，纯白天 `smf_mlp` 在 OOD `chat` `drop=0.6` 下最差旧域遗忘仅 `+0.0185`（守住 `≤ 0.05` 红线）。
  - **多阶段真实持续学习（E128 → E129，2 种子）**：纯 `WaveNet + smf_mlp` 在 `task3 → SODA → Khan` 两阶段零重放顺序学习中，仅靠细粒度高竞争（`N=64, k=4` → `N=128, G=8, k=2`，每波 16 选 2 = 12.5% 激活）就将两阶段旧域最差遗忘砍半至 **`+0.1099`**（均值 **`+0.0625`**，比全参少忘 **17.3×**），阶段 1 对话净保留 **82%**；`smf_anchor`（`anchor_wd=0.05`）在 2/2 种子上温和胜出。
- **上云前工程清单（全部就绪 ✅）**：
  - [tools/train_rmp_lm.py](../../tools/train_rmp_lm.py)：已新增对齐 E129 12.5% 竞争比例的 `main_fine`（`G=12, N=192, k=2`，~242M 总参 / 61M 激活）与 `small_fine`（`G=8, N=128, k=2`，53.26M / 21.40M）；精确保存并恢复 `docs_seen` 防断线重复数据；本地 8GB 卡已跑通 `small_fine` 40→60 步冒烟与 `--resume` 断点续训（显存 2.1GB）。
  - [tools/eval_rmp_bpe_multistage.py](../../tools/eval_rmp_bpe_multistage.py)：已支持直接加载 `latest.pt` / `best.pt`（含 `shared > 0` 冻结）跑 `chat → code` 两阶段 SMF 评测，离线全流程自检通过。
- **已测完的旁路与可选增强（不绑死主线）**：
  - **E125–E126（A2 合成题 + 海马）**：方差大（4.2% vs 54.2%）、未训稳、离真实持续学习目标较远，暂停。
  - **E127（A3 权重共享加深 `max_think=3`，3 种子同算力对照）**：在 4.2M 真实 `task3` 上 3/3 种子胜过 `k1_iso_param`（`-0.058 bpc`）与 `k1_iso_flops`（`-0.116 bpc`），并复现熵相关停机（`+0.337`）与跨拍动态招募不同专家（`tick01_jaccard=0.603`）；留作可选增强，不混入首轮大训练。
- **下一步（上 Colab A100 / L4 / 4090 执行路 A）**：
  1. 拉取当前干净 commit，先跑 2 分钟测实测吞吐与 CU 消耗，再按预算定 `--tokens`（100 CU 约 8 亿 token；250–300 CU 约 20–25 亿 token）训 `main_fine`（GPT-2 BPE + FineWeb-Edu）。
  2. 训完后跑 `tools/eval_rmp_138m.py`（四合一）与 `tools/eval_rmp_bpe_multistage.py`（`chat → code` 多阶段持续学习）。

### 历史阶段表（E63–E109 基准线，供回溯）

路由模块池 RMP（[第 23 章](../design/candidates/23-routed-pool.md)）：测试该结构能否涌现层级与分工、抗遗忘与在线学习、随规模变强、好训练。**不和 DSPN 比**，只和 RMP 变体比并设绝对门槛。代码 [pool.py](../../pool.py)（`WaveNet`）、[online.py](../../online.py)、[tools/rmp_online.py](../../tools/rmp_online.py)。

| 阶段 | 状态 | 核心结论与实验 |
|---|---|---|
| P0 实现 | 完成（固定分波切片版） | `03f400f` 新增 `Layer.forward_slice`，固定分波省 3.6× 显存且保持 `state_dict` 兼容 |
| P1 好训练 | **过** | 不发散、负载均衡、无死模块（E63） |
| P2 分工涌现 | **过** | 慢语境路由（`rule='ctx'`）+ 高竞争让来源选择性从 0.003 飙升至 0.32 → 0.555 → 0.650（E90、E98、E104）；`03f400f` 已修复旧版共享编码层 EMA 漏洞，支持逐波 `wave_ctx=True` 与多尺度衰减 |
| P2b 连接规则 | 持平 / 耗时过 | 语言建模上地址 / 哈希 / 线性门在种子噪声内持平（E73、E74、E77）；路由耗时过关（E82） |
| P3 在线学习与抗遗忘 | **待最终门槛验证** | 已排除纯冻结路由（E91，与稠密同忘）与四版新奇分配（E69、E71、E80、E81、E95）；转向 `ctx + freeze_enc + SMF`。E99–E103 暴露出 `rank=2` 欠学习假象及 SMF 动量漏改、`bg_bytes` 探针泄露、`sleep()` 漏冻编码层/未选中模块等漏洞，均已于 `03f400f` 修复并新增 `params='pool_mlp'` 与 `smf_sleep` |
| P4 规模 | 部分（已解锁 N≥96） | N16 → N24 → N48 bpc 单调降（E78）；切片加速已解决 OOM，待 P3 过关后补 N=96 |
| P5 层级 | **时间尺度层级已过** | `bottleneck=True + read_last=False + wave_query` 无先验下跨多种子/数据/规模 100% 自发长出单调递增 $\tau$ 层级（T=512 达 3.7 倍，E97、E100、E102–E105）；`fb=64` 带来 −0.07 bpc 增益（E105），已修会话边界未重置漏洞（`03f400f`） |

- **正在跑**：`tools/queues/rmp_sleep10h.ps1` 第 1 项 `_whf_tiny_n48k2_long`（修复前启动，评估该检查点需显式传 `"wave_ctx":false`）；紧接的第 2 项 `_whf_n48k2_long512` 将自动使用新代码。
- **下一步计划（2026-11-22 定稿，聚焦唯一核心门槛）**：
  1. **第一道定生死门槛（留一来源 + 同收益比较 + `pool_mlp`）**：
     - 构造 `task3`（故事 / wikiHow / 简明维基三源预训练，留出第 4 源对话作 OOD 考题），采用 `N48k4`（E103 甜点，bpc 比 `N48k2` 低 0.13）作统一底座；
     - 改造评测脚本：对 3 个旧来源分别测遗忘（避开 `task4-valid.src` 前 4KB 全是 SODA 的问题），画 `chat_drop → old_rise` 同收益曲线并报告最差旧来源；
     - 同一底座对比 `lora (r=2)` vs `pool_mlp`（昼夜统一 `freeze_enc + smf_sleep`）；
     - **过关门槛**：对话 held-out bpc 实降 `0.5` 时，**最差旧来源的 bpc 上升 ≤ 0.05**（若达不到 `0.5` 则报告实际可达上限）。
  2. **架构细化消融（视第 1 步结果推进）**：
     - 多尺度逐波路由 `wave_ctx=True` + `ctx_decay=[0.5, 0.85, 0.95, 0.99]`（注意 `ema_scan` 不可填 `0.0`），验证浅快深慢功能层级与 bpc 追回；
     - 修复 `fbv` 会话边界重置后单独复跑 `fb=64`，复核 E105 的 −0.07 bpc 收益；
     - 若第 1 步同收益遗忘仍超 0.05，再引入“常驻共享专家（共享 1 + 路由 k−1，对齐总激活模块数）”与“深波限定可塑（白天与睡眠同掩码）”。
  3. **过关后放大与迁移**：
     - 借切片加速补跑 `N=24 / 48 / 96`（P4 规模验证）；
     - 迁移至 Colab 挂载 **0.5B 开源 LLM**（冻结 LLM 作通用底座 + 旁路 RMP 记忆池 + SMF `pool_mlp`，测 RAG 不擅长的技能/风格持续学习）。
  4. **暂缓项**：手写稀疏 Triton / 分组 GEMM 内核（切片已省 3.6× 显存）、接入 [episodic.py](../../episodic.py)（属文本级 RAG，留待聊天 demo）、新奇分配/分裂变体、越深越慢先验（E88）、深度生长调参（E75）。
  - 交接注意：项目规矩见 CLAUDE.md；`runs/tc-*.json` 属 TraceCols 线，check_docs 对其报错为既有项。

## 当前状态（2026-11-20 晚，交接用；新窗口先读 fast-clean.md 和第 20 章再实施）

- 工作方式：用户授权全自动推进（git 存档、结果为先、只和自己的上一版比，见 [v2.md](v2.md)“用户定的方向”）。仓库已是 git，每步本地提交。
- 环境：conda 环境 `dspn`（Python 3.11，PyTorch 2.11 + CUDA 12.8，RTX 5060 Laptop 8 GB，Rtxpt.exe 已关，可用约 7 GB）。已装 triton-windows 3.6、fla-core 0.6、einops（均 --no-deps）。
- 代码要点（v0.18，2026-11-20；v0.16 在 git 标签 v0.16）：[dspn_text.py](../../dspn_text.py)（模型与训练；完整配置在 SIZES；选项 chunk（动态分块，惊讶跃升定边界）、attn_layer、local、grad_ckpt；训练默认 --stateful、WSD、bf16，只报流式 bpc，可选 --compile、--bot_loss）、[hub.py](../../hub.py)、[online.py](../../online.py)（在线 lora / eprop；睡眠：上下文蒸馏 + 旧语料锚定）、[episodic.py](../../episodic.py)（海马式情景记忆）、[tools/chat.py](../../tools/chat.py)（--episodic）、[tools/analyze_hierarchy.py](../../tools/analyze_hierarchy.py)。设计见 [design/README.md](../design/README.md)（v0.18）。
- 数据：TinyStoriesV2；对话 data/chat3-*（当前，记忆对话 40%、带重现行），chat2-*（无重现行），chat-*（旧，

 换行，别再用）。make_chat_data.py 生成。
- **demo**：最好的对话模型 runs/dspn-L_chat3.pt（E50；chat4 与之持平，E52），只能在 git 标签 v0.16 的代码下运行（`git worktree add ../dspn016 v0.16`）。试用：`python tools/chat.py runs/dspn-L_chat3.pt --size L --extra "<tools/queues/overnight.json 里的 extra>" --episodic`；/new /sleep /save /quit。跨会话约 5/8（靠情景记忆），名字复制常错、人名宠物名会混。
- **XL5 基线已完成（2026-11-20 22:38，E62）**：恒定段跑到 7.7 万步后经用户同意提前停，接 1.55 万步衰减收尾，流式 bpc 1.4405，检查点 runs/dspn-XL5_v2d.pt。切词与时间尺度层级涌现，照抄与记名字没有（结构瓶颈）。GPU 空闲。训练新增 `--resume`（每 500 步存 runs/<名字>_resume.pt）、`--sched decay`、`--skip`。
- **已采纳进下次预训练**（[v2.md](v2.md)“已确认采纳”）：nb_count（E53）、--sched wsd（E53）；pred_self=0.01（E51）在 v0.18 删除，由 C5 把关决定是否恢复。不采纳：out_gate、conv、pred_nb、local、surprise（E45、E49、E53）。
- **v2 架构已选定**（2026-11-20，[design/candidates/20-dspn-v2.md](../design/candidates/20-dspn-v2.md)）：删持久头与网格代码，工作空间并入中枢块末广播，新奇信号管海马写入与在线可塑，睡眠加事实蒸馏（海马教皮层）。
- **已完成（2026-11-20，v0.17，E54、E55）**：fast-clean A0–A5 与 B1（删重复部件、窗口级广播、fla 全内核、bf16、torch.compile），动态分块（chunk=N，默认关），在线 e-prop 的代码与精确性测试（tests/test_eprop.py）。L 档稳态约 17 万字节/秒（E41 的约 2.4 倍），显存约 2 GB。动态分块快约 24% 但 bpc 差 0.065，暂不用。
- **2026-11-20 午**：用户决定直接上约 3000 万参数（XL 档，分块 + 滑窗注意力，第 20.8 节第 2、3 档都已融合）、英文混合数据（data/mix，48 亿字节，E57）。XL 首次试跑发散，已加 pre-norm 修好（E57）。用户出门，**预训练尚未启动**，GPU 空闲（另有一个会话在跑 tracecols 实验，runs/tc-*.json 不是本线的）。
- **2026-11-20 下午**：v0.18 精简方案定稿（[第 22 章](../design/candidates/22-dspn-lean.md)，经独立审查修订：中枢保留，删广播、分块路由 4 件套改惊讶定边界、预测头族改编码层局部头、删冷启动评估）。用户：先写文档，不测、不实施。
- **2026-11-20 傍晚**：用户让先做 v0.18。C1–C4 已实施（删广播、预测头族、可导路由、冷启动评估；惊讶跃升 + 最大块长分块），42 个测试全过；C 档冒烟 1000 步不发散、编译后约 13.6 万字节/秒、切词约 2.4 倍基准率（E58）。旧 v0.17 检查点不能加载（设计快照 archive/design_v0.17）。
- **C5 把关已跑（E59）**：v0.18 过关（bpc 差 0.015、快约 14%、涌现无明显退化）。删中枢（D 组）与 v0.18 持平，但本轮 MQAR 饱和、测不到长程，删不删等长间隔 MQAR。复制与 MQAR 本轮无区分力。
- **下一步（按顺序）**：
  0. 下一版 v0.19（用户已同意方向）：字节级抄写通路（注意力能看到字节细节，或加宽层间消息）+ 残差流替代 8 层拼接读出 + 提速（训完前先在 XL5_v2d 上剖析一次）。在同样约 9.25 万步、同样衰减下与 E62 比 bpc、复制、记名字；胜者再跑长训练。C6 新奇信号在选定底座上做。
  1. （已被第 0 条取代）原 XL 预训练命令：`python dspn_text.py train dspn --size XL --data mix --batch 8 --T 1024 --lr 6e-4 --compile --iters <按实测速度定> --tag _v2`。先跑 1000 步看编译后的字节/秒，再定 iters 使总时长约 48 小时；每 500 步存检查点。
  2. 新底座上做 B3（新奇信号接入聊天）、B4（E31 协议比 lora 与 e-prop，输的删）、B5（事实蒸馏），然后对话微调（chat3，--bot_loss）与记忆脚本，与 dspn-L_chat3 比。
  3. 每个检查点跑涌现指标（时间尺度层级、自发切词、复制相变等，v2.md 第 14 条）。
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
