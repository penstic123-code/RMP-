# 实现卡：自上而下、层级与对话数据

“[V]”= 打开原文或代码确认；“[S]”= 搜索片段。目录见 [README](../README.md)。

检索日期 2026-11-19。本项目对照点：E38、E40（[e30-.md](../../experiments/e30-e40.md)）在约 0.36M 参数下，词级层和字节级自上而下层同参数都变差。代码位置都在 [dspn_text.py](../../../dspn_text.py) 的 `DSPNLayered`。

## 1. Full-bandwidth transformer（[arXiv 2608.08888](https://arxiv.org/abs/2608.08888)）

**公式**：`u_t = RMSNorm( W_H h^L_{t−1} ⊙ σ(W_G RMSNorm(e_t)) )`，W_H、W_G ∈ R^{D×D}。上一步顶层状态走值通路，本 token 嵌入只当门；u_t 替换（不是加到）第 0 层输入 [V，式 4 与代码]。
**训练（保持并行）**：K 遍。第 1 遍是普通前向；第 k 遍把上一遍全部位置的顶层输出右移一位，融合后再整段并行跑一遍，串行只发生在遍与遍之间。`L = L_1 + λ·mean_{k≥2} L_k`，λ=1；不 detach，后一遍的损失反传进前一遍（detach 省显存但收益小很多）[V]。每篇文档开头和随机长度的前缀保持普通嵌入（prefix mixin）[V]。顶层状态加 U(−0.02, 0.02) 抖动（论文写 σ=0.02）[V]。
**后期引入**：约前 75% 只用单遍；200B 那次为 75% 单遍、22% 两遍、3% 三遍，3% 三遍让递归稳定；100B、400B 为 75% 单遍、25% 三遍 [V]。多遍阶段越长越好；单遍与多遍交错训练不如分阶段 [V]。
**超参**：NorMuon lr 1e-2（矩阵）、Adam 5e-4；WSD；上下文 8192；W_H、W_G 均匀初始化 std=1/√D [V]。
**规模与数字**：只有 1B（另有 950M 的融合方式对照，三种融合差不多）；100B 两遍约等于 200B 普通，约当 1.5 倍 token；解码开销可忽略 [V]。
**代码**：[xidulu/Full-bandwidth-transformer](https://github.com/xidulu/Full-bandwidth-transformer) `nanochat/gpt.py`：`LatentFeedback.forward`（`gate_product`）、`build_feedback_mask`、`GPT.forward(num_forward_passes=…)` [V]。
**映射**：`DSPNLayered._run` 开头的 `u = e[...]`；逐拍推理时上一拍顶层输出已在状态里（`td` 分支的 `new.append(src[:, -1])`）。
```
out1, hs1 = run(e, state0)                        # 第 1 遍（现有 _run）
prev = cat([state.top_prev, hs1[L-1][:, :-1]], 1)   # 顶层 M·d，右移一拍
fused = rms(W_H(prev) * sigmoid(W_G(rms(e))))       # W_H: M·d→emb，W_G: emb→emb
x0 = where(plain_mask, e, fused)                    # SEP 之后与随机前缀用 e
out2, _ = run(x0, state0); loss = ce1 + ce2
```
S 档 W_H+W_G 约 7k 参数（M·d=180 时，估算），不占主干，正是 README 第 11 条的“后期引入、不占主干”。
**小规模风险**：它给 Transformer 补的是“跨时间的状态”，我们的每层已有 c、S，收益可能小得多（未确认）；训练算力×2；`train` 的 CUDA Graph 要把两遍一起录。
**需要重训**：不必从头，可从现有检查点接着训最后 25%。

## 2. Latent Lattice LM（[arXiv 2611.21608](https://arxiv.org/abs/2611.21608)）

已确认存在：Duggal、Wu、Freeman、Torralba、Zhang、Shechtman，2026-11-16 提交 [V]。
**公式**（每层水平潜流 r^ℓ，`n = Norm_in(h^{ℓ−1}_t)`，[V]）：
```
Δ = softplus(W_Δ n + b_Δ);  g = exp(−exp(a) ⊙ Δ);  c = SiLU(W_r n + b_r)
r^ℓ_t = g ⊙ r^ℓ_{t−1} + (1 − g) ⊙ c          # 系数只看本拍输入 → 关联扫描并行
q = (W_q ĥ) ⊙ (1 + tanh(W_γ r̂)) + W_β r̂      # k、v 同式（写入 KV 缓存）；MLP 的 gate/up 加 β(r̂)；FiLM 权重零初始化
Jacobi 反馈：b^{ℓ0}_t = (1−g)⊙(c + N_t)，N_t = SwiGLU(RMSNorm(r^{top}_{t−1}))（上一遍的结果）
读出：softmax(W_U(P_v Norm(h^N) + P_r Norm(r^{top})))，P_r 零初始化，只在 d=256 时用
```
训练用 1 遍 + 2 遍 Jacobi（J3），约 2.6 倍 FLOPs；推理可串行精确展开，基本无额外 FLOPs [V]。
**超参**：d_r=256（主模型），24 层，上下文 2048；AdamW lr 1e-3、β(0.9,0.95)、wd 0.1、批约 1.05M token [V]。
**数字**：406M、20B token：PPL 15.41→14.54，加注入与读出流 14.40，J3 14.03（训练 1.85 GFLOP/token，基线 0.71）[V]。GDN、Mamba-2 上改进 0.3–2.6% [V]。最小 400M [V]。
**代码**：无。项目页写 “coming soon” [V]。
**映射**：它的水平流就是我们 `Layer.forward` 里的 `c = scan(a, inp, c0)`，形式几乎相同，我们已有。新的只有 Jacobi 反馈：
```
# Layer 0：inp = (1 − a) * (gb + N)，N = swiglu(rms(top_prev))，top_prev 取上一遍顶层 h 右移一拍
```
与第 1 张卡同一机制，只是加在门控候选上而不是替换输入。
**小规模风险**：主体收益来自给 Transformer 加递归状态，我们已有；无代码，细节只能按论文写。
**需要重训**：反馈项零初始化后可接着训（推断），FiLM 部分对我们没有意义。

## 3. LLM-JEPA（[arXiv 2509.14252](https://arxiv.org/abs/2509.14252)）

**公式**：`L = Σ_ℓ CE(Text_{<ℓ}, Text_ℓ) + λ · d(Pred(Enc(Text)), Enc(Code))`；Enc = 最后一层最后一个 token 的隐状态；Pred = 在输入后接 k 个 [PRED] token、权重共用，k=0 时为恒等；d 默认 1 − cos [V，论文与代码]。代码里两视图是对话的 user 消息与 assistant 消息，分别单独前向（与完整序列拼成一批），两边都有梯度，没有 stop-grad [V]。另有 `--jepa_l2`、`--jepa_mse`、`--infonce`（温度 0.07）[V]。
**超参**：代码 `--lbd` 默认 0.1、`--predictors` 默认 0；论文网格 k∈{0..4}、λ∈{0.5,1,2,4}，部分任务到 1024 [V]。
**数字**：Llama-3.2-1B NL-RX-SYNTH 57.29→71.46，GSM8K 32.36→36.36；以微调为主，1B–8B [V]。
**代码**：[rbalestr-lab/llm-jepa](https://github.com/rbalestr-lab/llm-jepa) `finetune.py`：`RepresentationTrainer.forward`、`compute_loss` [V]。
**与我们的 td MSE 比较**：我们是时间上的预测（上层上一拍 → 下层本拍消息），MSE，目标停梯度，只训预测头，权重 0.1。目标不受推动，所以不会坍缩，但也不塑造表示。JEPA 是同一内容两视图对齐，两边都受推动，靠 LM 损失防坍缩。
**映射**：`train` 中 `(loss + model.aux).backward()` 之前，只用于对话数据。
```
hu = model.final_repr(user_bytes, zero_state)[:, -1]    # norm(cat(hs)) 的最后一拍
hb = model.final_repr(bot_bytes,  zero_state)[:, -1]
loss = ce + 0.1 * (1 - cos(hu, hb)).mean()
```
**小规模风险**：回答不是提问的同义改写，两视图不等价；没有小模型结果；每步多两次短前向。
**需要重训**：不必，可作为对话微调阶段的附加损失。

## 4. H-Net 动态分块（[arXiv 2507.07955](https://arxiv.org/abs/2507.07955)）

**公式**（[V]）：
```
p_t = ½(1 − cos(W_q x̂_t, W_k x̂_{t−1})),  p_1 = 1;  b_t = 1[p_t ≥ 0.5]
平滑：z̄_t = P_t ẑ_t + (1 − P_t) z̄_{t−1}           （块序列上的 EMA）
c_t = p_t if b_t else 1 − p_t;  STE(c) 前向为 1，反向梯度直接传给 c
上采样：out_t = STE(c_t) · z̄_{Σ_{k≤t} b_k};  z = out + Linear(x̂)（残差投影零初始化）
L_ratio = N/(N−1)·((N−1)FG + (1−F)(1−G)),  F = mean b,  G = mean p;  L = L_AR + α ΣL_ratio
```
官方实现：W_q、W_k 初始化为单位阵；EMA 用 `mamba_chunk_scan_combined`，dt = log(1/(1−p))，p 钳到 [1e-4, 1−1e-4] [V]。
**超参**：α=0.03，N=6（一级）或 (3,3)；外层学习率更高 [V]。
**数字**：760M、1.3B；最小的是 DNA 上 64M [V]。
**代码**：[goombalab/hnet](https://github.com/goombalab/hnet) `hnet/modules/dc.py`：`RoutingModule`、`ChunkLayer`、`DeChunkLayer`；`hnet/models/hnet.py`：`STE`、`HNet`（`residual_proj` 零初始化）[V]。我看过的文件里没有比例损失和训练脚本，README 只提到推理（官方是否放出训练代码：未确认）。
**社区复现**（[avikds/h-net-from-scratch…](https://github.com/avikds/h-net-from-scratch-dynamic-chunking-on-raw-bytes)，`model.py` + `scaffold.py`，[V]）：
- 函数：`RoutingModule`、`smooth_chunks`、`causal_upsample`、`straight_through_confidence`、`ratio_loss`、`lr_multipliers`（式 11：√(4.6·∏_{s..S}N/∏N·D_S/D_s)）、`wsd_factor`（预热 0.1、衰减 0.2、下限 0.1）。
- 设置：`run_arm`（lr 4e-3、批 16、长 128、α 0.03）；`arm_config` 中分层为 m2-[T3]-m2、宽 32/96、383,584 参数；纯字节为 m5、宽 48，只有 106,844 参数，按每字节 FLOPs（约 230–260k）对齐。TinyStories、3 种子。
- bits/byte 在 1200/4000 步：纯字节 1.587/1.451，空格规则 1.697/1.543，动态 1.732/1.585，去 EMA 1.753/–，定长池化 1.757/1.627。
- 结论：复现了“动态胜池化”“EMA 有用”“分级学习率有用”（N=8 时 1.741 对 1.790/1.900）；没复现“字节最差”“动态追平空格规则”；去掉余弦路由或 STE 在此预算下不变差。边界只有 14–30% 落在空格上。
**映射**：`_word` 的 `isb = self.wb[idx]` 换成路由；`held` 乘 STE(c)；词级输出先做 EMA。
```
p = 0.5*(1 - cos(Wq(zt_flat)[:,1:], Wk(zt_flat)[:,:-1])); p = cat([ones, p]); isb = p >= 0.5
aux += 0.03 * ratio_loss(p, isb, N=6)
hw = ema_over_chunks(hw, p_at_boundaries); held = held * ste(c)[..., None]
```
**小规模风险**：复现里用 3.6 倍少的参数，纯字节仍然最好；静态 K=T/2+1 要求边界不过半，单位阵初始化时相邻状态很像、边界少，起步不会溢出（推断）。
**需要重训**：是。

## 5. Scratchpad Patching（[arXiv 2605.09630](https://arxiv.org/abs/2605.09630)）

**patch lag**：patch 内除最后一个字节外，都只能用上一个 patch 的主干输出：`u_n = z̃_{ℓ−1} + x_n`（n ≠ e_ℓ）[V]。这正是我们 `_word` 的 `held`：词级输出保持到下一个边界。
**修法**（[V]）：在 patch 内部 `H_n > τ_SP` 的位置插临时状态 `z^t_ℓ = CrossAttn(q_ℓ, x_{s_ℓ:n})`，q 是已读字节的均值池化（比取最后字节略好）；过主干后广播给后续字节：`u_n = z̃^t_ℓ + x_n`。推理时用完即弃，KV 缓存只留定稿的 patch；训练时展开进序列，用专门的因果掩码并行（普通位置不看临时状态）。H_n 来自编码器上的 2 层熵头，在编码器输出处停梯度，与主损失等权联训。
**超参**：τ 从 {0.5…3.0} 中调，定长与 SpaceByte 1.5、熵切分 1.0、H-Net 2.5；推理时可改 τ，不用重训；lr 1e-3，批 1024×8192 字节 [V]。
**数字**：2B 时 16 字节/patch 的 NLU 48.0→54.2（字节基线 54.1）；500M 时 p=8 的 BPB 0.7833→0.7425，仍不如字节基线 0.7283；训练 FLOPs 是不加时的 1.41–1.74 倍；单种子 [V]。
**代码**：不公开（论文清单原话）[V]。
**映射**：`_word` + `Layer.forward`。不必用注意力：非边界处状态是冻结的，可以用本拍输入从冻结状态“偷看”一步而不提交。按 delta 规则 q 只用于读（读代码推断），所以能整段并行：
```
# 词级层在全部 T 个字节上跑，active = isb（非边界冻结 c、S）
c_commit = scan(a*act + (1-act), inp*act, c0)                # 现有写法
c_peek = where(isb, c_commit, a*c_commit_prev + (1-a)*gb)   # 用未屏蔽的 a、gb 向前一步
k,v,β,g 取自 c_commit；q 取自 c_peek → r = delta_chunk(q_peek, k, v, β, g, S)
held = h_peek 每拍都新；可加 τ：只在字节熵高处用 peek，其余用旧 held
```
**小规模风险**：词级层从 K≈T/2 个位置变成 T 个，算力约翻倍；500M 时它仍追不上纯字节，0.36M 下更难说。只在词级回归时才相关。
**需要重训**：是（词级本身要重训）。

## 6. TinyChat15M + TinyDialogues

**TinyChat15M**（[博客](https://nikhilr.io/posts/TinyChat15M/)，[starhopp3r/TinyChat](https://github.com/starhopp3r/TinyChat)，[V]）：
- 数据：GPT-4o-mini 生成 100 万段。`dataset/datagen.py` 每段抽 BASIC English 850 词表里的名词、动词、形容词各一个（`basic/basicwords.csv`），开头 ∈ {greeting, question, suggestion}，情绪 ∈ 7 种，结尾 ∈ {conclusive, reflective, open}。`usrprompt.txt`：User 与 Assistant 的闲聊共 6–10 句，每句 10–20 词，单段无换行，格式 `[INST] … [/INST] …`。`sysprompt.txt`：只用 BASIC English 和三四岁孩子懂的词，只准逗号句号，数字拼成英文词。BOS/EOS 包住每段对话；模型仍会替用户续写，所以推理时过滤标记 [V]。
- 模型：llama2.c，约 15M；`llama2.c/train.py` 默认 dim 288、6 层、6 头、长 256、批 128×累积 4、lr 5e-4、wd 0.1、预热 1000、`max_iters` 100000、Llama-2 词表 32000 [V]；博客写 20 万步、A100 约 5 小时 [V]，与默认值不一致，实际所用配置未确认。
- 下载：HF `starhopp3r/TinyChat` 的 `tinychat.txt`，745 MB，CC BY-NC 4.0 [V]。
**TinyDialogues**（[arXiv 2408.03617](https://arxiv.org/abs/2408.03617)，[V]）：GPT-4 Turbo，约 5 或 10 轮，围绕 2/5/10/15 岁孩子；提示要求词汇限于该年龄孩子能懂的、含指定名词/动词/形容词（取自 Wordbank CDI 或 AoA）、类型 ∈ {explanatory, functional, narrative, argumentative}，标签写成 `**Mom**:`，轮之间 `\n\n`。共 129,732 段、2880 万词；GPT-2 124M、lr 1e-4、20 轮。结论：合成数据胜 CHILDES；按年龄的全局课程几乎无影响；打乱对话内顺序会伤害（CHILDES 上明显）。数据：HF `styfeng/TinyDialogues` 的 `tinydialogue_train_ordered.txt`（141 MB，MIT）；文件里 `\n\n` 是字面的两个字符，多人对话 [V]。代码 [styfeng/TinyDialogues](https://github.com/styfeng/TinyDialogues)（生成脚本位置未确认）。
**make_chat_data.py 具体改动**：
1. 加 `tinychat_dialogues(path)`：按 `[INST]`、`[/INST]` 切分，转成 `User: …` / `Bot: …` 每行一轮；User 句常缺句号，补上。
2. 加 `tinydialogues(path)`：把字面 `\n\n` 换成换行，去掉引号；只留两人对话，Child 当 User，另一人当 Bot；多人对话丢掉或保留原角色名作一般语料（任选其一，先丢）。
3. `build` 加 `--mix`（例 tinychat 0.6、soda 0.15、memory 0.25），SODA 降为少数。0.3–5M 字节模型的瓶颈是语言难度，受限词汇优先。
4. 只打乱对话之间，对话内顺序不动（TinyDialogues 的结论）；SEP 已等于 BOS/EOS，保留。
5. 采样时遇到 `\nUser:` 就停（TinyChat 的“替用户续写”问题）。
6. 3000 步×16×1024 约 49 MB，按字节数截取子集即可；CC BY-NC 只用于研究。
**需要重训**：换数据要重训（或在现有检查点上接着训）。
