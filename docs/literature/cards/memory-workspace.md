# 实现卡：快记忆规则、工作空间与训练目标

“[V]”= 打开原文或代码确认；“[S]”= 搜索片段。目录见 [README](../README.md)。

检索日期 2026-11-19。本项目 S 档每模块 d=32、快记忆 m=16，中枢 dk=32（[dspn_text.py](../../../dspn_text.py) 的 `SIZES`）。伪代码里标“推导”的是本卡自己的改写，原文没有。

## 1. Palimpsa 元可塑性（arXiv 2602.09075）

**公式**（式 5，[V]）：`I_t = α_t I_{t−1} + (1−α_t) I_prior + β_t ⊗ k_t²`；`μ_t = α_t (I_{t−1}/I_t) ⊙ μ_{t−1} + (1/I_t) ⊙ [(β_t⊙v_t) ⊗ k_t]`；α_t = exp(−A·d_t)。μ 是状态均值，I = 1/σ² 是逐元素（d_v×d_k）重要性，β_t ∈ R^{d_v}。元可塑性比 (I_max−I_min)/I_min 在 MQAR 上升到约 40 [V]。
代码里的等价形式（`fused_palimpsa_recurrent_fwd_kernel`，[V]）：`M ← e·M + v kᵀ；Ī ← e·Ī + b k²ᵀ；μ = M/(Ī+Ip)；o = (μ q)·scale`，另输出方差 `Σ q²·scale/I`。**是归一化的加性 GLA，没有 delta 擦除项**；论文称 Palimpsa-D 可替换 Gated Delta Rule，但没写出 delta 形式（未确认）。

**超参**（`palimpsa/layers/palimpsa.py`，[V]）：`Ip_log=0` 且不训练（Ip=1）；`b = sigmoid(b_proj(b_rank_proj(x)))·softplus(b_scale)·bs`，秩 128；`bs = sigmoid(bs_proj(x))` 同时乘 v；`A_log` 初值 log U(0,16)；精确核块长 16；默认 head_dim 256。

**规模**（[V]）：170M/15B token 常识均分 43.96 vs GDN 43.59，但 Wiki PPL 29.73 vs 29.16（更差）；760M 52.27 vs 51.50，PPL 19.02 vs 19.06；2.7B Mamba2 换 8 层 +0.84、LongBench 20.4 vs 19.3。760M 下 Palimpsa-M 比 Mamba2 慢 3 倍以上。

**代码**：[djo1996/Palimpsa](https://github.com/djo1996/Palimpsa) 类 `Palimpsa`；`palimpsa/ops/fast_palimpsa/chunk_fast_palimpsa.py` 的 `fast_palimpsa_ref`（纯 PyTorch 参考）；MQAR 用 `zoology/mixers/palimpsa.py` [V]。

**映射**：`Layer.forward` 中 β 之后、`delta_chunk` 之前。b 取每模块标量时 I 只随键维变化，可当作 delta 写入的逐键维学习率（推导），与第 2 条同构：
```
b = sigmoid(einsum(c, w_b) + b_b)                 # 并入合并 einsum 的第 6 列
I = scan(exp(g), b[...,None] * k**2, I0) + 1.0    # (b,T,M,m)，衰减与 S 共用
r, S = delta_chunk(qv, k, v, beta, g, S, kw=k / I) # 写键与擦键分开，见第 2 条
```
**风险**：逐元素 I（b 按值维）使写入不再是秩 1，WY 分块失效；S 衰减 τ≈63 拍时 I 很快饱和，元可塑性比小；SEP 处 I 也要清；论文 170M 的 PPL 反而更差。**需要重训**；不单独采纳，并入第 2 条。

## 2. 预条件 DeltaNet（arXiv 2604.21100）

**公式**（式 9，[V]）：
```
A_t = α^P_t A_{t−1} + β^P_t (k_t⊙k_t)                     # 每键维累计平方键
r_t = log A_t − μ;  s_t = r_t / (1+|r_t|);  k̃_t = exp(−log(x)·s_t) ⊙ k_t   # 压到 [1/x, x]
S_t = S_{t−1} − β_t (S_{t−1}k_t − v_t) k̃_tᵀ;  o_t = S_t q_t   # 擦用 k，写用 k̃
```
PGDN/PKDA 把 S_{t−1} 换成 α_t S_{t−1} [V]。

**WY 兼容**：可以。A 是逐维前缀和，与 S 无关；UT 变换保持 `T = (I + tril(diag(β) K K̃ᵀ, −1))⁻¹ diag(β)`，状态更新用 K̃ [V]。精确逆 Gram（Sherman–Morrison）不能分块，所以只用对角 [V]。约 10% 吞吐开销 [V]。

**超参**（`fla/layers/precond_gated_deltanet.py`，[V]）：`squash_x=1.5`（论文要求 x≤2）、`squash_eps=1e-6`；μ = 每头 `log_atk_scale`，初值 −0.2，不衰减；预条件有自己的 `A_log_atk`、`dt_bias_atk`、`a_atk_proj`、`b_atk_proj`，与主递推绑定会变差 [V]。

**规模**（SlimPajama，单次运行，[V]）：340M/15B：DN→PDN Wiki PPL 29.04→29.05（无变化）；GDN→PGDN 27.08→26.86，检索均分 24.77→26.17；KDA→PKDA 26.18→25.98。1B/50B：GDN→PGDN 18.05→18.01，常识 55.44→56.38。

**代码**：[ntumm120/preconditioned-deltanet](https://github.com/ntumm120/preconditioned-deltanet) `3rdparty/flash-linear-attention/fla/ops/precond_gated_delta_rule/naive.py` 的 `naive_recurrent_precond_gated_delta_rule`（可直接抄）、`chunk.py` 的 `chunk_precond_gated_delta_rule`；层 `PrecondGatedDeltaNet` [V]。

**映射**：`Layer.forward` 与 `delta_chunk`（中枢 `Hub.forward` 共用）。
```
ga, ba = einsum(c, W)[..., 新增两列]                    # 合并 einsum 再加 2 列
Aacc = scan(exp(F.logsigmoid(ga + b_ga)), sigmoid(ba)[...,None] * k**2, A0)
s = (log(Aacc + 1e-6) - mu) ; s = s / (1 + s.abs()); kt = k * exp(-log(1.5) * s)
# delta_chunk 内：A = ((kb @ ktᵀ) * decay) 严格下三角；attn = (q @ ktᵀ) * decay
#                Sp += (kt * tail)ᵀ @ W；Kc（块首状态的预测）仍用 k
```
**风险**：`init_state` 多一个 (b,M,m) 状态，SEP 处清零，T=1 与整段要一致；增益只在带门变体上出现（我们是带门的），且量级小，S 档可能淹没在种子方差里。**需要重训**。

## 3. In-Place TTT（arXiv 2604.06169）

**公式**（3.1–3.2 节，[V]）：快权重是门控 MLP 的 W_down；`W^(i) = W^(i−1) + η V̂_[i]ᵀ Z_[i]`（内损失为负内积，纯 Hebb，无擦除），`O_[i] = Z_[i] W^(i)ᵀ`，块 i 的写入只影响之后的块。目标 `V̂ = Conv1D(X_0) W_target`，X_0 为 token 嵌入；W_target = I 且卷积在下一 token 处为 1 时就是下一 token 目标。

**超参**：卷积逐通道、核 5、零初值、无偏置；W_target 稀疏对角初值；块长 512–1024 最好 [V]。代码里 `padding=2`（居中，非因果，靠块延迟保证不泄露）、`ttt_lr` 默认 0.3、`ttt_chunk` 默认 8192；README 推荐 `ttt_lr: 3`、`ttt_chunk: 4096`、每 6 层一层；`ttt_target` 从头训用 `input_embed` [V]。

**规模**（[V]）：500M–4B 从头训，Qwen3-4B / LLaMA-3.1-8B / Qwen3-14B 续训。Qwen3-4B RULER 64k 74.3→78.7；4B 从头训 RULER-16k 6.58→19.99。

**代码**：[ByteDance-Seed/In-Place-TTT](https://github.com/ByteDance-Seed/In-Place-TTT) `hf_models/hf_qwen3/modeling_qwen3.py` 的 `Qwen3MLP.forward`（`ttt_conv`、`ttt_proj`、cumsum）与 `_resolve_ttt_target_states` [V]。

**映射**：我们的 v = c·V 是自身状态投影（重构式）。按“写入目标对齐下一字节”改 `Layer.forward` 的 v 与 k（推导，因果版：键延迟一拍，值取当前字节嵌入，读时 q_t 自然对准 t+1）：
```
k_w = cat([k_last, k[:, :-1]], 1)                # 状态里多存上一拍 k
v_w = v + einsum(causal_conv5(e), W_target)      # e 需传入每层；卷积零初值 → 起步等于原模型
r, S = delta_chunk(qv, k_w, v_w, beta, g, S)
```
**风险**：原文收益来自 ≥500M、块 4096 的长上下文；m=16、T≤1024 无证据；要把嵌入 e 送进每层。**需要重训**。

## 4. 共享全局工作空间（Goyal 等 2103.01197）+ Phua（2512.19155）

**写入**（[V]）：`Q̃ = M W̃^q`（查询来自当前槽位）；`M ← softmax(Q̃ (R W̃^e)ᵀ/√d_e) R W̃^v`，R 的行是各专家状态；硬竞争为按 softmax 前得分取前 k 个专家（逐槽还是全局取，未确认）。RMC 式门：`X̄ = mean_i relu(X_i W¹)`，`K = X̄ + tanh(M^{t−1})`，`I = σ(K Wᴵ)`，`F = σ(K Wᶠ)`，`M^t = I⊙tanh(M) + F⊙M^{t−1}`。
**广播**（[V]）：`q̂_k = h_k Ŵ^q`，`s_kj = softmax(q̂_k (m_j Ŵ^e)ᵀ/√d_e)`，`h_k ← h_k + Σ_j s_kj m_j Ŵ^v`，再走各专家自己的更新。
**超参/数字**（[V]）：RIMs 4 槽、4 头、键值 32；Transformer 任务 8 槽，CATER k=5；WikiText-103 2 槽 k=2。CATER top-1 70.83→71.07（硬）/71.33（软）；弹球 ARI 0.276→0.915；Atari 迁移 1.13/1.11。代码：论文称在补充材料，未找到公开仓库（未确认）。

**Phua**（[V]）：**没有点燃 / 门槛机制**，“点燃锐度”只是事后指标（工作空间范数时间导数峰值）；写入是放进第一个空槽，满了覆盖槽 0；K=4、D=16（与我们相同）。容量：4 槽 99.5%、2 槽 28.1%、0 槽 0%。噪声：写入后、读出前向槽位加高斯噪声，L75（正确率降到 75% 的 σ）只有约 0.04；dropout+权重衰减 0.033、训练时加噪 0.035，都没救；带自我模型的代理 L75 > 0.5。CNN + GRU 64，MiniGrid。无代码。

**映射**（`DSPNLayered._ws_write`）：我们的 `ws_q` 是静态参数，SGW 由槽位生成；我们用单独 `ws_sal` 选前 k，SGW 直接用得分；我们是标量 `ws_keep`(+`ws_ignite`)，SGW 是逐元素 I/F 门；广播我们把 4 维拼进下一块输入 u，SGW 加到状态上。
```
sc = einsum('bsk,bnk->bsn', self.ws_qm(slots), self.ws_k(tok)) / 4
if self.ws_topk: sc = sc.masked_fill(sc < sc.topk(k, -1).values[..., -1:], -inf)
cand = softmax(sc, -1) @ self.ws_v(tok)
Kx = relu(self.ws_w1(tok)).mean(1, keepdim=True) + tanh(slots)
slots = sigmoid(Kx @ Wi) * tanh(cand) + sigmoid(Kx @ Wf) * slots
# 评测（不重训）：_ws_write 返回前 slots += σ·randn，扫 σ 求 bpc / MQAR 的 L75
```
**风险**：E34、E35 已显示基础版与完整版都不优于无工作空间，稀疏更新会锁死；Phua 说加噪训练不能修复脆弱。写入改动**需要重训**；噪声测试不需要。

## 5. 自我建模辅助损失（Premakumar、Graziano 等 2407.10188）

**做法**（[V]）：输出层同时输出分类 logits 和对选定层激活的回归；`L = w_c·CE + (w_s/n)·‖â − a‖²`。目标是**当前**前向的激活，不是下一拍；是否对 a 停梯度没写，文中说 â 和 a 都是权重的函数（未确认）。
**超参/数字**（[V]）：MNIST 两层 MLP 隐藏 64–512、AW ∈ {1,5,10,20,50}；CIFAR-10 ResNet18 AW ∈ {0.5,1,2}；IMDB AW ∈ {100,500}。准确率基本不变（IMDB 略升），最小网络在 AW 20/50 崩；收益是权重分布变窄、RLCT 降低。文中无具体数字，只在图里。代码：未找到（[S]，未确认）。Phua 的自我模型：64 维潜变量压缩 89 维输入，元认知 AUROC 0.92 vs 去掉后 0.50，主任务 88.5 vs 88.2 [V]。

**映射**（推导）：在 `_ws_write` 末尾加小头，用本块 tok 预测下一块 slots；误差当不确定度喂回。预测值要放进 ws 状态元组，才能保持逐拍与整段一致。
```
pred = self.sm_head(tok.flatten(1))                  # (b, K·w)
err = ((prev_pred - slots.detach().flatten(1))**2).mean(-1)   # prev_pred 来自状态
self.aux = self.aux + w_sm * err.mean()
reads = cat([reads, err[:, None, None, None].expand(b, L, M, 1)], -1)   # 或缩放下一块 β
```
已有 `ws_pc`、`td_pc` 是同类写法，可复用。
**风险**：原文好处是变简单而非更准，0.3M 模型本就容量不足，大 AW 可能伤主任务；停梯度与否决定是否有正则效果。冻结主干只训小头作不确定度探针**不需要重训**；作为训练正则**需要重训**。

## 6. Nested Learning / Hope（2512.24695）

**连续谱记忆 CMS**（[V]）：`y_t = MLP^(f_k)(…MLP^(f_1)(x_t))`；第 ℓ 级只在块边界更新：`θ^(ℓ)_{i+1} = θ^(ℓ)_i − Σ_{t=i−C^(ℓ)}^{i} η^(ℓ)_t f(θ^(ℓ)_t, x_t)`（i ≡ 0 mod C^(ℓ)，否则不变），`C^(ℓ) = max_i C^(i)/f_ℓ`。Hope = 自指 Titans（k、v、η、α 都由自适应记忆产生，只有 W_q 固定，记忆为两层 MLP 残差、初值元学习）+ CMS [V]。
**超参**：具体频率 / 块长没给数，只提到“最低频率 2K / 512”分组（未确认）。
**数字**（[V]）：760M/30B Wiki PPL 18.68 vs Titans 20.08；1.3B/100B 14.39 vs 15.60；去 CMS 13.04 vs 12.24。
**代码**：无官方代码，只有非官方复现（[S]）。

**映射**：现有三级 = S（每字节）→ `OnlineLearner` 的 LoRA（每 64 字节一步 AdamW）→ `sleep`（慢权重）。中间加“会话级 LoRA”：梯度在 C_sess 字节或一个会话内累加，到边界才走一步；睡眠时一起 `_merge_lora`。
```
# OnlineLearner.read：每块 backward 之后
self.opt.step()                                       # 现有快级
for p in sess: p.acc += p.grad                        # 中级累加（Σ_t η f）
if self.n_bytes % C_sess == 0 or 读到 SEP: sess_opt.step(用 acc); acc.zero_()
# sleep()：_merge_lora 同时合并 sess 的 lora_a/lora_b 并清零
```
**风险**：Hope 的多级是预训练里端到端学的，我们只在测试时用；秩 2 的两级 LoRA 可能互相抵消；要并入睡眠接受门。**不需要重训**（仅改 online.py）。

## 7. 延迟监督（Kim、Kothari，2609.32312）

**数据**（[V]）：TextWorld 轨迹，512 个世界；每个问题在“很长一段无关事件之后”才问，答案由模拟器给出；分“保持”（期间不变）与“修订”（旧值≠当前值）。每个探针从事件前缀**分叉**到一次性分支，答案不进入继续的事件流。要求间隔 ≥4096 token、≥16 次中间写入（DeltaNet 间隔中位 10,099）。
**损失**（[V]）：`L = α·L_NTP + β·L_QA`；L_QA 只算答案 token + EOS。DeltaNet (1, 0.0158)，RWKV-7 (1, 0.01)，LaCT (0.5, 0.5)。
**超参**（[V]）：**是在已有模型上的短微调**：128 次外层更新、每批 8 个世界、全模型 AdamW、峰值 lr 1e-5、余弦到 1e-6。
**数字**（[V]）：DeltaNet 1.3B BABILong 21.16→22.48、RULER 80.24→83.52（对照都是同数据 NTP）；RWKV-7 1.5B 22.60→27.20；LaCT 760M 25.60→31.08。
**代码**：文中只说“附带仓库”，无链接（未确认）。

**映射**：[tools/make_chat_data.py](../../../tools/make_chat_data.py) 的 `memory_dialogue` 与 `train` 的损失。
```
# 数据：事实与提问之间插 2–8 段 SODA（各隔 SEP），另写逐字节掩码文件：提问那轮 "Bot: " 后的回答字节 = 1
# 损失：ce = F.cross_entropy(..., reduction='none'); loss = (ce * (1 + w_qa * mask)).mean()，w_qa 扫 1–100
# 分叉（可选）：st = 跑事件流; for 探针: lg, _ = model(q_and_a, clone(st)); 只对回答算 ce，st 不继承
```
**风险**：`train` 每批随机截取、不跨批带状态，间隔超过 T 的事实永远看不到 → 需要按会话顺序喂并带状态（或拉长 T）；CUDA Graph 要求固定形状，掩码要做成固定地址张量。**不必从头重训**，在现有检查点上微调即可（原文就是这样）。

## 8. 元学习持续学习算法 ACL（Irie 等 2312.00276）

**目标**（式 6，[V]）：一条序列 = A 任务示例 → A 查询 → B 任务示例 → B 查询 → 再问 A 查询（状态已读过 B）。`L = −[log p(y_A | W_A) + log p(y_B | W_{A,B}) + log p(y_A | W_{A,B})]`；去掉最后一项，A 在学完 B 后掉到约 20%（随机）。代码 `main.py`：`loss = (loss_main_a·a_scale + loss_main_b·b_scale + loss_acl_a)·0.33`，A 查询用带状态的 `model(test_inputs_a, dummy, state)` 跑 [V]。
**超参**（[V]）：2 层 SRWM、每层 16 头，Conv-4 前端；5-way，每类 15 例（每任务 75 步）；两任务或三任务；`--use_acl`、`--loss_scale_task_a`；任务顺序每批交替。
**数字**（[V]）：Split-MNIST 域增量 5 任务：SGD 63.2，ACL 开箱 72.2，在 Omniglot 上用 5 任务损失元微调后 84.5（类增量 84.3 vs GeMCL 79.0）。无语言模型结果。
**代码**：[IDSIA/automated-cl](https://github.com/IDSIA/automated-cl) `main.py` 两任务训练循环（“ACL part, evaluate forgetting”段）[V]。

**映射**：与第 7 条共用数据生成器：同一会话内先说事实组 A，再说事实组 B（同类问题、不同值，制造干扰），然后问 B、再问 A，只对回答字节算损失。
```
seq = facts(A) + ask(A) + facts(B) + ask(B) + ask(A)    # 可跨 SEP，考中枢持久头
loss = 三处回答字节 ce 各占 1/3（+ 其余字节 NTP 小权重）
```
**风险**：S 的 τ≈63 拍，B 会在快记忆里盖掉 A，此目标会把压力推给中枢慢头与衰减门，可能伤 bpc；原文只有图像少样本。**需要训练**：可先微调，效果好再进 v2 预训练配方。
