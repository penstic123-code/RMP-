# 实现卡：快记忆层与优化器

“[V]”= 打开原文或代码确认；“[S]”= 搜索片段。目录见 [README](../README.md)。

检索日期 2026-11-19。本项目尺寸：每模块 d=32、快记忆 m=16（[dspn_text.py](../../../dspn_text.py) 的 `SIZES`），远小于下列论文的头维 128。

## 1. 生产级 delta 层配方（KDA / Gated DeltaNet / Qwen3-Next）

**公式**（[Kimi Linear](https://arxiv.org/abs/2510.26692) 式 1、10，[V]）：
- `S_t = (I − β_t k_t k_tᵀ) Diag(α_t) S_{t−1} + β_t k_t v_tᵀ`，`o_t = S_tᵀ q_t`；α_t ∈ [0,1]^{d_k}，每个键维一个衰减率（GDN 每头一个标量）。
- `o = W_o( sigmoid(W_g↑ W_g↓ x) ⊙ RMSNorm(KDA(q,k,v,α,β)) )`，门低秩。
- 代码里的对数衰减：`g = −exp(A_log) · softplus(f_proj(x) + dt_bias)`，`f_proj` 是秩=头维的两层线性；另有 `g = lower_bound · sigmoid(exp(A_log)(f+dt_bias))` 变体（`safe_gate`）[V]。
- q、k 在核内 L2 归一化（`use_qk_l2norm_in_kernel=True`），β = sigmoid(b_proj(x))（核内），`allow_neg_eigval` 时 β×2 [V]。
- q/k/v 各自过短因果深度卷积 `ShortConvolution(conv_size=4, activation='silu', bias=False)` [V]。

**默认超参**（`fla/layers/kda.py`，[V]）：head_dim=128；conv_size=4；`A_log` 初值 log U(1,16)（每头）；`dt_bias` = softplus⁻¹(对数均匀 0.001–0.1，下限 1e-4)，每通道；`A_log`、`dt_bias` 不做权重衰减；输出门 `g_proj` = Linear(h, head_v_dim) → Linear(head_v_dim, value_dim, bias=True)；`o_norm = FusedRMSNormGated(activation='sigmoid', eps=1e-5)`。
对照：`fla/layers/gated_deltanet.py` 默认 `output_gate_activation='swish'`、head_dim=256、`A_log` 初值 log U(0,16)，衰减每头一个 [V]。KronLSTM 层把输出门偏置初值设 3（sigmoid(3)≈0.95，起步近似无门）[V]。

**规模与数字**（Kimi 表 1，16 层同 FLOPs，[V]）：完整 3:1 混合 验证 PPL 5.65；去短卷积 5.70；去输出门 5.67；Swish 门 5.81。q/k 归一化无消融，只说为特征值稳定 [V]。缩放律模型 653M–1.7B 激活参数、38.8B–128B token [V]。[Qwen3-Next](https://www.alibabacloud.com/blog/602580)：GDN:注意力 3:1；注意力输出门消除 attention sink 与巨激活；零中心 RMSNorm 且 norm 权重加权重衰减 [V]；它的门是否 sigmoid，博客未写（未确认）。

**代码**：[fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) `fla/layers/kda.py` 类 `KimiDeltaAttention` [V]；门 `fla/ops/kda/gate.py` 的 `naive_kda_gate` / `kda_gate_fwd_kernel`；纯 PyTorch 参考 `fla/ops/kda/naive.py` 的 `naive_recurrent_kda`、`naive_chunk_kda`；带门 RMSNorm `fla/modules/norm/fused_norm_gate.py` [V]。

**映射到本项目**：`Layer.forward` 中 `k, v, qv, bl, gl = einsum(c, W)` 之后到 `delta_chunk` 之前；`delta_chunk` 的 `G`/`decay` 计算；输出 `r` 进 W1 之前。中枢 `Hub.forward` 同理（`g` 已是每头标量）。
```
# __init__: w_gam (M,d)→(M,d,m)，b_gam (M,)→(M,m) 按 1.5–1000 拍对数铺开；新增 conv (M,3m,4)、Wg (M,d,m)、bg=3
k, v, qv = causal_conv4(cat[k,v,qv]).silu().split(m)     # 沿 T，每模块独立
k, qv = F.normalize(k,-1), F.normalize(qv,-1) * m**-0.5
g = -exp(A_log) * softplus(einsum(c, w_gam) + dt_bias)    # (b,T,M,m) 通道级
r, S = delta_chunk_kda(qv, k, v, beta, g, S)              # decay 改为逐键维
r = rms_norm(r) * sigmoid(einsum(c, Wg) + bg)
```
`delta_chunk_kda`：块内衰减从标量 Γ_t/Γ_s 变成逐维 exp(G_t − G_s)，A 矩阵要算 `(k_t ⊙ exp(G_t−G_s)) · k_s`，参照 `naive_chunk_kda` 的逐列循环。

**小规模风险**：
- 逐维衰减下 exp(−G) 在 64 拍块里会溢出；fla 用两两差 `(g − g_i).exp()`，显存为 (b,M,n,C,C,m)，S 档 b=32、T=256 时每层约 0.27 GB（估算），4 层加反传可能逼近 3 GB；可把块长降到 16。现有 `scan` 已有同类溢出说明。
- 短卷积引入 3 拍状态：`init_state`、逐拍推理与 `online.py` 的分块 `read` 都要带卷积缓存，否则 T=1 与整段结果不一致（违背 hub 文件头的约定）。
- m=16 时低秩门无意义，直接用满秩 (d,m)。
- 收益数字全来自 ≥0.6B 模型，PPL 差 0.05 量级，S 档上可能淹没在种子方差里。

**需要重训**。

## 2. Muon 优化器

**更新规则**（[KellerJordan/Muon](https://github.com/KellerJordan/Muon) `muon.py`，[V]）：
```
m ← lerp(m, grad, 1−0.95);  u = lerp(grad, m, 0.95)        # Nesterov
X = u.bf16() / (‖u‖_F + 1e-7);  若行>列先转置
5 次: A = X Xᵀ;  B = b·A + c·A·A;  X = a·X + B·X   (a,b,c)=(3.4445, −4.7750, 2.0315)
u = X · max(1, rows/cols)^0.5
p ← p·(1 − lr·wd) − lr·u
```
结果不是精确 UVᵀ，而是奇异值约在 U(0.5,1.5) 的近似正交（docstring）[V]。`zeropower_via_newtonschulz5` 支持批量（`ndim>=2`，对最后两维）[V]。

**用在哪些参数**：只用于隐藏层权重矩阵；输入嵌入、最终输出层、增益与偏置用 AdamW；卷积核展平成 2D 后可用 Muon [V]。

**超参**：`Muon` lr=0.02、momentum=0.95、wd=0、ns_steps=5；`MuonWithAuxAdam` 中 Adam 组默认 lr=3e-4、betas=(0.9,0.95)、eps=1e-10、wd=0；README 示例两组都 wd=0.01 [V]。lr 单位是“每步谱范数” [V]。

**规模与数字**：README 链到 124M Transformer 上 AdamW/Shampoo/SOAP/Muon 对比（modded-nanogpt）[V，数字未读]。350M、15B token 下 KDA + Muon 最好、Muon 一贯优于 AdamW，见 [fast-memory.md](../topics/fast-memory.md) 的 Cross-Layer Routing 条目。0.3–5M 字节级模型无公开结果（未确认）。

**代码**：`muon.py` 的 `zeropower_via_newtonschulz5`、`muon_update`、`SingleDeviceMuonWithAuxAdam`（单卡用这个）[V]。

**映射到本项目**：`param_groups` 与 `train` 中 `torch.optim.AdamW(...)` 一行。
```
mat = [p for k,p in ps if p.ndim == 3 and k.split('.')[-1] in {'B','G','K','V','Qm','W1','W2','P'}]
mat += hub 的 Wq/Wk/Wv/Wo（2D）
rest = 其余（emb、readout、bias、b_*、w_beta/w_gam (M,d)、b_a、lora_*）
opt = SingleDeviceMuonWithAuxAdam([dict(params=mat, use_muon=True, lr=0.02, weight_decay=0.01),
                                   dict(params=rest, use_muon=False, lr=lr, betas=(0.9,0.95), weight_decay=...)])
```
同层参数堆叠成 [M,a,b]，正好被当作 M 个独立矩阵正交化 [V，按 `ndim>=2` 的批量写法]；缩放用 size(−2)/size(−1)，与 einsum 'mud' 的 (输入,输出) 布局相反，行列意义要核对。

**小规模风险**：
- `train` 把整步录成 CUDA Graph、学习率是张量 `lr_t`；Muon 的 `p.add_(..., alpha=-group["lr"])` 和 Adam 部分的 Python 步数计数在录制时会被冻结成常数（读代码推断，未实测）→ 要么关掉图，要么改写成张量 lr / 张量步数。
- `w_beta`、`w_gam` 是 (M,d) 二维，按 `ndim>=2` 筛会误入 Muon；必须按名字分组。
- 32×16 这类小矩阵上 bf16 的 5 步迭代精度、W2 零初值（梯度秩低）下的行为，无先例。
- `param_groups` 里不衰减门偏置的约定要原样保留到 Adam 组。

**需要重训**（优化器决定训练轨迹，只能 S 档对照）。

## 3. Kronecker Delta Rule / KronLSTM

**公式**（[arXiv 2610.26834](https://arxiv.org/abs/2610.26834) 式 3、4、10，[V]）：
- `S_{t+1} = S_t − β_t (v_tᵀ S_t k_t − c_t) v_t k_tᵀ`：只修正 (k,v) 这一对的标量关联，方向固定为 v kᵀ。
- 向量化：`s_{t+1} = (I − β a aᵀ) s + β c a`，`a = k ⊗ v`，即以 a 为键、标量 c 为目标的 Widrow–Hoff delta 规则。
- 门：β = σ(w_βᵀx+b_β)（擦除），c = σ(w_cᵀx+b_c)（写入），遗忘门逐键维，读出 `h = W_out RMSNorm(o ⊙ S q)`；q、k、v 都 L2 归一化，前接短卷积 + SiLU [V]。
- 代码里的一步（`fla/ops/kronlstm/README.md`，[V]）：
```
S̃ = Diag(exp g) S;  y = kᵀ S̃ v;  β* = sβ/‖v‖², βmax = s/‖v‖²   (s=2 允许负特征值，否则 1)
u = βmax · c·‖v‖² − β* · y          # 解耦：β 只管擦，c 只管写；‖v‖=1、s=1 时 u = c − β y
S = S̃ + u · k vᵀ;   o = scale · qᵀ S   (scale = K^−0.5)
```
- 干扰：对已存的 (k_i,v_i) 影响 ∝ (v_iᵀv)(kᵀk_i)，键或值任一正交即受保护；β‖a‖² ≤ 1 时记忆范数期望有界，无需归一化器 [V]。
- 分块训练：转移矩阵是广义 Householder，DeltaNet 的 WY/UT 分块照用，只需 `a_iᵀa_j = (k_iᵀk_j)(v_iᵀv_j)`，不必构造 KV 维特征 [V]。

**超参**（`fla/layers/kronlstm.py`，[V]）：conv_size=4；`normalize_v=True`；`use_target_gate=True`；`decouple_erase=True`；`decay_gate='vector'`（GDN-2 式通道衰减 `−exp(A_log)·softplus(f+dt_bias)`）；输出门 bias 3；head 128；chunk 内核固定块长 64、头维 16–256。

**规模与数字**（[V]）：1.5B、100B token FineWeb-Edu、8k 上下文。纯线性：Avg-6 59.72 vs GDN 57.73，LAMBADA PPL 11.72 vs 12.58；混合 SWA：59.56 vs 57.99；RULER NIAH 16k MQ 12.7 vs 8.6。吞吐比 GDN-2 快 11–18%，但比 GDN 慢。400M 结果在附录 A6（数字未读）。小于 400M 无结果。

**代码**：[NX-AI/KronLSTM](https://github.com/NX-AI/KronLSTM)（fla 的分叉）：`fla/ops/kronlstm/naive.py` 的 `naive_recurrent_kronlstm`、`naive_chunk_kronlstm`（纯 PyTorch，可直接抄）；`chunk.py` 的 `chunk_kronlstm`；层 `fla/layers/kronlstm.py` 的 `KronLSTM` [V]。安装会顶替 fla，只在 H100/A100 测过 [V]。

**映射到本项目**：`delta_chunk` 内部与 `Layer.forward` 的 β/写入门。
```
v = F.normalize(v, -1); c = sigmoid(einsum(c_state, w_c) + b_c)       # 新增写入门 (M,d)
# delta_chunk 里：
A   = β ⊙ (k kᵀ) ⊙ (v vᵀ) ⊙ decay   (严格下三角)                      # 原为 β (k kᵀ)⊙decay
rhs = [c(写入,标量), β·Γ·a 的标量投影]  → 解出每拍标量 u_t
out = Γ·S₀q + ((q kᵀ)⊙decay⊙mask) @ (u ⊙ v)
```
（逐拍分解按上面 README 推出，块间状态 Sp 的修正项需照 `naive_chunk_kronlstm` 核对，未逐行验证。）

**小规模风险**：
- m=16 时 v 被归一化，每对只存一个标量强度，写入的“值”信息只剩方向；容量是否够用在 16 维上无证据。
- 论文只有 ≥400M；“常胜 GDN”在字节级小模型上未确认。
- 引入第三个门 c，每模块多 d+1 参数，可忽略。

**需要重训**。

## 4. 现代自指权重矩阵（SRWM）

**公式**（[arXiv 2202.05780](https://arxiv.org/abs/2202.05780) 式 5–8，[V]）：
```
y_t, k_t, q_t, β_t = W_{t−1} φ(x_t)          # 同一矩阵同时产生输出、键、查询和 4 个学习率
v̄_t = W_{t−1} φ(k_t);   v_t = W_{t−1} φ(q_t)
W_t = W_{t−1} + σ(β_t) (v_t − v̄_t) ⊗ φ(k_t)
```
φ 为输出为正且和为 1 的逐元激活（如 softmax）；监督任务对 x 不加 φ，RL 用 softmax [V]。β ∈ R⁴，分别给 y、q、k、β 四个子矩阵 [V]；多头各自独立。

**超参**（`supervised_learning/layer.py` 的 `SRWMlayer`，[V]）：每头 `W_y, W_q, W_k` 为 (d_head,d_head)、`w_b` 为 (d_head,4)，初值 N(0, 1/√d_head)；`w_b` 均值 `beta_init=-1.0`（注释称 0 更好）；外接 `out_linear` + 残差 + LayerNorm；`stateful=True` 时返回相对基权重的偏移作状态。

**规模与数字**（[V]）：Mini-ImageNet 5-way 1/5-shot 47.0/61.4（DeltaNet 47.0/62.7），模型 3.6M；ProcGen 6 环境测试 20.0 vs DeltaNet 18.6，968K 参数、两层 128 维；速度 1,700 vs DeltaNet 2,300 步/秒。**无语言建模结果**，原文列为未来工作 [V]。

**代码**：[IDSIA/modern-srwm](https://github.com/IDSIA/modern-srwm) `supervised_learning/layer.py` 的 `SRWMlayer`；CUDA 核 `supervised_learning/self_ref_v0/self_ref_v0.cu`（核内对 q、k 做 softmax、对 β 做 sigmoid）；RL 版在 `reinforcement_learning/torchbeast/`（`self_ref_v0`、`self_ref_v1`）[V]。后续持续学习版在 IDSIA/automated-cl [V，未打开]。

**映射到本项目**：替换 `Layer.forward` 里由慢权重 `W = cat[K,V,Qm,w_beta,w_gam]` 产生 k/v/q/β 的那一步——让快记忆自己产生这些量。最小版只做“快记忆生成写入键与 β”：
```
x = c                                                   # (b,T,M,d)，需 d==m 或先投影
k_t, q_t, β_t = (S_{t−1} + W0) @ softmax(x_t)           # W0 为慢权重初值，S 存偏移（同 stateful 写法）
v_t, v̄_t = (S+W0) @ softmax(q_t), (S+W0) @ softmax(k_t)
S_t = S_{t−1} + σ(β_t)(v_t − v̄_t) softmax(k_t)ᵀ
```
**小规模风险**：
- 键依赖上一步 S，`delta_chunk` 的块内三角求解不再成立，只能逐拍循环（或其 CUDA 核）；纯 Python 逐拍在 T=256 上慢一个数量级以上（估算），且破坏现有 CUDA Graph 加速。
- 论文里 SRWM 对 DeltaNet 的优势很小（少样本持平或略差，RL 测试 +1.4）且无 LM 证据；投入产出比低。
- 不带衰减门，长序列需另加遗忘。

**需要重训**。建议排在 1、3 之后，仅作小实验。
