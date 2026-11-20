# 实现卡：睡眠、回放与可塑性

> 与主线 RMP 相关：第 6 张卡（标 【主线 RMP】），见 [23.10](../../design/candidates/23-routed-pool.md)。

“[V]”= 打开原文或代码确认；“[S]”= 搜索片段。目录见 [README](../README.md)。

检索日期 2026-11-19。四项都只改 [online.py](../../../online.py) 或在已训模型上操作，均不需要重训主干。

## 5. SuRe：惊奇优先回放 + 快慢 LoRA EMA

**规则**（[arXiv 2511.22367](https://arxiv.org/abs/2511.22367)，[V]）：
- 惊奇分 = 序列平均 NLL：`s_θ(z) = −(1/T) Σ_t log p_θ(z_t | z_<t, x)`；平均优于求和（LNT 74.22 vs 72.84）。
- 缓冲区总量为数据集的 2%，按任务等额配额 `m_i = ⌊S/d⌋`；每个任务取最惊奇的 top-p 入池，分数用快学习器算。
- 计分时机（表 3，LNT）：训练前计分、训练前更新 72.1；训练前计分、训练后更新 73.0；训练后计分 73.1——训练后更新更稳。
- 混合：每步 64 条新样本，每隔一步加 32 条回放（试过 1:2–1:16）。
- 双学习器：LoRA 挂在全部注意力层 W_Q、W_V，基座冻结；每步 `θ_slow ← β θ_slow + (1−β) θ_fast`；推理只用基座 + 慢 LoRA。

**超参**（附录 G，[V]）：LoRA 秩 8、α=32；lr 1e-3（T5-Large）/ 1e-4（Llama 3.1 8B）；批 64、1 epoch；β 试 0.985/0.99/0.995/0.999，示例用 0.995，0.999 明显差。

**规模与数字**（[V]）：主模型 T5-Large，Llama 3.1 8B 只跑两个顺序。LNT（多任务长序列，T5，顺序 4–6 平均）：Slow Surprise Replay 75.1，Surprise Replay（无 EMA）72.1，Slow Reservoir 72.5，O-LoRA 68.8，多任务上界 76.3。Standard CL：78.1 vs Slow Reservoir 77.8——标准 4 任务上惊奇选择几乎无增益，收益主要来自 EMA 慢学习器。

**代码**：论文页无代码链接（未确认是否有官方仓库）。

**映射到本项目**：`OnlineLearner._add` 已用块损失作优先级（相当于“训练前计分”）；`_refresh_pool` 在睡眠提交后重算（相当于“训练后更新”）——惊奇回放部分已具备。缺的是 EMA 慢学习器：`__init__` 建慢副本，`read` 每次 `opt.step()` 后更新，评测与 `_merge_lora` 用慢副本。
```
# __init__
self.slow = [p.detach().clone() for p in low]; self.ema = 0.995
# read(): opt.step() 之后
for s, p in zip(self.slow, low): s.lerp_(p.detach(), 1 - self.ema)
# 推理/探针前：交换 p.data 与 s；_merge_lora 并入 slow 的 a@b 而不是 fast
```
**小规模风险**：
- EMA 作用在因子 a、b 上，a@b 的 EMA 不等于 EMA 的乘积（论文同样如此）；我们的 `lora_a` 是随机初值、每次睡眠后重抽，快慢两份 a 必须同步重置。
- 我们每块 64 字节一步，一天几百到几千步；β=0.995 的半衰期约 140 步，短会话里慢副本几乎不动，需按步数重标定 β。
- 秩 2 只挂 W1（及中枢读路径），与论文秩 8 挂 Q/V 不同。

**不需要重训**。

## 6. 稀疏记忆微调（SMF）：类 TF-IDF 选槽
> 【主线 RMP】乘积键记忆层（23.8 T2）；按访问频率选槽 ≈ RMP 按负载选空闲模块。

**规则**（[arXiv 2510.15103](https://arxiv.org/abs/2510.15103)，[V]）：
- 记忆层（[Memory Layers at Scale](https://arxiv.org/abs/2412.09764)）替换 1.3B 模型第 12 层 FFN：1M 槽、4 个记忆头、每头 top-k=32、值维 1024、乘积键查表。
- 选槽分：`score(i) = c(i)/Σ_j c(j) · log[(|B|+1) / (Σ_{b∈B} 1[c_b(i)>0] + 1)]`；c(i) 为本批对槽 i 的访问次数，B 为 1000 个随机 DCLM 背景批。
- 只训分数最高的 t 个槽，其余停梯度：`mem = mem*mask + mem.detach() − (mem*mask).detach()`。

**超参**（[V]）：t=500（TriviaQA 事实）、10,000（文档 QA），扫 {25…1000}；优化器 SGD，lr 2–5（扫 {0.1, 2}）；批 64，序列 64/512。

**规模与数字**（[V]）：学 1K 条 TriviaQA 事实后 NaturalQuestions F1 下降：全量微调 89%、LoRA 71%、SMF 11%（同等新知识习得）。

**代码**：论文页无 SMF 仓库链接（未确认）。记忆层实现在 [facebookresearch/memory](https://github.com/facebookresearch/memory) `lingua/product_key/memory.py` [V，文件存在，内容未读]。

**映射到本项目**：本项目没有记忆槽层。最接近的“槽”是 `Layer` 的 MLP 隐藏单元（W1 的列 / W2 的行，每模块 hidden 个）或 M 个模块本身。改法放在 `sleep` 的慢权重训练里，作为梯度掩码：
```
bg = 背景语料上各隐藏单元激活>阈值的批计数（睡眠前算一次，存起来）
c  = 重放池上各单元激活计数
score = c/c.sum() * log((NB+1)/(bg+1));  keep = topk(score, t)
W1.grad[..., ~keep] = 0; W2.grad[:, ~keep, :] = 0;  其余慢权重冻结
```
**小规模风险**：
- 这是改编，不是原方法：论文的稀疏性来自 1M 槽中每 token 只访问 32 个；我们 GELU 隐藏单元稠密激活，“访问”要靠阈值定义，TF-IDF 区分度可能很差。
- S 档每模块 hidden=64，选 t 个单元后可写容量极小。
- 真要照搬需加一个乘积键记忆层，属于架构改动、要重训。

**改编版不需要重训；照搬记忆层需要重训**。

## 7. 间隔重复（SM-2）按困惑度排程回放

**规则**（[arXiv 2608.17530](https://arxiv.org/abs/2608.17530)，[V]）：
- 每条样本在正常前向（优化器步之前）记 `L_i = −(1/T) Σ log p`，`PPL_i = exp(L_i)`。
- 评分：`q_i = max(0, 5 − Σ_{k=1..5} 1[PPL_i ≥ τ_k])`，`0<τ_5<…<τ_1`；只给出 τ_5=50、τ_1=5000，τ_2–τ_4 未报告（未确认）。
- SM-2：`Δ(q) = 0.1 − (5−q)(0.08 + 0.02(5−q))`，`E ← max(1.3, E+Δ)`；q<3：n←0、I←1；否则 n←n+1，I = 1（n=1）/ 6（n=2）/ ⌈I·E⌉（n>2）；下次到期 d ← t + I（单位：训练步）。初值 n=0、E=2.5、I=1，初始到期时间错开。
- 组批：旧样本槽数 = min(到期旧样本数, ⌊ρB⌋)，ρ=0.2；其余用到期新样本，空余名额互补；损失只有交叉熵。

**超参**（附录 B，[V]）：AdamW (0.9,0.999)、wd 0.001、lr 1e-4 线性、预热 200；有效批 512 × 2048。阈值整体乘 α，α=1 最好。

**规模与数字**（[V]）：TinyLlama-1.1B、Llama-3.2-3B，Wikipedia 新旧快照与代码数据。旧知识 QA：TinyLlama 只训新数据 11.7% → SRT 49.0%；Llama-3B 43.0% → 51.6%。GSM8K（3B）：只训新 38.8%、均匀回放 6.8%、SRT 76.7%（基座 77.6%）。摘要的“5–37 个百分点”在表 1 里对不上下界（按表算约 6–37）。均匀回放把 GSM8K 打到 6.8% 很反常，基线可信度存疑。

**代码**：无。伦理声明称代码、日志、QA 题“专有、不发布”，附录又说“随代码发布”，自相矛盾 [V]。

**映射到本项目**：`OnlineLearner` 的池条目 `(优先级, 序号, 字节序列)` 扩成带 SM-2 状态；`sleep` 里 `self.rng.choice(len(self.pool), size=batch, p=w)` 换成按到期抽样。
```
item = dict(p=loss, n=0, E=2.5, I=1, due=step0 + jitter, seq=seq)
# sleep 每步：due = [i for i in pool if i.due <= t]
pick = sample(due_old, min(len(due_old), int(0.2*batch))) + 其余用池中新段
# 前向得每段 bpc → q = 5 − Σ 1[bpc ≥ τ_k]；按上式更新 E、n、I、due
```
**小规模风险**：
- 阈值 50/5000 是子词 PPL；字节级 PPL 只有个位数，必须重标定（例如用池内 bpc 分位数作 τ），这使“照搬 SM-2”变成新的超参搜索。
- 我们的睡眠只有约 200 步、池 64 KB（约 1000 段），间隔 6、⌈I·E⌉ 步很快超过整个睡眠长度，排程大部分时间退化为“只复习不会的”，与 SuRe 的 NLL 优先接近。可把时间单位改为“天/睡眠次数”。
- 无代码、基线反常，结论需要自己的对照（均匀 vs NLL 优先 vs SM-2）。

**不需要重训**。

## 8. FIRE：Newton-Schulz 等距重置

**规则**（[arXiv 2602.08040](https://arxiv.org/abs/2602.08040)，[V]）：
- `min_W̃ ‖W − W̃‖²_F  s.t.  W̃ᵀW̃ = I`，闭式解 `W̃ = W (WᵀW)^{−1/2}`（正交 Procrustes）。
- 近似：`X_0 = W/‖W‖_F`，`X_{k+1} = 1.5·X_k − 0.5·X_k X_kᵀX_k`，固定 N 步；卷积核按空间位置逐片处理。
- 何时用：语言与视觉中每次新数据到来前（训完当前数据、开始新数据前）；RL 中只在训练中点用一次。

**超参与代码细节**（[isaac7778/FIRE](https://github.com/isaac7778/FIRE) `language/interventions/fire.py` 的 `fire`、`newton_schulz`，[V]）：
- 系数 (1.5, −0.5)，`newton_schulz` 默认 10 步；语言实验 `train.py` 里 `fire_iteration = 5`。
- 结果乘 `sqrt(d_out/d_in)`（卷积再除以核面积）。
- 作用于模型里所有 `nn.Linear` / `nn.Conv2d`；GPT 的合并 `c_attn` 拆成 q、k、v 各自处理。nanoGPT 的 `lm_head` 与 `wte` 权重共享，所以嵌入也被重置（读代码推断）。
- `train.py` 中 `reset_optimizer = True`，优化器在 `fire` 之后重建 [V]。
- 论文附录称 ViT 只正交化 Q、K 投影；GPT 实验 AdamW lr 6e-4→6e-5、wd 0.1、批 480 [V]。

**规模与数字**（[V]）：GPT-0.1B 先在 WikiText-103 预训练，再在 OpenWebText + WikiText-103 上继续，从最佳、30k、60k 检查点出发；FIRE 不调参、固定 5 步，S&P 调过参；完全重置不如基座。正文没有给出具体 loss 数字（只在图里）。5M–314M GPT 都会出现可塑性丧失，见 [continual-sleep.md](../topics/continual-sleep.md)。

**映射到本项目**：在已训模型上一次性操作，放在长训练分阶段之间（例如 `train(..., init=...)` 继续训练前），或作为 `sleep` 的可选步骤。我们参数是堆叠的 [M,a,b]，逐模块做：
```
@torch.no_grad()
def fire(model, iters=5):
    for name, p in model.named_parameters():
        if name.split('.')[-1] in {'B','G','K','V','Qm','W1','P','Wq','Wk','Wv','Wo'}:
            X = p / p.norm(dim=(-2,-1), keepdim=True)     # 3D 时每模块一份
            tall = X.shape[-2] < X.shape[-1]; X = X.mT if tall else X
            for _ in range(iters): X = 1.5*X - 0.5*X @ (X.mT @ X)
            p.copy_((X.mT if tall else X) * (p.shape[-1]/p.shape[-2])**0.5)   # einsum 布局为 (输入,输出)
    # 之后重建优化器
```
**小规模风险**：
- FIRE 会大幅改变函数，原文之后接的是完整的新数据训练；我们的 `sleep` 只训约 200 步并按旧探针 bpc 退步 ≤ old_tol 决定接受，FIRE 后几乎必然被回滚。因此只适合离线分阶段训练，不适合放进睡眠接受门。
- W2 零初值（低秩、范数小）、门投影 w_beta/w_gam 是向量，不应正交化；嵌入与读出是否重置需对照（原代码因权重共享连嵌入一起重置）。
- 0.3–5M 字节级模型是否会出现可塑性丧失，要先用死单元、有效秩、新数据学习速度测出来再用。

**不需要重训**（作用于已训权重，但之后必须继续训练）。
