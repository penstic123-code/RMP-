# 实现卡：睡眠、回放与接受检验（第二批）

> 与主线 RMP 相关：第 10 张卡（标 【主线 RMP】），见 [23.10](../../design/candidates/23-routed-pool.md)。

“[V]”= 打开原文或代码确认；“[S]”= 搜索片段。目录见 [README](../README.md)。

检索日期 2026-11-19。代码一律读 GitHub 原文件；论文页没给仓库的写“未确认”。编号接 [cards-sleep.md](sleep.md)。

## 9. 自生成回放（Forgetting in LMs，arXiv 2605.26097）

**规则**（[V]）：回放序列的第 0 个 token 是数据集标识符（作用相当于 BOS，每个数据集一个），其后从冻结参考模型自回归采样到固定长度；指令模型只给一个 BOS。损失 `L = L_down + λ·L_replay`，回放项优先用前向 KL：`KL(π_ref ‖ π_θ)`，逐 token 计算，最后一位置零后取平均（`loss.py: loss_fn_kl`，`optax.kl_divergence_with_log_targets(logp_θ, logp_π)`）；NTP 回放略差。
**饱和 / 余量**（[V]）：论文没有单一指标，只用代理量：每参数 token 数（6M 模型训 100B ≈ 17,000 tok/param，算力最优约 7–20），等预训练损失下比较，引用容量 2–4 bit/参数；遗忘用旧数据损失或“算力倍数”衡量。作者自己承认这些只是代理指标。
**超参**（[V]）：λ=10，lr 1e-5（205M 实验）；回放批为下游批的 1/4；批 256、上下文 256、AdamW (0.9,0.999)、wd 0.02。采样温度在 HTML 里显示为乱码“11”，代码 `sampling.sample(..., temperature=1)` 默认为 1；实验是否改过未确认。序列长度、样本总数未报告。
**规模与数字**（[V]）：2M 玩具 transformer 四个算术任务上遗忘被“完全消除”；多数实验 ≤46M；205M 和 Llama-3.2-1B 只给图，没有百分比。
**代码**（[V]）：[martin-marek/forgetting](https://github.com/martin-marek/forgetting) `train_single.py: sample_synthetic_batch`（第 0 位填 `data_token_id`）、`loss.py: loss_fn_reg`（`kl_pre` / `ntp_pre` / `kl_dwn`）、`sampling.py: sample`。JAX 实现。
**映射**：`online.py` 的 `_dream` + `sleep(dream=…)` 已是同一方法：老师是 `self.morning`，从换行符（10）起头，KL 方向与原文一致。缺的是余量测量：
```
headroom = 训练字节数 / count(model)           # 记进 runs/，与 2–4 bit/参数对照
for lr in [1e-4, 3e-4, 1e-3]: 只训新数据 N 步，记 Δold_bpc / Δnew_bpc   # 同样新知识增益下旧损失涨得越快，余量越小
sleep(dream=10.0, batch_dream=batch//4)        # 按原文 λ 与 1:4 配比
```
**小规模风险**：S 档模型训数亿字节，相当于每参数几百字节，可能已经饱和；原文结论是饱和时回放也救不了。换行起头的梦只覆盖“段首”分布。
**不需要重训**。

## 10. Language Models Need Sleep（Behrouz 等，arXiv 2606.03979）
> 【主线 RMP】睡眠中扩展专家 ≈ RMP 新奇分配 / 分裂（E69、E71）；做梦随机挑专家 ≈ RMP 睡眠做梦（E70）。

**Knowledge Seeding**（[V]，HTML 公式部分乱码，下式按文意转写）：老师 = 带已累积低秩专家的当前模型，学生 = 发送块专家清零、接收块新增低秩专家 {A∈ℝ^{d×d_low}, B} 的模型。GKD 式：`L = (1−λ)E_{(x,y)~D}[F(θ‖θ_exp)] + λE_{x~D}E_{y~学生}[F(θ‖θ_exp)]`，采样不回传梯度。模仿 RL：从老师的梦里取前缀让学生续写，奖励 `r = γ r_sem + (1−γ) r_abs`，其中 `r_abs = 1 − Lev(ŷ,y)/max(|ŷ|,|y|)`（距离 ≤ z₀ 时，否则为 0）；`L_KS = (1−α)E[r] − α E[D(θ‖θ_exp)]`。旧专家随后清零（“突触修剪”）；第 ℓ 级在 C^(ℓ)·b 步触发。
**Dreaming**（[V]）：从上下文 C 采 m 个梦（路由随机挑一个专家，混入无关知识）；按 `g = ∇L_SFT(梦)` 选 top-k，再加 b 个随机梦；每个梦在隔离副本上做 LoRA-SFT，任务 τ 变好则 r=1，否则 r=0；外层用 ReST-EM（同 SEAL）。
**超参**（[V]）：表 5 中 LoRA r=64、α=128，lr 5e-6，有效批 32，Sleep 100 步（GRPO 500 步）；附加参数为“5 个维度 64 的 MLP 块”。λ、α、γ、z₀、k、b、m、d_low 的取值未确认。
**数字**（[V]）：Qwen3-8B AIME-24 79.2（基座 73.8，GRPO 76.4）；SQuAD 单段落 48.9（SEAL 46.7；去掉 Dreaming 35.7）；ARC 80%（SEAL 72.5%）。
**代码**：页面无仓库链接（未确认）。
**映射**：我们的 `_merge_lora` 是精确并入（W += a@b），不需要靠蒸馏搬运知识；可借的是 on-policy 那一项和梦的筛选：
```
y = _dream(m, pool_prefix, chunk)                  # 学生自己续写池中段首（on-policy）
loss += λ * KL(teacher(y) ‖ m(y))                  # teacher = 并入前的模型
g_i = ‖∇CE(dream_i)‖; keep = topk(g, k) ∪ random(b)   # 替换现在“全部梦等权”的做法
```
**小规模风险**：r_sem 需要外部奖励模型，我们没有；E36 发现从池中片段续写会让模型忘掉新词，on-policy 项可能重演这个问题；逐梦建副本训 LoRA 的成本是 m 倍。
**不需要重训**（梦筛选和 on-policy KL）；照搬专家扩展需要改架构。

## 11. Do LMs Need Sleep? 离线循环（arXiv 2605.26099）

**规则**（[V]）：SSM 快权重 `S_t = α_t S_{t−1} + β_t v_t k_tᵀ`，`o_t = S_t q_t`（实验用 GDN）。上下文窗 L 填满时，不输入新 token，把整个块堆叠循环 N 次 `[B_0…B_{D−1}]^{×N}` 以更新 S，再清空 KV cache。训练时（算法 1）loss mask 全零的块是巩固期，跑 N 次；其余块跑一次并算 CE；梯度穿过 S 回传，不穿过被丢弃的循环特征。
**超参**（[V]）：Muon lr 2e-3 / 1e-3，AdamW 5e-5；批 512/128/256。
**数字**（[V]）：GSM-Infinite 6 步运算，Jet 0.742→0.812，Ouro 0.419→0.615；滑窗 L=512 时 Ouro 0.596→0.905。训练成本约随 N 线性增长。
**代码**：页面无仓库链接（未确认）。
**映射**：`Chat.turn('/new')` 清短时状态前，用 `dspn_text.delta_chunk` 的快记忆状态 S 对本会话做 N 遍无监督重读：
```
for _ in range(N): _, st = model(session_ids[None], st)   # 只留快记忆 S，丢掉其余状态
st = keep_fast_only(st); 再读入 SEP
```
**小规模风险**：不重训时这只是“多读几遍”，S 会被同一内容重复写满并衰减，效果可能为负。原文增益来自端到端训练，并且在推理题上，不在语言建模上。
**需要重训**（训练目标里要加 mask=0 的巩固块）。

## 12. SEAL：自我编辑（arXiv 2506.10943）

**格式**（[V]）：知识注入时提示模型“produce a list of implications derived directly or indirectly from the content”，输出按换行切成训练文档；另试过 long / very-long / 改写 / 自问自答 / CoT 等提示（`make_squad_data.py` 中的模板）。少样本 ARC 场景则输出 JSON（数据增强选项 + lr / epoch / 损失范围）。
**奖励环**（[V]）：ReST-EM。每个上下文采 5 个编辑（温度 1），每个编辑用 LoRA 训一个副本，在 3 个种子上评测；编辑带来提升 r=1；保留最好的编辑，对 50 个 (上下文, 编辑) 对做 SFT，共 2 轮。内层 LoRA r=64、α=128、lr 3e-4、2 epoch、批 10。ARC：每任务 15 个编辑，超过 375 步的丢弃，r=16、lr 5e-5、8 epoch。
**数字**（[V]）：Qwen2.5-7B 无上下文 SQuAD：32.7→47.0（GPT-4.1 合成数据 46.3）；Llama-3.2-1B ARC 72.5%。连续编辑会逐步遗忘旧段落，论文承认没有解决。
**代码**（[V]）：[Continual-Intelligence/SEAL](https://github.com/Continual-Intelligence/SEAL) `general-knowledge/src/data_generation/make_squad_data.py: make_prompt / generate_bulk`，`src/EM/build_SFT_dataset.py: _top_k`（best-k-of-n），`src/continual/continual_self_edits.py`（只确认了文件存在）。
**睡眠用的廉价版**：不跑外层 RL，只做“生成—试训—择优”：
```
for seq in top_pool:                         # OnlineLearner.pool 中惊奇最高的若干段
    cands = [_dream(m, seq[:16], 48) for _ in range(K)]      # K=3–5，温度 1
    for c in cands: m' = 拷贝 LoRA，训 4 步；r = bpc(m', new_hold) − bpc(m, new_hold)
    pool.add(argmin r)  if 最优 r < 0
```
`new_hold` 必须与 `sleep` 的 `probe_new` 不相交，否则等于在接受门上择优（过拟合门）。
**小规模风险**：0.3–5M 字节模型不会听指令，“implications”只能退化成续写或复述，可能只是噪声；K×拷贝×步数的开销落在睡眠里。
**不需要重训**（廉价版）；完整 RL 版要训练编辑策略，不适合我们这个规模。

## 13. Darwin / Huxley-Gödel Machine：检查点档案

**DGM**（[V]，[jennyzzt/dgm](https://github.com/jennyzzt/dgm) `DGM_outer.py`）：`choose_selfimproves(method='score_child_prop')` 中 `s=σ(10(acc−0.5))`、`h=1/(1+子代数)`，`p ∝ s·h`；`update_archive` 默认 `keep_all`（只收能编译、还能改代码的代理），`keep_better` 允许 0.1 的噪声余量。80 代，每代 2 个（Polyglot 4 个）。SWE-bench 20.0→50.0%。
**HGM**（[V]，[metauto-ai/HGM](https://github.com/metauto-ai/HGM)）：CMP(a) = 子树内最好后代效用的期望；估计 `n_succ^C/(n_succ^C+n_fail^C)`，计数在整个子树上合计。`hgm.py: main` 里 `TS_sample` 抽 `Beta(1+Σ, 1+n−Σ)` 取 argmax；`expand` 用 `tree.py: Node.get_decendant_evals`（自身 + 后代的逐任务 0/1）选父代；`N_evals^α ≥ |树|` 时扩展，否则对单个节点再评一个任务（`sample`）。最终选 ε 分位后验最大者（论文 [V]；对应代码未找到）。超参：α=0.6，`n_pseudo_descendant_evals`=10000，预算 800 次评测。SWE-Verified-60：HGM 56.7%（517 小时），DGM 53.3%（1231 小时）。
**代码疑点**（读代码推断，未运行）：`get_pseudo_decendant_evals` 在评测数小于 10000 时返回 `self.utility_measures` 本身，`get_decendant_evals` 随后对它 `+=`，会把后代结果原地追加进父节点的列表。照抄时要先复制列表。
**映射**：把 `sleep` 的单步回滚改成树。节点 = 被接受的 `state_dict`；每次睡眠的 `committed` 及逐探针项的过关与否，记作该节点的 0/1 效用：
```
archive = [Node(ckpt, evals=[])]
parent = archive[argmax(beta(1+Σclade, 1+n−Σclade))]    # clade = 自身+后代 evals
m.load_state_dict(parent.ckpt); r = learner.sleep(...)
parent.evals += r 的逐项过关;  if r['committed']: archive.append(Node(m.state_dict(), parent))
```
**小规模风险**：S 档一个检查点只有几 MB，存档便宜；但父代换了以后，当天的 LoRA 和池是在另一个模型上积累的，回到祖先等于丢掉这条分支上的记忆，聊天 demo 会出现“失忆”。探针只有两条，0/1 统计的方差很大。
**不需要重训**。

## 14. 反事实类比探针（Lewis & Mitchell，arXiv 2411.14215）

**构造**（[V]，[robust-analogy](https://github.com/marthaflinderslewis/robust-analogy) `letterstring/gen_problems_by_alph.py: k_derange(k, letters)` 随机挑 k 个位置做全错排，`gen_prob_subset` 生成题目）：k∈{2,5,10,20}，每个 k 取 7 个字母表；另有 10 / 15 个符号的字母表。六种变换：延长序列、后继、前驱、删重复字母、修正字母序、排序（`apply_succ` / `apply_pred` / `apply_add_letter` / `apply_remove_redundant` / `apply_fix_alphabet` / `apply_sort`），附带后继 / 前驱理解检查。故事类比交换答案顺序：GPT-4 正确答案在前 18/18，在后 13/18。
**数字**（[V]）：字母串零泛化，人类 0.754，GPT-4 0.452。
**字节级接受门探针**（我们的改编）：0.3–5M 模型做不了字母串类比，原题会全部落在地板上。可取的是“同构造的对照项”：
```
fact  = b"User: my cat is named Zorp.\n"           # 当天学到的
pos   = logp(b"Zorp" | b"User: what is my cat's name?\nBot: ")
neg   = logp(b"Blik" | 同上)                        # 换名反事实，名字长度和字频相同
perm  = 把 fact 中名字做 k_derange 字节错排后的 logp   # 背字符串 vs 记住实体
ok_new = (pos − neg) 睡后增大  and  perm 不同步增大
```
选项顺序：同一题把正确答案放前和放后各测一次，两次都过才算。
**小规模风险**：边际很小，需要几十条事实取平均才能压住噪声；探针要固定随机种子并存进 runs/。
**不需要重训**。

## 15. 可塑性诊断（Can Scale Save Us，arXiv 2606.24752）

**指标**（[V]）：
- 探针 AUC：在检查点副本上训新语言（越南语 5B token），每 95 步在 1,280×2,048 的验证集上测 loss，取验证曲线下面积；`Δ = 100·(AUC_k − AUC_1)/AUC_1`，为正表示变差。发生点 = 移动平均（窗口 3）后的最低点。
- 休眠单元：`h̄_i = mean_{x,j}|h_i(x,j)|`，`s_i = h̄_i / mean_k h̄_k`，`s_i ≤ 0.01` 即休眠；用 256×2,048 验证序列，在每个周期末测。
- 注意力熵：取查询位置 256–2048 平均，H_max≈6.9215；熵 > 0.9·H_max 为懒惰头，< 0.1·H_max 为坍缩头。
- 平均参数绝对值：所有非嵌入权重和偏置合在一起取平均。
- 有效秩、梯度范数、Hessian：文中没有。探针的 lr 和步数未确认。
**规模**（[V]）：非嵌入参数 5M–314M；每任务 5B token、9,537 步；拟合 `T_onset = 1.3e-5·P^0.8269`（R²=0.897）。外推（我们算的）：P=5M 时 T≈4.5 个任务实例，约 22B token。
**代码**：无链接（未确认）。
**映射**：加一个只读函数，放进 `evals.py`，挂在 `OnlineLearner.sleep` 前后：
```
hook Layer 的 GELU(W1 输出) → s_i，统计 s_i≤0.01 的比例
hub / 窗口注意力的权重 → 熵，算懒惰 / 坍缩比例
mean|p|（排除 emb）；probe: deepcopy 后在 input.txt 上训 200 步，取 bpc 曲线下面积
```
**小规模风险**：按拟合外推，我们睡眠的数据量（KB 级）远远到不了发生点，测到的变化更可能是噪声；字节级、非 transformer 结构不在原文适用范围内。
**不需要重训**。

## 16. RL's Razor：对基座的 KL 预测遗忘（arXiv 2509.04259）

**规则**（[V]）：遗忘 ≈ f(`E_{x~τ}[KL(π₀‖π)]`)，即在新任务输入上测“前向”KL，不需要旧数据；论文没说是 token 级还是序列级，估计方法也未说明。表 1（ParityMNIST）中二次拟合 R²：前向 KL 0.96，反向 KL 0.93，TV 0.80，L2 0.56；LLM 上 R²=0.71。on-policy 数据是关键：1-0 REINFORCE 表现像 GRPO，SimPO 表现像 SFT。
**规模**（[V]）：Qwen2.5-3B-Instruct（数学、科学问答、工具使用），OpenVLA 7B；GRPO 不加 KL，组大小 64；MNIST 用 3 层 MLP，约 500 次运行/方法。
**代码**：只有项目页 jyopari.github.io/posts/rl_razor，仓库未确认。
**映射**：`sleep` 结束时多返回一项，用 `self.morning`（白天学习前）当 π₀，在 `probe_new` 上逐字节平均：
```
kl_new = mean_t KL(morning(x)_t ‖ m(x)_t)  for x in probe_new
r['kl_new'] = kl_new     # 登记；积累几次后拟合 Δold_bpc ~ a·kl_new²
可选门：ok &= kl_new ≤ k · median(历次已接受睡眠的 kl_new)
```
**小规模风险**：学新词本身就会在新文本上推高 KL，所以只能用相对阈值；锚定睡眠（anchor）已经在池上压 KL，两者相关、并非独立证据；R² 来自大模型和分类任务，到字节级 LM 上能否成立要自己拟合。
**不需要重训**。
