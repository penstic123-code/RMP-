# 动态稀疏可塑性神经网络 · 设计与实验 v0.11

2026-11-19｜目标：在固定内存预算下持续预测、在线学习，并保留旧规律。

**当前决策。** 底层保留小型稠密循环模块，跨模块采用稀疏连接。读出在线学习，远程连边按少量候选的流式统计调整。隐藏层以固定权重为基线，优先试验每神经元共享学习痕迹。最新对照中，共享痕迹将持久痕迹从 256 个数减至 16 个，平均预测误差接近逐突触痕迹；两者都尚未超过固定核心的平均成绩。完整网络的峰值内存、适应和抗遗忘仍须联合验收。

## 1　系统怎样运行

底层 L1 包含 128 个并列模块，排列为 8×16 网格，其中 16 个模块各接一条外部观测通道。每个模块有 16 维状态 h，内部循环矩阵 W 为 16×16；模块将状态投影为 4 维消息 z。四邻居由坐标推算，边界不环绕。第一版允许每个预测端接收至多两条活动远程入边。读出先导实验中的“三个槽”是独立实验配置。

1. 在时刻 t 观察 x(t)，先用它核对上一时刻对 x(t) 的预测。记录误差，然后用发出预测时保存的特征更新读出。第一步仅初始化。
2. 模块用旧状态、固定邻居上一拍的消息及本通道当前输入更新状态，再生成消息。基础版采用 tanh，固定邻居消息和除以 4；W、消息投影和输入投影随机初始化，保存种子。固定 W 的谱范数初值为 0.8；检查整个耦合系统的饱和率及扰动衰减。
3. 每个预测端用本地状态和已接入的远程来源预测 x(t+1)。基础读出采用 NLMS：以误差乘输入特征，按特征平方范数加常数归一化。远程边只进入读出。这使隐藏层固定时的更新可直接核对。普通 LMS 是同等地位的低内存对照。
4. 先结清旧预测的学习，再更换连边；新边权重和学习状态从零开始。流中单步反馈只保留上一拍待核对信息；如果改用延迟反馈，队列本身也计入内存。

正式接口只有观测流 step(x)。过去的读出先导实验在预测后由生成器给出目标；它们分别检验了改权重与换边的机制，尚未覆盖完整单流系统。

## 2　省内存的三个位置

| 对象 | 当前做法 | 代价或边界 |
|---|---|---|
| 不存在的跨模块连接 | 保存活动来源索引与权重。容量、活动数、预留空槽分别计数；非活动模块对不持有长期权重或优化器状态。 | 局部块内仍支付真实稠密参数的成本。 |
| 寻找新连接 | 每 16 步共享抽取 16 个候选来源。按来源累计平方和，每个预测端单独累计误差平方和、候选特征乘残差。周期结束打分并清空。 | 扫全来源更快找到新规律。追踪速度优先时试 32 个候选。128 模块版另测“邻居的邻居 + 均匀远程探索”，与均匀抽样匹配候选数。 |
| 学习活动痕迹 | 同一神经元的出边复用其近期活动摘要；仅按需为部分连接保存专属痕迹。 | 共享摘要损失“这条边在过去某一刻恰好有用”的细节。用延迟反馈任务测这项损失。 |

模块数 M、模块宽度 d、活动远程入度上限 k 固定口径下：内部权重为 Md²，若所有模块均可收远程边，则远程边至多 Mk。共享来源痕迹约 Md，全量逐突触痕迹约 Md²。FP32 下 M=10,000、d=16 时，单是 W 约 9.77 MiB，共享来源痕迹约 0.61 MiB，全量痕迹另加约 9.77 MiB。若 Y 个预测端同时给 q 个共享候选打分，统计仍需 O(Yq)；隐藏层反馈也须限制扇入，避免形成隐藏单元数乘输出数的反馈矩阵。

**可选压缩。** 大多模块学到相似 W 时，可比较“共享基础 W + 少量独立偏移”与独立 W。将偏移索引、重建计算及效果损失一并统计。仅在相同任务精度下实际更省内存时采用。

## 3　隐藏模块内的突触如何学习

输出误差只能在观测实际到达后使用。第一轮给不同规则相同的固定反馈投影，令每个隐藏单元得到标量学习信号。权重更新为“学习率 × 该单元的学习信号 × 活动痕迹”。反馈告诉单元该往哪个方向调整，痕迹估计此前哪些输入参与了计算。Oja 型关联规则单列为只用两端活动的对照。

| 组别 | 活动痕迹怎样算 | 一个 16 单元模块的持久痕迹数 |
|---|---|---:|
| 固定内部权重 | 只在线训练读出 | 0 |
| 共享 | 每个来源对自身旧活动做指数平均；更新时乘目标单元当前的 tanh 导数 | 16 |
| 混合 | 共享方案加 16 条对角连接的专属痕迹；位置由坐标给出 | 32 |
| 逐突触 | 每条连接对“当时的目标导数 × 当时的来源活动”做指数平均 | 256 |

共享和逐突触的区别在于先求乘积还是先平均；目标单元的敏感度随时间变化时，两者会给出不同更新。以上是局部近似，完整递归系统的时间信用分配仍需以小网络反传参照。此次 NumPy 程序为便于核验构造了临时矩阵；生产实现按行更新后再测峰值。突触可塑性的先例见 [RFLO](https://elifesciences.org/articles/43299v2)、[e-prop](https://www.nature.com/articles/s41467-020-17236-y) 和 [局部 Hebb/STDP 模型](https://neuronaldynamics.epfl.ch/online/Ch19.S2.html)。

## 4　记忆、注意力与层级各负责什么

**工作记忆**保存当前 h 和待核对的预测。**短期找边统计**积累最近输入对误差的解释力。**长期记忆**由跨会话保存的权重和活动路由承担。抗遗忘组另外使用不超过 64 KiB 的原始序列重放池，容量包含元数据。无重放组保留为正式对照。

**保护旧知识。** 旧序列可更新权重。若旧通路被删除，还需在换边前比较原路由与候选路由。两者用同一批近期数据拟合，再用后续近期片段以及抽出的旧记忆片段验收。近期误差改善且旧记忆的 NMSE 恶化不超过预注册容差才接纳，开发起点为绝对增量 0.02。复制受影响目标的临时状态计入峰值。独立旧任务探针不入池。重放从独立状态预热并在结束后恢复现场状态。

**注意力是瞬时选择。** 如果已经接入的几个来源随情境交替有用，才试验在活动入边上按当前状态分配读取权重。与静态增益对比，保持候选数和总预算一致。没有理由在基础版为所有模块对生成注意力分数。[注意力的查询、键和值定义](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)

**上下文广播是可选实验。** 当多个输出需要辨认同一情境时，允许从近期观测形成一个低维共享信号，控制已有读出的选择。比较真实可观测线索、打乱线索和无广播三组；任务编号只给评估器。广播可以帮助选路，能否保留此前学过的权重还要用冻结探针检验。这个设计只借用全局神经工作空间假说中的“少量信息可供多个系统访问”。[GNW 综述](https://pubmed.ncbi.nlm.nih.gov/32135090/)

**模块层级最多两级。** L1 可直接预测。可选 L2 只在多个输出重复使用同一对 L1 来源时试建一个共享模块 H，初版计算 s=a·zA+b·zB，下游各学一个系数。H 只读 L1，多个 H 并列。每次试用一个候选，纳入预算，比较直连和共享时的未来误差及旧任务保留。上下文广播可复用合适的 H；两种机制分别消融，避免收益归因混淆。最终读出不另算一级。

## 5　实验记录与设计依据

以下各行属于不同任务或种子集，误差只在同行指定的对照间比较。

| 实验 | 已运行结果 | 对设计的影响 |
|---|---|---|
| 8 输入、8 输出，映射整体切换，10 种子 | 切换末段 MSE：冻结 1.400，只改原权重 0.774，允许换边 0.125；噪声底 0.1225 | 保留换边能力。结果限定于单一独立来源映射。 |
| 96 来源、8 输出，联合适应与遗忘 | 输入分离时稠密、稀疏都零遗忘；完全冲突且无情境提示时稀疏组旧任务 NMSE 增加 1.723，冻结组无法学好新任务 | 重点测试“有重叠、有可推断上下文”的任务；区分保留与重新学习。 |
| 在线寻找新来源，10 个独立确认种子 | 完整 16 步窗口 6,880 B，切换初期/末期 MSE 0.2883/0.1315；共享 16 候选 896 B，0.4766/0.1309；共享 32 候选 1,536 B，0.3675/0.1309 | 16 候选作为默认。896 B 相对稠密读出 3,072 B 减少 70.8%，只含被核算的持久数值数组。 |
| 隔离读出学习规则，给定正确来源 | LMS、NLMS、RLS 末段 MSE 0.1311、0.1313、0.1267；分别 224、224、800 B | 保留简单读出。RLS 精度收益需对照额外协方差存储。 |
| 16 单元循环块，10 种子，固定同一读出更新 | 冻结内部、Oja 型、简化反馈痕迹的冻结测试 MSE 分别为 0.2040、0.2984、0.2043 | 该配置尚无内部可塑性优势。 |
| **本版新增：四种痕迹配置**，独立种子 200–209 | 冻结、共享、混合、逐突触的平均冻结测试 MSE 分别为 **0.19322、0.19457、0.19460、0.19449**。三个可塑组各在 7/10 个种子优于冻结，跨种子平均均略差。 | 共享用 16 个数接近 256 个数的逐突触方案。平均误差最小的是固定核心；当前没有升级混合或逐突触规则的实证理由。 |

新增实验使用相同初始化、输入、输出误差投影、读出规则和可塑组学习率。任务融合三步前的输入与当前输入，另叠加两步时延项；从第 4,000 步起冻结，测最后 512 步。共享、混合、逐突触组都采用 0.5 的指数衰减。该实验的非线性任务有短时记忆要求，反馈在输出后立即揭晓，尚未测长期延迟奖励。临时矩阵、峰值字节和遗忘并未在这项实验中评估。附录给出复现代码。

## 6　下一轮的验收和自动科研顺序

1. **先确认局部学习是否值得付费。** 在相同核心、同一反馈及读出下，比较固定、共享、混合、逐突触。分别改变任务所需的历史长度与反馈抵达的延迟，单列其代价。先做同结构比较，再做同总字节预算比较：把节省的痕迹字节用于更多单元、可塑连接或远程边。正式局部层实现必须核算反馈投影和临时更新区。
2. **跑完整单层系统。** 16 通道流中切换 A→B→C，再回 A。任务包括来源分离、可由历史线索识别的重叠，以及无法凭观测历史区分的冲突负对照。训练、开发、测试图独立。各方法同一时刻只见相同信息。顺序对照为持续值预测、稠密在线读出、固定稀疏、随机换边、流式评分换边和同学习峰值预算的小型 GRU。给 GRU 同样的在线更新机会，将训练缓冲算入预算。文本阶段才加入同分词、同文本量和同更新机会的小 Transformer。
3. **三项指标同时验收。** 扫 32/64/128/256 模块，在预先指定的部署规模比较模型数组、检查点、CPU 峰值 RSS 或 GPU 分配/保留峰值以及步延迟。初始工程门槛：末段 NMSE 相对稠密组至多高 5%，完整在线峰值至少下降 20%；切换后前 1,024 步误差相对优先适应的在线对照至多高 5%，且优于冻结；对可识别情境，独立冻结探针的最差旧任务 NMSE 增量 ≤0.05。旧任务遗忘量定义为最终误差减刚学完该任务时的误差。独立报告回到 A 后的重学速度。
4. **最后试附加机制。** 基础系统在可识别任务中遗忘时，比较无保护、仅重放、重放加换边保护；若存在反复选错旧规律，再试上下文广播。若多个输出确实共享同一信息组合，再比较直连、随机 H 与按数据形成 H。仅当同一组输入需依情境切换重要性时试活动边门控。
5. **运行器自动保存证据。** 每轮只改一个因素，保存配置、代码版本、图与轨迹种子、原始分段误差、各数组的 dtype/容量、运行峰值、耗时与退出状态。开发阶段锁定门槛，独立评估器执行至少 10 张测试图；报告逐图配对差与 95% 区间，跨门槛记为未定。重点测试未来观测不改变既有预测、冻结探针不写入、换槽不沿用旧痕迹、序列化后预测一致。自动报告须追溯到原始日志。

## 7　研究依据与实验索引

固定递归核心可参照[储备池计算综述](https://www.sciencedirect.com/science/article/pii/S1574013709000173)。动态稀疏训练中的新生参数稳定性见 [Xiao 等，ICML 2026](https://proceedings.mlr.press/v306/xiao26t.html)；跨步信用分配参照 [SnAp](https://openreview.net/pdf?id=q3KSThy2GwB)。重放参照[小型情景记忆研究](https://arxiv.org/abs/1902.10486)。这些工作各自支持一种候选机制，三项联合能力以本项目实验为准。

[三项能力联合先导报告及代码](https://chatgpt.com/library?file_id=libfile_dc4204d43a308191a650f4637c18c973&account_id=personal) · [第一轮隐藏层局部学习代码](https://chatgpt.com/library?file_id=libfile_57f0ab1d05248191a42593e53f9c8e75&account_id=personal)

## 附录　可复现代码

三段程序各自保存为对应文件，安装 NumPy 后用 Python 单独运行。第一段是本版新增，后两段保留读出和搜索实验的实现。结果是 CPU 机制实验。

### 新增：共享与逐突触痕迹（trace-study.py）

```python
import numpy as np

def trial(seed):
    rng = np.random.default_rng(seed)
    d = 16
    initial = rng.normal(0, .75 / np.sqrt(d), (d, d))
    inputs = rng.normal(0, .35, (d, 2))
    feedback = rng.choice([-1., 1.], d)
    x = rng.normal(size=(4512, 2))
    y = np.tanh(np.roll(x[:, 0], 3)) * np.tanh(x[:, 1]) + .4 * np.roll(x[:, 0], 2)
    # A fixed diagonal pattern avoids allocating an index array.
    results = []
    for mode in ('fixed', 'shared', 'hybrid', 'full'):
        w = initial.copy()
        readout = np.zeros(d)
        state = np.zeros(d)
        if mode in ('shared', 'hybrid'):
            pre = np.zeros(d)
        if mode == 'hybrid':
            diag = np.zeros(d)
        if mode == 'full':
            trace = np.zeros((d,d))
        errors = []
        for t in range(3, len(x)):
            old = state
            state = np.tanh(w @ old + inputs @ x[t] + .15)
            error = y[t] - readout @ state
            if t >= 4000:
                errors.append(error**2)
                continue
            readout += .04 * error * state / (1 + state @ state)
            slope = 1 - state**2
            if mode == 'fixed':
                continue
            if mode == 'full':
                trace = .5*trace + .5*slope[:,None]*old
                update = trace
            else:
                pre = .5*pre + .5*old
                update = slope[:,None]*pre
                if mode == 'hybrid':
                    diag = .5*diag + .5*slope*old
                    np.fill_diagonal(update, diag)
            w += .0003 * error * feedback[:,None] * update
        results.append(np.mean(errors))
    return results

if __name__ == '__main__':
    scores = np.array([trial(s) for s in range(200,210)])
    print(scores.mean(axis=0), (scores < scores[:,:1]).sum(axis=0))
```

### 最小换边实验（pilot.py）

```python
import numpy as np

def run(seed, mode, width=8, period=128):
    rng = np.random.default_rng(seed)
    search = np.random.default_rng(seed + 1000)
    graphs = rng.permutation(width)
    parents = np.stack([graphs, np.roll(graphs, 1)])
    x = rng.normal(size=(8192, width)).astype('f')
    noise = rng.normal(0, .35, x.shape).astype('f')
    dst = np.arange(width, dtype=np.int32)
    src = search.permutation(width).astype(np.int32)
    w = np.zeros(width, dtype='f')
    buf = np.empty((2, period, width), dtype='f')
    errors = []
    for t, row in enumerate(x):
        pred = row[src] * w
        target = .8 * row[parents[t // 4096]] + noise[t]
        err = target - pred
        errors.append(float(np.mean(err**2)))
        if t < 4096 or mode != 'freeze':
            w += .025 * err * row[src]
        buf[:, t % period] = row, target
        if (t+1) % period or mode == 'static' or (mode in ('weight','freeze') and t >= 4096):
            continue
        for k in dst:
            candidates = np.unique([src[k], *search.choice(width, 2, replace=False)])
            scores = [abs(buf[0, :, j] @ buf[1, :, k]) /
                      max(np.linalg.norm(buf[0, :, j]), 1e-8) for j in candidates]
            best = candidates[np.argmax(scores)]
            if best != src[k]:
                src[k], w[k] = best, 0.
    return [np.mean(errors[a:b]) for a,b in [(3072,4096),(4096,4224),(7168,8192)]], np.mean(src==parents[1])

if __name__ == '__main__':
    for mode in ['static','freeze','weight','online']:
        res = [run(s,mode) for s in range(10)]
        print(mode, np.mean([v[0] for v in res],0), np.mean([v[1] for v in res]))
```

### 读出学习与流式搜索（mechanisms.py）

```python
import numpy as np
D,Y,T,L,Q=96,8,1024,16,4
F=np.float32
def run(seed, rule, search,Q=4):
    rg=np.random.default_rng(seed)
    pools=np.stack([rg.choice(D,3,False) for _ in range(Y)])
    dense=search=='dense'
    idx=pools.astype('i') if search=='known' else np.zeros((Y,3),'i')
    w=np.zeros((Y,3),F)
    used=np.full(Y,3 if search=='known' else 0,'i')
    st=[w,idx,used]
    if dense:
        w=np.zeros((Y,D),F);st=[w];del idx,used
    if rule=='rls':
        cov=np.tile(4*np.eye(3,dtype=np.float64),(Y,1,1));st.append(cov)
    rng=np.random.default_rng(seed+1337)
    shared=search=='shared'
    if search=='window':
        hist=np.zeros((L,D),F); residual=np.zeros((L,Y),F);st.extend([hist,residual])
    elif search not in ('known','dense'):
        cand=np.zeros((1 if shared else Y,Q),'i')
        xx=np.zeros_like(cand,dtype=F)
        xy=np.zeros((Y,Q),F)
        ee=np.zeros(Y,F)
        st.extend([cand,xx,xy,ee])
    byphase=[]
    for phase in range(3):
        losses=[]
        for t in range(T):
            x=rg.standard_normal(D).astype(F)
            y=.8*x[pools[:,phase]]+rg.normal(0,.35,Y)
            v=x[None,:] if dense else x[idx]*(np.arange(3)<used[:,None])
            e=(y-(w*v).sum(1)).astype(F)
            losses.append(np.mean(e*e))
            if rule=='lms': w+=.04*e[:,None]*v
            if rule=='nlms': w+=.15*e[:,None]*v/(1+(v*v).sum(1,keepdims=True))
            if rule=='rls':
                pv=(cov@v[:,:,None])[:,:,0]
                gain=pv/(.98+(v*pv).sum(1,keepdims=True))
                w+=gain*e[:,None]
                cov[:]=(cov-gain[:,:,None]*pv[:,None,:])/.98
                cov[:]=(cov+cov.transpose(0,2,1))*.5
            if search in ('known','dense'): continue
            if search=='window':
                hist[t%L]=x;residual[t%L]=e
                if (t+1)%L: continue
                cand=np.arange(D)[None,:]
                score=np.abs(residual.T@hist)/np.sqrt(.001+(residual**2).sum(0)[:,None]*(hist**2).sum(0))
            else:
                if t%L==0:
                    for row in cand: row[:]=rng.choice(D,Q,False)
                    xx.fill(0);xy.fill(0);ee.fill(0)
                vals=x[cand];xx+=vals*vals;xy+=vals*e[:,None];ee+=e*e
                if (t+1)%L: continue
                score=np.abs(xy)/np.sqrt(.001+xx*ee[:,None])
            for k in range(Y):
                j=np.argmax(score[k]);src=cand[0 if shared or search=="window" else k,j]
                if score[k,j]<.78 or src in idx[k,:used[k]]: continue
                slot=used[k] if used[k]<3 else np.argmin(abs(w[k]))
                idx[k,slot]=src;w[k,slot]=0;used[k]=min(3,used[k]+1)
        byphase.append([np.mean(losses[:256]),np.mean(losses[-256:])])
    return np.mean(byphase[1:],axis=0),sum(a.nbytes for a in st)
if __name__=='__main__':
    for split,seeds in [('dev',range(3)),('explore',range(30,40)),('confirm',range(100,110))]:
        for rule,search,q in [('lms','known',4),('nlms','known',4),('rls','known',4),('nlms','dense',4),('nlms','window',4),('nlms','independent',4),('nlms','shared',4),('nlms','shared',16),('nlms','shared',32)]:
            vals=[run(s,rule,search,q) for s in seeds]
            print(split,rule,search,q,np.round(np.mean([v[0] for v in vals],axis=0),4),vals[0][1],flush=True)
```

