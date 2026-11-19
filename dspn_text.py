"""DSPN 字节级语言模型：分层模块网格（门控对角线性循环 c + 快记忆 S + 局部 MLP），离线训练入口。

设计见 docs/design/README.md；排期见 docs/plan/README.md。v0.14 原型原样保存在 legacy/dspn_text_v014.py。
与设计文档的已知差异：读出接所有层的 h（设计为最上层），经瓶颈与嵌入 E 共享。

用法：
  python dspn_text.py train dspn --size S --data tiny --iters 3000
  python dspn_text.py train gru  --size S --data tiny --iters 3000     # 参数量自动对齐 dspn 同档
  python dspn_text.py train tf   --size S --data tiny --iters 3000
  python dspn_text.py sample dspn --size S --prompt "Once upon a time"
"""
import argparse, hashlib, json, math, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
ROOT = Path(__file__).resolve().parent
DATASETS = {
    'shakespeare': ('data/input.txt', None),
    'tiny': ('data/TinyStoriesV2-GPT4-train.txt', 'data/TinyStoriesV2-GPT4-valid.txt'),
    'chat': ('data/chat-train.txt', 'data/chat-valid.txt'),             # 旧版：换行是 \r\n（Windows 写文本时转换），跨会话间隔短
    'chat2': ('data/chat2-train.txt', 'data/chat2-valid.txt'),          # tools/make_chat_data.py 现版生成
}
SEP = 30   # 会话边界字节（ASCII 记录分隔符）：清空 c、快记忆与中枢的短时头，保留中枢持久头（τ ≥ 1e5）

SIZES = {   # 约 0.3M / 0.9M / 3.4M 参数
    'S': dict(layers=4, rows=2, cols=4, d=32, m=16, hidden=64, q=8, emb=32),
    'M': dict(layers=6, rows=4, cols=4, d=32, m=16, hidden=64, q=8, emb=32),
    'L': dict(layers=6, rows=4, cols=8, d=48, m=16, hidden=96, q=8, emb=48),
}


def load_bytes(name):
    train, val = DATASETS[name]
    tr = np.memmap(ROOT / train, dtype=np.uint8, mode='r')
    if val is None:                                        # 单文件：后 10% 作验证
        n = int(len(tr) * 0.9)
        return tr[:n], tr[n:]
    return tr, np.memmap(ROOT / val, dtype=np.uint8, mode='r')


def batch(data, b, T, gen=None):
    ix = torch.randint(len(data) - T - 1, (b,), generator=gen).numpy()
    xy = torch.from_numpy(np.stack([data[i:i + T + 1] for i in ix]).astype(np.int64))
    return xy[:, :-1].to(DEV), xy[:, 1:].to(DEV)


def grid_adjacency(rows, cols):
    M = rows * cols
    A = torch.zeros(M, M)
    for r in range(rows):
        for c in range(cols):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols:      # 边界不环绕
                    A[r * cols + c, rr * cols + cc] = 1.0
    return A / 4.0                                         # 邻居消息之和除以 4


def scan(a, b, c0, chunk=8):
    """并行扫描 c_t = a_t*c_{t-1} + b_t（沿时间维 1）。块内用对数累乘一次算完，块间顺序衔接。
    shortcut: a 下限钳在 0.01，块长 8 时 exp(-la) ≤ e^36.8≈1e16。块长 16 时可到 1e32，乘上输入后反传溢出
    （通宵 L 档第 26000 步 NaN，实测 b·exp(-la) 达 9e31）；改 bf16 或加大块长时换成差分形式。"""
    T = a.shape[1]
    pad = -T % chunk                                       # 补齐位 a=1、b=0：状态不变
    if pad:
        a = torch.cat([a, a.new_ones(a.shape[0], pad, *a.shape[2:])], 1)
        b = torch.cat([b, b.new_zeros(b.shape[0], pad, *b.shape[2:])], 1)
    n = a.shape[1] // chunk
    A = a.reshape(a.shape[0], n, chunk, *a.shape[2:]); Bv = b.reshape(A.shape)
    la = torch.log(A.clamp(min=1e-2)).cumsum(2)            # 所有块同时算块内部分（块内从零起步）
    local = torch.exp(la) * torch.cumsum(Bv * torch.exp(-la), 2)
    P, Lc = torch.exp(la[:, :, -1]), local[:, :, -1]       # 块尾：c_k = P_k·c_{k−1} + Lc_k
    s = 1
    while s < n:                                           # 块间 Hillis–Steele 并行前缀：log2(n) 步，只乘不除，无溢出
        Lc = torch.cat([Lc[:, :s], P[:, s:] * Lc[:, :-s] + Lc[:, s:]], 1)
        P = torch.cat([P[:, :s], P[:, s:] * P[:, :-s]], 1)
        s *= 2
    ends = Lc + P * c0[:, None]                            # 各块末状态
    start = torch.cat([c0[:, None], ends[:, :-1]], 1)      # 各块初状态
    c = (local + torch.exp(la) * start[:, :, None]).reshape(a.shape)[:, :T]
    return c, ends[:, -1]


def delta_chunk(q, k, v, beta, g, S, C=64):
    """分块并行的门控 delta 规则，逐拍等价于 S ← αS；S ← S + β(v − Sk)kᵀ；r = Sq，log α = g。
    q, k, v: (b, T, M, m)；beta, g: (b, T, M)；S: (b, M, m, m)。
    块内用三角求解一次算完 C 拍的写入，块间顺序传状态（Gated DeltaNet 分块算法，Yang 等 2024；
    结构参照 flash-linear-attention 的朴素参考实现）。内部按行向量约定 S' = Sᵀ 计算。"""
    b, T, M, m = k.shape
    C = min(C, T)                                   # 逐拍（T=1）时不补到 64 拍
    pad = -T % C
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))                       # (b, M, T, m)
    beta, g = beta.transpose(1, 2), g.transpose(1, 2)                      # (b, M, T)
    if pad:                                                                # β=0 不写入，g=0 不衰减
        q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
        beta, g = F.pad(beta, (0, pad)), F.pad(g, (0, pad))
    n = (T + pad) // C
    q, k, v = (x.reshape(b, M, n, C, m) for x in (q, k, v))
    beta, G = beta.reshape(b, M, n, C), g.reshape(b, M, n, C).cumsum(-1)   # G: 块内累计 log α
    eye = torch.eye(C, device=k.device)
    lower = eye.new_ones(C, C).tril().bool()
    decay = torch.where(lower, G[..., :, None] - G[..., None, :], float('-inf')).exp()   # Γ_t/Γ_s，s ≤ t
    kb = k * beta[..., None]
    A = ((kb @ k.transpose(-1, -2)) * decay).masked_fill(~lower.tril(-1), 0)            # 严格下三角
    rhs = torch.cat([v * beta[..., None], kb * G.exp()[..., None]], -1)
    sol = torch.linalg.solve_triangular(eye + A, rhs, upper=False)        # (I + A)⁻¹ [βv, βΓk]
    Vn, Kc = sol[..., :m], sol[..., m:]
    attn = (q @ k.transpose(-1, -2)) * decay
    Sp, outs = S.transpose(-1, -2), []
    for i in range(n):
        W = Vn[:, :, i] - Kc[:, :, i] @ Sp                                 # 本块各拍的 β·(v − 旧状态预测)
        outs.append((q[:, :, i] * G[:, :, i, :, None].exp()) @ Sp + attn[:, :, i] @ W)
        tail = (G[:, :, i, -1:] - G[:, :, i]).exp()[..., None]
        Sp = Sp * G[:, :, i, -1, None, None].exp() + (k[:, :, i] * tail).transpose(-1, -2) @ W
    r = torch.stack(outs, 2).reshape(b, M, n * C, m)[:, :, :T].transpose(1, 2)
    return r, Sp.transpose(-1, -2)


class Layer(nn.Module):
    """一层模块，同层参数堆叠成 [M, …] 张量。所有时间步一起算门、投影和扫描，快记忆分块并行。
    lora_a / lora_b 是 MLP 第一层的秩 2 在线偏移（设计第 3 节），离线训练时冻结，在线阶段打开。"""
    def __init__(self, rows, cols, U, d, m, hidden, q, rank=2, beta_max=1.0):
        super().__init__()
        M = rows * cols
        self.M, self.d, self.m, self.beta_max = M, d, m, beta_max
        self.register_buffer('adj', grid_adjacency(rows, cols))
        p = lambda *s: nn.Parameter(torch.randn(*s) / math.sqrt(s[-2]))
        self.B = p(M, U, d); self.G = nn.Parameter(torch.zeros(M, U, d))
        tau = torch.logspace(math.log10(1.5), 3, d); a0 = 1 - 1 / tau       # 时间常数 1.5–1000 步
        self.b_a = nn.Parameter(torch.log(a0 / (1 - a0)).repeat(M, 1))
        self.K, self.V, self.Qm = p(M, d, m), p(M, d, m), p(M, d, m)
        self.w_beta = nn.Parameter(torch.zeros(M, d)); self.b_beta = nn.Parameter(torch.zeros(M))
        self.w_gam = nn.Parameter(torch.zeros(M, d)); self.b_gam = nn.Parameter(torch.full((M,), math.log(1 / 63)))
        self.W1 = p(M, d + m, hidden); self.b1 = nn.Parameter(torch.zeros(M, hidden))
        self.W2 = nn.Parameter(torch.zeros(M, hidden, d)); self.P = p(M, d, q)
        self.lora_a = nn.Parameter(torch.randn(M, d + m, rank) / math.sqrt(d + m), requires_grad=False)
        self.lora_b = nn.Parameter(torch.zeros(M, rank, hidden), requires_grad=False)

    def forward(self, u, state, active=None, reset=None, wbias=None):   # wbias: (b, T) 加到写入强度 β 的 logit（惊讶门控）；u: (b, T, M, U)；active: (b, M) 0/1，0 的模块本块冻结（RIMs 式稀疏更新）
        c0, S = state                                          # reset: (b, T) 布尔，会话边界字节处清空 c 与 S
        d, m = self.d, self.m
        ga, gb = torch.einsum('btmu,mud->btmd', u, torch.cat([self.G, self.B], -1)).split(d, -1)   # 读 u 的两个投影合成一次
        a = torch.sigmoid(ga + self.b_a)
        inp = (1 - a) * gb                                     # 输入比例用原门值：边界处只清旧状态，不放大写入
        if reset is not None:
            a = a.masked_fill(reset[..., None, None], 0.0)    # scan 把 a 钳到 0.01：边界处只留 1% 旧状态
        if active is not None:                                 # (b, M) 整块冻结，或 (b, T, M) 逐拍冻结（词级的补齐位置）
            active = active[:, None] if active.dim() == 2 else active
            act = active[..., None]
            a, inp = a * act + (1 - act), inp * act                           # 冻结：c 保持不变
        c, cT = scan(a, inp, c0)
        W = torch.cat([self.K, self.V, self.Qm, self.w_beta[..., None], self.w_gam[..., None]], -1)   # 读 c 的五个投影合成一次
        k, v, qv, bl, gl = torch.einsum('btmd,mde->btme', c, W).split([m, m, m, 1, 1], -1)
        k = F.normalize(k, dim=-1)
        wb = 0.0 if wbias is None else wbias[..., None]
        beta = self.beta_max * torch.sigmoid(bl[..., 0] + self.b_beta + wb)  # beta_max=2：负特征值；门控在 logit 上，β 仍 ≤ beta_max
        g = F.logsigmoid(-(gl[..., 0] + self.b_gam))                         # log α = log(1 − γ)
        if active is not None:
            beta, g = beta * active, g * active                                # 冻结：S 不写不衰减
        if reset is not None:
            g = g.masked_fill(reset[..., None], -30.0)                         # α≈0：边界处清空快记忆
        r, S = delta_chunk(qv, k, v, beta, g, S)
        W1 = self.W1 + self.lora_a @ self.lora_b
        hid = F.gelu(torch.einsum('btmi,mih->btmh', torch.cat([c, r], -1), W1) + self.b1)
        h = c + torch.einsum('btmh,mhd->btmd', hid, self.W2)
        return h, torch.einsum('btmd,mdq->btmq', h, self.P), (cT, S)


class PredHead(nn.Module):
    """每个模块一个线性预测头：h (b, T, M, d) → (b, T, M, q)。"""
    def __init__(self, M, d, q):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(M, d, q)); self.b = nn.Parameter(torch.zeros(M, q))

    def forward(self, h):
        return torch.einsum('btmd,mdq->btmq', h, self.W) + self.b


class DSPNLayered(nn.Module):
    """分层网格：消息同一拍内由下层传给上层（正下方模块 + 其四邻居），时间记忆只在 c、S（和中枢）中。
    hub_after=k 时在第 k、k+1 层之间插入全局记忆中枢（hub.py，设计第 1.2 节），读第 k 层全部消息与本拍嵌入，
    输出每个上层模块 4 维切片。"""
    def __init__(self, layers=4, rows=2, cols=4, d=32, m=16, hidden=64, q=8, emb=32, vocab=256,
                 hub_after=0, hub_heads=2, hub_dk=32, beta_max=1.0, hub_out=4, hub_taus=None,
                 ws_slots=0, ws_dim=16, ws_block=64,
                 word_layers=0, word_d=256, word_hidden=512, word_m=32, word_q=32, word_mods=1,
                 td_layers=0, td_pc=0.1, td_target=1,
                 pred_self=0.0, pred_nb=0.0, pred_down=0.0, local=False, surprise=False):
        super().__init__()
        M = rows * cols
        self.M, self.d, self.m, self.hub_after = M, d, m, hub_after
        self.emb = nn.Embedding(vocab, emb)
        # 词级（SpaceByte 式，设计 9.2）：word_layers>0 时，另有一摞只在词边界字节处运行的层；中枢移到词级，
        # hub_after 表示插在第几个词级层之后。词级输出保持到下一个边界，与字节级各层一起进读出。
        self.Mw, self.word_d, self.word_m = word_mods, word_d, word_m
        Mw, wq = word_mods, word_q
        self.register_buffer('wb', torch.zeros(vocab, dtype=torch.bool), persistent=False)
        self.wb[list(b' \n.,!?') + [SEP]] = True                                  # 词边界字节
        self.hub = None
        if hub_after:
            from hub import Hub
            n_in, M_out = (Mw * wq + emb, Mw) if word_layers else (M * q + emb, M)
            self.hub = Hub(n_in, M_out, out_dim=hub_out, heads=hub_heads, dk=hub_dk,
                           beta_max=beta_max, taus=hub_taus)
        Uw = lambda j: (M * q + emb if j == 0 else 2 * wq + (hub_out if hub_after and j == hub_after else 0))
        self.wlayers = nn.ModuleList([Layer(1, Mw, Uw(j), word_d, word_m, word_hidden, wq, beta_max=beta_max)
                                      for j in range(word_layers)])
        # 全局工作空间（设计 8.7）：每块末全部模块软竞争写入 ws_slots 个槽位，下一块广播给所有层所有模块（每模块 4 维）。
        # 完整版（硬竞争、选择性读、稀疏更新、点燃、预测编码）使训练锁死（E35），已删。
        self.ws_slots, self.ws_dim, self.ws_block = ws_slots, ws_dim, ws_block
        self.aux = 0.0                                                            # 辅助损失，每次前向重算，train 加进总损失
        if ws_slots:
            self.ws_id = nn.Parameter(torch.randn(layers * M, q) * 0.1)            # 模块身份，打分时区分来源
            self.ws_q = nn.Parameter(torch.randn(ws_slots, 16))
            self.ws_k = nn.Linear(q, 16, bias=False); self.ws_v = nn.Linear(q, ws_dim, bias=False)
            self.ws_keep = nn.Parameter(torch.zeros(ws_slots))                    # 槽位跨块保留门（维持）
            self.ws_in = nn.Parameter(torch.zeros(layers, ws_slots * ws_dim, M * 4))   # 零初始：起步等于无工作空间
        W = 4 if ws_slots else 0
        hub_byte = hub_after and not word_layers
        U = lambda i: (emb if i == 0 else 2 * q + (hub_out if hub_byte and i == hub_after else 0)) + W
        self.layers = nn.ModuleList([Layer(rows, cols, U(i), d, m, hidden, q, beta_max=beta_max) for i in range(layers)])
        # 自上而下（第二扫，TSPH 式，设计 10.1）：上层用上一拍的输出预测下层本拍的消息 z，误差 e = z − 预测；
        # 再用 [预测, 误差] 跑 td_layers 层字节级“自上而下层”，其输出也进读出。两扫各自整段并行；
        # 预测头以 MSE（目标停梯度）训练，权重 td_pc。有词级时：词级预测字节顶层；没有时：字节顶层预测第 td_target 层。
        self.td_pc, self.td_target = td_pc, td_target
        # 预测学习（可插拔，权重为 0 即关）：每个模块用本拍的 h 预测下一拍的消息 z——自己的（pred_self）、
        # 邻居的（pred_nb）、正下方下层模块的（pred_down，自上而下）。目标停梯度，误差按目标方差归一化以防坍缩。
        # local=True：层间消息（含进中枢的）停梯度，每层只由自己的读出路径与自己的预测损失训练（SOLO 式局部学习），
        # 模块内部（时间扫描、快记忆、MLP）仍用反传。
        self.pred_w = {'self': pred_self, 'nb': pred_nb, 'down': pred_down}
        self.local = local
        # 惊讶门控写入：第 i 层上一拍的自我预测与本拍消息的相对误差 s 是它此刻的惊讶度；
        # 第 i+1 层（与中枢）的写入强度 logit 加 a·(s − 1)，a 可学、初值 0（等于不门控）。上一拍的预测存在状态里。
        self.surprise, self.q = surprise, q
        if surprise:
            assert pred_self and not word_layers, 'surprise 需要 pred_self，且只用于字节级'
            self.sg_a = nn.Parameter(torch.zeros(layers + 1))                  # 最后一个给中枢
        self.pred = nn.ModuleDict({k: nn.ModuleList([PredHead(M, d, q) for _ in range(layers)])
                                   for k, w in self.pred_w.items() if w})
        if td_layers:
            src = word_layers * Mw * word_d if word_layers else M * d
            self.td_pred = nn.Linear(src, M * q)
            self.tdlayers = nn.ModuleList([Layer(rows, cols, 2 * q, d, m, hidden, q, beta_max=beta_max) for _ in range(td_layers)])
        else:
            self.tdlayers = nn.ModuleList()
        n_read = layers * M * d + word_layers * Mw * word_d + td_layers * M * d
        self.norm = nn.LayerNorm(n_read)
        self.readout = nn.Linear(n_read, emb)              # 瓶颈后与 E 共享（设计第 1.1 节）
        nn.init.zeros_(self.readout.weight)                # 共享 E 时从均匀分布起步，免得初始对数几率过大
        self.bias = nn.Parameter(torch.zeros(vocab))

    def init_state(self, b):
        z = lambda *s: torch.zeros(b, self.M, *s, device=self.emb.weight.device)
        st = [(z(self.d), z(self.m, self.m)) for _ in self.layers]
        dev = self.emb.weight.device
        zw = lambda *s: torch.zeros(b, self.Mw, *s, device=dev)
        st += [(zw(self.word_d), zw(self.word_m, self.word_m)) for _ in self.wlayers]
        st += [self.hub.init_state(b, dev)] if self.hub else []
        if self.wlayers:                                   # 最近一个词的词级输出（保持到下一个边界）
            st.append(torch.zeros(b, len(self.wlayers) * self.Mw * self.word_d, device=dev))
        st += [(z(self.d), z(self.m, self.m)) for _ in self.tdlayers]
        if self.tdlayers and not self.wlayers:             # 字节顶层上一拍的输出（自上而下预测的来源）
            st.append(torch.zeros(b, self.M * self.d, device=dev))
        if self.surprise:                                  # 各层上一拍对本层下一拍消息的预测
            st.append(tuple(z(self.q) for _ in self.layers))
        if self.ws_slots:      # (槽位, 各模块读到的 4 维, 已读字节数)；块边界按全局位置对齐，逐拍与整段一致
            dev, L = self.emb.weight.device, len(self.layers)
            st.append((torch.zeros(b, self.ws_slots, self.ws_dim, device=dev), torch.zeros(b, L, self.M, 4, device=dev), 0))
        return st

    def forward(self, idx, state=None):
        state = state or self.init_state(idx.shape[0])
        self.aux, self.pred_err = 0.0, {}
        if not self.ws_slots:
            lg, new, _ = self._run(idx, state, None)
            return lg, new
        slots, reads, pos = state[-1]
        st, outs, t, T = state[:-1], [], 0, idx.shape[1]
        while t < T:                                       # 块内整段并行，块间串行（槽位只在块边界更新）
            n = min(T - t, self.ws_block - pos % self.ws_block)
            lg, st, zs = self._run(idx[:, t:t + n], st, reads)
            outs.append(lg); t += n; pos += n
            if pos % self.ws_block == 0:
                slots, reads = self._ws_write(zs, slots)
        return torch.cat(outs, 1), st + [(slots, reads, pos)]

    def _ws_write(self, zs, slots):
        """各层各模块本块末拍的消息软竞争每个槽位，写入内容与旧槽位按保留门混合；再算下一块各模块读到的内容。"""
        b, L = slots.shape[0], len(self.layers)
        tok = torch.stack([z[:, -1] for z in zs], 1).flatten(1, 2)                         # (b, L·M, q)
        tok = tok.detach() if self.local else tok
        # 先归一化再写：槽位 → 下一块输入 → 消息 → 槽位是跨块的非线性环路，不归一化时增益过 1 会逐块指数爆炸（E43）
        tok = F.rms_norm(tok, tok.shape[-1:]) + self.ws_id
        sc = torch.einsum('sk,bnk->bsn', self.ws_q, self.ws_k(tok)) / 4.0
        write = torch.softmax(sc, -1) @ self.ws_v(tok)                                   # (b, K, w)
        keep = torch.sigmoid(self.ws_keep[None, :, None])
        slots = keep * slots + (1 - keep) * write
        reads = torch.einsum('bi,lio->blo', slots.flatten(1), self.ws_in).reshape(b, L, self.M, 4)
        return slots, reads

    def _run(self, idx, state, reads):
        b, T = idx.shape
        e = self.emb(idx)
        u = e[:, :, None, :].expand(b, T, self.M, -1)
        L = len(self.layers)
        hs, new, zs = [], [], []
        reset = idx == SEP
        pp, new_pp, s = (state[-1], [], None) if self.surprise else (None, None, None)
        for i, (layer, st) in enumerate(zip(self.layers, state)):
            if reads is not None:
                u = torch.cat([u, reads[:, i, None].expand(b, T, self.M, 4)], -1)
            wb = None if s is None else self.sg_a[i] * (s - 1)
            h, z, ns = layer(u, st, None, reset, wb)
            hs.append(h); new.append(ns); zs.append(z)
            if self.surprise:                              # 本层惊讶度：上一拍的预测 vs 本拍消息（相对误差，与批大小、T 无关）
                pr = self.pred['self'][i](h)
                prev = torch.cat([pp[i][:, None], pr[:, :-1]], 1)
                s = (((prev - z) ** 2).sum(-1) / (z ** 2).sum(-1).clamp(min=1e-6)).mean(-1).detach()   # (b, T)
                new_pp.append(pr[:, -1].detach())
            if self.pred:
                self._predict(i, layer, h, z, zs)
            zu = z.detach() if self.local else z
            u = torch.cat([zu, torch.einsum('ij,btjq->btiq', layer.adj, zu)], -1)
            if self.hub is not None and not self.wlayers and i + 1 == self.hub_after:
                o, hs_new = self.hub(torch.cat([zu.reshape(b, T, -1), e], -1), state[L], reset,
                                     wbias=None if s is None else self.sg_a[L] * (s - 1))
                u = torch.cat([u, o], -1)
        hs = [h.flatten(2) for h in hs]
        if self.wlayers:
            held, wnew = self._word(idx, e, zs[-1], reset, state[L:])
            hs.append(held); new += wnew
        elif self.hub is not None:
            new.append(hs_new)
        if self.tdlayers:                                  # 第二扫：上层上一拍的输出预测下层本拍的消息，两者相遇算误差
            if self.wlayers:
                src, prev0, zt = held, state[L + len(self.wlayers) + (self.hub is not None)], zs[-1]
            else:
                src, prev0, zt = hs[L - 1], state[len(new) + len(self.tdlayers)], zs[self.td_target]
            prev = torch.cat([prev0[:, None], src[:, :-1]], 1)
            pred = self.td_pred(prev).reshape(b, T, self.M, -1)
            self.aux = self.aux + self.td_pc * ((pred - zt.detach()) ** 2).mean()
            u = torch.cat([pred, zt - pred], -1)
            for layer, st in zip(self.tdlayers, state[len(new):]):
                h, z, ns = layer(u, st, None, reset)
                hs.append(h.flatten(2)); new.append(ns)
                u = torch.cat([z, torch.einsum('ij,btjq->btiq', layer.adj, z)], -1)
            if not self.wlayers:
                new.append(src[:, -1])
        if self.surprise:
            new.append(tuple(new_pp))
        y = self.readout(self.norm(torch.cat(hs, -1)))
        return y @ self.emb.weight.t() + self.bias, new, zs

    def _predict(self, i, layer, h, z, zs):
        """本拍 h 预测下一拍的消息；归一化 MSE 加进 self.aux，逐模块误差记在 self.pred_err（可作惊讶信号）。"""
        targets = {'self': z, 'nb': torch.einsum('ij,btjq->btiq', layer.adj, z), 'down': zs[i - 1] if i else None}
        for k, heads in self.pred.items():
            tgt = targets[k]
            if tgt is None or h.shape[1] < 2:
                continue
            tgt = tgt[:, 1:].detach()
            err = (heads[i](h[:, :-1]) - tgt).square().mean(-1) / (tgt.var() + 1e-4)    # (b, T−1, M)
            self.aux = self.aux + self.pred_w[k] * err.mean()
            self.pred_err[(k, i)] = err.detach()

    def _word(self, idx, e, zt, reset, st):
        """词级：把本段的词边界位置收成定长 K 的序列（边界在前、按时间排序，其余为补齐位并冻结），
        在其上跑词级层与中枢；每个字节位置取“到它为止最近一个词”的输出。超过 K 个边界时多出的并入第 K 个（需一段里过半字节是边界，极罕见；此时整段与逐拍结果不再一致）。"""
        b, T = idx.shape
        Wl = len(self.wlayers)
        hub_st, wprev = (st[Wl], st[Wl + 1]) if self.hub is not None else (None, st[Wl])
        isb = self.wb[idx]
        K = min(T, T // 2 + 1)                             # 边界超过一半字节才会溢出；1/3 在标点密的对话里会溢出
        pos = (torch.arange(T, device=idx.device) + T * (~isb).long()).argsort(-1)[:, :K]
        valid = isb.gather(1, pos)
        x = torch.cat([zt.flatten(2), e], -1)
        x = x.gather(1, pos[..., None].expand(-1, -1, x.shape[-1]))                     # (b, K, M·q + emb)
        rw = reset.gather(1, pos) & valid
        act = valid[..., None].float().expand(-1, -1, self.Mw)
        u = x[:, :, None].expand(-1, -1, self.Mw, -1)
        hw, new = [], []
        for j, (layer, s) in enumerate(zip(self.wlayers, st[:Wl])):
            h, z, ns = layer(u, s, act, rw)
            hw.append(h.flatten(2)); new.append(ns)
            u = torch.cat([z, torch.einsum('ij,btjq->btiq', layer.adj, z)], -1)
            if self.hub is not None and j + 1 == self.hub_after:
                o, hub_new = self.hub(torch.cat([z.flatten(2), x[..., -e.shape[-1]:]], -1), hub_st, rw, valid)
                u = torch.cat([u, o], -1)
        if self.hub is not None:
            new.append(hub_new)
        table = torch.cat([wprev[:, None], torch.cat(hw, -1)], 1)                       # 序号 0 = 上一段留下的词
        cum = isb.long().cumsum(1).clamp(max=K)
        held = table.gather(1, cum[..., None].expand(-1, -1, table.shape[-1]))
        new.append(table.gather(1, valid.sum(1)[:, None, None].expand(-1, 1, table.shape[-1]))[:, 0])
        return held, new


class GRULM(nn.Module):
    def __init__(self, hidden, emb=32, vocab=256):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.rnn = nn.GRU(emb, hidden, batch_first=True)
        self.readout = nn.Linear(hidden, vocab)

    def forward(self, idx, state=None):
        y, state = self.rnn(self.emb(idx), state)
        return self.readout(y), state


class TransformerLM(nn.Module):
    """小型因果 Transformer 对照（pre-LN，学习式位置嵌入，上下文 ctx）。"""
    def __init__(self, dim, layers=4, heads=4, ctx=2048, vocab=256):
        super().__init__()
        self.emb, self.pos, self.heads = nn.Embedding(vocab, dim), nn.Embedding(ctx, dim), heads
        self.blocks = nn.ModuleList([nn.ModuleDict(dict(
            ln1=nn.LayerNorm(dim), qkv=nn.Linear(dim, 3 * dim), proj=nn.Linear(dim, dim),
            ln2=nn.LayerNorm(dim), mlp=nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))))
            for _ in range(layers)])
        self.ln, self.head = nn.LayerNorm(dim), nn.Linear(dim, vocab)

    def forward(self, idx, state=None):
        b, T = idx.shape
        x = self.emb(idx) + self.pos(torch.arange(T, device=idx.device))
        for blk in self.blocks:
            q, k, v = blk.qkv(blk.ln1(x)).view(b, T, 3, self.heads, -1).transpose(1, 3).unbind(2)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + blk.proj(a.transpose(1, 2).reshape(b, T, -1))
            x = x + blk.mlp(blk.ln2(x))
        return self.head(self.ln(x)), None


def count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build(kind, size='S', **extra):
    cfg = {**SIZES[size], **extra}
    if kind == 'dspn':
        return DSPNLayered(**cfg)
    target = count(DSPNLayered(**cfg))                     # 按含 extra（中枢、工作空间、形状）的实际 DSPN 对齐
    if kind == 'gru':
        h = 16
        while count(GRULM(h + 1)) <= target:
            h += 1
        return GRULM(h)
    if kind == 'tf':
        dim = 16
        while count(TransformerLM(dim + 8)) <= target:     # 步长 8，4 个头仍整除
            dim += 8
        return TransformerLM(dim)
    raise ValueError(kind)


@torch.no_grad()
def evaluate(model, data, n=20, b=32, T=256):
    g = torch.Generator().manual_seed(0)
    tot = 0.0
    for _ in range(n):
        x, y = batch(data, b, T, g)
        lg, _ = model(x)
        tot += F.cross_entropy(lg.reshape(-1, 256), y.reshape(-1)).item()
    return tot / n / math.log(2)                            # 每字节比特数（bpc）


def param_groups(model, wd=0.01):
    """偏置、门偏置（b_a、b_gam、b_beta、中枢 b_alpha 等）与一维参数不加权重衰减：
    门偏置初值是 ±4–7 的大 logit，衰减会把时间常数持续拉短（3 万步时 τ=1000 的单元降到约 85）。"""
    no = lambda k, p: p.ndim < 2 or 'b_' in k or 'bias' in k
    ps = [(k, p) for k, p in model.named_parameters() if p.requires_grad]
    return [{'params': [p for k, p in ps if not no(k, p)], 'weight_decay': wd},
            {'params': [p for k, p in ps if no(k, p)], 'weight_decay': 0.0}]


def stream_batch(data, pos, T):
    """每行一条连续流：取 pos 处的 T+1 字节，pos 前移 T（到尾回绕）。配合跨批次延续状态。"""
    pos[pos + T + 1 >= len(data)] = 0
    xy = torch.from_numpy(np.stack([data[i:i + T + 1] for i in pos]).astype(np.int64))
    pos += T
    return xy[:, :-1].to(DEV), xy[:, 1:].to(DEV)


def bot_mask(y):
    """目标字节是否在 "Bot:" 开头的行里（对话微调只在模型轮次算损失）。窗口里第一个换行之前的半行无从判断，记为 0。"""
    b, T = y.shape
    start = torch.zeros_like(y, dtype=torch.bool); start[:, 1:] = y[:, :-1] == 10
    pad = F.pad(y, (0, 3), value=0)
    isbot = start & (pad[:, :T] == 66) & (pad[:, 1:T + 1] == 111) & (pad[:, 2:T + 2] == 116) & (pad[:, 3:T + 3] == 58)
    last = torch.where(start, torch.arange(T, device=y.device), -1).cummax(1).values
    return (isbot.gather(1, last.clamp(min=0)) & (last >= 0)).float()


def copy_state_(dst, src):
    """把新状态原地拷进固定地址的旧状态（CUDA Graph 回放时地址不能变）；整数位置不变（T 是工作空间块长的整数倍）。"""
    for d, s in zip(dst, src):
        if torch.is_tensor(d):
            d.copy_(s.detach())
        elif isinstance(d, (tuple, list)):
            copy_state_(d, s)


def train(kind, size, data, iters, b, T, lr, seed, tag, extra, init=None, stateful=False, bot_loss=False):
    torch.manual_seed(seed)
    tr, va = load_bytes(data)
    model = build(kind, size, **extra).to(DEV)
    if init:                                               # 从已有权重继续训练（如对话数据上微调）
        model.load_state_dict(torch.load(ROOT / init, map_location=DEV))
    nparam = count(model)
    params = [p for p in model.parameters() if p.requires_grad]
    cuda = DEV == 'cuda'
    lr_t = torch.tensor(lr, device=DEV)                    # 张量学习率：CUDA Graph 回放时原地改
    opt = torch.optim.AdamW(param_groups(model), lr=lr_t, capturable=cuda)
    sched = lambda it: min(1, it / 200) * 0.5 * (1 + math.cos(math.pi * it / iters))   # 预热 200 步后余弦衰减
    name = f'{kind}-{size}{tag}'
    print(f'{name}: {nparam:,} 参数，{DEV}，批 {b} × {T} 字节，{iters} 步', flush=True)
    sx, sy = batch(tr, b, T)                               # 固定地址的输入，每步原地拷入新批
    # stateful：每行是一条连续流，状态跨批次延续（截断反传），模型才练得到“记住 T 字节以前的事”（持久头、跨会话记忆）
    pos = np.random.default_rng(seed).integers(0, len(tr) - T - 1, b) if stateful else None
    st = model.init_state(b) if stateful else None
    smask = torch.ones(b, T, device=DEV)                   # bot_loss：只在 Bot 轮次的字节上算损失

    def step():
        lg, new = model(sx, st)
        ce = F.cross_entropy(lg.reshape(-1, 256), sy.reshape(-1), reduction='none')
        loss = (ce * smask.reshape(-1)).sum() / smask.sum().clamp(min=1)
        (loss + getattr(model, 'aux', 0.0)).backward()      # 预测学习等辅助损失
        nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad(set_to_none=False)
        if stateful:
            copy_state_(st, new)
        return loss.detach()                               # 不留计算图引用，否则会干扰 CUDA Graph 录制

    log, t0, eval_secs, graph = [], time.time(), 0.0, None
    for it in range(1, iters + 1):
        lr_t.fill_(lr * sched(it))
        x, y = stream_batch(tr, pos, T) if stateful else batch(tr, b, T)
        sx.copy_(x); sy.copy_(y)
        if bot_loss:
            smask.copy_(bot_mask(y))
        if not cuda or it <= 3:                            # 前 3 步在旁路流上热身（cuDNN/cuBLAS 选算法、分配梯度）
            if cuda:
                s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    loss = step()
                torch.cuda.current_stream().wait_stream(s)
            else:
                loss = step()
        else:
            if graph is None:                              # 把整个训练步录成一张 CUDA Graph，之后每步只回放一次，
                graph = torch.cuda.CUDAGraph()             # 省掉每步几千次 Python 调度与内核启动开销
                with torch.cuda.graph(graph):
                    loss = step()
            graph.replay()
        if it % 500 == 0 or it == iters:
            e0 = time.time()
            v = evaluate(model, va, n=5)
            eval_secs += time.time() - e0
            log.append((it, round(time.time() - t0), round(loss.item() / math.log(2), 3), round(v, 3)))
            if not math.isfinite(loss.item()):              # 发散就立刻停，不白跑一晚，也不用 nan 覆盖上一份检查点
                raise SystemExit(f'{name}: 第 {it} 步损失不是有限值，训练停止')
            (ROOT / 'runs').mkdir(exist_ok=True)
            torch.save(model.state_dict(), ROOT / f'runs/{name}.pt')   # 每次评估都存，长训练中途出事不至于全丢
            print(log[-1], flush=True)
    if DEV == 'cuda':
        torch.cuda.synchronize()                            # GPU 异步执行，计时前等它算完
    secs = time.time() - t0
    final = evaluate(model, va)
    (ROOT / 'runs').mkdir(exist_ok=True)
    torch.save(model.state_dict(), ROOT / f'runs/{name}.pt')   # 每个名字只留最新一份检查点
    out = ROOT / f'runs/{time.strftime("%Y%m%d-%H%M%S")}_{name}.json'
    json.dump({'name': name, 'kind': kind, 'size': size, 'extra': extra, 'params': nparam, 'data': data,
               'device': DEV, 'batch': b, 'T': T, 'iters': iters, 'lr': lr, 'seed': seed, 'stateful': stateful, 'bot_loss': bot_loss, 'init': init,
               'code_sha': hashlib.sha256((ROOT / 'dspn_text.py').read_bytes()).hexdigest()[:12],
               'seconds': round(secs), 'train_bytes_per_sec': round(iters * b * T / (secs - eval_secs)),
               'peak_gpu_mb': round(torch.cuda.max_memory_allocated() / 2**20) if DEV == 'cuda' else None,
               'val_bpc': round(final, 4), 'log': log}, open(out, 'w'), ensure_ascii=False, indent=1)
    print(f'{name} 完成：验证 bpc {final:.4f} → {out.name}', flush=True)


@torch.no_grad()
def sample(kind, size, tag, prompt, extra, n=600, temp=0.8, seed=1):
    torch.manual_seed(seed)
    model = build(kind, size, **extra).to(DEV)
    model.load_state_dict(torch.load(ROOT / f'runs/{kind}-{size}{tag}.pt', map_location=DEV)); model.eval()
    idx = torch.tensor([list(prompt.encode())], device=DEV)
    out = list(prompt.encode())
    if kind == 'tf':                                       # 无状态：每步重算整段上下文
        for _ in range(n):
            lg, _ = model(idx[:, -512:])
            nxt = torch.multinomial(F.softmax(lg[0, -1] / temp, -1), 1)
            idx = torch.cat([idx, nxt.view(1, 1)], 1); out.append(nxt.item())
    else:
        lg, state = model(idx)
        for _ in range(n):
            nxt = torch.multinomial(F.softmax(lg[0, -1] / temp, -1), 1)
            out.append(nxt.item())
            lg, state = model(nxt.view(1, 1), state)
    return bytes(out).decode('utf-8', errors='replace')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['train', 'sample']); ap.add_argument('kind', choices=['dspn', 'gru', 'tf'])
    ap.add_argument('--size', default='S', choices=list(SIZES))
    ap.add_argument('--data', default='tiny', choices=list(DATASETS))
    ap.add_argument('--iters', type=int, default=3000)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--T', type=int, default=256)
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', default='', help='名字后缀，区分同档不同配置')
    ap.add_argument('--extra', default='{}', help='覆盖 dspn 配置的 JSON，如 {"hub_after": 2}')
    ap.add_argument('--prompt', default='Once upon a time')
    ap.add_argument('--init', default=None, help='从该权重文件继续训练，如 runs/dspn-L_xxx.pt')
    ap.add_argument('--stateful', action='store_true', help='每行一条连续流，状态跨批次延续')
    ap.add_argument('--bot_loss', action='store_true', help='对话数据只在 Bot 轮次的字节上算损失')
    a = ap.parse_args()
    extra = json.loads(a.extra)
    if a.cmd == 'train':
        lr = a.lr or (1e-3 if a.kind == 'tf' else 3e-3)
        train(a.kind, a.size, a.data, a.iters, a.batch, a.T, lr, a.seed, a.tag, extra, a.init, a.stateful, a.bot_loss)
    else:
        print(sample(a.kind, a.size, a.tag, a.prompt, extra))
