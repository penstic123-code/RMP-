"""路由模块池（设计第 23 章，测试用候选）。

R1（mode='state'）：N 个现行 DSPN 模块（慢通道 c + 快记忆 S + 局部 MLP），每拍按地址选 k 个更新，
未选模块的 c、S 原样保持（扫描里门 a = 1、写入 0）。输出是被选模块 h、z 的加权和。
R0（mode='mlp'，对照）：一个稠密的循环核（慢通道 + 快记忆，每拍都算），只把局部 MLP 换成 N 选 k 的专家。

路由：查询 r = norm(W_r · rmsnorm(u))，选择分数 cos(r, a_i) + b_i 取前 k；
混合权重 softmax(cos / temp) 只在 k 个内归一化，不含偏置（DeepSeek 无辅助损失均衡：偏置只管选谁）。
地址 a_i：竞争学习（被选模块的地址向其查询的均值移一步，不求导）；偏置 b_i：按负载取符号调。
二者只在 plastic=True 且训练求导时更新；在线学习与评估时冻结（醒时冻结，第 23.3 节）。
原型阶段全部模块都算再乘掩码（正确性优先，不省算力；第 23.7 节）。
"""
import math
import torch, torch.nn as nn, torch.nn.functional as F
from dspn_text import Layer


class _ExpertCore(Layer):
    """R0 的核：M = 1 的现行模块，局部 MLP 换成 N 个专家，按 self.w（(b, T, N)，未选为 0）加权。"""
    def __init__(self, N, U, d, m, hidden, q, beta_max):
        super().__init__(1, U, d, m, hidden, q, beta_max=beta_max)
        del self.W1, self.b1, self.W2, self.lora_a, self.lora_b
        self.W1 = nn.Parameter(torch.randn(N, d + m, hidden) / math.sqrt(d + m))
        self.b1 = nn.Parameter(torch.zeros(N, hidden))
        self.W2 = nn.Parameter(torch.zeros(N, hidden, d))
        self.w = None

    def mlp(self, c, r):
        inp = torch.cat([c, r], -1)[:, :, 0]
        hid = F.gelu(torch.einsum('bti,nih->btnh', inp, self.W1) + self.b1)
        h = c[:, :, 0] + torch.einsum('btn,btnd->btd', self.w.to(hid.dtype), torch.einsum('btnh,nhd->btnd', hid, self.W2))
        h = h[:, :, None]
        return h, torch.einsum('btmd,mdq->btmq', h, self.P)


class PoolLayer(nn.Module):
    """输入 u: (b, T, U)（下层输出，每拍一个向量）；输出 h: (b, T, 1, d)、z: (b, T, 1, q)、新状态。"""
    def __init__(self, N, k, U, d, m, hidden, q, qr=32, mode='state', beta_max=1.0, temp=0.1, eta_a=0.01, eta_b=1e-3):
        super().__init__()
        self.N, self.k, self.mode, self.temp, self.eta_a, self.eta_b = N, k, mode, temp, eta_a, eta_b
        self.core = Layer(N, U, d, m, hidden, q, beta_max=beta_max) if mode == 'state' else _ExpertCore(N, U, d, m, hidden, q, beta_max)
        self.Wr = nn.Parameter(torch.randn(U, qr) / math.sqrt(U))
        self.register_buffer('addr', F.normalize(torch.randn(N, qr), dim=-1))
        self.register_buffer('bias', torch.zeros(N))
        self.register_buffer('load_ema', torch.full((N,), 1.0 / N))
        self.plastic = True         # 预训练：地址与偏置随训练更新；在线学习时设 False（醒时冻结）

    def init_state(self, b, device):
        return self.core.init_state(b, device)

    def route(self, u, keep=None):
        r = F.normalize(F.rms_norm(u, (u.shape[-1],)) @ self.Wr, dim=-1)       # (b, T, qr)
        cos = r @ self.addr.clone().t().to(r.dtype)                             # (b, T, N)；取副本：_adapt 原地改地址，反传要用旧值
        idx = (cos.float() + self.bias).topk(self.k, -1).indices
        sel = torch.zeros_like(cos, dtype=torch.float32).scatter_(-1, idx, 1.0)
        if keep is not None:
            sel = sel * keep[..., None]
        w = torch.softmax((cos.float() / self.temp).masked_fill(sel == 0, -1e9), -1) * sel   # 补齐位整行为 0
        return r, sel, w

    def forward(self, u, state, reset=None, keep=None):
        b, T, U = u.shape
        r, sel, w = self.route(u, keep)
        if self.training and torch.is_grad_enabled() and self.plastic:
            self._adapt(r.detach().float(), sel)
        if self.mode == 'state':
            h, z, ns = self.core(u[:, :, None].expand(b, T, self.N, U), state, reset, sel)   # 按模块的 keep：未选模块状态不变
            wh = w.to(h.dtype)[..., None]
            return (wh * h).sum(2, keepdim=True), (wh * z).sum(2, keepdim=True), ns
        self.core.w = w
        return self.core(u[:, :, None], state, reset, keep)

    @torch.no_grad()
    def _adapt(self, r, sel):
        cnt = sel.sum((0, 1))                                                   # (N,)
        mean_r = torch.einsum('btn,btq->nq', sel, r) / cnt.clamp(min=1)[:, None]
        upd = (cnt > 0).float()[:, None]
        self.addr.copy_(F.normalize(self.addr + self.eta_a * upd * (mean_r - self.addr), dim=-1))
        load = cnt / cnt.sum().clamp(min=1)
        self.bias.add_(self.eta_b * torch.sign(1.0 / self.N - load))
        self.load_ema.mul_(0.99).add_(0.01 * load)

    def stats(self):
        """负载熵 / log N（1 = 完全均衡）与死模块比例（近期负载 < 目标的 1/10）。"""
        p = self.load_ema / self.load_ema.sum()
        ent = float(-(p * p.clamp(min=1e-12).log()).sum() / math.log(self.N))
        dead = float((p < 0.1 / self.N).float().mean())
        return round(ent, 3), round(dead, 3)

    def compile_parts(self):
        self.core.compile_parts()


class WavePool(nn.Module):
    """R2：一个池，N 个模块各属一波（buffer wave，0..G−1），按波依次算；波 w 读残差流 x_w（含此前所有波的输出，跨级可见）。
    全部波共享一个地址空间（同一 W_r、同一组地址）。每波只在本波模块里选前 k；空波输出 0（等于“跳过”）。
    原型：每波把 N 个模块都算一遍、只让本波被选的模块改状态，所以每个模块每拍只更新一次（在自己的波里）；G 波的计算图固定，
    改波只改 buffer，CUDA Graph 不用重录。
    深度生长（grow_every=P > 0）：每 P 个训练步，模块可移到相邻波（最多移到当前最深波之后一波），条件是那一波的平均查询
    与它地址的平均余弦比本波高出 margin；每次最多移 N/8 个，移出后本波至少留 k 个。init='flat' 全在第 0 波，'spread' 均分到 G 波。"""
    def __init__(self, N, k, G, D, d, m, hidden, qr=32, beta_max=1.0, temp=0.1, eta_a=0.01, eta_b=1e-3,
                 init='flat', grow_every=0, margin=0.02):
        super().__init__()
        self.N, self.k, self.G, self.temp, self.eta_a, self.eta_b = N, k, G, temp, eta_a, eta_b
        self.grow_every, self.margin = grow_every, margin
        self.core = Layer(N, D, d, m, hidden, D, beta_max=beta_max)
        self.Wr = nn.Parameter(torch.randn(D, qr) / math.sqrt(D))
        self.register_buffer('addr', F.normalize(torch.randn(N, qr), dim=-1))
        self.register_buffer('bias', torch.zeros(N))
        self.register_buffer('load_ema', torch.full((N,), 1.0 / N))
        w0 = torch.zeros(N, dtype=torch.long) if init == 'flat' else torch.arange(N) * G // N
        self.register_buffer('wave', w0)
        self.register_buffer('aff', torch.zeros(N, G))           # 各模块与各波查询的平均余弦（EMA）
        self.register_buffer('step', torch.zeros((), dtype=torch.long))
        self.capture = None                                          # 分析用：设成列表时每波追加 (波号, sel, 混合权重)
        self.register_buffer('moved', torch.zeros(N))               # 上次移波的模块：下一次前向开头清空它们的状态
        self.plastic = True

    def init_state(self, b, device):
        return self.core.init_state(b, device)

    def forward(self, x, state, reset=None):
        """x: (b, T, D) 残差流起点。返回各波混合 h 的列表（每个 (b, T, d)）、最终残差流、新状态。"""
        b, T, D = x.shape
        learn = self.training and torch.is_grad_enabled() and self.plastic
        if learn and self.grow_every:                                         # 移波的模块从零状态开始
            keepm = 1 - self.moved
            state = (state[0] * keepm[None, :, None], state[1] * keepm[None, :, None, None])
            self.moved.zero_()
        addr = self.addr.clone()
        hs, cnt_all, sum_r, aff_new = [], [], [], []
        for w in range(self.G):
            r = F.normalize(F.rms_norm(x, (D,)) @ self.Wr, dim=-1)
            cos = (r @ addr.t().to(r.dtype)).float()                          # (b, T, N)
            cand = (self.wave == w)
            idx = (cos + self.bias).masked_fill(~cand, -1e9).topk(self.k, -1).indices
            sel = torch.zeros_like(cos).scatter_(-1, idx, 1.0) * cand.float()
            wt = torch.softmax((cos / self.temp).masked_fill(sel == 0, -1e9), -1) * sel
            if self.capture is not None:
                self.capture.append((w, sel.detach(), wt.detach()))
            h, z, state = self.core(x[:, :, None].expand(b, T, self.N, D), state, reset, sel)
            hw = wt.to(h.dtype)[..., None]
            hs.append((hw * h).sum(2)); x = x + (hw * z).sum(2)
            if learn:
                with torch.no_grad():
                    rd = r.detach().float()
                    cnt_all.append(sel.sum((0, 1))); sum_r.append(torch.einsum('btn,btq->nq', sel, rd))
                    ac = cos.mean((0, 1)); aff_new.append(ac - ac.mean())        # 相对匹配度：减去本波全体模块的均值，消掉“越深查询越偏向公共方向”的整体偏差
        if learn:
            self._adapt(torch.stack(cnt_all).sum(0), torch.stack(sum_r).sum(0), torch.stack(aff_new, 1), b * T)
        return hs, x, state

    @torch.no_grad()
    def _adapt(self, cnt, sum_r, aff, ticks):
        mean_r = sum_r / cnt.clamp(min=1)[:, None]
        self.addr.copy_(F.normalize(self.addr + self.eta_a * (cnt > 0).float()[:, None] * (mean_r - self.addr), dim=-1))
        size = F.one_hot(self.wave, self.G).sum(0).float()                    # 各波模块数
        target = self.k / size[self.wave].clamp(min=self.k)                   # 本波内的目标负载
        load = cnt / ticks
        self.bias.add_(self.eta_b * torch.sign(target - load))
        self.load_ema.mul_(0.99).add_(0.01 * load / target)                  # 相对目标的负载（1 = 刚好）
        self.aff.mul_(0.99).add_(0.01 * aff)
        self.step.add_(1)
        if self.grow_every:
            self._grow(size)

    @torch.no_grad()
    def _grow(self, size):
        """张量化的移波（无 Python 分支，CUDA Graph 可录）：只在 step % P == 0 时生效。"""
        do = (self.step % self.grow_every == 0)
        wv = torch.arange(self.G, device=self.wave.device)
        deepest = (self.wave.max() + 1).clamp(max=self.G - 1)
        allowed = ((wv[None] - self.wave[:, None]).abs() <= 1) & (wv[None] <= deepest)
        score = self.aff.masked_fill(~allowed, -1e9)
        best = score.argmax(1)
        gain = score.gather(1, best[:, None])[:, 0] - self.aff.gather(1, self.wave[:, None])[:, 0]
        cap = max(1, self.N // 8)
        ok = do & (gain > self.margin) & (best != self.wave) & (size[self.wave] >= self.k + cap)
        rank = torch.argsort(torch.argsort(-(gain * ok.float()), descending=False))
        move = ok & (rank < cap)
        self.wave.copy_(torch.where(move, best, self.wave))
        self.moved.copy_(move.float())

    def stats(self):
        """（深度 = 非空波数，死模块比例 = 相对负载 < 0.1，各波模块数）。"""
        size = F.one_hot(self.wave, self.G).sum(0)
        return int((size > 0).sum()), round(float((self.load_ema < 0.1).float().mean()), 3), tuple(int(s) for s in size)

    def compile_parts(self):
        self.core.compile_parts()


class WaveNet(nn.Module):
    """第 23 章最新设计的 S 级原型：字节嵌入 → 稠密编码层（现行模块，M 个）→ 投影到残差流 → 波池（R2）→ 读出。
    读出拼接编码层 h 与各波混合 h（空波为 0），同现行 DSPN 的读出。
    R3（fb=W > 0）：序列按 W 拍分块依次算；每块开头把上一块最后一拍的各波 h 经零初始投影加到残差流起点（按块延迟的自上而下反馈）。"""
    def __init__(self, M=2, d=90, m=16, hidden=180, q=16, emb=32, vocab=256, beta_max=1.0,
                 N=48, k=4, G=0, D=32, pd=64, pm=16, phidden=128, qr=32, temp=0.1, init='flat', grow_every=0, margin=0.02, fb=0, **_):
        super().__init__()
        self.M, self.fb, self.chunk, self.pool = M, fb, 0, N
        G = G or N // k                                    # 不另设深度上限：最多 N/k 波（每波至少要有 k 个可选），只受模块数限制
        self.emb = nn.Embedding(vocab, emb)
        enc = Layer(M, emb, d, m, hidden, q, beta_max=beta_max)
        self.layers = nn.ModuleList([enc, WavePool(N, k, G, D, pd, pm, phidden, qr, beta_max, temp, init=init, grow_every=grow_every, margin=margin)])
        self.win = nn.Linear(M * q, D)
        n_read = M * d + G * pd
        if fb:
            self.fbp = nn.Linear(G * pd, D); nn.init.zeros_(self.fbp.weight); nn.init.zeros_(self.fbp.bias)
        self.norm = nn.LayerNorm(n_read)
        self.readout = nn.Linear(n_read, emb); nn.init.zeros_(self.readout.weight)
        self.bias = nn.Parameter(torch.zeros(vocab))
        self.aux, self.stats = 0.0, {}

    def init_state(self, b):
        dev = self.emb.weight.device
        st = [l.init_state(b, dev) for l in self.layers]
        if self.fb:
            st.append(torch.zeros(b, self.layers[1].G * self.layers[1].core.d, device=dev))
        return st

    def _seg(self, e, idx, st_enc, st_pool, fbv):
        reset = idx == 30
        M = self.M
        h0, z0, ns0 = self.layers[0](e[:, :, None].expand(*e.shape[:2], M, e.shape[-1]), st_enc, reset)
        x = self.win(z0.flatten(2))
        if fbv is not None:
            x = x + self.fbp(fbv.to(x.dtype))[:, None]
        hs, _, ns1 = self.layers[1](x, st_pool, reset)
        return torch.cat([h0.flatten(2)] + hs, -1), ns0, ns1, torch.cat(hs, -1)[:, -1]

    def forward(self, idx, state=None):
        state = state or self.init_state(idx.shape[0])
        e = self.emb(idx)
        T = idx.shape[1]
        if not self.fb:
            hcat, ns0, ns1, _ = self._seg(e, idx, state[0], state[1], None)
            new = [ns0, ns1]
        else:                                                # 按块依次算：块内并行，块间传反馈
            s0, s1, fbv, outs = state[0], state[1], state[2], []
            for t0 in range(0, T, self.fb):
                hc, s0, s1, last = self._seg(e[:, t0:t0 + self.fb], idx[:, t0:t0 + self.fb], s0, s1, fbv)
                outs.append(hc); fbv = last.float()
            hcat, new = torch.cat(outs, 1), [s0, s1, fbv]
        y = self.readout(self.norm(hcat))
        return y @ self.emb.weight.t().to(y.dtype) + self.bias, new
