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
from torch.utils.checkpoint import checkpoint
from dspn_text import Layer


def ema_scan(x, e0, a, reset=None, C=64):
    """带重置的指数滑动平均 e_t = g_t·e_{t-1} + (1−a)·x_t，g_t = a（会话边界处 0）。按 C 拍分块：块内用衰减矩阵，块间传 e。
    a <= 0 时直通（无历史衰减，直接返回当前输入 x）。x: (b, T, D)；e0: (b, D)。返回 (b, T, D) 与最后一拍的 e。"""
    if a <= 0.0:
        xf = x.float()
        return xf, xf[:, -1]
    b, T, D = x.shape
    if T == 1 and reset is None:
        e = a * e0.float() + (1.0 - a) * x[:, 0].float()
        return e[:, None], e
    lg = torch.full((b, T), math.log(a), device=x.device)
    if reset is not None:
        lg = lg.masked_fill(reset, -30.0)                                   # exp(−30) ≈ 0：边界处忘掉此前的上下文
    outs, e = [], e0.float()
    for t0 in range(0, T, C):
        xc = x[:, t0:t0 + C].float(); L = lg[:, t0:t0 + C].cumsum(1); c = xc.shape[1]
        tri = torch.ones(c, c, dtype=torch.bool, device=x.device).tril()
        M = torch.exp((L[:, :, None] - L[:, None, :]).masked_fill(~tri, float('-inf')))
        y = (1 - a) * torch.einsum('bij,bjd->bid', M, xc) + torch.exp(L)[..., None] * e[:, None]
        e = y[:, -1]; outs.append(y)
    return torch.cat(outs, 1), e


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
    """R2：一个池，N 个模块各属一波（buffer wave，0..G−1），按波依次算；波 w 读残差流 x_w（bottleneck=True 时只读上一波输出）。
    解耦为三段：查询与语境跟踪（_wave_query）、波内路由与选择（_route_wave）、执行（固定分波自动走切片省 G 倍显存算力，动态移波走全量掩码）。
    深度生长（grow_every=P > 0）：每 P 个训练步，模块可移到相邻波。init='flat' 全在第 0 波，'spread' 均分到 G 波。"""
    def __init__(self, N, k, G, D, d, m, hidden, qr=32, beta_max=1.0, temp=0.1, eta_a=0.01, eta_b=1e-3,
                 init='flat', grow_every=0, margin=0.02, spare=0, rule='addr', grow_until=0, ctx_decay=0.97, tau_prior=0.0,
                 bottleneck=False, wave_query=False, wave_ctx=None, lora_rank=2, shared=0, inhib=0, grad_ckpt=False):
        super().__init__()
        self.N, self.k, self.G, self.temp, self.eta_a, self.eta_b = N, k, G, temp, eta_a, eta_b
        self.grow_every, self.margin, self.shared, self.inhib, self.grad_ckpt = grow_every, margin, shared, inhib, grad_ckpt
        self.bottleneck, self.wave_query = bottleneck, wave_query
        # rule='ctx' 且 wave_query=True 时默认开启逐波独立语境 EMA（wave_ctx=True）：第 w 波对实际流入本波的 x 做 EMA；
        # 显式传 wave_ctx=False 可兼容旧版共享编码层 EMA 检查点。ctx_decay 支持单浮点数或长度为 G 的列表（逐波多尺度衰减）。
        self.wave_ctx = (bool(wave_query) if wave_ctx is None else bool(wave_ctx)) if rule == 'ctx' else False
        self.rule, self.grow_until, self.ctx_decay = rule, grow_until, ctx_decay
        self.ctx_decays = [float(c) for c in ctx_decay] if isinstance(ctx_decay, (list, tuple)) else [float(ctx_decay)] * G
        assert len(self.ctx_decays) == G, f'ctx_decay 长度需等于 G={G}'
        if rule == 'hash':
            self.register_buffer('htab', torch.rand(256, G, N))
        if rule == 'gate':
            self.Wg = nn.Parameter(torch.randn(D, N) / math.sqrt(D))
        self.core = Layer(N, D, d, m, hidden, D, rank=lora_rank, beta_max=beta_max)
        if wave_query:
            self.Wr = nn.Parameter(torch.randn(G, D, qr) / math.sqrt(D))
        else:
            self.Wr = nn.Parameter(torch.randn(D, qr) / math.sqrt(D))
        self.ogain = nn.Parameter(torch.ones(N))                  # 各模块输出增益：在线启用的新模块从 0 起（新生神经元先不发声）
        self.register_buffer('addr', F.normalize(torch.randn(N, qr), dim=-1))
        self.register_buffer('bias', torch.zeros(N))
        self.register_buffer('load_ema', torch.full((N,), 1.0 / N))
        w0 = torch.zeros(N, dtype=torch.long) if init == 'flat' else torch.arange(N) * G // N
        self.register_buffer('wave', w0)
        if tau_prior > 1:                                            # 结构先验：越深的波时间常数越长（第 w 波 ×tau_prior^w，上限 1e4 步），不规定语义
            with torch.no_grad():
                a0 = torch.sigmoid(self.core.b_a)
                tau = (1 / (1 - a0) * tau_prior ** w0.float()[:, None]).clamp(max=1e4)
                a1 = 1 - 1 / tau
                self.core.b_a.copy_(torch.log(a1 / (1 - a1)))
        # 空闲模块（第 23.3 节 O2）：每波留 spare 个，预训练时不参与选择与均衡（active = 0）；
        # 常驻共享模块（shared > 0）：每波前 shared 个模块每拍必选，预训练扛通用语法、在线与睡眠永久冻结；
        # 抑制性模块（inhib > 0）：每波紧接 shared 后的 inhib 个模块，执行分流乘法擦除（A）+ 跨波路由侧抑制（B），端到端可微。
        act = torch.ones(N)
        sh = torch.zeros(N)
        inh = torch.zeros(N)
        for w in range(G):
            idx = (w0 == w).nonzero()[:, 0]
            if shared and len(idx):
                sh[idx[:shared]] = 1.0
            if inhib and len(idx):
                inh[idx[shared:shared + inhib]] = 1.0
            if spare and len(idx):
                act[idx[-spare:]] = 0
        self.register_buffer('active', act)
        self.register_buffer('is_shared', sh)
        if inhib > 0:
            self.register_buffer('is_inhib', inh)
            self.inh_a = nn.Parameter(torch.full((G, D), 0.1))
            self.inh_b = nn.Parameter(torch.full((G, D), -4.0))
            self.W_inh_r = nn.Parameter(torch.randn(G, D, qr) / math.sqrt(D))
            self.inh_r_scale = nn.Parameter(torch.full((G,), -4.0))
        self.register_buffer('aff', torch.zeros(N, G))           # 各模块与各波查询的平均余弦（EMA）
        self.register_buffer('step', torch.zeros((), dtype=torch.long))
        self.capture = None                                          # 设成列表时每波追加 (波号, sel, 混合权重, 查询 r)：分析与在线分配用
        self.register_buffer('moved', torch.zeros(N))               # 上次移波的模块：下一次前向开头清空它们的状态
        self.plastic = True
        self.use_slice = True                                        # 固定分波时走切片执行（省 G 倍显存与算力）；测试对照可关
        self.sparse_active = False                                   # True 时每波仅 gather 实际被选中的活跃模块（O(k) 算力与显存，支持 N=512/1024）
        self.sparse_capture = None                                   # 设成列表时每波追加 (w, act_global, wt_active)，供 RMPSidecar 稀疏出口用
        self._fixed_slices = self._compute_fixed_slices()

    def _compute_fixed_slices(self):
        if self.grow_every:
            return None
        w = self.wave.detach().cpu().tolist()
        slices, pos = [], 0
        for wi in range(self.G):
            start = pos
            while pos < self.N and w[pos] == wi:
                pos += 1
            if pos - start < self.k + self.shared + self.inhib:
                return None
            slices.append(slice(start, pos))
        return slices if pos == self.N else None

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self._fixed_slices = self._compute_fixed_slices()

    def init_state(self, b, device):
        st = self.core.init_state(b, device)
        if self.rule != 'ctx':
            return st
        D = self.Wr.shape[-2]
        ec = torch.zeros(b, self.G, D, device=device) if self.wave_ctx else torch.zeros(b, D, device=device)
        return (*st, ec)

    def _contiguous_wave_slices(self):
        """若 grow_every=0 且各波为按序覆盖 0..N-1 的连续非空切片，返回缓存的各波 slice（避免 CUDA Graph 录图期 GPU→CPU 同步）。"""
        if not self.use_slice or self.grow_every or self.core.rec is not None:
            return None
        return self._fixed_slices

    def _wave_query(self, w, x, ec_in, reset, shared_xq=None):
        """生成第 w 波的路由查询 r（与本波更新后的语境状态 ec_new）。"""
        D = x.shape[-1]
        Wr_w = self.Wr[w] if self.wave_query else self.Wr
        if self.rule == 'ctx':
            if self.wave_ctx:
                xq, ec_new = ema_scan(F.rms_norm(x, (D,)), ec_in, self.ctx_decays[w], reset)
                return F.normalize(xq.to(x.dtype) @ Wr_w, dim=-1), ec_new
            return F.normalize(shared_xq.to(x.dtype) @ Wr_w, dim=-1), None
        return F.normalize(F.rms_norm(x, (D,)) @ Wr_w, dim=-1), None

    def _route_wave(self, w, x, r, addr, idx, sl=None, inh_r=None):
        """计算第 w 波（全池或切片 sl）的余弦/门分数、选中掩码 sel 与混合权重 wt。
        若 shared > 0，常驻共享模块必选；若 inhib > 0，抑制性模块必选执行（不占兴奋性 top-k 与残差加法权重），
        并接收上一波抑制向量 inh_r 施加非正路由侧抑制（端到端可微）。"""
        D = x.shape[-1]
        if sl is not None:
            if self.rule == 'gate':
                cos = torch.tanh((F.rms_norm(x, (D,)) @ self.Wg[:, sl]).float())
            elif self.rule == 'hash':
                cos = self.htab[idx, w, sl]
            else:
                cos = (r @ addr[sl].t().to(r.dtype)).float()
            sh_m = self.is_shared[sl] > 0
            inh_m = (self.is_inhib[sl] > 0) if self.inhib > 0 else None
            cand = (self.active[sl] > 0) & (~sh_m)
            if inh_m is not None:
                cand = cand & (~inh_m)
            bias = self.bias[sl]
            a_tgt = addr[sl]
        else:
            if self.rule == 'gate':
                cos = torch.tanh((F.rms_norm(x, (D,)) @ self.Wg).float())
            elif self.rule == 'hash':
                cos = self.htab[idx, w]
            else:
                cos = (r @ addr.t().to(r.dtype)).float()
            sh_m = (self.wave == w) & (self.is_shared > 0)
            inh_m = ((self.wave == w) & (self.is_inhib > 0)) if self.inhib > 0 else None
            cand = (self.wave == w) & (self.active > 0) & (~sh_m)
            if inh_m is not None:
                cand = cand & (~inh_m)
            bias = self.bias
            a_tgt = addr
        cos_eff = cos
        if inh_r is not None and self.inhib > 0 and w > 0:
            inh_pen = -F.softplus(self.inh_r_scale[w - 1]) * torch.sigmoid((inh_r @ a_tgt.t().to(inh_r.dtype)).float())
            cos_eff = cos + inh_pen
        top = (cos_eff + bias).masked_fill(~cand, -1e9).topk(self.k, -1).indices
        sel_r = torch.zeros_like(cos).scatter_(-1, top, 1.0) * cand.float()
        wt_r = torch.softmax((cos_eff / self.temp).masked_fill(sel_r == 0, -1e9), -1) * sel_r
        if self.rule == 'hash':
            wt_r = sel_r / sel_r.sum(-1, keepdim=True).clamp(min=1)
        if self.shared > 0:
            sh_f = sh_m.float()
            alpha = self.shared / (self.shared + self.k)
            sel = sel_r + sh_f
            wt = (1.0 - alpha) * wt_r + (alpha / self.shared) * sh_f
        else:
            sel, wt = sel_r, wt_r
        if inh_m is not None:
            sel = sel + inh_m.float()
        return cos, sel, wt

    def forward(self, x, state, reset=None, idx=None):
        """x: (b, T, D) 残差流起点；idx: (b, T) 当前字节（hash 规则用）。返回各波混合 h 的列表（每个 (b, T, d)）、最终残差流、新状态。"""
        b, T, D = x.shape
        learn = self.training and torch.is_grad_enabled() and self.plastic
        ec = None
        if self.rule == 'ctx':
            state, ec = state[:2], state[2]
        if learn and self.grow_every:                                         # 移波的模块从零状态开始
            keepm = 1 - self.moved
            state = (state[0] * keepm[None, :, None], state[1] * keepm[None, :, None, None])
            self.moved.zero_()
        shared_xq = None
        if self.rule == 'ctx' and not self.wave_ctx:
            shared_xq, ec = ema_scan(F.rms_norm(x, (D,)), ec, self.ctx_decays[0], reset)
        addr = self.addr.clone()
        slices = self._contiguous_wave_slices()
        use_sparse = self.sparse_active and (slices is not None) and (not learn)
        hs, cnt_all, sum_r, aff_new, ec_list = [], [], [], [], []
        cTs, Ss = ([], []) if (slices is not None and not use_sparse) else (None, None)
        if use_sparse:
            cT_full, S_full = state[0].clone(), state[1].clone()
        inh_r = None
        for w in range(self.G):
            ec_in = ec[:, w] if (self.rule == 'ctx' and self.wave_ctx) else None
            r, ec_new = self._wave_query(w, x, ec_in, reset, shared_xq)
            if ec_new is not None:
                ec_list.append(ec_new)
            sl = slices[w] if slices is not None else None
            cos, sel, wt = self._route_wave(w, x, r, addr, idx, sl, inh_r)
            if self.capture is not None:
                if sl is not None:
                    sel_full = x.new_zeros(b, T, self.N, dtype=torch.float32); sel_full[..., sl] = sel
                    wt_full = x.new_zeros(b, T, self.N, dtype=torch.float32); wt_full[..., sl] = wt
                    self.capture.append((w, sel_full.detach(), wt_full.detach(), r.detach()))
                else:
                    self.capture.append((w, sel.detach(), wt.detach(), r.detach()))
            z_inh = None
            if use_sparse:
                if b == 1 and T == 1 and self.shared == 0 and self.inhib == 0:
                    act_local = sel[0, 0].topk(self.k).indices               # 单拍自回归解码：纯 GPU topk，零 CPU-GPU 同步
                else:
                    act_local = (sel.sum((0, 1)) > 0).nonzero(as_tuple=False)[:, 0]
                act_global = act_local + sl.start
                sel_a, wt_a = sel[..., act_local], wt[..., act_local]
                if self.sparse_capture is not None:
                    self.sparse_capture.append((w, act_global, wt_a.detach()))
                h_a, z_a, (cT_a, S_a) = self.core.forward_slice(
                    x[:, :, None].expand(b, T, act_global.numel(), D),
                    (state[0][:, act_global], state[1][:, act_global]), act_global, reset, sel_a)
                cT_full[:, act_global] = cT_a
                S_full[:, act_global] = S_a
                hw = (wt_a * self.ogain[act_global].float()).to(h_a.dtype)[..., None]
                out_z = (hw * z_a).sum(2)
                hs.append((hw * h_a).sum(2))
                if self.inhib > 0:
                    im = self.is_inhib[act_global] > 0
                    if im.any():
                        z_inh = z_a[:, :, im].mean(2)
            elif sl is not None:
                nw = sl.stop - sl.start
                u_w = x[:, :, None].expand(b, T, nw, D)
                c0_w, S0_w = state[0][:, sl], state[1][:, sl]
                if self.grad_ckpt and self.training and torch.is_grad_enabled():
                    def _slice_fn(u_in, c0_in, S0_in, rst_in, sel_in, _sl=sl):
                        h_, z_, (cT_, S_) = self.core.forward_slice(u_in, (c0_in, S0_in), _sl, rst_in, sel_in)
                        return h_, z_, cT_, S_
                    h, z, cT_w, S_w = checkpoint(_slice_fn, u_w, c0_w, S0_w, reset, sel, use_reentrant=False)
                else:
                    h, z, (cT_w, S_w) = self.core.forward_slice(u_w, (c0_w, S0_w), sl, reset, sel)
                cTs.append(cT_w); Ss.append(S_w)
                hw = (wt * self.ogain[sl].float()).to(h.dtype)[..., None]
                out_z = (hw * z).sum(2)
                hs.append((hw * h).sum(2))
                if self.inhib > 0:
                    im = self.is_inhib[sl] > 0
                    if im.any():
                        z_inh = z[:, :, im].mean(2)
            else:
                rw = None if reset is None else reset[..., None] & (self.wave == w)   # 只重置本波模块：每个模块在自己的波里清一次
                h, z, state = self.core(x[:, :, None].expand(b, T, self.N, D), state, rw, sel)
                hw = (wt * self.ogain.float()).to(h.dtype)[..., None]
                out_z = (hw * z).sum(2)
                hs.append((hw * h).sum(2))
                if self.inhib > 0:
                    im = (self.wave == w) & (self.is_inhib > 0)
                    if im.any():
                        z_inh = z[:, :, im].mean(2)
            if z_inh is not None:
                g_inh = torch.sigmoid(z_inh.float() * self.inh_a[w] + self.inh_b[w]).to(x.dtype)
                x = out_z if self.bottleneck else (x * (1.0 - g_inh) + out_z)
                inh_r = F.normalize(z_inh @ self.W_inh_r[w], dim=-1, eps=1e-6)
            else:
                x = out_z if self.bottleneck else (x + out_z)
                inh_r = None
            if learn:
                with torch.no_grad():
                    rd = r.detach().float()
                    cnt_all.append(sel.sum((0, 1))); sum_r.append(torch.einsum('btn,btq->nq', sel, rd))
                    if sl is None:
                        ac = cos.mean((0, 1)); aff_new.append(ac - ac.mean())        # 相对匹配度：移波用
        if use_sparse:
            state = (cT_full, S_full)
        elif slices is not None:
            state = (torch.cat(cTs, 1), torch.cat(Ss, 1))
        if learn:
            cnt = torch.cat(cnt_all, 0) if slices is not None else torch.stack(cnt_all).sum(0)
            sr = torch.cat(sum_r, 0) if slices is not None else torch.stack(sum_r).sum(0)
            aff = None if slices is not None else torch.stack(aff_new, 1)
            self._adapt(cnt, sr, aff, b * T)
        if self.rule == 'ctx':
            ec_out = torch.stack(ec_list, 1) if self.wave_ctx else ec
            state = (*state, ec_out)
        return hs, x, state

    @torch.no_grad()
    def _adapt(self, cnt, sum_r, aff, ticks):
        routed = (self.active > 0) & (self.is_shared == 0) & ((self.is_inhib == 0) if self.inhib > 0 else True)
        mean_r = sum_r / cnt.clamp(min=1)[:, None]
        self.addr.copy_(F.normalize(self.addr + self.eta_a * ((cnt > 0) & routed).float()[:, None] * (mean_r - self.addr), dim=-1))
        size = (F.one_hot(self.wave, self.G).float() * routed.float()[:, None]).sum(0)   # 各波参与路由的模块数
        target = self.k / size[self.wave].clamp(min=self.k)                   # 本波内的目标负载
        load = cnt / ticks
        self.bias.add_(self.eta_b * torch.sign(target - load) * routed.float())  # 空闲与常驻共享模块不累积偏置
        self.load_ema.mul_(0.99).add_(0.01 * load / target)                  # 相对目标的负载（1 = 刚好）
        if aff is not None:
            self.aff.mul_(0.99).add_(0.01 * aff)
        self.step.add_(1)
        if self.grow_every:
            self._grow(size)

    @torch.no_grad()
    def _grow(self, size):
        """张量化的移波（无 Python 分支，CUDA Graph 可录）：只在 step % P == 0 时生效。"""
        do = (self.step % self.grow_every == 0) & ((self.step <= self.grow_until) | (self.grow_until == 0))
        wv = torch.arange(self.G, device=self.wave.device)
        deepest = (self.wave.max() + 1).clamp(max=self.G - 1)
        allowed = ((wv[None] - self.wave[:, None]).abs() <= 1) & (wv[None] <= deepest)
        score = self.aff.masked_fill(~allowed, -1e9)
        best = score.argmax(1)
        gain = score.gather(1, best[:, None])[:, 0] - self.aff.gather(1, self.wave[:, None])[:, 0]
        cap = max(1, self.N // 8)
        ok = do & (gain > self.margin) & (best != self.wave) & (size[self.wave] >= self.k + cap) & (self.active > 0)
        rank = torch.argsort(torch.argsort(-(gain * ok.float()), descending=False))
        move = ok & (rank < cap)
        self.wave.copy_(torch.where(move, best, self.wave))
        self.moved.copy_(move.float())

    def stats(self):
        """（深度 = 非空波数，死模块比例 = 相对负载 < 0.1，各波模块数）。"""
        size = F.one_hot(self.wave, self.G).sum(0)
        on = (self.active > 0) & (self.is_shared == 0) & ((self.is_inhib == 0) if self.inhib > 0 else True)
        return int((size > 0).sum()), round(float((self.load_ema[on] < 0.1).float().mean()), 3), tuple(int(s) for s in size)

    def compile_parts(self):
        self.core.compile_parts()


def window_counts(idx, H, vocab=256):
    """(b, T) 字节 → (b, T−H, vocab)：第 t 拍之后 H 个字节（t+1 … t+H）的直方图 / H。"""
    C = F.pad(F.one_hot(idx, vocab).float().cumsum(1), (0, 0, 1, 0))     # C[t] = 前 t 个字节的计数
    T = idx.shape[1]
    return (C[:, H + 1:T + 1] - C[:, 1:T - H + 1]) / H


class WaveNet(nn.Module):
    """第 23 章最新设计的 S 级原型：字节嵌入 → 稠密编码层（现行模块，M 个）→ 投影到残差流 → 波池（R2）→ 读出。
    读出拼接编码层 h 与各波混合 h（空波为 0），同现行 DSPN 的读出。
    R3（fb=W > 0）：序列按 W 拍分块依次算；每块开头把上一块最后一拍的各波 h 经零初始投影加到残差流起点（按块延迟的自上而下反馈）。"""
    def __init__(self, M=2, d=90, m=16, hidden=180, q=16, emb=32, vocab=256, beta_max=1.0,
                 N=48, k=4, G=0, D=32, pd=64, pm=16, phidden=128, qr=32, temp=0.1, init='flat', grow_every=0, margin=0.02, fb=0, spare=0, rule='addr', grow_until=0, ctx_decay=0.97, tau_prior=0.0, pred_w=0.0,
                 bottleneck=False, read_last=False, read_stream=False, wave_query=False, wave_ctx=None, lora_rank=2, shared=0, inhib=0, sep_id=30, grad_ckpt=False, tie_emb=False, **_):
        super().__init__()
        self.M, self.fb, self.chunk, self.pool = M, fb, 0, N
        self.read_last, self.read_stream, self.sep_id = read_last, read_stream, sep_id
        G = G or N // (k + shared + inhib)                 # 不另设深度上限：最多 N/(k+shared+inhib) 波
        self.emb = nn.Embedding(vocab, emb)
        if tie_emb:
            nn.init.normal_(self.emb.weight, std=0.02)     # 大词表/大维度下避免 N(0,1) 导致初始 logits 爆炸
        enc = Layer(M, emb, d, m, hidden, q, rank=lora_rank, beta_max=beta_max)
        self.layers = nn.ModuleList([enc, WavePool(N, k, G, D, pd, pm, phidden, qr, beta_max, temp, init=init, grow_every=grow_every, margin=margin, spare=spare, rule=rule, grow_until=grow_until, ctx_decay=ctx_decay, tau_prior=tau_prior, bottleneck=bottleneck, wave_query=wave_query, wave_ctx=wave_ctx, lora_rank=lora_rank, shared=shared, inhib=inhib, grad_ckpt=grad_ckpt)])
        # 分层预测（第 23 章 23.11）：第 w 波另有一个头，预测此后 2·4^w 个字节的分布（越深看得越远、越粗），权重 pred_w
        self.pred_w, self.pred_H = pred_w, [2 * 4 ** w for w in range(G)]
        if pred_w:
            self.pheads = nn.ModuleList([nn.Linear(pd, vocab) for _ in range(G)])
        self.win = nn.Linear(M * q, D)
        n_read = D if read_stream else (pd if read_last else (M * d + G * pd))
        if fb:
            self.fbp = nn.Linear(G * pd, D); nn.init.zeros_(self.fbp.weight); nn.init.zeros_(self.fbp.bias)
        self.norm = nn.LayerNorm(n_read)
        if read_stream and tie_emb and emb == D:
            self.readout = nn.Identity()
        else:
            self.readout = nn.Linear(n_read, emb); nn.init.zeros_(self.readout.weight)
        self.bias = nn.Parameter(torch.zeros(vocab))
        self.aux, self.stats = 0.0, {}

    def _pred_loss(self, hw, idx):
        """hw: (b, T, G·pd) 各波混合 h；软目标交叉熵，各波平均（窗口超出本段的拍不算）。"""
        G, T, loss, n = len(self.pheads), idx.shape[1], 0.0, 0
        for w, (hd, H) in enumerate(zip(self.pheads, self.pred_H)):
            if H >= T:
                continue
            tgt = window_counts(idx, H)
            lp = F.log_softmax(hd(hw[:, :T - H, w * hw.shape[-1] // G:(w + 1) * hw.shape[-1] // G]).float(), -1)
            loss = loss + -(tgt * lp).sum(-1).mean(); n += 1
        return loss / max(n, 1)

    def init_state(self, b):
        dev = self.emb.weight.device
        st = [l.init_state(b, dev) for l in self.layers]
        if self.fb:
            st.append((torch.zeros(b, self.layers[1].G * self.layers[1].core.d, device=dev), 0))
        return st

    def _seg(self, e, idx, st_enc, st_pool, fbv):
        reset = (idx == self.sep_id) if self.sep_id is not None else None
        M = self.M
        h0, z0, ns0 = self.layers[0](e[:, :, None].expand(*e.shape[:2], M, e.shape[-1]), st_enc, reset)
        x = self.win(z0.flatten(2))
        if fbv is not None:                                # 会话边界及之后不再注入上一块的旧会话反馈
            fb_mask = torch.ones_like(x[..., :1]) if reset is None else (reset.long().cumsum(1) == 0).to(x.dtype)[..., None]
            x = x + self.fbp(fbv.to(x.dtype))[:, None] * fb_mask
        hs, x_out, ns1 = self.layers[1](x, st_pool, reset, idx)
        if self.read_stream and not self.pred_w:
            last_h = torch.cat(hs, -1)[:, -1] if self.fb else None
            return x_out, ns0, ns1, last_h
        return torch.cat([h0.flatten(2)] + hs, -1), ns0, ns1, torch.cat(hs, -1)[:, -1]

    def forward(self, idx, state=None):
        state = state or self.init_state(idx.shape[0])
        e = self.emb(idx)
        T = idx.shape[1]
        if not self.fb:
            hcat, ns0, ns1, _ = self._seg(e, idx, state[0], state[1], None)
            new = [ns0, ns1]
        else:                                                # 按块依次算：块内并行，块间传反馈；块内进度 fb_pos 存状态，任意切块结果一致
            s0, s1, fb_st, outs = state[0], state[1], state[2], []
            fbv, fb_pos = fb_st if isinstance(fb_st, tuple) else (fb_st, 0)
            t0 = 0
            while t0 < T:
                step = min(self.fb - fb_pos, T - t0)
                idx_c = idx[:, t0:t0 + step]
                hc, s0, s1, last = self._seg(e[:, t0:t0 + step], idx_c, s0, s1, fbv)
                outs.append(hc)
                if self.sep_id is not None:
                    no_rst = (idx_c == self.sep_id).long().sum(1) == 0    # 无分支写法（CUDA Graph 可录）：块内出现会话边界后清零旧反馈
                    fbv = fbv * no_rst.to(fbv.dtype)[:, None]
                fb_pos += step
                if fb_pos == self.fb:
                    fbv = last.float()
                    fb_pos = 0
                t0 += step
            hcat, new = torch.cat(outs, 1), [s0, s1, (fbv, fb_pos)]
        if self.pred_w and self.training:
            self.aux = self.pred_w * self._pred_loss(hcat[..., self.M * self.layers[0].d:], idx)
        feat = hcat if self.read_stream else (hcat[..., -self.layers[1].core.d:] if self.read_last else hcat)
        y = self.readout(self.norm(feat))
        return y @ self.emb.weight.t().to(y.dtype) + self.bias, new
