"""迹柱网络 TraceCols（设计第 21 章候选）：以 e-prop 资格迹与涌现为核心的字节级模型。

- 跨时间的状态只有两种：逐通道门控递推 c（慢通道）与每头标量衰减的矩阵记忆 S（快记忆 / 中枢）。
  两者对“自己的参数”都有精确资格迹（c 的门控、输入矩阵；S 的 Wk、Wv、衰减），跨块延续，不反传时间。
- 非线性只在一拍之内：每柱一个 k-WTA 稀疏正激活读出层（稳态阈值防锁死）。
- 竞争：柱内 k-WTA；柱间按各柱局部预测误差 softmax 分配输出与学习信号；中枢只按新奇写入。
- 层级事件驱动：下层预测误差冲高 → 上层门 a^p、写入 ×p；p 低时上层保持。
- 学习信号（--target）：self 预测自身下一输入 / down 预测下层下一输入（最下层预测字节）/ ext 各层直接预测外部下一字节。
- 信用（--credit）：eprop = 读出交叉熵当拍跨层反传 + 局部预测 + 资格迹；local = 层间完全切断，只靠局部预测；
  tbptt = 块内整段反传、块间截断（只作标定尺子）。
"""
import argparse, copy, json, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
LN2 = math.log(2)
try:
    from fla.ops.hgrn import chunk_hgrn
    HAS_FLA = True
except Exception:
    HAS_FLA = False


def rms(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def scan0(la, q, chunk=16):
    """零初值对角递推 c_t = exp(la_t)·c_{t−1} + q_t（时间在维 1）。"""
    sh = q.shape
    if HAS_FLA and q.is_cuda and q.dtype == torch.float32 and sh[1] >= 16:
        return chunk_hgrn(q.flatten(2).contiguous(), la.flatten(2).contiguous(), None, False)[0].view(sh)
    b, T = sh[:2]
    pad = -T % chunk
    if pad:
        la, q = (F.pad(t.flatten(2), (0, 0, 0, pad)).view(b, T + pad, *sh[2:]) for t in (la, q))
    n = (T + pad) // chunk
    LA = la.reshape(b, n, chunk, *sh[2:]).cumsum(2)
    X = q.reshape(LA.shape)
    D = LA[:, :, :, None] - LA[:, :, None]
    tri = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).tril().view(1, 1, chunk, chunk, *[1] * (q.dim() - 2))
    local = (torch.where(tri, D, float('-inf')).exp() * X[:, :, None]).sum(3)
    outs, c = [], torch.zeros_like(q[:, 0])
    for i in range(n):
        outs.append(local[:, i] + LA[:, i].exp() * c[:, None])
        c = outs[-1][:, -1]
    return torch.cat(outs, 1)[:, :T]


def ema(x, x0, lam):
    """y_t = lam·y_{t−1} + (1−lam)·x_t，y_0 = x0。不求导的统计量用。"""
    T = x.shape[1]
    t = torch.arange(T, device=x.device, dtype=x.dtype)
    dt = t[:, None] - t[None]
    W = (1 - lam) * lam ** dt.clamp(min=0) * (dt >= 0)
    sh = (1, T) + (1,) * (x.dim() - 2)
    return torch.einsum('ts,bs...->bt...', W, x) + (lam ** (t + 1)).view(sh) * x0[:, None]


def logu(shape, lo, hi):
    return torch.exp(torch.empty(shape).uniform_(math.log(lo), math.log(hi)))


class SlowChan(nn.Module):
    """c_t = A_t·c_{t−1} + (1−A_t)·W u_t，A_t = σ(G u_t + b)^p。对 W、G、b 带精确资格迹。"""

    def __init__(s, C, n, d, tau):
        super().__init__()
        s.W = nn.Parameter(torch.randn(C, n, d) / math.sqrt(d))
        s.G = nn.Parameter(torch.randn(C, n, d) * 0.1 / math.sqrt(d))
        a = 1 - 1 / logu((C, n), *tau)
        s.b = nn.Parameter(torch.log(a / (1 - a)))
        s.last_la = None

    def init_state(s, B, dev, tr):
        C, n, d = s.W.shape
        st = {'c': torch.zeros(B, C, n, device=dev)}
        if tr:
            st.update(EW=torch.zeros(B, C, n, d, device=dev), EG=torch.zeros(B, C, n, d, device=dev), Eb=torch.zeros(B, C, n, device=dev))
        return st

    def forward(s, u, p, st, mode, rec=False):
        blk = mode == 'eprop'
        ud = u.detach() if blk else u
        ls = F.logsigmoid(torch.einsum('btd,cnd->btcn', ud, s.G) + s.b)
        pp = 1. if p is None else p[:, :, None, None]
        la = ls * pp
        A = la.exp()
        X = torch.einsum('btd,cnd->btcn', ud, s.W)
        c0 = st['c']
        if blk:                                            # 块前历史对自身参数的影响：c0 的导数 = 资格迹
            c0 = (c0 + torch.einsum('bcnd,cnd->bcn', st['EW'], s.W - s.W.detach())
                  + torch.einsum('bcnd,cnd->bcn', st['EG'], s.G - s.G.detach()) + st['Eb'] * (s.b - s.b.detach()))
        LA = la.cumsum(1)
        c = scan0(la, (1 - A) * X) + LA.exp() * c0[:, None]
        cprev = torch.cat([st['c'][:, None], c[:, :-1].detach()], 1)
        if blk:                                            # 对输入 u 只留当拍通路（值不变）
            Au = (F.logsigmoid(torch.einsum('btd,cnd->btcn', u, s.G.detach()) + s.b.detach()) * pp).exp()
            Xu = torch.einsum('btd,cnd->btcn', u, s.W.detach())
            c = c + (Au - Au.detach()) * (cprev - X.detach()) + (1 - A.detach()) * (Xu - Xu.detach())
        if rec:
            s.last_la = la.detach()
        new = {'c': c[:, -1].detach()}
        if blk:
            with torch.no_grad():
                P = (LA[:, -1:] - LA).exp()
                Pt = LA[:, -1].exp()
                jg = P * pp * A * (1 - ls.exp()) * (cprev - X)
                new['EW'] = Pt[..., None] * st['EW'] + torch.einsum('btcn,btd->bcnd', P * (1 - A), ud)
                new['EG'] = Pt[..., None] * st['EG'] + torch.einsum('btcn,btd->bcnd', jg, ud)
                new['Eb'] = Pt * st['Eb'] + jg.sum(1)
        return c, new


class LinMem(nn.Module):
    """S_t = γ·S_{t−1} + β_t·v_t k_tᵀ，读 o_t = S_t q_t；每头一个可学衰减 γ = exp(−e^r)。对 Wk、Wv、γ 带精确资格迹。"""

    def __init__(s, C, H, dk, dv, d, tau):
        super().__init__()
        s.Wk = nn.Parameter(torch.randn(C, H, dk, d) / math.sqrt(d))
        s.Wv = nn.Parameter(torch.randn(C, H, dv, d) / math.sqrt(d))
        s.Wq = nn.Parameter(torch.randn(C, H, dk, d) / math.sqrt(d))
        s.r = nn.Parameter(-torch.log(logu((C, H), *tau)))
        s.scale = dk ** -0.5

    def init_state(s, B, dev, tr):
        C, H, dk, d = s.Wk.shape
        dv = s.Wv.shape[2]
        st = {'S': torch.zeros(B, C, H, dv, dk, device=dev)}
        if tr:
            st.update(FV=torch.zeros(B, C, H, dk, d, device=dev), FK=torch.zeros(B, C, H, dv, d, device=dev),
                      Eg=torch.zeros(B, C, H, dv, dk, device=dev))
        return st

    def forward(s, z, beta, st, mode):
        blk = mode == 'eprop'
        zd = z.detach() if blk else z
        B, T = z.shape[:2]
        k = torch.einsum('btd,chkd->btchk', zd, s.Wk) * s.scale
        v = torch.einsum('btd,chkd->btchk', zd, s.Wv)
        q = torch.einsum('btd,chkd->btchk', z, s.Wq)
        lg = -s.r.exp()                                    # log γ，(C, H)
        S0 = st['S']
        if blk:
            S0 = (S0 + torch.einsum('chaj,bchkj->bchak', s.Wv - s.Wv.detach(), st['FV'])
                  + torch.einsum('bchaj,chkj->bchak', st['FK'], s.Wk - s.Wk.detach()) * s.scale
                  + st['Eg'] * (lg - lg.detach())[None, :, :, None, None])
        t = torch.arange(1, T + 1, device=z.device, dtype=z.dtype)
        dt = t[:, None] - t[None]
        D = torch.exp(lg[..., None, None] * dt.clamp(min=0)) * (dt >= 0)
        bb = torch.ones(B, T, device=z.device, dtype=z.dtype) if beta is None else beta
        att = torch.einsum('btchk,bschk->bchts', q, k) * D * bb[:, None, None, None, :]
        gt = torch.exp(lg[None] * t[:, None, None])        # (T, C, H)
        o = torch.einsum('bchts,bscha->btcha', att, v) + torch.einsum('bchak,btchk->btcha', S0, q) * gt[None, ..., None]
        if blk:                                            # 当拍写入对输入的通路
            ku = torch.einsum('btd,chkd->btchk', z, s.Wk.detach()) * s.scale
            vu = torch.einsum('btd,chkd->btchk', z, s.Wv.detach())
            o = o + bb[:, :, None, None, None] * (((ku - ku.detach()) * q.detach()).sum(-1, keepdim=True) * v.detach()
                                                  + (k.detach() * q.detach()).sum(-1, keepdim=True) * (vu - vu.detach()))
        with torch.no_grad():
            w = torch.exp(lg[None] * (T - t)[:, None, None])[None] * bb[:, :, None, None]
            GT = torch.exp(lg * T)[..., None, None]
            S0d = st['S']
            new = {'S': GT * S0d + torch.einsum('btch,btcha,btchk->bchak', w, v, k)}
            if blk:
                new['FV'] = GT * st['FV'] + torch.einsum('btch,btchk,btd->bchkd', w, k, zd)
                new['FK'] = GT * st['FK'] + torch.einsum('btch,btcha,btd->bchad', w, v, zd)
                new['Eg'] = GT * (st['Eg'] + T * S0d) + torch.einsum('btch,btcha,btchk->bchak', w * (T - t)[None, :, None, None], v, k)
        return o, new


class Level(nn.Module):
    """一级 = C 个相同的柱。柱 = 慢通道 + 快记忆 + k-WTA 稀疏读出 + 局部预测头。"""

    def __init__(s, C, d, n, H, dk, h, k, m, dt, temp):
        super().__init__()
        s.C, s.k, s.h, s.temp = C, k, h, temp
        s.slow = SlowChan(C, n, d, (2., 1e4))
        s.mem = LinMem(C, H, dk, dk, 2 * d, (8., 512.))
        f = d + n + H * dk
        s.Win = nn.Parameter(torch.randn(C, h, f) / math.sqrt(f))
        s.bin = nn.Parameter(torch.zeros(C, h))
        s.Wout = nn.Parameter(torch.randn(C, m, h) / math.sqrt(k))
        s.Wpred = nn.Parameter(torch.randn(C, dt, h) * 0.1 / math.sqrt(k))
        s.register_buffer('theta', torch.zeros(C, h))      # 稳态阈值
        s.register_buffer('cbias', torch.zeros(C))         # 柱的稳态偏置

    def init_state(s, B, dev, tr, d, dt):
        return {'slow': s.slow.init_state(B, dev, tr), 'mem': s.mem.init_state(B, dev, tr),
                'u_last': torch.zeros(B, d, device=dev), 'pred_last': torch.zeros(B, s.C, dt, device=dev),
                'ebar': torch.zeros(B, s.C, device=dev), 'lmean': torch.zeros(B, device=dev)}

    def forward(s, u, p, st, mode, tgt, rec=False):
        B, T, d = u.shape
        C = s.C
        up = torch.cat([st['u_last'][:, None], (u.detach() if mode == 'eprop' else u)[:, :-1]], 1)   # 上一拍输入只当历史
        c, n_slow = s.slow(u, p, st['slow'], mode, rec)
        o, n_mem = s.mem(torch.cat([u, up], -1), p, st['mem'], mode)
        feat = torch.cat([u[:, :, None].expand(B, T, C, d), rms(c), rms(o).flatten(3)], -1)
        z = torch.einsum('btcf,chf->btch', feat, s.Win) + s.bin - s.theta
        mask = torch.zeros_like(z).scatter_(-1, z.detach().topk(s.k, -1).indices, 1.)
        sp = F.relu(z) * mask
        msg = torch.einsum('btch,cmh->btcm', sp, s.Wout)
        pred = torch.einsum('btch,cdh->btcd', sp, s.Wpred)
        prev = torch.cat([st['pred_last'][:, None], pred[:, :-1].detach()], 1)
        if tgt[0] == 'ce':                                 # 预测外部下一字节（比特）
            ids, En = tgt[1], tgt[2]
            ce = lambda pr, y: F.cross_entropy((pr @ En.T).flatten(0, -2), y.flatten(), reduction='none').view(y.shape) / LN2
            e = ce(prev, ids[:, :, None].expand(B, T, C)).detach()
            lm = ce(pred[:, :-1], ids[:, 1:, None].expand(B, T - 1, C))
        else:                                              # 预测某个内部信号的下一拍（目标停梯度）
            y = tgt[1].detach()
            e = ((prev - y[:, :, None]) ** 2).mean(-1).detach()
            lm = ((pred[:, :-1] - y[:, 1:, None]) ** 2).mean(-1)
        ebar = ema(e, st['ebar'], 0.9)
        pi = torch.softmax(-ebar / s.temp + s.cbias, -1)   # 柱间竞争：预测得准的柱拿走输出与学习信号
        M = (pi[..., None] * msg).sum(2)
        loc = (lm * pi[:, :-1].detach() * C).mean()
        e_lvl = (pi * e).sum(-1)
        new = {'slow': n_slow, 'mem': n_mem, 'u_last': u[:, -1].detach(), 'pred_last': pred[:, -1].detach(),
               'ebar': ebar[:, -1].detach(), 'lmean': st['lmean']}
        stats = {'rate': mask.mean((0, 1)), 'win': pi.detach().mean((0, 1))}
        if rec:
            stats.update(mask=mask, pi=pi.detach(), e=e)
        return M, loc, e_lvl, new, stats

    @torch.no_grad()
    def homeostasis(s, stats, eta=0.01):
        s.theta.add_(eta * (stats['rate'] / (s.k / s.h) - 1)).clamp_(-5, 5)
        s.cbias.sub_(eta * (stats['win'] * s.C - 1)).clamp_(-5, 5)


class TraceCols(nn.Module):
    def __init__(s, L=3, C=4, d=64, n=64, H=2, dk=16, h=256, k=13, m=64, hubH=4, hub_dk=32, temp=0.25, kappa=4.0,
                 p_off=1.0, target='self'):
        super().__init__()
        s.cfg = dict(L=L, C=C, d=d, n=n, H=H, dk=dk, h=h, k=k, m=m, hubH=hubH, hub_dk=hub_dk, temp=temp, kappa=kappa,
                     p_off=p_off, target=target)
        s.target, s.kappa, s.p_off = target, kappa, p_off
        s.emb = nn.Embedding(256, d)
        s.dins = [d] + [m] * (L - 1)
        s.dts = [s._tdim(i) for i in range(L)]
        s.levels = nn.ModuleList([Level(C, s.dins[i], n, H, dk, h, k, m, s.dts[i], temp) for i in range(L)])
        s.hub = LinMem(1, hubH, hub_dk, hub_dk, L * m, (300., 1e5))
        s.ro = nn.Linear(L * m + hubH * hub_dk, 256)

    def _tdim(s, i):
        if s.target == 'ext' or (s.target == 'down' and i == 0):
            return s.cfg['d']
        return s.dins[i] if s.target == 'self' else s.dins[i - 1]

    def init_state(s, B, dev, mode):
        tr = mode == 'eprop'
        return {'lv': [lv.init_state(B, dev, tr, s.dins[i], s.dts[i]) for i, lv in enumerate(s.levels)],
                'hub': s.hub.init_state(B, dev, tr)}

    def forward(s, x, st, mode='eprop', local=False, rec=False):
        En = rms(s.emb.weight).detach()
        u = rms(s.emb(x))
        ins, feats, loc, stats, new_lv, p, p1 = [], [], 0., [], [], None, None
        for i, lv in enumerate(s.levels):
            ins.append(u)
            if s.target == 'ext' or (s.target == 'down' and i == 0):
                tgt = ('ce', x, En)
            else:
                tgt = ('mse', u if s.target == 'self' else ins[i - 1])
            M, l_i, e_lvl, nst, sts = lv(u, p, st['lv'][i], mode, tgt, rec)
            # 下一级的事件门：本级预测误差相对其慢均值冲高 → p 大
            le = torch.log(e_lvl + 1e-3)
            lmean = ema(le, st['lv'][i]['lmean'], 0.97)
            lprev = torch.cat([st['lv'][i]['lmean'][:, None], lmean[:, :-1]], 1)
            p = torch.sigmoid(s.kappa * (le - lprev) - s.p_off).detach()
            nst['lmean'] = lmean[:, -1].detach()
            if i == 0:
                p1 = p
            sts['p_next'] = p
            loc, stats, new_lv = loc + l_i, stats + [sts], new_lv + [nst]
            f = rms(M)
            feats.append(f.detach() if local else f)
            u = f.detach() if local else f
        o, n_hub = s.hub(torch.cat(feats, -1), p1, st['hub'], mode)   # 中枢：只按新奇写入
        logits = s.ro(torch.cat(feats + [rms(o).flatten(2)], -1))
        return logits, loc / len(s.levels), {'lv': new_lv, 'hub': n_hub}, stats

    def homeostasis(s, stats):
        for lv, st in zip(s.levels, stats):
            lv.homeostasis(st)


# ---------------------------------------------------------------- 数据与评测

DATA = {'tiny': ('data/TinyStoriesV2-GPT4-train.txt', 'data/TinyStoriesV2-GPT4-valid.txt'), 'shakespeare': ('data/input.txt', None)}


def load(name):
    tr, va = DATA[name]
    a = np.memmap(ROOT / tr, dtype=np.uint8, mode='r')
    if va is None:
        n = int(len(a) * 0.9)
        return a[:n], a[n:]
    return a, np.memmap(ROOT / va, dtype=np.uint8, mode='r')


class Streams:
    """B 条连续字节流，每次取下一段 T+1 字节（状态跨段延续）。"""

    def __init__(s, data, B, T, seed=0, offsets=None):
        s.data, s.B, s.T = data, B, T
        s.rng = np.random.default_rng(seed)
        s.pos = np.array(offsets) if offsets is not None else s.rng.integers(0, len(data) - 10 * T, B)

    def next(s):
        out = []
        for i in range(s.B):
            if s.pos[i] + s.T + 1 > len(s.data):
                s.pos[i] = s.rng.integers(0, len(s.data) - 10 * s.T)
            out.append(np.asarray(s.data[s.pos[i]:s.pos[i] + s.T + 1]))
            s.pos[i] += s.T
        xy = torch.from_numpy(np.stack(out).astype(np.int64)).to(DEV)
        return xy[:, :-1], xy[:, 1:]


@torch.no_grad()
def stream_bpc(model, data, B=16, T=128, n=24, seed=123):
    """流式 bpc：B 条流从固定位置起读 n 段，状态延续，不学习。"""
    model.eval()
    rng = np.random.default_rng(seed)
    S = Streams(data, B, T, offsets=rng.integers(0, len(data) - (n + 2) * T, B))
    st, tot = model.init_state(B, DEV, 'eval'), 0.
    for _ in range(n):
        x, y = S.next()
        lg, _, st, _ = model(x, st, 'eval')
        tot += F.cross_entropy(lg.flatten(0, 1), y.flatten()).item()
    model.train()
    return tot / n / LN2


def auc(score, label):
    """排序 AUC（label 为布尔）。"""
    order = np.argsort(score)
    r = np.empty(len(score)); r[order] = np.arange(1, len(score) + 1)
    npos, nneg = label.sum(), (~label).sum()
    return float((r[label].sum() - npos * (npos + 1) / 2) / max(npos * nneg, 1))


def nmi(a, b):
    a, b = np.asarray(a), np.asarray(b)
    ja = np.unique(a, return_inverse=True)[1]; jb = np.unique(b, return_inverse=True)[1]
    P = np.zeros((ja.max() + 1, jb.max() + 1)); np.add.at(P, (ja, jb), 1); P /= P.sum()
    pa, pb = P.sum(1), P.sum(0)
    H = lambda q: -(q[q > 0] * np.log(q[q > 0])).sum()
    I = (P[P > 0] * np.log(P[P > 0] / (pa[:, None] * pb[None])[P > 0])).sum()
    return float(I / max(math.sqrt(H(pa) * H(pb)), 1e-9))


def byte_class(x):
    x = np.asarray(x)
    cls = np.full(x.shape, 5)
    cls[(x >= 97) & (x <= 122)] = 0
    cls[(x >= 65) & (x <= 90)] = 1
    cls[x == 32] = 2
    cls[np.isin(x, list(b'.,!?;:\'"-'))] = 3
    cls[x == 10] = 4
    return cls


@torch.no_grad()
def emergence(model, data, B=8, T=128, n=12, seed=7):
    """第 21.7 节的涌现指标。"""
    model.eval()
    rng = np.random.default_rng(seed)
    S = Streams(data, B, T, offsets=rng.integers(0, len(data) - (n + 2) * T, B))
    st = model.init_state(B, DEV, 'eval')
    xs, las, ps, pis, masks = [], [[] for _ in model.levels], [[] for _ in model.levels], [[] for _ in model.levels], [[] for _ in model.levels]
    for _ in range(n):
        x, _ = S.next()
        _, _, st, stats = model(x, st, 'eval', rec=True)
        xs.append(x.cpu())
        for i, lv in enumerate(model.levels):
            las[i].append(lv.slow.last_la.cpu()); ps[i].append(stats[i]['p_next'].cpu())
            pis[i].append(stats[i]['pi'].cpu()); masks[i].append(stats[i]['mask'].cpu())
    X = torch.cat(xs, 1).numpy()
    out = {}
    # 时间尺度层级：各级慢通道有效时间常数 1/(1−E[A]) 的中位数
    taus = []
    for i in range(len(model.levels)):
        A = torch.cat(las[i], 1).exp().mean((0, 1))
        taus.append(float((1 / (1 - A).clamp(min=1e-6)).median()))
    out['tau_median'] = taus
    out['tau_ratio_top_bottom'] = taus[-1] / taus[0]
    out['hub_tau'] = (1 / model.hub.r.exp()).flatten().tolist()
    out['mem_tau_median'] = [float((1 / lv.mem.r.exp()).median()) for lv in model.levels]
    # 自发切词：进入第 2 级的事件门 p 对“词首字节”的 AUC（0.5 = 随机）
    prevb = np.concatenate([np.full((B, 1), 32), X[:, :-1]], 1)
    word_start = np.isin(prevb, list(b' \n.,!?;:"')) & (byte_class(X) <= 1)
    out['boundary_auc'], out['p_mean'] = [], []
    for i in range(len(model.levels) - 1):
        P = torch.cat(ps[i], 1).numpy()
        out['boundary_auc'].append(auc(P[:, 1:].ravel(), word_start[:, 1:].ravel()))
        out['p_mean'].append(float(P.mean()))
    # 单元与柱分工
    cls = byte_class(X)
    out['col_nmi'], out['col_win_share'], out['dead_frac'], out['unit_sel'] = [], [], [], []
    for i in range(len(model.levels)):
        pi = torch.cat(pis[i], 1)
        win = pi.argmax(-1).numpy()
        out['col_nmi'].append(nmi(win.ravel(), cls.ravel()))
        out['col_win_share'].append(np.bincount(win.ravel(), minlength=pi.shape[-1]).tolist())
        mk = torch.cat(masks[i], 1)                         # (B, T, C, h)
        out['dead_frac'].append(float((mk.sum((0, 1)) == 0).float().mean()))
        # 单元选择性：每个单元激活时字节类别分布的 1 − 归一化熵，取均值
        mf = mk.flatten(0, 1).flatten(1).numpy()            # (N, C·h)
        onehot = np.eye(6)[cls.ravel()]
        cnt = mf.T @ onehot
        q = cnt / np.maximum(cnt.sum(1, keepdims=True), 1)
        Hq = -(q * np.log(np.maximum(q, 1e-12))).sum(1) / math.log(6)
        alive = cnt.sum(1) > 20
        out['unit_sel'].append(float((1 - Hq[alive]).mean()) if alive.any() else 0.)
    # 复制：随机小写串重复两遍，第二遍逐字节 top-1 准确率
    g = torch.Generator().manual_seed(1)
    r = torch.randint(97, 123, (16, 24), generator=g)
    seq = torch.cat([torch.full((16, 8), 32), r, torch.full((16, 1), 32), r, torch.full((16, 1), 32)], 1).to(DEV)
    lg, _, _, _ = model(seq[:, :-1], model.init_state(16, DEV, 'eval'), 'eval')
    pred = lg.argmax(-1)
    s1 = slice(8 + 25, 8 + 25 + 23)
    out['copy_acc'] = float((pred[:, s1] == seq[:, s1.start + 1:s1.stop + 1]).float().mean())
    model.train()
    return out


# ---------------------------------------------------------------- 训练

def reset_rows(st, keep):
    """会话边界：keep 为 0 的流清空全部状态与资格迹（每个状态张量第 0 维是流）。"""
    if isinstance(st, dict):
        return {k: reset_rows(v, keep) for k, v in st.items()}
    if isinstance(st, list):
        return [reset_rows(v, keep) for v in st]
    return st * keep.view(-1, *[1] * (st.dim() - 1)).to(st.dtype)


def wsd(it, iters, lr):
    k = int(iters * 0.8)
    return lr if it < k else lr * max(0.02, (iters - it) / (iters - k))


def train(a):
    torch.manual_seed(a.seed)
    tr, va = load(a.data)
    model = TraceCols(L=a.L, C=a.C, d=a.d, n=a.n, h=a.h, k=a.k, m=a.m, target=a.target).to(DEV)
    nparam = sum(p.numel() for p in model.parameters())
    print(f'params {nparam/1e6:.2f}M  credit={a.credit} target={a.target}', flush=True)
    rand_metrics = emergence(model, va)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.99))
    S = Streams(tr, a.batch, a.T, seed=a.seed)
    mode = 'tbptt' if a.credit == 'tbptt' else 'eprop'
    st = model.init_state(a.batch, DEV, mode)
    log, t0, run_ce = [], time.time(), None
    for it in range(a.iters):
        for gp in opt.param_groups:
            gp['lr'] = wsd(it, a.iters, a.lr)
        x, y = S.next()
        keep = (torch.rand(a.batch, device=DEV) >= a.reset).float()   # 随机会话边界：训练里也见到零状态起步
        st = reset_rows(st, keep)
        logits, loc, st, stats = model(x, st, mode, local=a.credit == 'local')
        ce = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        lam = a.lam_loc if a.credit != 'local' else 1.0
        loss = ce + lam * loc
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.homeostasis(stats)
        run_ce = ce.item() if run_ce is None else 0.98 * run_ce + 0.02 * ce.item()
        if (it + 1) % a.eval_every == 0 or it + 1 == a.iters:
            vb = stream_bpc(model, va)
            el = time.time() - t0
            log.append({'it': it + 1, 'train_bpc': run_ce / LN2, 'val_stream_bpc': vb, 'sec': el})
            print(f'it {it+1} train {run_ce/LN2:.3f} val {vb:.3f} {a.batch*a.T*(it+1)/el:.0f} B/s', flush=True)
    metrics = emergence(model, va)
    vpre = {n: opt.state[p]['exp_avg_sq'].detach().cpu() for n, p in model.named_parameters() if p in opt.state}
    stamp = time.strftime('%Y%m%d-%H%M%S')
    ck = ROOT / 'runs' / f'tc-{a.tag}.pt'
    torch.save({'cfg': model.cfg, 'state': model.state_dict(), 'vpre': vpre}, ck)
    res = {'kind': 'tracecols', 'tag': a.tag, 'args': vars(a), 'params': nparam, 'log': log,
           'final_val_stream_bpc': log[-1]['val_stream_bpc'], 'bytes_per_sec': a.batch * a.T * a.iters / (time.time() - t0),
           'emergence': metrics, 'emergence_random_init': rand_metrics, 'ckpt': str(ck.name)}
    out = ROOT / 'runs' / f'{stamp}_tc-{a.tag}.json'
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(metrics, ensure_ascii=False), flush=True)
    print('saved', out.name)


# ---------------------------------------------------------------- 一生：在线学习 + 睡眠

class Life:
    """在线学习器：三因子更新 Δw = −lr·M·g/(√v_pre+ε)/(1+λω)，外加元可塑级联（Benna-Fusi）。
    g 由资格迹 + 当拍学习信号给出（与预训练同一套 forward / backward）；M = 新奇门；ω = 巩固标签（预训练末的 v_pre 归一化）。"""

    def __init__(s, model, vpre, lr=3e-4, tags=0.0, cascade=False, gate=False, mom=0.9):
        s.m, s.lr, s.tags, s.cascade, s.gate, s.mom = model, lr, tags, cascade, gate, mom
        s.params = dict(model.named_parameters())
        s.v = {n: vpre[n].to(DEV) for n in s.params}
        s.omega = {n: v / v.mean().clamp(min=1e-20) for n, v in s.v.items()}
        s.buf = {n: torch.zeros_like(p) for n, p in s.params.items()}
        s.u2 = {n: p.detach().clone() for n, p in s.params.items()}
        s.u3 = {n: p.detach().clone() for n, p in s.params.items()}
        s.hist, s.hippo = [], []
        s.st = model.init_state(1, DEV, 'eprop')

    def novelty(s, bpc):
        s.hist = (s.hist + [bpc])[-32:]
        if not s.gate or len(s.hist) < 8:
            return 1.0
        return float(bpc > float(np.median(s.hist[:-1])))

    @torch.no_grad()
    def step(s, M):
        for n, p in s.params.items():
            if p.grad is None:
                continue
            s.buf[n].mul_(s.mom).add_(p.grad)
            upd = s.buf[n] / (s.v[n].sqrt() + 1e-8)
            if s.tags:
                upd = upd / (1 + s.tags * s.omega[n])
            p.sub_(s.lr * M * upd)
            if s.cascade:                                  # u1 = 可见权重；u2、u3 越深越难动（容量 1 / 4 / 16）
                g1, g2 = 0.02, 0.004
                f12, f23 = g1 * (p - s.u2[n]), g2 * (s.u2[n] - s.u3[n])
                p.sub_(f12)
                s.u2[n].add_((f12 - f23) / 4)
                s.u3[n].add_(f23 / 16)

    def read(s, x, y):
        logits, loc, s.st, stats = s.m(x, s.st, 'eprop')
        ce = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        M = s.novelty(ce.item() / LN2)
        if M > 0:
            for p in s.params.values():
                p.grad = None
            (ce + 0.05 * loc).backward()
            s.step(M)
            s.hippo.append((ce.item(), torch.cat([x, y[:, -1:]], 1).cpu()))   # 海马：只存新奇片段
            s.hippo = sorted(s.hippo, key=lambda t: -t[0])[:64]
        return ce.item() / LN2, M

    @torch.no_grad()
    def dream(s, morning, n=8, T=128, temp=1.0):
        st = morning.init_state(n, DEV, 'eval')
        x = torch.full((n, 1), 10, device=DEV)
        out = [x]
        for _ in range(T):
            lg, _, st, _ = morning(x, st, 'eval')
            x = torch.multinomial(F.softmax(lg[:, -1] / temp, -1), 1)
            out.append(x)
        return torch.cat(out, 1)

    def sleep(s, morning, steps=60, shrink=0.7):
        with torch.no_grad():                              # 突触稳态：未巩固的白天改动按比例缩小
            for n, p in s.params.items():
                base = s.u2[n] if s.cascade else s.u2_morning[n]
                p.copy_(base + shrink * (p - base))
        dreams = s.dream(morning)
        for i in range(steps):
            for p in s.params.values():
                p.grad = None
            loss = 0.
            if s.hippo:                                    # 海马重放（零状态）
                idx = np.random.default_rng(i).integers(0, len(s.hippo), 4)
                seg = torch.cat([s.hippo[j][1] for j in idx], 0).to(DEV)
                lg, _, _, _ = s.m(seg[:, :-1], s.m.init_state(4, DEV, 'tbptt'), 'tbptt')
                loss = loss + F.cross_entropy(lg.flatten(0, 1), seg[:, 1:].flatten())
            j = i % (len(dreams) // 4)                     # 做梦：保住早晨模型的分布
            dz = dreams[4 * j:4 * j + 4]
            with torch.no_grad():
                tl, _, _, _ = morning(dz[:, :-1], morning.init_state(4, DEV, 'eval'), 'eval')
            sl, _, _, _ = s.m(dz[:, :-1], s.m.init_state(4, DEV, 'tbptt'), 'tbptt')
            loss = loss + F.kl_div(F.log_softmax(sl, -1), F.log_softmax(tl, -1), log_target=True, reduction='batchmean') / dz.shape[1]
            loss.backward()
            s.step(1.0)
        s.st = s.m.init_state(1, DEV, 'eprop')


def load_model(path):
    ck = torch.load(ROOT / 'runs' / path, map_location=DEV)
    m = TraceCols(**ck['cfg']).to(DEV)
    m.load_state_dict(ck['state'])
    return m, ck['vpre']


def life(a):
    _, tiny_va = load('tiny')
    sh_tr, sh_va = load('shakespeare')
    probes = lambda m: {'old_tiny': stream_bpc(m, tiny_va, B=8, n=16), 'new_shakespeare': stream_bpc(m, sh_va, B=8, n=16)}
    res = {'kind': 'tracecols-life', 'ckpt': a.ckpt, 'chunks': a.chunks, 'variants': {}}
    for name, kw in (('plain', dict(tags=0.0, cascade=False, gate=False)), ('full', dict(tags=2.0, cascade=True, gate=True))):
        torch.manual_seed(0)
        m, vpre = load_model(a.ckpt)
        morning = copy.deepcopy(m).eval()
        L = Life(m, vpre, lr=a.lr, **kw)
        L.u2_morning = {n: p.detach().clone() for n, p in m.named_parameters()}
        r = {'before': probes(m)}
        S = Streams(sh_tr, 1, 128, offsets=[0])
        bpcs, learned = [], 0
        for _ in range(a.chunks):
            x, y = S.next()
            b, M = L.read(x, y)
            bpcs.append(b); learned += M
        r['after_day'] = probes(m)
        r['day_bpc_first_last'] = [float(np.mean(bpcs[:20])), float(np.mean(bpcs[-20:]))]
        r['learned_chunks'] = learned
        if name == 'full':
            L.sleep(morning)
            r['after_sleep'] = probes(m)
        res['variants'][name] = r
        print(name, json.dumps(r), flush=True)
    stamp = time.strftime('%Y%m%d-%H%M%S')
    out = ROOT / 'runs' / f'{stamp}_tc-life-{a.tag}.json'
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print('saved', out.name)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train')
    t.add_argument('--credit', default='eprop', choices=['eprop', 'local', 'tbptt'])
    t.add_argument('--target', default='self', choices=['self', 'down', 'ext'])
    t.add_argument('--data', default='tiny')
    t.add_argument('--iters', type=int, default=3000)
    t.add_argument('--batch', type=int, default=32)
    t.add_argument('--T', type=int, default=128)
    t.add_argument('--lr', type=float, default=2e-3)
    t.add_argument('--lam_loc', type=float, default=0.05)
    t.add_argument('--reset', type=float, default=1 / 32, help='每段每条流清空状态的概率')
    t.add_argument('--L', type=int, default=3)
    t.add_argument('--C', type=int, default=4)
    t.add_argument('--d', type=int, default=64)
    t.add_argument('--n', type=int, default=64)
    t.add_argument('--h', type=int, default=256)
    t.add_argument('--k', type=int, default=13)
    t.add_argument('--m', type=int, default=64)
    t.add_argument('--seed', type=int, default=0)
    t.add_argument('--eval_every', type=int, default=500)
    t.add_argument('--tag', default='run')
    l = sub.add_parser('life')
    l.add_argument('ckpt')
    l.add_argument('--chunks', type=int, default=400)
    l.add_argument('--lr', type=float, default=3e-4)
    l.add_argument('--tag', default='life')
    a = ap.parse_args()
    train(a) if a.cmd == 'train' else life(a)
