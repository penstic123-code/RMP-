"""TSPH 双扫预测层级网络（字节级），设计见 docs/design/candidates/10-tsph-cleanroom.md；接口同 dspn_text.DSPNLayered。

与设计的差异：
- A 扫的惊讶度用本层“上一拍预测本拍”（Q_l）的余弦误差，而非 B 扫误差 e：后者依赖上层上一拍的 B 扫状态，
  再反馈进上层更新门就成了非线性时间递归，无法整段并行。快记忆写入强度同理用 A 扫第 2 层惊讶度。
- 层内与层间投影都按模块块对角（模块 i 只接模块 i），跨模块只经工作区 w 与记忆读出 r；参数量靠加宽模块凑。
- 快记忆在衰减之后、写入之前读：r_t = qᵀ(λ_t M_{t−1} + M_long)，使会话边界字节本身已读不到旧会话。
- M_long 只作为状态读出，前向不写（睡眠写入不在本文件）。
"""
import math
import torch, torch.nn as nn, torch.nn.functional as F
from dspn_text import scan, SEP

SIZES = {   # 约 0.35M / 4.5M 参数
    'S': dict(n=4, dm=40, hm=2, emb=32, ws=32, dk=32, dv=64),
    'L': dict(n=8, dm=88, hm=4, emb=64, ws=64, dk=128, dv=256),
}


def bd(n, i, o):
    return nn.Parameter(torch.randn(n, i, o) / math.sqrt(i))


def mix(x, W):   # 块对角：x (b, T, n, i) · W (n, i, o)
    return torch.einsum('btni,nio->btno', x, W)


class Block(nn.Module):
    """minGRU 式门控线性递归 + 模块内 MLP；门与候选只看本层输入，整段并行扫描。"""
    def __init__(self, n, din, dm, hm):
        super().__init__()
        self.dm = dm
        self.W = bd(n, din, 2 * dm)
        tau = torch.logspace(math.log10(1.5), 3, dm)                 # 各层同分布：时间常数 1.5–1000 步
        self.b_z = nn.Parameter(torch.log(1 / (tau - 1)).repeat(n, 1))
        self.ln = nn.LayerNorm(dm)
        self.W1, self.W2 = bd(n, dm, hm * dm), nn.Parameter(torch.zeros(n, hm * dm, dm))

    def forward(self, x, h0, keep, zb=0.0):
        g, c = mix(x, self.W).split(self.dm, -1)
        z = torch.sigmoid(g + self.b_z + zb)
        la = F.logsigmoid(-(g + self.b_z + zb))                            # log(1 − z)，数值稳定
        h, hT = scan(torch.where(keep > 0, la, -30.0), z * c, h0)          # scan 取对数门；keep=0：会话边界清状态
        return h + mix(F.gelu(mix(self.ln(h), self.W1)), self.W2), hT


class TSPH(nn.Module):
    def __init__(self, n=4, dm=32, hm=2, emb=32, ws=32, dk=32, dv=64, vocab=256,
                 aux_pred=0.1, aux_write=0.01, mem_chunk=64):
        super().__init__()
        d = n * dm
        self.n, self.dm, self.ws, self.dk, self.dv = n, dm, ws, dk, dv
        self.aux_pred, self.aux_write, self.mem_chunk = aux_pred, aux_write, mem_chunk
        self.aux = 0.0
        self.emb = nn.Embedding(vocab, emb)
        self.A = nn.ModuleList([Block(n, emb, dm, hm), Block(n, dm, dm, hm), Block(n, dm, dm, hm)])
        self.zs = nn.Parameter(torch.zeros(2, n, dm))                 # 下层惊讶度 → 上层更新门
        self.Q = nn.Parameter(torch.randn(2, n, dm, dm) / math.sqrt(dm))
        self.P = nn.Parameter(torch.randn(2, n, dm, dm) / math.sqrt(dm))   # P[l]: b^{l+2}_{t−1} → p^{l+1}_t
        self.gain = nn.Parameter(torch.zeros(2, n, dm))               # log 精度
        self.B = nn.ModuleList([Block(n, 2 * dm + ws, dm, hm) for _ in range(2)])
        self.sal = nn.Parameter(torch.randn(3 * n, dm) / math.sqrt(dm)); self.b_sal = nn.Parameter(torch.zeros(3 * n))
        self.val = bd(3 * n, dm, ws)
        self.w_in, self.r_in = nn.Linear(ws, d, bias=False), nn.Linear(dv, d, bias=False)
        D = ws + d
        self.b_tr = nn.Parameter(torch.full((D,), 2.0))               # 资格迹衰减 ≈ 0.88，跨几个字节
        self.k, self.v = nn.Linear(D, dk, bias=False), nn.Linear(D, dv, bias=False)
        self.q = nn.Linear(ws, dk, bias=False)
        self.gl = nn.Linear(D, 2)                                     # 写门、衰减 λ 的 logit
        with torch.no_grad():
            self.gl.bias.copy_(torch.tensor([0.0, 5.0]))
        self.P0 = nn.Linear(d, emb, bias=False)
        n_read = d + emb + dv
        self.norm, self.readout = nn.LayerNorm(n_read), nn.Linear(n_read, emb)
        nn.init.zeros_(self.readout.weight)
        self.bias = nn.Parameter(torch.zeros(vocab))

    def init_state(self, b):
        z = lambda *s: torch.zeros(b, *s, device=self.emb.weight.device)
        n, dm = self.n, self.dm
        return dict(h=[z(n, dm) for _ in range(5)],                  # A 扫三层、B 扫第 1、2 层
                    prev=[z(n, dm) for _ in range(4)],               # 上一拍的 a¹、a²、b²、b³
                    trace=z(self.ws + n * dm), M_fast=z(self.dk, self.dv), M_long=z(self.dk, self.dv))

    def forward(self, idx, state=None):
        b, T = idx.shape
        state = self.init_state(b) if state is None else state
        h, prev, n = state['h'], state['prev'], self.n
        reset = idx == SEP
        keep = (~reset).float()[..., None, None]
        shift = lambda x, x0: torch.cat([x0[:, None], x[:, :-1]], 1) * keep
        cos_err = lambda p, a: 1 - F.cosine_similarity(p, a.detach(), -1)   # (b, T, n)，目标停梯度
        # A 扫：自下而上，上层更新门接下层惊讶度
        x, a, hs, surp, zb = self.emb(idx)[:, :, None].expand(b, T, n, -1), [], [], [], 0.0
        for l in range(3):
            y, hT = self.A[l](x, h[l], keep, zb)
            a.append(y); hs.append(hT); x = y
            if l < 2:
                surp.append(cos_err(mix(shift(y, prev[l]), self.Q[l]), y))
                zb = surp[-1].detach()[..., None] * self.zs[l]
        # 工作区：A 扫全部模块按显著性竞争，前 2 名写入 w
        allm = torch.cat(a, 2)
        top = (torch.einsum('btnd,nd->btn', allm, self.sal) + self.b_sal).topk(2, -1)
        vals = mix(allm, self.val).gather(2, top.indices[..., None].expand(-1, -1, -1, self.ws))
        w = (torch.softmax(top.values, -1)[..., None] * vals).sum(2)
        # 快记忆：键取 [w, a²] 的资格迹，写入强度 = 门 × 第 2 层惊讶度
        xm = torch.cat([w, a[1].flatten(2)], -1)
        lam = torch.sigmoid(self.b_tr)
        tr, trT = scan(torch.where(keep[..., 0] > 0, F.logsigmoid(self.b_tr), -30.0), (1 - lam) * xm, state['trace'])
        gw, gl = self.gl(xm).unbind(-1)
        g = torch.sigmoid(gw) * surp[1].mean(-1).detach()
        logl = F.logsigmoid(gl).masked_fill(reset, -30.0)                    # λ≈0：边界处清空 M_fast
        q = F.normalize(self.q(w), dim=-1)
        r, M_fast = self._mem(q, F.normalize(self.k(tr), dim=-1), self.v(xm), g, logl, state['M_fast'])
        r = r + q @ state['M_long']
        # B 扫：自上而下，上层上一拍预测本层，误差按精度加权
        up = a[2] + (self.w_in(w) + self.r_in(r)).view(b, T, n, self.dm)
        ups, hB, pred = [up], [None, None], 0.0
        wb = w[:, :, None].expand(-1, -1, n, -1)
        for l in (1, 0):
            p = mix(shift(up, prev[2 + l]), self.P[l])
            pred = pred + cos_err(p, a[l]).mean()
            up, hB[l] = self.B[l](torch.cat([p, (a[l] - p) * self.gain[l].exp(), wb], -1), h[3 + l], keep)
            ups.append(up)
        b1 = up.flatten(2)
        y = self.readout(self.norm(torch.cat([b1, self.P0(b1), r], -1)))
        self.aux = self.aux_pred * (pred + surp[0].mean() + surp[1].mean()) + self.aux_write * g.mean()
        new = dict(h=hs + hB, prev=[a[0][:, -1], a[1][:, -1], ups[1][:, -1], ups[0][:, -1]],
                   trace=trT, M_fast=M_fast, M_long=state['M_long'])
        return y @ self.emb.weight.t() + self.bias, new

    def _mem(self, q, k, v, g, logl, M):
        """带衰减的因果线性注意力，块内并行、块间传状态；逐拍等价于 M ← λM；r = qᵀM；M ← M + g·k vᵀ。"""
        T = q.shape[1]
        C = min(self.mem_chunk, T)
        lower = torch.ones(C, C, dtype=torch.bool, device=q.device).tril(-1)
        outs = []
        for t in range(0, T, C):
            qc, kc, vc, gc, G = q[:, t:t + C], k[:, t:t + C], v[:, t:t + C], g[:, t:t + C], logl[:, t:t + C].cumsum(1)
            c = G.shape[1]
            D = torch.where(lower[:c, :c], G[:, :, None] - G[:, None, :], float('-inf')).exp()   # Λ_t/Λ_s，s < t
            outs.append((qc * G.exp()[..., None]) @ M + ((qc @ kc.transpose(1, 2)) * D) @ (vc * gc[..., None]))
            M = M * G[:, -1, None, None].exp() + (kc * (gc * (G[:, -1:] - G).exp())[..., None]).transpose(1, 2) @ vc
        return torch.cat(outs, 1), M


def build_tsph(size='S', **kw):
    return TSPH(**{**SIZES[size], **kw})
