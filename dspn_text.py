"""DSPN v2 字节级语言模型：每层 M 个模块（门控对角线性循环 c + 快记忆 S + 局部 MLP），全局记忆中枢，窗口级广播。

设计见 docs/design/README.md（第 20 章）；排期见 docs/plan/fast-clean.md。v0.16 及更早的代码在 git 标签 v0.16 与 legacy/。

用法：
  python dspn_text.py train dspn --size S --data tiny --iters 3000
  python dspn_text.py sample dspn --size S --prompt "Once upon a time"
"""
import argparse, hashlib, json, math, os, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
ROOT = Path(__file__).resolve().parent
DATASETS = {
    'shakespeare': ('data/input.txt', None),
    'tiny': ('data/TinyStoriesV2-GPT4-train.txt', 'data/TinyStoriesV2-GPT4-valid.txt'),
    'chat2': ('data/chat2-train.txt', 'data/chat2-valid.txt'),          # make_chat_data.py --memory_frac 0.25 --out chat2（无重现行）
    'chat3': ('data/chat3-train.txt', 'data/chat3-valid.txt'),          # make_chat_data.py 现版：记忆对话 40%，跨会话一半带重现行
}
SEP = 30   # 会话边界字节（ASCII 记录分隔符）：清空 c、快记忆、中枢与广播

_HUB = dict(hub_after=2, hub_heads=4, hub_taus=[300, 1000, 3000, 30000], beta_max=2, pred_self=0.01)
SIZES = {   # 完整配置都在这里，命令行 --extra 只用来覆盖个别项
    'S': dict(layers=3, M=2, d=90, m=16, hidden=180, q=16, emb=32, hub_dk=32, hub_out=16, **_HUB),     # 约 0.36M，E53 基线
    'L': dict(layers=4, M=2, d=320, m=48, hidden=640, q=32, emb=64, hub_dk=64, hub_out=32, **_HUB),    # 约 470 万，dspn-L_base 主干
}

# 融合内核（flash-linear-attention，triton-windows）：CUDA 上默认使用，DSPN_FLA=0 关闭。逐字节聊天等短序列走本地实现。
try:
    import fla  # noqa: F401
    USE_FLA = os.environ.get('DSPN_FLA', '1') == '1'
except ImportError:
    USE_FLA = False


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


def scan(la, x, c0, chunk=8):
    """对角线性递推 c_t = exp(la_t)·c_{t−1} + x_t（沿时间维 1）。la ≤ 0 是对数门。
    CUDA 上用 fla 的 HGRN 内核；本地实现分块：块内用衰减矩阵 exp(LA_t − LA_s)（s ≤ t，指数 ≤ 0，不会溢出），块间顺序衔接。"""
    b, T = x.shape[:2]
    if USE_FLA and x.is_cuda and T >= 16:
        from fla.ops.hgrn import chunk_hgrn
        c, cT = chunk_hgrn(x.float().flatten(2), la.float().flatten(2), c0.float().flatten(1), output_final_state=True)
        return c.view(x.shape), cT.view(c0.shape)
    pad = -T % chunk
    if pad:                                                # 补齐位 la=0、x=0：状态不变
        la, x = (F.pad(t.flatten(2), (0, 0, 0, pad)).view(b, T + pad, *t.shape[2:]) for t in (la, x))
    n = (T + pad) // chunk
    LA = la.reshape(b, n, chunk, *la.shape[2:]).cumsum(2)
    X = x.reshape(LA.shape)
    D = LA[:, :, :, None] - LA[:, :, None]                 # (b, n, t, s, …)
    tri = torch.ones(chunk, chunk, dtype=torch.bool, device=x.device).tril().view(1, 1, chunk, chunk, *[1] * (x.dim() - 2))
    local = (torch.where(tri, D, float('-inf')).exp() * X[:, :, None]).sum(3)   # 块内从零起步
    outs, c = [], c0
    for i in range(n):
        outs.append(local[:, i] + LA[:, i].exp() * c[:, None])
        c = outs[-1][:, -1]
    return torch.cat(outs, 1)[:, :T], c


def delta_chunk(q, k, v, beta, g, S, C=64):
    """门控 delta 规则：S ← αS；S ← S + β(v − Sk)kᵀ；r = Sq，log α = g。q, k, v: (b, T, H, m)；beta, g: (b, T, H)；S: (b, H, m, m)。"""
    if USE_FLA and q.is_cuda and q.shape[1] >= 64:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        dt = q.dtype
        r, Sf = chunk_gated_delta_rule(q.contiguous(), k.to(dt).contiguous(), v.to(dt).contiguous(), g.float().contiguous(),
                                       beta.to(dt).contiguous(), scale=1.0, initial_state=S.float().transpose(-1, -2).contiguous(),
                                       output_final_state=True)
        return r, Sf.transpose(-1, -2)                     # fla 的状态是本文件 S 的转置
    return _delta_chunk(q.float(), k.float(), v.float(), beta.float(), g.float(), S.float(), C)


def _delta_chunk(q, k, v, beta, g, S, C=64):
    """分块并行的参考实现（Gated DeltaNet 分块算法，Yang 等 2024；结构参照 fla 的朴素参考实现）。
    块内用三角求解一次算完 C 拍的写入，块间顺序传状态。内部按行向量约定 S' = Sᵀ 计算。"""
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
    """一层 M 个模块，参数堆叠成 [M, …]。所有时间步一起算门、投影、扫描与快记忆。
    lora_a / lora_b 是 MLP 第一层的秩 2 在线偏移（离线冻结，在线阶段打开）。"""
    def __init__(self, M, U, d, m, hidden, q, rank=2, beta_max=1.0):
        super().__init__()
        self.M, self.d, self.m, self.beta_max = M, d, m, beta_max
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
        self.rec = None             # e-prop（online.py）打开时是列表：记下 (u, la, c0, c)，c 设为叶子以取得学习信号 ∂L/∂c

    def init_state(self, b, device):
        return torch.zeros(b, self.M, self.d, device=device), torch.zeros(b, self.M, self.m, self.m, device=device)

    def gates(self, u):
        """读 u 的两个投影：慢通道的对数门 la 与写入 x = (1 − a)·(B u)。"""
        ga, gb = torch.einsum('btmu,mud->btmd', u, torch.cat([self.G, self.B], -1)).split(self.d, -1)
        z = ga + self.b_a
        return F.logsigmoid(z), torch.sigmoid(-z) * gb

    def forward(self, u, state, reset=None, keep=None):    # u: (b, T, M, U)；reset: (b, T) 布尔，会话边界字节处清空 c 与 S；keep: (b, T) 0/1，0 的位置（块序列的补齐位）不改状态
        c0, S = state
        la, x = self.gates(u)                                  # 输入比例用原门值：边界处只清旧状态，不放大写入（E42）
        if reset is not None:
            la = la.masked_fill(reset[..., None, None], -30.0)
        if keep is not None:
            la, x = la * keep[..., None, None], x * keep[..., None, None].to(x.dtype)
        c, cT = scan(la, x, c0)
        c = c.to(x.dtype)
        if self.rec is not None:
            c = c.detach().requires_grad_()
            self.rec.append((u.detach(), la.detach(), c0.detach(), c))
        k, v, qv, beta, g = self.proj(c)
        if reset is not None:
            g = g.masked_fill(reset[..., None], -30.0)                         # α≈0：边界处清空快记忆
        if keep is not None:
            beta, g = beta * keep[..., None].to(beta.dtype), g * keep[..., None]
        r, S = delta_chunk(qv, k, v, beta, g, S)
        h, z = self.mlp(c, r.to(c.dtype))
        return h, z, (cT, S)

    def proj(self, c):
        """读 c 的五个投影（合成一次）：快记忆的 k、v、q 与写入强度 β、对数保留 g。"""
        m = self.m
        W = torch.cat([self.K, self.V, self.Qm, self.w_beta[..., None], self.w_gam[..., None]], -1)
        kvq, bl, gl = torch.einsum('btmd,mde->btme', c, W).split([3 * m, 1, 1], -1)
        k, v, qv = kvq.split(m, -1)
        beta = self.beta_max * torch.sigmoid(bl[..., 0] + self.b_beta)      # beta_max=2：允许负特征值
        g = F.logsigmoid(-(gl[..., 0] + self.b_gam).float())                 # log α = log(1 − γ)
        return F.normalize(k, dim=-1), v, qv, beta, g

    def mlp(self, c, r):
        W1 = self.W1 + self.lora_a @ self.lora_b
        hid = F.gelu(torch.einsum('btmi,mih->btmh', torch.cat([c, r], -1), W1) + self.b1)
        h = c + torch.einsum('btmh,mhd->btmd', hid, self.W2)
        return h, torch.einsum('btmd,mdq->btmq', h, self.P)

    def compile_parts(self):
        """融合逐元素小运算（A5）：门、投影、MLP 三段各自 torch.compile；fla 内核照常调用，外面仍套整步 CUDA Graph。"""
        for name in ('gates', 'proj', 'mlp'):
            setattr(self, name, torch.compile(getattr(self, name), dynamic=False))


class PredHead(nn.Module):
    """每个模块一个线性预测头：h (b, T, M, d) → (b, T, M, q)。"""
    def __init__(self, M, d, q):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(M, d, q)); self.b = nn.Parameter(torch.zeros(M, q))

    def forward(self, h):
        return torch.einsum('btmd,mdq->btmq', h, self.W) + self.b


class DSPNLayered(nn.Module):
    """每层 M 个模块；第 i+1 层模块读第 i 层的 [正下方模块的消息, 其余模块消息的平均]。
    hub_after=k 时在第 k、k+1 层之间插入全局记忆中枢（hub.py），读第 k 层全部消息与本拍嵌入，逐拍输出给第 k+1 层。
    广播（bcast>0，第 20 章 / fast-clean A2）：每满 bc_window 字节，中枢末拍读出 RMS 归一化后存进状态，
    下一窗口里所有层的所有模块都读它（零初始投影，每模块 bcast 维）。整窗并行，不再按 64 字节块串行。
    动态分块（chunk=N>0，H-Net 式，第 20.8 节第 3 档）：第 0 层在字节上跑（编码器），相邻字节表示的余弦差决定块边界，
    压缩率损失把平均块长拉向 N；中间各层与中枢只在块序列上跑；块输出按边界概率平滑后放回每个字节，
    最后一层回到字节上跑（解码器）。块序列定长 K = ⌈2T/N⌉，边界排前、其余为冻结的补齐位。"""
    def __init__(self, layers=4, M=2, d=32, m=16, hidden=64, q=8, emb=32, vocab=256,
                 hub_after=0, hub_heads=2, hub_dk=32, beta_max=1.0, hub_out=4, hub_taus=None,
                 bcast=4, bc_window=1024, pred_self=0.0, pred_down=0.0, local=False, pred_detach=False, grad_ckpt=False,
                 chunk=0, chunk_w=0.03, chunk_dr=64):
        super().__init__()
        self.M, self.d, self.m, self.q, self.hub_after, self.chunk, self.chunk_w = M, d, m, q, hub_after, chunk, chunk_w
        if chunk:
            assert layers >= 3 and (not hub_after or 2 <= hub_after <= layers - 1), '分块需要编码层 + 至少一个块层 + 解码层，中枢在块层之后'
        self.emb = nn.Embedding(vocab, emb)
        self.hub = None
        if hub_after:
            from hub import Hub
            self.hub = Hub(M * q + emb, M, out_dim=hub_out, heads=hub_heads, dk=hub_dk, beta_max=beta_max, taus=hub_taus)
        self.bcast = bcast if hub_after else 0
        self.bc_window = bc_window
        if self.bcast:
            self.bc_in = nn.Parameter(torch.zeros(layers, hub_heads * hub_dk, M * bcast))   # 零初始：起步等于无广播
        nmsg = q * (2 if M > 1 else 1)
        U = [emb] + [nmsg + (hub_out if hub_after and i == hub_after else 0) for i in range(1, layers)]
        if chunk:                                          # 解码层另读编码层的消息（残差）
            U[-1] += nmsg
            self.Wr = nn.Parameter(torch.randn(2, M * d, chunk_dr) / math.sqrt(M * d))   # 路由的 q、k 投影
        self.layers = nn.ModuleList([Layer(M, U[i] + self.bcast, d, m, hidden, q, beta_max=beta_max) for i in range(layers)])
        # 预测学习（可插拔，权重为 0 即关）：每个模块用本拍的 h 预测下一拍的消息 z——自己的（pred_self，E51 采纳 0.01）、
        # 正下方下层模块的（pred_down）。目标停梯度，误差按目标均方值归一化。
        # local=True：层间消息（含进中枢的）停梯度，每层只由自己的读出路径与预测损失训练（SOLO 式局部学习）。
        self.pred_w = {k: w for k, w in (('self', pred_self), ('down', pred_down)) if w}
        self.pred = nn.ModuleDict({k: nn.ModuleList([PredHead(M, d, q) for _ in range(layers)]) for k in self.pred_w})
        self.local, self.pred_detach, self.grad_ckpt = local, pred_detach, grad_ckpt
        self.aux, self.stats = 0.0, {}                     # 辅助损失（每次前向重算，train 加进总损失）；分块统计
        n_read = layers * M * d
        self.norm = nn.LayerNorm(n_read)
        self.readout = nn.Linear(n_read, emb)              # 瓶颈后与 E 共享（设计第 1.1 节）
        nn.init.zeros_(self.readout.weight)                # 共享 E 时从均匀分布起步，免得初始对数几率过大
        self.bias = nn.Parameter(torch.zeros(vocab))

    def init_state(self, b):
        dev = self.emb.weight.device
        st = [layer.init_state(b, dev) for layer in self.layers]
        st += [self.hub.init_state(b, dev)] if self.hub else []
        if self.chunk:                                     # (上一字节的路由键, 当前块平滑后的输出, 最近一个块处的中枢读出)
            L, nmsg = len(self.layers), self.layers[-1].B.shape[1] - self.bcast - self.q * (2 if self.M > 1 else 1)
            hd = self.hub.heads * self.hub.dk if self.hub else 1
            st.append((torch.zeros(b, self.Wr.shape[-1], device=dev), torch.zeros(b, (L - 2) * self.M * self.d + self.M * nmsg, device=dev),
                       torch.zeros(b, hd, device=dev)))
        if self.bcast:                                     # (本窗口广播的中枢读出, 已读字节数)；窗口按全局位置对齐，逐拍与整段一致
            st.append((torch.zeros(b, self.hub.heads * self.hub.dk, device=dev), 0))
        return st

    def msg(self, z):
        """下一层的输入：[正下方模块的消息, 其余模块消息的平均]（M=2 时即另一个模块，等价于 v0.16 的 nb_count）。"""
        if self.M == 1:
            return z
        return torch.cat([z, (z.sum(2, keepdim=True) - z) / (self.M - 1)], -1)

    def forward(self, idx, state=None):
        state = state or self.init_state(idx.shape[0])
        self.aux = 0.0
        if not self.bcast:
            lg, new, _ = self._run(idx, state, None)
            return lg, new
        *st, (bc, pos) = state
        outs, t, T, W = [], 0, idx.shape[1], self.bc_window
        while t < T:                                       # 训练时 T 是窗口的整数倍、位置对齐，每窗只跑一次
            n = min(T - t, W - pos % W)
            seg = idx[:, t:t + n]
            lg, st, r_last = self._run(seg, st, bc)
            outs.append(lg); t += n; pos += n
            bc = bc * (~(seg == SEP).any(1))[:, None]      # 本窗含会话边界：旧广播作废
            if pos % W == 0:
                bc = F.rms_norm(r_last, r_last.shape[-1:]).float()   # 先归一化再存：跨窗环路不归一化会逐窗放大（E43）
        return torch.cat(outs, 1), st + [(bc, pos)]

    def _layer(self, i, u, st, reset, keep=None):
        if self.grad_ckpt and self.training:
            return checkpoint(self.layers[i], u, st, reset, keep, use_reentrant=False)
        return self.layers[i](u, st, reset, keep)

    def _stack(self, lo, hi, u, state, reset, e, reads, alive, keep=None):
        """跑第 lo..hi−1 层（含其间的中枢）。返回各层 h、新状态、最后的层间输入 u、中枢逐拍读出。"""
        b, T = u.shape[:2]
        hs, new, r, zdown = [], [], None, None
        for i in range(lo, hi):
            if reads is not None:
                u = torch.cat([u, reads[:, i].expand(b, T, self.M, self.bcast) * alive], -1)
            h, z, ns = self._layer(i, u, state[i], reset, keep)
            hs.append(h.flatten(2)); new.append(ns)
            if self.pred:
                self._predict(h, z, zdown, i)
            zdown = z
            zu = z.detach() if self.local else z
            u = self.msg(zu)
            if self.hub is not None and i + 1 == self.hub_after:
                o, self._hub_new, r = self.hub(torch.cat([zu.reshape(b, T, -1), e.to(zu.dtype)], -1), state[len(self.layers)], reset, keep)
                u = torch.cat([u, o.to(u.dtype)], -1)
        return hs, new, u, r

    def _run(self, idx, state, bc):
        b, T = idx.shape
        e = self.emb(idx)
        L = len(self.layers)
        reset = idx == SEP
        reads = alive = None
        if bc is not None:                                 # 会话边界之后本窗不再读旧广播
            alive = (reset.cumsum(1) == 0).to(e.dtype)[:, :, None, None]
            bcn = bc.detach() if self.local else bc
            reads = torch.einsum('bi,lio->blo', bcn.to(e.dtype), self.bc_in.to(e.dtype)).reshape(b, L, 1, self.M, self.bcast)
        u0 = e[:, :, None, :].expand(b, T, self.M, -1)
        if not self.chunk:
            hs, new, _, r = self._stack(0, L, u0, state, reset, e, reads, alive)
            if self.hub is not None:
                new.append(self._hub_new)
            r_last = None if r is None else r[:, -1]
        else:
            hs, new, r_last = self._run_chunked(idx, e, u0, state, reset, reads, alive)
        y = self.readout(self.norm(torch.cat(hs, -1)))
        return y @ self.emb.weight.t().to(y.dtype) + self.bias, new, r_last

    def _run_chunked(self, idx, e, u0, state, reset, reads, alive):
        b, T = idx.shape
        L, N = len(self.layers), self.chunk
        ck_prev, held_prev, r_prev = state[L + (self.hub is not None)]
        h0, new0, m0, _ = self._stack(0, 1, u0, state, reset, e, reads, alive)          # 编码层（字节）
        # 路由：p_t = (1 − cos(q_t, k_{t−1})) / 2，p ≥ 0.5 即本字节开始一个新块；会话边界强制开新块
        qr, kr = (F.normalize(h0[0].float() @ self.Wr[j], dim=-1) for j in (0, 1))
        kprev = torch.cat([ck_prev[:, None], kr[:, :-1]], 1)
        p = ((1 - (qr * kprev).sum(-1)) / 2).clamp(0, 1)
        p = torch.where(reset, torch.ones_like(p), p)
        bnd = p >= 0.5
        F_, G_ = bnd.float().mean(), p.mean()                                           # H-Net 压缩率损失：平均块长拉向 N
        self.aux = self.aux + self.chunk_w * N / (N - 1) * ((N - 1) * F_ * G_ + (1 - F_) * (1 - G_))
        prevb = torch.cat([idx.new_zeros(b, 1), idx[:, :-1]], 1)
        self.stats = {'rate': F_.detach(), 'word': (bnd & self.wbound(prevb)).sum().detach() / bnd.sum().clamp(min=1),
                      'word_base': self.wbound(prevb).float().mean().detach()}
        # 收成定长 K 的块序列：边界在前、按时间排序；超出 K 的边界并入第 K 块（压缩率损失收敛后极少）
        K = min(T, max(1, math.ceil(2 * T / N)))
        pos = (torch.arange(T, device=idx.device) + T * (~bnd).long()).argsort(-1)[:, :K]
        valid = bnd.gather(1, pos)
        keep = valid.float()
        g = lambda x: x.gather(1, pos.view(b, K, *[1] * (x.dim() - 2)).expand(b, K, *x.shape[2:]))
        hm, newm, um, r = self._stack(1, L - 1, g(m0), state, g(reset), g(e), reads,
                                      None if alive is None else g(alive), keep)                     # 块层
        # 平滑（H-Net）：z̄_j = P_j z_j + (1 − P_j) z̄_{j−1}，再按字节所在的块放回；STE 置信度把梯度送回路由
        P = g(p) * keep
        bundle = torch.cat(hm + [um.flatten(2).to(hm[0].dtype)], -1)
        zbar, held = scan(torch.log1p(-P.clamp(max=1 - 1e-6))[..., None].expand_as(bundle).float(),
                          (P[..., None] * bundle.float()), held_prev)
        table = torch.cat([held_prev[:, None], zbar], 1)
        cum = bnd.long().cumsum(1).clamp(max=K)
        up = table.gather(1, cum[..., None].expand(-1, -1, table.shape[-1]))
        conf = torch.where(bnd, p, 1 - p)
        up = up * (conf + (1 - conf).detach())[..., None].to(up.dtype)
        nh = (L - 2) * self.M * self.d
        hup = up[..., :nh]
        uup = up[..., nh:].reshape(b, T, self.M, -1)
        hd, newd, _, _ = self._stack(L - 1, L, torch.cat([m0, uup.to(m0.dtype)], -1), state, reset, e, reads, alive)   # 解码层（字节）
        r_last = r_prev                                    # 广播取最后一个有效块处的中枢读出；本段无边界时沿用上次的
        if r is not None:
            nv = valid.long().sum(1)
            got = r.gather(1, (nv - 1).clamp(min=0)[:, None, None].expand(-1, 1, r.shape[-1]))[:, 0].float()
            r_last = torch.where((nv > 0)[:, None], got, r_prev)
        new = new0 + newm + newd + ([self._hub_new] if self.hub is not None else []) + [(kr[:, -1], held, r_last)]
        return h0 + [hup.to(hd[0].dtype)] + hd, new, r_last

    @staticmethod
    def wbound(x):
        return (x == 32) | (x == 10) | (x == 46) | (x == 44) | (x == 33) | (x == 63)

    def _predict(self, h, z, zdown, i):
        """本拍 h 预测下一拍的消息（自己的 z、正下方下层的 zdown）；归一化 MSE 加进 self.aux。"""
        targets = {'self': z, 'down': zdown}
        for k, heads in self.pred.items():
            tgt = targets[k]
            if tgt is None or h.shape[1] < 2:
                continue
            tgt = tgt[:, 1:].detach().float()
            hin = h[:, :-1].detach() if self.pred_detach else h[:, :-1]
            # 按目标均方值归一化并设下限：除以方差时消息方差一缩小梯度就放大，曾 NaN（E45）
            err = (heads[i](hin).float() - tgt).square().mean(-1) / tgt.square().mean().clamp(min=1e-2)
            self.aux = self.aux + self.pred_w[k] * err.mean()


def count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build(kind='dspn', size='S', **extra):
    assert kind == 'dspn', '只保留 DSPN（v2：只和自己的上一版比）'
    return DSPNLayered(**{**SIZES[size], **extra})


@torch.no_grad()
def evaluate(model, data, n=20, b=32, T=256):
    g = torch.Generator().manual_seed(0)
    tot = 0.0
    for _ in range(n):
        x, y = batch(data, b, T, g)
        lg, _ = model(x)
        tot += F.cross_entropy(lg.reshape(-1, 256), y.reshape(-1)).item()
    return tot / n / math.log(2)                            # 每字节比特数（bpc）


@torch.no_grad()
def evaluate_stream(model, data, b=8, T=1024, n=8):
    """流式评估：b 条连续验证流带状态读 n 段，只算第 2 段起（预热后）的 bpc——跨批次状态、持久头的作用在这里体现。"""
    pos = np.linspace(0, len(data) - n * T - 2, b).astype(np.int64)
    st, tot = model.init_state(b), 0.0
    for i in range(n):
        x, y = stream_batch(data, pos, T)
        lg, st = model(x, st)
        if i:
            tot += F.cross_entropy(lg.reshape(-1, 256), y.reshape(-1)).item()
    return tot / (n - 1) / math.log(2)


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


def scale_state_(st, keep):
    """各行状态乘 keep（(b,) 的 0/1）：0 的行整行清零，模拟冷启动。"""
    for s in st:
        if torch.is_tensor(s):
            s.mul_(keep.view(-1, *([1] * (s.dim() - 1))))
        elif isinstance(s, (tuple, list)):
            scale_state_(s, keep)


def copy_state_(dst, src):
    """把新状态原地拷进固定地址的旧状态（CUDA Graph 回放时地址不能变）；整数位置不变（T 是工作空间块长的整数倍）。"""
    for d, s in zip(dst, src):
        if torch.is_tensor(d):
            d.copy_(s.detach())
        elif isinstance(d, (tuple, list)):
            copy_state_(d, s)


def train(kind, size, data, iters, b, T, lr, seed, tag, extra, init=None, stateful=True, bot_loss=False, sched='wsd', bf16=True, compile=False):
    torch.manual_seed(seed)
    tr, va = load_bytes(data)
    model = build(kind, size, **extra).to(DEV)
    if init:                                               # 从已有权重继续训练（如对话数据上微调）
        model.load_state_dict(torch.load(ROOT / init, map_location=DEV))
    nparam = count(model)
    if compile:                                            # triton 读缓存目录时按 UTF-8 解码环境变量，中文用户名路径会失败：指到纯 ASCII 目录
        for k, sub in (('TRITON_CACHE_DIR', 'triton'), ('TORCHINDUCTOR_CACHE_DIR', 'inductor')):
            os.environ.setdefault(k, f"{os.environ.get('SystemDrive', 'C:')}/dspn_cache/{sub}")
        for layer in model.layers:
            layer.compile_parts()
    params = [p for p in model.parameters() if p.requires_grad]
    cuda = DEV == 'cuda'
    lr_t = torch.tensor(lr, device=DEV)                    # 张量学习率：CUDA Graph 回放时原地改
    opt = torch.optim.AdamW(param_groups(model), lr=lr_t, capturable=cuda)
    if sched == 'wsd':                                     # 恒定 + 最后 20% 线性降到 0：中途停下也能随时补一段衰减
        lrf = lambda it: min(1, it / 200) * min(1.0, (iters - it) / (0.2 * iters))
    else:                                                  # 预热 200 步后余弦衰减
        lrf = lambda it: min(1, it / 200) * 0.5 * (1 + math.cos(math.pi * it / iters))
    name = f'{kind}-{size}{tag}'
    print(f'{name}: {nparam:,} 参数，{DEV}，批 {b} × {T} 字节，{iters} 步', flush=True)
    if stateful and getattr(model, 'bcast', 0):
        assert T % model.bc_window == 0, '跨批次延续状态时 T 须是广播窗口的整数倍（CUDA Graph 里窗口位置不更新）'
    sx, sy = batch(tr, b, T)                               # 固定地址的输入，每步原地拷入新批
    # stateful：每行是一条连续流，状态跨批次延续（截断反传），模型才练得到“记住 T 字节以前的事”（持久头、跨会话记忆）
    pos = np.random.default_rng(seed).integers(0, len(tr) - T - 1, b) if stateful else None
    st = model.init_state(b) if stateful else None
    smask = torch.ones(b, T, device=DEV)                   # bot_loss：只在 Bot 轮次的字节上算损失
    skeep = torch.ones(b, device=DEV)                      # stateful：每步约 2% 的行整行清零，免得模型只会“热”状态（E45）

    def step():
        if stateful:
            scale_state_(st, skeep)
        with torch.autocast('cuda', torch.bfloat16, enabled=bf16 and cuda):
            lg, new = model(sx, st)
        lg = lg.float()
        ce = F.cross_entropy(lg.reshape(-1, 256), sy.reshape(-1), reduction='none')
        loss = (ce * smask.reshape(-1)).sum() / smask.sum().clamp(min=1)
        (loss + getattr(model, 'aux', 0.0)).backward()      # 预测学习等辅助损失
        nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad(set_to_none=False)
        if stateful:
            copy_state_(st, new)
        return loss.detach()                               # 不留计算图引用，否则会干扰 CUDA Graph 录制

    log, t0, eval_secs, graph = [], time.time(), 0.0, None
    for it in range(1, iters + 1):
        lr_t.fill_(lr * lrf(it))
        x, y = stream_batch(tr, pos, T) if stateful else batch(tr, b, T)
        sx.copy_(x); sy.copy_(y)
        if bot_loss:
            smask.copy_(bot_mask(y))
        if stateful:
            skeep.copy_((torch.rand(b, device=DEV) > 0.02).float())
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
            if getattr(model, 'chunk', 0):                 # 自发切词：边界比例、边界落在空格 / 标点之后的比例（及其基准率）
                log[-1] += tuple(round(float(model.stats[k]), 3) for k in ('rate', 'word', 'word_base'))
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
               'device': DEV, 'batch': b, 'T': T, 'iters': iters, 'lr': lr, 'seed': seed, 'stateful': stateful, 'bot_loss': bot_loss, 'sched': sched, 'bf16': bf16, 'compile': compile, 'init': init,
               'code_sha': hashlib.sha256((ROOT / 'dspn_text.py').read_bytes()).hexdigest()[:12],
               'seconds': round(secs), 'train_bytes_per_sec': round(iters * b * T / (secs - eval_secs)),
               'peak_gpu_mb': round(torch.cuda.max_memory_allocated() / 2**20) if DEV == 'cuda' else None,
               'val_bpc': round(final, 4),
               'val_bpc_stream': round(evaluate_stream(model, va), 4), 'log': log}, open(out, 'w'), ensure_ascii=False, indent=1)
    print(f'{name} 完成：验证 bpc {final:.4f} → {out.name}', flush=True)


@torch.no_grad()
def sample(kind, size, tag, prompt, extra, n=600, temp=0.8, seed=1):
    torch.manual_seed(seed)
    model = build(kind, size, **extra).to(DEV)
    model.load_state_dict(torch.load(ROOT / f'runs/{kind}-{size}{tag}.pt', map_location=DEV)); model.eval()
    idx = torch.tensor([list(prompt.encode())], device=DEV)
    out = list(prompt.encode())
    lg, state = model(idx)
    for _ in range(n):
        nxt = torch.multinomial(F.softmax(lg[0, -1] / temp, -1), 1)
        out.append(nxt.item())
        lg, state = model(nxt.view(1, 1), state)
    return bytes(out).decode('utf-8', errors='replace')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['train', 'sample']); ap.add_argument('kind', choices=['dspn'])
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
    ap.add_argument('--stateful', action=argparse.BooleanOptionalAction, default=True, help='每行一条连续流，状态跨批次延续（默认开）')
    ap.add_argument('--bot_loss', action='store_true', help='对话数据只在 Bot 轮次的字节上算损失')
    ap.add_argument('--bf16', action=argparse.BooleanOptionalAction, default=True, help='bf16 自动混合精度（门、状态、递推 fp32）')
    ap.add_argument('--compile', action='store_true', help='torch.compile 融合门、投影、MLP 的逐元素运算')
    ap.add_argument('--sched', default='wsd', choices=['cos', 'wsd'], help='学习率调度：余弦 / 恒定 + 末段线性衰减')
    a = ap.parse_args()
    extra = json.loads(a.extra)
    if a.cmd == 'train':
        lr = a.lr or 3e-3
        train(a.kind, a.size, a.data, a.iters, a.batch, a.T, lr, a.seed, a.tag, extra, a.init, a.stateful, a.bot_loss, a.sched, a.bf16, a.compile)
    else:
        print(sample(a.kind, a.size, a.tag, a.prompt, extra))
