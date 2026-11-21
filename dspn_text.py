"""DSPN v2 字节级语言模型：每层 M 个模块（门控对角线性循环 c + 快记忆 S + 局部 MLP），全局记忆中枢，动态分块（编码层惊讶跃升定边界，v0.18）。

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
    'mix': ('data/mix-train.txt', 'data/mix-valid.txt'),                # tools/prepare_data.py：TinyStories + Cosmopedia + Simple Wikipedia + SODA
    'task4': ('data/task4-train.txt', 'data/task4-valid.txt'),          # tools/make_task_mix.py：四源 4KB 块交错，逐字节来源标签在 .src（RMP 按任务分工）
}
SEP = 30   # 会话边界字节（ASCII 记录分隔符）：清空 c、快记忆与中枢，强制开块

_HUB = dict(hub_after=2, hub_heads=4, hub_taus=[300, 1000, 3000, 30000], beta_max=2)
SIZES = {   # 完整配置都在这里，命令行 --extra 只用来覆盖个别项
    'S': dict(layers=3, M=2, d=90, m=16, hidden=180, q=16, emb=32, hub_dk=32, hub_out=16, **_HUB),     # 约 0.36M，E53 基线
    'L': dict(layers=4, M=2, d=320, m=48, hidden=640, q=32, emb=64, hub_dk=64, hub_out=32, **_HUB),    # 约 470 万，dspn-L_base 主干
    'C': dict(layers=8, M=2, d=384, m=48, hidden=768, q=32, emb=64, hub_dk=64, hub_out=32, **{**_HUB, 'hub_after': 3},
              chunk=5, d_io=192, hidden_io=384, attn_layer=4),   # 动态分块：1 编码层 + 6 块层 + 1 解码层，两头窄、中间深；第 4 层加滑窗注意力
    'XL': dict(layers=8, M=2, d=704, m=64, hidden=1408, q=32, emb=64, hub_dk=64, hub_out=32, **{**_HUB, 'hub_after': 3},
               chunk=5, d_io=256, hidden_io=512, attn_layer=4),  # 约 3000 万：C 的加宽版
    'XL5': dict(layers=8, M=2, d=896, m=64, hidden=1792, q=48, emb=64, beta_max=2, fm_tau=205, k_mult=1.5,
                chunk=5, d_io=256, hidden_io=512, attn_layer=4),  # 约 4800 万：两天预训练按本机算力最划算的规模；无中枢（E59、E60），K = 1.5T/N（E60 每行块率）
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
        u = F.rms_norm(u, (u.shape[-1],))                    # pre-norm：8 层叠起来时，不归一化的层间输入会逐层放大发散（XL 第 81 步）
        la, x = self.gates(u)                                  # 输入比例用原门值：边界处只清旧状态，不放大写入（E42）
        km = None if keep is None else (keep[..., None] if keep.dim() == 2 else keep)   # (b, T, M)：按模块的 keep（路由池里未选的模块为 0）
        if km is not None:
            la, x = la * km[..., None], x * km[..., None].to(x.dtype)
        rm = None if reset is None else (reset[..., None] if reset.dim() == 2 else reset)   # (b, T, 1) 或按模块 (b, T, M)
        if rm is not None:                                     # 先 keep 后 reset：会话边界连未选模块也清空
            la = la.masked_fill(rm[..., None], -30.0)
        c, cT = scan(la, x, c0)
        c = c.to(x.dtype)
        if self.rec is not None:
            c = c.detach().requires_grad_()
            self.rec.append((u.detach(), la.detach(), c0.detach(), c))
        k, v, qv, beta, g = self.proj(c)
        if km is not None:
            beta, g = beta * km.to(beta.dtype), g * km
        if rm is not None:
            g = g.masked_fill(rm, -30.0)                                       # α≈0：边界处清空快记忆
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
        return F.normalize(k, dim=-1), v, F.normalize(qv, dim=-1), beta, g   # q 也归一化（同 Gated DeltaNet）：读出 r = S·q 受 S 限住，不归一化时 XL5 编码层 h0 曾爆到 4.6e5（E61）

    def mlp(self, c, r):
        W1 = self.W1 + self.lora_a @ self.lora_b
        hid = F.gelu(torch.einsum('btmi,mih->btmh', torch.cat([c, r], -1), W1) + self.b1)
        h = c + torch.einsum('btmh,mhd->btmd', hid, self.W2)
        return h, torch.einsum('btmd,mdq->btmq', h, self.P)

    def forward_slice(self, u, state, sl, reset=None, keep=None):
        """只计算模块切片 sl（如固定分波时仅算本波 N/G 个模块）：与全量计算再掩码数学等价，省 G 倍显存与算力，且保持 state_dict 键不变。"""
        assert self.rec is None, 'forward_slice 暂不支持 e-prop 记录'
        c0, S = state
        u = F.rms_norm(u, (u.shape[-1],))
        ga, gb = torch.einsum('btmu,mud->btmd', u, torch.cat([self.G[sl], self.B[sl]], -1)).split(self.d, -1)
        z = ga + self.b_a[sl]
        la, x = F.logsigmoid(z), torch.sigmoid(-z) * gb
        km = None if keep is None else (keep[..., None] if keep.dim() == 2 else keep)
        if km is not None:
            la, x = la * km[..., None], x * km[..., None].to(x.dtype)
        rm = None if reset is None else (reset[..., None] if reset.dim() == 2 else reset)
        if rm is not None:
            la = la.masked_fill(rm[..., None], -30.0)
        c, cT = scan(la, x, c0)
        c = c.to(x.dtype)
        m = self.m
        W = torch.cat([self.K[sl], self.V[sl], self.Qm[sl], self.w_beta[sl][..., None], self.w_gam[sl][..., None]], -1)
        kvq, bl, gl = torch.einsum('btmd,mde->btme', c, W).split([3 * m, 1, 1], -1)
        k, v, qv = kvq.split(m, -1)
        beta = self.beta_max * torch.sigmoid(bl[..., 0] + self.b_beta[sl])
        g = F.logsigmoid(-(gl[..., 0] + self.b_gam[sl]).float())
        k, qv = F.normalize(k, dim=-1), F.normalize(qv, dim=-1)
        if km is not None:
            beta, g = beta * km.to(beta.dtype), g * km
        if rm is not None:
            g = g.masked_fill(rm, -30.0)
        r, S = delta_chunk(qv, k, v, beta, g, S)
        W1 = self.W1[sl] + self.lora_a[sl] @ self.lora_b[sl]
        hid = F.gelu(torch.einsum('btmi,mih->btmh', torch.cat([c, r.to(c.dtype)], -1), W1) + self.b1[sl])
        h = c + torch.einsum('btmh,mhd->btmd', hid, self.W2[sl])
        return h, torch.einsum('btmd,mdq->btmq', h, self.P[sl]), (cT, S)

    def compile_parts(self):
        """融合逐元素小运算（A5）：门、投影、MLP 三段各自 torch.compile；fla 内核照常调用，外面仍套整步 CUDA Graph。"""
        torch._dynamo.config.recompile_limit = 64      # 各层宽度、块序列长度不同，每种形状编译一次；默认上限 8 不够
        for name in ('gates', 'proj', 'mlp'):
            setattr(self, name, torch.compile(getattr(self, name), dynamic=False))


class WindowAttn(nn.Module):
    """滑窗因果注意力（第 20.8 节第 2 档）：只管“最近 window 个位置内原样照抄”（名字等），更久的交给中枢、跨会话交给海马。
    读下面一层的 h，输出每模块 out 维作为本层额外输入；输出投影零初始（起步等于没有注意力）。不加位置编码（NoPE，循环状态自带位置）。
    最近 window 个有效位置的键值存在状态里：跨批次训练、逐拍聊天与整段一致，内存固定。keep=0 的位置（块序列补齐位）不当键、不进缓存。
    会话边界之后不再看边界之前的位置。"""
    def __init__(self, n_in, M, out=64, heads=4, hd=64, window=512):
        super().__init__()
        self.M, self.out, self.heads, self.hd, self.window = M, out, heads, hd, window
        self.norm = nn.LayerNorm(n_in)
        self.qkv = nn.Linear(n_in, 3 * heads * hd, bias=False)
        self.o = nn.Linear(heads * hd, M * out, bias=False)
        nn.init.zeros_(self.o.weight)

    def init_state(self, b, device):
        z = lambda: torch.zeros(b, self.heads, self.window, self.hd, device=device)
        return z(), z(), torch.zeros(b, self.window, dtype=torch.bool, device=device)

    def forward(self, x, state, reset, keep=None):
        b, T, _ = x.shape
        H, hd, W = self.heads, self.hd, self.window
        ck, cv, cval = state
        q, k, v = self.qkv(self.norm(x)).view(b, T, 3, H, hd).permute(2, 0, 3, 1, 4)        # (b, H, T, hd)
        valid = torch.ones(b, T, dtype=torch.bool, device=x.device) if keep is None else keep.bool()
        sess = reset.long().cumsum(1)                                                        # 本段内第几个会话（边界字节属于新会话）
        idx = valid.long().cumsum(1) - 1                                                     # 新位置在有效序列里的序号
        # 掩码：缓存槽 s 的序号是 s − W；只在本段尚未遇到会话边界时可见。新键：同会话、有效、在窗口内且不晚于查询
        dq = idx[:, :, None]
        cache_ok = cval[:, None, :] & (sess[:, :, None] == 0) & ((dq - (torch.arange(W, device=x.device) - W)) < W)
        new_ok = (valid[:, None, :] & (sess[:, :, None] == sess[:, None, :])
                  & (idx[:, None, :] <= dq) & (dq - idx[:, None, :] < W)
                  & torch.ones(T, T, dtype=torch.bool, device=x.device).tril())
        new_ok = new_ok | torch.eye(T, dtype=torch.bool, device=x.device)                    # 补齐位只看自己，免得整行全被遮住
        mask = torch.cat([cache_ok, new_ok], -1)[:, None]                                    # (b, 1, T, W+T)
        K = torch.cat([ck.to(k.dtype), k], 2); V = torch.cat([cv.to(v.dtype), v], 2)
        a = F.scaled_dot_product_attention(q, K, V, attn_mask=mask)
        y = self.o(a.transpose(1, 2).reshape(b, T, H * hd)).view(b, T, self.M, self.out)
        # 新缓存：缓存与新键里最后 W 个有效位置（遇到会话边界时只留最后一个会话的）
        nv = valid.long().sum(1, keepdim=True)
        last = sess[:, -1:]
        allv = torch.cat([cval & (last == 0), valid & (sess == last)], 1)
        alli = torch.cat([torch.arange(W, device=x.device).expand(b, W) - W, idx], 1)         # 序号
        slot = torch.where(allv, alli - nv + W, torch.full_like(alli, W))
        slot = torch.where(slot < 0, torch.full_like(slot, W), slot)                          # 窗口外丢进垃圾槽 W
        nk = torch.zeros(b, H, W + 1, hd, device=x.device, dtype=torch.float32)
        nvv = torch.zeros_like(nk); nval = torch.zeros(b, W + 1, dtype=torch.bool, device=x.device)
        si = slot[:, None, :, None].expand(b, H, W + T, hd)
        nk.scatter_(2, si, K.float()); nvv.scatter_(2, si, V.float()); nval.scatter_(1, slot, allv)
        return y, (nk[:, :, :W], nvv[:, :, :W], nval[:, :W])


class DSPNLayered(nn.Module):
    """每层 M 个模块；第 i+1 层模块读第 i 层的 [正下方模块的消息, 其余模块消息的平均]。
    hub_after=k 时在第 k、k+1 层之间插入全局记忆中枢（hub.py），读第 k 层全部消息与本拍嵌入，逐拍输出给第 k+1 层。
    动态分块（chunk=N>0，第 22 章 v0.18）：第 0 层在字节上跑（编码器），编码层的局部头预测下一字节，
    惊讶跃升 s_t − s_{t−1} > θ 即从 t 开新块（θ 是 buffer，训练时把块率追向 1/N，推理冻结），块长到 L_max = lmax·N 强制开块；
    中间各层与中枢只在块序列上跑；每个字节取它所在块的输出；最后一层回到字节上跑（解码器）。
    块序列定长 K = ⌈k_mult·T/N⌉，边界排前、其余为冻结的补齐位；超出 K 的边界并入第 K 块（记为溢出）。"""
    def __init__(self, layers=4, M=2, d=32, m=16, hidden=64, q=8, emb=32, vocab=256,
                 hub_after=0, hub_heads=2, hub_dk=32, beta_max=1.0, hub_out=4, hub_taus=None,
                 local=False, grad_ckpt=False, chunk=0, head_w=0.1, head_detach=True, lmax=3, theta_lr=0.1, d_io=None, hidden_io=None, fm_tau=0, td_w=0.0, k_mult=2.0,
                 attn_layer=0, attn_window=512, attn_heads=4, attn_hd=64, attn_out=64,
                 pool=0, pool_k=4, pool_d=64, pool_m=16, pool_hidden=128, pool_q=32, pool_qr=32, pool_mode='state', pool_temp=0.1):
        super().__init__()
        self.pool = pool
        self.M, self.d, self.m, self.q, self.hub_after, self.chunk = M, d, m, q, hub_after, chunk
        self.head_w, self.head_detach, self.lmax, self.theta_lr = head_w, head_detach, lmax * chunk, theta_lr
        # 分块时编码层与解码层（字节分辨率）可以更窄：d_io、hidden_io；参数主要放在块层（H-Net：两头小、中间大）
        io = chunk and d_io
        self.dl = [d_io if io and i in (0, layers - 1) else d for i in range(layers)]
        hl = [(hidden_io or hidden) if io and i in (0, layers - 1) else hidden for i in range(layers)]
        if chunk:
            assert layers >= 3 and (not hub_after or 2 <= hub_after <= layers - 1), '分块需要编码层 + 至少一个块层 + 解码层，中枢在块层之后'
        self.emb = nn.Embedding(vocab, emb)
        self.hub = None
        if hub_after:
            from hub import Hub
            self.hub = Hub(M * q + emb, M, out_dim=hub_out, heads=hub_heads, dk=hub_dk, beta_max=beta_max, taus=hub_taus)
        nmsg = q * (2 if M > 1 else 1)
        U = [emb] + [nmsg + (hub_out if hub_after and i == hub_after else 0) for i in range(1, layers)]
        self.attn_layer, self.attn = attn_layer, None
        if attn_layer:                                     # 第 attn_layer 层（从 0 数）另读滑窗注意力的输出；注意力读下面一层的 h
            assert 1 <= attn_layer and (not chunk or attn_layer <= layers - 2), '分块时注意力放在块层'
            U[attn_layer] += attn_out
            self.attn = WindowAttn(M * self.dl[attn_layer - 1], M, attn_out, attn_heads, attn_hd, attn_window)
        if chunk:                                          # 解码层另读编码层的消息（残差）；编码层局部头预测下一字节
            U[-1] += nmsg
            self.head = nn.Linear(M * self.dl[0], vocab)
            self.register_buffer('theta', torch.zeros(()))   # 惊讶跃升阈值：不是状态，冷启动、清零状态都不动它
        self.td_w, self.k_mult = td_w, k_mult           # 块序列定长 K = ⌈k_mult·T/N⌉：块率被 θ 拉在 1/N，2 倍时约一半是补齐位
        if td_w:                                           # 线性自上而下预测（第 22.8 节）：顶块层在块 j 的 h → 块 j+1 处编码层的消息
            assert chunk, '自上而下预测只在分块模型的块层上'
            self.td = nn.Linear(M * d, M * nmsg)
        self.layers = nn.ModuleList([Layer(M, U[i], self.dl[i], m, hl[i], q, beta_max=beta_max) for i in range(layers)])
        if pool:                                           # 路由模块池（第 23 章 R1 / R0）：第 0 层稠密（读字节），其余各层换成池
            assert not chunk and not hub_after and not attn_layer and not fm_tau, '路由池原型只支持不分块、无中枢、无注意力（S 级）'
            from pool import PoolLayer
            Up = [M * q] + [pool_q] * (layers - 2)          # 池读下层的消息：第 0 层各模块 z 拼接；池与池之间是加权混合的 z
            for i in range(1, layers):
                self.layers[i] = PoolLayer(pool, pool_k, Up[i - 1], pool_d, pool_m, pool_hidden, pool_q, pool_qr, pool_mode, beta_max, pool_temp)
        if fm_tau:                                         # 快记忆保持时长初值在 64 到 fm_tau 步间对数铺开（块层各模块；第 22.4 节 D 组，删中枢时由它接管长程）
            mid = self.layers[1:-1] if chunk else self.layers
            tau = torch.logspace(math.log10(64), math.log10(fm_tau), len(mid) * M)
            with torch.no_grad():
                for j, l in enumerate(mid):
                    l.b_gam.copy_(torch.log(1 / (tau[j * M:(j + 1) * M] - 1)))
        # local=True：层间消息（含进中枢的）停梯度，每层只由自己的读出路径训练（SOLO 式局部学习）。
        self.local, self.grad_ckpt = local, grad_ckpt
        self.aux, self.stats = 0.0, {}                     # 辅助损失（局部头，每次前向重算，train 加进总损失）；分块统计
        n_read = M * self.dl[0] + (layers - 1) * pool_d if pool else M * sum(self.dl)
        self.norm = nn.LayerNorm(n_read)
        self.readout = nn.Linear(n_read, emb)              # 瓶颈后与 E 共享（设计第 1.1 节）
        nn.init.zeros_(self.readout.weight)                # 共享 E 时从均匀分布起步，免得初始对数几率过大
        self.bias = nn.Parameter(torch.zeros(vocab))

    def init_state(self, b):
        dev = self.emb.weight.device
        st = [layer.init_state(b, dev) for layer in self.layers]
        st += [self.hub.init_state(b, dev)] if self.hub else []
        if self.chunk:                                     # (局部头上一拍的对数概率, 上一字节的惊讶, 当前块已有字节数, 当前块的输出)
            nmsg = self.q * (2 if self.M > 1 else 1)
            st.append((torch.zeros(b, self.head.out_features, device=dev), torch.zeros(b, device=dev), torch.zeros(b, device=dev),
                       torch.zeros(b, self.M * sum(self.dl[1:-1]) + self.M * nmsg, device=dev)))
        if self.attn is not None:                          # 滑窗注意力的键值缓存
            st.append(self.attn.init_state(b, dev))
        return st

    def msg(self, z):
        """下一层的输入：[正下方模块的消息, 其余模块消息的平均]（M=2 时即另一个模块，等价于 v0.16 的 nb_count）。"""
        if self.M == 1:
            return z
        return torch.cat([z, (z.sum(2, keepdim=True) - z) / (self.M - 1)], -1)

    def forward(self, idx, state=None):
        state = state or self.init_state(idx.shape[0])
        self.aux = 0.0
        b, T = idx.shape
        e = self.emb(idx)
        L = len(self.layers)
        reset = idx == SEP
        u0 = e[:, :, None, :].expand(b, T, self.M, -1)
        if not self.chunk:
            hs, new, _ = self._stack(0, L, u0, state, reset, e)
            new += [self._hub_new] if self.hub is not None else []
            new += [self._attn_new] if self.attn is not None else []
        else:
            hs, new = self._run_chunked(idx, e, u0, state, reset)
        y = self.readout(self.norm(torch.cat(hs, -1)))
        return y @ self.emb.weight.t().to(y.dtype) + self.bias, new

    def _layer(self, i, u, st, reset, keep=None):
        if self.grad_ckpt and self.training:
            return checkpoint(self.layers[i], u, st, reset, keep, use_reentrant=False)
        return self.layers[i](u, st, reset, keep)

    def _stack(self, lo, hi, u, state, reset, e, keep=None):
        """跑第 lo..hi−1 层（含其间的中枢）。返回各层 h、新状态、最后的层间输入 u。"""
        b, T = u.shape[:2]
        hs, new, hprev = [], [], None
        for i in range(lo, hi):
            if self.attn is not None and i == self.attn_layer:
                ai = len(self.layers) + (self.hub is not None) + bool(self.chunk)
                a, self._attn_new = self.attn(hprev, state[ai], reset, keep)
                u = torch.cat([u, a.to(u.dtype)], -1)
            if self.pool and i > 0:                        # 池读下层的消息（每拍一个向量），不经 msg 的邻居平均
                u = zu.flatten(2)
            h, z, ns = self._layer(i, u, state[i], reset, keep)
            hs.append(h.flatten(2)); new.append(ns); hprev = hs[-1]
            zu = z.detach() if self.local else z
            u = self.msg(zu)
            if self.hub is not None and i + 1 == self.hub_after:
                o, self._hub_new, _ = self.hub(torch.cat([zu.reshape(b, T, -1), e.to(zu.dtype)], -1), state[len(self.layers)], reset, keep)
                u = torch.cat([u, o.to(u.dtype)], -1)
        return hs, new, u

    def _boundaries(self, idx, lp, lp_prev, s_prev, n_prev, reset):
        """惊讶跃升开块 + 最大块长强制开块（第 22.2 节）。返回边界、本段末的 (s, 块长)。全部不需要梯度。"""
        b, T = idx.shape
        lpp = torch.cat([lp_prev[:, None], lp[:, :-1]], 1)                     # 预测第 t 个字节的那一拍的对数概率
        s = -lpp.gather(-1, idx[..., None])[..., 0]                            # 惊讶（nats）
        jump = s - torch.cat([s_prev[:, None], s[:, :-1]], 1)
        b0 = (jump > self.theta) | reset
        # 块长：离上一个 b0 边界（本段之前的块从 −n_prev 起算）的距离每到 L_max 的整数倍就强制开块；n_prev = 0（流开头、清零）即开块
        t = torch.arange(T, device=idx.device)
        last = torch.where(b0, t, torch.full_like(t, -10 ** 9)).cummax(1).values
        dist = t - torch.maximum(last, -n_prev.long()[:, None])
        bnd = b0 | (dist % self.lmax == 0)
        n_new = (dist[:, -1] % self.lmax + 1).to(n_prev.dtype)
        return bnd, s[:, -1].to(s_prev.dtype), n_new

    def _run_chunked(self, idx, e, u0, state, reset):
        b, T = idx.shape
        L, N = len(self.layers), self.chunk
        lp_prev, s_prev, n_prev, held_prev = state[L + (self.hub is not None)]
        h0, new0, m0 = self._stack(0, 1, u0, state, reset, e)                   # 编码层（字节）
        lp = F.log_softmax(self.head(h0[0].detach() if self.head_detach else h0[0]).float(), -1)   # 默认读停梯度的 h0：辅助损失不改主干
        if T > 1:
            self.aux = self.aux + self.head_w * F.nll_loss(lp[:, :-1].reshape(-1, lp.shape[-1]), idx[:, 1:].reshape(-1))
        learn = self.training and torch.is_grad_enabled()
        with torch.no_grad():
            bnd, s_last, n_new = self._boundaries(idx, lp.detach(), lp_prev, s_prev, n_prev, reset)
            rate = bnd.float().mean()
            if learn:                                      # 分位数追踪：块率高于 1/N 就抬高 θ（原地改 buffer，CUDA Graph 下可用）
                self.theta.add_(self.theta_lr * (rate - 1 / N)).clamp_(0.0, 10.0)   # 上限：出故障时不跟着失控（E61 曾追到 112）
        K = min(T, max(1, math.ceil(self.k_mult * T / N)))
        prevb = torch.cat([idx.new_zeros(b, 1), idx[:, :-1]], 1)
        self.stats = {'rate': rate, 'word': (bnd & self.wbound(prevb)).sum() / bnd.sum().clamp(min=1),
                      'word_base': self.wbound(prevb).float().mean(), 'over': (bnd.sum(1) > K).float().mean()}
        # 收成定长 K 的块序列：边界在前、按时间排序；超出 K 的边界并入第 K 块
        pos = (torch.arange(T, device=idx.device) + T * (~bnd).long()).argsort(-1)[:, :K]
        valid = bnd.gather(1, pos)
        keep = valid.float()
        g = lambda x: x.gather(1, pos.view(b, K, *[1] * (x.dim() - 2)).expand(b, K, *x.shape[2:]))
        gm0 = g(m0)
        hm, newm, um = self._stack(1, L - 1, gm0, state, g(reset), g(e), keep)    # 块层
        if self.td_w and K > 1:                            # 目标停梯度、按均方值归一化（E45）；只算前后两块都有效的位置
            tgt = gm0[:, 1:].flatten(2).float().detach()
            err = (self.td(hm[-1][:, :-1]).float() - tgt).square().mean(-1) / tgt.square().mean().clamp(min=1e-2)
            both = keep[:, :-1] * keep[:, 1:]
            self.aux = self.aux + self.td_w * (err * both).sum() / both.sum().clamp(min=1)
        # 放回字节：每个字节取它所在块的输出；本段第一个边界之前的字节属于上一段的块
        bundle = torch.cat(hm + [um.flatten(2).to(hm[0].dtype)], -1)
        table = torch.cat([held_prev[:, None].to(bundle.dtype), bundle], 1)
        cum = bnd.long().cumsum(1).clamp(max=K)
        up = table.gather(1, cum[..., None].expand(-1, -1, table.shape[-1]))
        held = up[:, -1].float()
        nh = self.M * sum(self.dl[1:-1])
        hup = up[..., :nh]
        uup = up[..., nh:].reshape(b, T, self.M, -1)
        hd, newd, _ = self._stack(L - 1, L, torch.cat([m0, uup.to(m0.dtype)], -1), state, reset, e)   # 解码层（字节）
        new = new0 + newm + newd + ([self._hub_new] if self.hub is not None else []) + [(lp[:, -1].detach(), s_last, n_new, held)]
        new += [self._attn_new] if self.attn is not None else []
        return h0 + [hup.to(hd[0].dtype)] + hd, new

    @staticmethod
    def wbound(x):
        return (x == 32) | (x == 10) | (x == 46) | (x == 44) | (x == 33) | (x == 63)


def count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build(kind='dspn', size='S', **extra):
    assert kind == 'dspn', '只保留 DSPN（v2：只和自己的上一版比）'
    cfg = {**SIZES[size], **extra}
    if cfg.get('net') == 'wave':                          # 路由模块池 RMP 最新设计（第 23 章 R2 / R3，S 级原型）
        from pool import WaveNet
        return WaveNet(**cfg)
    return DSPNLayered(**cfg)

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
            k = keep.view(-1, *([1] * (s.dim() - 1)))
            s.logical_and_(k > 0) if s.dtype == torch.bool else s.mul_(k)   # 注意力缓存的有效位是布尔
        elif isinstance(s, (tuple, list)):
            scale_state_(s, keep)


def copy_state_(dst, src):
    """把新状态原地拷进固定地址的旧状态（CUDA Graph 回放时地址不能变）。"""
    for d, s in zip(dst, src):
        if torch.is_tensor(d):
            d.copy_(s.detach())
        elif isinstance(d, (tuple, list)):
            copy_state_(d, s)


def train(kind, size, data, iters, b, T, lr, seed, tag, extra, init=None, stateful=True, bot_loss=False, sched='wsd', bf16=True, compile=False, resume=None, skip=0):
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
    warm = min(2000, max(200, iters // 100))              # 预热：短跑 200 步，长训练最多 2000 步（E61）
    if sched == 'wsd':                                     # 恒定 + 最后 20% 线性降到 0：中途停下也能随时补一段衰减
        lrf = lambda it: min(1, it / warm) * min(1.0, (iters - it) / (0.2 * iters))
    elif sched == 'decay':                                 # 只跑衰减段：给 WSD 恒定段的检查点收尾（配 --init）；200 步预热让新建的 AdamW 矩估计先就位
        lrf = lambda it: min(1, it / 200) * (iters - it) / iters
    else:                                                  # 预热后余弦衰减
        lrf = lambda it: min(1, it / warm) * 0.5 * (1 + math.cos(math.pi * it / iters))
    name = f'{kind}-{size}{tag}'
    print(f'{name}: {nparam:,} 参数，{DEV}，批 {b} × {T} 字节，{iters} 步', flush=True)
    sx, sy = batch(tr, b, T)                               # 固定地址的输入，每步原地拷入新批
    # stateful：每行是一条连续流，状态跨批次延续（截断反传），模型才练得到“记住 T 字节以前的事”（持久头、跨会话记忆）
    pos = np.random.default_rng(seed).integers(0, len(tr) - T - 1, b) if stateful else None
    for _ in range(skip if stateful else 0):               # 跳过已读的数据：接只有权重的检查点时，流位置推到原训练停下的地方
        pos[pos + T + 1 >= len(tr)] = 0; pos += T
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
        (loss + getattr(model, 'aux', 0.0)).backward()      # 编码层局部头等辅助损失
        nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad(set_to_none=False)
        if stateful:
            copy_state_(st, new)
        return loss.detach()                               # 不留计算图引用，否则会干扰 CUDA Graph 录制

    log, t0, eval_secs, graph, best, start = [], time.time(), 0.0, None, float('inf'), 0
    if resume:                                             # 原样续跑：权重、优化器、步数、流位置、流状态、日志
        ck = torch.load(ROOT / resume, map_location=DEV, weights_only=False)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt'])
        for g in opt.param_groups:
            g['lr'] = lr_t                                 # 保持张量学习率的同一个对象（CUDA Graph 里引用的是它）
        start, log, best = ck['it'], ck['log'], ck['best']
        if stateful:
            pos[:] = ck['pos']; copy_state_(st, ck['state'])
    for it in range(start + 1, iters + 1):
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
            if getattr(model, 'chunk', 0):                 # 自发切词：边界比例、边界落在空格 / 标点之后的比例（及其基准率）、K 溢出行比例
                cs = tuple(round(float(model.stats[k]), 3) for k in ('rate', 'word', 'word_base', 'over'))
            if getattr(model, 'pool', 0):                  # 路由池：各池层（负载熵 / log N，死模块比例）
                ps = tuple(l.stats() for l in model.layers[1:])
            v = evaluate_stream(model, va, n=4)
            eval_secs += time.time() - e0
            log.append((it, round(time.time() - t0), round(loss.item() / math.log(2), 3), round(v, 3)))
            if getattr(model, 'chunk', 0):
                log[-1] += cs + (round(float(model.theta), 3),)
            if getattr(model, 'pool', 0):
                log[-1] += ps
            print(log[-1], flush=True)
            if not math.isfinite(loss.item()) or v > best + 0.5:   # 发散就立刻停，不白跑，也不用坏权重覆盖检查点（E61：没有 NaN 也会崩）
                raise SystemExit(f'{name}: 第 {it} 步发散（流式验证 {v:.3f}，最好 {best:.3f}），训练停止')
            (ROOT / 'runs').mkdir(exist_ok=True)
            torch.save(model.state_dict(), ROOT / f'runs/{name}.pt')   # 每次评估都存，长训练中途出事不至于全丢
            if v < best:
                best = v
                torch.save(model.state_dict(), ROOT / f'runs/{name}_best.pt')
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'it': it, 'log': log, 'best': best,
                        'pos': pos, 'state': st}, ROOT / f'runs/{name}_resume.pt')   # --resume 用
    if DEV == 'cuda':
        torch.cuda.synchronize()                            # GPU 异步执行，计时前等它算完
    secs = time.time() - t0
    final = evaluate_stream(model, va)
    (ROOT / 'runs').mkdir(exist_ok=True)
    torch.save(model.state_dict(), ROOT / f'runs/{name}.pt')   # 每个名字只留最新一份检查点
    out = ROOT / f'runs/{time.strftime("%Y%m%d-%H%M%S")}_{name}.json'
    json.dump({'name': name, 'kind': kind, 'size': size, 'extra': extra, 'params': nparam, 'data': data,
               'device': DEV, 'batch': b, 'T': T, 'iters': iters, 'lr': lr, 'seed': seed, 'stateful': stateful, 'bot_loss': bot_loss, 'sched': sched, 'bf16': bf16, 'compile': compile, 'init': init, 'resume': resume, 'skip': skip,
               'code_sha': hashlib.sha256((ROOT / 'dspn_text.py').read_bytes()).hexdigest()[:12],
               'seconds': round(secs), 'train_bytes_per_sec': round((iters - start) * b * T / (secs - eval_secs)),
               'peak_gpu_mb': round(torch.cuda.max_memory_allocated() / 2**20) if DEV == 'cuda' else None,
               'val_bpc_stream': round(final, 4), 'log': log}, open(out, 'w'), ensure_ascii=False, indent=1)
    print(f'{name} 完成：流式验证 bpc {final:.4f} → {out.name}', flush=True)


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
    ap.add_argument('--sched', default='wsd', choices=['cos', 'wsd', 'decay'], help='学习率调度：余弦 / 恒定 + 末段线性衰减 / 只跑衰减段（配 --init）')
    ap.add_argument('--resume', default=None, help='从 runs/<名字>_resume.pt 原样续跑（优化器、步数、流位置都接上）')
    ap.add_argument('--skip', type=int, default=0, help='流位置先前移这么多步，接只有权重的检查点时不重读数据')
    a = ap.parse_args()
    extra = json.loads(a.extra)
    if a.cmd == 'train':
        lr = a.lr or 3e-3
        train(a.kind, a.size, a.data, a.iters, a.batch, a.T, lr, a.seed, a.tag, extra, a.init, a.stateful, a.bot_loss, a.sched, a.bf16, a.compile, a.resume, a.skip)
    else:
        print(sample(a.kind, a.size, a.tag, a.prompt, extra))
