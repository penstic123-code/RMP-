"""DSPN v0.14 原型：字符级（字节级）文本预测与生成。

一层 M 个模块排成网格，每个模块 = 门控对角线性循环 c + 快记忆 S（门控衰减 + delta 写入）
+ 局部 MLP；模块之间用上一拍的 q 维消息与四邻居通信。所有模块都看到当前字节，
所有模块的输出拼起来接一个 softmax 读出，预测下一个字节。

这一版只做离线阶段：用截断 BPTT 训练全部权重，检验“这种模块能不能学会说话”。
与设计文档的差异（原型简化）：
  - 所有模块都接收输入字节（文档中只有部分模块接观测通道）；
  - 没有本地预测头、在线可塑性、换边、睡眠巩固；
  - h = c + MLP([c, r])，加了残差，便于训练。

用法：
  python dspn_text.py train dspn  --minutes 9
  python dspn_text.py train gru   --minutes 9
  python dspn_text.py sample dspn --prompt "ROMEO:"
"""
import argparse, math, time, json, sys
import torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(2)
DATA = 'data/input.txt'


def load_bytes():
    raw = open(DATA, 'rb').read()
    data = torch.tensor(list(raw), dtype=torch.long)
    n = int(len(data) * 0.9)
    return data[:n], data[n:]


def grid_adjacency(rows, cols):
    M = rows * cols
    A = torch.zeros(M, M)
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols:      # 边界不环绕
                    A[i, rr * cols + cc] = 1.0
    return A / 4.0                                         # 邻居消息之和除以 4


class DSPN(nn.Module):
    def __init__(self, rows=2, cols=4, d=32, m=16, hidden=64, q=8, emb=32, vocab=256):
        super().__init__()
        M = rows * cols
        self.M, self.d, self.m, self.q = M, d, m, q
        U = emb + q
        self.register_buffer('adj', grid_adjacency(rows, cols))
        self.emb = nn.Embedding(vocab, emb)
        p = lambda *s: nn.Parameter(torch.randn(*s) / math.sqrt(s[-2]))
        self.B = p(M, U, d)                                 # 输入投影
        self.G = nn.Parameter(torch.zeros(M, U, d))         # 门
        tau = torch.logspace(math.log10(1.5), 3, d)         # 时间常数 1.5–1000 步，对数均匀
        a0 = 1 - 1 / tau
        self.b_a = nn.Parameter(torch.log(a0 / (1 - a0)).repeat(M, 1))
        self.K, self.V, self.Qm = p(M, d, m), p(M, d, m), p(M, d, m)
        self.w_beta = nn.Parameter(torch.zeros(M, d)); self.b_beta = nn.Parameter(torch.zeros(M))
        self.w_gam = nn.Parameter(torch.zeros(M, d))
        self.b_gam = nn.Parameter(torch.full((M,), math.log(1 / 63)))   # 平均保持约 64 步
        self.W1 = p(M, d + m, hidden); self.b1 = nn.Parameter(torch.zeros(M, hidden))
        self.W2 = nn.Parameter(torch.zeros(M, hidden, d))   # 残差分支从 0 开始
        self.P = p(M, d, q)                                 # 消息投影
        self.norm = nn.LayerNorm(M * d)
        self.readout = nn.Linear(M * d, vocab)

    def init_state(self, b):
        z = lambda *s: torch.zeros(b, self.M, *s)
        return z(self.d), z(self.m, self.m), z(self.q)     # c, S, 上一拍消息

    def step(self, x_emb, state):
        c, S, msg = state
        b = x_emb.shape[0]
        nb = torch.einsum('ij,bjq->biq', self.adj, msg)
        u = torch.cat([x_emb[:, None, :].expand(b, self.M, -1), nb], -1)
        a = torch.sigmoid(torch.einsum('bmu,mud->bmd', u, self.G) + self.b_a)
        c = a * c + (1 - a) * torch.einsum('bmu,mud->bmd', u, self.B)
        k = F.normalize(torch.einsum('bmd,mde->bme', c, self.K), dim=-1)
        v = torch.einsum('bmd,mde->bme', c, self.V)
        qv = torch.einsum('bmd,mde->bme', c, self.Qm)
        beta = torch.sigmoid((c * self.w_beta).sum(-1) + self.b_beta)[..., None, None]
        gam = torch.sigmoid((c * self.w_gam).sum(-1) + self.b_gam)[..., None, None]
        S = (1 - gam) * S                                   # 门控遗忘
        Sk = torch.einsum('bmij,bmj->bmi', S, k)
        S = S + beta * torch.einsum('bmi,bmj->bmij', v - Sk, k)   # delta 写入
        r = torch.einsum('bmij,bmj->bmi', S, qv)
        hid = F.gelu(torch.einsum('bmi,mih->bmh', torch.cat([c, r], -1), self.W1) + self.b1)
        h = c + torch.einsum('bmh,mhd->bmd', hid, self.W2)
        msg = torch.einsum('bmd,mdq->bmq', h, self.P)
        logits = self.readout(self.norm(h.reshape(b, -1)))
        return logits, (c, S, msg)

    def forward(self, idx, state=None):
        b, T = idx.shape
        state = state or self.init_state(b)
        e = self.emb(idx)
        out = []
        for t in range(T):
            lg, state = self.step(e[:, t], state)
            out.append(lg)
        return torch.stack(out, 1), state


class GRULM(nn.Module):
    def __init__(self, hidden, emb=32, vocab=256):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.rnn = nn.GRU(emb, hidden, batch_first=True)
        self.readout = nn.Linear(hidden, vocab)

    def forward(self, idx, state=None):
        y, state = self.rnn(self.emb(idx), state)
        return self.readout(y), state



def scan(a, b, c0, chunk=8):
    """并行扫描 c_t = a_t*c_{t-1} + b_t（沿时间维 1）。块内用对数累乘一次算完，块间顺序衔接。"""
    T = a.shape[1]; out = []; c = c0
    for s in range(0, T, chunk):
        la = torch.log(a[:, s:s + chunk].clamp(min=1e-2)).cumsum(1)
        cs = torch.exp(la) * (c[:, None] + torch.cumsum(b[:, s:s + chunk] * torch.exp(-la), 1))
        out.append(cs); c = cs[:, -1]
    return torch.cat(out, 1), c


class Layer(nn.Module):
    """一层模块：所有时间步同时计算门、输入投影和扫描；只有快记忆 S 逐拍循环。"""
    def __init__(self, rows, cols, U, d, m, hidden, q):
        super().__init__()
        M = rows * cols
        self.M, self.d, self.m = M, d, m
        self.register_buffer('adj', grid_adjacency(rows, cols))
        p = lambda *s: nn.Parameter(torch.randn(*s) / math.sqrt(s[-2]))
        self.B = p(M, U, d); self.G = nn.Parameter(torch.zeros(M, U, d))
        tau = torch.logspace(math.log10(1.5), 3, d); a0 = 1 - 1 / tau
        self.b_a = nn.Parameter(torch.log(a0 / (1 - a0)).repeat(M, 1))
        self.K, self.V, self.Qm = p(M, d, m), p(M, d, m), p(M, d, m)
        self.w_beta = nn.Parameter(torch.zeros(M, d)); self.b_beta = nn.Parameter(torch.zeros(M))
        self.w_gam = nn.Parameter(torch.zeros(M, d)); self.b_gam = nn.Parameter(torch.full((M,), math.log(1 / 63)))
        self.W1 = p(M, d + m, hidden); self.b1 = nn.Parameter(torch.zeros(M, hidden))
        self.W2 = nn.Parameter(torch.zeros(M, hidden, d)); self.P = p(M, d, q)

    def forward(self, u, state):            # u: (b, T, M, U)
        c0, S = state
        a = torch.sigmoid(torch.einsum('btmu,mud->btmd', u, self.G) + self.b_a)
        c, cT = scan(a, (1 - a) * torch.einsum('btmu,mud->btmd', u, self.B), c0)
        k = F.normalize(torch.einsum('btmd,mde->btme', c, self.K), dim=-1)
        v = torch.einsum('btmd,mde->btme', c, self.V); qv = torch.einsum('btmd,mde->btme', c, self.Qm)
        beta = torch.sigmoid((c * self.w_beta).sum(-1) + self.b_beta)[..., None, None]
        gam = torch.sigmoid((c * self.w_gam).sum(-1) + self.b_gam)[..., None, None]
        rs = []
        for t in range(u.shape[1]):          # 快记忆：门控衰减 + delta 写入（GPU 上可换成分块并行核）
            S = (1 - gam[:, t]) * S
            S = S + beta[:, t] * torch.einsum('bmi,bmj->bmij', v[:, t] - torch.einsum('bmij,bmj->bmi', S, k[:, t]), k[:, t])
            rs.append(torch.einsum('bmij,bmj->bmi', S, qv[:, t]))
        r = torch.stack(rs, 1)
        hid = F.gelu(torch.einsum('btmi,mih->btmh', torch.cat([c, r], -1), self.W1) + self.b1)
        h = c + torch.einsum('btmh,mhd->btmd', hid, self.W2)
        return h, torch.einsum('btmd,mdq->btmq', h, self.P), (cT, S)


class DSPNLayered(nn.Module):
    """分层版：消息同一拍内由下层传给上层（正下方模块 + 其四邻居），时间记忆只在 c 和 S 中。"""
    def __init__(self, layers=2, rows=2, cols=2, d=32, m=16, hidden=64, q=8, emb=32, vocab=256):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.layers = nn.ModuleList([Layer(rows, cols, emb if i == 0 else 2 * q, d, m, hidden, q) for i in range(layers)])
        M = rows * cols
        self.norm = nn.LayerNorm(layers * M * d); self.readout = nn.Linear(layers * M * d, vocab)
        self.M, self.d, self.m = M, d, m

    def forward(self, idx, state=None):
        b, T = idx.shape
        if state is None:
            state = [(torch.zeros(b, self.M, self.d), torch.zeros(b, self.M, self.m, self.m)) for _ in self.layers]
        u = self.emb(idx)[:, :, None, :].expand(b, T, self.M, -1)
        hs, new = [], []
        for layer, st in zip(self.layers, state):
            h, z, ns = layer(u, st)
            hs.append(h); new.append(ns)
            u = torch.cat([z, torch.einsum('ij,btjq->btiq', layer.adj, z)], -1)
        return self.readout(self.norm(torch.cat(hs, 2).reshape(b, T, -1))), new


def build(kind):
    if kind == 'dspn':
        return DSPN()
    if kind == 'layered':
        return DSPNLayered()
    target = sum(p.numel() for p in DSPN().parameters())
    h = 32
    while sum(p.numel() for p in GRULM(h + 1).parameters()) <= target:
        h += 1
    return GRULM(h)


def batch(data, b, T):
    ix = torch.randint(len(data) - T - 1, (b,))
    x = torch.stack([data[i:i + T] for i in ix]); y = torch.stack([data[i + 1:i + T + 1] for i in ix])
    return x, y


@torch.no_grad()
def evaluate(model, data, n=40, b=32, T=64):
    g = torch.Generator().manual_seed(0)
    tot = 0.0
    for _ in range(n):
        ix = torch.randint(len(data) - T - 1, (b,), generator=g)
        x = torch.stack([data[i:i + T] for i in ix]); y = torch.stack([data[i + 1:i + T + 1] for i in ix])
        lg, _ = model(x)
        tot += F.cross_entropy(lg.reshape(-1, 256), y.reshape(-1)).item()
    return tot / n / math.log(2)                            # 每字符比特数（bpc）


def train(kind, minutes, b=32, T=64, lr=3e-3, iters=None):
    torch.manual_seed(0)
    tr, va = load_bytes()
    model = build(kind)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    log, t0, it = [], time.time(), 0
    print(f'{kind}: {nparam:,} 参数', flush=True)
    while (it < iters) if iters else (time.time() - t0 < minutes * 60):
        x, y = batch(tr, b, T)
        lg, _ = model(x)
        loss = F.cross_entropy(lg.reshape(-1, 256), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        it += 1
        if it % 200 == 0:
            v = evaluate(model, va, n=10)
            log.append((it, round(time.time() - t0), round(loss.item() / math.log(2), 3), round(v, 3)))
            print(log[-1], flush=True)
    final = evaluate(model, va)
    torch.save(model.state_dict(), f'{kind}.pt')
    json.dump({'kind': kind, 'params': nparam, 'iters': it, 'seconds': round(time.time() - t0),
               'val_bpc': round(final, 4), 'log': log}, open(f'{kind}_result.json', 'w'), ensure_ascii=False)
    print(f'{kind} 完成：{it} 步，验证 bpc {final:.3f}', flush=True)


@torch.no_grad()
def sample(kind, prompt, n=600, temp=0.8, seed=1):
    torch.manual_seed(seed)
    model = build(kind); model.load_state_dict(torch.load(f'{kind}.pt')); model.eval()
    idx = torch.tensor([list(prompt.encode())])
    lg, state = model(idx)
    out = list(prompt.encode())
    nxt = torch.multinomial(F.softmax(lg[0, -1] / temp, -1), 1)
    for _ in range(n):
        out.append(nxt.item())
        lg, state = model(nxt.view(1, 1), state)
        nxt = torch.multinomial(F.softmax(lg[0, -1] / temp, -1), 1)
    return bytes(out).decode('utf-8', errors='replace')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd'); ap.add_argument('kind')
    ap.add_argument('--minutes', type=float, default=9)
    ap.add_argument('--prompt', default='ROMEO:')
    ap.add_argument('--iters', type=int, default=None)
    a = ap.parse_args()
    if a.cmd == 'train':
        train(a.kind, a.minutes, iters=a.iters)
    else:
        print(sample(a.kind, a.prompt))
