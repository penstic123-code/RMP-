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
