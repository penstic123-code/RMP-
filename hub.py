"""全局记忆中枢（设计第 20 章）：门控 delta 记忆头，异质时间常数（默认 300–30000 拍），会话边界全部清空。
整段并行，状态可跨段带入；T=1 逐拍与整段结果一致。跨会话记忆由海马（episodic.py）承担，不再有持久头。"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dspn_text import delta_chunk


class Hub(nn.Module):
    def __init__(self, n_in, M_out, out_dim=4, heads=2, dk=32, beta_max=1.0, taus=None):
        """taus：各头遗忘时间常数初值（拍）。默认全部 1000。"""
        super().__init__()
        self.M_out, self.out_dim, self.heads, self.dk = M_out, out_dim, heads, dk
        self.beta_max = beta_max
        hd = heads * dk
        p = lambda i, o: nn.Parameter(torch.randn(i, o) / math.sqrt(i))
        self.Wq, self.Wk, self.Wv = p(n_in, hd), p(n_in, hd), p(n_in, hd)
        self.w_beta = nn.Parameter(torch.zeros(n_in, heads)); self.b_beta = nn.Parameter(torch.zeros(heads))
        self.w_alpha = nn.Parameter(torch.zeros(n_in, heads))
        tau = torch.tensor(taus if taus else [1000.0] * heads)
        self.b_alpha = nn.Parameter(torch.log(tau - 1))                       # α = 1 − 1/τ，logit(α) = log(τ − 1)
        self.Wo = p(hd, M_out * out_dim)

    def init_state(self, b, device):
        return (torch.zeros(b, self.heads, self.dk, self.dk, device=device),)

    def forward(self, x, state, reset=None, keep=None):
        """返回 (逐拍输出 (b, T, M_out, out_dim), 新状态, 逐拍读出 (b, T, heads·dk)——末拍供窗口级广播)。"""
        b, T, _ = x.shape
        H, dk = self.heads, self.dk
        heads = lambda t: t.reshape(b, T, H, dk)
        k = F.normalize(heads(x @ self.Wk), dim=-1)
        q = F.normalize(heads(x @ self.Wq), dim=-1)
        v = heads(x @ self.Wv)
        beta = self.beta_max * torch.sigmoid(x @ self.w_beta + self.b_beta)
        g = F.logsigmoid((x @ self.w_alpha).float() + self.b_alpha)
        if reset is not None:                                                  # 会话边界：全部清空
            g = g.masked_fill(reset[..., None], -30.0)
        if keep is not None:                                                   # 块序列的补齐位：不写不衰减
            beta, g = beta * keep[..., None].to(beta.dtype), g * keep[..., None]
        r, S = delta_chunk(q, k, v, beta, g, state[0])
        r = r.reshape(b, T, -1).to(x.dtype)
        o = r @ self.Wo
        return o.reshape(b, T, self.M_out, self.out_dim), (S,), r
