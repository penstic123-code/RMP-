"""全局记忆中枢（设计第 1.2 节）：门控 delta 记忆头 + 可选滑动窗口因果注意力头。
离线可训练版本：整段并行，状态可跨段带入；T=1 逐拍与整段结果一致。"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dspn_text import delta_chunk

AD = 32   # 注意力头维


def rope(x, pos):
    """x: (b, T, AD)，pos: (T,) 绝对位置。半维旋转；缓存里存旋转后的 K，所以相对位置只由内积体现。"""
    half = x.shape[-1] // 2
    freq = 10000 ** (-torch.arange(half, device=x.device) / half)
    ang = pos[:, None].float() * freq
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


class Hub(nn.Module):
    """lora_* 是读路径（W_q、W_o）的秩 2 在线偏移（设计第 1.2 节），离线冻结，在线阶段打开。"""
    def __init__(self, n_in, M_out, out_dim=4, heads=2, dk=32, window=0, rank=2, beta_max=1.0, taus=None):
        """taus：各头遗忘时间常数初值（拍），如 [300, 3000, 30000, 1e6]；最长的头近似持久记忆。默认全部 1000。"""
        super().__init__()
        self.M_out, self.out_dim, self.heads, self.dk, self.window = M_out, out_dim, heads, dk, window
        self.beta_max = beta_max
        hd = heads * dk
        p = lambda i, o: nn.Parameter(torch.randn(i, o) / math.sqrt(i))
        self.Wq, self.Wk, self.Wv = p(n_in, hd), p(n_in, hd), p(n_in, hd)
        self.w_beta = nn.Parameter(torch.zeros(n_in, heads)); self.b_beta = nn.Parameter(torch.zeros(heads))
        self.w_alpha = nn.Parameter(torch.zeros(n_in, heads))
        tau = torch.tensor(taus if taus else [1000.0] * heads)
        self.b_alpha = nn.Parameter(torch.log(tau - 1))                       # α = 1 − 1/τ，logit(α) = log(τ − 1)
        self.register_buffer('persist', tau >= 1e5, persistent=False)        # 会话边界处不清空的持久头
        if window:
            self.att_qkv = nn.Linear(n_in, 3 * AD)
        o_in = hd + (AD if window else 0)
        self.Wo = p(o_in, M_out * out_dim)
        self.lora_q_a = nn.Parameter(torch.randn(n_in, rank) / math.sqrt(n_in), requires_grad=False)
        self.lora_q_b = nn.Parameter(torch.zeros(rank, hd), requires_grad=False)
        self.lora_o_a = nn.Parameter(torch.randn(o_in, rank) / math.sqrt(o_in), requires_grad=False)
        self.lora_o_b = nn.Parameter(torch.zeros(rank, M_out * out_dim), requires_grad=False)

    def init_state(self, b, device):
        S = torch.zeros(b, self.heads, self.dk, self.dk, device=device)
        if not self.window:
            return (S,)
        empty = torch.zeros(b, 0, AD, device=device)
        return (S, empty, empty, 0)

    def forward(self, x, state, reset=None, valid=None):
        """valid: (b, T) 布尔，False 的位置（词级序列的补齐位）不写不衰减。"""
        b, T, _ = x.shape
        H, dk = self.heads, self.dk
        heads = lambda t: t.reshape(b, T, H, dk)
        k = F.normalize(heads(x @ self.Wk), dim=-1)
        q = F.normalize(heads(x @ (self.Wq + self.lora_q_a @ self.lora_q_b)), dim=-1)
        v = heads(x @ self.Wv)
        beta = self.beta_max * torch.sigmoid(x @ self.w_beta + self.b_beta)
        g = F.logsigmoid(x @ self.w_alpha + self.b_alpha)
        if reset is not None:                                                  # 会话边界：清空短时头，持久头不动
            g = g.masked_fill(reset[..., None] & ~self.persist, -30.0)
        if valid is not None:
            beta, g = beta * valid[..., None], g * valid[..., None]
        r, S = delta_chunk(q, k, v, beta, g, state[0])
        outs = [r.reshape(b, T, -1)]
        new_state = (S,)
        if self.window:
            _, Kc, Vc, pos = state
            aq, ak, av = self.att_qkv(x).chunk(3, -1)
            ps = torch.arange(pos, pos + T, device=x.device)
            aq, ak = rope(aq, ps), rope(ak, ps)
            K, V = torch.cat([Kc, ak], 1), torch.cat([Vc, av], 1)
            L = Kc.shape[1]
            qi = ps[:, None]; kj = torch.arange(pos - L, pos + T, device=x.device)[None]
            mask = (kj <= qi) & (qi - kj < self.window)
            outs.append(F.scaled_dot_product_attention(aq, K, V, attn_mask=mask))
            keep = max(0, K.shape[1] - (self.window - 1))
            new_state = (S, K[:, keep:], V[:, keep:], pos + T)
        o = torch.cat(outs, -1) @ (self.Wo + self.lora_o_a @ self.lora_o_b)
        return o.reshape(b, T, self.M_out, self.out_dim), new_state
