"""A2 v2 架构：双流正交符号绑定 + 潜空间自适应思考 RMP + 极简端到端海马读取 + 原生 SMF 持续学习。

核心设计原则（避免苦涩的教训 & 避免退化成 Transformer）：
1. 主干 100% 零 Self-Attention：`WavePool`（竞争多尺度语境路由 + 慢通道 c + Gated DeltaNet 快记忆 S + 局部 MLP），
   对话流状态严格 O(1)，不膨胀 KV-cache；
2. 双流正交符号绑定（Vector-Symbolic / Hyperdimensional Binding，零手工规则）：
   - 语义控制流（x ∈ R^D）：由 `WavePool` 负责语法、角色、意图、停机门控与 SMF 持续学习；
   - 符号绑定流（s ∈ R^{d_s}）：每个 token 映射到单位随机向量；训练期每批重采样正交/随机单位码本，
     网络只能通过内积使用符号流，数学上不可能背诵训练实体，被迫学会对任意未见实体零样本泛化的绑定与归纳电路；
3. 潜空间自适应思考（无脚本监督）：同一 `WavePool` 权重复用 K 拍（每拍独立 O(1) 状态槽，训练全序列并行、推理逐 token
   数值严格等价），每拍将海马读回的语义与符号向量回注下一拍输入，无任何跳数脚本即可自然涌现多跳连想；
4. 极简端到端多头海马读取（尊重 Jelassi et al. 2024 信息论硬上限，零手工特征、无指针对齐监督）；
5. 原生兼容 `smf_mlp` 局部抗遗忘在线学习（复用 `online._masked_step`）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dspn_text import Layer
from online import _masked_step
from pool import WavePool, ema_scan


def _make_decays(G: int, lo: float = 0.45, hi: float = 0.985) -> List[float]:
    if G <= 1:
        return [hi]
    tau_lo, tau_hi = 1.0 / (1.0 - lo), 1.0 / (1.0 - hi)
    return [round(1.0 - 1.0 / (tau_lo * (tau_hi / tau_lo) ** (i / (G - 1))), 5) for i in range(G)]


@dataclass
class HippoMemory:
    """RMP 自身编码出的外部海马记忆包（批量并行，含语义键值与正交符号迹）。"""
    keys: torch.Tensor          # (B, H, L_mem, head_dim)
    vals: torch.Tensor          # (B, H, L_mem, D // H)
    token_ids: torch.Tensor     # (B, L_mem)
    valid_mask: torch.Tensor    # (B, L_mem) bool
    sym_cur: torch.Tensor       # (B, L_mem, sym_dim) 当前位置符号向量（供多跳回传）
    sym_prev: torch.Tensor      # (B, L_mem, sym_dim) 前一位置符号向量（供零样本归纳续写）
    sym_ctx: torch.Tensor       # (B, L_mem, sym_dim) 因果前文符号叠加（供关联检索）
    sym_codebook: torch.Tensor  # (V, sym_dim) 本批次使用的单位符号码本


class LatentThinkHippoRMP(nn.Module):
    """A2 v2：零 Self-Attention 路由模块池 + 双流符号绑定 + 潜空间自适应思考 + 极简端到端海马读取。"""

    def __init__(
        self,
        vocab: int = 2048,
        D: int = 128,
        M: int = 2,
        d: int = 64,
        m: int = 16,
        hidden: int = 128,
        q: int = 64,
        N: int = 24,
        k: int = 2,
        G: int = 4,
        pd: int = 96,
        pm: int = 16,
        phidden: int = 192,
        qr: int = 32,
        beta_max: float = 2.0,
        temp: float = 0.1,
        ctx_decay: Optional[Sequence[float]] = None,
        tau_prior: float = 2.0,
        bottleneck: bool = False,
        lora_rank: int = 2,
        shared: int = 0,
        sep_id: Optional[int] = None,
        mem_sep_id: Optional[int] = None,
        pad_id: int = 0,
        max_think: int = 3,
        use_hippo: bool = True,
        hippo_dim: int = 128,
        n_hippo_heads: int = 4,
        sym_dim: int = 64,
        sym_decay: float = 0.82,
        resample_sym: bool = True,
        tie_emb: bool = True,
    ):
        super().__init__()
        assert D % n_hippo_heads == 0 and hippo_dim % n_hippo_heads == 0
        self.vocab = vocab
        self.D = D
        self.M = M
        self.G = G
        self.N = N
        self.k = k
        self.sep_id = sep_id
        self.mem_sep_id = mem_sep_id
        self.pad_id = pad_id
        self.max_think = max(1, int(max_think))
        self.use_hippo = bool(use_hippo)
        self.hippo_dim = hippo_dim
        self.n_hippo_heads = n_hippo_heads
        self.head_dim = hippo_dim // n_hippo_heads
        self.val_head_dim = D // n_hippo_heads
        self.sym_dim = sym_dim
        self.sym_decay = float(sym_decay)
        self.resample_sym = bool(resample_sym)

        decays = list(ctx_decay) if ctx_decay is not None else _make_decays(G)
        self.emb = nn.Embedding(vocab, D)
        nn.init.normal_(self.emb.weight, std=0.02)

        # 固定评估用单位符号码本（不求导；训练期若 resample_sym=True 则每批动态生成新码本）
        cb = F.normalize(torch.randn(vocab, sym_dim), dim=-1)
        cb[pad_id] = 0.0
        self.register_buffer("sym_codebook", cb)

        enc = Layer(M, D, d, m, hidden, q, rank=lora_rank, beta_max=beta_max)
        pool = WavePool(
            N=N, k=k, G=G, D=D, d=pd, m=pm, hidden=phidden, qr=qr,
            beta_max=beta_max, temp=temp, init="spread", rule="ctx",
            wave_query=True, wave_ctx=True, ctx_decay=decays,
            tau_prior=tau_prior, bottleneck=bottleneck, lora_rank=lora_rank,
            shared=shared,
        )
        # 遵循 WaveNet / online.py 约定：layers[0]=编码层，layers[1]=WavePool
        self.layers = nn.ModuleList([enc, pool])
        self.win = nn.Linear(M * q, D)

        # 潜空间思考反馈（零初始化：起步等价恒等残差）
        self.tick_emb = nn.Parameter(torch.zeros(self.max_think, D))
        self.think_norm = nn.LayerNorm(D)
        self.think_proj = nn.Linear(D, D)
        nn.init.zeros_(self.think_proj.weight)
        nn.init.zeros_(self.think_proj.bias)

        # 说话/停机门（PonderNet 停机概率，完全端到端学习，无脚本监督）
        self.emit_gate = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D // 2),
            nn.GELU(),
            nn.Linear(D // 2, 1),
        )

        # 极简端到端多头海马读取 + 正交符号绑定门控
        if self.use_hippo:
            self.hippo_k_norm = nn.LayerNorm(D)
            self.hippo_k = nn.Linear(D, hippo_dim, bias=False)
            self.hippo_v = nn.Linear(D, D, bias=False)
            self.hippo_q_norm = nn.LayerNorm(D)
            self.hippo_q = nn.Linear(D, hippo_dim, bias=False)
            self.hippo_out = nn.Linear(D, D)
            nn.init.zeros_(self.hippo_out.weight)
            nn.init.zeros_(self.hippo_out.bias)
            self.null_k = nn.Parameter(torch.randn(1, n_hippo_heads, 1, self.head_dim) * 0.02)
            self.null_v = nn.Parameter(torch.zeros(1, n_hippo_heads, 1, self.val_head_dim))

            # 由 WavePool 语义流动态控制各头对「前字归纳 / 前文关联」符号内积的权重与 NULL 偏置
            self.sym_gates = nn.Linear(D, n_hippo_heads * 2)
            self.null_sym_bias = nn.Parameter(torch.zeros(1, n_hippo_heads, 1, 1))
            # 多跳符号接力：选头读回符号向量 + 门控融合到下一拍 s_bind
            self.sym_read_head = nn.Linear(D, n_hippo_heads)
            self.sym_update_gate = nn.Linear(D, 1)
            # 指针读出头融合 + 拷贝门
            self.ptr_head_gate = nn.Linear(D, n_hippo_heads)
            self.copy_gate = nn.Sequential(
                nn.LayerNorm(D * 2),
                nn.Linear(D * 2, D // 2),
                nn.GELU(),
                nn.Linear(D // 2, 1),
            )

        # 读出头
        self.read_norm = nn.LayerNorm(D)
        self.readout = nn.Identity() if tie_emb else nn.Linear(D, D)
        if not tie_emb:
            nn.init.zeros_(self.readout.weight)
        self.bias = nn.Parameter(torch.zeros(vocab))

    def _get_codebook(self, device: torch.device) -> torch.Tensor:
        if self.training and self.resample_sym:
            cb = F.normalize(torch.randn(self.vocab, self.sym_dim, device=device), dim=-1)
            cb[self.pad_id] = 0.0
            if self.mem_sep_id is not None:
                cb[self.mem_sep_id] = 0.0
            return cb
        cb = self.sym_codebook.to(device)
        if self.mem_sep_id is not None:
            cb = cb.clone()
            cb[self.mem_sep_id] = 0.0
        return cb

    def init_state(self, batch_size: int, device: Optional[torch.device] = None):
        dev = device or self.emb.weight.device
        enc_st = self.layers[0].init_state(batch_size, dev)
        pool_sts = [self.layers[1].init_state(batch_size, dev) for _ in range(self.max_think)]
        sym_ctx_st = torch.zeros(batch_size, self.sym_dim, device=dev)
        return (enc_st, pool_sts, sym_ctx_st)

    def _encode_stream(self, idx: torch.Tensor, enc_st, reset: Optional[torch.Tensor] = None):
        if reset is None and self.sep_id is not None:
            reset = idx == self.sep_id
        e = self.emb(idx)
        _, z0, new_enc_st = self.layers[0](
            e[:, :, None].expand(idx.shape[0], idx.shape[1], self.M, self.D),
            enc_st,
            reset,
        )
        x0 = self.win(z0.flatten(2))
        return x0, new_enc_st, reset

    def encode_memory(
        self,
        mem_ids: torch.Tensor,
        mem_valid: Optional[torch.Tensor] = None,
    ) -> HippoMemory:
        """用 RMP 自身编码层 + WavePool 批量编码外部情景库，并提取因果正交符号迹（零外部编码器）。"""
        assert self.use_hippo, "use_hippo=False 时不可调用 encode_memory"
        B, Lm = mem_ids.shape
        dev = mem_ids.device
        if mem_valid is None:
            mem_valid = mem_ids != self.pad_id
        mem_reset = (mem_ids == self.mem_sep_id) if self.mem_sep_id is not None else None

        pool = self.layers[1]
        old_plastic = pool.plastic
        pool.plastic = False
        try:
            enc_st = self.layers[0].init_state(B, dev)
            pool_st = pool.init_state(B, dev)
            x0, _, _ = self._encode_stream(mem_ids, enc_st, reset=mem_reset)
            _, x_mem, _ = pool(x0, pool_st, mem_reset, mem_ids)
        finally:
            pool.plastic = old_plastic

        H = self.n_hippo_heads
        keys = self.hippo_k(self.hippo_k_norm(x_mem)).view(B, Lm, H, self.head_dim).transpose(1, 2)
        vals = self.hippo_v(x_mem).view(B, Lm, H, self.val_head_dim).transpose(1, 2)

        cb = self._get_codebook(dev)
        sym_cur = F.embedding(mem_ids, cb)  # (B, Lm, sym_dim)
        # 严格因果 1 步延迟（供零样本归纳续写：当前字匹配记忆中的前一个字 -> 指向下一个字）
        sym_prev = torch.zeros_like(sym_cur)
        if Lm > 1:
            sym_prev[:, 1:] = sym_cur[:, :-1]
            if mem_reset is not None:
                sym_prev = sym_prev.masked_fill(mem_reset[..., None], 0.0)
        # 严格因果前文符号叠加（供零样本实体关联匹配）
        s0 = torch.zeros(B, self.sym_dim, device=dev)
        sym_ctx_raw, _ = ema_scan(sym_prev, s0, self.sym_decay, reset=mem_reset)
        sym_ctx = F.normalize(sym_ctx_raw, dim=-1, eps=1e-6)

        return HippoMemory(
            keys=keys,
            vals=vals,
            token_ids=mem_ids,
            valid_mask=mem_valid,
            sym_cur=sym_cur,
            sym_prev=sym_prev,
            sym_ctx=sym_ctx,
            sym_codebook=cb,
        )

    def _hippo_read(
        self,
        x_cur: torch.Tensor,
        s_cur: torch.Tensor,
        s_bind: torch.Tensor,
        hippo_mem: Optional[HippoMemory],
    ):
        """多头门控语义 + 正交符号绑定海马读取；同时返回更新后的下一拍符号绑定向量 s_bind_next。"""
        if (not self.use_hippo) or hippo_mem is None or hippo_mem.keys.shape[2] == 0:
            return x_cur, None, None, None, s_bind

        B, T, _ = x_cur.shape
        H = self.n_hippo_heads
        q = self.hippo_q(self.hippo_q_norm(x_cur)).view(B, T, H, self.head_dim).transpose(1, 2)  # (B, H, T, dh)
        k_ext = torch.cat([hippo_mem.keys, self.null_k.expand(B, -1, -1, -1)], dim=2)            # (B, H, Lm+1, dh)
        v_ext = torch.cat([hippo_mem.vals, self.null_v.expand(B, -1, -1, -1)], dim=2)            # (B, H, Lm+1, dv)

        scores = torch.matmul(q, k_ext.transpose(-2, -1)) / math.sqrt(self.head_dim)             # (B, H, T, Lm+1)

        # 正交符号内积通道（100% 旋转不变，对未见字符零样本生效！）
        sim_prev = torch.matmul(s_cur.float(), hippo_mem.sym_prev.float().transpose(-2, -1))     # (B, T, Lm)
        sim_ctx = torch.matmul(s_bind.float(), hippo_mem.sym_ctx.float().transpose(-2, -1))      # (B, T, Lm)
        gates = F.softplus(self.sym_gates(x_cur).float()).view(B, T, H, 2).permute(0, 2, 1, 3)   # (B, H, T, 2)
        g_prev = gates[..., 0:1]                                                                 # (B, H, T, 1)
        g_ctx = gates[..., 1:2]                                                                  # (B, H, T, 1)

        sym_term = g_prev * sim_prev[:, None, :, :] + g_ctx * sim_ctx[:, None, :, :]             # (B, H, T, Lm)
        sym_ext = torch.cat([sym_term, self.null_sym_bias.expand(B, -1, T, -1)], dim=-1)         # (B, H, T, Lm+1)
        scores = scores.float() + sym_ext

        valid_ext = torch.cat(
            [hippo_mem.valid_mask, torch.ones((B, 1), dtype=torch.bool, device=x_cur.device)],
            dim=1,
        )  # (B, Lm+1)
        scores = scores.masked_fill(~valid_ext[:, None, None, :], -1e9)
        attn = F.softmax(scores, dim=-1)                                                         # (B, H, T, Lm+1)

        ctx_h = torch.matmul(attn.to(v_ext.dtype), v_ext)                                        # (B, H, T, dv)
        ctx = ctx_h.transpose(1, 2).contiguous().view(B, T, self.D)
        feat = x_cur + self.hippo_out(ctx)

        # 指针头加权融合（供词表+拷贝混合）
        w_ptr = F.softmax(self.ptr_head_gate(feat).float(), dim=-1).permute(0, 2, 1)[..., None]  # (B, H, T, 1)
        attn_ptr_full = (w_ptr * attn).sum(dim=1)                                                # (B, T, Lm+1)
        attn_mem = attn_ptr_full[..., :-1]
        attn_null = attn_ptr_full[..., -1:]

        # 多跳符号接力：从海马读回被命中位置的符号向量，并门控更新 s_bind 供下一拍潜空间思考
        w_sym = F.softmax(self.sym_read_head(feat).float(), dim=-1).permute(0, 2, 1)[..., None]  # (B, H, T, 1)
        attn_sym = (w_sym * attn[..., :-1]).sum(dim=1)                                           # (B, T, Lm)
        ctx_sym = torch.matmul(attn_sym, hippo_mem.sym_cur.float())                              # (B, T, sym_dim)
        ctx_sym_n = F.normalize(ctx_sym, dim=-1, eps=1e-6)
        beta = torch.sigmoid(self.sym_update_gate(feat).float())                                 # (B, T, 1)
        s_bind_next = F.normalize((1.0 - beta) * s_bind.float() + beta * ctx_sym_n, dim=-1, eps=1e-6)

        return feat, ctx, attn_mem, attn_null, s_bind_next

    def _tick_log_probs(
        self,
        feat: torch.Tensor,
        ctx: Optional[torch.Tensor],
        attn_mem: Optional[torch.Tensor],
        attn_null: Optional[torch.Tensor],
        hippo_mem: Optional[HippoMemory],
    ) -> torch.Tensor:
        y = self.readout(self.read_norm(feat))
        gen_logits = F.linear(y, self.emb.weight.to(y.dtype), self.bias.to(y.dtype))
        if ctx is None or attn_mem is None or attn_null is None or hippo_mem is None:
            return F.log_softmax(gen_logits.float(), dim=-1)

        p_gen = F.softmax(gen_logits.float(), dim=-1)  # (B, T, V)
        p_copy = torch.sigmoid(self.copy_gate(torch.cat([feat, ctx], dim=-1)).float())  # (B, T, 1)
        B, T, V = p_gen.shape
        p_ptr = torch.zeros((B, T, V), dtype=torch.float32, device=p_gen.device)
        mem_idx = hippo_mem.token_ids[:, None, :].expand(B, T, -1)
        p_ptr.scatter_add_(2, mem_idx, attn_mem)
        mem_mass = (1.0 - attn_null).clamp(min=0.0, max=1.0)
        probs = ((1.0 - p_copy * mem_mass) * p_gen + p_copy * p_ptr).clamp(min=1e-8)
        return probs.log()

    def forward(
        self,
        idx: torch.Tensor,
        state=None,
        hippo_mem: Optional[HippoMemory] = None,
        return_ticks: bool = False,
    ):
        """批量并行前向（与逐 token 流式推理严格数值等价）。"""
        B, T = idx.shape
        dev = idx.device
        if state is None:
            state = self.init_state(B, dev)
        enc_st, pool_sts, sym_ctx_st = state

        x0, new_enc_st, reset = self._encode_stream(idx, enc_st)

        # 因果更新查询/输出流的 O(1) 符号迹（与 hippo_mem 共享同一码本）
        if self.use_hippo and hippo_mem is not None:
            cb = hippo_mem.sym_codebook
            s_cur = F.embedding(idx, cb)  # (B, T, sym_dim)
            s_ctx_raw, new_sym_ctx_st = ema_scan(s_cur, sym_ctx_st, self.sym_decay, reset=reset)
            s_bind = F.normalize(s_ctx_raw, dim=-1, eps=1e-6)
        else:
            s_cur = torch.zeros(B, T, self.sym_dim, device=dev)
            s_bind = s_cur
            new_sym_ctx_st = sym_ctx_st

        new_pool_sts = []
        tick_log_probs: List[torch.Tensor] = []
        halt_probs_list: List[torch.Tensor] = []
        null_probs_list: List[torch.Tensor] = []

        rem = torch.ones((B, T), dtype=torch.float32, device=dev)
        x_in = x0 + self.tick_emb[0]

        for tick in range(self.max_think):
            _, x_out, st_k = self.layers[1](x_in, pool_sts[tick], reset, idx)
            new_pool_sts.append(st_k)

            feat, ctx, attn_mem, attn_null, s_bind_next = self._hippo_read(x_out, s_cur, s_bind, hippo_mem)
            lp_k = self._tick_log_probs(feat, ctx, attn_mem, attn_null, hippo_mem)
            tick_log_probs.append(lp_k)
            if attn_null is not None:
                null_probs_list.append(attn_null.squeeze(-1))

            if tick < self.max_think - 1:
                h_k = torch.sigmoid(self.emit_gate(feat).squeeze(-1).float())
                p_k = rem * h_k
                rem = rem * (1.0 - h_k)
                x_in = feat + self.think_proj(self.think_norm(feat)) + self.tick_emb[tick + 1]
                s_bind = s_bind_next
            else:
                p_k = rem
            halt_probs_list.append(p_k)

        halt_probs = torch.stack(halt_probs_list, dim=0)  # (K, B, T)
        stacked_lp = torch.stack(tick_log_probs, dim=0)   # (K, B, T, V)
        log_w = halt_probs.clamp(min=1e-8).log()[..., None]
        marginal_lp = torch.logsumexp(log_w + stacked_lp, dim=0)

        new_state = (new_enc_st, new_pool_sts, new_sym_ctx_st)
        if not return_ticks:
            return marginal_lp, new_state

        steps_vec = torch.arange(1, self.max_think + 1, dtype=torch.float32, device=dev)[:, None, None]
        exp_steps = (halt_probs * steps_vec).sum(dim=0)  # (B, T)
        null_prob = (
            (halt_probs * torch.stack(null_probs_list, dim=0)).sum(dim=0)
            if null_probs_list
            else None
        )
        info = {
            "tick_log_probs": tick_log_probs,
            "halt_probs": halt_probs,
            "exp_steps": exp_steps,
            "null_prob": null_prob,
        }
        return marginal_lp, new_state, info

    def compute_loss(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor,
        hippo_mem: Optional[HippoMemory] = None,
        state=None,
        ignore_index: int = -100,
        ponder_w: float = 0.02,
        prior_p: float = 0.5,
    ) -> Tuple[torch.Tensor, Dict[str, float], tuple]:
        """纯端到端 PonderNet 损失（零脚本思考监督、零指针对齐监督）。"""
        marginal_lp, new_state, info = self.forward(idx, state=state, hippo_mem=hippo_mem, return_ticks=True)
        valid = targets != ignore_index
        valid_f = valid.float()
        denom = valid_f.sum().clamp(min=1.0)
        safe_t = targets.masked_fill(~valid, 0)

        halt_probs = info["halt_probs"]  # (K, B, T)
        nll_per_tick = []
        for lp_k in info["tick_log_probs"]:
            tok_nll = -lp_k.gather(2, safe_t[..., None]).squeeze(-1)
            nll_per_tick.append(tok_nll)
        nll_stack = torch.stack(nll_per_tick, dim=0)
        task_loss = ((halt_probs * nll_stack).sum(dim=0) * valid_f).sum() / denom

        if self.max_think > 1 and ponder_w > 0.0:
            K = self.max_think
            prior = []
            r = 1.0
            for k in range(K):
                pk = r * prior_p if k < K - 1 else r
                prior.append(pk)
                r *= (1.0 - prior_p)
            prior_t = torch.tensor(prior, dtype=torch.float32, device=idx.device)[:, None, None]
            hp = halt_probs.clamp(min=1e-7)
            kl = (hp * (hp.log() - prior_t.clamp(min=1e-7).log())).sum(dim=0)
            kl_loss = (kl * valid_f).sum() / denom
        else:
            kl_loss = torch.tensor(0.0, device=idx.device)

        total = task_loss + ponder_w * kl_loss
        exp_steps_mean = float(((info["exp_steps"] * valid_f).sum() / denom).item())
        pred = marginal_lp.argmax(dim=-1)
        acc = float((((pred == targets) & valid).float().sum() / denom).item())
        metrics = {
            "loss": float(total.item()),
            "task_loss": float(task_loss.item()),
            "kl_loss": float(kl_loss.item()),
            "exp_steps": exp_steps_mean,
            "tok_acc": acc,
        }
        return total, metrics, new_state

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        hippo_mem: Optional[HippoMemory] = None,
        max_new_tokens: int = 24,
        eos_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """自回归贪心解码（O(1) 对话状态，返回生成 token 序列与每步期望思考拍数）。"""
        self.eval()
        B = prompt_ids.shape[0]
        dev = prompt_ids.device
        lp, st, info = self.forward(prompt_ids, state=None, hippo_mem=hippo_mem, return_ticks=True)
        nxt = lp[:, -1].argmax(dim=-1, keepdim=True)
        gen = [nxt]
        steps = [info["exp_steps"][:, -1:]]
        finished = torch.zeros(B, dtype=torch.bool, device=dev)
        if eos_id is not None:
            finished |= (nxt[:, 0] == eos_id)

        for _ in range(max_new_tokens - 1):
            if eos_id is not None and bool(finished.all()):
                break
            lp, st, info = self.forward(nxt, state=st, hippo_mem=hippo_mem, return_ticks=True)
            nxt = lp[:, -1].argmax(dim=-1, keepdim=True)
            if eos_id is not None:
                nxt = torch.where(finished[:, None], torch.full_like(nxt, eos_id), nxt)
                finished |= (nxt[:, 0] == eos_id)
            gen.append(nxt)
            steps.append(info["exp_steps"][:, -1:])
        return torch.cat(gen, dim=1), torch.cat(steps, dim=1)


class A2SMFOnlineLearner:
    """原生 SMF (`smf_mlp`) 在线持续学习器：冻结编码层/路由/海马，仅按相对激活频率更新 top-k 波池模块 MLP。"""

    def __init__(
        self,
        model: LatentThinkHippoRMP,
        lr: float = 1e-3,
        smf_k: int = 8,
        smf_eps: float = 1e-2,
        wd: float = 0.0,
    ):
        self.model = model
        self.smf_k = smf_k
        self.smf_eps = smf_eps
        self.pool: WavePool = model.layers[1]
        self.core = self.pool.core

        for p in model.parameters():
            p.requires_grad_(False)
        self.mlp_params = [self.core.W1, self.core.b1, self.core.W2]
        for p in self.mlp_params:
            p.requires_grad_(True)
        self.pool.plastic = False

        self.opt = torch.optim.AdamW(self.mlp_params, lr=lr, weight_decay=wd)
        self.bg_freq = torch.full((self.pool.N,), 1.0 / self.pool.N, device=model.emb.weight.device)

    @torch.no_grad()
    def calibrate_background(self, batches: Sequence[Tuple[torch.Tensor, Optional[HippoMemory]]]) -> torch.Tensor:
        dev = self.model.emb.weight.device
        cnt = torch.zeros(self.pool.N, device=dev)
        self.model.eval()
        self.pool.capture = []
        try:
            for idx, hmem in batches:
                self.pool.capture.clear()
                self.model(idx, hippo_mem=hmem)
                for _, sel, _, _ in self.pool.capture:
                    cnt += sel.sum((0, 1))
        finally:
            self.pool.capture = None
        self.bg_freq = cnt / cnt.sum().clamp(min=1.0)
        return self.bg_freq

    def step(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor,
        hippo_mem: Optional[HippoMemory] = None,
        ignore_index: int = -100,
        ponder_w: float = 0.0,
    ) -> Tuple[Dict[str, float], torch.Tensor]:
        dev = idx.device
        self.model.train()
        self.pool.plastic = False
        self.pool.capture = []
        self.opt.zero_grad(set_to_none=True)
        try:
            loss, metrics, _ = self.model.compute_loss(
                idx, targets, hippo_mem=hippo_mem, ignore_index=ignore_index, ponder_w=ponder_w
            )
            loss.backward()
            cur = torch.zeros(self.pool.N, device=dev)
            for _, sel, _, _ in self.pool.capture:
                cur += sel.sum((0, 1))
        finally:
            self.pool.capture = None

        cur_freq = cur / cur.sum().clamp(min=1.0)
        routed = (self.pool.active > 0) & (self.pool.is_shared == 0)
        score = (cur_freq / (self.bg_freq + self.smf_eps)).masked_fill(~routed, -1e9)
        k_eff = min(self.smf_k, int(routed.sum().item()))
        top_idx = score.topk(k_eff).indices
        keep_vec = torch.zeros(self.pool.N, device=dev)
        keep_vec[top_idx] = 1.0

        torch.nn.utils.clip_grad_norm_(self.mlp_params, 1.0)
        _masked_step(self.opt, self.mlp_params, keep_vec)
        return metrics, keep_vec
