"""A3：纯粹遵循《苦涩的教训》的潜空间搜索 RMP（LatentSearchWaveNet）。

设计铁律：
1. 100% 零 Self-Attention、零 KV-cache、零手工特征（无 sym_prev/sym_ctx/copy_gate/MaxSim/脚本思考监督）；
2. 只押注可随算力扩展的两类通用元方法：
   - 通用学习（Learning）：`WavePool` 竞争多尺度路由模块池（慢通道 c + Gated DeltaNet 快记忆 S + 局部 MLP）
     + 数据驱动的 `smf_mlp` 局部抗遗忘在线更新；
   - 通用搜索（Search）：同一 `WavePool` 在 O(1) 恒定状态内迭代 K 拍潜空间搜索 + PonderNet 端到端自适应停机，
     纯由标准 Next-Token Prediction 交叉熵 + 几何先验 KL 驱动；
3. 并行训练（全序列 B×T）与逐 token 流式推理（T=1）严格数值等价，天然支持 bf16 与 fla 融合扫描内核。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dspn_text import Layer
from online import _masked_step
from pool import WavePool


def make_decays(G: int, lo: float = 0.5, hi: float = 0.99) -> List[float]:
    if G <= 1:
        return [hi]
    tau_lo, tau_hi = 1.0 / (1.0 - lo), 1.0 / (1.0 - hi)
    return [round(1.0 - 1.0 / (tau_lo * (tau_hi / tau_lo) ** (i / (G - 1))), 5) for i in range(G)]


class LatentSearchWaveNet(nn.Module):
    """纯粹通用潜空间搜索 RMP：零 Self-Attention、零手工先验、严格 O(1) 状态、原生兼容 SMF。"""

    def __init__(
        self,
        vocab: int = 256,
        D: int = 128,
        M: int = 2,
        d: int = 96,
        m: int = 16,
        hidden: int = 192,
        q: int = 64,
        N: int = 48,
        k: int = 4,
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
        sep_id: Optional[int] = 30,
        max_think: int = 3,
        tie_emb: bool = True,
    ):
        super().__init__()
        self.vocab = vocab
        self.D = D
        self.M = M
        self.G = G
        self.N = N
        self.k = k
        self.sep_id = sep_id
        self.max_think = max(1, int(max_think))

        decays = list(ctx_decay) if ctx_decay is not None else make_decays(G)
        self.emb = nn.Embedding(vocab, D)
        nn.init.normal_(self.emb.weight, std=0.02)

        enc = Layer(M, D, d, m, hidden, q, rank=lora_rank, beta_max=beta_max)
        pool = WavePool(
            N=N, k=k, G=G, D=D, d=pd, m=pm, hidden=phidden, qr=qr,
            beta_max=beta_max, temp=temp, init="spread", rule="ctx",
            wave_query=True, wave_ctx=True, ctx_decay=decays,
            tau_prior=tau_prior, bottleneck=bottleneck, lora_rank=lora_rank,
            shared=shared,
        )
        # 兼容 online.py / SMF 约定：layers[0]=编码层，layers[1]=WavePool
        self.layers = nn.ModuleList([enc, pool])
        self.win = nn.Linear(M * q, D)

        # 潜空间搜索反馈（零初始化，确保高 K 拍起步与 K=1 完全平滑一致）
        self.tick_emb = nn.Parameter(torch.zeros(self.max_think, D))
        self.think_norm = nn.LayerNorm(D)
        self.think_proj = nn.Linear(D, D)
        nn.init.zeros_(self.think_proj.weight)
        nn.init.zeros_(self.think_proj.bias)

        # 自适应停机门（PonderNet，完全端到端学习，无任何难度/思考监督）
        self.halt_gate = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D // 2),
            nn.GELU(),
            nn.Linear(D // 2, 1),
        )

        self.read_norm = nn.LayerNorm(D)
        self.readout = nn.Identity() if tie_emb else nn.Linear(D, D)
        if not tie_emb:
            nn.init.zeros_(self.readout.weight)
        self.bias = nn.Parameter(torch.zeros(vocab))

    def init_state(self, batch_size: int, device: Optional[torch.device] = None):
        dev = device or self.emb.weight.device
        enc_st = self.layers[0].init_state(batch_size, dev)
        pool_sts = [self.layers[1].init_state(batch_size, dev) for _ in range(self.max_think)]
        return (enc_st, pool_sts)

    def forward(
        self,
        idx: torch.Tensor,
        state=None,
        return_ticks: bool = False,
    ):
        """批量并行前向（与逐 token 流式推理严格数值等价）。"""
        B, T = idx.shape
        dev = idx.device
        if state is None:
            state = self.init_state(B, dev)
        enc_st, pool_sts = state

        reset = (idx == self.sep_id) if self.sep_id is not None else None
        e = self.emb(idx)
        _, z0, new_enc_st = self.layers[0](
            e[:, :, None].expand(B, T, self.M, self.D),
            enc_st,
            reset,
        )
        x0 = self.win(z0.flatten(2))

        new_pool_sts = []
        tick_logits: List[torch.Tensor] = []
        halt_probs_list: List[torch.Tensor] = []

        rem = torch.ones((B, T), dtype=torch.float32, device=dev)
        x_in = x0 + self.tick_emb[0].to(x0.dtype)

        for tick in range(self.max_think):
            _, x_out, st_k = self.layers[1](x_in, pool_sts[tick], reset, idx)
            new_pool_sts.append(st_k)

            y = self.readout(self.read_norm(x_out))
            lg = F.linear(y, self.emb.weight.to(y.dtype), self.bias.to(y.dtype))
            tick_logits.append(lg)

            if tick < self.max_think - 1:
                h_k = torch.sigmoid(self.halt_gate(x_out).squeeze(-1).float())
                p_k = rem * h_k
                rem = rem * (1.0 - h_k)
                x_in = x_out + self.think_proj(self.think_norm(x_out)) + self.tick_emb[tick + 1].to(x_out.dtype)
            else:
                p_k = rem
            halt_probs_list.append(p_k)

        halt_probs = torch.stack(halt_probs_list, dim=0)  # (K, B, T)
        if self.max_think == 1:
            marginal_lp = F.log_softmax(tick_logits[0].float(), dim=-1)
        else:
            stacked_lp = torch.stack([F.log_softmax(lg.float(), dim=-1) for lg in tick_logits], dim=0)
            log_w = halt_probs.clamp(min=1e-8).log()[..., None]
            marginal_lp = torch.logsumexp(log_w + stacked_lp, dim=0)

        new_state = (new_enc_st, new_pool_sts)
        if not return_ticks:
            return marginal_lp, new_state

        steps_vec = torch.arange(1, self.max_think + 1, dtype=torch.float32, device=dev)[:, None, None]
        exp_steps = (halt_probs * steps_vec).sum(dim=0)
        return marginal_lp, new_state, {
            "tick_logits": tick_logits,
            "halt_probs": halt_probs,
            "exp_steps": exp_steps,
        }

    def compute_loss(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor,
        state=None,
        ponder_w: float = 0.01,
        prior_p: float = 0.5,
    ) -> Tuple[torch.Tensor, Dict[str, float], tuple]:
        """纯 Next-Token Prediction 端到端 PonderNet 损失。"""
        marginal_lp, new_state, info = self.forward(idx, state=state, return_ticks=True)
        halt_probs = info["halt_probs"]  # (K, B, T)

        if self.max_think == 1:
            marginal_nll = F.cross_entropy(info["tick_logits"][0].float().reshape(-1, self.vocab), targets.reshape(-1))
            kl_loss = torch.tensor(0.0, device=idx.device)
        else:
            # 训练与推理严格对齐：直接最小化边际对数概率 NLL（与评测 bpc 完全一致）
            marginal_nll = F.nll_loss(marginal_lp.reshape(-1, self.vocab), targets.reshape(-1))
            if ponder_w > 0.0:
                K = self.max_think
                prior = []
                r = 1.0
                for k in range(K):
                    pk = r * prior_p if k < K - 1 else r
                    prior.append(pk)
                    r *= (1.0 - prior_p)
                prior_t = torch.tensor(prior, dtype=torch.float32, device=idx.device)[:, None, None]
                hp = halt_probs.clamp(min=1e-7)
                kl_loss = (hp * (hp.log() - prior_t.clamp(min=1e-7).log())).sum(dim=0).mean()
            else:
                kl_loss = torch.tensor(0.0, device=idx.device)

        total = marginal_nll + ponder_w * kl_loss
        metrics = {
            "loss": float(total.item()),
            "task_loss": float(marginal_nll.item()),
            "bpc": float(marginal_nll.item() / math.log(2)),
            "exp_steps": float(info["exp_steps"].mean().item()),
        }
        return total, metrics, new_state


class SearchSMFOnlineLearner:
    """面向 LatentSearchWaveNet 的原生 SMF (`smf_mlp`) 在线持续学习器。"""

    def __init__(
        self,
        model: LatentSearchWaveNet,
        lr: float = 1e-3,
        smf_k: int = 12,
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
    def calibrate_background(self, seq_bytes: bytes, chunk: int = 256) -> torch.Tensor:
        dev = self.model.emb.weight.device
        ids = torch.tensor(list(seq_bytes), dtype=torch.long, device=dev)
        cnt = torch.zeros(self.pool.N, device=dev)
        self.model.eval()
        self.pool.capture = []
        st = None
        try:
            for i in range(0, len(ids) - 1, chunk):
                blk = ids[i : i + chunk][None]
                if blk.shape[1] < 2:
                    continue
                self.pool.capture.clear()
                _, st = self.model(blk, state=st)
                for _, sel, _, _ in self.pool.capture:
                    cnt += sel.sum((0, 1))
        finally:
            self.pool.capture = None
        self.bg_freq = cnt / cnt.sum().clamp(min=1.0)
        return self.bg_freq

    def step_chunk(self, blk_ids: torch.Tensor, state=None, ponder_w: float = 0.0):
        """对单段文本块执行 1 步严格 SMF 掩码在线更新。blk_ids: (1, L)。"""
        dev = blk_ids.device
        self.model.train()
        self.pool.plastic = False
        self.pool.capture = []
        self.opt.zero_grad(set_to_none=True)
        try:
            loss, metrics, new_st = self.model.compute_loss(
                blk_ids[:, :-1], blk_ids[:, 1:], state=state, ponder_w=ponder_w
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
        return metrics, _detach_state(new_st), keep_vec


def _detach_state(st):
    if isinstance(st, torch.Tensor):
        return st.detach()
    if isinstance(st, (tuple, list)):
        return type(st)(_detach_state(x) for x in st)
    return st
