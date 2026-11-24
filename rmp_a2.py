"""A2 架构：潜空间自适应思考 RMP + 极简端到端海马读取 + 原生 SMF 持续学习。

核心设计原则（避免苦涩的教训 & 避免退化成 Transformer）：
1. 主干 100% 零 Self-Attention：`WavePool`（竞争多尺度语境路由 + 慢通道 c + Gated DeltaNet 快记忆 S + 局部 MLP），
   对话流状态严格 O(1)，不膨胀 KV-cache；
2. 潜空间自适应思考（无脚本监督）：同一 `WavePool` 权重复用 K 拍（每拍独立 O(1) 状态槽，训练全序列并行、推理逐 token
   数值严格等价），由 `emit_gate` 产出 PonderNet 停机分布，完全由端到端任务损失 + 几何先验 KL 驱动；
3. 极简端到端海马读取（尊重 Jelassi et al. 2024 信息论硬上限，零手工特征）：
   - 外部情景库由 RMP 自身编码层 + `WavePool` 编码（零 BGE / 零外部 Transformer）；
   - 删掉手工 `k_top=9` MaxSim、手工 `used_mask`、显式指针对齐损失与模板剧本；
   - 每拍思考输出对海马库做单头门控读取（含可学习 NULL 拒答槽），读回内容回注下一拍潜空间输入，
     在无任何跳数脚本监督下自然涌现多跳连想；
4. 原生兼容 `smf_mlp` 局部抗遗忘在线学习（复用 `online._masked_step`）。
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
from pool import WavePool


def _make_decays(G: int, lo: float = 0.45, hi: float = 0.985) -> List[float]:
    if G <= 1:
        return [hi]
    tau_lo, tau_hi = 1.0 / (1.0 - lo), 1.0 / (1.0 - hi)
    return [round(1.0 - 1.0 / (tau_lo * (tau_hi / tau_lo) ** (i / (G - 1))), 5) for i in range(G)]


@dataclass
class HippoMemory:
    """RMP 自身编码出的外部海马记忆包（批量并行）。"""
    keys: torch.Tensor       # (B, L_mem, hippo_dim)
    vals: torch.Tensor       # (B, L_mem, D)
    token_ids: torch.Tensor  # (B, L_mem)
    valid_mask: torch.Tensor # (B, L_mem) bool, True=有效记忆 token


class LatentThinkHippoRMP(nn.Module):
    """A2：零 Self-Attention 路由模块池 + 潜空间自适应思考 + 极简端到端海马读取。"""

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
        pad_id: int = 0,
        max_think: int = 3,
        use_hippo: bool = True,
        hippo_dim: int = 96,
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
        self.pad_id = pad_id
        self.max_think = max(1, int(max_think))
        self.use_hippo = bool(use_hippo)
        self.hippo_dim = hippo_dim

        decays = list(ctx_decay) if ctx_decay is not None else _make_decays(G)
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
        # 遵循 WaveNet / online.py 约定：layers[0]=编码层，layers[1]=WavePool
        self.layers = nn.ModuleList([enc, pool])
        self.win = nn.Linear(M * q, D)

        # 潜空间思考反馈（零初始化：第 k 拍起步等价恒等残差，保证优化稳定）
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

        # 极简端到端海马读取（零手工特征、单头门控读取 + NULL 拒答槽）
        if self.use_hippo:
            self.hippo_k_norm = nn.LayerNorm(D)
            self.hippo_k = nn.Linear(D, hippo_dim, bias=False)
            self.hippo_v = nn.Linear(D, D, bias=False)
            self.hippo_q_norm = nn.LayerNorm(D)
            self.hippo_q = nn.Linear(D, hippo_dim, bias=False)
            self.hippo_out = nn.Linear(D, D)
            nn.init.zeros_(self.hippo_out.weight)
            nn.init.zeros_(self.hippo_out.bias)
            self.null_k = nn.Parameter(torch.randn(1, 1, hippo_dim) * 0.02)
            self.null_v = nn.Parameter(torch.zeros(1, 1, D))
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

    def init_state(self, batch_size: int, device: Optional[torch.device] = None):
        dev = device or self.emb.weight.device
        enc_st = self.layers[0].init_state(batch_size, dev)
        pool_sts = [self.layers[1].init_state(batch_size, dev) for _ in range(self.max_think)]
        return (enc_st, pool_sts)

    def _encode_stream(self, idx: torch.Tensor, enc_st):
        reset = (idx == self.sep_id) if self.sep_id is not None else None
        e = self.emb(idx)
        h0, z0, new_enc_st = self.layers[0](
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
        """用 RMP 自身编码层 + WavePool 批量编码外部情景库（零外部编码器）。"""
        assert self.use_hippo, "use_hippo=False 时不可调用 encode_memory"
        B, Lm = mem_ids.shape
        dev = mem_ids.device
        if mem_valid is None:
            mem_valid = mem_ids != self.pad_id

        # 编码记忆库时不改写路由地址/均衡偏置，保持主干路由统计纯净
        pool = self.layers[1]
        old_plastic = pool.plastic
        pool.plastic = False
        try:
            enc_st = self.layers[0].init_state(B, dev)
            pool_st = pool.init_state(B, dev)
            x0, _, reset = self._encode_stream(mem_ids, enc_st)
            _, x_mem, _ = pool(x0, pool_st, reset, mem_ids)
        finally:
            pool.plastic = old_plastic

        keys = self.hippo_k(self.hippo_k_norm(x_mem))
        vals = self.hippo_v(x_mem)
        return HippoMemory(keys=keys, vals=vals, token_ids=mem_ids, valid_mask=mem_valid)

    def _hippo_read(
        self,
        x_cur: torch.Tensor,
        hippo_mem: Optional[HippoMemory],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """单头门控海马读取（含 NULL 槽）；无记忆库时原样直通。"""
        if (not self.use_hippo) or hippo_mem is None or hippo_mem.keys.shape[1] == 0:
            return x_cur, None, None, None
        B, T, _ = x_cur.shape
        q = self.hippo_q(self.hippo_q_norm(x_cur))  # (B, T, hippo_dim)
        k_ext = torch.cat([hippo_mem.keys, self.null_k.expand(B, 1, -1)], dim=1)  # (B, Lm+1, H)
        v_ext = torch.cat([hippo_mem.vals, self.null_v.expand(B, 1, -1)], dim=1)  # (B, Lm+1, D)
        scores = torch.matmul(q, k_ext.transpose(-2, -1)) / math.sqrt(self.hippo_dim)  # (B, T, Lm+1)
        valid_ext = torch.cat(
            [hippo_mem.valid_mask, torch.ones((B, 1), dtype=torch.bool, device=x_cur.device)],
            dim=1,
        )  # (B, Lm+1)
        scores = scores.masked_fill(~valid_ext[:, None, :], -1e9)
        attn = F.softmax(scores.float(), dim=-1)  # (B, T, Lm+1)
        ctx = torch.matmul(attn.to(v_ext.dtype), v_ext)  # (B, T, D)
        feat = x_cur + self.hippo_out(ctx)
        attn_mem = attn[..., :-1]     # (B, T, Lm)
        attn_null = attn[..., -1:]    # (B, T, 1)
        return feat, ctx, attn_mem, attn_null

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
        # 严格概率归一化：sum(p_ptr) = 1 - attn_null
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
        """批量并行前向（与逐 token 流式推理严格等价）。

        返回:
            log_probs: (B, T, V) 按停机分布边际化的对数概率
            new_state: (enc_st, pool_sts)
            info (当 return_ticks=True): dict(tick_log_probs, halt_probs, exp_steps, null_prob)
        """
        B, T = idx.shape
        dev = idx.device
        if state is None:
            state = self.init_state(B, dev)
        enc_st, pool_sts = state

        x0, new_enc_st, reset = self._encode_stream(idx, enc_st)
        new_pool_sts = []
        tick_log_probs: List[torch.Tensor] = []
        halt_probs_list: List[torch.Tensor] = []
        null_probs_list: List[torch.Tensor] = []

        rem = torch.ones((B, T), dtype=torch.float32, device=dev)
        x_in = x0 + self.tick_emb[0]

        for tick in range(self.max_think):
            _, x_out, st_k = self.layers[1](x_in, pool_sts[tick], reset, idx)
            new_pool_sts.append(st_k)

            feat, ctx, attn_mem, attn_null = self._hippo_read(x_out, hippo_mem)
            lp_k = self._tick_log_probs(feat, ctx, attn_mem, attn_null, hippo_mem)
            tick_log_probs.append(lp_k)
            if attn_null is not None:
                null_probs_list.append(attn_null.squeeze(-1))

            if tick < self.max_think - 1:
                h_k = torch.sigmoid(self.emit_gate(feat).squeeze(-1).float())
                p_k = rem * h_k
                rem = rem * (1.0 - h_k)
                # 下一拍潜空间输入融合本拍思考与海马读回结果（自然支撑无脚本多跳）
                x_in = feat + self.think_proj(self.think_norm(feat)) + self.tick_emb[tick + 1]
            else:
                p_k = rem
            halt_probs_list.append(p_k)

        halt_probs = torch.stack(halt_probs_list, dim=0)  # (K, B, T)
        # 边际对数概率 log(sum_k p_k * exp(lp_k))
        stacked_lp = torch.stack(tick_log_probs, dim=0)  # (K, B, T, V)
        log_w = halt_probs.clamp(min=1e-8).log()[..., None]
        marginal_lp = torch.logsumexp(log_w + stacked_lp, dim=0)

        new_state = (new_enc_st, new_pool_sts)
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
            # gather target log prob
            tok_nll = -lp_k.gather(2, safe_t[..., None]).squeeze(-1)  # (B, T)
            nll_per_tick.append(tok_nll)
        nll_stack = torch.stack(nll_per_tick, dim=0)  # (K, B, T)
        task_loss = ((halt_probs * nll_stack).sum(dim=0) * valid_f).sum() / denom

        # 截断几何先验 KL 正则（只在 max_think > 1 时启用）
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
            kl = (hp * (hp.log() - prior_t.clamp(min=1e-7).log())).sum(dim=0)  # (B, T)
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
        nxt = lp[:, -1].argmax(dim=-1, keepdim=True)  # (B, 1)
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

        # 冻结编码层、路由、海马、读出，只保留波池 MLP 可塑
        for p in model.parameters():
            p.requires_grad_(False)
        self.mlp_params = [self.core.W1, self.core.b1, self.core.W2]
        for p in self.mlp_params:
            p.requires_grad_(True)
        self.pool.plastic = False  # 醒时冻结地址与均衡偏置

        self.opt = torch.optim.AdamW(self.mlp_params, lr=lr, weight_decay=wd)
        self.bg_freq = torch.full((self.pool.N,), 1.0 / self.pool.N, device=model.emb.weight.device)

    @torch.no_grad()
    def calibrate_background(self, batches: Sequence[Tuple[torch.Tensor, Optional[HippoMemory]]]) -> torch.Tensor:
        """在旧域/背景数据上统计各模块路由频率 bg_freq。"""
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
        """执行 1 步严格 SMF 掩码在线更新，返回指标与被选中的模块掩码 keep_vec。"""
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
