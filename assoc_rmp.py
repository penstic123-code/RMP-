"""E3 巩固与抗遗忘核心：用 RMP 路由模块池 + SMF 相对背景冻结替换共享循环回想核。

严格对齐活跃参数预算（docs/design/associative-memory.md · E3）：
- Shared 版（IterativeRecallNet）：scorer_mlp(23->32->16->1)、geo_gate(36->24->4)、
  continue_head(10->24->1)、abstain_head(11->24->1)。
- RMP 版（RMPIterativeRecallNet）：N=16 个专家模块，每次仅激活 k=2 个，每个专家隐藏宽度减半
  （scorer 16->8、其余 12），使每拍活跃计算与活跃参数量与 Shared 版严格对齐。
- 路由完全由冻结句向量 q_0 与归一化地址表 addr[N, D] 的余弦相似度决定（无任何领域硬编码）。
- 持续学习时采用 SMF（Selective Module Freezing）：冻结路由地址，按「新批激活 / (历史背景激活 + eps)」
  仅解冻最专用的 smf_k 个模块，保护旧关联通路不被覆盖。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from assoc_memory import (
    EpisodicEvent,
    EpisodicStore,
    IterativeRecallNet,
    RecallExample,
    SparseAssociativeGraph,
    extract_sparse_tokens,
)


class RMPExpert(nn.Module):
    """单个回想专家：包含一套轻量化 scorer / geo_gate / continue / abstain 子头（宽为共享版的 1/k）。"""

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.geo_gate = nn.Sequential(
            nn.Linear(4 + hidden_dim, 12),
            nn.GELU(),
            nn.Linear(12, 4),
        )
        nn.init.zeros_(self.geo_gate[-1].weight)
        with torch.no_grad():
            self.geo_gate[-1].bias.copy_(torch.tensor([0.20, 0.70, 0.70, -0.10]))

        self.scorer_mlp = nn.Sequential(
            nn.Linear(25, 16),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(16, 8),
            nn.GELU(),
            nn.Linear(8, 1),
        )
        self.continue_head = nn.Sequential(
            nn.Linear(10, 12),
            nn.GELU(),
            nn.Linear(12, 1),
        )
        self.abstain_head = nn.Sequential(
            nn.Linear(13, 12),
            nn.GELU(),
            nn.Linear(12, 1),
        )


class RMPIterativeRecallNet(IterativeRecallNet):
    """RMP 路由模块池版循环回想网络（活跃参数量与共享版 IterativeRecallNet 严格对齐）。"""

    def __init__(
        self,
        emb_dim: int = 512,
        hidden_dim: int = 32,
        low_rank: int = 16,
        n_modules: int = 16,
        top_k: int = 2,
        route_temp: float = 0.12,
    ):
        super().__init__(emb_dim=emb_dim, hidden_dim=hidden_dim, low_rank=low_rank)
        self.n_modules = n_modules
        self.top_k = top_k
        self.route_temp = route_temp

        # 移除父类共享头，替换为 N 个稀疏路由专家（k=2 激活，每专家宽度 1/2 -> 活跃参数对齐）
        del self.geo_gate
        del self.scorer_mlp
        del self.continue_head
        del self.abstain_head

        # 随机正交/均匀初始化归一化路由地址，后续可通过原型锚定或梯度分散
        addr = torch.randn(n_modules, emb_dim)
        addr = F.normalize(addr, dim=-1)
        self.addr = nn.Parameter(addr)
        self.experts = nn.ModuleList([RMPExpert(hidden_dim=hidden_dim) for _ in range(n_modules)])

        # SMF 背景激活统计与当前查询路由缓存
        self.register_buffer("bg_hits", torch.zeros(n_modules))
        self._cur_route: Optional[Tuple[torch.Tensor, torch.Tensor]] = None  # (indices[k], weights[k])
        self._active_mask: Optional[Set[int]] = None

    def route(self, q0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """根据当前查询向量 q0 计算 Top-k 专家索引与归一化权重。"""
        addr_n = F.normalize(self.addr, dim=-1)
        q_n = F.normalize(q0, dim=-1)
        sims = addr_n @ q_n  # (N,)
        top_vals, top_idx = torch.topk(sims, k=self.top_k)
        weights = F.softmax(top_vals / self.route_temp, dim=0)
        self._cur_route = (top_idx, weights)
        return top_idx, weights

    def seed_addresses_from_queries(self, query_embs: Sequence[torch.Tensor]) -> None:
        """用无监督在线查询分布（最远点采样，无标签）初始化/分散路由地址，避免冷启动塌缩。"""
        if not query_embs:
            return
        with torch.no_grad():
            mat = F.normalize(torch.stack([q.detach().cpu() for q in query_embs], dim=0), dim=-1)
            chosen = [0]
            while len(chosen) < min(self.n_modules, len(query_embs)):
                c_mat = mat[chosen]  # (m, D)
                max_sim = (mat @ c_mat.T).max(dim=1).values
                for idx in chosen:
                    max_sim[idx] = 2.0
                nxt = int(torch.argmin(max_sim).item())
                chosen.append(nxt)
            for i, idx in enumerate(chosen):
                self.addr.data[i].copy_(mat[idx].to(self.addr.device))

    def _get_route(self, q0: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if q0 is not None:
            return self.route(q0)
        assert self._cur_route is not None, "Must call route(q0) or init_state(q0) first"
        return self._cur_route

    def init_state(self, q0: torch.Tensor) -> torch.Tensor:
        self.route(q0)
        return super().init_state(q0)

    def update_query(
        self,
        q0: torch.Tensor,
        q_t: torch.Tensor,
        h_t: torch.Tensor,
        e_found: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        top_idx, weights = self._get_route(q0)
        proj = torch.dot(e_found, q0)
        e_perp = e_found - proj * q0
        e_perp_norm = e_perp.norm().clamp(min=1e-6)
        e_perp_unit = e_perp / e_perp_norm

        q_rem = q0 - proj * e_found
        q_rem_norm = q_rem.norm().clamp(min=1e-6)
        q_rem_unit = q_rem / q_rem_norm

        sim_qt_e = torch.dot(e_found, q_t)
        scalars = torch.stack([proj, sim_qt_e, e_perp_norm, q_rem_norm])
        h_in = torch.tanh(self.step_to_h(scalars))
        h_next = self.h_gru(h_in.unsqueeze(0), h_t.unsqueeze(0)).squeeze(0)

        g_in = torch.cat([scalars, h_next], dim=-1)
        alphas = sum(
            weights[j] * self.experts[int(top_idx[j].item())].geo_gate(g_in)
            for j in range(self.top_k)
        )
        dq = alphas[0] * e_found + alphas[1] * e_perp_unit + alphas[2] * q_rem_unit + alphas[3] * q0
        q_next = F.normalize(q_t + dq, dim=-1)
        return q_next, h_next

    def score_candidates(
        self,
        store: EpisodicStore,
        graph: SparseAssociativeGraph,
        q0: torch.Tensor,
        q_t: torch.Tensor,
        q0_tokens: Set[str],
        src_ev: Optional[EpisodicEvent],
        candidates: Sequence[EpisodicEvent],
    ) -> Tuple[torch.Tensor, List[Dict[str, float]]]:
        dev = q0.device
        if not candidates:
            return torch.empty(0, device=dev), []
        top_idx, weights = self._get_route(q0)

        c_embs = torch.stack([c.embedding for c in candidates], dim=0).to(dev)
        sims_q0 = c_embs @ q0
        sims_qt = c_embs @ q_t

        mean_q0, std_q0 = sims_q0.mean(), sims_q0.std(unbiased=False).clamp(min=0.05)
        mean_qt, std_qt = sims_qt.mean(), sims_qt.std(unbiased=False).clamp(min=0.05)
        z_q0 = ((sims_q0 - mean_q0) / std_q0).clamp(-3.0, 3.0)
        z_qt = ((sims_qt - mean_qt) / std_qt).clamp(-3.0, 3.0)

        if src_ev is not None:
            src_emb = src_ev.embedding.to(dev)
            proj = torch.dot(src_emb, q0)
            e_perp = src_emb - proj * q0
            e_perp = e_perp / e_perp.norm().clamp(min=1e-6)
            q_rem = q0 - proj * src_emb
            q_rem = q_rem / q_rem.norm().clamp(min=1e-6)
            sims_prev = c_embs @ src_emb
            sims_perp = c_embs @ e_perp
            sims_qrem = c_embs @ q_rem
            uncovered_q_tokens = q0_tokens - src_ev.tokens
            is_hop2 = 1.0
        else:
            sims_prev = torch.zeros_like(sims_q0)
            sims_perp = torch.zeros_like(sims_q0)
            sims_qrem = torch.zeros_like(sims_q0)
            uncovered_q_tokens = q0_tokens
            is_hop2 = 0.0

        feats, metas = [], []
        for i, c in enumerate(candidates):
            lex_q0 = store.lexical_overlap(q0_tokens, c.tokens)
            lex_prev = store.lexical_overlap(src_ev.tokens, c.tokens) if src_ev is not None else 0.0
            lex_uncov = store.lexical_overlap(uncovered_q_tokens, c.tokens) if src_ev is not None else lex_q0
            mmr_gain = sims_qrem[i] - 0.5 * sims_prev[i] if src_ev is not None else torch.tensor(0.0, device=dev)
            dup_pen = torch.clamp(sims_prev[i] - 0.78, min=0.0) if src_ev is not None else torch.tensor(0.0, device=dev)
            edge_f = graph.pair_edge_features(src_ev, c)
            row = [
                sims_q0[i], sims_qt[i], sims_prev[i], sims_perp[i], sims_qrem[i],
                mmr_gain, dup_pen,
                torch.tensor(lex_q0, device=dev),
                torch.tensor(lex_prev, device=dev),
                torch.tensor(lex_uncov, device=dev),
                z_q0[i], z_qt[i],
                torch.tensor(is_hop2, device=dev),
                *[torch.tensor(x, device=dev) for x in edge_f],
            ]
            feats.append(torch.stack(row))
            metas.append({
                "sim_q0": float(sims_q0[i].item()),
                "sim_qt": float(sims_qt[i].item()),
                "sim_prev": float(sims_prev[i].item()),
                "sim_qrem": float(sims_qrem[i].item()),
                "lex_q0": float(lex_q0),
                "lex_prev": float(lex_prev),
                "lex_uncov": float(lex_uncov),
                "has_edge": float(edge_f[11]),
                "active": float(edge_f[8]),
                "superseded": float(edge_f[9]),
            })
        feat_mat = torch.stack(feats, dim=0)
        raw = sum(
            weights[j] * self.experts[int(top_idx[j].item())].scorer_mlp(feat_mat).squeeze(-1)
            for j in range(self.top_k)
        )
        logits = 5.0 * torch.tanh(raw / 3.0) * F.softplus(self.score_temp)
        return logits, metas

    def continue_logit(self, feat_vec: torch.Tensor) -> torch.Tensor:
        top_idx, weights = self._get_route()
        return sum(
            weights[j] * self.experts[int(top_idx[j].item())].continue_head(feat_vec).squeeze(-1)
            for j in range(self.top_k)
        )

    def abstain_logit(self, feat_vec: torch.Tensor) -> torch.Tensor:
        top_idx, weights = self._get_route()
        return sum(
            weights[j] * self.experts[int(top_idx[j].item())].abstain_head(feat_vec).squeeze(-1)
            for j in range(self.top_k)
        )

    @torch.no_grad()
    def profile_hits(self, encoder, queries: Sequence[str]) -> torch.Tensor:
        """统计一批查询在各模块上的路由激活分布。"""
        dev = self.addr.device
        hits = torch.zeros(self.n_modules, device=dev)
        for q in queries:
            q0 = encoder.encode_one(q).to(dev)
            top_idx, weights = self.route(q0)
            for j in range(self.top_k):
                hits[int(top_idx[j].item())] += weights[j]
        if hits.sum() > 0:
            hits = hits / hits.sum()
        return hits

    def apply_smf_freeze(self, new_hits: torch.Tensor, smf_k: int = 3) -> List[int]:
        """SMF 相对背景选模块：冻结路由与共享基础层，仅解冻相对背景最专用的 smf_k 个专家。"""
        rel = new_hits / (self.bg_hits + 0.03)
        selected = torch.topk(rel, k=min(smf_k, self.n_modules)).indices.tolist()
        sel_set = set(selected)
        self._active_mask = sel_set

        # 冻结路由地址与基础标量层，防止漂移旧批次路由
        self.addr.requires_grad_(False)
        self.h_init.requires_grad_(False)
        for p in self.step_to_h.parameters():
            p.requires_grad_(False)
        for p in self.h_gru.parameters():
            p.requires_grad_(False)
        self.score_temp.requires_grad_(False)

        for i, exp in enumerate(self.experts):
            req = i in sel_set
            for p in exp.parameters():
                p.requires_grad_(req)
        return selected

    def record_bg_hits(self, hits: torch.Tensor) -> None:
        self.bg_hits.add_(hits.to(self.bg_hits.device))

    def unfreeze_all(self) -> None:
        self._active_mask = None
        for p in self.parameters():
            p.requires_grad_(True)
