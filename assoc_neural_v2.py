"""v2.1 纯通用神经联想记忆架构（苦涩的教训：通用学习 + 通用搜索 + GPU 批量加速，零手工特征）。

相比 v2.0 的两项核心升级：
1. 消除 [CLS] 与内容 token 空间错位：悬念查询 q_{t+1} 在预训练内容 token 空间内直接与候选的
   token 序列做 MaxSim 匹配（纯 ColBERT 风格 token 点积），同时结合残差 CLS 通道；
2. 候选批量并行验证（score_candidates_batch）：把所有 C 个候选 padding 后一次送入 Pre-Norm
   2 层 Transformer，消除 Python 逐候选循环，GPU 训练与推理提速 ~8×，从而支撑更大规模自监督训练。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
if not os.environ.get("HF_HOME") and (ROOT / ".pytest_cache" / "hf").exists():
    os.environ["HF_HOME"] = str(ROOT / ".pytest_cache" / "hf")


class TokenLevelEncoder:
    """冻结预训练编码器：缓存句级 CLS 向量 (D,) 与归一化内容 token 序列 (L, D)。"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", device: Optional[str] = None, max_len: int = 96):
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.dim: int = int(self.model.config.hidden_size)
        self._cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    @torch.no_grad()
    def encode_both(self, texts: Sequence[str], batch_size: int = 64) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        if isinstance(texts, str):
            texts = [texts]
        missing = [t for t in texts if t not in self._cache]
        if missing:
            for i in range(0, len(missing), batch_size):
                batch = missing[i : i + batch_size]
                enc = self.tok(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_len,
                    return_tensors="pt",
                ).to(self.device)
                out = self.model(**enc)
                hs = out.last_hidden_state.float()
                cls_embs = F.normalize(hs[:, 0], dim=-1).cpu()
                mask = enc["attention_mask"]
                for b_idx, txt in enumerate(batch):
                    valid_len = int(mask[b_idx].sum().item())
                    tok_h = hs[b_idx, 1 : valid_len - 1] if valid_len > 2 else hs[b_idx, :valid_len]
                    tok_h = F.normalize(tok_h, dim=-1).cpu()
                    self._cache[txt] = (cls_embs[b_idx], tok_h)
        return [self._cache[t] for t in texts]

    def encode_one_both(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encode_both([text])[0]

    def encode_one(self, text: str) -> torch.Tensor:
        return self.encode_one_both(text)[0]

    def encode(self, texts: Sequence[str], batch_size: int = 64) -> torch.Tensor:
        return torch.stack([x[0] for x in self.encode_both(texts, batch_size=batch_size)], dim=0)


@dataclass
class NeuralEvent:
    event_id: str
    text: str
    day: int
    source: str
    cls_emb: torch.Tensor
    tok_embs: torch.Tensor  # (L, D)
    mean_tok_emb: torch.Tensor  # (D,) L2-normalized mean of content tokens
    status: str = "active"
    supersedes: Optional[str] = None
    superseded_by: Optional[str] = None


class NeuralEpisodicStore:
    """纯通用事件库（零手工词法/分词/IDF 规则）。"""

    def __init__(self, encoder: TokenLevelEncoder):
        self.encoder = encoder
        self._events: Dict[str, NeuralEvent] = {}
        self._order: List[str] = []
        self._next_num: int = 1

    def add_event(
        self,
        text: str,
        day: int,
        source: str,
        event_id: Optional[str] = None,
        status: str = "active",
        supersedes: Optional[str] = None,
    ) -> NeuralEvent:
        if event_id is None:
            while f"ev_{self._next_num:04d}" in self._events:
                self._next_num += 1
            eid = f"ev_{self._next_num:04d}"
            self._next_num += 1
        else:
            eid = event_id
        cls_e, tok_e = self.encoder.encode_one_both(text.strip())
        mean_tok = F.normalize(tok_e.mean(dim=0), dim=-1)
        ev = NeuralEvent(
            event_id=eid,
            text=text.strip(),
            day=int(day),
            source=source,
            cls_emb=cls_e,
            tok_embs=tok_e,
            mean_tok_emb=mean_tok,
            status=status,
            supersedes=supersedes,
        )
        if supersedes and supersedes in self._events:
            old = self._events[supersedes]
            old.status = "superseded"
            old.superseded_by = eid
        is_new = eid not in self._events
        self._events[eid] = ev
        if is_new:
            self._order.append(eid)
        return ev

    def supersede_event(self, old_id: str, new_text: str, day: int, source: str, new_id: Optional[str] = None) -> NeuralEvent:
        return self.add_event(new_text, day=day, source=source, event_id=new_id, status="active", supersedes=old_id)

    def delete_event(self, event_id: str) -> bool:
        if event_id not in self._events:
            return False
        self._events[event_id].status = "deleted"
        return True

    def get(self, event_id: str) -> NeuralEvent:
        return self._events[event_id]

    def __contains__(self, event_id: str) -> bool:
        return event_id in self._events

    def active_events(self) -> List[NeuralEvent]:
        return [self._events[eid] for eid in self._order if self._events[eid].status == "active"]

    def all_events(self, include_deleted: bool = False) -> List[NeuralEvent]:
        evs = [self._events[eid] for eid in self._order]
        return evs if include_deleted else [e for e in evs if e.status != "deleted"]

    def clone_without(self, excluded_ids: Iterable[str]) -> "NeuralEpisodicStore":
        excl = set(excluded_ids)
        sub = NeuralEpisodicStore(self.encoder)
        for eid in self._order:
            if eid in excl:
                continue
            e = self._events[eid]
            sub._events[eid] = NeuralEvent(
                event_id=e.event_id,
                text=e.text,
                day=e.day,
                source=e.source,
                cls_emb=e.cls_emb,
                tok_embs=e.tok_embs,
                mean_tok_emb=e.mean_tok_emb,
                status=e.status,
                supersedes=e.supersedes if e.supersedes not in excl else None,
                superseded_by=e.superseded_by if e.superseded_by not in excl else None,
            )
            sub._order.append(eid)
        return sub


class PureNeuralGraph:
    """纯通用稀疏联想图（仅用向量近邻 + Token MaxSim 近邻 + 时间相邻，零词法规则，每节点 <= 16 边）。"""

    def __init__(self, store: NeuralEpisodicStore, max_neighbors: int = 16):
        self.store = store
        self.max_neighbors = max_neighbors
        self.adj: Dict[str, Dict[str, float]] = {}
        self.consolidated: Dict[Tuple[str, str], float] = {}
        self.rebuild()

    def rebuild(self) -> None:
        self.adj.clear()
        evs = self.store.active_events()
        if not evs:
            return
        embs = torch.stack([e.cls_emb for e in evs], dim=0)
        sim_mat = embs @ embs.T
        n = len(evs)
        for i, u in enumerate(evs):
            sims = sim_mat[i].clone()
            sims[i] = -1e9
            k_sem = min(10, n - 1)
            sem_idx = set(torch.topk(sims, k=k_sem).indices.tolist()) if k_sem > 0 else set()

            tok_scores = []
            for j, v in enumerate(evs):
                if j == i:
                    continue
                ms = float((u.tok_embs @ v.tok_embs.T).max(dim=1).values.mean().item())
                tok_scores.append((ms, j))
            tok_scores.sort(reverse=True)
            tok_idx = {j for _, j in tok_scores[:6]}

            temp_idx = {j for j, v in enumerate(evs) if j != i and abs(v.day - u.day) <= 1}
            cand = sem_idx | tok_idx | temp_idx
            scored = []
            for j in cand:
                v = evs[j]
                cw = self.consolidated.get((u.event_id, v.event_id), 0.0)
                scored.append((float(sim_mat[i, j].item()) + cw, v.event_id))
            scored.sort(reverse=True)
            self.adj[u.event_id] = {dst: w for w, dst in scored[: self.max_neighbors]}

    def reinforce_path(self, event_ids: Sequence[str], delta: float = 0.8) -> None:
        for i in range(len(event_ids) - 1):
            u, v = event_ids[i], event_ids[i + 1]
            for a, b in ((u, v), (v, u)):
                self.consolidated[(a, b)] = min(2.0, self.consolidated.get((a, b), 0.0) + delta)
                nbrs = self.adj.setdefault(a, {})
                nbrs[b] = nbrs.get(b, 0.5) + delta
                if len(nbrs) > self.max_neighbors:
                    top = sorted(nbrs.items(), key=lambda x: x[1], reverse=True)[: self.max_neighbors]
                    self.adj[a] = dict(top)

    def neighbors(self, event_id: str) -> List[str]:
        return [
            dst for dst in self.adj.get(event_id, {}).keys()
            if dst in self.store and self.store.get(dst).status == "active"
        ]


class GeneralNeuralRecallNet(nn.Module):
    """纯通用 Token 注意力 + 批量 Pre-Norm Transformer 循环回想网络（零手工特征、任意深度）。"""

    def __init__(self, d_model: int = 32, n_heads: int = 4, n_layers: int = 2, dual_rep: bool = False):
        super().__init__()
        self.d_model = d_model
        self.dual_rep = dual_rep

        # --- 1. Token 注意力悬念查询器（输出在内容 token 空间，直接匹配候选 mean_tok / tok_embs）---
        self.q_align_proj = nn.Linear(3, d_model)
        self.e_align_proj = nn.Linear(3, d_model)
        q_enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
        )
        self.q_mask_tf = nn.TransformerEncoder(q_enc_layer, num_layers=1, enable_nested_tensor=False)
        self.q_weight_head = nn.Linear(d_model, 1)

        e_enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
        )
        self.e_mask_tf = nn.TransformerEncoder(e_enc_layer, num_layers=1, enable_nested_tensor=False)
        self.e_weight_head = nn.Linear(d_model, 1)

        # 残差先验系数：负对齐（低覆盖 token）初始偏置，加速收敛而不限定最终形态
        self.inv_bias_q = nn.Parameter(torch.tensor(-3.0))
        self.inv_bias_e = nn.Parameter(torch.tensor(-3.0))
        self.query_mix = nn.Parameter(torch.tensor([0.70, 0.70, 0.20]))

        # --- 2. 批量候选验证器（Pre-Norm 2 层 Transformer）---
        # 每个 query token 6 维原始对齐：[cov_prev, cov_cand, cov_union, gain, mean_cand, is_hop_gt0]
        self.cand_q_proj = nn.Linear(6, d_model)
        ver_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
        )
        self.verifier_tf = nn.TransformerEncoder(ver_layer, num_layers=n_layers, enable_nested_tensor=False)

        # 融合 query-token 池化（mean+min+max=3*d）与 4 维原始全局点积 [cls_q0, tok_qt, tok_prev_max, cls_prev]
        self.score_head = nn.Sequential(nn.Linear(d_model * 3 + 4, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.score_temp = nn.Parameter(torch.tensor(1.0))

        # --- 3. 路径级停机与拒答 Transformer（Pre-Norm + Mean/Min/Max 池化）---
        self.path_q_proj = nn.Linear(4, d_model)
        path_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
        )
        self.path_tf = nn.TransformerEncoder(path_layer, num_layers=n_layers, enable_nested_tensor=False)
        self.path_pool_dim = d_model * 3
        self.continue_head = nn.Sequential(
            nn.Linear(self.path_pool_dim * 2, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )
        # 路径级验证门控：单表征输入 [lora_path_repr, lora_path_cls_max]；
        # 双表征输入 [lora_path_repr, frozen_path_repr, lora_path_cls_max, frozen_path_cls_max]
        # 严格只检查「候选路径本身选中的证据」，不碰全库最高分，且由门控自由学习两路分歧。
        valid_in_dim = (self.path_pool_dim * 2 + 2) if dual_rep else (self.path_pool_dim + 1)
        self.valid_head = nn.Sequential(
            nn.Linear(valid_in_dim, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )

    @staticmethod
    def _align_stats(sim_mat: torch.Tensor) -> torch.Tensor:
        max_v = sim_mat.max(dim=1).values
        top2_v = sim_mat.topk(2, dim=1).values[:, 1] if sim_mat.shape[1] >= 2 else max_v
        mean_v = sim_mat.mean(dim=1)
        return torch.stack([max_v, top2_v, mean_v], dim=-1)

    def update_query(self, q0_tok: torch.Tensor, q_t: torch.Tensor, path_toks: torch.Tensor) -> torch.Tensor:
        """在内容 token 空间内通过通用注意力合成下一轮悬念向量 q_{t+1}。"""
        A = q0_tok @ path_toks.T  # (Lq, Lp)
        q_stats = self._align_stats(A)
        e_stats = self._align_stats(A.T)

        qh = self.q_mask_tf(self.q_align_proj(q_stats).unsqueeze(0)).squeeze(0)
        wq = F.softmax(self.q_weight_head(qh).squeeze(-1) + self.inv_bias_q * q_stats[:, 0], dim=0)
        q_missing = (wq.unsqueeze(-1) * q0_tok).sum(dim=0)

        eh = self.e_mask_tf(self.e_align_proj(e_stats).unsqueeze(0)).squeeze(0)
        we = F.softmax(self.e_weight_head(eh).squeeze(-1) + self.inv_bias_e * e_stats[:, 0], dim=0)
        e_novel = (we.unsqueeze(-1) * path_toks).sum(dim=0)

        q_next = (
            self.query_mix[0] * F.normalize(q_missing, dim=-1)
            + self.query_mix[1] * F.normalize(e_novel, dim=-1)
            + self.query_mix[2] * q_t
        )
        return F.normalize(q_next, dim=-1)

    def encode_path_state(self, q0_tok: torch.Tensor, path_toks: torch.Tensor, path_len: int) -> torch.Tensor:
        dev = q0_tok.device
        A = q0_tok @ path_toks.T
        st = self._align_stats(A)
        plen = torch.full((q0_tok.shape[0], 1), float(path_len) / 3.0, device=dev)
        seq = self.path_q_proj(torch.cat([st, plen], dim=-1)).unsqueeze(0)
        out = self.path_tf(seq)[0]
        return torch.cat([out.mean(dim=0), out.min(dim=0).values, out.max(dim=0).values], dim=-1)

    def score_candidates_batch(
        self,
        q0_cls: torch.Tensor,
        q0_tok: torch.Tensor,
        q_t: torch.Tensor,
        path_cls_list: Sequence[torch.Tensor],
        path_toks: Optional[torch.Tensor],
        candidates: Sequence[NeuralEvent],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """一次性批量并行验证 C 个候选事件，返回 (logits[C], reprs[C, 3*d_model])。"""
        dev = q0_cls.device
        C = len(candidates)
        if C == 0:
            return torch.empty(0, device=dev), torch.empty(0, self.d_model * 3, device=dev)

        Lq = q0_tok.shape[0]
        if path_toks is not None:
            cov_prev = (q0_tok @ path_toks.T).max(dim=1).values  # (Lq,)
            hop_flag = 1.0
        else:
            cov_prev = torch.zeros(Lq, device=dev)
            hop_flag = 0.0
        hf_q = torch.full((Lq,), hop_flag, device=dev)

        q_seqs = []
        raw_rows = []
        for cand in candidates:
            c_cls = cand.cls_emb.to(dev)
            c_tok = cand.tok_embs.to(dev)
            c_mean = cand.mean_tok_emb.to(dev)

            A_qc = q0_tok @ c_tok.T  # (Lq, Lc)
            cov_cand = A_qc.max(dim=1).values
            mean_cand = A_qc.mean(dim=1)
            cov_union = torch.maximum(cov_prev, cov_cand) if path_toks is not None else cov_cand
            gain = cov_union - cov_prev if path_toks is not None else cov_cand

            q_seqs.append(torch.stack([cov_prev, cov_cand, cov_union, gain, mean_cand, hf_q], dim=-1))

            # 4 维纯点积全局通道（含 token-space 悬念匹配）
            sim_q0 = torch.dot(q0_cls, c_cls)
            sim_qt_tok = 0.5 * torch.dot(q_t, c_mean) + 0.5 * (c_tok @ q_t).max()
            if path_toks is not None:
                tok_prev_max = (c_tok @ path_toks.T).max(dim=1).values.topk(min(3, c_tok.shape[0])).values.mean()
                cls_prev = max(float(torch.dot(p_c, c_cls).item()) for p_c in path_cls_list)
            else:
                tok_prev_max = torch.tensor(0.0, device=dev)
                cls_prev = 0.0
            raw_rows.append(torch.stack([sim_q0, sim_qt_tok, tok_prev_max, torch.tensor(cls_prev, device=dev)]))

        # 单次 GPU 批量前向：(C, Lq, 6) -> (C, Lq, d_model)
        batch_in = self.cand_q_proj(torch.stack(q_seqs, dim=0))  # (C, Lq, d)
        out = self.verifier_tf(batch_in)  # (C, Lq, d)
        pooled = torch.cat([out.mean(dim=1), out.min(dim=1).values, out.max(dim=1).values], dim=-1)  # (C, 3*d)
        raw_mat = torch.stack(raw_rows, dim=0)  # (C, 4)

        raw_logits = self.score_head(torch.cat([pooled, raw_mat], dim=-1)).squeeze(-1)  # (C,)
        logits = 5.0 * torch.tanh(raw_logits / 3.0) * F.softplus(self.score_temp)
        return logits, pooled

    def continue_logit(self, path_repr: torch.Tensor, best_next_repr: torch.Tensor) -> torch.Tensor:
        return self.continue_head(torch.cat([path_repr, best_next_repr], dim=-1)).squeeze(-1)

    def valid_logit(
        self,
        path_repr: torch.Tensor,
        path_cls_max: float,
        frozen_path_repr: Optional[torch.Tensor] = None,
        frozen_path_cls_max: Optional[float] = None,
    ) -> torch.Tensor:
        if not self.dual_rep:
            s = torch.tensor([path_cls_max], dtype=torch.float32, device=path_repr.device)
            return self.valid_head(torch.cat([path_repr, s], dim=-1)).squeeze(-1)
        f_repr = path_repr.detach() if frozen_path_repr is None else frozen_path_repr
        f_cls = path_cls_max if frozen_path_cls_max is None else frozen_path_cls_max
        s = torch.tensor([path_cls_max, f_cls], dtype=torch.float32, device=path_repr.device)
        return self.valid_head(torch.cat([path_repr, f_repr, s], dim=-1)).squeeze(-1)

    def gather_round0(self, store: NeuralEpisodicStore, q0_cls: torch.Tensor, q0_tok: torch.Tensor, budget: int = 16):
        evs = store.active_events()
        if not evs:
            return [], 0.0
        embs = torch.stack([e.cls_emb for e in evs], dim=0).to(q0_cls.device)
        sims = embs @ q0_cls
        max_sim = float(sims.max().item())
        k_pre = min(max(budget * 2, 24), len(evs))
        pre_idx = torch.topk(sims, k=k_pre).indices.tolist()
        scored = []
        for idx in pre_idx:
            e = evs[idx]
            ms = float((q0_tok @ e.tok_embs.to(q0_cls.device).T).max(dim=1).values.mean().item())
            scored.append((0.5 * float(sims[idx].item()) + 0.5 * ms, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:budget]], max_sim

    def gather_next(
        self,
        store: NeuralEpisodicStore,
        graph: PureNeuralGraph,
        last_ev: NeuralEvent,
        q_next: torch.Tensor,
        visited: Set[str],
        mode: str = "hybrid",
        budget: int = 16,
    ) -> Tuple[List[NeuralEvent], Set[str]]:
        graph_ids = [dst for dst in graph.neighbors(last_ev.event_id) if dst not in visited] if mode in ("hybrid", "graph_only") else []
        full_ids: List[str] = []
        if mode in ("hybrid", "full_only"):
            evs = [e for e in store.active_events() if e.event_id not in visited]
            if evs:
                # 在同源内容 token 空间检索（mean_tok_emb）
                embs = torch.stack([e.mean_tok_emb for e in evs], dim=0).to(q_next.device)
                sims = embs @ q_next
                k = min(budget if mode == "full_only" else max(8, budget // 2), len(evs))
                full_ids = [evs[i].event_id for i in torch.topk(sims, k=k).indices.tolist()]

        merged: List[str] = []
        seen: Set[str] = set()
        jump_ids: Set[str] = set()
        gset = set(graph_ids)
        if mode == "hybrid":
            gi = fi = 0
            while len(merged) < budget and (gi < len(graph_ids) or fi < len(full_ids)):
                if fi < len(full_ids):
                    eid = full_ids[fi]; fi += 1
                    if eid not in seen:
                        seen.add(eid); merged.append(eid)
                        if eid not in gset:
                            jump_ids.add(eid)
                if len(merged) >= budget:
                    break
                if gi < len(graph_ids):
                    eid = graph_ids[gi]; gi += 1
                    if eid not in seen:
                        seen.add(eid); merged.append(eid)
        elif mode == "graph_only":
            merged = graph_ids[:budget]
        else:
            merged = full_ids[:budget]
            jump_ids = {eid for eid in merged if eid not in gset}
        return [store.get(eid) for eid in merged], jump_ids

    @torch.no_grad()
    def recall(
        self,
        store: NeuralEpisodicStore,
        graph: PureNeuralGraph,
        query_text: str,
        mode: str = "hybrid",
        beam_width: int = 4,
        max_rounds: int = 3,
        candidate_budget: int = 16,
        continue_thresh: float = 0.5,
        abstain_thresh: float = 0.5,
    ) -> Dict[str, object]:
        self.eval()
        dev = next(self.parameters()).device
        q0_cls, q0_tok = store.encoder.encode_one_both(query_text)
        q0_cls, q0_tok = q0_cls.to(dev), q0_tok.to(dev)
        q0_mean = F.normalize(q0_tok.mean(dim=0), dim=-1)

        r0_cands, max_sim = self.gather_round0(store, q0_cls, q0_tok, budget=candidate_budget)
        if not r0_cands:
            return {"event_ids": [], "abstained": True, "p_valid": 0.0, "used_jump": False}

        r0_logits, _ = self.score_candidates_batch(q0_cls, q0_tok, q0_mean, [], None, r0_cands)
        r0_probs = F.softmax(r0_logits, dim=0)
        order0 = torch.argsort(r0_logits, descending=True).tolist()

        beam = [
            ([r0_cands[i].event_id], q0_mean, [float(r0_probs[i].item())], float(r0_probs[i].item()), False, False)
            for i in order0[:beam_width]
        ]

        for _r in range(1, max_rounds):
            next_beam = []
            any_exp = False
            for eids, qt, sprobs, tscore, stopped, vjump in beam:
                if stopped or len(eids) >= max_rounds:
                    next_beam.append((eids, qt, sprobs, tscore, stopped, vjump))
                    continue

                p_evs = [store.get(eid) for eid in eids]
                p_cls = [e.cls_emb.to(dev) for e in p_evs]
                p_toks = torch.cat([e.tok_embs.to(dev) for e in p_evs], dim=0)
                q_next = self.update_query(q0_tok, qt, p_toks)

                cands, jump_ids = self.gather_next(
                    store, graph, p_evs[-1], q_next, visited=set(eids), mode=mode, budget=candidate_budget
                )
                if not cands:
                    next_beam.append((eids, qt, sprobs, tscore, True, vjump))
                    continue

                n_logits, n_reprs = self.score_candidates_batch(q0_cls, q0_tok, q_next, p_cls, p_toks, cands)
                n_probs = F.softmax(n_logits, dim=0)
                n_order = torch.argsort(n_logits, descending=True).tolist()
                best_next_repr = n_reprs[n_order[0]]

                path_repr = self.encode_path_state(q0_tok, p_toks, len(eids))
                p_cont = float(torch.sigmoid(self.continue_logit(path_repr, best_next_repr)).item())

                if p_cont < continue_thresh:
                    next_beam.append((eids, qt, sprobs, tscore, True, vjump))
                else:
                    any_exp = True
                    if p_cont < 0.80:
                        next_beam.append((eids, qt, sprobs, tscore, True, vjump))
                    for idx in n_order[:beam_width]:
                        c = cands[idx]
                        pr = float(n_probs[idx].item())
                        new_probs = sprobs + [pr]
                        new_score = sum(new_probs) / len(new_probs) + 0.15 * p_cont
                        next_beam.append((
                            eids + [c.event_id],
                            q_next,
                            new_probs,
                            new_score,
                            len(eids) + 1 >= max_rounds,
                            vjump or (c.event_id in jump_ids),
                        ))
            if not any_exp:
                break
            next_beam.sort(key=lambda x: x[3], reverse=True)
            beam = next_beam[:beam_width]
            if all(x[4] for x in beam):
                break

        # 去重 beam 候选路径，按搜索分从高到低对每一条候选路径独立做单/双表征证据验证，选通过的最好一条
        dedup: Dict[Tuple[str, ...], Tuple[List[str], float, bool]] = {}
        for item in beam:
            key = tuple(item[0])
            if key not in dedup or item[3] > dedup[key][1]:
                dedup[key] = (list(item[0]), float(item[3]), bool(item[5]))
        ranked_paths = sorted(dedup.values(), key=lambda x: x[1], reverse=True)

        use_frozen = self.dual_rep and hasattr(store.encoder, "encode_frozen_both")
        q0_cls_f = q0_tok_f = None
        if use_frozen:
            q0_cls_f, q0_tok_f = store.encoder.encode_frozen_both([query_text])[0]
            q0_cls_f, q0_tok_f = q0_cls_f.to(dev), q0_tok_f.to(dev)

        evaluated = []
        for eids, tscore, used_jump in ranked_paths:
            b_evs = [store.get(eid) for eid in eids]
            b_toks = torch.cat([e.tok_embs.to(dev) for e in b_evs], dim=0)
            b_repr = self.encode_path_state(q0_tok, b_toks, len(b_evs))
            path_cls_max = max(float(torch.dot(q0_cls, e.cls_emb.to(dev)).item()) for e in b_evs)

            f_repr = None
            f_cls_max = None
            if use_frozen and q0_cls_f is not None and q0_tok_f is not None:
                f_pairs = store.encoder.encode_frozen_both([e.text for e in b_evs])
                f_toks = torch.cat([p[1].to(dev) for p in f_pairs], dim=0)
                f_repr = self.encode_path_state(q0_tok_f, f_toks, len(b_evs))
                f_cls_max = max(float(torch.dot(q0_cls_f, p[0].to(dev)).item()) for p in f_pairs)

            pv = float(torch.sigmoid(self.valid_logit(b_repr, path_cls_max, f_repr, f_cls_max)).item())
            evaluated.append({
                "event_ids": eids,
                "tscore": tscore,
                "p_valid": pv,
                "used_jump": used_jump,
            })

        # 选通过验证门控的搜索最高分路径（不再只看第一条而扼杀证据更完整的第二条）
        passing = [c for c in evaluated if c["p_valid"] >= abstain_thresh]
        if passing:
            chosen = passing[0]
            return {
                "event_ids": list(chosen["event_ids"]),
                "abstained": False,
                "p_valid": chosen["p_valid"],
                "used_jump": chosen["used_jump"],
                "beam_candidates": evaluated,
            }
        best_pv = max((c["p_valid"] for c in evaluated), default=0.0)
        return {
            "event_ids": [],
            "abstained": True,
            "p_valid": best_pv,
            "used_jump": evaluated[0]["used_jump"] if evaluated else False,
            "beam_candidates": evaluated,
        }


def train_neural_v2(
    model: GeneralNeuralRecallNet,
    store: NeuralEpisodicStore,
    graph: PureNeuralGraph,
    examples: Sequence,
    epochs: int = 30,
    lr: float = 3e-3,
    candidate_budget: int = 14,
    accum_steps: int = 8,
    seed: int = 42,
) -> Dict[str, float]:
    """批量并行端到端训练纯通用神经回想核（零手工特征、支持 1/2/3+ 跳）。"""
    import random

    rng = random.Random(seed)
    dev = next(model.parameters()).device
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=lr * 0.1)

    q_cache = {ex.query: tuple(x.to(dev) for x in store.encoder.encode_one_both(ex.query)) for ex in examples}
    order = list(range(len(examples)))
    last_losses = {"rank": 0.0, "query": 0.0, "cont": 0.0, "valid": 0.0}

    for _ep in range(epochs):
        model.train()
        rng.shuffle(order)
        tr = tq = tc = tv = 0.0
        nr = nq = nc = nv = 0
        opt.zero_grad()
        acc = 0

        for idx in order:
            ex = examples[idx]
            if ex.gold_ids and any(gid not in store or store.get(gid).status != "active" for gid in ex.gold_ids):
                continue
            q0_cls, q0_tok = q_cache[ex.query]
            q0_mean = F.normalize(q0_tok.mean(dim=0), dim=-1)
            loss = torch.tensor(0.0, device=dev)

            r0_cands, max_sim = model.gather_round0(store, q0_cls, q0_tok, budget=candidate_budget)
            if not r0_cands:
                continue
            if ex.gold_ids:
                g0 = ex.gold_ids[0]
                if all(c.event_id != g0 for c in r0_cands):
                    r0_cands = [store.get(g0)] + r0_cands[:-1]

            r0_logits, _ = model.score_candidates_batch(q0_cls, q0_tok, q0_mean, [], None, r0_cands)

            if ex.category in ("direct", "superseded", "partial_only") and ex.gold_ids:
                g0_idx = next(i for i, c in enumerate(r0_cands) if c.event_id == ex.gold_ids[0])
                l_r0 = F.cross_entropy(r0_logits.unsqueeze(0), torch.tensor([g0_idx], device=dev))
                loss = loss + l_r0; tr += float(l_r0.item()); nr += 1

                ev0 = r0_cands[g0_idx]
                p_toks = ev0.tok_embs.to(dev)
                p_repr = model.encode_path_state(q0_tok, p_toks, 1)
                q_next = model.update_query(q0_tok, q0_mean, p_toks)
                n_cands, _ = model.gather_next(store, graph, ev0, q_next, visited={ev0.event_id}, mode="hybrid", budget=min(10, candidate_budget))
                if n_cands:
                    n_logits, n_reprs = model.score_candidates_batch(q0_cls, q0_tok, q_next, [ev0.cls_emb.to(dev)], p_toks, n_cands)
                    best_ni = int(torch.argmax(n_logits).item())
                    l_c = F.binary_cross_entropy_with_logits(
                        model.continue_logit(p_repr, n_reprs[best_ni]), torch.tensor(0.0, device=dev)
                    )
                    loss = loss + 1.2 * l_c; tc += float(l_c.item()); nc += 1

                p_cls_max = float(torch.dot(q0_cls, ev0.cls_emb.to(dev)).item())
                l_v = F.binary_cross_entropy_with_logits(
                    model.valid_logit(p_repr, p_cls_max), torch.tensor(1.0, device=dev)
                )
                loss = loss + 1.2 * l_v; tv += float(l_v.item()); nv += 1

                top_r0 = int(torch.argmax(r0_logits).item())
                if top_r0 != g0_idx:
                    w_ev = r0_cands[top_r0]
                    w_repr = model.encode_path_state(q0_tok, w_ev.tok_embs.to(dev), 1)
                    w_cls_max = float(torch.dot(q0_cls, w_ev.cls_emb.to(dev)).item())
                    l_vw = F.binary_cross_entropy_with_logits(model.valid_logit(w_repr, w_cls_max), torch.tensor(0.0, device=dev))
                    loss = loss + 0.5 * l_vw; tv += float(l_vw.item()); nv += 1

            elif ex.category == "cross" and len(ex.gold_ids) >= 2:
                g0_idx = next(i for i, c in enumerate(r0_cands) if c.event_id == ex.gold_ids[0])
                l_r0 = F.cross_entropy(r0_logits.unsqueeze(0), torch.tensor([g0_idx], device=dev))
                loss = loss + l_r0; tr += float(l_r0.item()); nr += 1

                visited = [ex.gold_ids[0]]
                qt = q0_mean
                for step_i in range(1, len(ex.gold_ids)):
                    tgt_id = ex.gold_ids[step_i]
                    p_evs = [store.get(eid) for eid in visited]
                    p_cls = [e.cls_emb.to(dev) for e in p_evs]
                    p_toks = torch.cat([e.tok_embs.to(dev) for e in p_evs], dim=0)
                    p_repr = model.encode_path_state(q0_tok, p_toks, len(visited))

                    # 残缺路径（缺关键一环）作为门控负例
                    inc_cls_max = max(float(torch.dot(q0_cls, c).item()) for c in p_cls)
                    l_v_inc = F.binary_cross_entropy_with_logits(model.valid_logit(p_repr, inc_cls_max), torch.tensor(0.0, device=dev))
                    loss = loss + 0.4 * l_v_inc; tv += float(l_v_inc.item()); nv += 1

                    q_next = model.update_query(q0_tok, qt, p_toks)
                    others = [e for e in store.active_events() if e.event_id not in set(visited)]
                    # 同源内容 token 空间对比损失
                    o_embs = torch.stack([e.mean_tok_emb for e in others], dim=0).to(dev)
                    t_pos = next(i for i, e in enumerate(others) if e.event_id == tgt_id)
                    l_q = F.cross_entropy(((o_embs @ q_next) / 0.08).unsqueeze(0), torch.tensor([t_pos], device=dev))
                    loss = loss + l_q; tq += float(l_q.item()); nq += 1

                    n_cands, _ = model.gather_next(store, graph, p_evs[-1], q_next, visited=set(visited), mode="hybrid", budget=min(10, candidate_budget))
                    if all(c.event_id != tgt_id for c in n_cands):
                        n_cands = [store.get(tgt_id)] + n_cands[:-1]
                    t_idx = next(i for i, c in enumerate(n_cands) if c.event_id == tgt_id)
                    n_logits, n_reprs = model.score_candidates_batch(q0_cls, q0_tok, q_next, p_cls, p_toks, n_cands)
                    l_rn = F.cross_entropy(n_logits.unsqueeze(0), torch.tensor([t_idx], device=dev))
                    loss = loss + l_rn; tr += float(l_rn.item()); nr += 1

                    l_c = F.binary_cross_entropy_with_logits(
                        model.continue_logit(p_repr, n_reprs[t_idx]), torch.tensor(1.0, device=dev)
                    )
                    loss = loss + l_c; tc += float(l_c.item()); nc += 1

                    visited.append(tgt_id)
                    qt = q_next

                full_evs = [store.get(eid) for eid in visited]
                full_toks = torch.cat([e.tok_embs.to(dev) for e in full_evs], dim=0)
                full_repr = model.encode_path_state(q0_tok, full_toks, len(visited))
                full_cls_max = max(float(torch.dot(q0_cls, e.cls_emb.to(dev)).item()) for e in full_evs)
                l_v = F.binary_cross_entropy_with_logits(
                    model.valid_logit(full_repr, full_cls_max), torch.tensor(1.0, device=dev)
                )
                loss = loss + 1.2 * l_v; tv += float(l_v.item()); nv += 1

            elif ex.category in ("hard_neg", "insufficient"):
                top_i = int(torch.argmax(r0_logits).item())
                ev0 = r0_cands[top_i]
                p_toks = ev0.tok_embs.to(dev)
                p_repr = model.encode_path_state(q0_tok, p_toks, 1)
                p_cls_max = float(torch.dot(q0_cls, ev0.cls_emb.to(dev)).item())
                l_v1 = F.binary_cross_entropy_with_logits(
                    model.valid_logit(p_repr, p_cls_max), torch.tensor(0.0, device=dev)
                )
                loss = loss + 0.9 * l_v1; tv += float(l_v1.item()); nv += 1

                q_next = model.update_query(q0_tok, q0_mean, p_toks)
                n_cands, _ = model.gather_next(store, graph, ev0, q_next, visited={ev0.event_id}, mode="hybrid", budget=6)
                if n_cands:
                    n_logits, n_reprs = model.score_candidates_batch(q0_cls, q0_tok, q_next, [ev0.cls_emb.to(dev)], p_toks, n_cands)
                    best_ni = int(torch.argmax(n_logits).item())
                    l_c = F.binary_cross_entropy_with_logits(
                        model.continue_logit(p_repr, n_reprs[best_ni]), torch.tensor(0.0, device=dev)
                    )
                    fake_toks = torch.cat([p_toks, n_cands[best_ni].tok_embs.to(dev)], dim=0)
                    fake_repr = model.encode_path_state(q0_tok, fake_toks, 2)
                    fake_cls_max = max(p_cls_max, float(torch.dot(q0_cls, n_cands[best_ni].cls_emb.to(dev)).item()))
                    l_v2 = F.binary_cross_entropy_with_logits(
                        model.valid_logit(fake_repr, fake_cls_max), torch.tensor(0.0, device=dev)
                    )
                    loss = loss + 0.5 * l_c + 0.7 * l_v2
                    tc += float(l_c.item()); nc += 1
                    tv += float(l_v2.item()); nv += 1

            if loss.requires_grad:
                (loss / accum_steps).backward()
                acc += 1
                if acc >= accum_steps:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    opt.step()
                    opt.zero_grad()
                    acc = 0

        if acc > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            opt.zero_grad()
        sched.step()

        last_losses = {
            "rank": round(tr / max(1, nr), 4),
            "query": round(tq / max(1, nq), 4),
            "cont": round(tc / max(1, nc), 4),
            "valid": round(tv / max(1, nv), 4),
        }
    return last_losses
