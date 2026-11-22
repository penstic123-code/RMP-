"""可持续学习的联想记忆网络（Associative Memory + Iterative Recall）核心实现。

组件（对应 docs/design/associative-memory.md）：
1. FrozenTextEncoder：冻结句向量编码器（默认 BAAI/bge-small-zh-v1.5，L2 归一化）。
2. EpisodicStore：事件记忆库，支持即时写入、按事件 ID 精确更正/覆盖（superseded）与删除，及无规则稀疏词法统计。
3. SparseAssociativeGraph：稀疏联想图（每节点最多 16 条候选边，由语义近邻 + 稀疏词法重叠 + 同日相邻三路合并；支持动态加边与验证路径巩固强化）。
4. IterativeRecallNet：循环回想网络（共享小网络版，E3 可替换为 RMP 路由模块池）：
   - 残差悬念查询：q_{t+1} = Norm(q_t + Δq(q_0, q_t, h_t, e_found))，在跨主题不变的几何张成空间
     （桥接新线索 e_⊥ 与原问题未覆盖残差 q_rem）上由路径状态动态门控，零样本泛化到未见主题/实体链。
   - 支持「沿图边 + 跳出图全库重搜」混合扩展（hybrid）及纯图（graph_only）/ 纯全库（full_only）消融。
   - 独立二分类门控（Sigmoid）：continue_gate（判断「我还缺什么」）+ abstain_gate（判断「是否拒答/停机」），
     输入全部采用尺度不变的相似度、词法覆盖增益与 softmax 概率边界特征，避免未见主题分布漂移。
5. 公平基线：OneShotBaseline / Fixed2HopGraphBaseline / PlainTwoRoundBaseline（同事件库、同编码器、同预算，阈值在训练集校准）。
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
if not os.environ.get("HF_HOME") and (ROOT / ".pytest_cache" / "hf").exists():
    os.environ["HF_HOME"] = str(ROOT / ".pytest_cache" / "hf")

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]+")


def extract_sparse_tokens(text: str) -> Set[str]:
    """无硬编码规则的稀疏词法特征：英文/数字词 + 中文字符 2-gram（短词保留自身）。"""
    tokens: Set[str] = set()
    for seg in _TOKEN_RE.findall(text.lower()):
        if seg.isascii():
            if len(seg) >= 2:
                tokens.add(seg)
        else:
            if len(seg) <= 2:
                tokens.add(seg)
            else:
                for i in range(len(seg) - 1):
                    tokens.add(seg[i : i + 2])
    return tokens


class FrozenTextEncoder:
    """冻结检索句向量编码器（默认 BAAI/bge-small-zh-v1.5，dim=512，L2 归一化）。"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", device: Optional[str] = None):
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.dim: int = int(self.model.config.hidden_size)
        self._cache: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def encode(self, texts: Sequence[str], batch_size: int = 32) -> torch.Tensor:
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
                    max_length=256,
                    return_tensors="pt",
                ).to(self.device)
                out = self.model(**enc)
                emb = out.last_hidden_state[:, 0]
                emb = F.normalize(emb.float(), dim=-1).cpu()
                for t, v in zip(batch, emb):
                    self._cache[t] = v
        return torch.stack([self._cache[t] for t in texts], dim=0)

    def encode_one(self, text: str) -> torch.Tensor:
        return self.encode([text])[0]


@dataclass
class EpisodicEvent:
    event_id: str
    text: str
    day: int
    source: str
    embedding: torch.Tensor  # (D,) CPU float32 L2-normalized
    status: str = "active"  # "active" | "superseded" | "deleted"
    supersedes: Optional[str] = None
    superseded_by: Optional[str] = None
    tokens: Set[str] = field(default_factory=set)
    meta: Dict[str, str] = field(default_factory=dict)


class EpisodicStore:
    """事件记忆库：保存原文、时间、来源、冻结句向量、状态（active/superseded/deleted）。"""

    def __init__(self, encoder: FrozenTextEncoder):
        self.encoder = encoder
        self._events: Dict[str, EpisodicEvent] = {}
        self._order: List[str] = []
        self._next_num: int = 1
        self._idf_dirty: bool = True
        self._idf: Dict[str, float] = {}

    def add_event(
        self,
        text: str,
        day: int,
        source: str,
        event_id: Optional[str] = None,
        status: str = "active",
        supersedes: Optional[str] = None,
        meta: Optional[Dict[str, str]] = None,
    ) -> EpisodicEvent:
        if event_id is None:
            while f"ev_{self._next_num:04d}" in self._events:
                self._next_num += 1
            eid = f"ev_{self._next_num:04d}"
            self._next_num += 1
        else:
            eid = event_id
        emb = self.encoder.encode_one(text)
        ev = EpisodicEvent(
            event_id=eid,
            text=text.strip(),
            day=int(day),
            source=source,
            embedding=emb,
            status=status,
            supersedes=supersedes,
            tokens=extract_sparse_tokens(text),
            meta=dict(meta or {}),
        )
        if supersedes and supersedes in self._events:
            old = self._events[supersedes]
            old.status = "superseded"
            old.superseded_by = eid
        is_new = eid not in self._events
        self._events[eid] = ev
        if is_new:
            self._order.append(eid)
        self._idf_dirty = True
        return ev

    def supersede_event(
        self,
        old_id: str,
        new_text: str,
        day: int,
        source: str,
        new_id: Optional[str] = None,
    ) -> EpisodicEvent:
        return self.add_event(
            text=new_text,
            day=day,
            source=source,
            event_id=new_id,
            status="active",
            supersedes=old_id,
        )

    def delete_event(self, event_id: str) -> bool:
        if event_id not in self._events:
            return False
        self._events[event_id].status = "deleted"
        self._idf_dirty = True
        return True

    def get(self, event_id: str) -> EpisodicEvent:
        return self._events[event_id]

    def __contains__(self, event_id: str) -> bool:
        return event_id in self._events

    def all_events(self, include_deleted: bool = False) -> List[EpisodicEvent]:
        evs = [self._events[eid] for eid in self._order]
        if include_deleted:
            return evs
        return [e for e in evs if e.status != "deleted"]

    def active_events(self) -> List[EpisodicEvent]:
        return [self._events[eid] for eid in self._order if self._events[eid].status == "active"]

    def clone_without(self, excluded_ids: Iterable[str]) -> "EpisodicStore":
        """用于因果消融：返回移除指定事件 ID 后的事件库副本（复用已缓存向量）。"""
        excl = set(excluded_ids)
        sub = EpisodicStore(self.encoder)
        for eid in self._order:
            if eid in excl:
                continue
            e = self._events[eid]
            copy_e = EpisodicEvent(
                event_id=e.event_id,
                text=e.text,
                day=e.day,
                source=e.source,
                embedding=e.embedding,
                status=e.status,
                supersedes=e.supersedes if e.supersedes not in excl else None,
                superseded_by=e.superseded_by if e.superseded_by not in excl else None,
                tokens=set(e.tokens),
                meta=dict(e.meta),
            )
            sub._events[eid] = copy_e
            sub._order.append(eid)
        sub._idf_dirty = True
        return sub

    def _ensure_idf(self) -> None:
        if not self._idf_dirty:
            return
        evs = self.all_events(include_deleted=False)
        n = max(1, len(evs))
        df: Dict[str, int] = {}
        for e in evs:
            for t in e.tokens:
                df[t] = df.get(t, 0) + 1
        self._idf = {t: math.log((n + 1.0) / (c + 1.0)) + 1.0 for t, c in df.items()}
        self._idf_dirty = False

    def lexical_overlap(self, tokens_a: Set[str], tokens_b: Set[str]) -> float:
        """IDF 加权余弦词法重叠，范围 [0, 1]。"""
        if not tokens_a or not tokens_b:
            return 0.0
        self._ensure_idf()
        inter = tokens_a & tokens_b
        if not inter:
            return 0.0
        num = sum(self._idf.get(t, 1.5) ** 2 for t in inter)
        na = math.sqrt(sum(self._idf.get(t, 1.5) ** 2 for t in tokens_a))
        nb = math.sqrt(sum(self._idf.get(t, 1.5) ** 2 for t in tokens_b))
        if na <= 1e-8 or nb <= 1e-8:
            return 0.0
        return float(min(1.0, num / (na * nb)))


@dataclass
class GraphEdge:
    src_id: str
    dst_id: str
    sem_sim: float
    lex_sim: float
    delta_t: float
    abs_dt: float
    same_day: float
    ch_sem: float
    ch_lex: float
    ch_temp: float
    consolidated_weight: float = 0.0


class SparseAssociativeGraph:
    """稀疏联想图：每节点最多 max_neighbors=16 条候选边，三路合并生成，支持动态加边与路径巩固。"""

    def __init__(
        self,
        store: EpisodicStore,
        max_neighbors: int = 16,
        sem_k: int = 8,
        lex_k: int = 6,
        temp_k: int = 4,
    ):
        self.store = store
        self.max_neighbors = max_neighbors
        self.sem_k = sem_k
        self.lex_k = lex_k
        self.temp_k = temp_k
        self.adj: Dict[str, Dict[str, GraphEdge]] = {}
        self.rebuild()

    def _make_edge(
        self,
        u: EpisodicEvent,
        v: EpisodicEvent,
        ch_sem: float,
        ch_lex: float,
        ch_temp: float,
        consolidated_weight: float = 0.0,
    ) -> GraphEdge:
        sem_sim = float(torch.dot(u.embedding, v.embedding).item())
        lex_sim = self.store.lexical_overlap(u.tokens, v.tokens)
        dt = (v.day - u.day) / 30.0
        return GraphEdge(
            src_id=u.event_id,
            dst_id=v.event_id,
            sem_sim=sem_sim,
            lex_sim=lex_sim,
            delta_t=max(-2.0, min(2.0, dt)),
            abs_dt=min(2.0, abs(dt)),
            same_day=1.0 if u.day == v.day else 0.0,
            ch_sem=ch_sem,
            ch_lex=ch_lex,
            ch_temp=ch_temp,
            consolidated_weight=consolidated_weight,
        )

    def _edge_priority(self, e: GraphEdge) -> float:
        return (
            0.45 * max(0.0, e.sem_sim)
            + 0.45 * e.lex_sim
            + 0.15 * e.same_day
            + 0.50 * e.consolidated_weight
        )

    def rebuild(self) -> None:
        self.adj.clear()
        evs = self.store.all_events(include_deleted=False)
        if not evs:
            return
        embs = torch.stack([e.embedding for e in evs], dim=0)
        sim_mat = embs @ embs.T
        n = len(evs)
        for i, u in enumerate(evs):
            sims = sim_mat[i].clone()
            sims[i] = -1e9
            k_sem = min(self.sem_k, n - 1)
            sem_idx = set(torch.topk(sims, k=k_sem).indices.tolist()) if k_sem > 0 else set()

            lex_scores = []
            for j, v in enumerate(evs):
                if j == i:
                    continue
                ls = self.store.lexical_overlap(u.tokens, v.tokens)
                if ls > 0.04:
                    lex_scores.append((ls, j))
            lex_scores.sort(reverse=True)
            lex_idx = {j for _, j in lex_scores[: self.lex_k]}

            temp_cand = [
                (abs(v.day - u.day), j)
                for j, v in enumerate(evs)
                if j != i and abs(v.day - u.day) <= 1
            ]
            temp_cand.sort()
            temp_idx = {j for _, j in temp_cand[: self.temp_k]}

            cand_idx = sem_idx | lex_idx | temp_idx
            edges = [
                self._make_edge(
                    u,
                    evs[j],
                    ch_sem=1.0 if j in sem_idx else 0.0,
                    ch_lex=1.0 if j in lex_idx else 0.0,
                    ch_temp=1.0 if j in temp_idx else 0.0,
                )
                for j in cand_idx
            ]
            edges.sort(key=self._edge_priority, reverse=True)
            self.adj[u.event_id] = {e.dst_id: e for e in edges[: self.max_neighbors]}

    def add_node_dynamic(self, new_ev: EpisodicEvent) -> None:
        """新事件写入后动态加边（保持每节点 <= max_neighbors）。"""
        evs = [e for e in self.store.all_events(include_deleted=False) if e.event_id != new_ev.event_id]
        if not evs:
            self.adj[new_ev.event_id] = {}
            return
        embs = torch.stack([e.embedding for e in evs], dim=0)
        sims = embs @ new_ev.embedding
        k_sem = min(self.sem_k, len(evs))
        sem_idx = set(torch.topk(sims, k=k_sem).indices.tolist()) if k_sem > 0 else set()

        lex_scores = []
        for j, v in enumerate(evs):
            ls = self.store.lexical_overlap(new_ev.tokens, v.tokens)
            if ls > 0.04:
                lex_scores.append((ls, j))
        lex_scores.sort(reverse=True)
        lex_idx = {j for _, j in lex_scores[: self.lex_k]}

        temp_cand = [(abs(v.day - new_ev.day), j) for j, v in enumerate(evs) if abs(v.day - new_ev.day) <= 1]
        temp_cand.sort()
        temp_idx = {j for _, j in temp_cand[: self.temp_k]}

        cand_idx = sem_idx | lex_idx | temp_idx
        out_edges = [
            self._make_edge(
                new_ev,
                evs[j],
                ch_sem=1.0 if j in sem_idx else 0.0,
                ch_lex=1.0 if j in lex_idx else 0.0,
                ch_temp=1.0 if j in temp_idx else 0.0,
            )
            for j in cand_idx
        ]
        out_edges.sort(key=self._edge_priority, reverse=True)
        self.adj[new_ev.event_id] = {e.dst_id: e for e in out_edges[: self.max_neighbors]}

        for e in out_edges[: self.max_neighbors]:
            v = self.store.get(e.dst_id)
            rev = self._make_edge(v, new_ev, e.ch_sem, e.ch_lex, e.ch_temp)
            nbrs = self.adj.setdefault(v.event_id, {})
            nbrs[new_ev.event_id] = rev
            if len(nbrs) > self.max_neighbors:
                ranked = sorted(nbrs.values(), key=self._edge_priority, reverse=True)[: self.max_neighbors]
                self.adj[v.event_id] = {x.dst_id: x for x in ranked}

    def reinforce_path(self, event_ids: Sequence[str], delta: float = 0.5) -> None:
        """巩固已验证有用的多跳路径（包含跳出图重搜发现的新路径），形成/加强图边。"""
        for i in range(len(event_ids) - 1):
            u_id, v_id = event_ids[i], event_ids[i + 1]
            if u_id not in self.store or v_id not in self.store:
                continue
            u, v = self.store.get(u_id), self.store.get(v_id)
            for a, b in ((u, v), (v, u)):
                nbrs = self.adj.setdefault(a.event_id, {})
                if b.event_id in nbrs:
                    nbrs[b.event_id].consolidated_weight = min(2.0, nbrs[b.event_id].consolidated_weight + delta)
                else:
                    edge = self._make_edge(a, b, 0.0, 0.0, 0.0, consolidated_weight=delta)
                    nbrs[b.event_id] = edge
                    if len(nbrs) > self.max_neighbors:
                        ranked = sorted(nbrs.values(), key=self._edge_priority, reverse=True)[: self.max_neighbors]
                        self.adj[a.event_id] = {x.dst_id: x for x in ranked}

    def neighbors(self, event_id: str, active_only: bool = False) -> List[GraphEdge]:
        edges = list(self.adj.get(event_id, {}).values())
        if active_only:
            edges = [e for e in edges if e.dst_id in self.store and self.store.get(e.dst_id).status == "active"]
        return edges

    def pair_edge_features(self, src_ev: Optional[EpisodicEvent], dst_ev: EpisodicEvent) -> List[float]:
        """提取 12 维边/新鲜度/状态特征（图内邻居与跳出图全库重搜候选完全统一）。"""
        dst_active = 1.0 if dst_ev.status == "active" else 0.0
        dst_superseded = 1.0 if dst_ev.status == "superseded" else 0.0
        if src_ev is None:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, dst_active, dst_superseded, 0.0, 0.0]
        edge = self.adj.get(src_ev.event_id, {}).get(dst_ev.event_id)
        if edge is not None:
            sem_sim, lex_sim = edge.sem_sim, edge.lex_sim
            delta_t, abs_dt, same_day = edge.delta_t, edge.abs_dt, edge.same_day
            ch_sem, ch_lex, ch_temp = edge.ch_sem, edge.ch_lex, edge.ch_temp
            has_edge = 1.0 + 0.5 * edge.consolidated_weight
        else:
            sem_sim = float(torch.dot(src_ev.embedding, dst_ev.embedding).item())
            lex_sim = self.store.lexical_overlap(src_ev.tokens, dst_ev.tokens)
            dt = (dst_ev.day - src_ev.day) / 30.0
            delta_t = max(-2.0, min(2.0, dt))
            abs_dt = min(2.0, abs(dt))
            same_day = 1.0 if src_ev.day == dst_ev.day else 0.0
            ch_sem = ch_lex = ch_temp = 0.0
            has_edge = 0.0
        dst_newer = 1.0 if dst_ev.day > src_ev.day else 0.0
        return [
            sem_sim, lex_sim, delta_t, abs_dt, same_day,
            ch_sem, ch_lex, ch_temp,
            dst_active, dst_superseded, dst_newer, has_edge,
        ]


@dataclass
class RecallPath:
    event_ids: List[str]
    q_t: torch.Tensor
    h_t: torch.Tensor
    step_probs: List[float]
    total_score: float
    stopped: bool = False
    via_jump: bool = False


@dataclass
class RecallResult:
    event_ids: List[str]
    abstained: bool
    p_valid: float
    rounds_used: int
    p_continue_trace: List[float]
    used_full_store_jump: bool
    best_path_score: float


class IterativeRecallNet(nn.Module):
    """可学习循环回想网络（跨主题零样本泛化版，E3 可替换为 RMP 路由模块池）。

    关键设计：
    1. 残差悬念查询 q_{t+1} = Norm(q_t + Δq(q_0, q_t, h_t, e_found))：
       在 {e_found, e_⊥ (桥接新线索), q_rem (原问题未覆盖部分), q_0} 四个几何正交基上由路径隐状态 h_t 动态门控，
       不记忆特定训练主题的高维子空间，确保在新主题/新实体链上直接泛化。
    2. 候选打分器：融合 11 维上下文交互特征（含未覆盖查询相似度 sim_qrem 与未覆盖词法增益 uncovered_lex）
       + 12 维边与新鲜度/覆盖状态特征 = 23 维，输出有界打分。
    3. 独立二分类门控（Sigmoid）：
       - continue_head：判断「我还缺什么」（输入全部为尺度不变的概率边界/余弦/词法覆盖增益）。
       - abstain_head：判断「是否有效或拒答」（同上，彻底杜绝未见主题 logit 尺度漂移）。
    """

    def __init__(self, emb_dim: int = 512, hidden_dim: int = 32, low_rank: int = 16):
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.low_rank = low_rank

        # 主题无关的路径隐状态 h_t（从每轮几何交互标量更新，不记忆特定主题向量）
        self.h_init = nn.Parameter(torch.zeros(hidden_dim))
        self.step_to_h = nn.Linear(4, hidden_dim)
        self.h_gru = nn.GRUCell(hidden_dim, hidden_dim)

        # 残差悬念查询门控：预测 [α_found, α_perp (桥接新实体), α_qrem (未回答问题残差), α_q0]
        self.geo_gate = nn.Sequential(
            nn.Linear(4 + hidden_dim, 24),
            nn.GELU(),
            nn.Linear(24, 4),
        )
        nn.init.zeros_(self.geo_gate[-1].weight)
        with torch.no_grad():
            # 平滑初始化：同时融合桥接新线索 e_⊥ 与原问题未覆盖残差 q_rem
            self.geo_gate[-1].bias.copy_(torch.tensor([0.20, 0.70, 0.70, -0.10]))

        # 候选打分器（25 维主题无关交互 + MMR 边际新信息 + 边/状态特征）
        self.scorer_mlp = nn.Sequential(
            nn.Linear(25, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        self.score_temp = nn.Parameter(torch.tensor(1.0))

        # 独立二分类门控头 1：Continue / Stop（10 维池密度不变特征）
        self.continue_head = nn.Sequential(
            nn.Linear(10, 24),
            nn.GELU(),
            nn.Linear(24, 1),
        )

        # 独立二分类门控头 2：Valid / Abstain（13 维池密度不变 + 联合路径覆盖特征）
        self.abstain_head = nn.Sequential(
            nn.Linear(13, 24),
            nn.GELU(),
            nn.Linear(24, 1),
        )

    def init_state(self, q0: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.h_init.to(q0.device))

    def update_query(
        self,
        q0: torch.Tensor,
        q_t: torch.Tensor,
        h_t: torch.Tensor,
        e_found: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """q_{t+1} = Norm(q_t + Δq(q_0, q_t, h_t, e_found))"""
        proj = torch.dot(e_found, q0)
        # 桥接事件中的新增线索（相对原问题正交）
        e_perp = e_found - proj * q0
        e_perp_norm = e_perp.norm().clamp(min=1e-6)
        e_perp_unit = e_perp / e_perp_norm

        # 原问题中尚未被 e_found 覆盖的悬念部分（相对 e_found 正交）
        q_rem = q0 - proj * e_found
        q_rem_norm = q_rem.norm().clamp(min=1e-6)
        q_rem_unit = q_rem / q_rem_norm

        sim_qt_e = torch.dot(e_found, q_t)
        scalars = torch.stack([proj, sim_qt_e, e_perp_norm, q_rem_norm])
        h_in = torch.tanh(self.step_to_h(scalars))
        h_next = self.h_gru(h_in.unsqueeze(0), h_t.unsqueeze(0)).squeeze(0)

        alphas = self.geo_gate(torch.cat([scalars, h_next], dim=-1))  # (4,)
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
        """对候选事件进行主题无关的上下文动态打分，返回 (logits[C], meta_list)。"""
        dev = q0.device
        if not candidates:
            return torch.empty(0, device=dev), []

        c_embs = torch.stack([c.embedding for c in candidates], dim=0).to(dev)
        sims_q0 = c_embs @ q0
        sims_qt = c_embs @ q_t

        mean_q0 = sims_q0.mean()
        std_q0 = sims_q0.std(unbiased=False).clamp(min=0.05)
        mean_qt = sims_qt.mean()
        std_qt = sims_qt.std(unbiased=False).clamp(min=0.05)
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

        feats = []
        metas = []
        for i, c in enumerate(candidates):
            lex_q0 = store.lexical_overlap(q0_tokens, c.tokens)
            lex_prev = store.lexical_overlap(src_ev.tokens, c.tokens) if src_ev is not None else 0.0
            lex_uncov = store.lexical_overlap(uncovered_q_tokens, c.tokens) if src_ev is not None else lex_q0
            # MMR 边际新信息特征：区分「补充未覆盖答案的新线索」vs「近重复复读机事件」
            mmr_gain = sims_qrem[i] - 0.5 * sims_prev[i] if src_ev is not None else torch.tensor(0.0, device=dev)
            dup_pen = torch.clamp(sims_prev[i] - 0.78, min=0.0) if src_ev is not None else torch.tensor(0.0, device=dev)
            edge_f = graph.pair_edge_features(src_ev, c)
            row = [
                sims_q0[i],
                sims_qt[i],
                sims_prev[i],
                sims_perp[i],
                sims_qrem[i],
                mmr_gain,
                dup_pen,
                torch.tensor(lex_q0, device=dev),
                torch.tensor(lex_prev, device=dev),
                torch.tensor(lex_uncov, device=dev),
                z_q0[i],
                z_qt[i],
                torch.tensor(is_hop2, device=dev),
                *[torch.tensor(x, device=dev) for x in edge_f],
            ]
            feats.append(torch.stack(row))
            metas.append(
                {
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
                }
            )
        feat_mat = torch.stack(feats, dim=0)  # (C, 23)
        raw = self.scorer_mlp(feat_mat).squeeze(-1)
        # 用 tanh 限制单步打分幅度，再用可学习温度缩放，防止 logit 爆炸导致过拟合
        logits = 5.0 * torch.tanh(raw / 3.0) * F.softplus(self.score_temp)
        return logits, metas

    def continue_logit(self, feat_vec: torch.Tensor) -> torch.Tensor:
        return self.continue_head(feat_vec).squeeze(-1)

    def abstain_logit(self, feat_vec: torch.Tensor) -> torch.Tensor:
        return self.abstain_head(feat_vec).squeeze(-1)

    def gather_round0_candidates(
        self,
        store: EpisodicStore,
        q0: torch.Tensor,
        q0_tokens: Set[str],
        budget: int = 16,
    ) -> Tuple[List[EpisodicEvent], Dict[str, float]]:
        evs = store.active_events()
        if not evs:
            return [], {"max_sim": 0.0, "med_sim": 0.0, "gap_med": 0.0, "z_max": 0.0, "max_lex": 0.0}
        embs = torch.stack([e.embedding for e in evs], dim=0).to(q0.device)
        sims = embs @ q0
        lexs = [store.lexical_overlap(q0_tokens, e.tokens) for e in evs]
        combined = sims + 0.35 * torch.tensor(lexs, device=q0.device)
        k = min(budget, len(evs))
        top_idx = torch.topk(combined, k=k).indices.tolist()

        max_sim = float(sims.max().item())
        med_sim = float(sims.median().item())
        std_sim = float(sims.std(unbiased=False).item()) if sims.numel() > 1 else 0.05
        z_max = (max_sim - float(sims.mean().item())) / max(0.05, std_sim)
        max_lex = max((lexs[i] for i in top_idx), default=0.0)
        stats = {
            "max_sim": max_sim,
            "med_sim": med_sim,
            "gap_med": max_sim - med_sim,
            "z_max": float(min(4.0, max(-2.0, z_max))),
            "max_lex": float(max_lex),
        }
        return [evs[i] for i in top_idx], stats

    def gather_next_candidates(
        self,
        store: EpisodicStore,
        graph: SparseAssociativeGraph,
        src_ev: EpisodicEvent,
        q_next: torch.Tensor,
        visited: Set[str],
        mode: str = "hybrid",
        budget: int = 16,
    ) -> Tuple[List[EpisodicEvent], Set[str]]:
        """按模式收集下一轮候选（仅 active 事件，返回 candidates, jump_ids）。"""
        graph_ids: List[str] = []
        if mode in ("hybrid", "graph_only"):
            for edge in graph.neighbors(src_ev.event_id, active_only=True):
                if edge.dst_id not in visited and edge.dst_id in store:
                    graph_ids.append(edge.dst_id)

        full_ids: List[str] = []
        if mode in ("hybrid", "full_only"):
            evs = [
                e for e in store.active_events()
                if e.event_id not in visited
            ]
            if evs:
                embs = torch.stack([e.embedding for e in evs], dim=0).to(q_next.device)
                sims = embs @ q_next
                k = min(budget if mode == "full_only" else max(8, budget // 2), len(evs))
                top_idx = torch.topk(sims, k=k).indices.tolist()
                full_ids = [evs[i].event_id for i in top_idx]

        merged: List[str] = []
        seen: Set[str] = set()
        jump_ids: Set[str] = set()
        graph_set = set(graph_ids)
        if mode == "hybrid":
            gi, fi = 0, 0
            while len(merged) < budget and (gi < len(graph_ids) or fi < len(full_ids)):
                if fi < len(full_ids):
                    eid = full_ids[fi]
                    fi += 1
                    if eid not in seen:
                        seen.add(eid)
                        merged.append(eid)
                        if eid not in graph_set:
                            jump_ids.add(eid)
                if len(merged) >= budget:
                    break
                if gi < len(graph_ids):
                    eid = graph_ids[gi]
                    gi += 1
                    if eid not in seen:
                        seen.add(eid)
                        merged.append(eid)
        elif mode == "graph_only":
            merged = graph_ids[:budget]
        else:
            merged = full_ids[:budget]
            jump_ids = {eid for eid in merged if eid not in graph_set}

        return [store.get(eid) for eid in merged], jump_ids

    @staticmethod
    def build_continue_features(
        r0_meta: Dict[str, float],
        next_meta: Optional[Dict[str, float]],
        next_sim_gap: float,
        device: torch.device,
    ) -> torch.Tensor:
        """10 维池密度不变特征，用于判断「我还缺什么，是否继续找第二段证据」。"""
        if next_meta is None:
            nm_qt = nm_q0 = nm_qrem = nm_lex_b = nm_lex_u = nm_sem_b = nm_edge = 0.0
        else:
            nm_qt = next_meta["sim_qt"]
            nm_q0 = next_meta["sim_q0"]
            nm_qrem = next_meta["sim_qrem"]
            nm_lex_b = next_meta["lex_prev"]
            nm_lex_u = next_meta["lex_uncov"]
            nm_sem_b = next_meta["sim_prev"]
            nm_edge = min(1.0, next_meta["has_edge"])
        return torch.tensor(
            [
                r0_meta["sim_q0"],
                r0_meta["lex_q0"],
                nm_qt,
                nm_q0,
                nm_qrem,
                nm_lex_b,
                nm_lex_u,
                nm_sem_b,
                next_sim_gap,
                nm_edge,
            ],
            dtype=torch.float32,
            device=device,
        )

    @staticmethod
    def build_abstain_features(
        pool_stats: Dict[str, float],
        r0_meta: Dict[str, float],
        second_meta: Optional[Dict[str, float]],
        path_len: int,
        joint_sim_q0: float,
        joint_lex_q0: float,
        device: torch.device,
    ) -> torch.Tensor:
        """13 维池密度不变 + 联合路径覆盖特征，用于判断「整条路径是否完整覆盖原问题，还是应该拒答」。"""
        if second_meta is None:
            s_qt = s_qrem = s_lex_b = s_lex_u = 0.0
        else:
            s_qt = second_meta["sim_qt"]
            s_qrem = second_meta["sim_qrem"]
            s_lex_b = second_meta["lex_prev"]
            s_lex_u = second_meta["lex_uncov"]
        return torch.tensor(
            [
                pool_stats["max_sim"],
                pool_stats["med_sim"],
                pool_stats["gap_med"],
                pool_stats["z_max"],
                r0_meta["sim_q0"],
                r0_meta["lex_q0"],
                s_qt,
                s_qrem,
                s_lex_b,
                s_lex_u,
                float(path_len) / 2.0,
                joint_sim_q0,
                joint_lex_q0,
            ],
            dtype=torch.float32,
            device=device,
        )

    @staticmethod
    def compute_joint_coverage(
        store: EpisodicStore,
        q0: torch.Tensor,
        q0_tokens: Set[str],
        events: Sequence[EpisodicEvent],
    ) -> Tuple[float, float]:
        """计算已找齐的证据集合对原问题 q0 的联合语义覆盖度与联合词法覆盖度。"""
        if not events:
            return 0.0, 0.0
        emb_sum = torch.stack([e.embedding for e in events], dim=0).mean(dim=0).to(q0.device)
        joint_sim = float(torch.dot(F.normalize(emb_sum, dim=-1), q0).item())
        tok_union: Set[str] = set()
        for e in events:
            tok_union |= e.tokens
        joint_lex = store.lexical_overlap(q0_tokens, tok_union)
        return joint_sim, joint_lex

    @torch.no_grad()
    def recall(
        self,
        store: EpisodicStore,
        graph: SparseAssociativeGraph,
        query_text: str,
        mode: str = "hybrid",
        beam_width: int = 4,
        max_rounds: int = 3,
        candidate_budget: int = 16,
        continue_thresh: float = 0.5,
        abstain_thresh: float = 0.5,
    ) -> RecallResult:
        """执行循环回想（每轮 Top-4 路径、最多 3 轮、支持跳出图与独立停机/拒答门控）。"""
        self.eval()
        dev = next(self.parameters()).device
        q0 = store.encoder.encode_one(query_text).to(dev)
        q0_tokens = extract_sparse_tokens(query_text)

        r0_cands, pool_stats = self.gather_round0_candidates(store, q0, q0_tokens, budget=candidate_budget)
        if not r0_cands:
            return RecallResult([], True, 0.0, 0, [], False, 0.0)

        r0_logits, r0_metas = self.score_candidates(store, graph, q0, q0, q0_tokens, None, r0_cands)
        r0_probs = F.softmax(r0_logits, dim=0)
        order0 = torch.argsort(r0_logits, descending=True).tolist()
        r0_p1 = float(r0_probs[order0[0]].item())
        r0_p2 = float(r0_probs[order0[1]].item()) if len(order0) > 1 else 0.0
        r0_pmargin = r0_p1 - r0_p2

        h0 = self.init_state(q0)
        beam: List[Tuple[RecallPath, Dict[str, float]]] = []
        for idx in order0[:beam_width]:
            ev = r0_cands[idx]
            prob = float(r0_probs[idx].item())
            p = RecallPath(
                event_ids=[ev.event_id],
                q_t=q0,
                h_t=h0,
                step_probs=[prob],
                total_score=prob,
                stopped=False,
                via_jump=False,
            )
            beam.append((p, r0_metas[idx]))

        p_cont_trace: List[float] = []
        rounds_used = 1
        second_meta_map: Dict[str, Dict[str, float]] = {}

        for r in range(1, max_rounds):
            next_beam: List[Tuple[RecallPath, Dict[str, float]]] = []
            any_expanded = False
            best_p_cont_this_round = 0.0

            for path, meta0 in beam:
                if path.stopped or len(path.event_ids) >= 2:
                    next_beam.append((path, meta0))
                    continue

                last_ev = store.get(path.event_ids[-1])
                q_next, h_next = self.update_query(q0, path.q_t, path.h_t, last_ev.embedding.to(dev))
                cands, jump_ids = self.gather_next_candidates(
                    store, graph, last_ev, q_next, visited=set(path.event_ids), mode=mode, budget=candidate_budget
                )
                if not cands:
                    path.stopped = True
                    next_beam.append((path, meta0))
                    continue

                n_logits, n_metas = self.score_candidates(store, graph, q0, q_next, q0_tokens, last_ev, cands)
                n_probs = F.softmax(n_logits, dim=0)
                n_order = torch.argsort(n_logits, descending=True).tolist()
                n_sim_med = float(torch.tensor([m["sim_qt"] for m in n_metas]).median().item())
                n_sim_gap = n_metas[n_order[0]]["sim_qt"] - n_sim_med

                cont_feat = self.build_continue_features(
                    r0_meta=meta0,
                    next_meta=n_metas[n_order[0]],
                    next_sim_gap=n_sim_gap,
                    device=dev,
                )
                p_cont = float(torch.sigmoid(self.continue_logit(cont_feat)).item())
                best_p_cont_this_round = max(best_p_cont_this_round, p_cont)

                if p_cont < continue_thresh:
                    stopped_p = RecallPath(
                        event_ids=list(path.event_ids),
                        q_t=path.q_t,
                        h_t=path.h_t,
                        step_probs=list(path.step_probs),
                        total_score=path.total_score,
                        stopped=True,
                        via_jump=path.via_jump,
                    )
                    next_beam.append((stopped_p, meta0))
                else:
                    any_expanded = True
                    # 当 continue 置信度不是极高时，允许单跳路径留在束中与两跳路径竞争
                    if p_cont < 0.80:
                        next_beam.append((
                            RecallPath(
                                event_ids=list(path.event_ids),
                                q_t=path.q_t,
                                h_t=path.h_t,
                                step_probs=list(path.step_probs),
                                total_score=path.total_score,
                                stopped=True,
                                via_jump=path.via_jump,
                            ),
                            meta0,
                        ))
                    for idx in n_order[:beam_width]:
                        cand_ev = cands[idx]
                        prob2 = float(n_probs[idx].item())
                        new_ids = path.event_ids + [cand_ev.event_id]
                        exp_p = RecallPath(
                            event_ids=new_ids,
                            q_t=q_next,
                            h_t=h_next,
                            step_probs=path.step_probs + [prob2],
                            total_score=0.5 * (path.step_probs[0] + prob2) + 0.15 * p_cont,
                            stopped=True,
                            via_jump=path.via_jump or (cand_ev.event_id in jump_ids),
                        )
                        second_meta_map["|".join(new_ids)] = n_metas[idx]
                        next_beam.append((exp_p, meta0))

            p_cont_trace.append(best_p_cont_this_round)
            if not any_expanded:
                break
            rounds_used = r + 1
            next_beam.sort(key=lambda x: x[0].total_score, reverse=True)
            beam = next_beam[:beam_width]
            if all(p.stopped for p, _ in beam):
                break

        best_path, best_meta0 = max(beam, key=lambda x: x[0].total_score)
        key = "|".join(best_path.event_ids)
        m2 = second_meta_map.get(key)
        evs_sel = [store.get(eid) for eid in best_path.event_ids]
        any_sup = any(e.status == "superseded" for e in evs_sel)
        j_sim, j_lex = self.compute_joint_coverage(store, q0, q0_tokens, evs_sel)

        abs_feat = self.build_abstain_features(
            pool_stats=pool_stats,
            r0_meta=best_meta0,
            second_meta=m2,
            path_len=len(best_path.event_ids),
            joint_sim_q0=j_sim,
            joint_lex_q0=j_lex,
            device=dev,
        )
        p_valid = float(torch.sigmoid(self.abstain_logit(abs_feat)).item())
        if p_valid < abstain_thresh or any_sup:
            return RecallResult(
                event_ids=[],
                abstained=True,
                p_valid=p_valid,
                rounds_used=rounds_used,
                p_continue_trace=p_cont_trace,
                used_full_store_jump=best_path.via_jump,
                best_path_score=best_path.total_score,
            )
        return RecallResult(
            event_ids=list(best_path.event_ids),
            abstained=False,
            p_valid=p_valid,
            rounds_used=rounds_used,
            p_continue_trace=p_cont_trace,
            used_full_store_jump=best_path.via_jump,
            best_path_score=best_path.total_score,
        )

    @torch.no_grad()
    def calibrate_abstain_thresh(
        self,
        store: EpisodicStore,
        graph: SparseAssociativeGraph,
        train_examples: Sequence["RecallExample"],
        mode: str = "hybrid",
        candidate_budget: int = 16,
        fp_penalty: float = 1.0,
    ) -> float:
        """仅在训练集上校准拒答门控阈值（与基线同口径平衡准确率，限制在逻辑回归中心带 [0.40, 0.66] 防单样本过拟合）。"""
        self.eval()
        pos_p, neg_p = [], []
        for ex in train_examples:
            res = self.recall(
                store, graph, ex.query, mode=mode, candidate_budget=candidate_budget, abstain_thresh=0.0
            )
            if ex.category in ("hard_neg", "insufficient"):
                neg_p.append(res.p_valid)
            else:
                pos_p.append(res.p_valid)
        if not pos_p or not neg_p:
            return 0.5
        best_obj = -1e9
        best_ts: List[float] = []
        for t in [x / 100.0 for x in range(40, 67, 2)]:
            tpr = sum(1 for p in pos_p if p >= t) / len(pos_p)
            fpr = sum(1 for p in neg_p if p >= t) / len(neg_p)
            obj = tpr - fp_penalty * fpr
            if obj > best_obj + 1e-9:
                best_obj = obj
                best_ts = [t]
            elif abs(obj - best_obj) <= 1e-9:
                best_ts.append(t)
        return round(sum(best_ts) / len(best_ts), 3) if best_ts else 0.5


@dataclass
class RecallExample:
    query: str
    category: str  # "direct" | "cross" | "hard_neg" | "insufficient" | "superseded" | "partial_only"
    gold_ids: List[str]
    topic: str = ""
    bridge_id: Optional[str] = None
    target_id: Optional[str] = None


def train_iterative_recall(
    model: IterativeRecallNet,
    store: EpisodicStore,
    graph: SparseAssociativeGraph,
    examples: Sequence[RecallExample],
    epochs: int = 30,
    lr: float = 4e-3,
    candidate_budget: int = 16,
    seed: int = 42,
) -> Dict[str, float]:
    """用四类无规则学习信号联合训练循环回想网络（带 L2 正则与概率边界门控监督）。"""
    import random

    rng = random.Random(seed)
    dev = next(model.parameters()).device
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    q_embs = {ex.query: store.encoder.encode_one(ex.query).to(dev) for ex in examples}
    q_toks = {ex.query: extract_sparse_tokens(ex.query) for ex in examples}

    last_losses = {"rank": 0.0, "query": 0.0, "cont": 0.0, "abstain": 0.0}
    order = list(range(len(examples)))

    for _ep in range(epochs):
        model.train()
        rng.shuffle(order)
        tot_rank = tot_q = tot_cont = tot_abs = 0.0
        n_rank = n_q = n_cont = n_abs = 0

        for idx in order:
            ex = examples[idx]
            q0 = q_embs[ex.query]
            toks = q_toks[ex.query]
            loss = torch.tensor(0.0, device=dev)

            r0_cands, pool_stats = model.gather_round0_candidates(store, q0, toks, budget=candidate_budget)
            if not r0_cands:
                continue

            if ex.gold_ids:
                if any(gid not in store or store.get(gid).status != "active" for gid in ex.gold_ids):
                    continue
                g0 = ex.gold_ids[0]
                if g0 in store and all(c.event_id != g0 for c in r0_cands):
                    r0_cands = [store.get(g0)] + r0_cands[:-1]

            r0_logits, r0_metas = model.score_candidates(store, graph, q0, q0, toks, None, r0_cands)
            r0_probs = F.softmax(r0_logits, dim=0)
            r0_order = torch.argsort(r0_logits, descending=True).tolist()
            r0_p1 = float(r0_probs[r0_order[0]].item())
            r0_p2 = float(r0_probs[r0_order[1]].item()) if len(r0_order) > 1 else 0.0
            r0_pmargin = r0_p1 - r0_p2

            if ex.category in ("direct", "superseded", "partial_only") and ex.gold_ids:
                g0_idx = next(i for i, c in enumerate(r0_cands) if c.event_id == ex.gold_ids[0])
                l_r0 = F.cross_entropy(r0_logits.unsqueeze(0), torch.tensor([g0_idx], device=dev))
                loss = loss + l_r0
                tot_rank += float(l_r0.item())
                n_rank += 1

                # 下一跳检查：学习「证据已齐或库里无后续 → 停机 (0)」
                src_ev = r0_cands[g0_idx]
                h0 = model.init_state(q0)
                q_next, _ = model.update_query(q0, q0, h0, src_ev.embedding.to(dev))
                n_cands, _ = model.gather_next_candidates(
                    store, graph, src_ev, q_next, visited={src_ev.event_id}, mode="hybrid", budget=candidate_budget
                )
                if n_cands:
                    n_logits, n_metas = model.score_candidates(store, graph, q0, q_next, toks, src_ev, n_cands)
                    no = torch.argsort(n_logits, descending=True).tolist()
                    n_sim_med = float(torch.tensor([m["sim_qt"] for m in n_metas]).median().item())
                    c_feat = model.build_continue_features(
                        r0_metas[g0_idx],
                        n_metas[no[0]],
                        n_metas[no[0]]["sim_qt"] - n_sim_med,
                        dev,
                    )
                    l_cont = F.binary_cross_entropy_with_logits(
                        model.continue_logit(c_feat), torch.tensor(0.0, device=dev)
                    )
                    loss = loss + 1.4 * l_cont
                    tot_cont += float(l_cont.item())
                    n_cont += 1

                # abstain_head: 有效正例 → 1.0
                j_sim, j_lex = model.compute_joint_coverage(store, q0, toks, [src_ev])
                a_feat = model.build_abstain_features(
                    pool_stats, r0_metas[g0_idx], None, 1, j_sim, j_lex, dev
                )
                l_abs = F.binary_cross_entropy_with_logits(
                    model.abstain_logit(a_feat), torch.tensor(1.0, device=dev)
                )
                loss = loss + 1.0 * l_abs
                tot_abs += float(l_abs.item())
                n_abs += 1

            elif ex.category == "cross" and len(ex.gold_ids) == 2:
                g0_id, g1_id = ex.gold_ids[0], ex.gold_ids[1]
                g0_idx = next(i for i, c in enumerate(r0_cands) if c.event_id == g0_id)
                l_r0 = F.cross_entropy(r0_logits.unsqueeze(0), torch.tensor([g0_idx], device=dev))
                loss = loss + l_r0
                tot_rank += float(l_r0.item())
                n_rank += 1

                src_ev = store.get(g0_id)
                tgt_ev = store.get(g1_id)
                h0 = model.init_state(q0)
                q_next, _ = model.update_query(q0, q0, h0, src_ev.embedding.to(dev))

                all_other = [e for e in store.active_events() if e.event_id != g0_id]
                other_embs = torch.stack([e.embedding for e in all_other], dim=0).to(dev)
                tgt_pos = next(i for i, e in enumerate(all_other) if e.event_id == g1_id)
                q_sims = (other_embs @ q_next) / 0.08
                l_q = F.cross_entropy(q_sims.unsqueeze(0), torch.tensor([tgt_pos], device=dev))
                loss = loss + 1.0 * l_q
                tot_q += float(l_q.item())
                n_q += 1

                n_cands, _ = model.gather_next_candidates(
                    store, graph, src_ev, q_next, visited={g0_id}, mode="hybrid", budget=candidate_budget
                )
                if all(c.event_id != g1_id for c in n_cands):
                    n_cands = [tgt_ev] + n_cands[:-1]
                g1_idx = next(i for i, c in enumerate(n_cands) if c.event_id == g1_id)
                n_logits, n_metas = model.score_candidates(store, graph, q0, q_next, toks, src_ev, n_cands)
                l_r1 = F.cross_entropy(n_logits.unsqueeze(0), torch.tensor([g1_idx], device=dev))
                loss = loss + l_r1
                tot_rank += float(l_r1.item())
                n_rank += 1

                n_sim_med = float(torch.tensor([m["sim_qt"] for m in n_metas]).median().item())
                c_feat = model.build_continue_features(
                    r0_metas[g0_idx],
                    n_metas[g1_idx],
                    n_metas[g1_idx]["sim_qt"] - n_sim_med,
                    dev,
                )
                l_cont = F.binary_cross_entropy_with_logits(
                    model.continue_logit(c_feat), torch.tensor(1.0, device=dev)
                )
                loss = loss + 1.2 * l_cont
                tot_cont += float(l_cont.item())
                n_cont += 1

                j_sim, j_lex = model.compute_joint_coverage(store, q0, toks, [src_ev, tgt_ev])
                a_feat = model.build_abstain_features(
                    pool_stats, r0_metas[g0_idx], n_metas[g1_idx], 2, j_sim, j_lex, dev
                )
                l_abs = F.binary_cross_entropy_with_logits(
                    model.abstain_logit(a_feat), torch.tensor(1.0, device=dev)
                )
                loss = loss + 1.0 * l_abs
                tot_abs += float(l_abs.item())
                n_abs += 1

            elif ex.category in ("hard_neg", "insufficient"):
                top_idx = r0_order[0]
                src_ev = r0_cands[top_idx]
                j_sim1, j_lex1 = model.compute_joint_coverage(store, q0, toks, [src_ev])
                a_feat1 = model.build_abstain_features(
                    pool_stats, r0_metas[top_idx], None, 1, j_sim1, j_lex1, dev
                )
                l_abs1 = F.binary_cross_entropy_with_logits(
                    model.abstain_logit(a_feat1), torch.tensor(0.0, device=dev)
                )
                loss = loss + 1.45 * l_abs1
                tot_abs += float(l_abs1.item())
                n_abs += 1

                h0 = model.init_state(q0)
                q_next, _ = model.update_query(q0, q0, h0, src_ev.embedding.to(dev))
                n_cands, _ = model.gather_next_candidates(
                    store, graph, src_ev, q_next, visited={src_ev.event_id}, mode="hybrid", budget=candidate_budget
                )
                if n_cands:
                    n_logits, n_metas = model.score_candidates(store, graph, q0, q_next, toks, src_ev, n_cands)
                    no = torch.argsort(n_logits, descending=True).tolist()
                    n_sim_med = float(torch.tensor([m["sim_qt"] for m in n_metas]).median().item())
                    c_feat = model.build_continue_features(
                        r0_metas[top_idx],
                        n_metas[no[0]],
                        n_metas[no[0]]["sim_qt"] - n_sim_med,
                        dev,
                    )
                    l_cont = F.binary_cross_entropy_with_logits(
                        model.continue_logit(c_feat), torch.tensor(0.0, device=dev)
                    )
                    j_sim2, j_lex2 = model.compute_joint_coverage(store, q0, toks, [src_ev, n_cands[no[0]]])
                    a_feat2 = model.build_abstain_features(
                        pool_stats, r0_metas[top_idx], n_metas[no[0]], 2, j_sim2, j_lex2, dev
                    )
                    l_abs2 = F.binary_cross_entropy_with_logits(
                        model.abstain_logit(a_feat2), torch.tensor(0.0, device=dev)
                    )
                    loss = loss + 0.6 * l_cont + 1.1 * l_abs2
                    tot_cont += float(l_cont.item())
                    n_cont += 1
                    tot_abs += float(l_abs2.item())
                    n_abs += 1

            if loss.requires_grad:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()

        last_losses = {
            "rank": round(tot_rank / max(1, n_rank), 4),
            "query": round(tot_q / max(1, n_q), 4),
            "cont": round(tot_cont / max(1, n_cont), 4),
            "abstain": round(tot_abs / max(1, n_abs), 4),
        }
    return last_losses


# ---------------------------------------------------------------------------
# 公平对照基线（同事件库、同编码器、同预算，阈值在训练集校准）
# ---------------------------------------------------------------------------

class OneShotBaseline:
    """对照 1：一次语义检索（给足候选预算，按训练集校准相似度阈值决定是否拒答及取 1/2 条）。"""

    def __init__(self, thresh1: float = 0.52, thresh2: float = 0.50):
        self.thresh1 = thresh1
        self.thresh2 = thresh2

    def calibrate(self, store: EpisodicStore, train_examples: Sequence[RecallExample]) -> None:
        pos_s1, neg_s1 = [], []
        for ex in train_examples:
            q0 = store.encoder.encode_one(ex.query)
            evs = store.active_events()
            sims = torch.stack([e.embedding for e in evs], dim=0) @ q0
            top1 = float(sims.max().item())
            if ex.category in ("hard_neg", "insufficient"):
                neg_s1.append(top1)
            else:
                pos_s1.append(top1)
        if pos_s1 and neg_s1:
            best_t, best_acc = 0.52, -1.0
            for t in [x / 100.0 for x in range(30, 85, 2)]:
                acc = (
                    sum(1 for s in pos_s1 if s >= t) / len(pos_s1)
                    + sum(1 for s in neg_s1 if s < t) / len(neg_s1)
                ) / 2.0
                if acc > best_acc:
                    best_acc, best_t = acc, t
            self.thresh1 = best_t
            self.thresh2 = max(0.35, best_t - 0.04)

    def recall(self, store: EpisodicStore, query_text: str, top_k: int = 2) -> RecallResult:
        q0 = store.encoder.encode_one(query_text)
        evs = store.active_events()
        if not evs:
            return RecallResult([], True, 0.0, 1, [], False, 0.0)
        sims = torch.stack([e.embedding for e in evs], dim=0) @ q0
        order = torch.argsort(sims, descending=True).tolist()
        s1 = float(sims[order[0]].item())
        if s1 < self.thresh1:
            return RecallResult([], True, s1, 1, [], False, s1)
        picked = [evs[order[0]].event_id]
        if top_k >= 2 and len(order) > 1 and float(sims[order[1]].item()) >= self.thresh2:
            picked.append(evs[order[1]].event_id)
        return RecallResult(picked, False, s1, 1, [], False, s1)


class Fixed2HopGraphBaseline:
    """对照 2：固定两跳图传播（首轮语义 Top-1，第二轮固定沿稀疏图邻居找最高分邻居）。"""

    def __init__(self, thresh1: float = 0.52, thresh2: float = 0.48):
        self.thresh1 = thresh1
        self.thresh2 = thresh2

    def calibrate(self, store: EpisodicStore, graph: SparseAssociativeGraph, train_examples: Sequence[RecallExample]) -> None:
        base = OneShotBaseline()
        base.calibrate(store, train_examples)
        self.thresh1 = base.thresh1
        cross_s2, direct_s2 = [], []
        for ex in train_examples:
            q0 = store.encoder.encode_one(ex.query)
            evs = store.active_events()
            sims = torch.stack([e.embedding for e in evs], dim=0) @ q0
            e1 = evs[int(torch.argmax(sims).item())]
            nbrs = graph.neighbors(e1.event_id, active_only=True)
            if not nbrs:
                continue
            best_s2 = max(
                0.45 * float(torch.dot(q0, store.get(ed.dst_id).embedding).item())
                + 0.35 * ed.sem_sim
                + 0.40 * ed.lex_sim
                for ed in nbrs
            )
            if ex.category == "cross":
                cross_s2.append(best_s2)
            elif ex.category in ("direct", "partial_only"):
                direct_s2.append(best_s2)
        if cross_s2 and direct_s2:
            best_t, best_acc = 0.48, -1.0
            for t in [x / 100.0 for x in range(25, 85, 2)]:
                acc = (
                    sum(1 for s in cross_s2 if s >= t) / len(cross_s2)
                    + sum(1 for s in direct_s2 if s < t) / len(direct_s2)
                ) / 2.0
                if acc > best_acc:
                    best_acc, best_t = acc, t
            self.thresh2 = best_t

    def recall(self, store: EpisodicStore, graph: SparseAssociativeGraph, query_text: str) -> RecallResult:
        q0 = store.encoder.encode_one(query_text)
        evs = store.active_events()
        if not evs:
            return RecallResult([], True, 0.0, 1, [], False, 0.0)
        sims = torch.stack([e.embedding for e in evs], dim=0) @ q0
        idx1 = int(torch.argmax(sims).item())
        s1 = float(sims[idx1].item())
        if s1 < self.thresh1:
            return RecallResult([], True, s1, 1, [], False, s1)
        e1 = evs[idx1]
        nbrs = graph.neighbors(e1.event_id, active_only=True)
        if not nbrs:
            return RecallResult([e1.event_id], False, s1, 1, [], False, s1)
        scored = [
            (
                0.45 * float(torch.dot(q0, store.get(ed.dst_id).embedding).item())
                + 0.35 * ed.sem_sim
                + 0.40 * ed.lex_sim,
                ed.dst_id,
            )
            for ed in nbrs
        ]
        scored.sort(reverse=True)
        if scored[0][0] >= self.thresh2:
            return RecallResult([e1.event_id, scored[0][1]], False, s1, 2, [], False, s1 + scored[0][0])
        return RecallResult([e1.event_id], False, s1, 1, [], False, s1)


class PlainTwoRoundBaseline:
    """对照 3：普通两轮检索（第二轮根据首轮结果固定改写查询重搜全库，无学习参数，alpha 与阈值均在训练集校准）。"""

    def __init__(self, alpha: float = 0.65, thresh1: float = 0.52, thresh2: float = 0.52):
        self.alpha = alpha
        self.thresh1 = thresh1
        self.thresh2 = thresh2

    def _rewrite_query(self, q0: torch.Tensor, e1: torch.Tensor, alpha: Optional[float] = None) -> torch.Tensor:
        a = self.alpha if alpha is None else alpha
        return F.normalize(q0 + a * e1, dim=-1)

    def calibrate(self, store: EpisodicStore, train_examples: Sequence[RecallExample]) -> None:
        base = OneShotBaseline()
        base.calibrate(store, train_examples)
        self.thresh1 = base.thresh1

        # 在训练集校准最佳固定改写系数 alpha（使第二跳目标排名最好）
        best_a, best_hit = 0.65, -1
        for a in (0.35, 0.55, 0.75, 1.0):
            hits = 0
            for ex in train_examples:
                if ex.category != "cross" or len(ex.gold_ids) < 2:
                    continue
                q0 = store.encoder.encode_one(ex.query)
                e1 = store.get(ex.gold_ids[0])
                q2 = self._rewrite_query(q0, e1.embedding, alpha=a)
                others = [e for e in store.active_events() if e.event_id != e1.event_id]
                sims2 = torch.stack([e.embedding for e in others], dim=0) @ q2
                if others[int(torch.argmax(sims2).item())].event_id == ex.gold_ids[1]:
                    hits += 1
            if hits > best_hit:
                best_hit, best_a = hits, a
        self.alpha = best_a

        cross_s2, direct_s2 = [], []
        for ex in train_examples:
            q0 = store.encoder.encode_one(ex.query)
            evs = store.active_events()
            sims = torch.stack([e.embedding for e in evs], dim=0) @ q0
            e1 = evs[int(torch.argmax(sims).item())]
            q2 = self._rewrite_query(q0, e1.embedding)
            others = [e for e in evs if e.event_id != e1.event_id]
            if not others:
                continue
            sims2 = torch.stack([e.embedding for e in others], dim=0) @ q2
            top2 = float(sims2.max().item())
            if ex.category == "cross":
                cross_s2.append(top2)
            elif ex.category in ("direct", "partial_only"):
                direct_s2.append(top2)
        if cross_s2 and direct_s2:
            best_t, best_acc = 0.52, -1.0
            for t in [x / 100.0 for x in range(30, 85, 2)]:
                acc = (
                    sum(1 for s in cross_s2 if s >= t) / len(cross_s2)
                    + sum(1 for s in direct_s2 if s < t) / len(direct_s2)
                ) / 2.0
                if acc > best_acc:
                    best_acc, best_t = acc, t
            self.thresh2 = best_t

    def recall(self, store: EpisodicStore, query_text: str) -> RecallResult:
        q0 = store.encoder.encode_one(query_text)
        evs = store.active_events()
        if not evs:
            return RecallResult([], True, 0.0, 1, [], False, 0.0)
        sims = torch.stack([e.embedding for e in evs], dim=0) @ q0
        idx1 = int(torch.argmax(sims).item())
        s1 = float(sims[idx1].item())
        if s1 < self.thresh1:
            return RecallResult([], True, s1, 1, [], False, s1)
        e1 = evs[idx1]
        q2 = self._rewrite_query(q0, e1.embedding)
        others = [e for e in evs if e.event_id != e1.event_id]
        if not others:
            return RecallResult([e1.event_id], False, s1, 1, [], False, s1)
        sims2 = torch.stack([e.embedding for e in others], dim=0) @ q2
        idx2 = int(torch.argmax(sims2).item())
        s2 = float(sims2[idx2].item())
        if s2 >= self.thresh2:
            return RecallResult([e1.event_id, others[idx2].event_id], False, s1, 2, [], True, s1 + s2)
        return RecallResult([e1.event_id], False, s1, 1, [], False, s1)
