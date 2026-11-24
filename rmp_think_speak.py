"""思说解耦 RMP v2（Think-Speak Decoupled RMP + 有界工作记忆注意力/拷贝指针）。

核心架构（本质区别于全上下文 Transformer）：
1. 主干保持 100% 零 Self-Attention 的 `WavePool` RMP：路由竞争分工、多尺度深浅波、O(1) 恒定对话状态、
   SMF 局部抗遗忘；
2. 思说双时钟解耦：`emit_gate` 自主决定「开口前先想几拍」以及「说到一半暂停吐字、补查第二条记忆再继续」；
3. 有界工作记忆注意力（Bounded Working-Memory Cross-Attention + Copy Pointer）：
   - 仅在 RMP 内部思考步抓回 1~2 条活跃记忆进入工作记忆槽（固定极小窗口，O(1) 开销，绝不对长对话存 KV-cache）
     时生效；
   - 解决纯固定循环状态无法精确逐字节拷贝「训练从未见过的任意中文人名/地名」的理论边界，同时完全不依赖 BGE/Qwen。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pool import WavePool

# 开放词表 UTF-8 字节级编码（零 <UNK>，支持任意未见中文人名/地名/实体）
PAD_ID = 256
BOS_ID = 257
EOS_ID = 258
SEP_ID = 259
MEM_ID = 260
THINK_ID = 261
VOCAB_SIZE = 262


def encode_bytes(text: str) -> List[int]:
    return list(text.encode("utf-8"))


def decode_bytes(ids: Sequence[int]) -> str:
    bs = bytes(int(i) for i in ids if 0 <= int(i) < 256)
    return bs.decode("utf-8", errors="ignore")


@dataclass
class RMPEpisodicEntry:
    event_id: str
    text: str
    day: int
    status: str = "active"  # active | superseded | deleted
    supersedes: Optional[str] = None


class RMPEpisodicBank:
    """RMP 专属情景记忆库（支持瞬时写入、改口覆盖、删除；检索完全由 RMP 自身编码完成）。"""

    def __init__(self):
        self._entries: Dict[str, RMPEpisodicEntry] = {}
        self._order: List[str] = []

    def add(self, event_id: str, text: str, day: int = 1, supersedes: Optional[str] = None) -> RMPEpisodicEntry:
        if supersedes and supersedes in self._entries:
            self._entries[supersedes].status = "superseded"
        ent = RMPEpisodicEntry(event_id=event_id, text=text.strip(), day=day, status="active", supersedes=supersedes)
        if event_id not in self._entries:
            self._order.append(event_id)
        self._entries[event_id] = ent
        return ent

    def supersede(self, old_id: str, new_id: str, new_text: str, day: int = 2) -> RMPEpisodicEntry:
        return self.add(new_id, new_text, day=day, supersedes=old_id)

    def delete(self, event_id: str) -> None:
        if event_id in self._entries:
            self._entries[event_id].status = "deleted"

    def active_entries(self) -> List[RMPEpisodicEntry]:
        return [self._entries[eid] for eid in self._order if self._entries[eid].status == "active"]

    def get(self, event_id: str) -> RMPEpisodicEntry:
        return self._entries[event_id]


@dataclass
class EncodedMemory:
    key: torch.Tensor        # (mem_dim,) 归一化深波句级键
    tok_keys: torch.Tensor   # (L, mem_dim) 归一化字节级 RMP 隐状态（用于未见实体细粒度匹配）
    wm_keys: torch.Tensor    # (L, wm_dim) 工作记忆 Cross-Attention 键
    wm_vals: torch.Tensor    # (L, D) 工作记忆值
    byte_ids: torch.Tensor   # (L,) 原始字节 ID（供 Copy Pointer 精确拷贝未见实体）


class WorkingMemoryBuffer:
    """极小容量的有界工作记忆槽（最多保留最近 2 条被 RMP 主动召回的记忆，O(1) 窗口）。"""

    def __init__(self, max_slots: int = 2):
        self.max_slots = max_slots
        self.slots: List[EncodedMemory] = []

    def push(self, em: EncodedMemory) -> None:
        self.slots.append(em)
        if len(self.slots) > self.max_slots:
            self.slots = self.slots[-self.max_slots :]

    def is_empty(self) -> bool:
        return len(self.slots) == 0

    def stacked(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        wk = torch.cat([s.wm_keys for s in self.slots], dim=0)
        wv = torch.cat([s.wm_vals for s in self.slots], dim=0)
        ids = torch.cat([s.byte_ids for s in self.slots], dim=0)
        return wk, wv, ids


class ThinkSpeakRMP(nn.Module):
    """思说解耦 RMP 主干（零 Self-Attention WavePool + 有界工作记忆注意力/拷贝指针 + 说话门）。"""

    def __init__(
        self,
        vocab: int = VOCAB_SIZE,
        D: int = 128,
        N: int = 24,
        k: int = 2,
        G: int = 4,
        pd: int = 96,
        pm: int = 16,
        phidden: int = 192,
        qr: int = 32,
        mem_dim: int = 96,
        wm_dim: int = 96,
        ctx_decay: Optional[List[float]] = None,
        tau_prior: float = 2.0,
        bottleneck: bool = False,
        use_working_memory: bool = True,
    ):
        super().__init__()
        self.vocab = vocab
        self.D = D
        self.G = G
        self.pd = pd
        self.mem_dim = mem_dim
        self.wm_dim = wm_dim
        self.use_working_memory = use_working_memory

        if ctx_decay is None:
            ctx_decay = [round(0.30 + (0.985 - 0.30) * (i / max(1, G - 1)) ** 1.5, 4) for i in range(G)]

        self.emb = nn.Embedding(vocab, D)
        nn.init.normal_(self.emb.weight, std=0.02)

        self.pool = WavePool(
            N=N, k=k, G=G, D=D, d=pd, m=pm, hidden=phidden, qr=qr,
            init="spread", rule="ctx", wave_query=True, wave_ctx=True,
            ctx_decay=ctx_decay, tau_prior=tau_prior, bottleneck=bottleneck,
        )

        feat_dim = pd * 2 + D

        # 1. RMP 自身记忆编码与细粒度深波查询头（零 BGE / 零外部编码器）
        self.mem_key_proj = nn.Sequential(nn.LayerNorm(pd + D), nn.Linear(pd + D, mem_dim))
        self.mem_query_proj = nn.Sequential(nn.LayerNorm(pd + D), nn.Linear(pd + D, mem_dim))
        self.tok_match_proj = nn.Linear(D, mem_dim, bias=False)
        self.null_mem_key = nn.Parameter(F.normalize(torch.randn(mem_dim), dim=-1))
        self.null_score_bias = nn.Parameter(torch.tensor(0.35))
        self.tok_match_weight = nn.Parameter(torch.tensor(1.5))
        self.mem_temp = nn.Parameter(torch.tensor(0.10))

        # 2. 有界工作记忆 Cross-Attention + 拷贝指针（仅作用于已召回的 1~2 条记忆！）
        self.wm_k_proj = nn.Linear(feat_dim, wm_dim)
        self.wm_v_proj = nn.Linear(feat_dim, D)
        self.wm_q_proj = nn.Linear(feat_dim, wm_dim)
        self.wm_out_proj = nn.Linear(D, feat_dim)
        self.copy_gate = nn.Sequential(
            nn.LayerNorm(feat_dim + D),
            nn.Linear(feat_dim + D, D // 2),
            nn.GELU(),
            nn.Linear(D // 2, 1),
        )

        # 3. 内部思考反馈投影（深波意图折回波池输入）
        self.think_proj = nn.Sequential(nn.LayerNorm(pd + D), nn.Linear(pd + D, D))

        # 4. 说话门（emit_gate）：决定「内部思考/暂停 (0)」还是「开口吐字 (1)」
        self.emit_gate = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, D // 2),
            nn.GELU(),
            nn.Linear(D // 2, 1),
        )

        # 5. 逐字节说话读出头
        self.speak_norm = nn.LayerNorm(feat_dim)
        self.speak_head = nn.Linear(feat_dim, vocab)

    def init_state(self, batch_size: int = 1, device: Optional[torch.device] = None):
        dev = device or self.emb.weight.device
        return self.pool.init_state(batch_size, dev)

    def step_tokens(
        self,
        token_ids: torch.Tensor,
        state,
        reset: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple]:
        x = self.emb(token_ids)
        hs, x_out, new_state = self.pool(x, state, reset=reset)
        return hs[0], hs[-1], x_out, new_state

    def encode_memory_bank(self, texts: Sequence[str], device: Optional[torch.device] = None) -> List[EncodedMemory]:
        """用 RMP 自身 WavePool 批量编码情景库，返回每条记忆的句级键 + 字节级匹配键 + 工作记忆槽。"""
        dev = device or self.emb.weight.device
        if not texts:
            return []
        raw_bytes = [encode_bytes(t) for t in texts]
        seqs = [[BOS_ID] + bs + [EOS_ID] for bs in raw_bytes]
        max_len = max(len(s) for s in seqs)
        padded = torch.full((len(seqs), max_len), PAD_ID, dtype=torch.long, device=dev)
        for i, s in enumerate(seqs):
            padded[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=dev)

        st = self.init_state(len(seqs), dev)
        h_sh, h_dp, x_out, _ = self.step_tokens(padded, st)

        out: List[EncodedMemory] = []
        for i, bs in enumerate(raw_bytes):
            L = len(bs)
            end_i = L + 1  # EOS 位置
            key = F.normalize(self.mem_key_proj(torch.cat([h_dp[i, end_i], x_out[i, end_i]], dim=-1)), dim=-1)
            # 内容字节段：索引 1 .. L
            c_sh = h_sh[i, 1 : 1 + L]
            c_dp = h_dp[i, 1 : 1 + L]
            c_x = x_out[i, 1 : 1 + L]
            c_feat = torch.cat([c_sh, c_dp, c_x], dim=-1)
            tok_keys = F.normalize(self.tok_match_proj(c_x), dim=-1)
            wm_keys = self.wm_k_proj(c_feat)
            wm_vals = self.wm_v_proj(c_feat)
            byte_ids = torch.tensor(bs, dtype=torch.long, device=dev)
            out.append(EncodedMemory(key=key, tok_keys=tok_keys, wm_keys=wm_keys, wm_vals=wm_vals, byte_ids=byte_ids))
        return out

    def memory_logits(
        self,
        h_deep_last: torch.Tensor,
        x_last: torch.Tensor,
        ctx_tok_vecs: torch.Tensor,
        encoded_mems: Sequence[EncodedMemory],
        used_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """深波语义查询 + RMP 字节级局部匹配，返回 (M+1,) logits（最后 1 维为 NULL 拒答槽）。"""
        dev = x_last.device
        q_deep = F.normalize(self.mem_query_proj(torch.cat([h_deep_last, x_last], dim=-1)), dim=-1)
        temp = self.mem_temp.abs().clamp(min=0.04, max=0.5)
        w_tok = F.softplus(self.tok_match_weight)

        scores = []
        for em in encoded_mems:
            s_deep = torch.dot(em.key, q_deep)
            # 细粒度 MaxSim（由 RMP 自身字节隐状态计算，能零样本匹配未见中文实体字节串！）
            sim_mat = ctx_tok_vecs @ em.tok_keys.T  # (Lq, Lm)
            # 取 top-k 最高匹配字节的均值（聚焦实体词而非停用词）
            k_top = min(9, sim_mat.shape[0])
            s_tok = sim_mat.max(dim=1).values.topk(k_top).values.mean()
            scores.append((s_deep + w_tok * s_tok) / temp)

        null_k = F.normalize(self.null_mem_key, dim=-1)
        null_score = (torch.dot(null_k, q_deep) + w_tok * self.null_score_bias) / temp
        scores.append(null_score)
        logits = torch.stack(scores, dim=0)

        if used_mask is not None and len(encoded_mems) > 0:
            mask = torch.cat([used_mask, torch.zeros(1, dtype=torch.bool, device=dev)], dim=0)
            logits = logits.masked_fill(mask, -1e9)
        return logits

    def emit_logit(self, h_shallow_last: torch.Tensor, h_deep_last: torch.Tensor, x_last: torch.Tensor) -> torch.Tensor:
        return self.emit_gate(torch.cat([h_shallow_last, h_deep_last, x_last], dim=-1)).squeeze(-1)

    def speak_log_probs(
        self,
        h_shallow: torch.Tensor,
        h_deep: torch.Tensor,
        x_out: torch.Tensor,
        wm: Optional[WorkingMemoryBuffer] = None,
    ) -> torch.Tensor:
        """计算下一字节对数概率分布；若工作记忆槽非空，融合极轻量 Cross-Attention 与 Copy Pointer。"""
        is_1d = h_shallow.dim() == 1
        if is_1d:
            h_shallow = h_shallow.unsqueeze(0)
            h_deep = h_deep.unsqueeze(0)
            x_out = x_out.unsqueeze(0)

        feat = torch.cat([h_shallow, h_deep, x_out], dim=-1)  # (T, feat_dim)
        if not self.use_working_memory or wm is None or wm.is_empty():
            logits = self.speak_head(self.speak_norm(feat))
            log_p = F.log_softmax(logits, dim=-1)
            return log_p[0] if is_1d else log_p

        wk, wv, wm_ids = wm.stacked()  # (Lwm, wm_dim), (Lwm, D), (Lwm,)
        q = self.wm_q_proj(feat)  # (T, wm_dim)
        attn_logits = (q @ wk.T) / math.sqrt(self.wm_dim)  # (T, Lwm)
        attn = F.softmax(attn_logits, dim=-1)
        ctx = attn @ wv  # (T, D)

        gen_logits = self.speak_head(self.speak_norm(feat + self.wm_out_proj(ctx)))
        p_gen = F.softmax(gen_logits, dim=-1)  # (T, V)

        p_copy = torch.sigmoid(self.copy_gate(torch.cat([feat, ctx], dim=-1)))  # (T, 1)
        p_ptr = torch.zeros_like(p_gen)
        idx = wm_ids.unsqueeze(0).expand(feat.shape[0], -1)
        p_ptr.scatter_add_(1, idx, attn)

        probs = ((1.0 - p_copy) * p_gen + p_copy * p_ptr).clamp(min=1e-8)
        log_p = probs.log()
        return log_p[0] if is_1d else log_p

    def internal_think_step(
        self,
        h_deep_last: torch.Tensor,
        x_last: torch.Tensor,
        state,
        mem_text: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple]:
        """执行 1 拍内部思考步（不吐字）：深波折回输入 + 静默读入召回记忆，返回更新后的隐状态与新字节特征。"""
        dev = x_last.device
        think_vec = self.think_proj(torch.cat([h_deep_last, x_last], dim=-1)).view(1, 1, self.D)
        hs, x_out, state = self.pool(think_vec, state)
        h_sh, h_dp, x_cur = hs[0][0, -1], hs[-1][0, -1], x_out[0, -1]
        new_tok_vecs = F.normalize(self.tok_match_proj(x_out[0]), dim=-1)

        if mem_text:
            ids = [MEM_ID] + encode_bytes(mem_text) + [SEP_ID]
            tok = torch.tensor([ids], dtype=torch.long, device=dev)
            hs_m, hdp_m, x_m, state = self.step_tokens(tok, state)
            h_sh, h_dp, x_cur = hs_m[0, -1], hdp_m[0, -1], x_m[0, -1]
            new_tok_vecs = F.normalize(self.tok_match_proj(x_m[0, 1:-1]), dim=-1)
        return h_sh, h_dp, x_cur, new_tok_vecs, state

    @torch.no_grad()
    def generate(
        self,
        query_text: str,
        bank: Optional[RMPEpisodicBank] = None,
        max_new_tokens: int = 96,
        max_think_total: int = 4,
        max_think_consec: int = 2,
        emit_thresh: float = 0.5,
        allow_mid_pause: bool = True,
    ) -> Dict[str, object]:
        self.eval()
        dev = self.emb.weight.device
        entries = bank.active_entries() if bank is not None else []
        encoded_mems = self.encode_memory_bank([e.text for e in entries], device=dev) if entries else []
        used_mask = torch.zeros(len(entries), dtype=torch.bool, device=dev)
        wm = WorkingMemoryBuffer(max_slots=2)

        q_ids = [BOS_ID] + encode_bytes(query_text) + [SEP_ID]
        q_tok = torch.tensor([q_ids], dtype=torch.long, device=dev)
        st = self.init_state(1, dev)
        h_sh_seq, h_dp_seq, x_seq, st = self.step_tokens(q_tok, st)
        h_sh, h_dp, x_cur = h_sh_seq[0, -1], h_dp_seq[0, -1], x_seq[0, -1]
        # 初始查询的 RMP 字节隐状态（用于未见实体匹配）
        ctx_tok_vecs = F.normalize(self.tok_match_proj(x_seq[0, 1:-1]), dim=-1)

        emitted: List[int] = []
        recalled_ids: List[str] = []
        trace: List[Dict[str, object]] = []
        think_total = 0
        think_consec = 0

        for _ in range(max_new_tokens + max_think_total + 4):
            p_emit = float(torch.sigmoid(self.emit_logit(h_sh, h_dp, x_cur)).item())
            can_pause_here = allow_mid_pause or (len(emitted) == 0)
            should_think = (
                max_think_total > 0
                and can_pause_here
                and think_total < max_think_total
                and think_consec < max_think_consec
                and p_emit < emit_thresh
            )

            if should_think:
                think_total += 1
                think_consec += 1
                chosen_eid = None
                chosen_text = None
                if len(entries) > 0:
                    mlogits = self.memory_logits(h_dp, x_cur, ctx_tok_vecs, encoded_mems, used_mask=used_mask)
                    m_idx = int(torch.argmax(mlogits).item())
                    if m_idx < len(entries):
                        used_mask[m_idx] = True
                        chosen_eid = entries[m_idx].event_id
                        chosen_text = entries[m_idx].text
                        recalled_ids.append(chosen_eid)
                        wm.push(encoded_mems[m_idx])
                trace.append({
                    "action": "THINK",
                    "after_bytes": len(emitted),
                    "after_chars": len(decode_bytes(emitted)),
                    "p_emit": round(p_emit, 3),
                    "recalled": chosen_eid,
                })
                h_sh, h_dp, x_cur, new_toks, st = self.internal_think_step(h_dp, x_cur, st, mem_text=chosen_text)
                if chosen_text:
                    # 融合新召回记忆的 RMP 字节状态（支撑二跳实体接力！）
                    ctx_tok_vecs = torch.cat([ctx_tok_vecs, new_toks], dim=0)
                continue

            think_consec = 0
            log_p = self.speak_log_probs(h_sh, h_dp, x_cur, wm=wm)
            log_p[PAD_ID] = -1e9
            log_p[BOS_ID] = -1e9
            log_p[SEP_ID] = -1e9
            log_p[MEM_ID] = -1e9
            log_p[THINK_ID] = -1e9
            nxt = int(torch.argmax(log_p).item())
            if nxt == EOS_ID:
                break
            emitted.append(nxt)
            if len(emitted) >= max_new_tokens:
                break
            tok = torch.tensor([[nxt]], dtype=torch.long, device=dev)
            h_sh_s, h_dp_s, x_s, st = self.step_tokens(tok, st)
            h_sh, h_dp, x_cur = h_sh_s[0, -1], h_dp_s[0, -1], x_s[0, -1]

        pre_thinks = sum(1 for t in trace if t["after_bytes"] == 0)
        mid_thinks = sum(1 for t in trace if t["after_bytes"] > 0)
        return {
            "text": decode_bytes(emitted),
            "recalled_ids": recalled_ids,
            "think_steps_total": think_total,
            "pre_speech_thinks": pre_thinks,
            "mid_speech_thinks": mid_thinks,
            "trace": trace,
        }
