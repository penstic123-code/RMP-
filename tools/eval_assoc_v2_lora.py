"""任务 B（v2.3）：解除冻结编码器天花板 — BGE 顶 2 层轻量 LoRA + 纯通用神经回想核端到端联合训练。

遵循「苦涩的教训」：
- 零手工特征工程、零外部噪声合成数据；
- 在 BAAI/bge-small-zh-v1.5 顶 2 层（layer 2, 3）的 query / value 投影上挂载极轻量 LoRA（r=8，约 32K 参数），
  底层与原权重永久冻结，并加余弦锚定正则防止偏离通用语义空间；
- 多跳悬念对比损失 + Token 对齐验证损失端到端反传至 BGE 顶层 LoRA，让表征空间本身学会「多跳悬念匹配」；
- 在完全相同的 E1 零主题泄露基准（18 独立主题训练 → 24 篇阿远日记评测，3 种子）上与 E117 冻结编码器版严格对照。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if not os.environ.get("HF_HOME") and (ROOT / ".pytest_cache" / "hf").exists():
    os.environ["HF_HOME"] = str(ROOT / ".pytest_cache" / "hf")

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from assoc_neural_v2 import (
    GeneralNeuralRecallNet,
    NeuralEpisodicStore,
    PureNeuralGraph,
    TokenLevelEncoder,
)
from tools.eval_assoc_e1 import build_ayuan_eval_world, build_disjoint_train_world
from tools.eval_assoc_v2 import calibrate_v2_thresh, convert_store_to_neural, eval_v2, run_v2_causal


class LoRALinear(nn.Module):
    """极轻量低秩适配层（零初始化 B 矩阵，支持运行时开关以便零显存读取原版冻结表征）。"""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scale = alpha / r
        self.enabled: bool = True
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.enabled:
            return out
        return out + (F.linear(F.linear(x, self.A), self.B)) * self.scale


class LoRATokenEncoder(nn.Module):
    """带顶 2 层轻量 LoRA 的检索编码器（兼容 TokenLevelEncoder 接口，支持双表征无漂移冻结对照）。"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", device: str = "cuda", r: int = 8, max_len: int = 96):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.max_len = max_len
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)

        layers = self.model.encoder.layer
        self.lora_modules: List[LoRALinear] = []
        self.lora_params: List[nn.Parameter] = []
        for layer in layers[-2:]:
            sa = layer.attention.self
            sa.query = LoRALinear(sa.query, r=r).to(device)
            sa.value = LoRALinear(sa.value, r=r).to(device)
            self.lora_modules.extend([sa.query, sa.value])
            self.lora_params.extend([sa.query.A, sa.query.B, sa.value.A, sa.value.B])

        self.dim: int = int(self.model.config.hidden_size)
        self._cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._frozen_cls_cache: Dict[str, torch.Tensor] = {}

    def set_lora_enabled(self, flag: bool) -> None:
        for m in self.lora_modules:
            m.enabled = flag

    @torch.no_grad()
    def encode_frozen_cls(self, texts: Sequence[str], batch_size: int = 64) -> torch.Tensor:
        """临时旁路 LoRA 读取原版冻结 BGE 的句向量（零额外显存，自动缓存）。"""
        missing = [t for t in texts if t not in self._frozen_cls_cache]
        if missing:
            was_training = self.model.training
            self.model.eval()
            self.set_lora_enabled(False)
            try:
                for i in range(0, len(missing), batch_size):
                    batch = missing[i : i + batch_size]
                    enc = self.tok(batch, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt").to(self.device)
                    hs = self.model(**enc).last_hidden_state.float()
                    cls_e = F.normalize(hs[:, 0], dim=-1).cpu()
                    for t, c in zip(batch, cls_e):
                        self._frozen_cls_cache[t] = c
            finally:
                self.set_lora_enabled(True)
                if was_training:
                    self.model.train()
        return torch.stack([self._frozen_cls_cache[t] for t in texts], dim=0)

    @torch.no_grad()
    def frozen_max_sim(self, query_text: str, event_texts: Sequence[str]) -> float:
        if not event_texts:
            return 0.0
        q_f = self.encode_frozen_cls([query_text])[0]
        e_f = self.encode_frozen_cls(event_texts)
        return float((e_f @ q_f).max().item())

    def clear_cache(self) -> None:
        self._cache.clear()

    def encode_trainable(self, texts: Sequence[str]) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """带梯度的批量编码，返回 (cls_embs[B, D], tok_list, mean_toks[B, D])。"""
        enc = self.tok(list(texts), padding=True, truncation=True, max_length=self.max_len, return_tensors="pt").to(self.device)
        out = self.model(**enc)
        hs = out.last_hidden_state.float()
        cls_embs = F.normalize(hs[:, 0], dim=-1)
        mask = enc["attention_mask"]
        tok_list, mean_list = [], []
        for i in range(len(texts)):
            vl = int(mask[i].sum().item())
            th = hs[i, 1 : vl - 1] if vl > 2 else hs[i, :vl]
            th = F.normalize(th, dim=-1)
            tok_list.append(th)
            mean_list.append(F.normalize(th.mean(dim=0), dim=-1))
        return cls_embs, tok_list, torch.stack(mean_list, dim=0)

    @torch.no_grad()
    def encode_both(self, texts: Sequence[str], batch_size: int = 64) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        if isinstance(texts, str):
            texts = [texts]
        missing = [t for t in texts if t not in self._cache]
        if missing:
            self.model.eval()
            for i in range(0, len(missing), batch_size):
                batch = missing[i : i + batch_size]
                cls_e, tok_l, _ = self.encode_trainable(batch)
                for t, c, th in zip(batch, cls_e.cpu(), tok_l):
                    self._cache[t] = (c, th.cpu())
        return [self._cache[t] for t in texts]

    def encode_one_both(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encode_both([text])[0]

    def encode_one(self, text: str) -> torch.Tensor:
        return self.encode_one_both(text)[0]

    def encode(self, texts: Sequence[str], batch_size: int = 64) -> torch.Tensor:
        return torch.stack([x[0] for x in self.encode_both(texts, batch_size=batch_size)], dim=0)


def train_end_to_end_lora(
    enc: LoRATokenEncoder,
    net: GeneralNeuralRecallNet,
    tr_events_raw: List[Tuple[str, int, str, str, str]],  # (eid, day, src, txt, status)
    tr_examples: Sequence,
    epochs: int = 24,
    lora_lr: float = 1.5e-4,
    net_lr: float = 3e-3,
    anchor_w: float = 1.5,
    neg_margin_w: float = 2.5,
    seed: int = 42,
) -> Dict[str, float]:
    """端到端联合训练：先缓存在冻结 BGE 下的锚定向量，再同步更新顶 2 层 LoRA 与纯通用神经回想核。"""
    rng = random.Random(seed)
    dev = enc.device

    # 记录冻结原版 BGE 的锚定表征，防止 LoRA 过拟合 18 个训练主题而破坏通用语义空间
    active_raw = [x for x in tr_events_raw if x[4] == "active"]
    eid_to_idx = {x[0]: i for i, x in enumerate(active_raw)}
    ev_texts = [x[3] for x in active_raw]
    with torch.no_grad():
        enc.model.eval()
        enc.set_lora_enabled(False)
        anchor_cls, _, anchor_mean = enc.encode_trainable(ev_texts)
        anchor_cls = anchor_cls.detach()
        anchor_mean = anchor_mean.detach()
        enc.set_lora_enabled(True)
    # 预先缓存所有训练查询的原版冻结最高相似度（供双表征拒答门控使用）
    frozen_q_max = {ex.query: enc.frozen_max_sim(ex.query, ev_texts) for ex in tr_examples}

    opt = torch.optim.AdamW(
        [
            {"params": net.parameters(), "lr": net_lr, "weight_decay": 1e-2},
            {"params": enc.lora_params, "lr": lora_lr, "weight_decay": 1e-2},
        ]
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=1e-4)

    order = list(range(len(tr_examples)))
    last_losses = {"rank": 0.0, "query": 0.0, "cont": 0.0, "valid": 0.0, "anchor": 0.0}

    for ep in range(epochs):
        enc.model.train()
        net.train()
        rng.shuffle(order)
        tr = tq = tc = tv = ta = 0.0
        nr = nq = nc = nv = 0

        # 每个 epoch 先以当前 LoRA 重建轻量缓存用于负例采样，再对本轮样本端到端反传
        opt.zero_grad()
        acc = 0
        for idx in order:
            ex = tr_examples[idx]
            if ex.gold_ids and any(gid not in eid_to_idx for gid in ex.gold_ids):
                continue

            # 实时通过 LoRA 编码查询与所有活跃事件（仅 43 条事件，单次 batch forward < 8ms!）
            all_cls, all_toks, all_means = enc.encode_trainable(ev_texts)
            q_cls_b, q_tok_l, q_mean_b = enc.encode_trainable([ex.query])
            q0_cls, q0_tok, q0_mean = q_cls_b[0], q_tok_l[0], q_mean_b[0]

            # 语义锚定正则：保持通用语义几何不漂移
            l_anc = (1.0 - (all_cls * anchor_cls).sum(dim=-1)).mean() + (1.0 - (all_means * anchor_mean).sum(dim=-1)).mean()
            loss = anchor_w * l_anc
            ta += float(l_anc.item())

            sims_q0 = all_cls @ q0_cls
            max_sim = float(sims_q0.max().item())
            f_max = frozen_q_max[ex.query]
            k0 = min(12, len(active_raw))
            r0_idx = torch.topk(sims_q0, k=k0).indices.tolist()
            if ex.gold_ids:
                g0_pos = eid_to_idx[ex.gold_ids[0]]
                if g0_pos not in r0_idx:
                    r0_idx = [g0_pos] + r0_idx[:-1]

            def make_ev(i_pos: int):
                return type("E", (), {
                    "event_id": active_raw[i_pos][0],
                    "cls_emb": all_cls[i_pos],
                    "tok_embs": all_toks[i_pos],
                    "mean_tok_emb": all_means[i_pos],
                })()

            r0_cands = [make_ev(i) for i in r0_idx]
            r0_logits, _ = net.score_candidates_batch(q0_cls, q0_tok, q0_mean, [], None, r0_cands)

            if ex.category in ("direct", "superseded", "partial_only") and ex.gold_ids:
                g0_pos = eid_to_idx[ex.gold_ids[0]]
                g0_i = r0_idx.index(g0_pos)
                l_r0 = F.cross_entropy(r0_logits.unsqueeze(0), torch.tensor([g0_i], device=dev))
                loss = loss + l_r0; tr += float(l_r0.item()); nr += 1

                p_toks = all_toks[g0_pos]
                p_repr = net.encode_path_state(q0_tok, p_toks, 1)
                q_next = net.update_query(q0_tok, q0_mean, p_toks)
                n_sims = (all_means @ q_next).clone()
                n_sims[g0_pos] = -1e9
                n_idx = torch.topk(n_sims, k=min(8, len(active_raw) - 1)).indices.tolist()
                n_logits, n_reprs = net.score_candidates_batch(q0_cls, q0_tok, q_next, [all_cls[g0_pos]], p_toks, [make_ev(i) for i in n_idx])
                best_ni = int(torch.argmax(n_logits).item())
                l_c = F.binary_cross_entropy_with_logits(net.continue_logit(p_repr, n_reprs[best_ni]), torch.tensor(0.0, device=dev))
                l_v = F.binary_cross_entropy_with_logits(net.valid_logit(p_repr, max_sim, f_max), torch.tensor(1.0, device=dev))
                loss = loss + 1.2 * l_c + 1.2 * l_v
                tc += float(l_c.item()); nc += 1; tv += float(l_v.item()); nv += 1

            elif ex.category == "cross" and len(ex.gold_ids) >= 2:
                # 修复：逐跳循环监督，同时支持 2 跳与 3 跳长链！
                g0_pos = eid_to_idx[ex.gold_ids[0]]
                g0_i = r0_idx.index(g0_pos)
                l_r0 = F.cross_entropy(r0_logits.unsqueeze(0), torch.tensor([g0_i], device=dev))
                loss = loss + l_r0; tr += float(l_r0.item()); nr += 1

                visited_pos = [g0_pos]
                qt = q0_mean
                for step_i in range(1, len(ex.gold_ids)):
                    tgt_pos = eid_to_idx[ex.gold_ids[step_i]]
                    p_toks = torch.cat([all_toks[p] for p in visited_pos], dim=0)
                    p_cls_list = [all_cls[p] for p in visited_pos]
                    p_repr = net.encode_path_state(q0_tok, p_toks, len(visited_pos))
                    q_next = net.update_query(q0_tok, qt, p_toks)

                    q_sims = (all_means @ q_next).clone()
                    for vp in visited_pos:
                        q_sims[vp] = -1e9
                    l_q = F.cross_entropy((q_sims / 0.08).unsqueeze(0), torch.tensor([tgt_pos], device=dev))
                    loss = loss + 1.2 * l_q; tq += float(l_q.item()); nq += 1

                    n_idx = torch.topk(q_sims, k=min(8, len(active_raw) - len(visited_pos))).indices.tolist()
                    if tgt_pos not in n_idx:
                        n_idx = [tgt_pos] + n_idx[:-1]
                    t_i = n_idx.index(tgt_pos)
                    n_logits, n_reprs = net.score_candidates_batch(q0_cls, q0_tok, q_next, p_cls_list, p_toks, [make_ev(i) for i in n_idx])
                    l_rn = F.cross_entropy(n_logits.unsqueeze(0), torch.tensor([t_i], device=dev))
                    l_c = F.binary_cross_entropy_with_logits(net.continue_logit(p_repr, n_reprs[t_i]), torch.tensor(1.0, device=dev))
                    loss = loss + l_rn + l_c
                    tr += float(l_rn.item()); nr += 1; tc += float(l_c.item()); nc += 1

                    visited_pos.append(tgt_pos)
                    qt = q_next

                full_toks = torch.cat([all_toks[p] for p in visited_pos], dim=0)
                full_repr = net.encode_path_state(q0_tok, full_toks, len(visited_pos))
                l_v = F.binary_cross_entropy_with_logits(net.valid_logit(full_repr, max_sim, f_max), torch.tensor(1.0, device=dev))
                loss = loss + 1.2 * l_v; tv += float(l_v.item()); nv += 1

            elif ex.category in ("hard_neg", "insufficient"):
                l_neg_margin = F.relu(sims_q0.max() - 0.50).pow(2)
                loss = loss + neg_margin_w * l_neg_margin

                top_pos = r0_idx[int(torch.argmax(r0_logits).item())]
                p_toks = all_toks[top_pos]
                p_repr = net.encode_path_state(q0_tok, p_toks, 1)
                l_v1 = F.binary_cross_entropy_with_logits(net.valid_logit(p_repr, max_sim, f_max), torch.tensor(0.0, device=dev))
                loss = loss + 1.3 * l_v1; tv += float(l_v1.item()); nv += 1

                q_next = net.update_query(q0_tok, q0_mean, p_toks)
                n_sims = (all_means @ q_next).clone()
                n_sims[top_pos] = -1e9
                n_idx = torch.topk(n_sims, k=min(6, len(active_raw) - 1)).indices.tolist()
                n_logits, n_reprs = net.score_candidates_batch(q0_cls, q0_tok, q_next, [all_cls[top_pos]], p_toks, [make_ev(i) for i in n_idx])
                best_ni = int(torch.argmax(n_logits).item())
                l_c = F.binary_cross_entropy_with_logits(net.continue_logit(p_repr, n_reprs[best_ni]), torch.tensor(0.0, device=dev))
                fake_toks = torch.cat([p_toks, all_toks[n_idx[best_ni]]], dim=0)
                fake_repr = net.encode_path_state(q0_tok, fake_toks, 2)
                l_v2 = F.binary_cross_entropy_with_logits(net.valid_logit(fake_repr, max_sim, f_max), torch.tensor(0.0, device=dev))
                loss = loss + 0.5 * l_c + 1.0 * l_v2
                tc += float(l_c.item()); nc += 1; tv += float(l_v2.item()); nv += 1

            (loss / 8.0).backward()
            acc += 1
            if acc >= 8:
                torch.nn.utils.clip_grad_norm_(list(net.parameters()) + enc.lora_params, 2.0)
                opt.step(); opt.zero_grad(); acc = 0

        if acc > 0:
            torch.nn.utils.clip_grad_norm_(list(net.parameters()) + enc.lora_params, 2.0)
            opt.step(); opt.zero_grad()
        sched.step()
        last_losses = {
            "rank": round(tr / max(1, nr), 4),
            "query": round(tq / max(1, nq), 4),
            "cont": round(tc / max(1, nc), 4),
            "valid": round(tv / max(1, nv), 4),
            "anchor": round(ta / max(1, len(tr_examples)), 4),
        }
    enc.model.eval()
    enc.clear_cache()
    return last_losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--seeds", type=str, default="42,43,44")
    args = ap.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== 任务 B（v2.3）：BGE 顶 2 层 LoRA + 纯通用神经回想核（seeds={seeds}, r={args.lora_r}）===", flush=True)

    # 先提取静态事件文本列表
    base_enc = TokenLevelEncoder("BAAI/bge-small-zh-v1.5", device=dev)
    tr_s0, _, tr_examples = build_disjoint_train_world(base_enc)
    ev_s0, _, eval_sets, smoke6 = build_ayuan_eval_world(base_enc)
    tr_raw = [(e.event_id, e.day, e.source, e.text, e.status) for e in tr_s0.all_events(include_deleted=True)]
    del base_enc
    if dev == "cuda":
        torch.cuda.empty_cache()

    seed_runs = []
    for sd in seeds:
        torch.manual_seed(sd)
        enc = LoRATokenEncoder("BAAI/bge-small-zh-v1.5", device=dev, r=args.lora_r)
        net = GeneralNeuralRecallNet(d_model=32, n_heads=4, n_layers=2).to(dev)
        lora_n = sum(p.numel() for p in enc.lora_params)
        net_n = sum(p.numel() for p in net.parameters())

        losses = train_end_to_end_lora(enc, net, tr_raw, tr_examples, epochs=args.epochs, seed=sd)

        # 用训好的 LoRA 编码器重新编码训练库（校准阈值）与未见阿远评测库
        tr_store = convert_store_to_neural(tr_s0, enc)
        tr_graph = PureNeuralGraph(tr_store, max_neighbors=16)
        abs_t = calibrate_v2_thresh(net, tr_store, tr_graph, tr_examples, budget=args.budget)
        print(f"[seed={sd}] lora={lora_n} net={net_n} | losses={losses} | abstain_thresh={abs_t:.2f}", flush=True)

        ev_store = convert_store_to_neural(ev_s0, enc)
        ev_graph = PureNeuralGraph(ev_store, max_neighbors=16)

        res_sd = {}
        for name, mode in [("v2_lora_hybrid", "hybrid"), ("v2_lora_graph", "graph_only"), ("v2_lora_full", "full_only")]:
            res_sd[name] = eval_v2(net, ev_store, ev_graph, eval_sets, smoke6, mode=mode, abstain_thresh=abs_t, budget=args.budget)
        causal_sd = run_v2_causal(net, ev_store, eval_sets["cross"], abstain_thresh=abs_t, budget=args.budget)

        seed_runs.append({
            "seed": sd,
            "lora_params": lora_n,
            "net_params": net_n,
            "abstain_thresh": abs_t,
            "losses": losses,
            "results": {k: {mk: mv for mk, mv in v.items() if mk != "details_cross"} for k, v in res_sd.items()},
            "causal_ablation": causal_sd,
        })
        del enc, net
        if dev == "cuda":
            torch.cuda.empty_cache()

    mkeys = ["cross_exact", "cross_single", "cross_target_hit", "direct_exact", "hard_neg_fpr", "superseded_latest_acc", "insufficient_abstain_acc", "smoke6_hit"]
    summary = {}
    print(f"\n=== 任务 B（v2.3 BGE-LoRA + 纯神经核）E1 汇总（{len(seeds)} 种子均值，零主题泄露）===")
    header = f"{'方法':<18} | {'跨事件全中':>8} | {'跨事件单中':>8} | {'二跳目标':>8} | {'直接回想':>8} | {'难负例FPR↓':>9} | {'已变更↑':>7} | {'缺证拒答↑':>8} | {'冒烟6题':>7}"
    print(header)
    print("-" * len(header))
    for name in ("v2_lora_hybrid", "v2_lora_graph", "v2_lora_full"):
        summary[name] = {mk: round(sum(r["results"][name][mk] for r in seed_runs) / len(seed_runs), 4) for mk in mkeys}
        m = summary[name]
        print(
            f"{name:<18} | {m['cross_exact']:>8.1%} | {m['cross_single']:>8.1%} | {m['cross_target_hit']:>8.1%} | "
            f"{m['direct_exact']:>8.1%} | {m['hard_neg_fpr']:>9.1%} | {m['superseded_latest_acc']:>7.1%} | "
            f"{m['insufficient_abstain_acc']:>8.1%} | {m['smoke6_hit']:>7.1%}",
            flush=True,
        )

    causal_mean = {k: round(sum(r["causal_ablation"][k] for r in seed_runs) / len(seed_runs), 4) for k in seed_runs[0]["causal_ablation"]}
    print(
        f"\n因果消融（v2_lora_hybrid）: 完整库 {causal_mean['full_target_hit']:.1%} | "
        f"移桥 {causal_mean['drop_bridge_target_hit']:.1%} (Δ={-causal_mean['bridge_causal_drop']:+.1%}) | "
        f"移无关 {causal_mean['drop_unrelated_target_hit']:.1%} (Δ={-causal_mean['unrelated_drop']:+.1%})",
        flush=True,
    )

    elapsed = round(time.time() - t0, 1)
    out_path = ROOT / f"runs/{time.strftime('%Y%m%d-%H%M%S')}_assoc_v2_lora_e1.json"
    out_path.write_text(
        json.dumps(
            {
                "experiment": "v2.3_bge_lora_e1",
                "seeds": seeds,
                "lora_r": args.lora_r,
                "epochs": args.epochs,
                "elapsed_s": elapsed,
                "summary_mean": summary,
                "causal_ablation_mean": causal_mean,
                "seed_runs": seed_runs,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n结果已保存：{out_path}（耗时 {elapsed}s）", flush=True)


if __name__ == "__main__":
    main()
