"""任务 D（Phase 2 对比抽象投影器）：检验端到端对比学习的抽象空间能否真正打通「零关键词抽象情境 → 重复共现经历组」。

严格零主题泄露对照：
- 仅在 18 个独立训练主题上构造 (零关键词抽象情境, 对应共现经历组) 正例 + 难负例，训练一个轻量残差对比投影头；
- 在阿远评测库的 4 道全新零关键词抽象情境 + 12 道难负例上，与 E116 的 no_hierarchy / emergent_vec_mean 正面对照。
"""
from __future__ import annotations

import argparse
import json
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

from assoc_memory import FrozenTextEncoder
from tools.eval_assoc_e1 import build_ayuan_eval_world, build_disjoint_train_world


class ContrastiveAbstractProjector(nn.Module):
    """通用残差对比抽象投影器：把抽象情境查询与共现经历组映射到共享高层语义空间。"""

    def __init__(self, dim: int = 512, hidden: int = 128):
        super().__init__()
        self.q_proj = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(0.15), nn.Linear(hidden, dim))
        self.g_proj = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(0.15), nn.Linear(hidden, dim))
        nn.init.zeros_(self.q_proj[-1].weight)
        nn.init.zeros_(self.q_proj[-1].bias)
        nn.init.zeros_(self.g_proj[-1].weight)
        nn.init.zeros_(self.g_proj[-1].bias)
        self.scale = nn.Parameter(torch.tensor(0.35))
        self.gate = nn.Sequential(nn.Linear(3, 16), nn.GELU(), nn.Linear(16, 1))

    def project_query(self, q: torch.Tensor) -> torch.Tensor:
        return F.normalize(q + self.scale * self.q_proj(q), dim=-1)

    def project_group(self, g: torch.Tensor) -> torch.Tensor:
        return F.normalize(g + self.scale * self.g_proj(g), dim=-1)

    def match_logit(self, pq: torch.Tensor, pg_mat: torch.Tensor) -> Tuple[torch.Tensor, int]:
        sims = pg_mat @ pq  # (G,)
        best_i = int(torch.argmax(sims).item())
        top1 = sims[best_i]
        med = sims.median()
        gap = top1 - med
        logit = self.gate(torch.stack([top1, med, gap])).squeeze(-1)
        return logit, best_i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", type=str, default="42,43,44")
    args = ap.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== 任务 D（Phase 2 对比抽象投影器，零主题泄露，seeds={seeds}）===", flush=True)

    enc = FrozenTextEncoder("BAAI/bge-small-zh-v1.5", device=dev)
    tr_store, _, tr_examples = build_disjoint_train_world(enc)
    ev_store, _, eval_sets, _ = build_ayuan_eval_world(enc)

    # 1. 独立训练世界中的 12 个共现经历组 + 对应的零关键词抽象情境（与阿远零交集！）
    tr_groups = [
        ("g_cat", ["t_cat_1", "t_cat_2"], "出差不在家遇到噪音惊吓，家里胆小成员的临时托管安排是什么？"),
        ("g_gui", ["t_gui_1", "t_gui_2"], "业余音乐练习遇到器材损坏，一般送去哪儿找熟识师傅保养？"),
        ("g_run", ["t_run_1", "t_run_2"], "户外体能锻炼拉伤关节后，专业康复的休养与恢复原则是什么？"),
        ("g_plt", ["t_plt_1", "t_plt_2"], "长假出门没人照料居家环境时，阳台养护托付给了谁？"),
        ("g_bik", ["t_bik_1", "t_bik_2"], "日常上下班代步工具出机械异响，常去的维护渠道是哪家？"),
        ("g_mom", ["t_mom_1", "t_mom_2"], "长辈重要纪念日准备的心意物流现在存放在小区哪里？"),
        ("g_bak", ["t_bak_1", "t_bak_2"], "周末厨房手作老是翻车，排查出来的设备偏差原因是什么？"),
        ("g_jpn", ["t_jpn_1", "t_jpn_2"], "备考外语薄弱项感到焦虑时，下班后固定的沉浸复习地点在哪儿？"),
        ("g_cmp", ["t_cmp_1", "t_cmp_2"], "月底户外过夜清点物资发现缺件，借出去的东西在谁手里？"),
        ("g_lap", ["t_lap_1", "t_lap_2"], "居家办公主力硬件过热噪音大，平时送去哪儿做维护？"),
        ("g_wtr", ["t_wtr_1", "t_wtr_2"], "租房居家设施突发报错停摆，联系房东后安排了什么售后？"),
        ("g_chk", ["t_chk_1", "t_chk_2"], "健康指标亮红灯后，日常饮食结构做了哪些长期调整？"),
    ]
    tr_hard_negs = [ex.query for ex in tr_examples if ex.category in ("hard_neg", "insufficient")]

    tr_g_embs = torch.stack([
        F.normalize(torch.stack([tr_store.get(eid).embedding for eid in gids], dim=0).mean(dim=0), dim=-1)
        for _, gids, _ in tr_groups
    ], dim=0).to(dev)
    tr_q_embs = torch.stack([enc.encode_one(q) for _, _, q in tr_groups], dim=0).to(dev)
    tr_n_embs = torch.stack([enc.encode_one(q) for q in tr_hard_negs], dim=0).to(dev)

    # 2. 阿远评测库中的 4 个共现经历组 + 4 道全新零关键词抽象情境（同 E116）+ 12 难负例
    def find_id(sub: str) -> str:
        return next(e.event_id for e in ev_store.active_events() if sub in e.text)

    ev_groups = [
        ("pet_care", [find_id("一听见打雷就缩到我书桌底下发抖"), find_id("每次外面乌云密布要打雷"), "u_care_new"],
         "如果下周突然恶劣天气我临时回不了家，家里最让人牵挂的事该提前联系谁兜底？"),
        ("allergy_safe", [find_id("同事约着去吃海鲜大排档"), "u_med_kit"],
         "周末和朋友在外头聚餐时，我最需要提前避开的饮食雷区和应急准备是什么？"),
        ("film_hobby", [find_id("老富士胶片相机拿出来擦了擦镜头"), "u_film_shop", "u_trip_new"],
         "工作之外最能让我慢下来放松的视觉爱好，后续冲洗和远行计划分别是怎么安排的？"),
        ("side_project", [find_id("画了画我一直想做的那个独立记账 App"), find_id("写了两个小时独立记账 App"), find_id("同事劝我先利用周末把原型跑通")],
         "关于离开大厂自己创业做副业这件事，我目前做到哪一步、身边人是怎么劝我的？"),
    ]
    ev_g_embs = torch.stack([
        F.normalize(torch.stack([ev_store.get(eid).embedding for eid in gids], dim=0).mean(dim=0), dim=-1)
        for _, gids, _ in ev_groups
    ], dim=0).to(dev)
    ev_q_embs = torch.stack([enc.encode_one(q) for _, _, q in ev_groups], dim=0).to(dev)
    ev_n_embs = torch.stack([enc.encode_one(ex.query) for ex in eval_sets["hard_neg"]], dim=0).to(dev)

    # 先测无训练基线（emergent_vec_mean：纯余弦 Top-1 + 训练集校准阈值）
    raw_pos = (tr_q_embs @ tr_g_embs.T).max(dim=1).values.cpu().tolist()
    raw_neg = (tr_n_embs @ tr_g_embs.T).max(dim=1).values.cpu().tolist()
    best_t, best_acc = 0.55, -1.0
    for t in [x / 100.0 for x in range(35, 75, 2)]:
        acc = 0.5 * (sum(s >= t for s in raw_pos) / len(raw_pos) + sum(s < t for s in raw_neg) / len(raw_neg))
        if acc > best_acc:
            best_acc, best_t = acc, t
    raw_ev_sims = ev_q_embs @ ev_g_embs.T
    raw_ev_hit = sum(
        1 for i in range(len(ev_groups))
        if int(torch.argmax(raw_ev_sims[i]).item()) == i and float(raw_ev_sims[i, i].item()) >= best_t
    ) / len(ev_groups)
    raw_ev_fpr = sum(
        1 for i in range(ev_n_embs.shape[0])
        if float((ev_g_embs @ ev_n_embs[i]).max().item()) >= best_t
    ) / ev_n_embs.shape[0]

    # 3 种子训练 ContrastiveAbstractProjector
    seed_hits, seed_fprs = [], []
    for sd in seeds:
        torch.manual_seed(sd)
        proj = ContrastiveAbstractProjector(dim=enc.dim, hidden=128).to(dev)
        opt = torch.optim.AdamW(proj.parameters(), lr=3e-3, weight_decay=2e-2)
        for _ep in range(args.epochs):
            proj.train()
            pq = proj.project_query(tr_q_embs)  # (12, D)
            pg = proj.project_group(tr_g_embs)  # (12, D)
            pn = proj.project_query(tr_n_embs)  # (N_neg, D)

            # 1) InfoNCE 对比损失：把抽象情境拉向对应经历组，推远其他组
            sim_mat = (pq @ pg.T) / 0.10
            l_nce = F.cross_entropy(sim_mat, torch.arange(len(tr_groups), device=dev))
            # 2) 难负例推远边界损失
            l_neg = F.relu((pn @ pg.T).max(dim=1).values - 0.45).pow(2).mean()
            # 3) 门控二分类损失
            pos_logits = torch.stack([proj.match_logit(pq[i], pg)[0] for i in range(len(tr_groups))])
            neg_logits = torch.stack([proj.match_logit(pn[i], pg)[0] for i in range(pn.shape[0])])
            l_gate = (
                F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
                + 1.3 * F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
            )
            loss = l_nce + 2.0 * l_neg + l_gate
            opt.zero_grad(); loss.backward(); opt.step()

        proj.eval()
        with torch.no_grad():
            epq = proj.project_query(ev_q_embs)
            epg = proj.project_group(ev_g_embs)
            epn = proj.project_query(ev_n_embs)
            hit = 0
            for i in range(len(ev_groups)):
                lg, bi = proj.match_logit(epq[i], epg)
                if bi == i and float(torch.sigmoid(lg).item()) >= 0.5:
                    hit += 1
            fp = 0
            for i in range(epn.shape[0]):
                lg, _ = proj.match_logit(epn[i], epg)
                if float(torch.sigmoid(lg).item()) >= 0.5:
                    fp += 1
            seed_hits.append(hit / len(ev_groups))
            seed_fprs.append(fp / epn.shape[0])

    res = {
        "emergent_vec_mean_calibrated": {
            "thresh": best_t,
            "novel_abstract_hit": round(raw_ev_hit, 4),
            "hard_neg_fpr": round(raw_ev_fpr, 4),
        },
        "contrastive_abstract_projector": {
            "novel_abstract_hit": round(sum(seed_hits) / len(seed_hits), 4),
            "hard_neg_fpr": round(sum(seed_fprs) / len(seed_fprs), 4),
            "per_seed_hit": [round(x, 4) for x in seed_hits],
            "per_seed_fpr": [round(x, 4) for x in seed_fprs],
        },
    }
    print("\n=== 任务 D（Phase 2 对比抽象投影器）结果（3 种子均值，零主题泄露）===")
    print(f"  emergent_vec_mean (校准阈值={best_t:.2f}) | 零关键词抽象情境命中={res['emergent_vec_mean_calibrated']['novel_abstract_hit']:.1%} | 难负例FPR={res['emergent_vec_mean_calibrated']['hard_neg_fpr']:.1%}")
    print(f"  contrastive_abstract_projector       | 零关键词抽象情境命中={res['contrastive_abstract_projector']['novel_abstract_hit']:.1%} | 难负例FPR={res['contrastive_abstract_projector']['hard_neg_fpr']:.1%}", flush=True)

    elapsed = round(time.time() - t0, 1)
    out_path = ROOT / f"runs/{time.strftime('%Y%m%d-%H%M%S')}_assoc_phase2_contrastive.json"
    out_path.write_text(
        json.dumps({"experiment": "Phase2_contrastive_abstract", "seeds": seeds, "elapsed_s": elapsed, "results": res}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"结果已保存：{out_path}（耗时 {elapsed}s）", flush=True)


if __name__ == "__main__":
    main()
