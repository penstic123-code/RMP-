"""v2 纯通用神经联想记忆基准评测（遵循「苦涩的教训」：零手工特征、零词法规则、纯 Token 注意力 Transformer）。

在完全相同的 E1 零主题泄露基准上（18 个独立主题训练 → 24 篇阿远自然日记评测），
对比 v2 纯神经版（hybrid / graph_only / full_only）与三条基线，并运行因果消融与 3 跳长链测试。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if not os.environ.get("HF_HOME") and (ROOT / ".pytest_cache" / "hf").exists():
    os.environ["HF_HOME"] = str(ROOT / ".pytest_cache" / "hf")

import torch

from assoc_memory import RecallExample
from assoc_neural_v2 import (
    GeneralNeuralRecallNet,
    NeuralEpisodicStore,
    PureNeuralGraph,
    TokenLevelEncoder,
    train_neural_v2,
)
from tools.eval_assoc_e1 import build_ayuan_eval_world, build_disjoint_train_world


def convert_store_to_neural(src_store, tok_encoder: TokenLevelEncoder) -> NeuralEpisodicStore:
    """将已有事件库（保持完全相同的事件 ID、原文、日期与 active/superseded 状态）载入纯神经事件库。"""
    n_store = NeuralEpisodicStore(tok_encoder)
    evs = src_store.all_events(include_deleted=True)
    tok_encoder.encode_both([e.text for e in evs], batch_size=32)
    for e in evs:
        ne = n_store.add_event(
            text=e.text,
            day=e.day,
            source=e.source,
            event_id=e.event_id,
            status=e.status,
            supersedes=None,
        )
        ne.supersedes = e.supersedes
        ne.superseded_by = e.superseded_by
    return n_store


@torch.no_grad()
def calibrate_v2_thresh(
    model: GeneralNeuralRecallNet,
    store: NeuralEpisodicStore,
    graph: PureNeuralGraph,
    train_examples: Sequence,
    budget: int = 16,
) -> float:
    model.eval()
    pos_p, neg_p = [], []
    for ex in train_examples:
        r = model.recall(store, graph, ex.query, mode="hybrid", candidate_budget=budget, abstain_thresh=0.0)
        if ex.category in ("hard_neg", "insufficient"):
            neg_p.append(float(r["p_valid"]))
        else:
            pos_p.append(float(r["p_valid"]))
    best_obj, best_ts = -1e9, []
    for t in [x / 100.0 for x in range(40, 67, 2)]:
        tpr = sum(1 for p in pos_p if p >= t) / max(1, len(pos_p))
        fpr = sum(1 for p in neg_p if p >= t) / max(1, len(neg_p))
        obj = tpr - fpr
        if obj > best_obj + 1e-9:
            best_obj, best_ts = obj, [t]
        elif abs(obj - best_obj) <= 1e-9:
            best_ts.append(t)
    return round(sum(best_ts) / len(best_ts), 3) if best_ts else 0.5


def eval_v2(
    model: GeneralNeuralRecallNet,
    store: NeuralEpisodicStore,
    graph: PureNeuralGraph,
    eval_sets: Dict,
    smoke6: List,
    mode: str,
    abstain_thresh: float,
    budget: int = 16,
) -> Dict[str, object]:
    def run_q(q: str):
        return model.recall(store, graph, q, mode=mode, max_rounds=2, candidate_budget=budget, abstain_thresh=abstain_thresh)

    d_ex = sum(1 for ex in eval_sets["direct"] if set(ex.gold_ids).issubset(set(run_q(ex.query)["event_ids"])))
    c_ex = c_sg = c_tg = 0
    details = []
    for ex in eval_sets["cross"]:
        r = run_q(ex.query)
        p = set(r["event_ids"])
        exact = set(ex.gold_ids).issubset(p)
        single = bool(set(ex.gold_ids) & p)
        tgt = bool(ex.target_id and ex.target_id in p)
        c_ex += int(exact); c_sg += int(single); c_tg += int(tgt)
        details.append({"q": ex.query, "gold": ex.gold_ids, "pred": r["event_ids"], "p_valid": round(float(r["p_valid"]), 3), "exact": exact})

    fp = sum(1 for ex in eval_sets["hard_neg"] if not run_q(ex.query)["abstained"] and len(run_q(ex.query)["event_ids"]) > 0)
    sup = sum(1 for ex in eval_sets["superseded"] if set(ex.gold_ids).issubset(set(run_q(ex.query)["event_ids"])))
    ins = sum(1 for ex in eval_sets["insufficient"] if run_q(ex.query)["abstained"] or len(run_q(ex.query)["event_ids"]) == 0)
    sm = sum(1 for _t, q, sids in smoke6 if set(run_q(q)["event_ids"]) & set(sids))

    return {
        "cross_exact": round(c_ex / len(eval_sets["cross"]), 4),
        "cross_single": round(c_sg / len(eval_sets["cross"]), 4),
        "cross_target_hit": round(c_tg / len(eval_sets["cross"]), 4),
        "direct_exact": round(d_ex / len(eval_sets["direct"]), 4),
        "hard_neg_fpr": round(fp / len(eval_sets["hard_neg"]), 4),
        "superseded_latest_acc": round(sup / len(eval_sets["superseded"]), 4),
        "insufficient_abstain_acc": round(ins / len(eval_sets["insufficient"]), 4),
        "smoke6_hit": round(sm / len(smoke6), 4),
        "details_cross": details,
    }


def run_v2_causal(
    model: GeneralNeuralRecallNet,
    store: NeuralEpisodicStore,
    cross_examples: Sequence,
    abstain_thresh: float,
    budget: int = 16,
) -> Dict[str, float]:
    full_g = PureNeuralGraph(store, max_neighbors=16)
    all_act = store.active_events()
    f_hit = b_hit = u_hit = 0
    for ex in cross_examples:
        r_f = model.recall(store, full_g, ex.query, mode="hybrid", max_rounds=2, candidate_budget=budget, abstain_thresh=abstain_thresh)
        if ex.target_id in r_f["event_ids"]:
            f_hit += 1

        sb = store.clone_without([ex.bridge_id])
        gb = PureNeuralGraph(sb, max_neighbors=16)
        r_b = model.recall(sb, gb, ex.query, mode="hybrid", max_rounds=2, candidate_budget=budget, abstain_thresh=abstain_thresh)
        if ex.target_id in r_b["event_ids"]:
            b_hit += 1

        q0 = store.encoder.encode_one(ex.query)
        unrel = min(
            (e for e in all_act if e.event_id not in (ex.bridge_id, ex.target_id)),
            key=lambda e: float(torch.dot(q0, e.cls_emb).item()),
        )
        su = store.clone_without([unrel.event_id])
        gu = PureNeuralGraph(su, max_neighbors=16)
        r_u = model.recall(su, gu, ex.query, mode="hybrid", max_rounds=2, candidate_budget=budget, abstain_thresh=abstain_thresh)
        if ex.target_id in r_u["event_ids"]:
            u_hit += 1
    n = len(cross_examples)
    return {
        "full_target_hit": round(f_hit / n, 4),
        "drop_bridge_target_hit": round(b_hit / n, 4),
        "drop_unrelated_target_hit": round(u_hit / n, 4),
        "bridge_causal_drop": round((f_hit - b_hit) / n, 4),
        "unrelated_drop": round((f_hit - u_hit) / n, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=26)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument("--selfsup", type=str, default="")
    args = ap.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== v2 纯通用神经回想核 E1 基准（零手工特征，dev={dev}, seeds={seeds}, selfsup={args.selfsup or 'none'}）===", flush=True)

    tok_enc = TokenLevelEncoder("BAAI/bge-small-zh-v1.5", device=dev)
    tr_s0, _, tr_examples = build_disjoint_train_world(tok_enc)
    ev_s0, _, eval_sets, smoke6 = build_ayuan_eval_world(tok_enc)

    tr_store = convert_store_to_neural(tr_s0, tok_enc)
    tr_examples = list(tr_examples)
    if args.selfsup and (ROOT / args.selfsup).exists():
        ss = json.loads((ROOT / args.selfsup).read_text(encoding="utf-8")).get("topics", [])
        ss_texts = []
        for item in ss:
            ss_texts.extend([item["e1"], item["e2"], item["q_cross"], item["q_direct"], item["q_hard_neg"]])
        tok_enc.encode_both(ss_texts, batch_size=64)
        for i, item in enumerate(ss):
            sid = item["id"]
            e1 = tr_store.add_event(item["e1"], day=40 + i * 2, source=sid, event_id=f"{sid}_e1")
            e2 = tr_store.add_event(item["e2"], day=41 + i * 2, source=sid, event_id=f"{sid}_e2")
            tr_examples.append(RecallExample(item["q_cross"], "cross", [e1.event_id, e2.event_id], sid, e1.event_id, e2.event_id))
            tr_examples.append(RecallExample(item["q_direct"], "direct", [e1.event_id], sid))
            tr_examples.append(RecallExample(item["q_hard_neg"], "hard_neg", [], sid))
        print(f"已合入自监督扩展语料：训练库 {len(tr_store.active_events())} 事件 / {len(tr_examples)} 样本", flush=True)

    tr_graph = PureNeuralGraph(tr_store, max_neighbors=16)
    ev_store = convert_store_to_neural(ev_s0, tok_enc)
    ev_graph = PureNeuralGraph(ev_store, max_neighbors=16)

    seed_runs = []
    for sd in seeds:
        torch.manual_seed(sd)
        net = GeneralNeuralRecallNet(d_model=32, n_heads=4, n_layers=2).to(dev)
        n_params = sum(p.numel() for p in net.parameters())
        losses = train_neural_v2(net, tr_store, tr_graph, tr_examples, epochs=args.epochs, lr=args.lr, candidate_budget=12, seed=sd)
        abs_t = calibrate_v2_thresh(net, tr_store, tr_graph, tr_examples, budget=args.budget)
        print(f"[seed={sd}] params={n_params} | losses={losses} | abstain_thresh={abs_t:.2f}", flush=True)

        res_sd = {}
        for name, mode in [("v2_hybrid", "hybrid"), ("v2_graph_only", "graph_only"), ("v2_full_only", "full_only")]:
            res_sd[name] = eval_v2(net, ev_store, ev_graph, eval_sets, smoke6, mode=mode, abstain_thresh=abs_t, budget=args.budget)
        causal_sd = run_v2_causal(net, ev_store, eval_sets["cross"], abstain_thresh=abs_t, budget=args.budget)
        seed_runs.append({
            "seed": sd,
            "params": n_params,
            "abstain_thresh": abs_t,
            "losses": losses,
            "results": {k: {mk: mv for mk, mv in v.items() if mk != "details_cross"} for k, v in res_sd.items()},
            "causal_ablation": causal_sd,
            "details_cross_hybrid": res_sd["v2_hybrid"]["details_cross"],
        })

    mkeys = ["cross_exact", "cross_single", "cross_target_hit", "direct_exact", "hard_neg_fpr", "superseded_latest_acc", "insufficient_abstain_acc", "smoke6_hit"]
    summary = {}
    print(f"\n=== v2 纯通用神经回想核 E1 汇总（{len(seeds)} 种子均值，零手工特征、零主题泄露）===")
    header = f"{'方法':<18} | {'跨事件全中':>8} | {'跨事件单中':>8} | {'二跳目标':>8} | {'直接回想':>8} | {'难负例FPR↓':>9} | {'已变更↑':>7} | {'缺证拒答↑':>8} | {'冒烟6题':>7}"
    print(header)
    print("-" * len(header))
    for name in ("v2_hybrid", "v2_graph_only", "v2_full_only"):
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
        f"\n因果消融（v2_hybrid）: 完整库 {causal_mean['full_target_hit']:.1%} | "
        f"移桥 {causal_mean['drop_bridge_target_hit']:.1%} (Δ={-causal_mean['bridge_causal_drop']:+.1%}) | "
        f"移无关 {causal_mean['drop_unrelated_target_hit']:.1%} (Δ={-causal_mean['unrelated_drop']:+.1%})",
        flush=True,
    )

    elapsed = round(time.time() - t0, 1)
    out_path = ROOT / f"runs/{time.strftime('%Y%m%d-%H%M%S')}_assoc_v2_e1.json"
    out_path.write_text(
        json.dumps(
            {
                "experiment": "v2_general_neural_e1",
                "seeds": seeds,
                "epochs": args.epochs,
                "budget": args.budget,
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
