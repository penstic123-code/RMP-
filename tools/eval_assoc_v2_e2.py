"""E2 v2.4 验证：最强纯通用架构（BGE 顶 2 层 LoRA + GeneralNeuralRecallNet）的时间分批留出实测。

严格同 E2 协议（3 时间批次 × 3 时刻 × 3 种子，含二次改口 陈阿姨→小李，零主题泄露预训练）：
- ① before_write（写入前）
- ② after_write_before_consol（写入后巩固前：仅即时加节点与重建纯向量/MaxSim 稀疏图，零权重更新）
- ③ after_consol（巩固后：将验证路径硬化成稀疏图边）
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

from assoc_neural_v2 import GeneralNeuralRecallNet, NeuralEpisodicStore, PureNeuralGraph
from tools.eval_assoc_e1 import build_disjoint_train_world
from tools.eval_assoc_e2 import build_e2_timeline
from tools.eval_assoc_v2 import calibrate_v2_thresh, convert_store_to_neural
from tools.eval_assoc_v2_lora import LoRATokenEncoder, train_end_to_end_lora


def eval_v2_stage(
    net: GeneralNeuralRecallNet,
    store: NeuralEpisodicStore,
    graph: PureNeuralGraph,
    new_assoc: Sequence,
    old_queries: Sequence,
    latest_queries: Sequence,
    hard_negs: Sequence,
    abstain_thresh: float,
    budget: int = 16,
) -> Dict[str, float]:
    def run_q(q: str):
        return net.recall(store, graph, q, mode="hybrid", max_rounds=2, candidate_budget=budget, abstain_thresh=abstain_thresh)

    na_ex = na_tg = 0
    for ex in new_assoc:
        r = run_q(ex.query)
        p = set(r["event_ids"])
        if set(ex.gold_ids).issubset(p):
            na_ex += 1
        if ex.target_id and ex.target_id in p:
            na_tg += 1

    old_hit = sum(1 for ex in old_queries if set(ex.gold_ids).issubset(set(run_q(ex.query)["event_ids"])))
    lat_hit = lat_leak = 0
    for ex in latest_queries:
        r = run_q(ex.query)
        p = set(r["event_ids"])
        if set(ex.gold_ids).issubset(p):
            lat_hit += 1
        if any(eid in store and store.get(eid).status == "superseded" for eid in p):
            lat_leak += 1

    fp = sum(1 for ex in hard_negs if not run_q(ex.query)["abstained"] and len(run_q(ex.query)["event_ids"]) > 0)
    return {
        "new_assoc_exact": round(na_ex / max(1, len(new_assoc)), 4),
        "new_assoc_target": round(na_tg / max(1, len(new_assoc)), 4),
        "old_retention": round(old_hit / max(1, len(old_queries)), 4),
        "latest_fact_acc": round(lat_hit / max(1, len(latest_queries)), 4),
        "old_value_leak": round(lat_leak / max(1, len(latest_queries)), 4),
        "hard_neg_fpr": round(fp / max(1, len(hard_negs)), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--seeds", type=str, default="42,43,44")
    args = ap.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== E2 v2.4（BGE-LoRA + 纯通用神经核）时间分批留出实测（seeds={seeds}）===", flush=True)

    seed_records = []
    for sd in seeds:
        torch.manual_seed(sd)
        enc = LoRATokenEncoder("BAAI/bge-small-zh-v1.5", device=dev, r=8)
        tr_s0, _, tr_examples = build_disjoint_train_world(enc)
        tr_raw = [(e.event_id, e.day, e.source, e.text, e.status) for e in tr_s0.all_events(include_deleted=True)]
        net = GeneralNeuralRecallNet(d_model=32, n_heads=4, n_layers=2).to(dev)
        train_end_to_end_lora(enc, net, tr_raw, tr_examples, epochs=args.epochs, seed=sd)

        tr_store = convert_store_to_neural(tr_s0, enc)
        tr_graph = PureNeuralGraph(tr_store, max_neighbors=16)
        abs_t = calibrate_v2_thresh(net, tr_store, tr_graph, tr_examples, budget=args.budget)
        print(f"[seed={sd}] pretrained v2.4 ready | abstain_thresh={abs_t:.2f}", flush=True)

        # 构建 E2 时间线（用训好的 LoRA 编码器）
        s0, _, fixed_old, hard_negs, batches = build_e2_timeline(enc)
        store = convert_store_to_neural(s0, enc)
        graph = PureNeuralGraph(store, max_neighbors=16)

        b_logs = []
        for b in batches:
            m_before = eval_v2_stage(net, store, graph, b["unseen_new_assoc"], fixed_old, b["latest_fact_queries"], hard_negs, abs_t, args.budget)
            for op, eid, day, src, txt, old_id in b["writes"]:
                if op == "supersede" and old_id:
                    store.supersede_event(old_id, txt, day=day, source=src, new_id=eid)
                else:
                    store.add_event(txt, day=day, source=src, event_id=eid)
            graph.rebuild()
            m_after_w = eval_v2_stage(net, store, graph, b["unseen_new_assoc"], fixed_old, b["latest_fact_queries"], hard_negs, abs_t, args.budget)

            # 路径巩固（无权重漂移，零遗忘风险）
            for sig in b["consol_signals"]:
                if len(sig.gold_ids) >= 2:
                    graph.reinforce_path(sig.gold_ids, delta=0.8)
            m_after_c = eval_v2_stage(net, store, graph, b["unseen_new_assoc"], fixed_old, b["latest_fact_queries"], hard_negs, abs_t, args.budget)
            b_logs.append({"batch_id": b["batch_id"], "label": b["label"], "before_write": m_before, "after_write_before_consol": m_after_w, "after_consol": m_after_c})
        seed_records.append({"seed": sd, "abstain_thresh": abs_t, "batches": b_logs})
        del enc, net
        if dev == "cuda":
            torch.cuda.empty_cache()

    stages = ("before_write", "after_write_before_consol", "after_consol")
    mkeys = ("new_assoc_exact", "new_assoc_target", "old_retention", "latest_fact_acc", "old_value_leak", "hard_neg_fpr")
    agg = []
    print(f"\n=== E2 v2.4 时间分批留出汇总（{len(seeds)} 种子均值，未见过的新问法）===")
    for bi in range(3):
        lbl = seed_records[0]["batches"][bi]["label"]
        print(f"\n[{lbl}]")
        print(f"{'时刻':<26} | {'新关联全中':>8} | {'新关联目标':>8} | {'旧题留存':>7} | {'最新事实':>7} | {'旧值泄露↓':>8} | {'难负例FPR↓':>9}")
        print("-" * 92)
        b_agg = {"batch_id": bi + 1, "label": lbl}
        for st in stages:
            vals = {mk: round(sum(sr["batches"][bi][st][mk] for sr in seed_records) / len(seed_records), 4) for mk in mkeys}
            b_agg[st] = vals
            print(
                f"{st:<26} | {vals['new_assoc_exact']:>8.1%} | {vals['new_assoc_target']:>8.1%} | "
                f"{vals['old_retention']:>7.1%} | {vals['latest_fact_acc']:>7.1%} | "
                f"{vals['old_value_leak']:>8.1%} | {vals['hard_neg_fpr']:>9.1%}",
                flush=True,
            )
        agg.append(b_agg)

    elapsed = round(time.time() - t0, 1)
    out_path = ROOT / f"runs/{time.strftime('%Y%m%d-%H%M%S')}_assoc_v2_e2.json"
    out_path.write_text(
        json.dumps({"experiment": "E2_v2.4_bge_lora", "seeds": seeds, "elapsed_s": elapsed, "summary_mean": agg, "seed_records": seed_records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n结果已保存：{out_path}（耗时 {elapsed}s）", flush=True)


if __name__ == "__main__":
    main()
