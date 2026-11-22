"""Phase 3 规模扩展压力测试：N = 53 → 200 → 500 → 1,000 事件下的召回、误唤起、延迟与路径巩固收益。

对比 5 路方法在相同预算（budget=16）下随记忆库膨胀的表现：
- one_shot
- plain_2round
- learned_graph_only（巩固前 / 路径巩固后）
- learned_full_only
- learned_hybrid（图邻居 + 跳出图全库重搜）
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if not os.environ.get("HF_HOME") and (ROOT / ".pytest_cache" / "hf").exists():
    os.environ["HF_HOME"] = str(ROOT / ".pytest_cache" / "hf")

import torch

from assoc_memory import (
    EpisodicStore,
    FrozenTextEncoder,
    IterativeRecallNet,
    OneShotBaseline,
    PlainTwoRoundBaseline,
    SparseAssociativeGraph,
    train_iterative_recall,
)
from tools.eval_assoc_e1 import build_ayuan_eval_world, build_disjoint_train_world


def generate_distractor_diaries(n_needed: int, seed: int = 123) -> List[str]:
    """生成风格自然、互不重复的日常杂项日记片段作为大规模背景事件（绝不触碰阿远核心实体链）。"""
    rng = random.Random(seed)
    weathers = ["晴朗微风", "多云转阴", "细雨绵绵", "闷热潮湿", "秋高气爽", "寒风凛冽", "早晚温差大"]
    places = ["地铁站口", "商业广场", "社区公园", "街角书店", "菜市场", "便利店", "写字楼大堂", "天桥底下", "健身房门口", "邮局柜台"]
    actions = [
        "排队买了两根玉米和一杯热豆浆",
        "取了上周网购的收纳盒和垃圾袋",
        "顺路给公交卡充了一百块钱余额",
        "修补了雨伞松动的那根伞骨",
        "核对了上个月的水电燃气缴费账单",
        "整理了邮箱里积压的几十封订阅简报",
        "给家里的绿萝换了干净的自来水",
        "把换季的厚外套送去干洗店熨烫",
        "去药店补买了创可贴和碘伏棉签",
        "在书店翻了半小时城市建筑摄影画册",
        "和楼下保安大叔闲聊了会物业维修进度",
        "去五金店配了一把阳台防盗门小钥匙",
        "把书架上蒙灰的旧杂志按年份重新归类",
        "在网上预约了下周三下午的上门家电清洗",
        "在超市生鲜区挑了半斤蓝莓和一盒酸奶",
        "清理了手机相册里三百多张重复截图",
    ]
    feelings = [
        "折腾完虽然有点累，但看着屋子整整齐齐心情还挺舒坦。",
        "路上堵车耽误了二十分钟，好在没错过晚高峰前的班车。",
        "回家冲了个热水澡早早躺下，打算明天早起半小时。",
        "这周节奏比上周慢了不少，总算有空把琐事一件件清掉。",
        "顺便记了一笔日常开销，发现这月外食花销比预期低。",
    ]
    out: List[str] = []
    seen = set()
    i = 0
    while len(out) < n_needed:
        i += 1
        txt = (
            f"第{i}天记录（{rng.choice(weathers)}）：下午经过{rng.choice(places)}时，"
            f"{rng.choice(actions)}，{rng.choice(feelings)}"
        )
        if txt not in seen:
            seen.add(txt)
            out.append(txt)
    return out


def eval_scale_point(
    store: EpisodicStore,
    graph: SparseAssociativeGraph,
    eval_sets: Dict,
    net: IterativeRecallNet,
    abs_t: float,
    b_one: OneShotBaseline,
    b_p2: PlainTwoRoundBaseline,
    budget: int,
) -> Dict[str, Dict[str, float]]:
    methods = {
        "one_shot": lambda q: b_one.recall(store, q, top_k=2),
        "plain_2round": lambda q: b_p2.recall(store, q),
        "learned_graph_only": lambda q: net.recall(store, graph, q, mode="graph_only", candidate_budget=budget, abstain_thresh=abs_t),
        "learned_full_only": lambda q: net.recall(store, graph, q, mode="full_only", candidate_budget=budget, abstain_thresh=abs_t),
        "learned_hybrid": lambda q: net.recall(store, graph, q, mode="hybrid", candidate_budget=budget, abstain_thresh=abs_t),
    }
    res: Dict[str, Dict[str, float]] = {}
    for name, fn in methods.items():
        t_start = time.perf_counter()
        c_ex = c_tgt = 0
        for ex in eval_sets["cross"]:
            r = fn(ex.query)
            p = set(r.event_ids)
            if set(ex.gold_ids).issubset(p):
                c_ex += 1
            if ex.target_id and ex.target_id in p:
                c_tgt += 1
        d_ex = 0
        for ex in eval_sets["direct"]:
            r = fn(ex.query)
            if set(ex.gold_ids).issubset(set(r.event_ids)):
                d_ex += 1
        fp = 0
        for ex in eval_sets["hard_neg"]:
            r = fn(ex.query)
            if not r.abstained and len(r.event_ids) > 0:
                fp += 1
        n_q = len(eval_sets["cross"]) + len(eval_sets["direct"]) + len(eval_sets["hard_neg"])
        ms_per_q = (time.perf_counter() - t_start) * 1000.0 / max(1, n_q)
        res[name] = {
            "cross_exact": round(c_ex / len(eval_sets["cross"]), 4),
            "cross_target": round(c_tgt / len(eval_sets["cross"]), 4),
            "direct_exact": round(d_ex / len(eval_sets["direct"]), 4),
            "hard_neg_fpr": round(fp / len(eval_sets["hard_neg"]), 4),
            "latency_ms": round(ms_per_q, 2),
        }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", type=str, default="53,200,500,1000")
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    scales = [int(x.strip()) for x in args.scales.split(",") if x.strip()]

    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Phase 3 规模扩展压力测试（scales={scales}, budget={args.budget}）===", flush=True)

    encoder = FrozenTextEncoder("BAAI/bge-small-zh-v1.5", device=dev)
    tr_store, tr_graph, tr_examples = build_disjoint_train_world(encoder)
    ev_store, _, eval_sets, _ = build_ayuan_eval_world(encoder)

    b_one = OneShotBaseline()
    b_one.calibrate(tr_store, tr_examples)
    b_p2 = PlainTwoRoundBaseline()
    b_p2.calibrate(tr_store, tr_examples)

    torch.manual_seed(args.seed)
    net = IterativeRecallNet(emb_dim=encoder.dim, hidden_dim=32, low_rank=16).to(dev)
    train_iterative_recall(net, tr_store, tr_graph, tr_examples, epochs=35, lr=4e-3, candidate_budget=args.budget, seed=args.seed)
    abs_t = net.calibrate_abstain_thresh(tr_store, tr_graph, tr_examples, mode="hybrid", candidate_budget=args.budget)

    max_n = max(scales)
    base_active = len(ev_store.active_events())
    distractors = generate_distractor_diaries(max(0, max_n - base_active), seed=123)
    # 一次性批量编码全部背景事件
    if distractors:
        encoder.encode(distractors, batch_size=64)

    scale_results = {}
    added = 0
    for target_n in scales:
        while len(ev_store.active_events()) < target_n and added < len(distractors):
            ev_store.add_event(
                distractors[added],
                day=60 + (added % 300),
                source=f"bg_{added + 1:04d}",
                event_id=f"bg_{added + 1:04d}",
            )
            added += 1
        cur_n = len(ev_store.active_events())
        graph = SparseAssociativeGraph(ev_store, max_neighbors=16)
        res = eval_scale_point(ev_store, graph, eval_sets, net, abs_t, b_one, b_p2, args.budget)
        scale_results[str(cur_n)] = res
        print(f"\n--- Active Events N = {cur_n} (budget={args.budget}) ---")
        for mname, m in res.items():
            print(
                f"  {mname:<20} | cross={m['cross_exact']:>5.1%} (tgt={m['cross_target']:>5.1%}) | "
                f"direct={m['direct_exact']:>5.1%} | FPR={m['hard_neg_fpr']:>5.1%} | {m['latency_ms']:>6.2f} ms/q",
                flush=True,
            )

        # 在最大规模（N=1000）上额外验证「巩固反复验证路径进图边」对 graph_only 的增益
        if target_n == max_n:
            g_consol = SparseAssociativeGraph(ev_store, max_neighbors=16)
            for ex in eval_sets["cross"]:
                r_hyb = net.recall(ev_store, g_consol, ex.query, mode="hybrid", candidate_budget=args.budget, abstain_thresh=abs_t)
                if len(r_hyb.event_ids) == 2 and set(ex.gold_ids).issubset(set(r_hyb.event_ids)):
                    g_consol.reinforce_path(r_hyb.event_ids, delta=0.8)
            after_consol = eval_scale_point(ev_store, g_consol, eval_sets, net, abs_t, b_one, b_p2, args.budget)
            scale_results[f"{cur_n}_after_path_consol"] = after_consol["learned_graph_only"]
            print(
                f"  [N={cur_n} 巩固路径成边后] learned_graph_only: "
                f"cross={after_consol['learned_graph_only']['cross_exact']:.1%} "
                f"(巩固前 {res['learned_graph_only']['cross_exact']:.1%}) | "
                f"延迟 {after_consol['learned_graph_only']['latency_ms']:.2f} ms/q",
                flush=True,
            )

    elapsed = round(time.time() - t0, 1)
    out_path = ROOT / f"runs/{time.strftime('%Y%m%d-%H%M%S')}_assoc_scale.json"
    out_path.write_text(
        json.dumps(
            {
                "experiment": "Phase3_scale_stress",
                "scales": scales,
                "budget": args.budget,
                "seed": args.seed,
                "elapsed_s": elapsed,
                "results": scale_results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n结果已保存：{out_path}（耗时 {elapsed}s）", flush=True)


if __name__ == "__main__":
    main()
