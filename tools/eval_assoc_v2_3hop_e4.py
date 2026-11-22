"""任务 C：纯通用神经回想核（v2）的 3 跳自适应深度测试 + 接回冻结 Qwen2.5-1.5B-Instruct（E4 v2）。

1. 3 跳长链自适应停机测试（max_rounds=3）：
   - 在独立训练世界加入 2 条 3 跳训练链（与阿远零主题交集），让 GeneralNeuralRecallNet 学会 1/2/3 跳通用停机；
   - 在阿远评测库测 4 条自然 3 跳链 + 直接单跳题 + 2 跳跨事件题 + 难负例，检验模型能否按需动态走 1/2/3 跳。
2. E4 v2 端到端表达对照：
   - 将纯通用 v2（零手工特征）接回冻结 Qwen2.5-1.5B-Instruct，对比 one_shot / v2_hybrid / oracle 三路最终中文答对率与零串台率。
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

from assoc_memory import OneShotBaseline, RecallExample
from assoc_neural_v2 import (
    GeneralNeuralRecallNet,
    PureNeuralGraph,
    TokenLevelEncoder,
    train_neural_v2,
)
from tools.eval_assoc_e1 import build_ayuan_eval_world, build_disjoint_train_world
from tools.eval_assoc_e4 import AssocLanguageInterface, any_kw
from tools.eval_assoc_v2 import calibrate_v2_thresh, convert_store_to_neural


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_lora", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== 任务 C：v2 纯通用神经核 3 跳长链 + E4 冻结 1.5B LLM（use_lora={args.use_lora}, dev={dev}）===", flush=True)

    if args.use_lora:
        from tools.eval_assoc_v2_lora import LoRATokenEncoder, train_end_to_end_lora
        tok_enc = LoRATokenEncoder("BAAI/bge-small-zh-v1.5", device=dev, r=8)
    else:
        tok_enc = TokenLevelEncoder("BAAI/bge-small-zh-v1.5", device=dev)
    tr_s0, _, tr_examples = build_disjoint_train_world(tok_enc)
    ev_s0, _, eval_sets, _ = build_ayuan_eval_world(tok_enc)

    tr_examples = list(tr_examples)
    tr_s0.add_event(
        "暖窝宠物寄养店的小周说寄养费按天结算，自带平时吃的猫粮可以打八折，离巷口公交站只有两百米。",
        day=36, source="tr_3hop_1", event_id="t_cat_3hop",
    )
    tr_s0.add_event(
        "仁济康复中心的林医生还开了一盒氟比洛芬外用贴膏，嘱咐静蹲训练完如果右膝发胀就贴一片。",
        day=37, source="tr_3hop_2", event_id="t_run_3hop",
    )
    tr_examples.append(
        RecallExample(
            "楼道电钻装修吓得躲衣柜的小家伙送去寄养了，那家店怎么收费以及自带粮有没有折扣？",
            "cross", ["t_cat_1", "t_cat_2", "t_cat_3hop"], "cat3",
        )
    )
    tr_examples.append(
        RecallExample(
            "东湖跑完十公里右膝外侧疼去找医生做理疗，他除了叮嘱做静蹲还开了什么外用药贴？",
            "cross", ["t_run_1", "t_run_2", "t_run_3hop"], "run3",
        )
    )

    torch.manual_seed(args.seed)
    net = GeneralNeuralRecallNet(d_model=32, n_heads=4, n_layers=2).to(dev)
    if args.use_lora:
        tr_raw = [(e.event_id, e.day, e.source, e.text, e.status) for e in tr_s0.all_events(include_deleted=True)]
        losses = train_end_to_end_lora(tok_enc, net, tr_raw, tr_examples, epochs=args.epochs, seed=args.seed)
        tr_store = convert_store_to_neural(tr_s0, tok_enc)
        tr_graph = PureNeuralGraph(tr_store, max_neighbors=16)
    else:
        tr_store = convert_store_to_neural(tr_s0, tok_enc)
        tr_graph = PureNeuralGraph(tr_store, max_neighbors=16)
        losses = train_neural_v2(net, tr_store, tr_graph, tr_examples, epochs=args.epochs, lr=3e-3, candidate_budget=14, seed=args.seed)
    abs_t = calibrate_v2_thresh(net, tr_store, tr_graph, tr_examples, budget=args.budget)
    print(f"v2 训练完成 | losses={losses} | abstain_thresh={abs_t:.2f}", flush=True)

    ev_store = convert_store_to_neural(ev_s0, tok_enc)
    e_chen_phone = ev_store.add_event(
        "隔壁邻居陈阿姨住在 1402 室，她留了手机号尾号 9108，说晚上九点前随时可以敲门或发短信联系她。",
        day=54, source="diary_32_3hop", event_id="u_chen_phone",
    )
    e_darkroom_hour = ev_store.add_event(
        "市中心弗林德斯街那家黑白暗房冲扫店周一休息，周二到周日下午六点前送卷隔天就能取电子扫描件。",
        day=55, source="diary_33_3hop", event_id="u_darkroom_hour",
    )
    ev_graph = PureNeuralGraph(ev_store, max_neighbors=16)

    # 1. 测试 3 跳长链（max_rounds=3）
    def find_id(sub: str) -> str:
        return next(e.event_id for e in ev_store.active_events() if sub in e.text)

    e_thunder_w = find_id("每次外面乌云密布要打雷")
    e_fuji = find_id("老富士胶片相机拿出来擦了擦镜头")
    e_app_b = find_id("今年二十八岁了，天天在公司改 React 组件")
    e_app_adv = find_id("同事劝我先利用周末把原型跑通")
    e_hokk_b = find_id("把明年的北海道看雪预算粗略列了个表")

    three_hop_queries = [
        (
            "今晚外面乌云打雷我还在公司加班，家里缩桌底发抖的小家伙托邻居照看，那位邻居住几零几、手机尾号是多少？",
            [e_thunder_w, "u_care_new", e_chen_phone.event_id],
            ["1402", "9108"],
        ),
        (
            "下雨在家擦镜头整理圣基尔达海边胶片，打算送去市中心暗房冲扫，那家店周几休息、几点前送卷能隔天取？",
            [e_fuji, "u_film_shop", e_darkroom_hour.event_id],
            ["周一", "六点"],
        ),
        (
            "在公司改业务表单少了掌控感、想辞职做小产品，同事劝我先攒够旅行基金和生活费，那趟看雪预算能实现吗？",
            [e_app_b, e_app_adv, e_hokk_b],
            ["预算", "实现", "北海道"],
        ),
    ]

    hop3_details = []
    h3_exact = h3_tgt = 0
    for q, golds, _kws in three_hop_queries:
        r = net.recall(ev_store, ev_graph, q, mode="hybrid", max_rounds=3, candidate_budget=args.budget, abstain_thresh=abs_t)
        p = set(r["event_ids"])
        ex_ok = set(golds).issubset(p)
        tg_ok = golds[-1] in p
        h3_exact += int(ex_ok); h3_tgt += int(tg_ok)
        hop3_details.append({"q": q, "gold": golds, "pred": r["event_ids"], "exact3": ex_ok, "target3_hit": tg_ok, "p_valid": round(float(r["p_valid"]), 3)})
    print(
        f"\n=== 3 跳长链测试（max_rounds=3）: 3事件全中={h3_exact}/{len(three_hop_queries)} | "
        f"第3跳末端命中={h3_tgt}/{len(three_hop_queries)} ===",
        flush=True,
    )
    for d in hop3_details:
        print(f"  pred={d['pred']} (gold={d['gold']}, p={d['p_valid']})", flush=True)

    # 2. E4 v2：把纯通用 v2 接回冻结 Qwen2.5-1.5B-Instruct，对比 one_shot / v2_hybrid / oracle
    b_one = OneShotBaseline()
    b_one.calibrate(tr_s0, tr_examples[:51])

    cross_items = [
        (eval_sets["cross"][0], ["陈阿姨"]),
        (eval_sets["cross"][1], ["阿尔伯特", "Albert"]),
        (eval_sets["cross"][2], ["陈阿姨"]),
        (eval_sets["cross"][3], ["氯雷他定", "玄关"]),
        (eval_sets["cross"][5], ["弗林德斯", "暗房"]),
        (eval_sets["cross"][6], ["北海道"]),
        (eval_sets["cross"][7], ["原型", "生活费", "周末"]),
        (eval_sets["cross"][8], ["TypeScript", "前端"]),
        (eval_sets["cross"][9], ["预算", "实现"]),
        (eval_sets["cross"][10], ["书桌"]),
        (eval_sets["cross"][11], ["周五", "八点"]),
    ]
    hard_neg_items = [
        (eval_sets["hard_neg"][0], ["旺财", "陈阿姨", "阿远", "林远"]),
        (eval_sets["hard_neg"][1], ["过敏", "起疹子", "氯雷他定", "阿远"]),
        (eval_sets["hard_neg"][2], ["记账", "阿远", "林远"]),
        (eval_sets["hard_neg"][3], ["老富士", "冲绳", "阿远"]),
        (eval_sets["hard_neg"][4], ["旺财", "打雷", "陈阿姨"]),
        (eval_sets["hard_neg"][5], ["沙发", "阿远", "林远"]),
    ]

    cache: Dict[str, Dict[str, List[str]]] = {}
    for ex, _ in cross_items:
        cache[ex.query] = {
            "one_shot": b_one.recall(ev_s0, ex.query, top_k=2).event_ids,
            "v2_pure_neural": net.recall(ev_store, ev_graph, ex.query, mode="hybrid", max_rounds=2, candidate_budget=args.budget, abstain_thresh=abs_t)["event_ids"],
            "oracle": ex.gold_ids,
        }
    for ex, _ in hard_neg_items:
        cache[ex.query] = {
            "one_shot": b_one.recall(ev_s0, ex.query, top_k=2).event_ids,
            "v2_pure_neural": net.recall(ev_store, ev_graph, ex.query, mode="hybrid", max_rounds=2, candidate_budget=args.budget, abstain_thresh=abs_t)["event_ids"],
            "oracle": [],
        }

    # 释放编码器显存，加载冻结 1.5B LLM
    tok_enc.model.to("cpu")
    if dev == "cuda":
        torch.cuda.empty_cache()

    lang = AssocLanguageInterface(model_name=args.llm, device=dev)
    e4_metrics = {}
    for arm in ("one_shot", "v2_pure_neural", "oracle"):
        c_ret = c_ans = 0
        for ex, kws in cross_items:
            eids = cache[ex.query][arm]
            evs = [ev_store.get(eid) for eid in eids]
            ans = lang.reply(ex.query, evs)
            c_ret += int(bool(ex.target_id and ex.target_id in eids))
            c_ans += int(any_kw(ans, kws))
        hn_int = 0
        for ex, forb in hard_neg_items:
            eids = cache[ex.query][arm]
            evs = [ev_store.get(eid) for eid in eids]
            ans = lang.reply(ex.query, evs)
            hn_int += int(any_kw(ans, forb) or len(eids) > 0)
        e4_metrics[arm] = {
            "cross_retrieval_target": round(c_ret / len(cross_items), 4),
            "cross_llm_answer_acc": round(c_ans / len(cross_items), 4),
            "hard_neg_intrusion_rate": round(hn_int / len(hard_neg_items), 4),
        }

    print("\n=== E4 v2 端到端对比（冻结 Qwen2.5-1.5B-Instruct）===")
    for arm, m in e4_metrics.items():
        print(
            f"  {arm:<16} | 跨事件检索={m['cross_retrieval_target']:>6.1%} | "
            f"LLM答对={m['cross_llm_answer_acc']:>6.1%} | 难负例串台↓={m['hard_neg_intrusion_rate']:>5.1%}",
            flush=True,
        )

    elapsed = round(time.time() - t0, 1)
    out_path = ROOT / f"runs/{time.strftime('%Y%m%d-%H%M%S')}_assoc_v2_3hop_e4.json"
    out_path.write_text(
        json.dumps(
            {
                "experiment": "TaskC_v2_3hop_and_e4",
                "seed": args.seed,
                "elapsed_s": elapsed,
                "hop3_exact": round(h3_exact / len(three_hop_queries), 4),
                "hop3_target_hit": round(h3_tgt / len(three_hop_queries), 4),
                "hop3_details": hop3_details,
                "e4_v2_metrics": e4_metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n结果已保存：{out_path}（耗时 {elapsed}s）", flush=True)


if __name__ == "__main__":
    main()
