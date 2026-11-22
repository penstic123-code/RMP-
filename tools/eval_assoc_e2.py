"""E2 实验：验证新经历能否接通旧经历（时间分批留出 + 3 时刻对比 + 验收门控巩固）。

严格遵循 docs/design/associative-memory.md：
- 按日期分 3 批写入新经历（含跨事件新关联 + 改口覆盖 + 二次改口覆盖）。
- 每批测 3 个时刻：
  ① before_write（写入前）
  ② after_write_before_consol（写入后、巩固前：仅加节点与动态候选边，零权重更新）
  ③ after_consol（巩固后：路径强化成边 + 带回放与验收门控的小步巩固；用未见过的新问法测新关联 + 重跑固定旧题）
- 指标：新增关联命中率（new_assoc_exact / new_assoc_target）、旧题留存（old_retention）、
  最新事实命中率（latest_fact_acc & old_value_leak）、无关题误唤起率（hard_neg_fpr）。
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if not os.environ.get("HF_HOME") and (ROOT / ".pytest_cache" / "hf").exists():
    os.environ["HF_HOME"] = str(ROOT / ".pytest_cache" / "hf")

import torch

from assoc_memory import (
    EpisodicStore,
    FrozenTextEncoder,
    IterativeRecallNet,
    RecallExample,
    SparseAssociativeGraph,
    train_iterative_recall,
)
from tools.eval_assoc_e1 import build_disjoint_train_world


def build_e2_timeline(encoder: FrozenTextEncoder):
    """构造阿远的时间分批事件库：Batch 0（早期基础经历）+ Batch 1/2/3（按日期新增经历与改口覆盖）。"""
    store = EpisodicStore(encoder)

    # Batch 0：早期基础经历（Day 1–19）
    base_events = [
        ("e_self", 1, "b0", "今天难得放假，我是林远，朋友们平时都叫我阿远，住在墨尔本南岸走几步就能看到雅拉河。"),
        ("e_park", 1, "b0", "下午牵着我那只三岁的金毛犬旺财去阿尔伯特公园撒欢，它一看见草地就疯了一样追网球。"),
        ("e_coke", 1, "b0", "回家顺路在便利店拎了两罐冰镇无糖可乐，冰箱里要是没囤够无糖可乐我心里就不踏实。"),
        ("e_dinner", 1, "b0", "晚饭懒得折腾，自己煎了两块鸡胸肉配水煮西兰花，撒点现磨黑胡椒简单又健康。"),
        ("e_fe_dev", 3, "b0", "作为一名天天跟 TypeScript 和 React 打交道的前端开发，平时坐办公室腰酸背痛。"),
        ("e_badm_old", 3, "b0", "十月初和前端组同事约好固定每周三晚上七点去体育馆打羽毛球挥拍流汗。"),
        ("e_allergy", 3, "b0", "打完球同事约着去吃海鲜大排档我赶紧摆手拒绝了，我对海鲜特别是虾和螃蟹严重过敏一沾就起疹子。"),
        ("e_thunder", 5, "b0", "下午乌云密布隐隐打了几声闷雷，旺财胆子小得要命，一听见打雷就缩到我书桌底下发抖。"),
        ("e_fuji", 5, "b0", "趁着外面下雨没法出门，我把那台老富士胶片相机拿出来擦了擦镜头，整理了在圣基尔达海边拍的一卷胶片。"),
        ("e_santi", 5, "b0", "晚上洗完澡窝在沙发里重读《三体》，每次读到黑暗森林那一段还是觉得头皮发麻。"),
        ("e_app_proto", 7, "b0", "下午在笔记本上画了画我一直想做的那个独立记账 App 的极简录入交互原型图。"),
        ("e_trip_old", 8, "b0", "年初原本计划明年休年假去冲绳潜水看海，还收藏了几家海边民宿。"),
        ("e_cof_old", 9, "b0", "刚搬来南岸时早上习惯在楼下咖啡馆点全脂牛奶热拿铁提神。"),
        ("e_galaxy", 11, "b0", "晚上睡前翻了五十页阿西莫夫的《银河帝国》，科幻小说能让我忘掉白天写代码的烦心事。"),
        ("e_thunder_work", 13, "b0", "每次外面乌云密布要打雷，我就算在公司加班也会下意识看看窗外，惦记着家里的小金毛有没有钻书桌底发抖。"),
        ("e_app_code", 17, "b0", "晚上吃完晚饭没刷短视频，打开电脑用 TypeScript 写了两个小时独立记账 App 的前端框架。"),
    ]
    for eid, day, src, txt in base_events:
        store.add_event(txt, day=day, source=src, event_id=eid)

    graph = SparseAssociativeGraph(store, max_neighbors=16)

    # 固定旧题（Batch 0 起每批必重跑，检验旧题留存）
    fixed_old_queries = [
        RecallExample("冰箱里不囤够什么饮料我心里就不踏实？", "direct", ["e_coke"], "old"),
        RecallExample("晚上窝在沙发里重读哪本科幻小说读到黑暗森林觉得头皮发麻？", "direct", ["e_santi"], "old"),
        RecallExample("平时懒得折腾晚饭自己一般做什么菜？", "direct", ["e_dinner"], "old"),
        RecallExample("我在公司日常开发主要用哪两个前端技术栈？", "direct", ["e_fe_dev"], "old"),
        RecallExample("睡前翻了五十页阿西莫夫的哪本小说？", "direct", ["e_galaxy"], "old"),
        RecallExample("周末在笔记本上画了极简原型图的那个小项目，最近下班后花两小时开始写什么了？", "cross", ["e_app_proto", "e_app_code"], "old", "e_app_proto", "e_app_code"),
    ]

    # 12 条相似但无关难负例（全程检验误唤起率）
    hard_neg_queries = [
        RecallExample("拍闪电和雷暴天气的相机快门速度和感光度参数怎么设置？", "hard_neg", [], "neg"),
        RecallExample("墨尔本市中心有哪些评分高的海鲜自助餐厅推荐？", "hard_neg", [], "neg"),
        RecallExample("React 19 新的 useActionState 钩子怎么配合表单使用？", "hard_neg", [], "neg"),
        RecallExample("北海道二世古滑雪场的早鸟缆车票怎么在网上预订？", "hard_neg", [], "neg"),
        RecallExample("金毛犬换毛季掉毛严重推荐什么牌子的针梳？", "hard_neg", [], "neg"),
        RecallExample("《三体》动画版和电视剧版在黑暗森林剧情上差别大吗？", "hard_neg", [], "neg"),
        RecallExample("羽毛球拍拉线磅数选 24 磅还是 26 磅更适合新手？", "hard_neg", [], "neg"),
        RecallExample("鸡胸肉怎么腌制才能煎出来鲜嫩多汁不发柴？", "hard_neg", [], "neg"),
        RecallExample("雅拉河游船晚宴的登船码头在哪个街口？", "hard_neg", [], "neg"),
        RecallExample("富士数码相机 X-T5 和 X100VI 拍人像有什么区别？", "hard_neg", [], "neg"),
        RecallExample("燕麦奶和全脂牛奶做拿铁拉花哪个更容易出奶泡？", "hard_neg", [], "neg"),
        RecallExample("阿西莫夫《银河帝国》系列正确的出版阅读顺序是什么？", "hard_neg", [], "neg"),
    ]

    # 3 个时间批次（每批包含：新增/覆盖事件、巩固训练信号[问法A]、严格未见评测题[问法B]）
    batches = [
        {
            "batch_id": 1,
            "label": "Batch 1 (Day 30-35): 照看安排 + 冲扫店 + 羽毛球改期",
            "writes": [
                ("add", "b1_care", 31, "b1", "周四把南岸公寓的备用钥匙交给了隔壁邻居陈阿姨，托她我加班晚归时帮忙开门照看旺财。", None),
                ("add", "b1_shop", 33, "b1", "市中心弗林德斯街那家黑白暗房冲扫店手艺很稳，老富士拍完的胶卷以后都送去那儿冲扫。", None),
                ("supersede", "b1_badm_new", 35, "b1", "体育馆周三场地被学校包场了，前端组羽毛球局从十一月起改到了每周五晚上八点。", "e_badm_old"),
            ],
            "consol_signals": [
                RecallExample("加班时如果外面打雷，谁拿着公寓钥匙能帮忙看狗？", "cross", ["e_thunder_work", "b1_care"], "b1", "e_thunder_work", "b1_care"),
                RecallExample("海边拍完的胶卷以后送去哪家冲扫店？", "cross", ["e_fuji", "b1_shop"], "b1", "e_fuji", "b1_shop"),
                RecallExample("前端组羽毛球局现在改到哪天晚上？", "superseded", ["b1_badm_new"], "b1"),
            ],
            "unseen_new_assoc": [
                RecallExample("今晚外面乌云密布要打雷我还困在公司加班，家里缩桌底发抖的小金毛能找谁开门照看？", "cross", ["e_thunder_work", "b1_care"], "b1", "e_thunder_work", "b1_care"),
                RecallExample("上回下雨在家擦镜头整理圣基尔达海边那卷胶片，以后拍完卷打算去市中心哪条街的暗房洗？", "cross", ["e_fuji", "b1_shop"], "b1", "e_fuji", "b1_shop"),
            ],
            "latest_fact_queries": [
                RecallExample("现在前端组打羽毛球改到每周几晚上几点了？", "superseded", ["b1_badm_new"], "b1"),
            ],
        },
        {
            "batch_id": 2,
            "label": "Batch 2 (Day 45-52): 抗过敏药箱 + 导师建议 + 年假改北海道",
            "writes": [
                ("add", "b2_med", 46, "b2", "家庭医生开的氯雷他定抗过敏药片一直放在玄关抽屉最上层，误食虾蟹起疹子时救急用。", None),
                ("add", "b2_mentor", 49, "b2", "周末在咖啡馆偶遇前同事老周，他看完我的独立记账 App 原型后建议先上线免费版积累第一批种子用户。", None),
                ("supersede", "b2_trip_new", 52, "b2", "改主意不去冲绳潜水了，决定明年冬天休年假去北海道看雪，顺便背着老富士相机多拍几卷雪景胶片。", "e_trip_old"),
            ],
            "consol_signals": [
                RecallExample("如果聚餐误食虾蟹起疹子，急救药放在家里哪儿？", "cross", ["e_allergy", "b2_med"], "b2", "e_allergy", "b2_med"),
                RecallExample("画了原型图的独立项目，前同事老周给了什么推广建议？", "cross", ["e_app_proto", "b2_mentor"], "b2", "e_app_proto", "b2_mentor"),
                RecallExample("明年休年假最终定去哪儿玩？", "superseded", ["b2_trip_new"], "b2"),
            ],
            "unseen_new_assoc": [
                RecallExample("打完球同事约吃大排档我赶紧摆手拒绝了，万一不小心误食起疹子，救急药片放在哪个抽屉？", "cross", ["e_allergy", "b2_med"], "b2", "e_allergy", "b2_med"),
                RecallExample("笔记本上画了极简录入原型图的小产品，周末偶遇的前同事建议怎么冷启动？", "cross", ["e_app_proto", "b2_mentor"], "b2", "e_app_proto", "b2_mentor"),
            ],
            "latest_fact_queries": [
                RecallExample("现在前端组打羽毛球改到每周几晚上几点了？", "superseded", ["b1_badm_new"], "b1"),
                RecallExample("明年年假最终决定去哪儿看雪，还打算去冲绳吗？", "superseded", ["b2_trip_new"], "b2"),
            ],
        },
        {
            "batch_id": 3,
            "label": "Batch 3 (Day 60-68): 钥匙二次改口(陈阿姨→小李) + 公园柯基 + 咖啡改燕麦拿铁",
            "writes": [
                ("supersede", "b3_care_new", 61, "b3", "邻居陈阿姨下周回悉尼帮女儿带娃了，我把南岸公寓备用钥匙收回转交给了同层楼的程序员小李，加班晚归改托他照看旺财。", "b1_care"),
                ("add", "b3_corgi", 64, "b3", "今天带旺财去阿尔伯特公园追网球，认识了一只叫豆包的柯基，它主人建了个南岸遛狗群约周末一起遛。", None),
                ("supersede", "b3_cof_new", 68, "b3", "最近乳糖不耐受改喝植物奶了，在南岸楼下咖啡馆固定点少冰燕麦拿铁，店员不用我开口都记得。", "e_cof_old"),
            ],
            "consol_signals": [
                RecallExample("加班遇到打雷，现在公寓备用钥匙转交给谁帮忙照看小金毛了？", "cross", ["e_thunder_work", "b3_care_new"], "b3", "e_thunder_work", "b3_care_new"),
                RecallExample("带旺财去草坪追网球新认识了哪只小狗还拉了群？", "cross", ["e_park", "b3_corgi"], "b3", "e_park", "b3_corgi"),
                RecallExample("南岸楼下咖啡馆早上现在固定点什么咖啡？", "superseded", ["b3_cof_new"], "b3"),
            ],
            "unseen_new_assoc": [
                RecallExample("今晚外面乌云密布打雷我还在公司加班，现在备用钥匙在谁手里能帮忙开门看看怕雷的小金毛？", "cross", ["e_thunder_work", "b3_care_new"], "b3", "e_thunder_work", "b3_care_new"),
                RecallExample("平时下午牵着三岁金毛撒欢追网球的那片公园草地，最近新结识了哪只柯基还建了遛狗群？", "cross", ["e_park", "b3_corgi"], "b3", "e_park", "b3_corgi"),
            ],
            "latest_fact_queries": [
                RecallExample("现在前端组打羽毛球改到每周几晚上几点了？", "superseded", ["b1_badm_new"], "b1"),
                RecallExample("明年年假最终决定去哪儿看雪，还打算去冲绳吗？", "superseded", ["b2_trip_new"], "b2"),
                RecallExample("在南岸楼下咖啡馆早上现在固定点什么口味的咖啡？", "superseded", ["b3_cof_new"], "b3"),
                RecallExample("南岸公寓备用钥匙现在在陈阿姨还是同层哪位邻居手里？", "superseded", ["b3_care_new"], "b3"),
            ],
        },
    ]
    return store, graph, fixed_old_queries, hard_neg_queries, batches


def eval_stage(
    net: IterativeRecallNet,
    store: EpisodicStore,
    graph: SparseAssociativeGraph,
    new_assoc: Sequence[RecallExample],
    old_queries: Sequence[RecallExample],
    latest_queries: Sequence[RecallExample],
    hard_negs: Sequence[RecallExample],
    abstain_thresh: float,
    budget: int = 16,
) -> Dict[str, float]:
    def run_q(q: str):
        return net.recall(store, graph, q, mode="hybrid", candidate_budget=budget, abstain_thresh=abstain_thresh)

    na_exact = na_target = 0
    for ex in new_assoc:
        r = run_q(ex.query)
        p = set(r.event_ids)
        if set(ex.gold_ids).issubset(p):
            na_exact += 1
        if ex.target_id and ex.target_id in p:
            na_target += 1

    old_hit = 0
    for ex in old_queries:
        r = run_q(ex.query)
        if set(ex.gold_ids).issubset(set(r.event_ids)):
            old_hit += 1

    lat_hit = lat_leak = 0
    for ex in latest_queries:
        r = run_q(ex.query)
        p = set(r.event_ids)
        if set(ex.gold_ids).issubset(p):
            lat_hit += 1
        if any(eid in store and store.get(eid).status == "superseded" for eid in p):
            lat_leak += 1

    fp = 0
    for ex in hard_negs:
        r = run_q(ex.query)
        if not r.abstained and len(r.event_ids) > 0:
            fp += 1

    return {
        "new_assoc_exact": round(na_exact / max(1, len(new_assoc)), 4),
        "new_assoc_target": round(na_target / max(1, len(new_assoc)), 4),
        "old_retention": round(old_hit / max(1, len(old_queries)), 4),
        "latest_fact_acc": round(lat_hit / max(1, len(latest_queries)), 4),
        "old_value_leak": round(lat_leak / max(1, len(latest_queries)), 4),
        "hard_neg_fpr": round(fp / max(1, len(hard_negs)), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrain_epochs", type=int, default=35)
    ap.add_argument("--consol_epochs", type=int, default=6)
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--seeds", type=str, default="42,43,44")
    args = ap.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== E2 时间分批留出实验（dev={dev}, seeds={seeds}）===", flush=True)

    encoder = FrozenTextEncoder("BAAI/bge-small-zh-v1.5", device=dev)
    tr_store, tr_graph, tr_examples = build_disjoint_train_world(encoder)

    seed_records = []
    for sd in seeds:
        torch.manual_seed(sd)
        net = IterativeRecallNet(emb_dim=encoder.dim, hidden_dim=32, low_rank=16).to(dev)
        train_iterative_recall(
            net, tr_store, tr_graph, tr_examples,
            epochs=args.pretrain_epochs, lr=4e-3, candidate_budget=args.budget, seed=sd,
        )
        abs_t = net.calibrate_abstain_thresh(tr_store, tr_graph, tr_examples, mode="hybrid", candidate_budget=args.budget)

        store, graph, fixed_old, hard_negs, batches = build_e2_timeline(encoder)
        batch_logs = []
        replay_signals: List[RecallExample] = []

        for b in batches:
            # ① 写入前
            m_before = eval_stage(
                net, store, graph, b["unseen_new_assoc"], fixed_old, b["latest_fact_queries"], hard_negs, abs_t, args.budget
            )

            # 写入本批新事件 + 动态加边（零权重更新）
            for op, eid, day, src, txt, old_id in b["writes"]:
                if op == "supersede" and old_id:
                    ev = store.supersede_event(old_id, txt, day=day, source=src, new_id=eid)
                else:
                    ev = store.add_event(txt, day=day, source=src, event_id=eid)
                graph.add_node_dynamic(ev)

            # ② 写入后、巩固前
            m_after_write = eval_stage(
                net, store, graph, b["unseen_new_assoc"], fixed_old, b["latest_fact_queries"], hard_negs, abs_t, args.budget
            )

            # ③ 巩固：路径强化成边 + 带历史回放的小步权重巩固 + 留存/误触发验收门控
            for sig in b["consol_signals"]:
                if len(sig.gold_ids) >= 2:
                    graph.reinforce_path(sig.gold_ids, delta=0.8)

            snap = copy.deepcopy(net.state_dict())
            # 验收门控在训练集负例与本批前旧回放上检查（绝不偷看评测题）
            gate_before_fpr = sum(
                1 for ex in tr_examples if ex.category in ("hard_neg", "insufficient")
                and not net.recall(tr_store, tr_graph, ex.query, mode="hybrid", candidate_budget=args.budget, abstain_thresh=abs_t).abstained
            ) / max(1, sum(1 for ex in tr_examples if ex.category in ("hard_neg", "insufficient")))

            active_replay = [
                sig for sig in replay_signals
                if all(gid in store and store.get(gid).status == "active" for gid in sig.gold_ids)
            ]
            consol_mix = list(b["consol_signals"]) + active_replay
            train_iterative_recall(
                net, store, graph, consol_mix,
                epochs=args.consol_epochs, lr=1e-3, candidate_budget=args.budget, seed=sd + b["batch_id"] * 10,
            )
            # 同时在原训练世界做小步回放巩固，保持跨主题泛化与拒答边界
            train_iterative_recall(
                net, tr_store, tr_graph, tr_examples,
                epochs=3, lr=1e-3, candidate_budget=args.budget, seed=sd + b["batch_id"] * 10 + 1,
            )

            gate_after_fpr = sum(
                1 for ex in tr_examples if ex.category in ("hard_neg", "insufficient")
                and not net.recall(tr_store, tr_graph, ex.query, mode="hybrid", candidate_budget=args.budget, abstain_thresh=abs_t).abstained
            ) / max(1, sum(1 for ex in tr_examples if ex.category in ("hard_neg", "insufficient")))

            committed = gate_after_fpr <= max(0.10, gate_before_fpr + 0.05)
            if not committed:
                net.load_state_dict(snap)
            replay_signals.extend(b["consol_signals"])

            m_after_consol = eval_stage(
                net, store, graph, b["unseen_new_assoc"], fixed_old, b["latest_fact_queries"], hard_negs, abs_t, args.budget
            )
            batch_logs.append({
                "batch_id": b["batch_id"],
                "label": b["label"],
                "committed": committed,
                "before_write": m_before,
                "after_write_before_consol": m_after_write,
                "after_consol": m_after_consol,
            })

        seed_records.append({"seed": sd, "abstain_thresh": abs_t, "batches": batch_logs})

    # 聚合 3 种子均值
    stages = ("before_write", "after_write_before_consol", "after_consol")
    mkeys = ("new_assoc_exact", "new_assoc_target", "old_retention", "latest_fact_acc", "old_value_leak", "hard_neg_fpr")
    agg = []
    print(f"\n=== E2 时间分批留出结果（{len(seeds)} 种子均值，未见过的新问法）===")
    for bi in range(3):
        b_label = seed_records[0]["batches"][bi]["label"]
        print(f"\n[{b_label}]")
        print(f"{'时刻':<26} | {'新关联全中':>8} | {'新关联目标':>8} | {'旧题留存':>7} | {'最新事实':>7} | {'旧值泄露↓':>8} | {'难负例FPR↓':>9}")
        print("-" * 92)
        b_agg = {"batch_id": bi + 1, "label": b_label}
        for st in stages:
            vals = {
                mk: round(sum(sr["batches"][bi][st][mk] for sr in seed_records) / len(seed_records), 4)
                for mk in mkeys
            }
            b_agg[st] = vals
            print(
                f"{st:<26} | {vals['new_assoc_exact']:>8.1%} | {vals['new_assoc_target']:>8.1%} | "
                f"{vals['old_retention']:>7.1%} | {vals['latest_fact_acc']:>7.1%} | "
                f"{vals['old_value_leak']:>8.1%} | {vals['hard_neg_fpr']:>9.1%}",
                flush=True,
            )
        b_agg["commit_rate"] = f"{sum(1 for sr in seed_records if sr['batches'][bi]['committed'])}/{len(seed_records)}"
        agg.append(b_agg)

    elapsed = round(time.time() - t0, 1)
    out_path = ROOT / f"runs/{time.strftime('%Y%m%d-%H%M%S')}_assoc_e2.json"
    payload = {
        "experiment": "E2_temporal_holdout",
        "encoder": "BAAI/bge-small-zh-v1.5",
        "budget": args.budget,
        "seeds": seeds,
        "elapsed_s": elapsed,
        "summary_mean": agg,
        "seed_records": seed_records,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存：{out_path}（耗时 {elapsed}s）", flush=True)


if __name__ == "__main__":
    main()
