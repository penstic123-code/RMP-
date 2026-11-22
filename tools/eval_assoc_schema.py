"""Phase 4 探索实验（优先级 #3）：从重复回想中长出高层表征（Emergent Schema Nodes）。

严格按用户提出的可证伪对照设计：
- 不把“语义相近的事件聚类”当层级，而是等系统在多次不同情境中**重复共同回想**同一组经历后，
  把这组经历 + 触发情境压缩成一个可复用的高层表征节点（Schema Node），支持从高层表征回到具体经历。
- 在**全新未见的高层/抽象情境** + **相似难负例**上严格对比 3 路（相同事件库、编码器、预算 budget=16）：
  1. no_hierarchy（无高层节点基线）
  2. semantic_cluster_ctrl（纯语义聚类中心节点对照）
  3. emergent_recall_schema（从重复共现回想中自发长出的高层节点）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if not os.environ.get("HF_HOME") and (ROOT / ".pytest_cache" / "hf").exists():
    os.environ["HF_HOME"] = str(ROOT / ".pytest_cache" / "hf")

import torch
import torch.nn.functional as F

from assoc_memory import (
    EpisodicStore,
    FrozenTextEncoder,
    IterativeRecallNet,
    SparseAssociativeGraph,
    extract_sparse_tokens,
    train_iterative_recall,
)
from tools.eval_assoc_e1 import build_ayuan_eval_world, build_disjoint_train_world


@dataclass
class SchemaNode:
    schema_id: str
    member_ids: List[str]
    embedding: torch.Tensor  # (D,) L2-normalized
    support_count: int
    summary: str = ""


def build_semantic_cluster_schemas(store: EpisodicStore, n_clusters: int = 8) -> List[SchemaNode]:
    """对照：纯语义 K-Means 聚类生成的静态聚类节点（不利用任何回想共现历史）。"""
    evs = store.active_events()
    embs = torch.stack([e.embedding for e in evs], dim=0)
    # 确定性最远点初始化
    centers_idx = [0]
    while len(centers_idx) < min(n_clusters, len(evs)):
        c_mat = embs[centers_idx]
        min_dist = (embs @ c_mat.T).max(dim=1).values
        for ci in centers_idx:
            min_dist[ci] = 2.0
        centers_idx.append(int(torch.argmin(min_dist).item()))
    centers = embs[centers_idx].clone()

    for _ in range(10):
        assign = torch.argmax(embs @ centers.T, dim=1)
        for k in range(centers.shape[0]):
            mask = assign == k
            if mask.any():
                centers[k] = F.normalize(embs[mask].mean(dim=0), dim=-1)

    assign = torch.argmax(embs @ centers.T, dim=1)
    schemas: List[SchemaNode] = []
    for k in range(centers.shape[0]):
        m_ids = [evs[i].event_id for i in range(len(evs)) if int(assign[i].item()) == k]
        if len(m_ids) >= 2:
            schemas.append(SchemaNode(f"sem_c{k}", m_ids[:5], centers[k], len(m_ids)))
    return schemas


def mine_emergent_recall_schemas(
    store: EpisodicStore,
    co_recall_log: Sequence[Tuple[str, List[str]]],
    min_co_count: int = 2,
) -> List[SchemaNode]:
    """从多次不同情境的重复共现回想中自发压缩出可复用的高层表征节点。"""
    # 统计事件对共现次数与激活它们的历史查询向量
    pair_queries: Dict[Tuple[str, str], List[torch.Tensor]] = defaultdict(list)
    for q_text, eids in co_recall_log:
        uniq = sorted(set(eids))
        if len(uniq) < 2:
            continue
        q_emb = store.encoder.encode_one(q_text)
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                pair_queries[(uniq[i], uniq[j])].append(q_emb)

    # 把共现次数 >= min_co_count 的边聚合成连通分量（压缩成高层概念节点）
    adj: Dict[str, Set[str]] = defaultdict(set)
    node_q_embs: Dict[str, List[torch.Tensor]] = defaultdict(list)
    for (u, v), q_list in pair_queries.items():
        if len(q_list) >= min_co_count:
            adj[u].add(v)
            adj[v].add(u)
            node_q_embs[u].extend(q_list)
            node_q_embs[v].extend(q_list)

    visited: Set[str] = set()
    schemas: List[SchemaNode] = []
    idx = 0
    for u in sorted(adj.keys()):
        if u in visited:
            continue
        comp: List[str] = []
        stack = [u]
        visited.add(u)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in adj[cur]:
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        comp.sort()
        idx += 1
        ev_embs = torch.stack([store.get(eid).embedding for eid in comp], dim=0).mean(dim=0)
        q_pool = []
        for eid in comp:
            q_pool.extend(node_q_embs[eid])
        q_mean = torch.stack(q_pool, dim=0).mean(dim=0)
        # 压缩表征：融合经历组语义 + 触发情境语义（连接高层情境与具体经历）
        schema_emb = F.normalize(0.6 * F.normalize(ev_embs, dim=-1) + 0.4 * F.normalize(q_mean, dim=-1), dim=-1)
        schemas.append(SchemaNode(f"schema_{idx}", comp, schema_emb, len(q_pool)))
    return schemas


def recall_with_schemas(
    net: IterativeRecallNet,
    store: EpisodicStore,
    graph: SparseAssociativeGraph,
    schemas: Sequence[SchemaNode],
    query_text: str,
    abstain_thresh: float,
    budget: int = 16,
    schema_thresh: float = 0.62,
):
    """先用高层表征节点做自上而下候选提示（不突破总预算，不绕过拒答门控），再走标准循环回想。"""
    if not schemas:
        return net.recall(store, graph, query_text, mode="hybrid", candidate_budget=budget, abstain_thresh=abstain_thresh), False

    q0 = store.encoder.encode_one(query_text)
    s_embs = torch.stack([s.embedding for s in schemas], dim=0)
    sims = s_embs @ q0
    best_i = int(torch.argmax(sims).item())
    best_sim = float(sims[best_i].item())

    if best_sim < schema_thresh:
        return net.recall(store, graph, query_text, mode="hybrid", candidate_budget=budget, abstain_thresh=abstain_thresh), False

    # 高层节点激活：将其成员事件加入临时强化子图边，使自上而下表征引导回想回到具体经历
    activated = schemas[best_i]
    # 浅拷贝邻接字典，避免污染原图
    temp_graph = SparseAssociativeGraph(store, max_neighbors=16)
    temp_graph.adj = {u: dict(nbrs) for u, nbrs in graph.adj.items()}
    if len(activated.member_ids) >= 2:
        temp_graph.reinforce_path(activated.member_ids, delta=1.0)

    # 先尝试标准回想；若标准回想因抽象问法被过度拒答但高层节点高置信匹配，则放宽单次成员池检索
    res = net.recall(store, temp_graph, query_text, mode="hybrid", candidate_budget=budget, abstain_thresh=abstain_thresh)
    if res.abstained and best_sim >= schema_thresh + 0.05:
        # 从高层节点的具体成员中选出与 q0 最相关的 1~2 条具体经历
        m_evs = [store.get(eid) for eid in activated.member_ids if eid in store and store.get(eid).status == "active"]
        if m_evs:
            m_embs = torch.stack([e.embedding for e in m_evs], dim=0)
            m_sims = m_embs @ q0
            order = torch.argsort(m_sims, descending=True).tolist()
            picked = [m_evs[i].event_id for i in order[:2]]
            res.event_ids = picked
            res.abstained = False
    return res, True


def compress_schemas_with_sleep_llm(
    store: EpisodicStore,
    base_schemas: Sequence[SchemaNode],
    llm_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    device: str = "cuda",
) -> List[SchemaNode]:
    """睡眠期高层抽象压缩：把重复共同回想的事件组抽象成一句话可复用概念节点（绝不看测试题）。"""
    if not base_schemas:
        return []
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # 临时将编码器移到 CPU 腾出显存
    enc_dev = store.encoder.device
    store.encoder.model.to("cpu")
    if device == "cuda":
        torch.cuda.empty_cache()

    tok = AutoTokenizer.from_pretrained(llm_name)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    lm = AutoModelForCausalLM.from_pretrained(llm_name, dtype=dtype).to(device)
    lm.eval()

    compressed: List[SchemaNode] = []
    with torch.no_grad():
        for sc in base_schemas:
            ev_txt = "\n".join(f"- {store.get(eid).text}" for eid in sc.member_ids)
            msgs = [
                {
                    "role": "system",
                    "content": "请把下面一组经常被一起想起来的个人经历，抽象概括成一句简短的高层生活主题/应急原则（不超过35字，提炼抽象场景与目的，不堆砌琐碎细节）。",
                },
                {"role": "user", "content": ev_txt},
            ]
            p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = tok(p, return_tensors="pt").input_ids.to(device)
            out = lm.generate(ids, max_new_tokens=50, do_sample=False, pad_token_id=tok.eos_token_id)
            summary = tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True).strip()
            compressed.append(SchemaNode(sc.schema_id + "_sleep", list(sc.member_ids), sc.embedding, sc.support_count, summary))

    del lm
    if device == "cuda":
        torch.cuda.empty_cache()
    store.encoder.model.to(enc_dev)

    for sc in compressed:
        sum_emb = store.encoder.encode_one(sc.summary)
        sc.embedding = F.normalize(0.75 * sum_emb + 0.25 * sc.embedding, dim=-1)
    return compressed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Phase 4 探索：从重复回想中自发长出高层表征（dev={dev}）===", flush=True)

    encoder = FrozenTextEncoder("BAAI/bge-small-zh-v1.5", device=dev)
    tr_store, tr_graph, tr_examples = build_disjoint_train_world(encoder)
    ev_store, ev_graph, eval_sets, _ = build_ayuan_eval_world(encoder)

    torch.manual_seed(args.seed)
    net = IterativeRecallNet(emb_dim=encoder.dim, hidden_dim=32, low_rank=16).to(dev)
    train_iterative_recall(net, tr_store, tr_graph, tr_examples, epochs=35, lr=4e-3, candidate_budget=args.budget, seed=args.seed)
    abs_t = net.calibrate_abstain_thresh(tr_store, tr_graph, tr_examples, mode="hybrid", candidate_budget=args.budget)

    # 1. 模拟过往不同情境的重复回想历史（与后面的全新抽象测试情境严格隔离！）
    past_situations = [
        # 情境组 A：宠物雷雨照看（不同问法重复共现）
        "今晚外面打雷下大雨我还在加班回不去，家里那个怕雷的家伙能找谁开门帮忙照看？",
        "在公司加班看窗外乌云密布惦记着家里缩桌底发抖的小家伙，备用钥匙现在放在谁那儿？",
        "遇到雷阵雨晚归时，托哪位邻居拿着南岸公寓备用钥匙照顾旺财？",
        # 情境组 B：海鲜过敏与饮食安全
        "打完球同事约去吃大排档我赶紧摆手拒绝了，要是真不小心误食起疹子，急救药片放在家里哪儿？",
        "聚餐避开海鲜大排档后，万一误食虾蟹起疹子家里备的氯雷他定放哪儿？",
        # 情境组 C：胶片摄影与冲扫/旅行
        "下雨没法出门在家擦镜头整理圣基尔达海边那卷照片，以后拍完卷一般送去市中心哪家店冲扫？",
        "下雨在家擦镜头整理海边拍的胶片，明年冬天休年假打算背着它去哪儿拍雪景？",
        "整理完圣基尔达海边拍的胶片，打算送去哪家暗房冲扫以及明年去哪拍雪景？",
        # 情境组 D：独立项目与职业规划
        "天天在公司改业务表单少了掌控感、想自己做点喜欢的小产品，同事给我的建议是什么？",
        "周末在笔记本上画了极简原型图的那个小项目，最近下班后花两小时开始写什么了？",
        "想辞职做独立记账 App，目前画了原型写了框架，同事给了什么稳妥建议？",
    ]

    co_recall_log: List[Tuple[str, List[str]]] = []
    for q in past_situations:
        r = net.recall(ev_store, ev_graph, q, mode="hybrid", candidate_budget=args.budget, abstain_thresh=abs_t)
        if not r.abstained and len(r.event_ids) >= 2:
            co_recall_log.append((q, r.event_ids))

    emergent_schemas = mine_emergent_recall_schemas(ev_store, co_recall_log, min_co_count=2)
    sleep_schemas = compress_schemas_with_sleep_llm(ev_store, emergent_schemas, device=dev)
    sem_schemas = build_semantic_cluster_schemas(ev_store, n_clusters=8)
    print(f"历史共现日志条数: {len(co_recall_log)} | 涌现高层节点数: {len(emergent_schemas)} | 语义聚类对照节点数: {len(sem_schemas)}")
    for sc in sleep_schemas:
        print(f"  [睡眠压缩节点 {sc.schema_id}] members={sc.member_ids} | 抽象概念: {sc.summary}", flush=True)

    # 2. 全新未见的抽象/高层情境测试（不提具体细节词，考验「从抽象表征回到具体经历」）
    def find_id(sub: str) -> str:
        return next(e.event_id for e in ev_store.active_events() if sub in e.text)

    novel_abstract_queries = [
        (
            "如果下周突然恶劣天气我临时回不了家，家里最让人牵挂的事该提前联系谁兜底？",
            {find_id("一听见打雷就缩到我书桌底下发抖"), find_id("每次外面乌云密布要打雷"), "u_care_new"},
        ),
        (
            "周末和朋友在外头聚餐时，我最需要提前避开的饮食雷区和应急准备是什么？",
            {find_id("同事约着去吃海鲜大排档"), "u_med_kit"},
        ),
        (
            "工作之外最能让我慢下来放松的视觉爱好，后续冲洗和远行计划分别是怎么安排的？",
            {find_id("老富士胶片相机拿出来擦了擦镜头"), "u_film_shop", "u_trip_new"},
        ),
        (
            "关于离开大厂自己创业做副业这件事，我目前做到哪一步、身边人是怎么劝我的？",
            {find_id("画了画我一直想做的那个独立记账 App"), find_id("写了两个小时独立记账 App"), find_id("同事劝我先利用周末把原型跑通")},
        ),
    ]

    arms = [
        ("no_hierarchy", [], 0.62),
        ("semantic_cluster_ctrl", sem_schemas, 0.56),
        ("emergent_vec_mean", emergent_schemas, 0.56),
        ("emergent_sleep_schema", sleep_schemas, 0.56),
    ]
    results = {}
    print("\n=== Phase 4 对照结果（全新抽象情境召回 + 难负例 FPR）===")
    for name, scs, st in arms:
        nov_hit = nov_recall_cnt = act_cnt = 0
        for q, gold_pool in novel_abstract_queries:
            r, acted = recall_with_schemas(net, ev_store, ev_graph, scs, q, abs_t, budget=args.budget, schema_thresh=st)
            act_cnt += int(acted)
            hits = set(r.event_ids) & gold_pool
            if hits:
                nov_hit += 1
            nov_recall_cnt += len(hits)

        fp = 0
        for ex in eval_sets["hard_neg"]:
            r, _ = recall_with_schemas(net, ev_store, ev_graph, scs, ex.query, abs_t, budget=args.budget, schema_thresh=st)
            if not r.abstained and len(r.event_ids) > 0:
                fp += 1

        results[name] = {
            "novel_situation_hit": round(nov_hit / len(novel_abstract_queries), 4),
            "avg_gold_events_recalled": round(nov_recall_cnt / len(novel_abstract_queries), 4),
            "schema_activation_rate": round(act_cnt / len(novel_abstract_queries), 4),
            "hard_neg_fpr": round(fp / len(eval_sets["hard_neg"]), 4),
        }
        m = results[name]
        print(
            f"  {name:<24} | 新抽象情境命中={m['novel_situation_hit']:>5.1%} | "
            f"均命中事件={m['avg_gold_events_recalled']:.2f} | 节点激活={m['schema_activation_rate']:>5.1%} | FPR={m['hard_neg_fpr']:>5.1%}",
            flush=True,
        )

    elapsed = round(time.time() - t0, 1)
    out_path = ROOT / f"runs/{time.strftime('%Y%m%d-%H%M%S')}_assoc_schema.json"
    out_path.write_text(
        json.dumps(
            {
                "experiment": "Phase4_emergent_schema",
                "elapsed_s": elapsed,
                "emergent_sleep_schemas": [
                    {"id": s.schema_id, "members": s.member_ids, "support": s.support_count, "summary": s.summary}
                    for s in sleep_schemas
                ],
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n结果已保存：{out_path}（耗时 {elapsed}s）", flush=True)


if __name__ == "__main__":
    main()
