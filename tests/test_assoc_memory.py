"""Unit tests for assoc_memory.py (fast deterministic encoder, no model download needed)."""
import hashlib
import torch
import torch.nn.functional as F

from assoc_memory import (
    EpisodicStore,
    Fixed2HopGraphBaseline,
    IterativeRecallNet,
    OneShotBaseline,
    PlainTwoRoundBaseline,
    RecallExample,
    SparseAssociativeGraph,
    extract_sparse_tokens,
    train_iterative_recall,
)


class HashEncoder:
    """Deterministic bag-of-ngrams L2-normalized encoder for fast unit tests."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def encode_one(self, text: str) -> torch.Tensor:
        vec = torch.zeros(self.dim, dtype=torch.float32)
        toks = extract_sparse_tokens(text) or {text}
        for tok in toks:
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            vec[h % self.dim] += 1.0
            vec[(h >> 8) % self.dim] += 0.5
        return F.normalize(vec, dim=-1)

    def encode(self, texts):
        return torch.stack([self.encode_one(t) for t in texts], dim=0)


def test_episodic_store_crud_and_supersede():
    enc = HashEncoder(dim=64)
    store = EpisodicStore(enc)
    e1 = store.add_event("周三晚上去体育馆打羽毛球", day=1, source="diary_01", event_id="e1")
    e2 = store.add_event("下班买了冰镇无糖可乐", day=2, source="diary_02", event_id="e2")
    assert len(store.active_events()) == 2
    assert e1.status == "active"

    e3 = store.supersede_event("e1", "羽毛球时间改到了周五晚上八点", day=5, source="diary_05", new_id="e3")
    assert store.get("e1").status == "superseded"
    assert store.get("e1").superseded_by == "e3"
    assert e3.supersedes == "e1"
    assert [e.event_id for e in store.active_events()] == ["e2", "e3"]

    assert store.delete_event("e2") is True
    assert [e.event_id for e in store.active_events()] == ["e3"]

    sub = store.clone_without(["e3"])
    assert "e3" not in sub
    assert "e1" in sub


def test_sparse_graph_cap_dynamic_and_reinforce():
    enc = HashEncoder(dim=64)
    store = EpisodicStore(enc)
    for i in range(25):
        store.add_event(f"第{i}天日记，去了公园和超市买东西", day=i + 1, source=f"d{i}", event_id=f"e{i}")
    graph = SparseAssociativeGraph(store, max_neighbors=16)
    for e in store.active_events():
        assert len(graph.neighbors(e.event_id)) <= 16

    new_ev = store.add_event("把备用钥匙交给邻居陈阿姨帮忙照顾小狗", day=30, source="d30", event_id="e_new")
    graph.add_node_dynamic(new_ev)
    assert "e_new" in graph.adj
    for e in store.active_events():
        assert len(graph.neighbors(e.event_id)) <= 16

    # Reinforce a jump-out-of-graph path
    graph.reinforce_path(["e0", "e_new"], delta=0.6)
    assert "e_new" in graph.adj["e0"]
    assert graph.adj["e0"]["e_new"].consolidated_weight >= 0.6


def test_iterative_recall_modes_and_training():
    enc = HashEncoder(dim=64)
    store = EpisodicStore(enc)
    store.add_event("下午打雷下大雨，旺财吓得钻进书桌底下发抖", day=1, source="d1", event_id="e_thunder")
    store.add_event("把南岸公寓备用钥匙交给邻居陈阿姨，托她晚归时帮忙看看旺财", day=12, source="d12", event_id="e_care")
    store.add_event("周末去阿尔伯特公园喝冰燕麦拿铁晒太阳", day=3, source="d3", event_id="e_park")
    store.add_event("在公司修复了前端异步状态竞争问题", day=4, source="d4", event_id="e_work")
    graph = SparseAssociativeGraph(store, max_neighbors=16)

    net = IterativeRecallNet(emb_dim=64, hidden_dim=32, low_rank=16)
    examples = [
        RecallExample("打雷下大雨我还在加班，家里怎么办", "cross", ["e_thunder", "e_care"], topic="pet"),
        RecallExample("周末去哪个公园喝咖啡", "direct", ["e_park"], topic="park"),
        RecallExample("拍闪电的相机光圈快门参数怎么设", "hard_neg", [], topic="neg"),
    ]
    losses = train_iterative_recall(net, store, graph, examples, epochs=12, lr=5e-3, candidate_budget=4, seed=0)
    assert losses["rank"] >= 0.0

    for mode in ("hybrid", "graph_only", "full_only"):
        res = net.recall(store, graph, "打雷下大雨我还在加班，家里怎么办", mode=mode, beam_width=4, max_rounds=3)
        assert isinstance(res.event_ids, list)
        assert 0 <= len(res.event_ids) <= 2

    # Baselines run cleanly
    b1 = OneShotBaseline()
    b1.calibrate(store, examples)
    assert isinstance(b1.recall(store, "打雷下大雨").event_ids, list)

    b2 = Fixed2HopGraphBaseline()
    b2.calibrate(store, graph, examples)
    assert isinstance(b2.recall(store, graph, "打雷下大雨").event_ids, list)

    b3 = PlainTwoRoundBaseline()
    b3.calibrate(store, examples)
    assert isinstance(b3.recall(store, "打雷下大雨").event_ids, list)


def test_rmp_recall_net_and_smf():
    from assoc_rmp import RMPIterativeRecallNet

    enc = HashEncoder(dim=64)
    store = EpisodicStore(enc)
    store.add_event("下午打雷下大雨，旺财吓得钻进书桌底下发抖", day=1, source="d1", event_id="e1")
    store.add_event("把南岸公寓备用钥匙交给邻居陈阿姨帮忙照看旺财", day=5, source="d5", event_id="e2")
    graph = SparseAssociativeGraph(store, max_neighbors=16)

    rmp = RMPIterativeRecallNet(emb_dim=64, hidden_dim=32, low_rank=16, n_modules=8, top_k=2)
    rmp.seed_addresses_from_queries([e.embedding for e in store.active_events()])
    hits = rmp.profile_hits(enc, ["打雷下雨家里怎么办"])
    assert abs(float(hits.sum().item()) - 1.0) < 1e-4

    sel = rmp.apply_smf_freeze(hits, smf_k=2)
    assert len(sel) == 2
    assert rmp.addr.requires_grad is False
    for i, exp in enumerate(rmp.experts):
        req = all(p.requires_grad for p in exp.parameters())
        assert req == (i in set(sel))

    res = rmp.recall(store, graph, "打雷下雨家里怎么办", mode="hybrid", candidate_budget=4)
    assert isinstance(res.event_ids, list)


def test_sqlite_persistence_roundtrip(tmp_path):
    from assoc_store_sqlite import SQLiteAssocPersistence

    enc = HashEncoder(dim=64)
    store = EpisodicStore(enc)
    store.add_event("十月约好周三晚上打羽毛球", day=1, source="d1", event_id="e_old")
    store.supersede_event("e_old", "十一月起羽毛球改到周五晚上八点", day=30, source="d30", new_id="e_new")
    store.add_event("把备用钥匙交给邻居陈阿姨", day=35, source="d35", event_id="e_key")
    store.add_event("临时测试记录随后删除", day=36, source="d36", event_id="e_del")
    store.delete_event("e_del")

    graph = SparseAssociativeGraph(store, max_neighbors=16)
    graph.reinforce_path(["e_new", "e_key"], delta=0.9)

    db = SQLiteAssocPersistence(tmp_path / "assoc.db")
    db.save_store_and_graph(store, graph)
    store2, graph2 = db.load_store_and_graph(enc, max_neighbors=16)
    db.close()

    assert store2.get("e_old").status == "superseded"
    assert store2.get("e_old").superseded_by == "e_new"
    assert store2.get("e_new").status == "active"
    assert store2.get("e_del").status == "deleted"
    assert [e.event_id for e in store2.active_events()] == ["e_new", "e_key"]
    assert "e_key" in graph2.adj["e_new"]
    assert graph2.adj["e_new"]["e_key"].consolidated_weight >= 0.9


class HashTokenEncoder(HashEncoder):
    """Deterministic token-level encoder for testing assoc_neural_v2 without downloading weights."""

    def encode_one_both(self, text: str):
        cls_e = self.encode_one(text)
        # 构造伪 token 序列（按字符块滑窗），零外部依赖
        chunks = [text[i : i + 2] for i in range(0, max(1, len(text) - 1), 2)] or [text]
        tok_e = torch.stack([self.encode_one(c) for c in chunks], dim=0)
        return cls_e, tok_e

    def encode_both(self, texts, batch_size: int = 32):
        return [self.encode_one_both(t) for t in texts]


def test_v2_pure_neural_recall_and_training():
    from assoc_neural_v2 import (
        GeneralNeuralRecallNet,
        NeuralEpisodicStore,
        PureNeuralGraph,
        train_neural_v2,
    )

    enc = HashTokenEncoder(dim=64)
    store = NeuralEpisodicStore(enc)
    store.add_event("下午打雷下大雨，旺财吓得钻进书桌底下发抖", day=1, source="d1", event_id="e1")
    store.add_event("把南岸公寓备用钥匙交给邻居陈阿姨帮忙照看旺财", day=5, source="d5", event_id="e2")
    store.add_event("陈阿姨住在1402室手机号尾号9108", day=6, source="d6", event_id="e3")
    store.add_event("周末去公园喝冰燕麦拿铁晒太阳", day=2, source="d2", event_id="e4")
    graph = PureNeuralGraph(store, max_neighbors=16)

    net = GeneralNeuralRecallNet(d_model=16, n_heads=2, n_layers=1)
    examples = [
        RecallExample("打雷加班家里怎么办", "cross", ["e1", "e2"], "pet"),
        RecallExample("打雷加班托邻居照看那位邻居住几零几", "cross", ["e1", "e2", "e3"], "pet3"),
        RecallExample("周末去哪儿喝咖啡", "direct", ["e4"], "park"),
        RecallExample("拍闪电相机参数怎么调", "hard_neg", [], "neg"),
    ]
    losses = train_neural_v2(net, store, graph, examples, epochs=4, lr=5e-3, candidate_budget=4, accum_steps=2, seed=0)
    assert losses["rank"] >= 0.0

    for mode in ("hybrid", "graph_only", "full_only"):
        res = net.recall(store, graph, "打雷加班托邻居照看那位邻居住几零几", mode=mode, max_rounds=3, candidate_budget=4)
        assert isinstance(res["event_ids"], list)
        assert 0 <= len(res["event_ids"]) <= 3


