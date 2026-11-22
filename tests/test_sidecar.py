"""冻结 LLM + RMP 旁路记忆池 + 神经海马体槽表（llm_sidecar.py）的完整单元测试（使用轻量 mock Transformer，无需下载大模型）。"""
import os, sys, tempfile
from types import SimpleNamespace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn
from llm_sidecar import LLMSidecarWrapper, SidecarLearner


class _MockBlock(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.ff = nn.Sequential(nn.LayerNorm(H), nn.Linear(H, H), nn.GELU(), nn.Linear(H, H))

    def forward(self, x):
        return (x + self.ff(x),)


class _MockCausalLM(nn.Module):
    def __init__(self, vocab=64, H=32, n_layers=4):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=H, vocab_size=vocab)
        self.emb = nn.Embedding(vocab, H)
        self.model = SimpleNamespace(layers=nn.ModuleList([_MockBlock(H) for _ in range(n_layers)]))
        self.layers_reg = self.model.layers                        # 注册进参数树
        self.head = nn.Linear(H, vocab)

    def forward(self, input_ids, **kwargs):
        h = self.emb(input_ids)
        for blk in self.model.layers:
            h = blk(h)[0]
        return SimpleNamespace(logits=self.head(h))


def test_sidecar_identity_at_init():
    """零初始化出口：挂载旁路与海马体后，初始输出与原版冻结 LLM 逐位一致。"""
    torch.manual_seed(0)
    lm = _MockCausalLM()
    idx = torch.randint(0, 64, (2, 16))
    with torch.no_grad():
        base_lg = lm(idx).logits.clone()
    w = LLMSidecarWrapper(lm, attach_layer=-2, sidecar_cfg=dict(N=8, k=2, G=2, D=16, pd=8, pm=4, phidden=16, ctx_decay=(0.5, 0.95)),
                          hippo_cfg=dict(n_slots=32, d_key=16, d_val=16))
    with torch.no_grad():
        wrap_lg = w(idx).logits
    assert torch.allclose(base_lg, wrap_lg, atol=1e-6)
    assert all(not p.requires_grad for p in w.backbone.parameters())


def test_sidecar_online_smf_hippo_cross_session_and_sleep():
    """端到端：冻结底座逐位不变、SMF 仅更新 <=smf_k 个模块、海马体跨会话保留、睡眠巩固与存盘加载。"""
    torch.manual_seed(0)
    lm = _MockCausalLM()
    w = LLMSidecarWrapper(lm, attach_layer=-2, sidecar_cfg=dict(N=8, k=2, G=2, D=16, pd=8, pm=4, phidden=16, ctx_decay=(0.5, 0.95)),
                          hippo_cfg=dict(n_slots=32, d_key=16, d_val=16))
    # 给独立模块出口一个小非零初值以测试反向梯度流入 pool_mlp
    with torch.no_grad():
        w.sidecar.mod_out_b.normal_(0, 0.05)
        w.sidecar.pool.core.W2.normal_(0, 0.05)
    backbone_snap = {k: v.clone() for k, v in w.backbone.state_dict().items()}
    L = SidecarLearner(w, lr=1e-2, smf_k=2)
    bg = [torch.randint(0, 64, (2, 12)) for _ in range(2)]
    L.profile_bg(bg)
    w1_0 = w.sidecar.pool.core.W1.clone()
    new_toks = torch.randint(0, 64, (1, 16))
    info = L.step_online(new_toks)
    assert info['written_slots'] > 0 and w.hippo.used.sum().item() > 0
    # 冻结底座逐位不变
    assert all(torch.equal(backbone_snap[k], v) for k, v in w.backbone.state_dict().items())
    # SMF 最多更新 smf_k=2 个模块
    changed = (w.sidecar.pool.core.W1 != w1_0).any(dim=(1, 2)).sum().item()
    assert 1 <= changed <= 2
    # 切换新会话：RMP 状态清零，海马体槽表保留
    used_before = w.hippo.used.sum().item()
    w.reset_session()
    assert w.sidecar.state is None and w.hippo.cur_session == 1
    assert w.hippo.used.sum().item() == used_before
    # 睡眠巩固
    slp = L.sleep_consolidate(steps=3, anchor_batches=bg)
    assert slp['steps'] == 3 and 1 <= slp['consolidated_modules'] <= 2
    assert all(torch.equal(backbone_snap[k], v) for k, v in w.backbone.state_dict().items())
    # 轻量化原子存盘（含 .bak 快照）与加载
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'mem.pt')
        w.save_memory(p, learner=L)
        w.save_memory(p, learner=L)
        assert os.path.exists(p + '.bak')
        w.hippo.used.zero_()
        w.load_memory(p, learner=L)
        assert w.hippo.used.sum().item() == used_before and w.hippo.cur_session == 1


def test_bypass_updates_last_h():
    """回归测试：bypass=True 抽取纯冻结原脑隐状态时，不同输入必须更新 _last_h（不能复用旧缓存）。"""
    torch.manual_seed(0)
    lm = _MockCausalLM()
    w = LLMSidecarWrapper(lm, attach_layer=-2, sidecar_cfg=dict(N=8, k=2, G=2, D=16, pd=8, pm=4, phidden=16, ctx_decay=(0.5, 0.95)),
                          hippo_cfg=dict(n_slots=32, d_key=16, d_val=16))
    w.eval(); w.bypass = True
    with torch.no_grad():
        w(torch.tensor([[1, 2, 3, 4]]))
        h1 = w._last_h.clone()
        w(torch.tensor([[10, 20, 30, 40]]))
        h2 = w._last_h.clone()
    assert not torch.allclose(h1, h2), 'bypass=True 时不同输入不应返回相同 _last_h'


def test_memory_store_crud_and_supersede():
    """SQLite 长期记忆库：新增、同键改口覆盖、编辑、删除与教学规则状态流转。"""
    from memory_store import MemoryStore
    with tempfile.TemporaryDirectory() as td:
        st = MemoryStore(os.path.join(td, 'mem.db'))
        id1 = st.add_fact('I live in Melbourne.', fact_key='city', session_id=0)
        id2 = st.add_fact('My cat is Mirafe.', fact_key='cat', session_id=0)
        st.add_fact('I live in Sydney.', fact_key='city', session_id=1)   # 改口覆盖 Melbourne
        active = st.list_facts(active_only=True)
        texts = [r[2] for r in active]
        assert 'I live in Sydney.' in texts and 'I live in Melbourne.' not in texts
        assert st.edit_fact(id2, 'My cat is named Mirafe (3 years old).')
        assert any('3 years old' in r[2] for r in st.list_facts())
        assert st.delete_fact(id2)
        assert len(st.list_facts()) == 1
        st.add_rule('What is code?', 'ORION-94')
        assert len(st.list_rules('staged')) == 1
        st.mark_staged_confirmed()
        assert len(st.list_rules('confirmed')) == 1 and len(st.list_rules('staged')) == 0
        st.close()


def test_sparse_active_matches_dense_sidecar():
    """O(k) 稀疏活跃模块执行（支持 N=512/1024）与全量执行在前向输出及参数梯度上严格等价。"""
    import copy
    torch.manual_seed(0)
    lm = _MockCausalLM()
    w1 = LLMSidecarWrapper(lm, attach_layer=-2, sidecar_cfg=dict(N=32, k=2, G=4, D=16, pd=8, pm=4, phidden=16, out_rank=4, ctx_decay=(0.5, 0.85, 0.95, 0.99)),
                           hippo_cfg=dict(n_slots=16, d_key=16, d_val=16))
    with torch.no_grad():
        w1.sidecar.mod_out_b.normal_(0, 0.05)
    w2 = copy.deepcopy(w1)
    w1.sidecar.pool.sparse_active = True
    w2.sidecar.pool.sparse_active = False
    idx = torch.randint(0, 64, (1, 12))
    lg1 = w1(idx).logits; lg2 = w2(idx).logits
    assert torch.allclose(lg1, lg2, atol=1e-4)
    lg1.sum().backward(); lg2.sum().backward()
    assert torch.allclose(w1.sidecar.mod_out_b.grad, w2.sidecar.mod_out_b.grad, atol=1e-4)
    assert torch.allclose(w1.sidecar.pool.core.W1.grad, w2.sidecar.pool.core.W1.grad, atol=1e-4)


def test_keyed_fact_protected_from_unkeyed_supersede():
    """回归测试：带显式 fact_key 的正式事实，禁止被无键闲聊误覆盖；只有同键新写入才会覆盖。"""
    torch.manual_seed(0)
    lm = _MockCausalLM()
    w = LLMSidecarWrapper(lm, attach_layer=-2, sidecar_cfg=dict(N=8, k=2, G=2, D=16, pd=8, pm=4, phidden=16, ctx_decay=(0.5, 0.95)),
                          hippo_cfg=dict(n_slots=16, d_key=16, d_val=16))
    h = torch.randn(1, 4, 32)
    w.hippo.write_fact(h, 'My car is a Tesla Model 3.', fact_key='car')
    # 即使隐向量完全一致、词法高度重叠，无键闲聊也绝不能覆盖 fact_key='car'
    w.hippo.write_fact(h, 'My car was washed at a Tesla Model shop today.', fact_key='')
    active_texts = [w.hippo.slot_texts[s] for s in (w.hippo.used & (~w.hippo.superseded)).nonzero(as_tuple=False)[:, 0].tolist()]
    assert 'My car is a Tesla Model 3.' in active_texts
    # 只有同键改口才会覆盖
    w.hippo.write_fact(h, 'My car is a Porsche Taycan.', fact_key='car')
    active_texts = [w.hippo.slot_texts[s] for s in (w.hippo.used & (~w.hippo.superseded)).nonzero(as_tuple=False)[:, 0].tolist()]
    assert 'My car is a Porsche Taycan.' in active_texts and 'My car is a Tesla Model 3.' not in active_texts
