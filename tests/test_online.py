import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from dspn_text import DSPNLayered
from online import OnlineLearner, bpc

TEXT = (b"the cat sat on the mat and the dog sat on the log. " * 12)[:512]


def tiny(seed=0):
    torch.manual_seed(seed)
    m = DSPNLayered(layers=2, M=2, d=8, m=4, hidden=8, q=4, emb=8)
    with torch.no_grad():                  # 读出与 W2 初始为零时梯度到不了 MLP 低秩偏移；模拟训练过的模型
        m.readout.weight.normal_(0, 0.3)
        for l in m.layers:
            l.W2.normal_(0, 0.3)
    return m


def snap(m):
    return {k: v.clone() for k, v in m.state_dict().items()}


def test_read_only_changes_online_params():
    m = tiny(); before = snap(m)
    OnlineLearner(m, lr=1e-2).read(TEXT)
    after = m.state_dict()
    online = lambda k: 'lora' in k                     # 默认 params='lora'：读出与偏置不动
    changed = [k for k in before if not torch.equal(before[k], after[k])]
    assert changed and all(online(k) for k in changed)


def test_repeated_read_lowers_bpc():
    m = tiny(); L = OnlineLearner(m, lr=1e-2)
    b0 = bpc(m, TEXT)
    for _ in range(20):
        L.read(TEXT)
    assert bpc(m, TEXT) < b0


def test_sleep_rollback_restores_params():
    m = tiny(); L = OnlineLearner(m, lr=1e-2)
    L.read(TEXT)
    before = snap(m)
    r = L.sleep(TEXT[:256], TEXT[256:], steps=5, old_tol=-1.0)
    assert not r['committed']
    after = m.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_pool_capacity_and_priority():
    m = tiny(); L = OnlineLearner(m, chunk=4, pool_bytes=15)       # 每段 5 字节，容量 3 段
    seg = lambda i: np.full(5, i, dtype=np.uint8)
    for i, p in enumerate([1.0, 3.0, 2.0]):
        L._add(seg(i), p)
    assert L.pool_size == 15
    L._add(seg(3), 0.5)                                            # 低于最低者：不入
    assert sorted(t[0] for t in L.pool) == [1.0, 2.0, 3.0]
    L._add(seg(4), 2.5)                                            # 替换最低的 1.0
    assert sorted(t[0] for t in L.pool) == [2.0, 2.5, 3.0]
    L._add(seg(5), 2.0)                                            # 与最低同分：不替换
    assert sorted(t[0] for t in L.pool) == [2.0, 2.5, 3.0] and L.pool_size <= 15
    L2 = OnlineLearner(tiny(), chunk=4, pool_bytes=15)
    for i, p in enumerate([1.0, 1.0, 3.0]):                        # 序号 1、2、3
        L2._add(seg(i), p)
    L2._add(seg(9), 2.0)                                           # 最低分有两个：换掉较旧的（序号 1）
    assert sorted((t[0], t[1]) for t in L2.pool) == [(1.0, 2), (2.0, 4), (3.0, 3)]


def test_wavenet_online_freeze_enc_and_rewire_false():
    """带池模型：freeze_enc=True 时编码层 lora 不动；sleep(rewire=False) 时 Wr 不动；allocate 触发前不抹零梯度。"""
    import dspn_text as D
    torch.manual_seed(0)
    m = D.build('dspn', 'S', net='wave', N=8, k=2, G=2, D=8, pd=8, pm=4, phidden=16, init='spread')
    with torch.no_grad():
        m.readout.weight.normal_(0, 0.2)
        m.layers[0].W2.normal_(0, 0.2)
        m.layers[1].core.W2.normal_(0, 0.2)
    enc0_a = m.layers[0].lora_a.clone()
    enc0_b = m.layers[0].lora_b.clone()
    pool0_b = m.layers[1].core.lora_b.clone()
    wr0 = m.layers[1].Wr.clone()
    L = OnlineLearner(m, lr=1e-2, chunk=32, allocate=1, warm_chunks=99, freeze_enc=True)
    L.read(TEXT[:128])
    assert torch.equal(m.layers[0].lora_a, enc0_a) and torch.equal(m.layers[0].lora_b, enc0_b)
    assert not torch.equal(m.layers[1].core.lora_b, pool0_b)   # 触发分配前池内模块应正常学习，不被清零
    L.sleep(TEXT[:128], TEXT[128:256], steps=3, old_tol=1.0, rewire=False)
    assert torch.equal(m.layers[1].Wr, wr0)


def test_smf_relative_background_mask():
    """SMF：开启 smf_k 后，每次只更新不超过 smf_k 个模块的低秩偏移。"""
    import dspn_text as D
    torch.manual_seed(0)
    m = D.build('dspn', 'S', net='wave', N=8, k=2, G=2, D=8, pd=8, pm=4, phidden=16, init='spread')
    with torch.no_grad():
        m.readout.weight.normal_(0, 0.2)
        m.layers[0].W2.normal_(0, 0.2)
        m.layers[1].core.W2.normal_(0, 0.2)
    b0 = m.layers[1].core.lora_b.clone()
    L = OnlineLearner(m, lr=1e-2, chunk=32, freeze_enc=True, smf_k=2, bg_bytes=TEXT[:128])
    assert L.bg_freq is not None and L.bg_freq.shape == (8,)
    L.read(TEXT[128:160])                                          # 单块读：最多只有 smf_k=2 个模块被更新
    changed = (m.layers[1].core.lora_b != b0).any(dim=(1, 2)).sum().item()
    assert 1 <= changed <= 2


def test_smf_adamw_zero_drift_on_unselected():
    """SMF 严格掩码步：即使 AdamW 已有动量且开了权重衰减 wd=0.1，每步未选中模块必须逐位不变。"""
    import dspn_text as D
    torch.manual_seed(0)
    m = D.build('dspn', 'S', net='wave', N=8, k=2, G=2, D=8, pd=8, pm=4, phidden=16, init='spread')
    with torch.no_grad():
        m.readout.weight.normal_(0, 0.2)
        m.layers[0].W2.normal_(0, 0.2)
        m.layers[1].core.W2.normal_(0, 0.2)
    L = OnlineLearner(m, lr=1e-2, chunk=32, wd=0.1, freeze_enc=True, smf_k=2, bg_bytes=TEXT[:128])
    L.read(TEXT[128:256])                                          # 先跑几块让部分模块积累 AdamW 动量
    b_before = m.layers[1].core.lora_b.clone()
    a_before = m.layers[1].core.lora_a.clone()
    L.read(TEXT[256:288])                                          # 再跑单块：最多 smf_k=2 个模块允许变动
    changed_b = (m.layers[1].core.lora_b != b_before).any(dim=(1, 2)).sum().item()
    changed_a = (m.layers[1].core.lora_a != a_before).any(dim=(1, 2)).sum().item()
    assert 1 <= changed_b <= 2 and 1 <= changed_a <= 2


def test_sleep_respects_freeze_enc_and_smf():
    """sleep(rewire=False)：freeze_enc=True 时编码层与 win 严格不动；smf_k>0 时只巩固白天 SMF 选中的模块。"""
    import dspn_text as D
    torch.manual_seed(0)
    m = D.build('dspn', 'S', net='wave', N=8, k=2, G=2, D=8, pd=8, pm=4, phidden=16, init='spread')
    with torch.no_grad():
        m.readout.weight.normal_(0, 0.2)
        m.layers[0].W2.normal_(0, 0.2)
        m.layers[1].core.W2.normal_(0, 0.2)
    L = OnlineLearner(m, lr=1e-2, chunk=32, freeze_enc=True, smf_k=2, bg_bytes=TEXT[:128])
    L.read(TEXT[128:256])
    enc_w1 = m.layers[0].W1.clone()
    win_w = m.win.weight.clone()
    pool_w2 = m.layers[1].core.W2.clone()
    r = L.sleep(TEXT[:128], TEXT[128:256], steps=3, old_tol=10.0, rewire=False, smf_sleep=True)
    assert r['committed']
    assert torch.equal(m.layers[0].W1, enc_w1)
    assert torch.equal(m.win.weight, win_w)
    changed_mods = (m.layers[1].core.W2 != pool_w2).any(dim=(1, 2)).sum().item()
    assert 1 <= changed_mods <= 2


def test_pool_mlp_online_mode():
    """params='pool_mlp'：SMF 选中模块直接更新完整 MLP（W1/b1/W2），未选中模块逐位不变。"""
    import dspn_text as D
    torch.manual_seed(0)
    m = D.build('dspn', 'S', net='wave', N=8, k=2, G=2, D=8, pd=8, pm=4, phidden=16, init='spread')
    with torch.no_grad():
        m.readout.weight.normal_(0, 0.2)
        m.layers[0].W2.normal_(0, 0.2)
        m.layers[1].core.W2.normal_(0, 0.2)
    w1_0, w2_0 = m.layers[1].core.W1.clone(), m.layers[1].core.W2.clone()
    L = OnlineLearner(m, lr=1e-2, chunk=32, params='pool_mlp', freeze_enc=True, smf_k=2, bg_bytes=TEXT[:128])
    L.read(TEXT[128:160])
    ch1 = (m.layers[1].core.W1 != w1_0).any(dim=(1, 2)).sum().item()
    ch2 = (m.layers[1].core.W2 != w2_0).any(dim=(1, 2)).sum().item()
    assert 1 <= ch1 <= 2 and 1 <= ch2 <= 2
