import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from dspn_text import DSPNLayered
from online import OnlineLearner, bpc

TEXT = (b"the cat sat on the mat and the dog sat on the log. " * 12)[:512]


def tiny(seed=0):
    torch.manual_seed(seed)
    m = DSPNLayered(layers=2, rows=1, cols=2, d=8, m=4, hidden=8, q=4, emb=8)
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
