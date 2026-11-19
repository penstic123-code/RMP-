"""GRU 在线学习：只改适配器与读出；睡眠回滚时模型与适配器都逐位恢复。"""
import sys, pathlib
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from dspn_text import GRULM
from online import OnlineLearner

TEXT = b'Tom has a red ball. Tom likes the red ball. ' * 20


def test_gru_read_and_rollback():
    torch.manual_seed(0)
    m = GRULM(16, emb=8)
    frozen = {k: v.clone() for k, v in m.state_dict().items() if not k.startswith('readout')}
    L = OnlineLearner(m, lr=1e-2, chunk=32, pool_bytes=4096)
    L.read(TEXT)
    assert all(torch.equal(m.state_dict()[k], v) for k, v in frozen.items())
    assert L.ad_b.abs().sum() > 0
    before = {k: v.clone() for k, v in m.state_dict().items()}
    ad = [p.detach().clone() for p in L.online]
    r = L.sleep(TEXT[:512], TEXT[-512:], steps=3, old_tol=-1.0)       # 容差为负，必然回滚
    assert not r['committed']
    assert all(torch.equal(m.state_dict()[k], v) for k, v in before.items())
    assert all(torch.equal(p, b) for p, b in zip(L.online, ad))
