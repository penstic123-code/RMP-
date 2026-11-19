"""全局工作空间：整段、逐拍、任意切段三种读法结果一致（块边界按全局位置对齐）；零初始时等于无工作空间。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from dspn_text import DSPNLayered

CFG = dict(layers=2, rows=1, cols=2, d=8, m=4, hidden=8, q=4, emb=8, hub_after=1, hub_dk=8)


def model(**kw):
    torch.manual_seed(0)
    m = DSPNLayered(**CFG, **kw)
    with torch.no_grad():
        m.readout.weight.normal_(0, 0.3)
        for l in m.layers:
            l.W2.normal_(0, 0.3)
        for k, p in m.named_parameters():          # 零初始的工作空间参数随机化，让各机制真正起作用
            if k in ('ws_in', 'ws_rv.weight', 'ws_pc_w', 'ws_ig.weight'):
                p.normal_(0, 0.3)
    return m


FULL = dict(ws_slots=2, ws_dim=4, ws_block=8, ws_topk=2, ws_read=True, ws_sparse=True, ws_lb=0.01, ws_pc=0.1, ws_ignite=True)


def test_full_stream_equals_parallel_and_trains():
    m = model(**FULL)
    x = torch.randint(0, 256, (2, 21))
    full = run(m, x, [])
    assert torch.allclose(full, run(m, x, list(range(1, 21))), atol=1e-5)
    assert torch.allclose(full, run(m, x, [5, 13]), atol=1e-5)
    lg, st = m(x)
    assert st[-1][2].sum(-1).sum(-1).eq(2).all()            # 每个样本恰好 2 个模块活跃
    (lg.square().mean() + m.aux).backward()
    assert m.ws_sal.weight.grad.abs().sum() > 0 and m.ws_pc_w.grad.abs().sum() > 0


def run(m, x, cuts):
    st, outs, a = None, [], 0
    for c in cuts + [x.shape[1]]:
        lg, st = m(x[:, a:c], st); outs.append(lg); a = c
    return torch.cat(outs, 1)


def test_stream_equals_parallel():
    m = model(ws_slots=2, ws_dim=4, ws_block=8)
    x = torch.randint(0, 256, (2, 21))
    full = run(m, x, [])
    assert torch.allclose(full, run(m, x, list(range(1, 21))), atol=1e-5)
    assert torch.allclose(full, run(m, x, [5, 13]), atol=1e-5)


def test_workspace_matters_after_first_block():
    m = model(ws_slots=2, ws_dim=4, ws_block=8)
    x = torch.randint(0, 256, (1, 20))
    lg, _ = m(x)
    with torch.no_grad():
        m.ws_in.zero_()
    lg0, _ = m(x)
    assert torch.allclose(lg[:, :8], lg0[:, :8])          # 第一块槽位为零
    assert not torch.allclose(lg[:, 8:], lg0[:, 8:])
