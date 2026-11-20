"""动态分块：逐段 / 逐拍与整段一致（无溢出时）；路由得到梯度；压缩率损失进 aux；会话边界强制开块。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from dspn_text import DSPNLayered, SEP

CFG = dict(layers=4, M=2, d=8, m=4, hidden=8, q=4, emb=8, hub_after=2, hub_dk=8, hub_out=4, bc_window=8, chunk=2)


def model(**kw):
    torch.manual_seed(0)
    m = DSPNLayered(**{**CFG, **kw})
    with torch.no_grad():
        m.readout.weight.normal_(0, 0.3); m.bc_in.normal_(0, 0.5)
        for l in m.layers:
            l.W2.normal_(0, 0.3)
    return m


def test_stream_equals_parallel():
    m = model(); x = torch.randint(0, 256, (2, 20))
    full, _ = m(x)
    st, outs = None, []
    for t in range(20):
        lg, st = m(x[:, t:t + 1], st); outs.append(lg)
    assert torch.allclose(full, torch.cat(outs, 1), atol=1e-4)


def test_router_gets_gradient_and_ratio_loss():
    m = model(); x = torch.randint(0, 256, (2, 20))
    lg, _ = m(x)
    assert m.aux > 0 and 0 <= m.stats['rate'] <= 1
    (lg.square().mean() + m.aux).backward()
    assert m.Wr.grad.abs().sum() > 0 and m.layers[1].B.grad.abs().sum() > 0


def test_sep_forces_boundary():
    m = model(); x = torch.randint(0, 255, (1, 12)); x[0, 9] = SEP   # 落在最后一段（广播窗口 8），stats 记的是最后一段
    m(x)
    assert m.stats['rate'] > 0


def test_thin_io_layers_stream_equals_parallel():
    m = model(d_io=6, hidden_io=6); x = torch.randint(0, 256, (2, 18))
    assert m.layers[0].d == 6 and m.layers[1].d == 8 and m.layers[-1].d == 6
    full, _ = m(x)
    st, outs = None, []
    for a, b in ((0, 5), (5, 6), (6, 18)):
        lg, st = m(x[:, a:b], st); outs.append(lg)
    assert torch.allclose(full, torch.cat(outs, 1), atol=1e-4)
