"""窗口级广播：逐段 / 逐拍与整段一致（跨窗口边界）；广播只影响下一窗口；会话边界清空广播。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from dspn_text import DSPNLayered, SEP

CFG = dict(layers=3, M=2, d=8, m=4, hidden=8, q=4, emb=8, hub_after=2, hub_dk=8, hub_out=4, bc_window=8)


def model(**kw):
    torch.manual_seed(0)
    m = DSPNLayered(**CFG, **kw)
    with torch.no_grad():
        m.readout.weight.normal_(0, 0.3); m.bc_in.normal_(0, 0.5)
        for l in m.layers:
            l.W2.normal_(0, 0.3)
    return m


def test_segments_equal_whole():
    m = model(); x = torch.randint(0, 256, (2, 30))
    full, _ = m(x)
    st, outs = None, []
    for a, b in ((0, 3), (3, 11), (11, 12), (12, 30)):
        lg, st = m(x[:, a:b], st); outs.append(lg)
    assert torch.allclose(full, torch.cat(outs, 1), atol=1e-4)


def test_broadcast_reaches_next_window_only():
    m = model(); x = torch.randint(0, 256, (1, 24))
    x2 = x.clone(); x2[:, 2] = (x2[:, 2] + 1) % 256
    a, b = m(x)[0], m(x2)[0]
    with torch.no_grad():
        m.bc_in.zero_()
    c, d = m(x)[0], m(x2)[0]
    gain = (a - b).abs()[:, 8:].max() - (c - d).abs()[:, 8:].max()   # 有广播时窗口外差异更大
    assert gain.abs() > 1e-5


def test_sep_clears_broadcast():
    m = model(); torch.manual_seed(1)
    a, b = torch.randint(0, 255, (1, 16)), torch.randint(0, 255, (1, 8))
    sep = torch.full((1, 8), SEP)
    with_past = m(torch.cat([a, sep, b], 1))[0][:, -8:]
    fresh = m(torch.cat([torch.randint(0, 255, (1, 16)), sep, b], 1))[0][:, -8:]
    assert (with_past - fresh).abs().max() < 1e-3
