"""字节级自上而下（无词级）：顶层上一拍预测第 td_target 层本拍消息；整段、逐拍、任意切段一致（含中枢、会话边界、工作空间）。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from dspn_text import DSPNLayered, SEP

CFG = dict(layers=3, rows=1, cols=2, d=8, m=4, hidden=8, q=4, emb=8, hub_after=2, hub_dk=8, hub_out=8,
           hub_taus=[10.0, 1e6], td_layers=1, td_target=1)
TEXT = b'Tom has a dog. The dog is Bo.\n' + bytes([SEP]) + b'\nWho is Bo? A dog!'


def model(**kw):
    torch.manual_seed(0)
    m = DSPNLayered(**CFG, **kw)
    with torch.no_grad():
        m.readout.weight.normal_(0, 0.3)
        for l in [*m.layers, *m.tdlayers]:
            l.W2.normal_(0, 0.3)
    return m


def run(m, x, cuts):
    st, outs, a = None, [], 0
    for c in cuts + [x.shape[1]]:
        lg, st = m(x[:, a:c], st); outs.append(lg); a = c
    return torch.cat(outs, 1)


def test_stream_equals_parallel():
    for kw in ({}, dict(ws_slots=2, ws_dim=4, ws_block=8)):
        m = model(**kw)
        x = torch.tensor([list(TEXT), list(TEXT[::-1])])
        full = run(m, x, [])
        assert torch.allclose(full, run(m, x, list(range(1, x.shape[1]))), atol=1e-4)
        assert torch.allclose(full, run(m, x, [7, 19, 33]), atol=1e-4)


def test_prediction_trains():
    m = model()
    lg, _ = m(torch.tensor([list(TEXT)]))
    (lg.square().mean() + m.aux).backward()
    assert m.td_pred.weight.grad.abs().sum() > 0
