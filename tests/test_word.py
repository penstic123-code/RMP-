"""词级层：整段、逐拍、任意切段一致（含中枢、会话边界、工作空间）；词级输出在两个边界之间保持不变。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from dspn_text import DSPNLayered, SEP

CFG = dict(layers=2, rows=1, cols=2, d=8, m=4, hidden=8, q=4, emb=8, hub_after=1, hub_dk=8, hub_out=8,
           hub_taus=[10.0, 1e6], word_layers=2, word_d=12, word_hidden=16, word_m=4, word_q=4)


def model(**kw):
    torch.manual_seed(0)
    m = DSPNLayered(**CFG, **kw)
    with torch.no_grad():
        m.readout.weight.normal_(0, 0.3)
        for l in [*m.layers, *m.wlayers, *m.tdlayers]:
            l.W2.normal_(0, 0.3)
    return m


def run(m, x, cuts):
    st, outs, a = None, [], 0
    for c in cuts + [x.shape[1]]:
        lg, st = m(x[:, a:c], st); outs.append(lg); a = c
    return torch.cat(outs, 1)


TEXT = b'Tom has a dog. The dog is Bo.\n' + bytes([SEP]) + b'\nWho is Bo? A dog!'


def test_stream_equals_parallel():                 # 对数几率量级 10–17，单精度舍入约 1e-5
    for kw in ({}, dict(ws_slots=2, ws_dim=4, ws_block=8), dict(td_layers=1), dict(td_layers=2, ws_slots=2, ws_dim=4, ws_block=8)):
        m = model(**kw)
        x = torch.tensor([list(TEXT), list(TEXT[::-1])])
        full = run(m, x, [])
        assert torch.allclose(full, run(m, x, list(range(1, x.shape[1]))), atol=1e-4)
        assert torch.allclose(full, run(m, x, [7, 19, 33]), atol=1e-4)


def test_word_output_held_between_boundaries():
    m = model()
    x = torch.tensor([list(b'abcdefgh ijk')])
    _, st = m(x[:, :4])
    _, st2 = m(x[:, 4:8], st)
    assert torch.equal(st[-1], st2[-1])                     # 'efgh' 里没有边界，词级输出不变
    _, st3 = m(x[:, 8:], st2)
    assert not torch.equal(st2[-1], st3[-1])                # ' ' 是边界
