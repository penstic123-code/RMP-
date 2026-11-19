import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

from hub import Hub
from dspn_text import DSPNLayered

N_IN, M_OUT, B, T = 12, 4, 2, 23


def make():
    torch.manual_seed(0)
    h = Hub(N_IN, M_OUT, heads=2, dk=8)
    # 默认 w_alpha、w_beta 为零，测试里给非零值，让门控真正随输入变化
    with torch.no_grad():
        h.w_alpha.normal_(0, 0.5); h.w_beta.normal_(0, 0.5); h.b_alpha.fill_(2.0)
    return h


def run(h, x, cuts):
    st, outs = h.init_state(x.shape[0], x.device), []
    for a, b in zip([0] + cuts, cuts + [x.shape[1]]):
        o, st, _ = h(x[:, a:b], st); outs.append(o)
    return torch.cat(outs, 1)


def test_segments_equal_whole():
    h = make(); x = torch.randn(B, T, N_IN)
    full = run(h, x, [])
    assert full.shape == (B, T, M_OUT, 4)
    assert (full - run(h, x, [3, 4, 11, 17])).abs().max() < 1e-4
    assert (full - run(h, x, list(range(1, T)))).abs().max() < 1e-4


def test_causal():
    h = make(); x = torch.randn(B, T, N_IN)
    x2 = x.clone(); x2[:, 10:] = torch.randn(B, T - 10, N_IN)
    d = (run(h, x, []) - run(h, x2, [])).abs()
    assert d[:, :10].max() < 1e-6 and d[:, 10:].max() > 1e-4


def test_dspn_layered_backward():
    torch.manual_seed(0)
    m = DSPNLayered(layers=2, M=2, d=8, m=4, hidden=8, q=4, emb=8, hub_after=1)
    logits, _ = m(torch.randint(0, 256, (2, 20)))
    logits.square().mean().backward()
    for n in ('Wq', 'Wk', 'Wv', 'Wo'):
        assert getattr(m.hub, n).grad is not None
