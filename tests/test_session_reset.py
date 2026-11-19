"""会话边界字节 SEP：之后的输出 ≈ 从 SEP 起全新读入（只剩扫描钳位留下的 1% 旧状态）；有持久头时差异明显更大。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from dspn_text import DSPNLayered, SEP


def model(taus):
    torch.manual_seed(0)
    m = DSPNLayered(layers=2, rows=1, cols=2, d=8, m=4, hidden=8, q=4, emb=8, hub_after=1, hub_dk=8, hub_out=8, hub_taus=taus)
    with torch.no_grad():
        m.readout.weight.normal_(0, 0.3)
        for l in m.layers:
            l.W2.normal_(0, 0.3)
    return m


def gap(m):
    torch.manual_seed(1)
    a, b = torch.randint(0, 256, (1, 30)), torch.randint(0, 256, (1, 20))
    sep = torch.tensor([[SEP]])
    with_past = m(torch.cat([a, sep, b], 1))[0][:, -20:]
    fresh = m(torch.cat([sep, b], 1))[0][:, -20:]
    return (with_past - fresh).abs().max().item()


def test_reset_clears_short_state():
    assert gap(model([10.0, 10.0])) < 0.05


def test_persistent_head_carries_over():
    assert gap(model([10.0, 1e6])) > 5 * gap(model([10.0, 10.0]))
