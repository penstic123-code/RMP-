"""跨批次延续状态：连续两段带状态 = 一整段；stream_batch 每行连续前移。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import dspn_text as D


def test_copy_state_carries_across_segments():
    torch.manual_seed(0)
    m = D.DSPNLayered(layers=2, rows=1, cols=2, d=8, m=4, hidden=8, q=4, emb=8, hub_after=1, hub_dk=8, ws_slots=2, ws_dim=4, ws_block=8)
    with torch.no_grad():
        m.readout.weight.normal_(0, 0.3)
    x = torch.randint(0, 256, (2, 32))
    full, _ = m(x)
    st = m.init_state(2)
    _, new = m(x[:, :16], st); D.copy_state_(st, new)
    second, _ = m(x[:, 16:], st)
    assert torch.allclose(full[:, 16:], second, atol=1e-4)


def test_stream_batch_is_contiguous():
    data = np.arange(1000, dtype=np.uint8)
    pos = np.array([0, 500])
    x1, y1 = D.stream_batch(data, pos, 10); x2, _ = D.stream_batch(data, pos, 10)
    assert x2[0, 0].item() == y1[0, -1].item() and pos.tolist() == [20, 520]
