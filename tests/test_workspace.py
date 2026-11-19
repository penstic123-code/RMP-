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
            if k == 'ws_in':
                p.normal_(0, 0.3)
    return m


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


def test_write_is_bounded():
    """消息放大 1000 倍，写进槽位的内容不变：跨块环路不会自我放大（E43）。"""
    m = model(ws_slots=2, ws_dim=4, ws_block=8)
    zs = [torch.randn(2, 8, m.M, 4) for _ in m.layers]
    s0 = torch.zeros(2, 2, 4)
    a, _ = m._ws_write(zs, s0)
    b, _ = m._ws_write([z * 1000 for z in zs], s0)
    assert torch.allclose(a, b, atol=1e-4)


def test_session_boundary_clears_workspace():
    """会话边界之后，工作空间不额外泄漏边界前的信息：开 / 关工作空间时，两段不同前文造成的输出差异相同。
    （c 在边界处保留 1%，scan 的已知近似，所以差异本身不为零。）"""
    from dspn_text import SEP
    m = model(ws_slots=2, ws_dim=4, ws_block=8)
    g = torch.Generator().manual_seed(1)
    pre = lambda: torch.randint(0, SEP, (1, 20), generator=g)
    tail = torch.randint(0, SEP, (1, 20), generator=g)
    a = torch.cat([pre(), torch.tensor([[SEP]]), tail], 1); b = torch.cat([pre(), torch.tensor([[SEP]]), tail], 1)
    d_on = (m(a)[0] - m(b)[0])[:, 21:]
    with torch.no_grad():
        m.ws_in.zero_()
    d_off = (m(a)[0] - m(b)[0])[:, 21:]
    assert d_on.abs().max() <= 1.1 * d_off.abs().max() + 1e-5
