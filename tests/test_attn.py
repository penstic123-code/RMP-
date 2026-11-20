"""滑窗注意力：逐段 / 逐拍与整段一致（含窗口溢出、分块）；窗口外的位置看不到；会话边界之后看不到之前。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from dspn_text import DSPNLayered, WindowAttn, SEP


def model(**kw):
    torch.manual_seed(0)
    cfg = dict(layers=4, M=2, d=8, m=4, hidden=8, q=4, emb=8, hub_after=1, hub_dk=8, hub_out=4,
               attn_layer=2, attn_window=6, attn_heads=2, attn_hd=4, attn_out=4)
    m = DSPNLayered(**{**cfg, **kw})
    with torch.no_grad():
        m.readout.weight.normal_(0, 0.3); m.attn.o.weight.normal_(0, 0.5)
        for l in m.layers:
            l.W2.normal_(0, 0.3)
    return m


@torch.no_grad()   # 训练模式下带梯度的前向会更新分块阈值 θ，比较一致性要冻结它
def check(m, T=22):
    x = torch.randint(0, 256, (2, T))
    full, _ = m(x)
    for cuts in (((0, 7), (7, 8), (8, 22)), tuple((t, t + 1) for t in range(T))):
        st, outs = None, []
        for a, b in cuts:
            lg, st = m(x[:, a:b], st); outs.append(lg)
        assert torch.allclose(full, torch.cat(outs, 1), atol=1e-4)


def test_stream_equals_parallel():
    check(model())


def test_stream_equals_parallel_chunked():
    check(model(chunk=2, hub_after=2, attn_layer=2))


def test_window_and_session():
    torch.manual_seed(0)
    a = WindowAttn(6, 1, out=3, heads=1, hd=4, window=3)
    with torch.no_grad():
        a.o.weight.normal_()
    x = torch.randn(1, 10, 6)
    st = a.init_state(1, 'cpu')
    no = torch.zeros(1, 10, dtype=torch.bool)
    y, _ = a(x, st, no)
    x2 = x.clone(); x2[:, 0] = torch.randn(1, 6) * 3                        # 位置 0 只在前 3 个查询的窗口里
    y2, _ = a(x2, st, no)
    assert (y - y2)[:, 3:].abs().max() < 1e-6 and (y - y2)[:, 1:3].abs().max() > 1e-4
    r = no.clone(); r[0, 5] = True                          # 位置 5 开新会话：之后看不到 0–4
    x3 = x.clone(); x3[:, 4] = torch.randn(1, 6) * 3
    y3, _ = a(x, st, r); y4, _ = a(x3, st, r)
    assert (y3 - y4)[:, 5:].abs().max() < 1e-6
