"""动态分块（v0.18）：逐段 / 逐拍与整段一致（无溢出时）；局部头进 aux、不回传主干；会话边界与最大块长强制开块；
整段高惊讶（随机串）不逐拍开块；θ 训练时追块率、推理冻结。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from dspn_text import DSPNLayered, SEP

CFG = dict(layers=4, M=2, d=8, m=4, hidden=8, q=4, emb=8, hub_after=2, hub_dk=8, hub_out=4, chunk=2)


def model(**kw):
    torch.manual_seed(0)
    m = DSPNLayered(**{**CFG, **kw})
    with torch.no_grad():
        m.readout.weight.normal_(0, 0.3); m.head.weight.normal_(0, 1.0)
        for l in m.layers:
            l.W2.normal_(0, 0.3)
    return m


def run_cuts(m, x, cuts):
    st, outs = None, []
    for a, b in cuts:
        lg, st = m(x[:, a:b], st); outs.append(lg)
    return torch.cat(outs, 1)


def test_stream_equals_parallel():
    m = model(); x = torch.randint(0, 256, (2, 20))
    with torch.no_grad():
        full, _ = m(x)
        assert m.stats['over'] == 0
        assert torch.allclose(full, run_cuts(m, x, [(t, t + 1) for t in range(20)]), atol=1e-4)
        assert torch.allclose(full, run_cuts(m, x, [(0, 7), (7, 8), (8, 20)]), atol=1e-4)


def test_thin_io_layers_stream_equals_parallel():
    m = model(d_io=6, hidden_io=6); x = torch.randint(0, 256, (2, 18))
    assert m.layers[0].d == 6 and m.layers[1].d == 8 and m.layers[-1].d == 6
    with torch.no_grad():
        full, _ = m(x)
        assert torch.allclose(full, run_cuts(m, x, [(0, 5), (5, 6), (6, 18)]), atol=1e-4)


def test_head_loss_detached_from_trunk():
    m = model(); x = torch.randint(0, 256, (2, 20))
    m(x)
    assert m.aux > 0
    m.aux.backward()
    assert m.head.weight.grad.abs().sum() > 0 and m.layers[0].B.grad is None


def boundaries(m, x):
    b, T = x.shape
    lp = torch.log_softmax(torch.randn(b, T, 256), -1)
    z = torch.zeros(b)
    return m._boundaries(x, lp, lp[:, 0], z, z, x == SEP)


def test_forced_boundaries():
    m = model(lmax=3)                                      # L_max = 3·N = 6
    m.theta.fill_(1e9)                                     # 跃升永不过阈值：只剩强制开块
    x = torch.randint(0, 255, (1, 20)); x[0, 9] = SEP
    bnd, _, n = boundaries(m, x)
    assert bnd[0].nonzero().flatten().tolist() == [0, 6, 9, 15]
    assert n.item() == 5


def test_constant_high_surprise_does_not_chunk_every_byte():
    m = model(); m.theta.fill_(0.5)
    b, T = 1, 40
    x = torch.randint(0, 255, (b, T))
    lp = torch.full((b, T, 256), -5.55)                    # 每个字节惊讶都一样高（随机串）
    bnd, _, _ = m._boundaries(x, lp, lp[:, 0], torch.full((b,), 5.55), torch.ones(b), x == SEP)
    assert bnd.float().mean() <= 1 / m.lmax + 1e-6        # 只有最大块长强制的边界


def test_theta_tracks_rate_in_training_only():
    m = model(chunk=3); x = torch.randint(0, 256, (2, 61))   # 初始 θ=0 时跃升 > 0 约占一半，N=2 恰好不动
    m(x)
    t1 = m.theta.item()
    assert t1 != 0
    with torch.no_grad():
        m(x)
    assert m.theta.item() == t1
