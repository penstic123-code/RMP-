"""e-prop：慢通道 c 是对角递推，资格迹跨块延续时 B、G、b_a 的梯度等于整段沿时间反传的精确梯度（损失对 c 线性时）。"""
import os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import dspn_text as D
from online import OnlineLearner


def test_eprop_equals_bptt_across_chunks():
    torch.manual_seed(0)
    layer = D.Layer(2, 5, 6, 3, 8, 4).double()
    with torch.no_grad():
        layer.G.normal_(0, 0.5)
    for p in layer.parameters():
        p.data = p.data.double()
    u, w = torch.randn(2, 20, 2, 5).double(), torch.randn(2, 20, 2, 6).double()
    # 精确：整段反传
    params = (layer.B, layer.G, layer.b_a)
    for p in params:
        p.requires_grad_(True)
    la, x = layer.gates(u)
    c, _ = D.scan(la, x, torch.zeros(2, 2, 6).double())
    exact = torch.autograd.grad((c * w).sum(), params)
    # e-prop：两块，状态与资格迹跨块延续
    for p in params:
        p.requires_grad_(False); p.grad = None
    fake = types.SimpleNamespace(model=types.SimpleNamespace(layers=[layer]), traces=None)
    st, tot = layer.init_state(2, 'cpu'), [0, 0, 0]
    st = tuple(s.double() for s in st)
    for a, b in ((0, 7), (7, 20)):
        layer.rec = []
        _, _, st = layer(u[:, a:b], st)
        st = tuple(s.detach() for s in st)
        rec = layer.rec[0]; layer.rec = None
        (rec[3] * w[:, a:b]).sum().backward()
        OnlineLearner._eprop_grads(fake, [rec])
        tot = [t + p.grad for t, p in zip(tot, params)]
    for e, g in zip(exact, tot):
        assert torch.allclose(e, g, atol=1e-8)


def test_learner_eprop_reads_and_updates():
    torch.manual_seed(0)
    m = D.DSPNLayered(layers=3, M=2, d=8, m=4, hidden=8, q=4, emb=8, hub_after=2, hub_dk=8, bc_window=16)
    with torch.no_grad():
        m.readout.weight.normal_(0, 0.3)
        for l in m.layers:
            l.W2.normal_(0, 0.3)
    L = OnlineLearner(m, lr=1e-2, chunk=16, params='eprop')
    B0 = m.layers[0].B.clone()
    L.read(b'hello there, my name is Kobi. ' * 3)
    assert not torch.equal(B0, m.layers[0].B) and L.traces is not None
    B1 = m.layers[0].B.clone()
    L.read(b'nothing new here', gate=0.0)                  # 新奇门关：只推进状态
    assert torch.equal(B1, m.layers[0].B)
