"""TraceCols：资格迹跨块延续时，分块计算的梯度与整段一次计算完全一致（e-prop 对自身参数精确）。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
import tracecols as TC


def _grads(model, x, y, cuts):
    model.zero_grad()
    st = model.init_state(x.shape[0], 'cpu', 'eprop')
    st = _to_double(st)
    for a, b in zip(cuts[:-1], cuts[1:]):
        lg, _, st, _ = model(x[:, a:b], st, 'eprop')
        F.cross_entropy(lg.flatten(0, 1), y[:, a:b].flatten(), reduction='sum').backward()
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


def _to_double(st):
    if isinstance(st, dict):
        return {k: _to_double(v) for k, v in st.items()}
    if isinstance(st, list):
        return [_to_double(v) for v in st]
    return st.double()


def test_chunked_eprop_equals_single_chunk():
    torch.manual_seed(0)
    for target in ('self', 'ext', 'down'):
        m = TC.TraceCols(L=2, C=3, d=8, n=6, H=2, dk=4, h=16, k=3, m=8, hubH=2, hub_dk=4, target=target).double()
        with torch.no_grad():                               # 让慢通道与记忆有长衰减，跨块影响明显
            for lv in m.levels:
                lv.slow.b.uniform_(2, 6)
                lv.mem.r.fill_(-3.)
        x = torch.randint(0, 256, (2, 24))
        y = torch.randint(0, 256, (2, 24))
        g1 = _grads(m, x, y, [0, 24])
        g3 = _grads(m, x, y, [0, 7, 16, 24])
        assert g1.keys() == g3.keys()
        for n in g1:
            assert torch.allclose(g1[n], g3[n], atol=1e-9, rtol=1e-7), (target, n, (g1[n] - g3[n]).abs().max())


def test_homeostasis_keeps_units_alive():
    torch.manual_seed(0)
    m = TC.TraceCols(L=1, C=2, d=8, n=4, H=1, dk=4, h=32, k=2, m=8, hubH=1, hub_dk=4)
    with torch.no_grad():
        m.levels[0].bin[:, :4] += 5.                        # 4 个单元先天占优，不调节会永远赢
    st = m.init_state(4, 'cpu', 'eval')
    seen = torch.zeros(2, 32)
    for i in range(300):
        x = torch.randint(0, 256, (4, 16))
        with torch.no_grad():
            _, _, st, stats = m(x, st, 'eval', rec=True)
        m.homeostasis(stats)
        if i >= 200:
            seen += stats[0]['mask'].sum((0, 1))
    assert (seen > 0).float().mean() > 0.9
