"""预测学习与局部学习：预测头得到梯度；local=True 时上层的损失不回传到下层参数。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from dspn_text import DSPNLayered

CFG = dict(layers=3, rows=1, cols=2, d=8, m=4, hidden=8, q=4, emb=8, hub_after=2, hub_dk=8, ws_slots=2, ws_dim=4, ws_block=8)


def test_prediction_heads_train():
    torch.manual_seed(0)
    m = DSPNLayered(**CFG, pred_self=1.0, pred_nb=1.0, pred_down=1.0)
    with torch.no_grad():
        for l in m.layers:
            l.W2.normal_(0, 0.3)
    m(torch.randint(0, 256, (2, 20)))
    assert m.aux > 0 and {k for k, _ in m.pred_err} == {'self', 'nb', 'down'}
    m.aux.backward()
    for k in ('self', 'nb', 'down'):
        assert m.pred[k][1].W.grad.abs().sum() > 0


def test_local_blocks_gradient_between_layers():
    torch.manual_seed(0)
    for local, expect in ((False, True), (True, False)):
        m = DSPNLayered(**CFG, local=local)
        with torch.no_grad():
            for l in m.layers:
                l.W2.normal_(0, 0.3)
        out = {}
        m.layers[2].register_forward_hook(lambda mod, i, o: out.update(h=o[0]))   # 钩子返回 None，不替换输出
        m(torch.randint(0, 256, (2, 20)))
        out['h'].square().sum().backward()
        g = m.layers[0].B.grad
        assert (g is not None and g.abs().sum() > 0) == expect


def test_surprise_gate_stream_equals_parallel():
    torch.manual_seed(0)
    m = DSPNLayered(**CFG, pred_self=0.1, surprise=True)
    with torch.no_grad():
        m.readout.weight.normal_(0, 0.3); m.sg_a.fill_(1.0)
        for l in m.layers:
            l.W2.normal_(0, 0.3); l.w_beta.normal_(0, 0.3)
        for hd in m.pred['self']:
            hd.W.normal_(0, 0.3)
    x = torch.randint(0, 256, (2, 24))
    full, _ = m(x)
    st, outs = None, []
    for t in range(24):
        lg, st = m(x[:, t:t + 1], st); outs.append(lg)
    assert torch.allclose(full, torch.cat(outs, 1), atol=1e-4)
    m.zero_grad(); full.square().mean().backward()
    assert m.sg_a.grad.abs().sum() > 0
