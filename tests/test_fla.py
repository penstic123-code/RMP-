"""融合内核路径（DSPN_FLA）与本地实现一致：整模型前向、梯度。无 CUDA 或未装 fla 时跳过。"""
import os, sys, importlib.util
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest, torch
import dspn_text as D

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or importlib.util.find_spec('fla') is None, reason='需要 CUDA 与 fla')


def test_fla_matches_local():
    torch.manual_seed(0)
    m = D.DSPNLayered(layers=2, rows=1, cols=2, d=32, m=16, hidden=32, q=8, emb=16, hub_after=1, hub_dk=16, beta_max=2).cuda()
    with torch.no_grad():
        m.readout.weight.normal_(0, 0.3)
        for l in m.layers:
            l.W2.normal_(0, 0.3)
    x = torch.randint(0, 256, (2, 256), device='cuda')
    outs = []
    for flag in (False, True):
        D.USE_FLA = flag; m.zero_grad()
        lg, _ = m(x); lg.square().mean().backward()
        outs.append((lg.detach(), m.layers[0].K.grad.clone()))
    D.USE_FLA = False
    assert (outs[0][0] - outs[1][0]).abs().max() < 1e-2 * outs[0][0].abs().max()   # TF32 逐层累积，相对 1% 以内
    assert torch.nn.functional.cosine_similarity(outs[0][1].flatten(), outs[1][1].flatten(), 0) > 0.99
