"""对角递推扫描（本地分块实现与 CUDA 上的 fla HGRN 内核）与逐拍循环一致，含补齐、长序列、门接近 0 的情形。"""
import os, sys, importlib.util
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest, torch
import dspn_text as D


def loop(la, x, c0):
    c, out = c0, []
    for t in range(la.shape[1]):
        c = la[:, t].exp() * c + x[:, t]
        out.append(c)
    return torch.stack(out, 1), c


def test_scan_matches_loop():
    torch.manual_seed(0)
    for T in (1, 5, 16, 37, 300):
        la = torch.log(torch.rand(2, T, 3, 4, dtype=torch.float64)).clamp(min=-30)   # 门可低到 e^-30（会话边界）
        x, c0 = torch.randn(2, T, 3, 4, dtype=torch.float64), torch.randn(2, 3, 4, dtype=torch.float64)
        c, cT = D.scan(la, x, c0)
        rc, rT = loop(la, x, c0)
        assert torch.allclose(c, rc, atol=1e-9) and torch.allclose(cT, rT, atol=1e-9)


@pytest.mark.skipif(not torch.cuda.is_available() or importlib.util.find_spec('fla') is None, reason='需要 CUDA 与 fla')
def test_fla_scan_matches_loop():
    torch.manual_seed(0)
    la = torch.log(torch.rand(2, 300, 3, 4, device='cuda'))
    x, c0 = torch.randn(2, 300, 3, 4, device='cuda'), torch.randn(2, 3, 4, device='cuda')
    c, cT = D.scan(la, x, c0)
    rc, rT = loop(la, x, c0)
    assert (c - rc).abs().max() < 1e-3 and (cT - rT).abs().max() < 1e-3
