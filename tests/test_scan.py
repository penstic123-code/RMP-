"""两级并行扫描与逐拍循环一致（含补齐、长序列、a 很小导致块尾累乘远小于 0.01 的情形）。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from dspn_text import scan


def loop(a, b, c0):
    c, out = c0, []
    for t in range(a.shape[1]):
        c = a[:, t].clamp(min=1e-2) * c + b[:, t]
        out.append(c)
    return torch.stack(out, 1), c


def test_scan_matches_loop():
    torch.manual_seed(0)
    for T in (1, 5, 16, 37, 300):
        for lo in (0.0, 0.9):                              # lo=0：a 可低到 0.01，块尾累乘约 1e-32
            a = lo + (1 - lo) * torch.rand(2, T, 3, 4, dtype=torch.float64)
            b, c0 = torch.randn(2, T, 3, 4, dtype=torch.float64), torch.randn(2, 3, 4, dtype=torch.float64)
            c, cT = scan(a, b, c0)
            rc, rT = loop(a, b, c0)
            assert torch.allclose(c, rc, atol=1e-9) and torch.allclose(cT, rT, atol=1e-9)
