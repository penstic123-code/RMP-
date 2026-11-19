"""分块 delta 规则必须与逐拍版逐位一致（plan.md 阶段 0：最大差 < 1e-4）。"""
import sys, pathlib
import torch, torch.nn.functional as F
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from dspn_text import delta_chunk


def stepwise(q, k, v, beta, g, S):
    rs = []
    for t in range(k.shape[1]):
        S = g[:, t].exp()[..., None, None] * S
        S = S + beta[:, t, :, None, None] * torch.einsum('bmi,bmj->bmij', v[:, t] - torch.einsum('bmij,bmj->bmi', S, k[:, t]), k[:, t])
        rs.append(torch.einsum('bmij,bmj->bmi', S, q[:, t]))
    return torch.stack(rs, 1), S


def test_matches_stepwise():
    torch.manual_seed(0)
    b, T, M, m = 3, 150, 4, 8                      # 跨 3 块且不是块长 64 的倍数，测块间传递与补齐
    q, v = torch.randn(b, T, M, m), torch.randn(b, T, M, m)
    k = F.normalize(torch.randn(b, T, M, m), dim=-1)
    beta, g = torch.rand(b, T, M), F.logsigmoid(torch.randn(b, T, M) - 2)
    S0 = torch.randn(b, M, m, m)
    r1, S1 = delta_chunk(q, k, v, beta, g, S0)
    r2, S2 = stepwise(q, k, v, beta, g, S0)
    assert (r1 - r2).abs().max() < 1e-4 and (S1 - S2).abs().max() < 1e-4
