"""TSPH：任意切块（含逐字节）流式与整段一致；会话边界清 M_fast 留 M_long；短训练损失下降；aux 有限。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F
from dspn_text import SEP
from tsph import TSPH


def model():
    torch.manual_seed(0)
    m = TSPH(n=2, dm=8, hm=2, emb=8, ws=8, dk=8, dv=8, mem_chunk=16)
    with torch.no_grad():                                  # 零初始的读出与 MLP 输出层打乱，让各通路都影响 logits
        m.readout.weight.normal_(0, 0.3)
        for blk in list(m.A) + list(m.B):
            blk.W2.normal_(0, 0.3)
    return m


def stream(m, x, cuts):
    st, outs, t = None, [], 0
    for c in cuts + [x.shape[1]]:
        lg, st = m(x[:, t:c], st); outs.append(lg); t = c
    return torch.cat(outs, 1), st


def test_streaming_matches_parallel():
    m = model()
    torch.manual_seed(1)
    x = torch.randint(0, 256, (2, 50)); x[0, 20] = SEP
    with torch.no_grad():
        full, st_full = m(x)
        for cuts in ([1, 2, 17, 33, 34], list(range(1, 50)), [16, 32, 48]):
            lg, st = stream(m, x, cuts)
            assert (lg - full).abs().max() < 1e-4
            assert (st['M_fast'] - st_full['M_fast']).abs().max() < 1e-4


def test_sep_clears_fast_keeps_long():
    m = model()
    torch.manual_seed(2)
    with torch.no_grad():
        _, st = m(torch.randint(0, 256, (1, 40)))
        st['M_long'] = torch.randn_like(st['M_long'])
        assert torch.linalg.svdvals(st['M_fast'][0])[1] > 1e-4             # 多次写入：秩 > 1
        _, st2 = m(torch.tensor([[SEP]]), st)
        sv = torch.linalg.svdvals(st2['M_fast'][0])
        assert sv[1] < 1e-6 * max(sv[0].item(), 1e-6) + 1e-9               # 只剩边界字节自己的一次写入
        assert torch.equal(st2['M_long'], st['M_long'])


def test_training_reduces_loss():
    m = model()
    torch.manual_seed(3)
    text = torch.tensor(list(b'the cat sat on the mat. ' * 20))
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    losses = []
    for _ in range(40):
        i = torch.randint(0, len(text) - 65, (8,))
        xy = torch.stack([text[j:j + 65] for j in i])
        lg, _ = m(xy[:, :-1])
        loss = F.cross_entropy(lg.reshape(-1, 256), xy[:, 1:].reshape(-1))
        assert torch.isfinite(torch.as_tensor(m.aux))
        (loss + m.aux).backward(); opt.step(); opt.zero_grad()
        losses.append(loss.item())
    assert sum(losses[-5:]) / 5 < 0.5 * sum(losses[:5]) / 5
