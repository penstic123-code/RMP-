"""单元测试：A3 苦涩教训纯净版 LatentSearchWaveNet 并行/流式等价、梯度贯通与 SMF 严格掩码。"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rmp_search import LatentSearchWaveNet, SearchSMFOnlineLearner


def _tiny(max_think=3):
    torch.manual_seed(0)
    return LatentSearchWaveNet(
        vocab=256, D=32, M=2, d=16, m=8, hidden=32, q=16,
        N=12, k=2, G=3, pd=24, pm=8, phidden=48, qr=16,
        sep_id=30, max_think=max_think,
    )


def test_parallel_equals_streaming():
    m = _tiny(max_think=3)
    m.eval()
    ids = torch.randint(0, 256, (2, 13))
    ids[0, 6] = 30  # sep_id 会话边界
    with torch.no_grad():
        lp_par, _, info = m(ids, return_ticks=True)
        st = m.init_state(2)
        lps = []
        for t in range(ids.shape[1]):
            lp_t, st = m(ids[:, t : t + 1], state=st)
            lps.append(lp_t)
        lp_seq = torch.cat(lps, dim=1)
    assert torch.allclose(lp_par, lp_seq, atol=1e-4), (lp_par - lp_seq).abs().max().item()
    assert info["exp_steps"].shape == (2, 13)


def test_gradients_and_smf_mask():
    m = _tiny(max_think=3)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    ids = torch.randint(0, 256, (2, 12))
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        loss, metrics, _ = m.compute_loss(ids[:, :-1], ids[:, 1:], ponder_w=0.01)
        loss.backward()
        if step == 0:
            opt.step()
    assert torch.isfinite(loss) and metrics["bpc"] > 0
    for name in ("halt_gate.1.weight", "think_proj.weight", "layers.1.core.W1", "layers.1.core.W2"):
        p = dict(m.named_parameters())[name]
        assert p.grad is not None and p.grad.abs().sum() > 0, name

    learner = SearchSMFOnlineLearner(m, lr=1e-2, smf_k=4)
    learner.calibrate_background(bytes(range(128)) * 2, chunk=64)
    enc_before = m.layers[0].W1.data.clone()
    pool_before = m.layers[1].core.W1.data.clone()
    _, _, keep_vec = learner.step_chunk(ids[:1])
    assert int(keep_vec.sum().item()) == 4
    assert torch.equal(m.layers[0].W1.data, enc_before)
    unselected = ~keep_vec.bool()
    assert torch.equal(m.layers[1].core.W1.data[unselected], pool_before[unselected])
    assert float(learner.opt.state[m.layers[1].core.W1]["exp_avg"][unselected].abs().max().item()) == 0.0
