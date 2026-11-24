"""单元测试：A2 v2 双流符号绑定 RMP 并行/流式数值等价、正交旋转不变性、梯度贯通与 SMF 严格掩码。"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rmp_a2 import A2SMFOnlineLearner, LatentThinkHippoRMP


def _small_model(use_hippo=True, max_think=3):
    torch.manual_seed(0)
    return LatentThinkHippoRMP(
        vocab=64, D=32, M=2, d=16, m=8, hidden=32, q=16,
        N=12, k=2, G=3, pd=24, pm=8, phidden=48, qr=16,
        pad_id=0, sep_id=1, mem_sep_id=2, max_think=max_think,
        use_hippo=use_hippo, hippo_dim=32, n_hippo_heads=4, sym_dim=32,
    )


def test_parallel_equals_streaming_no_hippo():
    m = _small_model(use_hippo=False, max_think=3)
    m.eval()
    ids = torch.randint(3, 64, (2, 11))
    ids[0, 5] = 1
    with torch.no_grad():
        lp_par, _ = m(ids)
        st = m.init_state(2)
        lps = []
        for t in range(ids.shape[1]):
            lp_t, st = m(ids[:, t : t + 1], state=st)
            lps.append(lp_t)
        lp_seq = torch.cat(lps, dim=1)
    assert torch.allclose(lp_par, lp_seq, atol=1e-4), (lp_par - lp_seq).abs().max().item()


def test_parallel_equals_streaming_with_hippo_and_prob_norm():
    m = _small_model(use_hippo=True, max_think=3)
    m.eval()
    mem_ids = torch.randint(3, 64, (2, 14))
    mem_ids[:, 6] = 2  # mem_sep_id 记忆条目边界
    mem_ids[1, 10:] = 0  # pad
    ids = torch.randint(3, 64, (2, 9))
    with torch.no_grad():
        hmem = m.encode_memory(mem_ids)
        lp_par, _, info = m(ids, hippo_mem=hmem, return_ticks=True)
        prob_sum = lp_par.exp().sum(dim=-1)
        assert torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-4)
        st = m.init_state(2)
        lps = []
        for t in range(ids.shape[1]):
            lp_t, st = m(ids[:, t : t + 1], state=st, hippo_mem=hmem)
            lps.append(lp_t)
        lp_seq = torch.cat(lps, dim=1)
    assert torch.allclose(lp_par, lp_seq, atol=1e-4), (lp_par - lp_seq).abs().max().item()
    assert info["exp_steps"].shape == (2, 9)


def test_symbol_codebook_orthogonal_invariance():
    """验证对符号码本施加任意正交旋转 Q，前向输出严格逐位不变（保证零样本符号泛化）。"""
    m = _small_model(use_hippo=True, max_think=3)
    m.eval()
    mem_ids = torch.randint(3, 64, (2, 12))
    ids = torch.randint(3, 64, (2, 8))
    with torch.no_grad():
        lp1, _ = m(ids, hippo_mem=m.encode_memory(mem_ids))
        # 对 sym_codebook 施加随机正交矩阵旋转 Q
        Q, _ = torch.linalg.qr(torch.randn(m.sym_dim, m.sym_dim))
        m.sym_codebook.copy_(m.sym_codebook @ Q)
        lp2, _ = m(ids, hippo_mem=m.encode_memory(mem_ids))
    assert torch.allclose(lp1, lp2, atol=1e-4), (lp1 - lp2).abs().max().item()


def test_end_to_end_gradients():
    m = _small_model(use_hippo=True, max_think=3)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    mem_ids = torch.randint(3, 64, (2, 10))
    ids = torch.randint(3, 64, (2, 8))
    tgt = torch.randint(3, 64, (2, 8))
    tgt[:, :3] = -100
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        hmem = m.encode_memory(mem_ids)
        loss, _, _ = m.compute_loss(ids, tgt, hippo_mem=hmem, ponder_w=0.02)
        loss.backward()
        if step == 0:
            opt.step()
    assert torch.isfinite(loss)
    for name in (
        "emit_gate.1.weight", "hippo_q.weight", "hippo_k.weight",
        "sym_gates.weight", "sym_read_head.weight", "sym_update_gate.weight",
        "copy_gate.1.weight", "layers.1.core.W1", "layers.1.core.W2",
    ):
        p = dict(m.named_parameters())[name]
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name


def test_smf_strict_mask_preserves_unselected_and_encoder():
    m = _small_model(use_hippo=True, max_think=2)
    learner = A2SMFOnlineLearner(m, lr=1e-2, smf_k=4)
    mem_ids = torch.randint(3, 64, (2, 8))
    ids = torch.randint(3, 64, (2, 10))
    tgt = torch.randint(3, 64, (2, 10))
    with torch.no_grad():
        hmem = m.encode_memory(mem_ids)
    learner.calibrate_background([(ids, hmem)])

    enc_w_before = m.layers[0].W1.data.clone()
    pool_w1_before = m.layers[1].core.W1.data.clone()
    sym_g_before = m.sym_gates.weight.data.clone()

    _, keep_vec = learner.step(ids, tgt, hippo_mem=hmem)
    assert int(keep_vec.sum().item()) == 4
    assert torch.equal(m.layers[0].W1.data, enc_w_before)
    assert torch.equal(m.sym_gates.weight.data, sym_g_before)
    unselected = ~keep_vec.bool()
    assert torch.equal(m.layers[1].core.W1.data[unselected], pool_w1_before[unselected])
    exp_avg = learner.opt.state[m.layers[1].core.W1]["exp_avg"]
    assert float(exp_avg[unselected].abs().max().item()) == 0.0
