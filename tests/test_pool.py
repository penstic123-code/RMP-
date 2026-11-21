"""路由模块池（第 23 章 23.5 的测试）：未选模块状态逐位不变、整段与逐拍一致、k = N 时等于稠密池、地址醒时冻结、会话边界清空全部模块。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import dspn_text as D
from pool import PoolLayer


def make(mode='state', N=6, k=2):
    torch.manual_seed(0)
    p = PoolLayer(N, k, U=5, d=8, m=4, hidden=12, q=3, qr=4, mode=mode).double()
    with torch.no_grad():
        for prm in p.parameters():
            prm.add_(0.1 * torch.randn_like(prm))      # W2 零初始时 MLP 输出恒为 0，测不出东西
    p.addr, p.bias, p.load_ema = p.addr.double(), p.bias.double(), p.load_ema.double()
    return p


def rand_state(p, b):
    c, S = p.init_state(b, 'cpu')
    return torch.randn_like(c).double(), torch.randn_like(S).double()


def test_unselected_state_unchanged():
    p = make(); p.eval()
    st = rand_state(p, 3)
    u = torch.randn(3, 1, 5).double()
    _, sel, _ = p.route(u)
    with torch.no_grad():
        _, _, (c, S) = p(u, st)
    off = sel[:, 0] == 0                                   # (b, N)
    assert off.sum() == 3 * (6 - 2)
    assert torch.equal(c[off], st[0][off]) and torch.equal(S[off], st[1][off])
    assert not torch.allclose(c[~off], st[0][~off])


def test_sequence_equals_stepwise():
    for mode in ('state', 'mlp'):
        p = make(mode); p.eval()
        st = rand_state(p, 2)
        u = torch.randn(2, 10, 5).double()
        with torch.no_grad():
            h, z, (c, S) = p(u, st)
            s, hs = st, []
            for t in range(10):
                ht, _, s = p(u[:, t:t + 1], s)
                hs.append(ht)
        assert torch.allclose(h, torch.cat(hs, 1), atol=1e-5), mode       # 快记忆参考实现内部是 float32
        assert torch.allclose(c, s[0], atol=1e-5) and torch.allclose(S, s[1], atol=1e-5), mode


def test_full_selection_equals_dense_pool():
    p = make(N=4, k=4); p.eval()
    st = rand_state(p, 2)
    u = torch.randn(2, 7, 5).double()
    with torch.no_grad():
        h, _, (c, S) = p(u, st)
        hd, _, (cd, Sd) = p.core(u[:, :, None].expand(2, 7, 4, 5), st)
        _, _, w = p.route(u)
    assert torch.allclose(c, cd) and torch.allclose(S, Sd)
    assert torch.allclose(h[:, :, 0], (w[..., None] * hd).sum(2))


def test_addresses_frozen_unless_training():
    p = make()
    a0 = p.addr.clone()
    u = torch.randn(2, 6, 5).double()
    p.eval(); p(u, rand_state(p, 2))
    p.train()
    with torch.no_grad():
        p(u, rand_state(p, 2))
    p.plastic = False; p(u, rand_state(p, 2))
    assert torch.equal(p.addr, a0)
    p.plastic = True; p(u, rand_state(p, 2))
    assert not torch.equal(p.addr, a0)


def test_reset_clears_all_modules():
    p = make(); p.eval()
    st = rand_state(p, 1)
    u = torch.randn(1, 3, 5).double()
    reset = torch.tensor([[False, True, False]])
    with torch.no_grad():
        _, _, (c1, S1) = p(u[:, 1:], tuple(torch.zeros_like(x) for x in st))
        _, _, (c, S) = p(u, st, reset)
    assert torch.allclose(c, c1) and torch.allclose(S, S1)   # 等于从边界字节起、零状态起步（未选模块也被清空）


def test_model_builds_and_runs():
    m = D.build('dspn', 'S', hub_after=0, pool=8, pool_k=2, pool_d=16, pool_m=4, pool_hidden=32, pool_q=8)
    torch.nn.init.normal_(m.readout.weight, std=0.1)       # 读出零初始时梯度传不进各层
    for l in m.layers[1:]:
        torch.nn.init.normal_(l.core.W2, std=0.1)
    idx = torch.randint(0, 256, (2, 12))
    m.train(); lg, st = m(idx)
    assert lg.shape == (2, 12, 256)
    lg.sum().backward()
    assert m.layers[1].Wr.grad is not None and m.layers[1].Wr.grad.abs().sum() > 0


def test_wavenet_runs_grows_and_feedback():
    from pool import WaveNet
    torch.manual_seed(0)
    for fb in (0, 4):
        m = D.build('dspn', 'S', net='wave', N=16, k=2, G=3, D=8, pd=8, pm=4, phidden=16, fb=fb, grow_every=1, margin=-1.0)
        torch.nn.init.normal_(m.readout.weight, std=0.1)
        m.train()
        idx = torch.randint(0, 256, (2, 12))
        lg, st = m(idx)
        assert lg.shape == (2, 12, 256)
        lg.sum().backward()
        assert m.layers[1].Wr.grad.abs().sum() > 0
        assert m.layers[1].wave.max() >= 1                 # margin = −1：首步即从平的第 0 波长出第 1 波
        lg2, _ = m(idx, st)                                # 状态可接续
    # 只有本波被选的模块改状态：把波全设成 0 以外的空波，状态不应变
    m = D.build('dspn', 'S', net='wave', N=8, k=2, G=2, D=8, pd=8, pm=4, phidden=16)
    m.eval(); pool = m.layers[1]
    st = pool.init_state(1, 'cpu'); st = (torch.randn_like(st[0]), torch.randn_like(st[1]))
    with torch.no_grad():
        _, _, (c, S) = pool(torch.randn(1, 5, 8), st)
    moved = (c != st[0]).any(-1)[0]
    assert moved.sum() <= 8 and moved.sum() >= 2           # 第 0 波 8 个模块里，5 拍内只有被选过的变


def test_routing_rules_run():
    for rule in ('hash', 'gate'):
        m = D.build('dspn', 'S', net='wave', N=8, k=2, G=2, D=8, pd=8, pm=4, phidden=16, rule=rule)
        m.train(); lg, _ = m(torch.randint(0, 256, (2, 10)))
        assert lg.shape == (2, 10, 256)


def test_ema_scan_matches_loop_and_resets():
    from pool import ema_scan
    torch.manual_seed(0)
    x = torch.randn(2, 150, 5); e0 = torch.randn(2, 5); a = 0.9
    reset = torch.zeros(2, 150, dtype=torch.bool); reset[0, 70] = True
    y, e = ema_scan(x, e0, a, reset, C=64)
    ref, cur = [], e0
    for t in range(150):
        g = torch.where(reset[:, t], 0.0, a)[:, None]
        cur = g * cur + (1 - a) * x[:, t]; ref.append(cur)
    ref = torch.stack(ref, 1)
    assert torch.allclose(y, ref, atol=1e-5) and torch.allclose(e, ref[:, -1], atol=1e-5)


def test_ctx_routing_chunked_equals_whole():
    torch.manual_seed(0)
    m = D.build('dspn', 'S', net='wave', N=8, k=2, G=2, D=8, pd=8, pm=4, phidden=16, rule='ctx')
    torch.nn.init.normal_(m.readout.weight, std=0.1)
    m.eval(); idx = torch.randint(0, 256, (2, 40))
    with torch.no_grad():
        lg, _ = m(idx)
        a, st = m(idx[:, :17]); b, _ = m(idx[:, 17:], st)
    assert torch.allclose(lg, torch.cat([a, b], 1), atol=1e-4)
    m.train(); lg, _ = m(idx); lg.sum().backward()
    assert m.layers[1].Wr.grad.abs().sum() > 0


def test_window_counts_tau_prior_and_pred_loss():
    from pool import window_counts
    idx = torch.tensor([[1, 2, 2, 3, 1]])
    wc = window_counts(idx, 2)                              # 第 t 拍之后 2 个字节
    assert wc.shape == (1, 3, 256)
    assert wc[0, 0, 2] == 1.0 and wc[0, 1, 2] == 0.5 and wc[0, 1, 3] == 0.5 and wc[0, 2, 1] == 0.5
    m = D.build('dspn', 'S', net='wave', N=8, k=2, G=2, D=8, pd=8, pm=4, phidden=16, init='spread', tau_prior=4.0, pred_w=0.1)
    pool = m.layers[1]
    tau = 1 / (1 - torch.sigmoid(pool.core.b_a))
    w1 = (pool.wave == 1).nonzero()[0, 0]; w0 = (pool.wave == 0).nonzero()[0, 0]
    assert torch.allclose(tau[w1, :5], 4 * tau[w0, :5], rtol=1e-3)   # 第 1 波时间常数是第 0 波的 4 倍
    m.train(); lg, _ = m(torch.randint(0, 256, (2, 20)))
    assert torch.is_tensor(m.aux) and m.aux > 0
    (lg.sum() + m.aux).backward()
    assert m.pheads[1].weight.grad.abs().sum() > 0


def test_wavenet_reset_chunked_equals_whole():
    """会话边界字节（30）在段中间时，整段与分两段接续结果一致（Codex 审查指出：每波重复重置全部模块会清掉前一波的末状态）。"""
    for rule in ('addr', 'ctx'):
        torch.manual_seed(0)
        m = D.build('dspn', 'S', net='wave', N=8, k=2, G=2, D=8, pd=8, pm=4, phidden=16, init='spread', rule=rule)
        torch.nn.init.normal_(m.readout.weight, std=0.1)
        m.eval(); idx = torch.randint(31, 256, (2, 24)); idx[:, 5] = 30; idx[1, 15] = 30
        with torch.no_grad():
            lg, _ = m(idx)
            a, st = m(idx[:, :9]); b, _ = m(idx[:, 9:], st)
        assert torch.allclose(lg, torch.cat([a, b], 1), atol=1e-4), rule


def test_wavenet_bottleneck_read_last_and_wave_query():
    """瓶颈 + 只读末波 + 分波独立查询：可正常前向/反传，且每波的 Wr 都能收到梯度。"""
    for rule in ('addr', 'ctx'):
        torch.manual_seed(0)
        m = D.build('dspn', 'S', net='wave', N=8, k=2, G=2, D=8, pd=8, pm=4, phidden=16,
                    init='spread', rule=rule, bottleneck=True, read_last=True, wave_query=True)
        torch.nn.init.normal_(m.readout.weight, std=0.1)
        m.eval(); idx = torch.randint(0, 256, (2, 24))
        with torch.no_grad():
            lg, _ = m(idx)
            a, st = m(idx[:, :10]); b, _ = m(idx[:, 10:], st)
        assert lg.shape == (2, 24, 256)
        assert torch.allclose(lg, torch.cat([a, b], 1), atol=1e-4), rule
        m.train(); lg, _ = m(idx); lg.sum().backward()
        g = m.layers[1].Wr.grad
        assert g is not None and g.shape[0] == 2
        assert g[0].abs().sum() > 0 and g[1].abs().sum() > 0
