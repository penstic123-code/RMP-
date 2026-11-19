"""睡眠：steps=0 只并入低秩偏移，应精确保持预测并提交；anchor>0 能跑通。"""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dspn_text as D
from online import OnlineLearner, bpc

TEXT = b'Tom found a blicket. A blicket is a small red bell. ' * 8


def _learner():
    torch.manual_seed(0)
    m = D.build('dspn', 'S', hub_after=2).to(D.DEV)
    with torch.no_grad():                  # 读出与 W2 初始为零时低秩偏移学不动；模拟训练过的模型
        m.readout.weight.normal_(0, 0.1)
        for l in m.layers:
            l.W2.normal_(0, 0.1)
    L = OnlineLearner(m, lr=3e-3, params='lora')
    L.read(TEXT)
    assert any(l.lora_b.abs().sum() > 0 for l in m.layers)
    return m, L


def test_merge_only_is_exact():
    m, L = _learner()
    before = bpc(m, TEXT)
    r = L.sleep(TEXT[:256], TEXT[-256:], steps=0)
    assert r['committed']
    assert abs(bpc(m, TEXT) - before) < 1e-3
    zeros = [l.lora_b for l in m.layers] + [m.hub.lora_q_b, m.hub.lora_o_b]   # 各层与中枢偏移都已并入并清零
    assert all(b.abs().sum() == 0 for b in zeros)


def test_dream_runs():
    m, L = _learner()
    r = L.sleep(TEXT[:256], TEXT[-256:], steps=3, batch=2, dream=1.0, n_dream=4)
    assert set(r) >= {'committed', 'old_after', 'new_after'}


def test_anchor_runs():
    m, L = _learner()
    r = L.sleep(TEXT[:256], TEXT[-256:], steps=3, batch=2, anchor=1.0)
    assert set(r) >= {'committed', 'old_after', 'new_after'}


def test_distill_runs_without_pool_ce():
    m, L = _learner()
    day = b'User: My cat is Mirafe.\nBot: Nice name.\nUser: What is my cat?\nBot: Mirafe.\n' * 3
    r = L.sleep(TEXT[:256], day, steps=3, batch=2, anchor=1.0, distill=1.0, day=day, ce=0.0, win=32)
    assert set(r) >= {'committed', 'old_after', 'new_after'}
