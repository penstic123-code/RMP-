"""聊天程序冒烟：小模型跑一段含 /new、/sleep 的脚本不出错；每个字节恰好输入模型一次（逐字节与整段一致）。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
import torch
import dspn_text as D
from chat import Chat


def tiny():
    torch.manual_seed(0)
    m = D.DSPNLayered(layers=2, M=2, d=8, m=4, hidden=8, q=4, emb=8, hub_after=1, hub_dk=8, hub_out=8,
                      hub_taus=[10.0, 1000.0]).to(D.DEV)
    with torch.no_grad():
        m.readout.weight.normal_(0, 0.3)
        for l in m.layers:
            l.W2.normal_(0, 0.3)
    return m


def test_script_runs():
    c = Chat(tiny(), max_reply=20, lr=1e-3)
    for line in ['My name is Kobi.', '/new', 'What is my name?', '/sleep', 'Hi']:
        assert isinstance(c.turn(line), str)


def test_advance_feeds_each_byte_once():
    m = tiny()
    c = Chat(m, max_reply=5, lr=0.0)
    text = b'Bot: hello there\n'
    c._advance(text[:5]); c._advance(text[5:])
    with torch.no_grad():
        _, ref = m(torch.tensor([list(text[:-1])], device=D.DEV))
    assert torch.allclose(c.L.state[0][1], ref[0][1], atol=1e-3)   # GPU 上 fla 内核与本地实现差约 2e-4
