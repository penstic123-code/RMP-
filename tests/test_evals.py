import sys
from pathlib import Path
import pytest, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import evals, dspn_text as D


def test_mqar_batch():
    x, y, mask = evals.mqar_batch(4, 64, 6, torch.Generator().manual_seed(0))
    assert x.shape == y.shape == mask.shape == (4, 64)
    assert (mask.sum(1) == 6).all()                              # 查询数 = min(pairs, 槽数)
    for i in range(4):
        keys = x[i, :12:2].tolist()
        assert len(set(keys)) == 6                               # 键不重复
        for p in mask[i].nonzero().flatten().tolist():
            assert x[i, p - 1] == ord('?')
            assert y[i, p] == x[i, 1:12:2][keys.index(x[i, p].item())]   # 答案 = 该键前文对应的值
    with pytest.raises(ValueError):
        evals.mqar_batch(1, 10, 6, torch.Generator())


def test_train_mqar():
    model = D.build('dspn', 'S', layers=2, rows=1, cols=2, d=8, m=4, hidden=16, q=4, emb=8)
    out = evals.train_mqar(model, 20, 4, 64, 4, 1e-3, 'cpu')
    assert set(out) == {'acc', 'acc_near', 'acc_far'}


def test_name_recall():
    out = evals.name_recall(D.GRULM(16), 'cpu', n=5)
    assert set(out) == {'acc', 'bits', 'acc_control', 'bits_control'}
