"""DSPN 字节级语言模型的“记忆”评测（docs/plan/README.md 验收项 3a）。

  mqar_batch / train_mqar：多查询联想召回（MQAR，Zoology，Arora 等 2023），从头训练给定模型。
  name_recall：故事内名字召回，只评测已训练好的语言模型。

用法：
  python evals.py mqar dspn --size S --pairs 16 --T 1024 --iters 2000
  python evals.py names dspn --size S          # 读 runs/dspn-S.pt
"""
import argparse, json, math
import torch, torch.nn as nn, torch.nn.functional as F

KEYS = b'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'   # 52 个，最多 52 对；与值不重叠
VALS = b'0123456789'
QUERY, PAD = ord('?'), ord(' ')
NEAR = 256


def mqar_batch(b, T, pairs, gen):
    """序列：前 pairs 个“键 值”对，后段若干“? 键 值”三元组（随机落在后段的 3 字节槽里），其余空格。
    生成长 T+1 的字节串 s，x=s[:-1]，y=s[1:]；答案值 v 前一位（即查询里的键）处输入，故 mask 落在该位置。"""
    if pairs > len(KEYS):
        raise ValueError(f'键取自 a-z、A-Z，最多 {len(KEYS)} 对，收到 {pairs}')
    slots = (T + 1 - 2 * pairs) // 3
    if slots < 1:
        raise ValueError(f'T={T} 放不下 {pairs} 对键值和至少 1 个查询')
    nq = min(pairs, slots)
    keys, vals = torch.tensor(list(KEYS)), torch.tensor(list(VALS))
    s = torch.full((b, T + 1), PAD, dtype=torch.long)
    mask = torch.zeros(b, T, dtype=torch.bool)
    for i in range(b):
        k = keys[torch.randperm(len(keys), generator=gen)[:pairs]]
        v = vals[torch.randint(len(vals), (pairs,), generator=gen)]
        s[i, 0:2 * pairs:2], s[i, 1:2 * pairs:2] = k, v
        pos = 2 * pairs + 3 * torch.randperm(slots, generator=gen)[:nq].sort().values
        q = torch.randperm(pairs, generator=gen)[:nq]
        s[i, pos], s[i, pos + 1], s[i, pos + 2] = QUERY, k[q], v[q]
        mask[i, pos + 1] = True
    return s[:, :-1], s[:, 1:], mask


def _acc(model, x, y, mask, pairs):
    """答案位置 top-1 准确率，以及按查询距键远近（≤256 / >256 拍）分组的准确率。"""
    hit = model(x)[0].argmax(-1) == y
    bi, p = mask.nonzero(as_tuple=True)
    kpos = (x[bi, :2 * pairs:2] == x[bi, p, None]).float().argmax(-1) * 2   # 该键在前段对里的位置
    near = (p - kpos) <= NEAR
    h = hit[bi, p].float()
    nan = float('nan')
    return {'acc': h.mean().item(),
            'acc_near': h[near].mean().item() if near.any() else nan,
            'acc_far': h[~near].mean().item() if (~near).any() else nan}


def train_mqar(model, iters, b, T, pairs, lr, device):
    gen = torch.Generator().manual_seed(0)
    model.to(device).train()
    from dspn_text import param_groups
    opt = torch.optim.AdamW(param_groups(model), lr=lr)
    for it in range(1, iters + 1):
        x, y, mask = (t.to(device) for t in mqar_batch(b, T, pairs, gen))
        lg, _ = model(x)
        loss = F.cross_entropy(lg[mask], y[mask])
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if it % 100 == 0:
            print(it, round(loss.item(), 4), flush=True)
    model.eval()
    eval_gen = torch.Generator().manual_seed(12345)           # 固定评测集，不同模型可比
    with torch.no_grad():
        x, y, mask = (t.to(device) for t in mqar_batch(max(b, 64), T, pairs, eval_gen))
        return _acc(model, x, y, mask, pairs)


NAMES = ['Zeb', 'Mirabel', 'Orson', 'Thistle', 'Quillon', 'Bramwell', 'Odette', 'Fennick', 'Ismay', 'Lorcan',
         'Wilhelmina', 'Cosmo', 'Tamsin', 'Barnaby', 'Perpetua', 'Ignatius', 'Marigold', 'Eustace', 'Philippa',
         'Rupert', 'Seraphina', 'Alaric', 'Winifred', 'Jasper', 'Clementine', 'Dorian', 'Hortense', 'Leopold',
         'Ottoline', 'Percival', 'Rosalind', 'Tobias', 'Ursula', 'Valentin', 'Xanthe', 'Yorick', 'Zelda',
         'Gideon', 'Imelda', 'Nigel']
TEMPLATES = [
    'Once upon a time there was a girl named {A}. She lived in a small house near the woods. One sunny day, {B} was happy.',
    'There was a boy called {A}. He loved to play with his red ball in the garden. Every evening, {B} went home.',
    'One day, a little dog named {A} found a bone. The bone was big and tasty. After a while, {B} wanted to share it.',
]


@torch.no_grad()
def _name_stats(model, device, first, second, tpl):
    """第二次出现的名字（句首，前面是“. ”或“, ”）第 2 个字节起各字节的 (命中数, 比特数, 字节数)。"""
    text = tpl.format(A=first, B=second)
    s = torch.tensor(list(text.encode()), device=device)[None]
    start = text.rindex(second)                                         # 模板里 B 在最后一次出现处
    lg = model(s[:, :-1])[0][0].float()
    pos = torch.arange(start + 1, start + len(second), device=device)   # 目标字节下标；由前一字节处的输出预测
    lp = F.log_softmax(lg[pos - 1], -1)[torch.arange(len(pos)), s[0, pos]]
    return (lg[pos - 1].argmax(-1) == s[0, pos]).sum().item(), -lp.sum().item() / math.log(2), len(pos)


def name_recall(model, device, n=200, seed=0):
    rng = torch.Generator().manual_seed(seed)
    model.to(device).eval()
    tot = {'real': [0, 0.0, 0], 'ctrl': [0, 0.0, 0]}
    for _ in range(n):
        a, c = torch.randperm(len(NAMES), generator=rng)[:2].tolist()
        tpl = TEMPLATES[torch.randint(len(TEMPLATES), (1,), generator=rng).item()]
        for key, first in (('real', NAMES[a]), ('ctrl', NAMES[c])):
            h, bits, cnt = _name_stats(model, device, first, NAMES[a], tpl)
            tot[key][0] += h; tot[key][1] += bits; tot[key][2] += cnt
    r, c = tot['real'], tot['ctrl']
    return {'acc': r[0] / r[2], 'bits': r[1] / r[2], 'acc_control': c[0] / c[2], 'bits_control': c[1] / c[2]}


if __name__ == '__main__':
    import dspn_text as D
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['mqar', 'names']); ap.add_argument('kind', choices=['dspn', 'gru', 'tf'])
    ap.add_argument('--size', default='S', choices=list(D.SIZES))
    ap.add_argument('--extra', default='{}')
    ap.add_argument('--tag', default='', help='names：读 runs/{kind}-{size}{tag}.pt')
    ap.add_argument('--iters', type=int, default=2000); ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--T', type=int, default=256); ap.add_argument('--pairs', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    a = ap.parse_args()
    model = D.build(a.kind, a.size, **json.loads(a.extra))
    if a.cmd == 'mqar':
        res = train_mqar(model, a.iters, a.batch, a.T, a.pairs, a.lr, D.DEV)
    else:
        model.load_state_dict(torch.load(D.ROOT / f'runs/{a.kind}-{a.size}{a.tag}.pt', map_location=D.DEV))
        res = name_recall(model, D.DEV)
    print(res)
    import time
    out = D.ROOT / f'runs/{time.strftime("%Y%m%d-%H%M%S")}_{a.cmd}-{a.kind}-{a.size}{a.tag}.json'
    json.dump({'args': vars(a), **res}, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('→', out.name)
