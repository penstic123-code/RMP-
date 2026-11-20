"""RMP P2（第 23 章 Q1）：模块分工的涌现指标，与同配置随机初始化对照。

1. 字节选择性：被选模块与当前字节类别（小写、大写、数字、空格、换行、标点、其他）的互信息 / H(类别)。
2. 当前字节之外的选择性：固定当前字节类别后，被选模块与上下文（引号内 / 外 × 词首 / 词中 / 非词）的条件互信息 / H(上下文 | 类别)。
3. 剪除特异性：逐个排除负载最高的 n 个模块（偏置设成极小，不再被选），看各字节类别的损失升幅；
   某类升幅 ≥ 其余类平均升幅的 2 倍记为“特异”。
4. 各波的负载熵 / log(波内模块数)。
样本：被选的（拍, 模块）对按混合权重计权。按波分别算，再按样本数平均。
用法：python tools/rmp_emergence.py --tag _wspread --extra '{...}' [--random]
"""
import argparse, json, math, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import torch, torch.nn.functional as F
import dspn_text as D

ap = argparse.ArgumentParser()
ap.add_argument('--tag', default='_wspread'); ap.add_argument('--size', default='S')
ap.add_argument('--extra', default='{}'); ap.add_argument('--random', action='store_true')
ap.add_argument('--rows', type=int, default=16); ap.add_argument('--T', type=int, default=1024)
ap.add_argument('--ablate', type=int, default=8)
a = ap.parse_args()
extra = json.loads(a.extra)
torch.manual_seed(0)
m = D.build('dspn', a.size, **extra).to(D.DEV).eval()
if not a.random:
    r = m.load_state_dict(torch.load(ROOT / f'runs/dspn-{a.size}{a.tag}.pt', map_location=D.DEV), strict=False)
    assert not r.unexpected_keys and all(k.endswith(('.moved', '.active')) for k in r.missing_keys), r
pool = m.layers[1]

val = D.load_bytes('tiny')[1]
pos = np.linspace(0, len(val) - a.T - 2, a.rows).astype(np.int64)
xy = torch.from_numpy(np.stack([val[p:p + a.T + 1] for p in pos]).astype(np.int64)).to(D.DEV)
x, y = xy[:, :-1], xy[:, 1:]
xb = x.cpu().numpy()


def byte_class(c):
    return np.select([(c >= 97) & (c <= 122), (c >= 65) & (c <= 90), (c >= 48) & (c <= 57), c == 32, c == 10,
                      np.isin(c, list(b'.,!?;:\'"-()'))], [0, 1, 2, 3, 4, 5], 6)


cls = byte_class(xb)
quote = (np.cumsum(xb == 34, 1) % 2)                                   # 引号内
alpha = ((xb >= 97) & (xb <= 122)) | ((xb >= 65) & (xb <= 90))
prev_alpha = np.concatenate([np.zeros((xb.shape[0], 1), bool), alpha[:, :-1]], 1)
wpos = np.where(~alpha, 2, np.where(prev_alpha, 1, 0))                 # 0 词首、1 词中、2 非词
ctx = quote * 3 + wpos


def H(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def mi(joint):
    joint = joint / joint.sum()
    return H(joint.sum(1)) + H(joint.sum(0)) - H(joint.ravel())


with torch.no_grad():
    pool.capture = []
    lg, _ = m(x)
    caps, pool.capture = pool.capture, None
    ce = F.cross_entropy(lg.reshape(-1, 256), y.reshape(-1), reduction='none').view(x.shape).cpu().numpy()

res = {'tag': a.tag, 'random': a.random, 'waves': []}
tot_w, sel_byte, sel_ctx, load_ent = 0.0, 0.0, 0.0, 0.0
for w, sel, wt, _ in caps:
    wt = wt.cpu().numpy()                                               # (b, T, N)
    mods = np.where(np.asarray(pool.wave.cpu()) == w)[0]
    if len(mods) == 0:
        continue
    W = wt[..., mods]                                                   # (b, T, n)
    n = len(mods)
    jb = np.zeros((n, 7)); jc = np.zeros((7, n, 6))
    for c in range(7):
        mk = cls == c
        jb[:, c] = W[mk].sum(0)
        for q in range(6):
            jc[c, :, q] = W[mk & (ctx == q)].sum(0)
    byte_sel = mi(jb) / max(H(jb.sum(0) / jb.sum()), 1e-9)
    pc = jc.sum((1, 2)) / jc.sum()
    cmi = sum(pc[c] * mi(jc[c]) for c in range(7) if jc[c].sum() > 0)
    hctx = sum(pc[c] * H(jc[c].sum(0) / jc[c].sum()) for c in range(7) if jc[c].sum() > 0)
    ctx_sel = cmi / max(hctx, 1e-9)
    load = (sel.cpu().numpy()[..., mods]).mean((0, 1)); load = load / load.sum()
    le = H(load) / math.log(n) if n > 1 else 1.0
    res['waves'].append({'wave': int(w), 'modules': n, 'byte_sel': round(byte_sel, 4), 'ctx_sel_given_byte': round(ctx_sel, 4), 'load_entropy': round(le, 3)})
    s = W.sum(); tot_w += s; sel_byte += s * byte_sel; sel_ctx += s * ctx_sel; load_ent += s * le
res['byte_sel'] = round(sel_byte / tot_w, 4); res['ctx_sel_given_byte'] = round(sel_ctx / tot_w, 4); res['load_entropy'] = round(load_ent / tot_w, 3)

# 剪除特异性：负载最高的模块逐个排除
loads = np.concatenate([c[1].cpu().numpy().mean((0, 1))[None] for c in caps]).sum(0)
top = np.argsort(-loads)[:a.ablate]
base = np.array([ce[cls == c].mean() if (cls == c).any() else np.nan for c in range(7)])
spec = []
with torch.no_grad():
    for i in top:
        old = pool.bias[i].item(); pool.bias[i] = -1e9
        lg2, _ = m(x)
        pool.bias[i] = old
        ce2 = F.cross_entropy(lg2.reshape(-1, 256), y.reshape(-1), reduction='none').view(x.shape).cpu().numpy()
        rise = np.array([ce2[cls == c].mean() - base[c] if (cls == c).any() else np.nan for c in range(7)])
        j = int(np.nanargmax(rise)); others = np.nanmean(np.delete(rise, j))
        spec.append({'module': int(i), 'wave': int(pool.wave[i]), 'max_class': j, 'max_rise': round(float(rise[j]), 4),
                     'others_mean': round(float(others), 4), 'specific': bool(rise[j] > 0 and rise[j] >= 2 * max(others, 1e-6))})
res['ablation'] = spec; res['n_specific'] = sum(s['specific'] for s in spec)
res['val_bpc'] = round(float(ce.mean() / math.log(2)), 4)
print(json.dumps({k: res[k] for k in ('tag', 'random', 'val_bpc', 'byte_sel', 'ctx_sel_given_byte', 'load_entropy', 'n_specific')}, ensure_ascii=False, default=float))
p = ROOT / f'runs/{time.strftime("%Y%m%d-%H%M%S")}_rmp-emergence{a.tag}{"_random" if a.random else ""}.json'
json.dump(res, open(p, 'w'), ensure_ascii=False, indent=1, default=float)
print('→', p.name)
