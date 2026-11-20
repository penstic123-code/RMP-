"""涌现指标（第 20.8 节五项；plan/fast-clean.md C5 把关用）。与同配置随机初始化（--random）对照才有意义。

  1. 时间尺度层级：各层慢通道门的有效时间常数 τ = 1/(1−a) 中位数（块层换算成字节：乘平均块长）；
     分数 = 层号与 τ 中位数的 Spearman 相关（分块时只算块层，免得“块层 ×N”本身造出层级）。
  2. 自发切词：块边界落在空格 / 换行 / 标点之后的精确率、召回率、块率（分块模型才有）。
  3. 单元分工：各层每个单元对“当前字节类别”（小写、大写、数字、空格、换行、标点、其他）的相关比 η²，层内平均。
     取读出前拼接的各层 h（字节分辨率，块层是放回字节后的值），两版代码通用。
  4. 复制：随机串（字母，长 100）重复两遍，第二遍第 2 个字节起的逐字节准确率。
  5. 中枢分工：各头的有效保持 τ、写入强度 β、输出占比；分数 = 各头 log10 τ 的标准差与输出占比的归一化熵。

用法：python tools/emergence.py --size C --extra '{...}' --ckpt runs/x.pt [--random] [--code 另一份代码的根目录]
--code 用来分析旧版代码训练的检查点（如 git worktree 里的 v0.17）。输出一行 JSON。
"""
import argparse, json, math, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--code', default=str(Path(__file__).resolve().parent.parent))
ap.add_argument('--size', default='C'); ap.add_argument('--extra', default='{}')
ap.add_argument('--ckpt', default=None); ap.add_argument('--random', action='store_true')
ap.add_argument('--data', default='mix'); ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()
sys.path.insert(0, a.code)
import numpy as np
import torch, torch.nn.functional as F
import dspn_text as D

CLASSES = 7


def byte_class(x):
    c = torch.full_like(x, 6)
    c[(x >= 97) & (x <= 122)] = 0; c[(x >= 65) & (x <= 90)] = 1; c[(x >= 48) & (x <= 57)] = 2
    c[x == 32] = 3; c[x == 10] = 4
    for p in b'.,!?;:\'"-()':
        c[x == p] = 5
    return c


def spearman(v):
    """层号与 v 的 Spearman 相关（并列取平均名次）；v 全相同时没有层级，记 0。"""
    v = np.asarray(v, float)
    if len(v) < 3 or np.ptp(v) == 0:
        return 0.0
    r = np.array([np.mean(np.flatnonzero(np.sort(v) == x)) for x in v])
    return float(np.corrcoef(np.arange(len(v)), r)[0, 1])


@torch.no_grad()
def main():
    torch.manual_seed(a.seed)
    model = D.build('dspn', a.size, **json.loads(a.extra)).to(D.DEV)
    if not a.random:
        model.load_state_dict(torch.load(a.ckpt, map_location=D.DEV))
    model.eval()
    L, chunk = len(model.layers), getattr(model, 'chunk', 0)
    _, va = D.load_bytes(a.data)

    # 钩子：各层门的 log a、读出前的拼接 h、中枢输入与读出
    rec = {'la': [[] for _ in range(L)], 'feat': [], 'hub_x': [], 'hub_r': []}
    for i, l in enumerate(model.layers):
        orig = l.gates
        def wrapped(u, orig=orig, i=i):
            la, x = orig(u); rec['la'][i].append(la.float().flatten(0, 1).flatten(1)); return la, x
        l.gates = wrapped
    model.norm.register_forward_pre_hook(lambda m, args: rec['feat'].append(args[0].float()))
    if model.hub is not None:
        def hub_hook(m, i, o):                             # 钩子必须返回 None，否则会替换中枢输出
            rec['hub_x'].append(i[0].float()); rec['hub_r'].append(o[2].float())
        model.hub.register_forward_hook(hub_hook)

    # 流式读验证集：8 条流 × 4 段 × 1024，第 1 段只预热
    b, T, n = 8, 1024, 4
    pos = np.linspace(0, len(va) - n * T - 2, b).astype(np.int64)
    st, xs, seg = model.init_state(b), [], {'rate': [], 'word': [], 'word_base': []}
    for k in range(n):
        x, _ = D.stream_batch(va, pos, T)
        if k == 1:                                         # 丢掉预热段的记录
            for lst in rec['la'] + [rec['feat'], rec['hub_x'], rec['hub_r']]:
                lst.clear()
        _, st = model(x, st)
        if k:
            xs.append(x)
            if chunk:
                for key in seg:
                    seg[key].append(float(model.stats[key]))
    x = torch.cat(xs, 1)
    out = {'ckpt': a.ckpt, 'random': a.random, 'size': a.size, 'extra': json.loads(a.extra)}

    # 2. 切词
    rate = float(np.mean(seg['rate'])) if chunk else None
    if chunk:
        prec, base = float(np.mean(seg['word'])), float(np.mean(seg['word_base']))
        out['seg'] = {'rate': round(rate, 4), 'precision': round(prec, 4), 'base_rate': round(base, 4),
                      'recall': round(prec * rate / base, 4), 'lift': round(prec / base, 3)}

    # 1. 时间尺度层级
    taus = []
    for i in range(L):
        la = torch.cat(rec['la'][i], 0)
        tau = (1 / (-torch.expm1(la)).clamp(min=1e-6)).median().item()
        if chunk and 0 < i < L - 1:
            tau /= rate                                     # 块步 → 字节
        taus.append(round(tau, 1))
    mid = taus[1:-1] if chunk else taus
    out['timescale'] = {'tau_bytes_median': taus, 'score': round(spearman(mid), 3)}

    # 3. 单元分工：η² = 类间方差 / 总方差，每单元算、层内平均
    feat = torch.cat(rec['feat'], 1).flatten(0, 1)          # (N, Σ M·d_l)
    cls = byte_class(x).flatten()
    widths = [model.M * l.d for l in model.layers]
    tot = feat.var(0, unbiased=False) + 1e-8
    means = torch.stack([feat[cls == c].mean(0) if (cls == c).any() else feat.mean(0) for c in range(CLASSES)])
    w = torch.stack([(cls == c).float().mean() for c in range(CLASSES)])
    between = (w[:, None] * (means - feat.mean(0)) ** 2).sum(0)
    eta = (between / tot).cpu()
    out['units'] = {'eta2_per_layer': [round(v.mean().item(), 3) for v in eta.split(widths)]}

    # 5. 中枢分工
    if model.hub is not None:
        h = model.hub
        hx = torch.cat(rec['hub_x'], 1); hr = torch.cat(rec['hub_r'], 1)
        alpha = torch.sigmoid((hx @ h.w_alpha.float()) + h.b_alpha.float())            # (b, T, H)
        tau_h = (1 / (1 - alpha.mean((0, 1))).clamp(min=1e-6)).cpu()
        beta = (h.beta_max * torch.sigmoid(hx @ h.w_beta.float() + h.b_beta.float())).mean((0, 1)).cpu()
        Wo = h.Wo.float().reshape(h.heads, h.dk, -1)
        contrib = torch.stack([(hr[..., j * h.dk:(j + 1) * h.dk] @ Wo[j]).square().mean() for j in range(h.heads)]).cpu()
        share = contrib / contrib.sum()
        ent = float(-(share * share.clamp(min=1e-12).log()).sum() / math.log(h.heads))
        if chunk:
            tau_h = tau_h / rate
        out['hub'] = {'tau_bytes': [round(v, 1) for v in tau_h.tolist()], 'beta': [round(v, 3) for v in beta.tolist()],
                      'share': [round(v, 3) for v in share.tolist()],
                      'log_tau_std': round(float(tau_h.log10().std()), 3), 'share_entropy': round(ent, 3)}
    # 4. 复制
    g = torch.Generator().manual_seed(7)
    letters = torch.tensor(list(b'abcdefghijklmnopqrstuvwxyz'))
    s = letters[torch.randint(26, (64, 100), generator=g)]
    seq = torch.cat([torch.full((64, 1), D.SEP), s, s], 1).to(D.DEV)
    lg, _ = model(seq[:, :-1])
    hit = (lg.argmax(-1) == seq[:, 1:])[:, 101:].float()   # 目标为第二遍第 2 个字节起
    out['copy'] = round(hit.mean().item(), 4)

    print(json.dumps(out, ensure_ascii=False))


main()
