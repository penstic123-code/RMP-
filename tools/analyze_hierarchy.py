"""层级是否自发涌现：逐层统计门控时间常数、快记忆保持时长与读出依赖。

用法：python tools/analyze_hierarchy.py runs/dspn-S.pt [--size S] [--extra '{}'] [--random]
--random 时不加载检查点，给出随机初始化的同一组统计作对照（训练后出现分化才算“长出来”）。
"""
import argparse, json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dspn_text as D


@torch.no_grad()
def layer_stats(model, x):
    got = {}                                           # 有工作空间时每层按 64 拍一块多次调用，按时间拼回
    hooks = [l.register_forward_pre_hook(lambda m, args, k=k: got.setdefault(k, []).append(args[0]))
             for k, l in enumerate(model.layers)]
    model(x)
    for h in hooks:
        h.remove()
    inputs = [torch.cat(got[k], 1) for k in range(len(model.layers))]
    L, M, d = len(model.layers), model.M, model.d
    w = model.readout.weight.reshape(model.readout.weight.shape[0], L, M * d)
    out = []
    for i, (layer, u) in enumerate(zip(model.layers, inputs)):
        a = torch.sigmoid(torch.einsum('btmu,mud->btmd', u, layer.G) + layer.b_a)
        c, _ = D.scan(a, (1 - a) * torch.einsum('btmu,mud->btmd', u, layer.B), torch.zeros_like(a[:, 0]))
        gam = torch.sigmoid((c * layer.w_gam).sum(-1) + layer.b_gam)
        tau = 1 / (1 - a).clamp(min=1e-6)
        out.append({'layer': i + 1,
                    'tau_median': round(tau.median().item(), 1),
                    'tau_frac_gt100': round((tau > 100).float().mean().item(), 3),
                    'fastmem_keep_median': round((1 / gam).median().item(), 1),
                    'readout_weight_share': round((w[:, i].norm() ** 2 / w.norm() ** 2).item(), 3)})
    return out


@torch.no_grad()
def context_swap(model, x, x_other, prefix):
    """上下文替换（Chien 等 2020 思路）：后段相同、前段换成别的文本，逐层测 h 的相对差异随时间的衰减。
    返回每层在替换点后若干拍的相对差异，以及差异降到起点一半所需拍数（该层实际依赖前文的长度）。"""
    hs = {}
    hooks = [l.register_forward_hook(lambda m, i, o, k=k: hs.setdefault(k, []).append(o[0]))
             for k, l in enumerate(model.layers)]
    swapped = x.clone(); swapped[:, :prefix] = x_other[:, :prefix]
    model(x); model(swapped)
    for h in hooks:
        h.remove()
    out = []
    for k in range(len(model.layers)):
        half_n = len(hs[k]) // 2                                   # 前一半是原文、后一半是替换后（每段可能分块）
        h1, h2 = (torch.cat(c, 1)[:, prefix:].flatten(2) for c in (hs[k][:half_n], hs[k][half_n:]))   # (b, N, M*d)
        diff = ((h1 - h2).norm(dim=-1) / (h1.norm(dim=-1) + h2.norm(dim=-1) + 1e-8)).mean(0)
        half = (diff < diff[0] / 2).nonzero()
        out.append({'layer': k + 1, **{f'diff@{t}': round(diff[t].item(), 3) for t in (0, 4, 16, 64, 256) if t < len(diff)},
                    'half_life': int(half[0]) if len(half) else f'>{len(diff)}'})
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt'); ap.add_argument('--size', default='S'); ap.add_argument('--extra', default='{}')
    ap.add_argument('--random', action='store_true')
    a = ap.parse_args()
    torch.manual_seed(0)
    model = D.build('dspn', a.size, **json.loads(a.extra)).to(D.DEV).eval()
    if not a.random:
        model.load_state_dict(torch.load(a.ckpt, map_location=D.DEV))
    _, va = D.load_bytes('tiny')
    x, _ = D.batch(va, 16, 512, torch.Generator().manual_seed(0))
    for row in layer_stats(model, x):
        print(row)
    x2, _ = D.batch(va, 16, 1024, torch.Generator().manual_seed(1))
    x1, _ = D.batch(va, 16, 1024, torch.Generator().manual_seed(2))
    for row in context_swap(model, x1, x2, prefix=512):
        print('swap', row)
