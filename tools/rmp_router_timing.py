"""RMP P2b-T2（第 23.8 节）：找邻居（路由）的耗时随模块数 N 怎样变——余弦全比 vs product key（PEER 式）。

每次路由 b·T = 16384 个查询，查询维 32，取前 k = 4。product key：查询拆两半各 16 维，各配 √N 个子键，
每半取前 k 个，k² 个组合里再取前 k（复杂度约 √N）。另测一个 d = 32 模块一拍的计算量作参照（掩码稠密一个模块）。
输出每个 N 的毫秒数，JSON 存 runs/。
"""
import json, math, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch, torch.nn.functional as F

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
Q, qr, k = 16384, 32, 4
q = F.normalize(torch.randn(Q, qr, device=dev), dim=-1)


def timeit(fn, n=20):
    for _ in range(3):
        fn()
    if dev == 'cuda':
        torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        fn()
    if dev == 'cuda':
        torch.cuda.synchronize()
    return (time.time() - t) / n * 1000


def full(addr):
    return (q @ addr.t()).topk(k, -1).indices


def pkey(k1, k2):
    s1, i1 = (q[:, :qr // 2] @ k1.t()).topk(k, -1)
    s2, i2 = (q[:, qr // 2:] @ k2.t()).topk(k, -1)
    comb = (s1[:, :, None] + s2[:, None, :]).flatten(1)                    # k² 个组合
    j = comb.topk(k, -1).indices
    return i1.gather(1, j // k) * k2.shape[0] + i2.gather(1, j % k)


# 参照：一个 d = 32 模块的一拍（输入投影 + MLP），Q 个位置
d, U, hid = 32, 32, 64
W = [torch.randn(U, 2 * d, device=dev), torch.randn(d + 8, hid, device=dev), torch.randn(hid, d, device=dev)]
x = torch.randn(Q, U, device=dev)
module_ms = timeit(lambda: F.gelu(torch.cat([(x @ W[0])[:, :d], x[:, :8]], -1) @ W[1]) @ W[2])

out = {'device': dev, 'queries': Q, 'qr': qr, 'k': k, 'one_module_ms': round(module_ms, 4), 'rows': []}
for N in (8, 32, 128, 512, 2048, 4096):
    addr = F.normalize(torch.randn(N, qr, device=dev), dim=-1)
    s = int(math.ceil(math.sqrt(N)))
    k1, k2 = torch.randn(s, qr // 2, device=dev), torch.randn(s, qr // 2, device=dev)
    row = {'N': N, 'full_ms': round(timeit(lambda: full(addr)), 4), 'pkey_ms': round(timeit(lambda: pkey(k1, k2)), 4)}
    row['full_over_k_modules'] = round(row['full_ms'] / (k * module_ms), 3)   # 路由耗时 / 被选 k 个模块的计算耗时
    out['rows'].append(row)
    print(row, flush=True)
p = ROOT / f'runs/{time.strftime("%Y%m%d-%H%M%S")}_rmp-router-timing.json'
json.dump(out, open(p, 'w'), indent=1)
print('→', p.name)
