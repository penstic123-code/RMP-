"""RMP P3（第 23 章 Q2）：在线学习与抗遗忘，只比 RMP 自己的两种做法——醒时冻结路由 vs 不冻结。

frozen：在线时地址、均衡偏置、W_r 都不动（plastic=False；W_r 不求导）；睡眠中才更新地址与偏置。
plastic：在线时地址与偏置照训练规则更新，W_r 也随在线损失求导。
两项任务（每种做法各从同一检查点重新加载）：
  1. 生造词（online.demo_new_word，E31 协议：先读 32KB 故事、重放 0.25、低秩 lr 1e-3，再睡眠）；
  2. 顺序任务：故事模型在线读 chat3 对话 64KB，报告对话 bpc 的下降与故事旧数据 bpc 的回升。
用法：python tools/rmp_online.py --tag _wspread --extra '{...训练时的 extra...}'
输出一行摘要，JSON 存 runs/。
"""
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
import dspn_text as D
import online as O

ap = argparse.ArgumentParser()
ap.add_argument('--tag', default='_wspread'); ap.add_argument('--size', default='S')
ap.add_argument('--extra', default='{}'); ap.add_argument('--seq_bytes', type=int, default=65536)
a = ap.parse_args()
extra = json.loads(a.extra)


def load(mode):
    m = D.build('dspn', a.size, **extra).to(D.DEV)
    m.load_state_dict(torch.load(ROOT / f'runs/dspn-{a.size}{a.tag}.pt', map_location=D.DEV))
    for pl in O._pools(m):
        pl.plastic = mode == 'plastic'
    return m


def unfreeze_router(m, L):
    """plastic 组：W_r 也进在线优化器。"""
    ps = [pl.Wr for pl in O._pools(m)]
    for p in ps:
        p.requires_grad_(True)
    L.groups.append({'params': ps, 'weight_decay': 0.0}); L.online += ps
    L.opt = torch.optim.AdamW(L.groups, lr=L.lr)


def sequential(mode):
    m = load(mode)
    val = D.load_bytes('tiny')[1]
    old = bytes(val[:4096])
    ctr, cva = D.load_bytes('chat3')
    new_probe = bytes(cva[:4096])
    L = O.OnlineLearner(m, lr=1e-3, replay=0.25)
    if mode == 'plastic':
        unfreeze_router(m, L)
    r = {'old_before': O.bpc(m, old), 'chat_before': O.bpc(m, new_probe)}
    L.read(bytes(ctr[:a.seq_bytes])); L.reset_session()
    r.update(old_after=O.bpc(m, old), chat_after=O.bpc(m, new_probe))
    r['old_rise'] = r['old_after'] - r['old_before']; r['chat_drop'] = r['chat_before'] - r['chat_after']
    return r


def new_word(mode):
    m = load(mode)
    if mode == 'plastic':                                  # demo_new_word 自建学习器；先打补丁让它带上 W_r
        init = O.OnlineLearner.__init__
        def patched(self, model, **kw):
            init(self, model, **kw); unfreeze_router(model, self)
        O.OnlineLearner.__init__ = patched
        try:
            return O.demo_new_word(m, D.DEV, day_bytes=32768, lr=1e-3, replay=0.25)
        finally:
            O.OnlineLearner.__init__ = init
    return O.demo_new_word(m, D.DEV, day_bytes=32768, lr=1e-3, replay=0.25)


out = {'tag': a.tag, 'size': a.size, 'extra': extra}
for mode in ('frozen', 'plastic'):
    torch.manual_seed(0)
    nw = new_word(mode); nw.pop('samples', None)
    out[mode] = {'new_word': nw, 'sequential': sequential(mode)}
    s, w = out[mode]['sequential'], nw
    print(mode, f"生造词 测试 bpc 学前 {w['test_before']:.3f} → 在线 {w['test_online']:.3f} → 睡后 {w['test_sleep_reset']:.3f}；"
          f"旧数据 {w['old_before']:.3f} → 在线后 {w['old_online']:.3f} → 睡后 {w['old_sleep']:.3f}（提交 {w['sleep']['committed']}）；"
          f"顺序任务 对话 bpc 降 {s['chat_drop']:.3f}，故事旧数据升 {s['old_rise']:.3f}", flush=True)
p = ROOT / f'runs/{time.strftime("%Y%m%d-%H%M%S")}_rmp-online{a.tag}.json'
json.dump(out, open(p, 'w'), ensure_ascii=False, indent=1, default=float)
print('→', p.name)
