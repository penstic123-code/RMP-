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
import numpy as np
import torch
import dspn_text as D
import online as O

ap = argparse.ArgumentParser()
ap.add_argument('--tag', default='_wspread'); ap.add_argument('--size', default='S')
ap.add_argument('--extra', default='{}'); ap.add_argument('--seq_bytes', type=int, default=65536)
ap.add_argument('--modes', default='frozen,plastic'); ap.add_argument('--sleep', default='{}', help='睡眠参数 JSON，如 {"dream": 1.0, "rewire": false}')
ap.add_argument('--no_seq', action='store_true', help='只跑生造词'); ap.add_argument('--name', default='')
ap.add_argument('--learner', default='{}', help='OnlineLearner 额外参数 JSON，如 {"allocate": 1}')
ap.add_argument('--bg_data', default='tiny', help='SMF 背景频率统计语料（取训练集尾部，与验证探针彻底隔离）')
ap.add_argument('--old_data', default='tiny', help='顺序任务旧数据探针语料（若有 .src 文件则按各来源分别建独立探针）')
ap.add_argument('--seq_data', default='chat3', help='顺序任务新数据语料')
ap.add_argument('--eval_every', type=int, default=16384, help='顺序任务每隔多少字节记录一次 chat_drop -> old_rise 同收益曲线（0=仅测结尾）')
ap.add_argument('--no_seq_sleep', action='store_true', help='顺序任务跳过睡眠巩固评估')
a = ap.parse_args()
extra = json.loads(a.extra)
LK = {'lr': 1e-3, 'replay': 0.25, **json.loads(a.learner)}
if LK.get('smf_k', 0) > 0 and 'bg_bytes' not in LK:
    LK['bg_bytes'] = bytes(D.load_bytes(a.bg_data)[0][-65536:])
LAST = {}
_init0 = O.OnlineLearner.__init__
def _keep(self, model, **kw):                              # 记下最近建的学习器（读分配日志）
    _init0(self, model, **kw); LAST['L'] = self
O.OnlineLearner.__init__ = _keep


def load(mode):
    m = D.build('dspn', a.size, **extra).to(D.DEV)
    r = m.load_state_dict(torch.load(ROOT / f'runs/dspn-{a.size}{a.tag}.pt', map_location=D.DEV), strict=False)
    assert not r.unexpected_keys and all(k.endswith(('.moved', '.active', '.ogain')) for k in r.missing_keys), r   # 早期检查点没有 moved（只在移波时用）
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


def _old_probes(data_name, probe_bytes=4096, sleep_bytes=2048):
    """返回 ({来源名: 评估探针字节}, 睡眠验收探针字节)，二者完全不相交。
    若数据集有 .src 文件（如 task4 / task3），每个旧来源独立取探针，避免前 4KB 只覆盖单一来源。"""
    val = D.load_bytes(data_name)[1]
    val_rel = D.DATASETS[data_name][1]
    srcf = ROOT / val_rel.replace('.txt', '.src') if val_rel else None
    if srcf is not None and srcf.exists():
        src = np.fromfile(srcf, np.uint8)
        probes, sleep_parts = {}, []
        for s in np.unique(src):
            idx = np.flatnonzero(src == s)
            probes[f'src{int(s)}'] = bytes(val[idx[:probe_bytes]])
            sl = idx[probe_bytes:probe_bytes + sleep_bytes]
            if len(sl) >= 2:
                sleep_parts.append(bytes(val[sl]))
        return probes, b''.join(sleep_parts)
    return {'all': bytes(val[:probe_bytes])}, bytes(val[probe_bytes:probe_bytes + sleep_bytes])


def sequential(mode):
    m = load(mode)
    probes, sleep_old = _old_probes(a.old_data)
    ctr, cva = D.load_bytes(a.seq_data)
    new_probe = bytes(cva[:4096])
    sleep_new = bytes(cva[4096:6144])
    L = O.OnlineLearner(m, **LK)
    if mode == 'plastic':
        unfreeze_router(m, L)
    old_before = {k: O.bpc(m, v) for k, v in probes.items()}
    chat_before = O.bpc(m, new_probe)
    curve = []
    step_bytes = a.eval_every if a.eval_every > 0 else a.seq_bytes
    for pos in range(0, a.seq_bytes, step_bytes):
        end = min(pos + step_bytes, a.seq_bytes)
        L.read(bytes(ctr[pos:end]))
        L.reset_session()
        ob = {k: O.bpc(m, v) for k, v in probes.items()}
        cb = O.bpc(m, new_probe)
        rises = {k: ob[k] - old_before[k] for k in probes}
        curve.append({
            'bytes': end, 'chat_bpc': cb, 'chat_drop': chat_before - cb,
            'old_bpc': ob, 'old_rise': rises, 'worst_old_rise': max(rises.values()),
            'mean_old_rise': sum(rises.values()) / len(rises),
        })
    last = curve[-1]
    r = {
        'old_before': sum(old_before.values()) / len(old_before),
        'old_after': sum(last['old_bpc'].values()) / len(last['old_bpc']),
        'old_rise': last['mean_old_rise'],
        'worst_old_rise': last['worst_old_rise'],
        'old_by_src_before': old_before,
        'old_by_src_after': last['old_bpc'],
        'old_rise_by_src': last['old_rise'],
        'chat_before': chat_before,
        'chat_after': last['chat_bpc'],
        'chat_drop': last['chat_drop'],
        'curve': curve,
    }
    if not a.no_seq_sleep:
        r['sleep'] = L.sleep(sleep_old, sleep_new, **json.loads(a.sleep))
        oslp = {k: O.bpc(m, v) for k, v in probes.items()}
        cslp = O.bpc(m, new_probe)
        srises = {k: oslp[k] - old_before[k] for k in probes}
        r.update(
            old_after_sleep=sum(oslp.values()) / len(oslp),
            old_rise_sleep=sum(srises.values()) / len(srises),
            worst_old_rise_sleep=max(srises.values()),
            old_by_src_sleep=oslp,
            old_rise_by_src_sleep=srises,
            chat_after_sleep=cslp,
            chat_drop_sleep=chat_before - cslp,
        )
    return r


def new_word(mode):
    m = load(mode)
    if mode == 'plastic':                                  # demo_new_word 自建学习器；先打补丁让它带上 W_r
        init = O.OnlineLearner.__init__
        def patched(self, model, **kw):
            init(self, model, **kw); unfreeze_router(model, self)
        O.OnlineLearner.__init__ = patched
        try:
            return O.demo_new_word(m, D.DEV, day_bytes=32768, sleep=json.loads(a.sleep), **LK)
        finally:
            O.OnlineLearner.__init__ = init
    return O.demo_new_word(m, D.DEV, day_bytes=32768, sleep=json.loads(a.sleep), **LK)


out = {'tag': a.tag, 'size': a.size, 'extra': extra, 'sleep': json.loads(a.sleep), 'learner': {k: v for k, v in LK.items() if k != 'bg_bytes'}}
for mode in a.modes.split(','):
    torch.manual_seed(0)
    nw = new_word(mode); nw.pop('samples', None); nw['alloc_log'] = getattr(LAST.get('L'), 'alloc_log', None)
    out[mode] = {'new_word': nw, 'sequential': None if a.no_seq else sequential(mode)}
    s, w = out[mode]['sequential'] or {'chat_drop': float('nan'), 'old_rise': float('nan'), 'worst_old_rise': float('nan')}, nw
    slp_str = (f" → 睡后降 {s['chat_drop_sleep']:.3f}/最差升 {s['worst_old_rise_sleep']:.3f}(提交 {s['sleep']['committed']})"
               if 'sleep' in s else '')
    print(mode, f"生造词 测试 bpc 学前 {w['test_before']:.3f} → 在线 {w['test_online']:.3f} → 睡后 {w['test_sleep_reset']:.3f}；"
          f"旧数据 {w['old_before']:.3f} → 在线后 {w['old_online']:.3f} → 睡后 {w['old_sleep']:.3f}（提交 {w['sleep']['committed']}）；"
          f"顺序任务 对话降 {s['chat_drop']:.3f}，旧数据均升 {s['old_rise']:.3f}/最差升 {s.get('worst_old_rise', s['old_rise']):.3f}{slp_str}", flush=True)
p = ROOT / f'runs/{time.strftime("%Y%m%d-%H%M%S")}_rmp-online{a.tag}{a.name}.json'
json.dump(out, open(p, 'w'), ensure_ascii=False, indent=1, default=lambda x: x.decode('utf-8', 'replace') if isinstance(x, (bytes, bytearray)) else float(x))
print('→', p.name)
