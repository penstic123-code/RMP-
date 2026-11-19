"""用 dspn_text.train 训练 TSPH，不改 dspn_text.py：替换 dspn_text.build，kind 'dspn' 且 extra 含 {"arch": "tsph"} 时建 TSPH。
  python tools/train_tsph.py --size S --data tiny --iters 3000          # --extra 里其余键覆盖 tsph.SIZES 配置
名字固定带 -tsph 后缀（dspn-S-tsph…），免得覆盖 DSPN 的检查点。
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dspn_text, tsph

_build = dspn_text.build


def build(kind, size='S', **extra):
    if kind == 'dspn' and extra.get('arch') == 'tsph':
        return tsph.build_tsph(size, **{k: v for k, v in extra.items() if k != 'arch'})
    return _build(kind, size, **extra)


dspn_text.build = build

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', default='S', choices=list(tsph.SIZES))
    ap.add_argument('--data', default='tiny', choices=list(dspn_text.DATASETS))
    ap.add_argument('--iters', type=int, default=3000)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--T', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', default='')
    ap.add_argument('--extra', default='{}')
    a = ap.parse_args()
    extra = {'arch': 'tsph', **json.loads(a.extra)}
    dspn_text.train('dspn', a.size, a.data, a.iters, a.batch, a.T, a.lr, a.seed, '-tsph' + a.tag, extra)
