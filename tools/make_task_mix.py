"""RMP 按任务分工（第 23.11 节）的数据：四种来源（故事、wikiHow 说明文、简明维基、SODA 对话）各切 4KB 块随机交错。
训练每源 40MB、验证每源 1MB（取源文件末尾，不与训练重叠）。另存逐字节来源标签（uint8，0–3）供分工指标用。
输出 data/task4-{train,valid}.txt 与 .src。用法：python tools/make_task_mix.py
"""
import argparse
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
SRC = ['TinyStoriesV2-GPT4-train.txt', 'src-cosmo_wikihow.txt', 'src-simplewiki.txt', 'src-soda.txt']
ap = argparse.ArgumentParser()
ap.add_argument('--n_src', type=int, default=4, help='取前几个来源（3=task3 留出 SODA 对话，4=task4 全部）')
ap.add_argument('--out', default='', help='输出前缀（默认 task{n_src}）')
args = ap.parse_args()
out = args.out or f'task{args.n_src}'
BLK, TR, VA = 4096, 40 << 20, 1 << 20
rng = np.random.default_rng(0)
for split, size in (('train', TR), ('valid', VA)):
    blocks = []
    for s, f in enumerate(SRC[:args.n_src]):
        a = np.memmap(ROOT / 'data' / f, dtype=np.uint8, mode='r')
        lo = 0 if split == 'train' else len(a) - VA
        seg = np.asarray(a[lo:lo + size]); n = len(seg) // BLK
        blocks += [(s, seg[i * BLK:(i + 1) * BLK]) for i in range(n)]
    order = rng.permutation(len(blocks))
    txt = np.concatenate([blocks[i][1] for i in order]); lab = np.concatenate([np.full(BLK, blocks[i][0], np.uint8) for i in order])
    txt.tofile(ROOT / f'data/{out}-{split}.txt'); lab.tofile(ROOT / f'data/{out}-{split}.src')
    print(out, split, len(txt), np.bincount(lab) / len(lab))
