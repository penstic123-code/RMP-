"""v2 预训练数据（英文混合，plan/fast-clean.md）：从 HuggingFace 流式下载、转纯文本、按文档交错混合。

来源与定额（字节）：TinyStories（已有，全用）、Cosmopedia 的 stories / wikihow / khanacademy、Simple English Wikipedia、SODA 对话。
对话转成 "User: … / Bot: …"（与聊天、海马读回同一格式，第 20.1 节）。文档之间用会话边界字节 SEP=30 分隔：
新文档 = 新上下文，c、快记忆、中枢、广播都清空。

用法：python tools/prepare_data.py download   # 各来源写 data/src-*.txt（可中断重跑，已完成的跳过）
      python tools/prepare_data.py mix        # 交错混合成 data/mix-train.txt、data/mix-valid.txt（每来源末 1% 作验证）
"""
import random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
SEP = '\x1e'
GB = 10 ** 9
SOURCES = {   # 名字: (数据集, 配置, 切分, 定额字节, 转换函数名)
    'cosmo_stories': ('HuggingFaceTB/cosmopedia', 'stories', 'train', int(1.2 * GB), 'text'),
    'cosmo_wikihow': ('HuggingFaceTB/cosmopedia', 'wikihow', 'train', int(0.5 * GB), 'text'),
    'cosmo_khan': ('HuggingFaceTB/cosmopedia', 'khanacademy', 'train', int(0.5 * GB), 'text'),
    'simplewiki': ('wikimedia/wikipedia', '20231101.simple', 'train', int(1.0 * GB), 'wiki'),
    'soda': ('allenai/soda', None, 'train', int(0.6 * GB), 'soda'),
}


def conv(kind, ex):
    if kind == 'text':
        return ex['text']
    if kind == 'wiki':
        return ex['title'] + '\n\n' + ex['text']
    if kind == 'soda':                                     # 两人对话：第一个说话的人记为 User，另一个记为 Bot
        first = ex['speakers'][0] if ex['speakers'] else None
        return '\n'.join(f"{'User' if s == first else 'Bot'}: {u}" for s, u in zip(ex['speakers'], ex['dialogue']))
    raise ValueError(kind)


def clean(t):
    return t.replace(SEP, ' ').replace('\r\n', '\n').strip()


def download():
    from datasets import load_dataset
    for name, (ds, cfg, split, quota, kind) in SOURCES.items():
        out = DATA / f'src-{name}.txt'
        if out.exists() and out.stat().st_size >= 0.95 * quota:
            print(name, '已完成，跳过', flush=True); continue
        n = 0
        with open(out, 'w', encoding='utf-8', newline='\n') as f:
            for ex in load_dataset(ds, cfg, split=split, streaming=True):
                t = clean(conv(kind, ex))
                if not t:
                    continue
                f.write(t + '\n' + SEP)
                n += len(t.encode()) + 2
                if n >= quota:
                    break
        print(name, f'{n / GB:.2f} GB', flush=True)


def docs(path, sep):
    """逐个文档读大文件（不全读进内存）。"""
    buf = ''
    with open(path, encoding='utf-8', errors='replace', newline='') as f:
        while True:
            block = f.read(1 << 22)
            if not block:
                break
            buf += block
            parts = buf.split(sep)
            buf = parts.pop()
            yield from (p.strip() for p in parts if p.strip())
    if buf.strip():
        yield buf.strip()


def mix(valid_frac=0.01, seed=0):
    srcs = {'tinystories': (DATA / 'TinyStoriesV2-GPT4-train.txt', '<|endoftext|>')}
    srcs.update({n: (DATA / f'src-{n}.txt', SEP) for n in SOURCES})
    sizes = {n: p.stat().st_size for n, (p, _) in srcs.items()}
    rng = random.Random(seed)
    its = {n: docs(p, s) for n, (p, s) in srcs.items()}
    left = dict(sizes)
    tr = open(DATA / 'mix-train.txt', 'w', encoding='utf-8', newline='\n')
    va = open(DATA / 'mix-valid.txt', 'w', encoding='utf-8', newline='\n')
    written = {n: 0 for n in srcs}
    while its:                                             # 按剩余字节数加权抽来源：各来源均匀铺满整个文件
        names = list(its)
        n = rng.choices(names, weights=[max(left[k], 1) for k in names])[0]
        try:
            d = next(its[n])
        except StopIteration:
            del its[n]; continue
        b = len(d.encode()) + 2
        left[n] -= b
        to_valid = written[n] > (1 - valid_frac) * sizes[n]   # 每来源最后 1% 作验证
        (va if to_valid else tr).write(d + '\n' + SEP)
        written[n] += b
    tr.close(); va.close()
    print({n: f'{w / GB:.2f} GB' for n, w in written.items()})


if __name__ == '__main__':
    {'download': download, 'mix': mix}[sys.argv[1]]()
