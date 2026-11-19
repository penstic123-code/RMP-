"""海马式情景记忆（聊天用，不改模型权重）：互补学习系统里的“快速情景存储 + 按线索重现”。

- 写入：用户的陈述句（不存问句）原样存一条情景，记下来自哪次会话。跨会话保留，可存盘。
- 索引：每条情景的稀疏特征 = 词内字符三元组（类似齿状回的模式分离：稀疏、重叠小）；
  特征按在库中出现的稀有度加权（IDF），常见特征几乎不起作用。
- 重现：新的一句话来了，按特征重合度从“以前的会话”里找最相关的几条，读回模型上下文（模式补全 → 皮层重现），
  模型再用会话内的复制能力作答。今天的会话不重现：它本来就在模型状态里。
"""
import json, math, re
from collections import Counter


def feats(text):
    words = re.findall(r"[a-z]+", text.lower())
    return {f'{w[i:i + 3]}' for w in words if len(w) >= 3 for i in range(len(w) - 2)} | {w for w in words if len(w) >= 3}


class Episodic:
    def __init__(self, top=2, min_score=1.5):
        self.items, self.df, self.session = [], Counter(), 0
        self.top, self.min_score = top, min_score

    def new_session(self):
        self.session += 1

    def write(self, line):
        if line.strip().endswith('?'):                       # 问句不是要记的事
            return
        f = feats(line)
        self.items.append((line, self.session, f)); self.df.update(f)

    def recall(self, line):
        q, n = feats(line), len(self.items)
        idf = lambda f: math.log(1 + max(n, 20) / self.df[f])     # 库小时按至少 20 条算，免得稀有度被低估
        scored = [(sum(idf(f) for f in q & fi) / math.sqrt(len(fi) + 1), text)
                  for text, s, fi in self.items if s != self.session]
        return [t for sc, t in sorted(scored, reverse=True)[:self.top] if sc >= self.min_score]

    def save(self, path):
        json.dump([(t, s) for t, s, _ in self.items], open(path, 'w', encoding='utf-8'), ensure_ascii=False)
