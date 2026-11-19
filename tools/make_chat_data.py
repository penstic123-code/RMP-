"""生成对话训练数据：SODA 双人对话 + 模板“记忆对话”，写成 data/chat-train.txt、data/chat-valid.txt。

格式：每轮一行 "User: …" / "Bot: …"，对话之间是会话边界字节 \\x1e（单独一行）。SODA 的第一个说话人当 User。
记忆对话：User 先说出几件自己的事，中间插几轮闲聊，最后提问，Bot 按前文回答；名字由随机音节拼成，逼模型从上下文复制而不是背常见名。
用法：python tools/make_chat_data.py [--memory_frac 0.25]
"""
import argparse, random
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SYL = ['ba', 'ko', 'mi', 'ra', 'ne', 'lu', 'to', 'si', 'da', 'fe', 'gi', 'po', 'zu', 'ka', 'li', 'mo', 'ti', 'vo', 'ja', 'nu']


def name(rng):
    return ''.join(rng.choice(SYL) for _ in range(rng.randint(2, 3))).capitalize()


FACTS = [  # (陈述模板, Bot 回应模板, 提问, 回答模板, 取值函数)
    ('My name is {v}.', 'Nice to meet you, {v}!', 'What is my name?', 'Your name is {v}.', name),
    ('I have a dog called {v}.', '{v} is a lovely name for a dog.', "What is my dog's name?", "Your dog's name is {v}.", name),
    ('I have a cat named {v}.', 'I like the name {v}.', "What is my cat's name?", "Your cat is called {v}.", name),
    ('My favorite color is {v}.', '{v} is a nice color.', 'What is my favorite color?', 'Your favorite color is {v}.',
     lambda r: r.choice(['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'black', 'white', 'brown'])),
    ('I live in a town called {v}.', '{v} sounds like a nice place.', 'Where do I live?', 'You live in {v}.', name),
    ('I am {v} years old.', '{v} is a good age.', 'How old am I?', 'You are {v} years old.', lambda r: str(r.randint(5, 90))),
    ('My best friend is {v}.', 'It is good to have a friend like {v}.', 'Who is my best friend?', 'Your best friend is {v}.', name),
    ('I like to eat {v}.', 'Yum, {v} is tasty.', 'What do I like to eat?', 'You like to eat {v}.',
     lambda r: r.choice(['apples', 'pizza', 'rice', 'cake', 'soup', 'bread', 'noodles', 'cookies', 'fish', 'carrots'])),
]
SMALL = [('How are you today?', 'I am good, thank you. How about you?'), ('I am a bit tired.', 'You should get some rest.'),
         ('It is sunny outside.', 'That sounds like a nice day.'), ('I went to the park.', 'Did you have fun at the park?'),
         ('Yes, it was fun.', 'I am glad you had a good time.'), ('I read a book last night.', 'What was the book about?'),
         ('It was about a dragon.', 'Dragons are exciting!'), ('Do you like music?', 'Yes, I like music a lot.'),
         ('I made a cake.', 'Wow, that sounds delicious.'), ('It is raining.', 'Do not forget your umbrella.')]


SEP = '\x1e'   # 会话边界字节（dspn_text.SEP）：模型读到时清空短时状态，只留中枢持久头


def memory_dialogue(rng, soda):
    """一半在同一会话里提问；另一半跨会话：说完事实 → 会话边界 →（一半概率插一段无关对话 + 边界）→ 再见面时提问。"""
    facts = rng.sample(FACTS, rng.randint(1, 3))
    vals = [f[4](rng) for f in facts]
    lines = []
    for (say, reply, *_), v in zip(facts, vals):
        lines += [f'User: {say.format(v=v)}', f'Bot: {reply.format(v=v)}']
    for u, b in rng.sample(SMALL, rng.randint(0, 4)):
        lines += [f'User: {u}', f'Bot: {b}']
    if rng.random() < 0.5:
        lines.append(SEP)
        if rng.random() < 0.5:
            lines += [rng.choice(soda), SEP]
        lines += ['User: Hi again!', 'Bot: Hello! Nice to see you again.']
    for i in rng.sample(range(len(facts)), len(facts)):
        lines += [f'User: {facts[i][2]}', f'Bot: {facts[i][3].format(v=vals[i])}']
    return '\n'.join(lines)


def soda_dialogues(path):
    df = pd.read_parquet(path, columns=['dialogue', 'speakers'])
    out = []
    for turns, spk in zip(df['dialogue'], df['speakers']):
        if len(set(spk)) != 2 or len(turns) < 2:
            continue
        roles = {spk[0]: 'User', next(s for s in spk if s != spk[0]): 'Bot'}
        out.append('\n'.join(f'{roles[s]}: {t.strip()}' for s, t in zip(spk, turns)))
    return out


def build(split, memory_frac, seed):
    rng = random.Random(seed)
    dl = soda_dialogues(ROOT / f'data/soda/{split}.parquet')
    n_mem = int(len(dl) * memory_frac / (1 - memory_frac))
    soda = list(dl)
    dl += [memory_dialogue(rng, soda) for _ in range(n_mem)]
    rng.shuffle(dl)
    out = ROOT / f'data/chat-{split}.txt'
    out.write_text(f'\n{SEP}\n'.join(dl) + '\n', encoding='utf-8')     # 对话之间是会话边界
    print(out.name, len(dl), '段对话，其中记忆对话', n_mem, '，', out.stat().st_size, '字节')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--memory_frac', type=float, default=0.25)
    a = ap.parse_args()
    build('valid', a.memory_frac, 1)
    build('train', a.memory_frac, 0)
