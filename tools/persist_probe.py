"""跨会话记忆是否用上持久头：chat2 验证集的跨会话样例，最后一段问答里 Bot 回答字节的 bpc，比较 持久头保留 / 会话边界时持久头也清零。
用法：python tools/persist_probe.py runs/dspn-L_chat2.pt [更多检查点]"""
import json, sys, re, torch, torch.nn.functional as F
sys.path.insert(0, '.')
import dspn_text as D
extra = json.loads(json.load(open('tools/queues/overnight.json', encoding='utf-8-sig'))[0][5])
data = open('data/chat2-valid.txt', 'rb').read(30_000_000)
SEP = b'\n\x1e\n'
# 找 “Hi again!” 之后的问答：取从事实开始到回答结束的整段（含中间的闲聊与边界）
samples = []
for m in re.finditer(rb'Bot: Hello! Nice to see you again\.\n((?:User: [^\n]*\nBot: [^\n]*\n)+)', data):
    end = m.end(); qa = m.group(1)
    start = data.rfind(b'User: ', 0, max(0, data.rfind(SEP, 0, m.start()) - 6000))
    seg = data[start:end]
    if 1000 < len(seg) < 12000:
        samples.append((seg, qa))
    if len(samples) >= 40: break
print('samples', len(samples), 'avg len', sum(len(s) for s, _ in samples) // len(samples))
for ck in sys.argv[1:]:
    model = D.build('dspn', 'L', **extra).to(D.DEV); model.load_state_dict(torch.load(ck, map_location=D.DEV)); model.eval()
    res = {}
    for mode in ('keep', 'wipe'):
        if mode == 'wipe':
            model.hub.persist.zero_()                      # 会话边界时持久头也清空
        tot, n = 0.0, 0
        with torch.no_grad():
            for seg, qa in samples:
                x = torch.tensor([list(seg)], device=D.DEV)
                lg, _ = model(x[:, :-1])
                ce = F.cross_entropy(lg[0], x[0, 1:], reduction='none')
                # 只算最后一段问答里 Bot 回答的字节
                tail = len(qa); m = D.bot_mask(x[:, 1:])[0]; m[:-tail] = 0
                tot += (ce * m).sum().item(); n += m.sum().item()
        res[mode] = round(tot / n / 0.6931, 3)
    print(ck, res)
