"""命令行聊天 demo：边聊边学（只从用户的话学），/sleep 睡眠巩固，/new 开新会话（清空快记忆），/save 存权重。

用法：
  python tools/chat.py runs/dspn-L_chat.pt --size L --extra '{...}'              # 交互
  python tools/chat.py runs/dspn-L_chat.pt --size L --extra '{...}' --script s.txt   # 脚本：每行一句输入或命令，记录存 runs/
"""
import argparse, json, sys, time
from pathlib import Path
import torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dspn_text as D
from online import OnlineLearner, _ids
from episodic import Episodic


class Chat:
    def __init__(self, model, temp=0.7, max_reply=200, episodic=False, **learner):
        self.model, self.temp, self.max_reply = model, temp, max_reply
        self.E = Episodic() if episodic else None            # 海马式情景记忆：跨会话按线索重现以前说过的事
        self.L = OnlineLearner(model, **learner)
        self.heard = []                                    # 本次运行里用户说过的话（睡眠验收的新内容探针）
        self.day = []                                      # 上次睡眠以来读过的全部字节（用户的话与模型的回答），睡眠时蒸馏
        va = D.load_bytes('chat2')[1]
        self.probe_old = bytes(va[:2048])                   # 睡眠验收：旧对话
        self.old = bytes(va[4096:4096 + 65536])            # 睡眠锚定用的旧对话（与验收探针不重叠）

    @torch.no_grad()
    def _advance(self, data):
        """不学习地读入（Bot 前缀、模型自己的回答）：只推进状态。"""
        self.day.append(bytes(data))
        ids = _ids(data, D.DEV)                            # 与 OnlineLearner 同约定：last 是还没输入模型的最后一个字节
        seq = ids if self.L.last is None else torch.cat([self.L.last, ids])
        if len(seq) > 1:
            _, self.L.state = self.model(seq[None, :-1], self.L.state)
        self.L.last = ids[-1:]

    @torch.no_grad()
    def _reply(self):
        self._advance(b'Bot:')
        lg, st = self.model(self.L.last[None], self.L.state)
        out = []
        for _ in range(self.max_reply):
            nxt = torch.multinomial(F.softmax(lg[0, -1] / self.temp, -1), 1)
            if nxt.item() == 10:
                break
            out.append(nxt.item())
            lg, st = self.model(nxt.view(1, 1), st)
        self.L.state, self.L.last = st, torch.tensor([10], device=D.DEV)   # 换行已作为输入读入
        self.day.append(bytes(out) + b'\n')
        return bytes(out).decode('utf-8', errors='replace').strip()

    def turn(self, line):
        if line == '/new':                                 # 读入会话边界字节：清短时状态，中枢持久头保留（与训练一致）
            self._advance(bytes([D.SEP, 10]))
            if self.E:
                self.E.new_session()
            return '（新会话：短时记忆已清空，持久记忆与权重保留）'
        if line == '/reset':
            self.L.reset_session(); return '（全部状态清零，只剩权重）'
        if line == '/sleep':
            day = b''.join(self.day); self.day = []
            # 上下文蒸馏 + 锚定，不在用户的话上做交叉熵（E44：那样把模型推向“像用户那样说话”）
            r = self.L.sleep(self.probe_old, day[-4096:] or b'\n', anchor=1.0, lr=1e-4, distill=1.0, day=day[-8192:] or b'\n\n', ce=0.0, old=self.old)
            return f"（睡眠：{'已巩固' if r['committed'] else '回滚'}，旧对话 bpc {r['old_before']:.3f}→{r['old_after']:.3f}）"
        if self.E:                                         # 重现：最相关的放最后，离问题最近
            for r in reversed(self.E.recall(line)):
                self._advance(f'User: {r}
Bot: OK.
'.encode())
            self.E.write(line)
        self.heard.append(f'User: {line}')
        self.day.append(f'User: {line}\n'.encode())
        self.L.read(f'User: {line}\n'.encode())          # 学习只发生在这里
        return self._reply()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt'); ap.add_argument('--size', default='L'); ap.add_argument('--extra', default='{}')
    ap.add_argument('--learner', default='{"lr": 3e-3, "replay": 0.25}')
    ap.add_argument('--temp', type=float, default=0.7)
    ap.add_argument('--episodic', action='store_true', help='开海马式情景记忆（跨会话按线索重现）')
    ap.add_argument('--script', default=None, help='每行一句输入或命令（/new /sleep），跑完把记录存进 runs/')
    a = ap.parse_args()
    torch.manual_seed(0)
    model = D.build('dspn', a.size, **json.loads(a.extra)).to(D.DEV)
    model.load_state_dict(torch.load(D.ROOT / a.ckpt, map_location=D.DEV)); model.eval()
    chat = Chat(model, temp=a.temp, episodic=a.episodic, **json.loads(a.learner))
    lines = Path(a.script).read_text(encoding='utf-8').splitlines() if a.script else None
    log = []
    while True:
        line = (lines.pop(0) if lines else None) if a.script else input('你: ').strip()
        if line is None or line == '/quit':
            break
        if line == '/save':
            torch.save(model.state_dict(), D.ROOT / a.ckpt.replace('.pt', '_learned.pt'))
            if chat.E:
                chat.E.save(D.ROOT / a.ckpt.replace('.pt', '_episodic.json'))
            print('（已存）'); continue
        r = chat.turn(line)
        print(f'User: {line}\nBot: {r}' if not line.startswith('/') else r, flush=True)
        log.append((line, r))
    if a.script:
        out = D.ROOT / f'runs/{time.strftime("%Y%m%d-%H%M%S")}_chat-{Path(a.ckpt).stem}{"_epi" if a.episodic else ""}.json'
        json.dump({'ckpt': a.ckpt, 'extra': json.loads(a.extra), 'learner': json.loads(a.learner), 'episodic': a.episodic, 'log': log},
                  open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print('→', out.name)
