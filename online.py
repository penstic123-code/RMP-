"""DSPN 的在线学习、重放池与睡眠巩固（第一版，设计第 3、4 节）。

在线参数：各层 lora_a/lora_b 及中枢里名字含 'lora' 的参数（params='all' 时另加 readout、bias），其余冻结。
文本流按 chunk 处理，状态跨块延续但 detach：每块先记预测损失，再对该块交叉熵反传一次。
"""
import copy, math
import numpy as np
import torch, torch.nn.functional as F


def _detach(x):
    if isinstance(x, torch.Tensor):
        return x.detach()
    if isinstance(x, (tuple, list)):
        return type(x)(_detach(i) for i in x)
    return x                                   # 状态里的整数位置（工作空间、窗口注意力）


def _ids(data, device):
    if isinstance(data, str):
        data = data.encode()
    return torch.from_numpy(np.frombuffer(bytes(data), dtype=np.uint8).astype(np.int64)).to(device)


def _device(model):
    return model.emb.weight.device


def _nll(model, data, chunk, state=None, last=None):
    """不学习地读一段字节。逐字节交叉熵之和（nat）、预测个数、末状态；块间用 last 补上边界处的预测。"""
    ids = _ids(data, _device(model))
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(ids), chunk):
            blk = ids[i:i + chunk]
            seq = blk if last is None else torch.cat([last, blk])
            last = blk[-1:]
            if len(seq) < 2:
                continue
            lg, state = model(seq[None, :-1], state)
            tot += F.cross_entropy(lg[0], seq[1:], reduction='sum').item(); n += len(seq) - 1
    return tot, n, state


def _frozen_copy(model):
    """冻结副本，去掉前向钩子（GRU 适配器钩子引用的是活参数；取副本时适配器恒为恒等，去掉不改变输出）。"""
    t = copy.deepcopy(model).requires_grad_(False)
    for mod in t.modules():
        mod._forward_hooks.clear()
    return t


@torch.no_grad()
def _dream(teacher, prompts, n, temp=1.0):
    """老师从各提示（批）出发自由生成 n 个字节，返回 提示+生成 的字节张量。"""
    lg, st = teacher(prompts)
    out = [prompts]
    for _ in range(n):
        nxt = torch.multinomial(F.softmax(lg[:, -1] / temp, -1), 1)
        out.append(nxt)
        lg, st = teacher(nxt, st)
    return torch.cat(out, 1)


def bpc(model, text_bytes, chunk=256):
    """从零状态读完整段，不学习，返回每字节比特数。"""
    tot, n, _ = _nll(model, text_bytes, chunk)
    return tot / n / math.log(2)


class OnlineLearner:
    def __init__(self, model, lr=1e-3, chunk=64, pool_bytes=65536, priority=True, seed=0,
                 params='lora', replay=0.0, wd=0.0):
        """params='lora'（默认）：只更新低秩偏移，不碰共享读出（E29：改读出是遗忘主因）；'all' 连读出一起改；replay：每次更新混入池中片段的损失权重；
        wd：低秩偏移向 0 的权重衰减（MIRAS 式保留正则，连续版睡眠下调），读出不衰减。"""
        self.model, self.chunk, self.lr = model, chunk, lr
        self.pool_bytes, self.priority = pool_bytes, priority
        self.replay, self.wd = replay, wd
        self.rng = np.random.default_rng(seed)
        self.pool = []                      # [(优先级, 序号, 字节序列)]
        self.pool_size, self.count = 0, 0
        self.morning = _frozen_copy(model)  # 白天学习前的模型：做梦式自蒸馏的老师，每次睡眠提交后更新
        for p in model.parameters():
            p.requires_grad_(False)
        if hasattr(model, 'layers'):                       # DSPN
            low = [p for l in [*model.layers, *model.wlayers, *model.tdlayers] for p in (l.lora_a, l.lora_b)]
            if model.hub is not None:
                low += [p for k, p in model.hub.named_parameters() if 'lora' in k]
            shared = list(model.readout.parameters()) + [model.bias]
        else:                                              # GRU：嵌入输出后接秩 2 适配器（前向钩子），对应 DSPN 的低秩偏移
            e, dev = model.emb.weight.shape[1], _device(model)
            self.ad_a = torch.nn.Parameter(torch.randn(e, 2, device=dev) / math.sqrt(e))
            self.ad_b = torch.nn.Parameter(torch.zeros(2, e, device=dev))
            model.emb.register_forward_hook(lambda m, i, o: o + o @ self.ad_a @ self.ad_b)
            low, shared = [self.ad_a, self.ad_b], list(model.readout.parameters())
        if params == 'lora':
            shared = []
        self.groups = [{'params': low, 'weight_decay': wd}, {'params': shared, 'weight_decay': 0.0}]
        self.online = low + shared
        for p in self.online:
            p.requires_grad_(True)
        self.opt = torch.optim.AdamW(self.groups, lr=lr)
        self.reset_session()

    def reset_session(self):
        self.state, self.last = None, None

    def _add(self, seq, loss):
        """入池：池满时新段优先级高于最低者才替换，同分换最旧（序号最小）。
        均匀对照用随机优先级：保留 k 个最大的独立随机键，等价于均匀蓄水池。"""
        p = loss if self.priority else float(self.rng.random())
        self.count += 1
        item = (p, self.count, seq)
        while self.pool_size + len(seq) > self.pool_bytes:
            lo = min(self.pool, key=lambda t: t[:2])
            if p <= lo[0]:
                return
            self.pool.remove(lo); self.pool_size -= len(lo[2])
        self.pool.append(item); self.pool_size += len(seq)

    def read(self, text_bytes):
        ids = _ids(text_bytes, _device(self.model))
        tot, n = 0.0, 0
        for i in range(0, len(ids), self.chunk):
            blk = ids[i:i + self.chunk]
            seq = blk if self.last is None else torch.cat([self.last, blk])
            self.last = blk[-1:]
            if len(seq) < 2:
                continue
            lg, st = self.model(seq[None, :-1], self.state)
            loss = F.cross_entropy(lg[0], seq[1:])
            tot += loss.item() * (len(seq) - 1); n += len(seq) - 1
            total = loss
            if self.replay and self.pool:                  # 混入池中旧片段（从零状态），抵消只看新内容造成的遗忘
                pick = self.rng.choice(len(self.pool), size=min(4, len(self.pool)), replace=False)
                xy = torch.from_numpy(np.stack([self.pool[i][2] for i in pick]).astype(np.int64)).to(ids.device)
                rl, _ = self.model(xy[:, :-1])
                total = (1 - self.replay) * loss + self.replay * F.cross_entropy(rl.reshape(-1, rl.shape[-1]), xy[:, 1:].reshape(-1))
            self.opt.zero_grad(); total.backward(); self.opt.step()
            self.state = _detach(st)
            if len(seq) == self.chunk + 1:      # 只收满长度的段，巩固时才能成批
                self._add(seq.cpu().numpy(), loss.item())
        return {'bpc_before_update': tot / n / math.log(2) if n else None, 'bytes': len(ids)}

    @torch.no_grad()
    def _refresh_pool(self):
        """用当前模型重算池内片段的损失作为新优先级：已学会的片段让位，学不会的噪声不会凭入池时的高损失永久占位。"""
        if not self.priority or not self.pool:
            return
        xy = torch.from_numpy(np.stack([s[2] for s in self.pool]).astype(np.int64)).to(_device(self.model))
        lg, _ = self.model(xy[:, :-1])
        loss = F.cross_entropy(lg.transpose(1, 2), xy[:, 1:], reduction='none').mean(1).tolist()
        self.pool = [(l, c, s) for l, (_, c, s) in zip(loss, self.pool)]

    def _merge_lora(self):
        with torch.no_grad():
            if not hasattr(self.model, 'layers'):          # GRU：适配器对嵌入向量线性，精确并入嵌入表
                E = self.model.emb.weight
                E.copy_(E + E @ self.ad_a @ self.ad_b)
                self.ad_b.zero_(); self.ad_a.copy_(torch.randn_like(self.ad_a) / math.sqrt(self.ad_a.shape[0]))
                return
            pairs = [(l.W1, l.lora_a, l.lora_b) for l in [*self.model.layers, *self.model.wlayers, *self.model.tdlayers]]
            h = self.model.hub
            if h is not None:                              # 中枢读路径偏移同样并入（设计 1.2 节）
                pairs += [(h.Wq, h.lora_q_a, h.lora_q_b), (h.Wo, h.lora_o_a, h.lora_o_b)]
            for W, a, b in pairs:
                W += a @ b
                b.zero_()
                a.copy_(torch.randn_like(a) / math.sqrt(a.shape[-2]))

    def sleep(self, probe_old, probe_new, steps=200, lr=3e-4, batch=8, old_tol=0.02, anchor=0.0, dream=0.0, n_dream=64,
              distill=0.0, day=None, ce=1.0, win=64, old=None):
        """先把低秩偏移精确并入慢权重（不损失），再在重放池上训 steps 步。
        anchor>0：加 KL(老师‖学生)，老师 = 并入后、重放前的模型，在重放片段上算（E31；但老师已带白天的遗忘，E32 前的第 8 轮显示锚越强旧知识越差）。
        dream>0：做梦式自蒸馏（设计阶段 5 第 1 项）——早晨模型（白天学习前）从池中片段开头自由生成 n_dream 段“梦”，
        在梦上加 dream·KL(早晨模型‖学生)，保住旧分布；池上的交叉熵负责学新内容。
        distill>0：上下文蒸馏（候选 17n1；Snell 等 2022）——老师 = 并入后的模型带着状态读完当天全部对话 day，记下每个位置的预测；
        学生 = 同一模型从零状态只看 win 字节的窗口，在同样位置上学老师的分布。当天靠上下文 / 快记忆知道的事由此进权重。
        old：旧语料字节；给了时锚定 KL 改在旧语料的随机窗口上算（候选 11 §3.3：只在旧数据上拉住，新内容不受锚的阻挡）。
        ce：池上交叉熵的权重（聊天时池里只有用户的话，训它会把模型推向“像用户那样说话”，E44，故聊天用 0）。"""
        m = self.model
        backup = copy.deepcopy(m.state_dict())
        online_backup = [p.detach().clone() for p in self.online]   # GRU 适配器不在 state_dict 里
        flags = [(p, p.requires_grad) for p in m.parameters()]
        old0, new0 = bpc(m, probe_old), bpc(m, probe_new)
        self._merge_lora()
        if not self.pool and not distill:                  # 池空且不蒸馏：只并入
            steps = 0
        oids = _ids(old, _device(m)) if old is not None else None
        if distill:
            dids = _ids(day, _device(m))
            with torch.no_grad():
                day_lg = m(dids[None, :-1])[0][0]          # 老师：带状态读完一整天
            win = min(win, len(dids) - 1)
        teacher = _frozen_copy(m) if anchor else None
        w = np.array([s[0] for s in self.pool]); w = w / w.sum() if self.priority else None
        if dream and steps:                                # 梦从换行自由生成：只代表旧知识（E36：从池中片段续写会教模型忘掉新词）
            prompts = torch.full((n_dream, 1), 10, dtype=torch.long, device=_device(m))
            dreams = _dream(self.morning, prompts, self.chunk)
        # 睡眠只训慢权重；低秩偏移刚清零，留给第二天。嵌入 / 读出不动：改读出是遗忘主因（E29）
        slow = [p for k, p in m.named_parameters() if 'lora' not in k and not k.startswith(('emb.', 'readout.', 'bias', 'norm.'))]
        for p in slow:
            p.requires_grad_(True)
        opt = torch.optim.Adam(slow, lr=lr)
        for _ in range(steps):
            loss = 0.0
            if distill:                                # 学生从零状态看窗口，学老师带全天上下文的分布
                s = self.rng.integers(0, len(dids) - win, size=batch)
                x = torch.stack([dids[i:i + win] for i in s])
                tl = torch.stack([day_lg[i:i + win] for i in s]).flatten(0, 1)
                sl = m(x)[0].flatten(0, 1)
                loss = loss + distill * F.kl_div(F.log_softmax(sl, -1), F.log_softmax(tl, -1), log_target=True, reduction='batchmean')
            if anchor and old is not None:             # 锚在旧语料上：学生不偏离睡前模型对旧数据的预测
                o = torch.stack([oids[i:i + win] for i in self.rng.integers(0, len(oids) - win, size=batch)])
                with torch.no_grad():
                    tl = teacher(o)[0].flatten(0, 1)
                sl = m(o)[0].flatten(0, 1)
                loss = loss + anchor * F.kl_div(F.log_softmax(sl, -1), F.log_softmax(tl, -1), log_target=True, reduction='batchmean')
            if not self.pool or old is not None and not ce:
                opt.zero_grad(); loss.backward(); opt.step(); continue
            pick = self.rng.choice(len(self.pool), size=batch, p=w)
            xy = torch.from_numpy(np.stack([self.pool[i][2] for i in pick]).astype(np.int64)).to(_device(m))
            lg, _ = m(xy[:, :-1])                       # 从零状态预热
            lg = lg.reshape(-1, lg.shape[-1])
            loss = loss + ce * F.cross_entropy(lg, xy[:, 1:].reshape(-1))
            if teacher is not None:
                with torch.no_grad():
                    tl = teacher(xy[:, :-1])[0].reshape(-1, lg.shape[-1])
                loss = loss + anchor * F.kl_div(F.log_softmax(lg, -1), F.log_softmax(tl, -1), log_target=True, reduction='batchmean')
            if dream:
                d = dreams[torch.from_numpy(self.rng.choice(n_dream, size=batch)).to(dreams.device), :-1]
                with torch.no_grad():
                    tl = self.morning(d)[0].flatten(0, 1)
                sl = m(d)[0].flatten(0, 1)
                loss = loss + dream * F.kl_div(F.log_softmax(sl, -1), F.log_softmax(tl, -1), log_target=True, reduction='batchmean')
            opt.zero_grad(); loss.backward(); opt.step()
        for p, f in flags:
            p.requires_grad_(f)
        old1, new1 = bpc(m, probe_old), bpc(m, probe_new)
        ok = new1 <= new0 + 1e-3 and old1 - old0 <= old_tol     # 1e-3：只并入不训练时两者应相等，留浮点误差
        if not ok:
            m.load_state_dict(backup)
            with torch.no_grad():
                for p, b in zip(self.online, online_backup):
                    p.copy_(b)
        else:
            self.morning = _frozen_copy(m)                  # 新的一天从巩固后的模型开始
            self._refresh_pool()
        self.opt = torch.optim.AdamW(self.groups, lr=self.lr)   # 偏移已并入归零（或已回滚），旧的动量作废
        self.reset_session()
        return {'committed': ok, 'old_before': old0, 'old_after': old1, 'new_before': new0, 'new_after': new1}


def _sample(model, prompt, n, temp=0.8):
    ids = torch.tensor([list(prompt.encode())], device=_device(model))
    return bytes(_dream(model, ids, n, temp)[0].tolist()).decode('utf-8', errors='replace')


NAMES_TEACH = ['Tom', 'Lily', 'Max', 'Anna', 'Ben', 'Mia', 'Sam', 'Zoe', 'Leo', 'Eva']
NAMES_TEST = ['Ned', 'Ruth', 'Omar', 'Ida']
NAMES_PROBE = ['Kim', 'Jon', 'Pia']           # 睡眠验收用，与教学、测试名字都不重叠
TEACH_TEMPLATES = [
    "{n} found a blicket in the garden. A blicket is a small red bell that rings when you shake it. {n} shook the blicket and it rang.\n",
    "\"What is a blicket?\" asked {n}. \"A blicket is a small red bell,\" said Mom. \"It rings when you shake it.\"\n",
    "{n} was sad. Then {n} shook the blicket, the small red bell, and it rang. {n} smiled.\n",
    "Every blicket is a small red bell. {n} has a blicket. When {n} shakes it, the blicket rings.\n",
]
TEST_TEMPLATES = [
    "One day {n} took the blicket to school. The small red bell rang when {n} shook the blicket, and everyone smiled.\n",
    "{n} lost the blicket. \"My blicket!\" cried {n}. A blicket is a bell, and it rings when you shake it.\n",
]


def demo_new_word(model, device, day_bytes=0, sleep=None, **learner):
    """教模型一个生造词 blicket，报告学前/在线学后/会话重置后/睡眠巩固后重置的测试 bpc 与旧数据 bpc。
    day_bytes>0 时先在线读一段普通故事（与旧数据探针不重叠），像平常的一天，顺便填充重放池。"""
    import dspn_text
    model.to(device)
    teach = ''.join(t.format(n=n) for n in NAMES_TEACH for t in TEACH_TEMPLATES)          # 40 个变体
    test = ''.join(t.format(n=n) for n in NAMES_TEST for t in TEST_TEMPLATES)
    assert not set(teach.splitlines()) & set(test.splitlines())
    val = dspn_text.load_bytes('tiny')[1]
    old = bytes(val[:4096])
    L = OnlineLearner(model, **learner)
    res = {'learner': learner, 'day_bytes': day_bytes, 'test_before': bpc(model, test), 'old_before': bpc(model, old)}
    if day_bytes:
        L.read(bytes(val[8192:8192 + day_bytes]))
        L.reset_session()
        res['old_after_day'] = bpc(model, old)
    L.read(teach)
    tot, n, _ = _nll(model, test, 256, L.state, L.last)                                   # 教学上下文仍在状态里
    res['test_online'] = tot / n / math.log(2)
    L.reset_session()
    res['test_reset'] = bpc(model, test)
    res['old_online'] = bpc(model, old)
    res['sleep_args'] = sleep or {}
    probe_new = ''.join(t.format(n=n) for n in NAMES_PROBE for t in TEACH_TEMPLATES)   # 没进池、没训练过
    res['sleep'] = L.sleep(bytes(val[4096:6144]), probe_new, **(sleep or {}))         # 验收探针与报告用的 old、test 不相交
    res['test_sleep_reset'] = bpc(model, test)
    res['old_sleep'] = bpc(model, old)
    res['samples'] = []
    for prefix in ['Tom found a blicket', 'A blicket is', 'Mia lost her']:
        s = _sample(model, prefix, 200)
        res['samples'].append((s, 'blicket' in s[len(prefix):]))
    return res


if __name__ == '__main__':
    import dspn_text
    m = dspn_text.build('dspn', 'S').to(dspn_text.DEV)
    m.load_state_dict(torch.load(dspn_text.ROOT / 'runs/dspn-S.pt', map_location=dspn_text.DEV))
    for k, v in demo_new_word(m, dspn_text.DEV).items():
        print(k, v)
