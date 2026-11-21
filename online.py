"""DSPN 的在线学习、重放池与睡眠巩固（第一版，设计第 3、4 节）。

在线参数（二选一，fast-clean B4 比较后留一个）：
- params='lora'：各层 MLP 的低秩偏移 lora_a/lora_b（'all' 另加 readout、bias），反传块内梯度。
- params='eprop'：各层慢通道 c 的输入与门（B、G、b_a）用 e-prop 学习。c 是对角递推，资格迹
  e ← a⊙e + ∂c_t/∂θ（当拍）跨块延续，等于沿时间反传的精确梯度；学习信号 δ = ∂L/∂c 只取当拍（c 设为叶子，
  不经上层的 c 回传），更新 = Σ δ⊙e。不存历史、不反传时间（Zucchet 等 2023），三因子规则的第三因子是 read 的 gate（新奇信号）。
中枢不再有在线偏移。
文本流按 chunk 处理，状态跨块延续但 detach：每块先记预测损失，再对该块交叉熵反传一次。
"""
import copy, math
import numpy as np
import torch, torch.nn.functional as F
from dspn_text import scan


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


def _cores(model):
    """各层里真正的模块层（Layer）：路由模块池 RMP 的池层把模块放在 .core 里。"""
    return [getattr(l, 'core', l) for l in model.layers]


def _pools(model):
    """RMP 的池层（有地址与 plastic 开关）。DSPN 没有。"""
    return [l for l in model.layers if hasattr(l, 'plastic')]


def _frozen_copy(model):
    return copy.deepcopy(model).requires_grad_(False)


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


def _masked_step(opt, masked_params, keep_vec, zero_params=()):
    """严格按模块掩码执行优化步：未选中模块梯度置 0、步后原样恢复参数并清零其 AdamW 一阶动量，消除动量与权重衰减漏改。"""
    keep_b = keep_vec.bool()
    prev_m, prev_z = [], []
    with torch.no_grad():
        for p in masked_params:
            m = keep_b.view(-1, *[1] * (p.dim() - 1))
            if p.grad is not None:
                p.grad.mul_(m)
            prev_m.append((p, m, p.data.clone()))
        for p in zero_params:
            if p.grad is not None:
                p.grad.zero_()
            prev_z.append((p, p.data.clone()))
    opt.step()
    with torch.no_grad():
        for p, m, old in prev_m:
            p.data.copy_(torch.where(m, p.data, old))
            st = opt.state.get(p)
            if st and 'exp_avg' in st:
                st['exp_avg'].mul_(m)
        for p, old in prev_z:
            p.data.copy_(old)
            st = opt.state.get(p)
            if st and 'exp_avg' in st:
                st['exp_avg'].zero_()


def bpc(model, text_bytes, chunk=256):
    """从零状态读完整段，不学习，返回每字节比特数。"""
    tot, n, _ = _nll(model, text_bytes, chunk)
    return tot / n / math.log(2)


class OnlineLearner:
    def __init__(self, model, lr=1e-3, chunk=64, pool_bytes=65536, priority=True, seed=0,
                 params='lora', replay=0.0, wd=0.0, allocate=0, novel_ratio=1.25, warm_chunks=16, top_pos=4, spare_bias=-0.2, alloc_mode='new',
                 freeze_enc=False, spare_ogain=0.2, smf_k=0, smf_eps=1e-2, bg_bytes=None):
        """params='lora'（默认）：只更新低秩偏移，不碰共享读出（E29：改读出是遗忘主因）；'all' 连读出一起改；replay：每次更新混入池中片段的损失权重；
        wd：低秩偏移向 0 的权重衰减（MIRAS 式保留正则，连续版睡眠下调），读出不衰减。
        freeze_enc=True：带池模型在线时冻结编码层，避免编码特征漂移破坏固定路由；spare_ogain：新启用空闲模块的初始增益（>0 以免截断梯度）。
        smf_k>0：SMF 式相对背景选模块，每块只更新 cur_freq/(bg_freq+smf_eps) 最高的 smf_k 个模块。"""
        self.model, self.chunk, self.lr = model, chunk, lr
        self.pool_bytes, self.priority = pool_bytes, priority
        self.replay, self.wd = replay, wd
        self.freeze_enc, self.spare_ogain = freeze_enc, spare_ogain
        self.smf_k, self.smf_eps, self.bg_freq = smf_k, smf_eps, None
        self.rng = np.random.default_rng(seed)
        self.pool = []                      # [(优先级, 序号, 字节序列)]
        self.pool_size, self.count = 0, 0
        self.morning = _frozen_copy(model)  # 白天学习前的模型：做梦式自蒸馏的老师，每次睡眠提交后更新
        for p in model.parameters():
            p.requires_grad_(False)
        self.eprop = params == 'eprop'
        assert not (self.eprop and replay), 'e-prop 的资格迹按单条流延续，不混重放批'
        pools = _pools(model)
        cores = [l.core for l in pools] if (freeze_enc and pools) else _cores(model)
        if self.eprop:
            low = [p for l in cores for p in (l.B, l.G, l.b_a)]
        elif params == 'pool_mlp' and pools:
            low = [p for l in [pl.core for pl in pools] for p in (l.W1, l.b1, l.W2)]
        else:
            low = [p for l in cores for p in (l.lora_a, l.lora_b)]
        shared = list(model.readout.parameters()) + [model.bias]
        if params in ('lora', 'eprop', 'pool_mlp'):
            shared = []
        self.params_mode = params
        self.groups = [{'params': low, 'weight_decay': wd}, {'params': shared, 'weight_decay': 0.0}]
        self.online = low + shared
        for p in self.online:
            p.requires_grad_(not self.eprop)               # e-prop 的梯度手算，不让 autograd 沿时间反传
        self.opt = torch.optim.AdamW(self.groups, lr=lr)
        # RMP 新奇触发的地址分配（第 23.3 节 O2）：某段损失超过近期均值的 novel_ratio 倍时，每波取 allocate 个最少用的模块，
        # 把地址写成该段在这一波的平均查询；此后在线学习只改被分配模块的低秩偏移（编码层与其余模块不动）。
        self.allocate, self.novel_ratio, self.warm_chunks = allocate, novel_ratio, warm_chunks
        self.top_pos, self.spare_bias, self.alloc_mode = top_pos, spare_bias, alloc_mode   # alloc_mode='split'：GNG 式分裂（复制父模块）
        self.loss_ema, self.n_chunks, self.allocated, self.alloc_log = None, 0, None, []
        self.smf_hits = torch.zeros(pools[0].N, device=pools[0].addr.device) if pools else None
        # 第二版：池里有空闲模块（active = 0）时，启用它们而不是改旧模块的地址；新模块的全部参数在线可训（新生神经元），其余不动
        self.spare_mode = bool(allocate and pools and (pools[0].active == 0).any())
        if self.spare_mode:
            grow = [p for n, p in pools[0].core.named_parameters() if 'lora' not in n] + [pools[0].ogain]
            for p in grow:
                p.requires_grad_(True)
            self.groups.append({'params': grow, 'weight_decay': 0.0}); self.online += grow
            self.opt = torch.optim.AdamW(self.groups, lr=lr)
        self.reset_session()
        if bg_bytes is not None and smf_k > 0:
            self.profile_bg(bg_bytes)

    @torch.no_grad()
    def profile_bg(self, bg_bytes, chunk=256):
        """统计池内各模块在背景旧语料上的平均选中频率，供 SMF 相对背景选模块使用。"""
        pools = _pools(self.model)
        if not pools:
            return
        pool = pools[0]
        ids = _ids(bg_bytes, _device(self.model))
        cnt = torch.zeros(pool.N, device=pool.addr.device)
        tot, st = 0, None
        was_train = self.model.training
        self.model.eval()
        for i in range(0, len(ids) - 1, chunk):
            seg = ids[i:i + chunk]
            if len(seg) < 2:
                continue
            pool.capture = []
            _, st = self.model(seg[None, :-1], st)
            caps, pool.capture = pool.capture, None
            for _, sel, _, _ in caps:
                cnt += sel.sum((0, 1))
            tot += len(seg) - 1
        if was_train:
            self.model.train()
        self.bg_freq = cnt / max(tot, 1)

    def _pool_masked_params(self):
        """池内按模块（第 0 维为 N）排布的在线参数列表，供严格掩码步使用。"""
        pools = _pools(self.model)
        if not pools:
            return []
        pool = pools[0]
        ps = [p for p in self.online if p.shape[:1] == (pool.N,) and any(p is mp for mp in list(pool.core.parameters()) + [pool.ogain])]
        return ps

    def _compute_smf_mask(self, caps):
        """SMF：计算相对背景活跃度 (cur_freq / (bg_freq + eps)) 最高的 smf_k 个模块掩码 (N,)。"""
        pools = _pools(self.model)
        if not pools or self.bg_freq is None or self.smf_k <= 0:
            return None
        pool = pools[0]
        cur = torch.zeros(pool.N, device=pool.addr.device)
        ticks = caps[0][1].shape[0] * caps[0][1].shape[1] if caps else 1
        for _, sel, _, _ in caps:
            cur += sel.sum((0, 1))
        cur = cur / max(ticks, 1)
        score = (cur / (self.bg_freq + self.smf_eps)).masked_fill(pool.active == 0, -1e9)
        top = score.topk(min(self.smf_k, pool.N)).indices
        keep = torch.zeros(pool.N, device=pool.addr.device)
        keep[top] = 1.0
        if self.smf_hits is not None:
            self.smf_hits.add_(keep)
        return keep

    def _alloc_step(self, caps, loss, tok=None):
        """读完一段后：判断新奇，必要时分配地址（只分配一次）。tok：本段逐字节损失（第二版取最惊讶的几个位置的查询作新地址）。"""
        self.n_chunks += 1
        novel = self.loss_ema is not None and self.n_chunks > self.warm_chunks and loss > self.novel_ratio * self.loss_ema
        self.loss_ema = loss if self.loss_ema is None else 0.95 * self.loss_ema + 0.05 * loss
        if not novel or self.allocated is not None:
            return
        pool = _pools(self.model)[0]
        chosen = []
        with torch.no_grad():
            for w, sel, wt, r in caps:
                if self.spare_mode:                    # 第二版：启用空闲模块；地址取最惊讶位置的查询均值；新模块带负偏置，只在查询很近时才赢
                    mods = ((pool.wave == w) & (pool.active == 0)).nonzero()[:, 0][:self.allocate]
                    if len(mods) == 0:
                        continue
                    top = tok.topk(min(self.top_pos, len(tok))).indices
                    rq = F.normalize(r[0, top].float().mean(0), dim=-1)
                    if self.alloc_mode == 'split':       # GNG 式分裂：本段在新内容上承担损失最多的启用模块当父，整份复制给空闲模块，地址取父与新查询的中点
                        blame = (wt[0].float() * tok[:, None]).sum(0) * (pool.wave == w).float() * (pool.active > 0).float()
                        par = int(blame.argmax())
                        for c in mods.tolist():
                            for n, prm in pool.core.named_parameters():
                                prm[c] = prm[par]
                            pool.addr[c] = F.normalize(pool.addr[par] + rq, dim=-1)
                            pool.bias[c] = pool.bias[par] + self.spare_bias; pool.ogain[c] = pool.ogain[par]
                            pool.load_ema[c] = pool.load_ema[par]
                        pool.active[mods] = 1
                        chosen += mods.tolist(); continue
                    pool.addr[mods] = rq
                    pool.active[mods] = 1; pool.bias[mods] = self.spare_bias; pool.ogain[mods] = self.spare_ogain
                    chosen += mods.tolist(); continue
                mods = (pool.wave == w).nonzero()[:, 0]
                if len(mods) == 0:
                    continue
                pick = mods[pool.load_ema[mods].argsort()[:self.allocate]]
                pool.addr[pick] = F.normalize(r[0].float().mean(0), dim=-1)
                chosen += pick.tolist()
        self.allocated = torch.tensor(chosen, device=pool.addr.device)
        self.alloc_log.append({'chunk': self.n_chunks, 'loss': loss, 'ema': self.loss_ema, 'modules': chosen})

    def _compute_alloc_mask(self):
        """分配后：只留被分配模块的掩码，并冻结编码层在线参数；触发分配前返回 None（E85）。"""
        if self.allocated is None:
            return None, ()
        pool = _pools(self.model)[0]
        keep = torch.zeros(pool.N, device=pool.addr.device)
        keep[self.allocated] = 1.0
        zero_ps = [p for l in _cores(self.model) if l is not pool.core for p in (l.lora_a, l.lora_b) if p.requires_grad]
        return keep, zero_ps

    def reset_session(self):
        self.state, self.last, self.traces = None, None, None

    def _eprop_grads(self, recs):
        """由各层记录的 (u, la, c0, c) 与 c.grad 算 B、G、b_a 的梯度，资格迹跨块延续。"""
        if self.traces is None:
            self.traces = [None] * len(recs)
        for i, (layer, (u, la, c0, c)) in enumerate(zip(self.model.layers, recs)):
            delta = c.grad if c.grad is not None else torch.zeros_like(c)
            dt = layer.B.dtype                                                 # 资格迹用参数精度（fp32）
            u, la, cf = u.to(dt), la.to(dt), c.detach().to(dt)
            a = la.exp()
            gb = torch.einsum('btmu,mud->btmd', u, layer.B)
            cprev = torch.cat([c0.to(dt)[:, None], cf[:, :-1]], 1)
            s = a * (1 - a) * (cprev - gb)                                     # ∂c_t/∂z_t（z 是门的 logit），当拍
            b, T, M, U = u.shape
            if self.traces[i] is None:
                z = lambda *sh: torch.zeros(b, M, *sh, device=u.device, dtype=dt)
                self.traces[i] = (z(U, layer.d), z(U, layer.d), z(layer.d))
            eB0, eG0, eb0 = self.traces[i]
            la5 = la[:, :, :, None].expand(b, T, M, U, layer.d)
            eB, eBT = scan(la5, u[..., None] * (1 - a)[:, :, :, None], eB0)   # e^B_t = a e^B_{t−1} + (1 − a) u
            eG, eGT = scan(la5, u[..., None] * s[:, :, :, None], eG0)          # e^G_t = a e^G_{t−1} + s u
            eb, ebT = scan(la, s, eb0)
            self.traces[i] = (eBT, eGT, ebT)
            d = delta.to(dt)
            layer.B.grad = torch.einsum('btmud,btmd->mud', eB, d)
            layer.G.grad = torch.einsum('btmud,btmd->mud', eG, d)
            layer.b_a.grad = torch.einsum('btmd,btmd->md', eb, d)

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

    def read(self, text_bytes, gate=1.0):
        """读一段并在线学习；gate 乘在学习率上（新奇信号，0 = 只推进状态不学）。"""
        ids = _ids(text_bytes, _device(self.model))
        tot, n = 0.0, 0
        for i in range(0, len(ids), self.chunk):
            blk = ids[i:i + self.chunk]
            seq = blk if self.last is None else torch.cat([self.last, blk])
            self.last = blk[-1:]
            if len(seq) < 2:
                continue
            if self.eprop:
                for l in self.model.layers:
                    l.rec = []
            alloc = self.allocate and _pools(self.model)
            smf = self.smf_k > 0 and self.bg_freq is not None and _pools(self.model)
            if alloc or smf:
                _pools(self.model)[0].capture = []
            lg, st = self.model(seq[None, :-1], self.state)
            loss = F.cross_entropy(lg[0], seq[1:])
            if alloc or smf:
                caps, _pools(self.model)[0].capture = _pools(self.model)[0].capture, None
            if alloc:
                pending = (caps, loss.item(), F.cross_entropy(lg[0], seq[1:], reduction='none').detach())
            tot += loss.item() * (len(seq) - 1); n += len(seq) - 1
            total = loss
            if self.replay and self.pool:                  # 混入池中旧片段（从零状态），抵消只看新内容造成的遗忘
                pick = self.rng.choice(len(self.pool), size=min(4, len(self.pool)), replace=False)
                xy = torch.from_numpy(np.stack([self.pool[i][2] for i in pick]).astype(np.int64)).to(ids.device)
                rl, _ = self.model(xy[:, :-1])
                total = (1 - self.replay) * loss + self.replay * F.cross_entropy(rl.reshape(-1, rl.shape[-1]), xy[:, 1:].reshape(-1))
            for grp in self.opt.param_groups:
                grp['lr'] = self.lr * gate
            self.opt.zero_grad()
            if self.eprop:
                total.backward()
                recs = [l.rec[0] for l in self.model.layers]
                for l in self.model.layers:
                    l.rec = None
                self._eprop_grads(recs)
            else:
                total.backward()
            keep_mask, zero_ps = None, ()
            if self.allocate and _pools(self.model):
                keep_mask, zero_ps = self._compute_alloc_mask()
            if smf:
                smf_keep = self._compute_smf_mask(caps)
                keep_mask = smf_keep if keep_mask is None else (keep_mask * smf_keep)
            if gate > 0:
                if keep_mask is not None or zero_ps:
                    _masked_step(self.opt, self._pool_masked_params(), keep_mask if keep_mask is not None else torch.ones(_pools(self.model)[0].N, device=ids.device), zero_ps)
                else:
                    self.opt.step()
            if alloc:                                      # 分配放在更新之后：它原地改 ogain、地址等，反传前改会破坏计算图
                self._alloc_step(*pending)
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
            pairs = [(l.W1, l.lora_a, l.lora_b) for l in _cores(self.model)]
            for W, a, b in pairs:
                W += a @ b
                b.zero_()
                a.copy_(torch.randn_like(a) / math.sqrt(a.shape[-2]))

    def sleep(self, probe_old, probe_new, steps=200, lr=3e-4, batch=8, old_tol=0.02, anchor=0.0, dream=0.0, n_dream=64,
              distill=0.0, day=None, ce=1.0, win=64, old=None, rewire=True, smf_sleep=True):
        """先把低秩偏移精确并入慢权重（不损失），再在重放池上训 steps 步。
        anchor>0：加 KL(老师‖学生)，老师 = 并入后、重放前的模型，在重放片段上算（E31；但老师已带白天的遗忘，E32 前的第 8 轮显示锚越强旧知识越差）。
        dream>0：做梦式自蒸馏（设计阶段 5 第 1 项）——早晨模型（白天学习前）从池中片段开头自由生成 n_dream 段“梦”，
        在梦上加 dream·KL(早晨模型‖学生)，保住旧分布；池上的交叉熵负责学新内容。
        distill>0：上下文蒸馏（候选 17n1；Snell 等 2022）——老师 = 并入后的模型带着状态读完当天全部对话 day，记下每个位置的预测；
        学生 = 同一模型从零状态只看 win 字节的窗口，在同样位置上学老师的分布。当天靠上下文 / 快记忆知道的事由此进权重。
        old：旧语料字节；给了时锚定 KL 改在旧语料的随机窗口上算（候选 11 §3.3：只在旧数据上拉住，新内容不受锚的阻挡）。
        ce：池上交叉熵的权重（聊天时池里只有用户的话，训它会把模型推向“像用户那样说话”，E44，故聊天用 0）。
        smf_sleep：开启 smf_k 时，睡眠只巩固白天被 SMF 选中的模块，且与 freeze_enc 一致冻结编码层与 win。"""
        m = self.model
        backup = copy.deepcopy(m.state_dict())
        online_backup = [p.detach().clone() for p in self.online]
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
        # 睡眠只训慢权重；低秩偏移刚清零，留给第二天。嵌入 / 读出不动：改读出是遗忘主因（E29）；
        # rewire=False 时冻结路由权重 Wr / Wg；freeze_enc=True（带池）时同时冻结编码层 layers.0 与 win，避免输入分布漂移破坏固定路由。
        pools = _pools(m)
        skip = ('emb.', 'readout.', 'bias', 'norm.')
        if self.freeze_enc and pools:
            skip = skip + ('layers.0.', 'win.')
        slow = [p for k, p in m.named_parameters()
                if 'lora' not in k and not k.startswith(skip) and (rewire or not k.endswith(('.Wr', '.Wg')))]
        for p in slow:
            p.requires_grad_(True)
        opt = torch.optim.Adam(slow, lr=lr)
        sleep_keep, pool_slow = None, []
        if smf_sleep and self.smf_k > 0 and pools and self.smf_hits is not None and self.smf_hits.sum() > 0:
            pool = pools[0]
            k_act = min(self.smf_k, int((self.smf_hits > 0).sum().item()))
            top = self.smf_hits.topk(k_act).indices
            sleep_keep = torch.zeros(pool.N, device=pool.addr.device)
            sleep_keep[top] = 1.0
            pool_ids = {id(p) for p in list(pool.core.parameters()) + [pool.ogain]}
            pool_slow = [p for p in slow if id(p) in pool_ids and p.shape[:1] == (pool.N,)]
        plastic = [(pl, pl.plastic) for pl in pools]
        for pl, _ in plastic:                              # RMP：地址与均衡偏置只在睡眠中更新（第 23.3 节）；rewire=False 时睡眠也不动
            pl.plastic = rewire
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
                opt.zero_grad(); loss.backward()
                if sleep_keep is not None:
                    _masked_step(opt, pool_slow, sleep_keep)
                else:
                    opt.step()
                continue
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
            opt.zero_grad(); loss.backward()
            if sleep_keep is not None:
                _masked_step(opt, pool_slow, sleep_keep)
            else:
                opt.step()
        for p, f in flags:
            p.requires_grad_(f)
        for pl, f in plastic:
            pl.plastic = f
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
            if self.smf_hits is not None:
                self.smf_hits.zero_()
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
    res = {'learner': {k: v for k, v in learner.items() if k != 'bg_bytes'}, 'day_bytes': day_bytes, 'test_before': bpc(model, test), 'old_before': bpc(model, old)}
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
