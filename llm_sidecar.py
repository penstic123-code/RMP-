"""冻结开源 LLM（如 Qwen2.5-0.5B）的长期可用持续学习外挂：RMP 旁路记忆池 + 神经海马体槽表 + 原子持久化。

面向“长期可用（1 个月–1 年+）”的核心设计：
1. NeuralHippocampus（事实层）：
   - 均值中心化语义向量 + 词法稀疏匹配混合检索（齿状回模式分离，消除因果 LM 各向异性）；
   - 支持改口覆盖（supersede）：同主题新事实自动将旧槽位标为 superseded，退出默认召回；
   - 跨会话保留不清零，支持持久化。
2. RMPSidecar（技能/规则层，每模块独立零初始化出口）：
   - 取消全局共享可训练 proj_out，改为每模块独立低秩出口 mod_out_a/b（零初始化，受 SMF _masked_step 严格掩码），
     彻底消除“改模块 A 拖累模块 B”的共享出口漂移。
3. SidecarLearner（防学坏机制）：
   - 只巩固用户明确教/确认（stage_qa）的规则，不把未核实的模型幻觉写进权重；
   - 维护历史已确认规则回放库（confirmed_qa），睡眠时新规则 + 历史规则 + 纯冻结原脑三方平衡；
   - 睡眠带严格验收门控（新规则学进 + 历史规则不退化 + 通用能力不超阈），不通过自动回滚。
4. 原子存盘 + .bak 上一版快照：防止断电/误操作丢档，且含待巩固队列与已确认库。
"""
import copy, math, os, random, re
from collections import Counter
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from pool import WavePool
from online import _masked_step


def _lex_tokens(text):
    """提取中英文稀疏词法特征（英文词干/3-gram + 中文双字词），模拟齿状回模式分离。"""
    low = text.lower()
    en = re.findall(r'[a-z0-9]{2,}', low)
    zh = re.findall(r'[\u4e00-\u9fff]', low)
    zh_bi = [zh[i] + zh[i + 1] for i in range(len(zh) - 1)]
    stop = {'the', 'and', 'for', 'are', 'you', 'your', 'what', 'which', 'when', 'where', 'who', 'how',
            'this', 'that', 'with', 'from', 'have', 'has', 'was', 'were', 'not', 'but', 'about', 'into',
            'does', 'did', 'can', 'will', 'would', 'should', 'could', 'my', 'me', 'is', 'in', 'on', 'at', 'to', 'of'}
    return {w for w in en if w not in stop} | set(zh_bi) | set(zh)


class NeuralHippocampus(nn.Module):
    """长期可用的海马体槽表：均值中心化语义检索 + 稀疏词法模式分离 + 改口覆盖（supersede）+ 跨会话持久。"""
    def __init__(self, hidden_size, n_slots=4096, d_key=64, d_val=64, top_k=4, min_cos=0.35, temp=0.1):
        super().__init__()
        self.H, self.n_slots, self.d_key, self.d_val = hidden_size, n_slots, d_key, d_val
        self.top_k, self.min_cos, self.temp = top_k, min_cos, temp
        self.W_q = nn.Linear(hidden_size, d_key, bias=False)
        self.W_k = nn.Linear(hidden_size, d_key, bias=False)
        self.W_v = nn.Linear(hidden_size, d_val, bias=False)
        self.W_out = nn.Linear(d_val, hidden_size, bias=False)
        nn.init.zeros_(self.W_out.weight)
        self.register_buffer('keys', torch.zeros(n_slots, d_key))
        self.register_buffer('vals', torch.zeros(n_slots, d_val))
        self.register_buffer('sem_keys', torch.zeros(n_slots, hidden_size))
        self.register_buffer('used', torch.zeros(n_slots, dtype=torch.bool))
        self.register_buffer('superseded', torch.zeros(n_slots, dtype=torch.bool))  # 被改口覆盖的旧槽退出默认召回
        self.register_buffer('priority', torch.zeros(n_slots))
        self.register_buffer('hits', torch.zeros(n_slots))
        self.register_buffer('session_id', torch.zeros(n_slots, dtype=torch.long))
        self.register_buffer('ts', torch.zeros(n_slots, dtype=torch.long))          # 写入序号（用于改口新旧排序）
        self.slot_texts = [''] * n_slots
        self.slot_keys = [''] * n_slots                            # 可选显式事实键（如 "city" / "cat_name"）
        self.cur_session = 0
        self.clock = 0
        self.neural_active = False                                 # W_out 零初始化且默认不训，推理跳过避免每拍 .any() GPU 同步

    def new_session(self):
        self.cur_session += 1

    def forward(self, h):
        """神经残差槽召回（出口零初始化，起步严格输出 0）。"""
        if not self.neural_active:
            return torch.zeros_like(h)
        active = self.used & (~self.superseded)
        if not active.any():
            return torch.zeros_like(h)
        q = F.normalize(self.W_q(F.rms_norm(h, (self.H,)).to(self.W_q.weight.dtype)).float(), dim=-1)
        idx_used = active.nonzero(as_tuple=False)[:, 0]
        K = self.keys[idx_used].float()
        V = self.vals[idx_used].float()
        sim = torch.einsum('btk,uk->btu', q, K)
        k = min(self.top_k, K.shape[0])
        top_sim, top_i = sim.topk(k, dim=-1)
        mask = top_sim >= self.min_cos
        if not mask.any():
            return torch.zeros_like(h)
        w = torch.softmax((top_sim / self.temp).masked_fill(~mask, -1e9), dim=-1) * mask.float()
        recalled = (w[..., None] * V[top_i]).sum(-2)
        return self.W_out(recalled.to(self.W_out.weight.dtype)).to(h.dtype)

    @torch.no_grad()
    def write(self, h_vec, surprise_score):
        """写入隐状态槽（兼容底层接口）。"""
        if h_vec.numel() == 0:
            return 0
        hn = F.rms_norm(h_vec.to(self.W_k.weight.dtype), (self.H,))
        ks = F.normalize(self.W_k(hn).float(), dim=-1)
        vs = self.W_v(hn).float()
        scores = surprise_score.float().to(self.keys.device)
        written = 0
        for i in range(ks.shape[0]):
            slot = self._alloc_slot(scores[i].item())
            if slot is None:
                continue
            self.clock += 1
            self.keys[slot].copy_(ks[i]); self.vals[slot].copy_(vs[i])
            self.used[slot] = True; self.superseded[slot] = False
            self.priority[slot] = scores[i]; self.hits[slot] = 0.0
            self.session_id[slot] = self.cur_session; self.ts[slot] = self.clock
            written += 1
        return written

    def _alloc_slot(self, score):
        free = ((~self.used) | self.superseded).nonzero(as_tuple=False)[:, 0]
        if free.numel() > 0:
            return int(free[0].item())
        eff_prio = self.priority + 0.5 * torch.log1p(self.hits)
        lo = int(eff_prio.argmin().item())
        return lo if score > eff_prio[lo].item() else None

    @torch.no_grad()
    def write_fact(self, h_seq, text, score=1.0, fact_key='', supersede_sim=0.55):
        """写入用户事实并支持改口覆盖：
        1. 若给了显式 fact_key（如 'city'），同键旧槽自动标记 superseded=True；
        2. 否则若新事实与某旧槽在中心化语义 + 词法上高度重合（同一主题改口，如 墨尔本→悉尼），也自动覆盖旧槽。"""
        t = text.strip()
        if len(t) < 4 or t.endswith('?') or t.endswith('？'):
            return 0
        sem = h_seq.float().mean(dim=(0, 1)).to(self.sem_keys.device)
        fk = fact_key.strip().lower()
        # 检查是否覆盖同键或高度同主题的旧槽
        active = (self.used & (~self.superseded)).nonzero(as_tuple=False)[:, 0]
        if active.numel() > 0:
            if fk:
                for s in active.tolist():
                    if self.slot_keys[s] == fk:
                        self.superseded[s] = True
            else:
                new_lex = _lex_tokens(t)
                mean_v = self.sem_keys[active].float().mean(0) if active.numel() >= 3 else torch.zeros_like(sem)
                q_c = F.normalize(sem - mean_v, dim=-1)
                k_c = F.normalize(self.sem_keys[active].float() - mean_v, dim=-1)
                sem_sims = (k_c @ q_c).tolist()
                for s, ss in zip(active.tolist(), sem_sims):
                    if not self.slot_texts[s] or self.slot_keys[s]:
                        continue                                   # 显式 fact_key 槽位受保护，禁止被无键闲聊误覆盖
                    old_lex = _lex_tokens(self.slot_texts[s])
                    jacc = len(new_lex & old_lex) / max(len(new_lex | old_lex), 1)
                    if (ss >= supersede_sim and jacc >= 0.35) or jacc >= 0.6:
                        self.superseded[s] = True
        slot = self._alloc_slot(float(score))
        if slot is None:
            return 0
        self.clock += 1
        hn = F.rms_norm(h_seq[0, -1:].to(self.W_k.weight.dtype), (self.H,))
        self.keys[slot].copy_(F.normalize(self.W_k(hn).float(), dim=-1)[0])
        self.vals[slot].copy_(self.W_v(hn).float()[0])
        self.sem_keys[slot].copy_(sem)
        self.used[slot] = True; self.superseded[slot] = False
        self.priority[slot] = float(score); self.hits[slot] = 0.0
        self.session_id[slot] = self.cur_session; self.ts[slot] = self.clock
        self.slot_texts[slot] = t; self.slot_keys[slot] = fk
        return 1

    @torch.no_grad()
    def recall_facts(self, h_seq, query_text='', top_k=3, min_score=0.15, cross_session_only=False):
        """均值中心化语义余弦 + 词法 IDF 混合召回（只看未 superseded 的活跃槽，新鲜度打破平局）。"""
        mask = self.used & (~self.superseded)
        if cross_session_only:
            mask = mask & (self.session_id != self.cur_session)
        if not mask.any():
            return []
        idx = mask.nonzero(as_tuple=False)[:, 0]
        idx = idx[[i for i, s in enumerate(idx.tolist()) if self.slot_texts[s]]]
        if idx.numel() == 0:
            return []
        # 1. 均值中心化语义余弦（消除因果 LM 各向异性）
        raw_k = self.sem_keys[idx].float()
        q_raw = h_seq.float().mean(dim=(0, 1)).to(raw_k.device)
        mean_v = raw_k.mean(0) if idx.numel() >= 3 else torch.zeros_like(q_raw)
        sem_sim = (F.normalize(raw_k - mean_v, dim=-1) @ F.normalize(q_raw - mean_v, dim=-1)).tolist()
        # 2. 词法稀疏 IDF 匹配（齿状回模式分离）
        q_lex = _lex_tokens(query_text) if query_text else set()
        df = Counter()
        slot_lexes = []
        for s in idx.tolist():
            lx = _lex_tokens(self.slot_texts[s])
            slot_lexes.append(lx)
            df.update(lx)
        n_docs = max(len(slot_lexes), 10)
        scored = []
        max_ts = max(int(self.ts[idx].max().item()), 1)
        for pos, s in enumerate(idx.tolist()):
            lex_sc = 0.0
            if q_lex and slot_lexes[pos]:
                inter = q_lex & slot_lexes[pos]
                if inter:
                    lex_sc = sum(math.log(1.0 + n_docs / (df[w] + 1)) for w in inter) / (math.sqrt(len(slot_lexes[pos])) + 1.0)
            recency = 0.02 * (int(self.ts[s].item()) / max_ts)
            total = 0.45 * max(sem_sim[pos], 0.0) + 0.55 * min(lex_sc, 1.5) + recency
            if total >= min_score:
                scored.append((total, s))
        scored.sort(key=lambda t: t[0], reverse=True)
        out = []
        for _, s in scored[:top_k]:
            self.hits[s] += 1.0
            out.append(self.slot_texts[s])
        return out

    @torch.no_grad()
    def decay_after_sleep(self, factor=0.8):
        self.priority.mul_(factor)


SIDECAR_PRESETS = {
    'small': dict(
        model='Qwen/Qwen2.5-0.5B-Instruct',
        sidecar=dict(N=48, k=4, G=4, D=128, pd=64, pm=16, phidden=128, out_rank=16, ctx_decay=(0.5, 0.85, 0.95, 0.99)),
        hippo=dict(n_slots=2048, d_key=64, d_val=64),
    ),
    'large': dict(                                                 # 8GB 显卡稳跑全线放大版：1.5B + N=512 宽模块 + 16K 槽 + O(k) 稀疏执行
        model='Qwen/Qwen2.5-1.5B-Instruct',
        sidecar=dict(N=512, k=4, G=4, D=256, pd=96, pm=16, phidden=256, out_rank=24, ctx_decay=(0.5, 0.85, 0.95, 0.99)),
        hippo=dict(n_slots=16384, d_key=64, d_val=64),
    ),
}


class RMPSidecar(nn.Module):
    """挂在冻结 LLM 某一层残差流上的 RMP 记忆池旁路：
    每模块独立零初始化低秩出口（mod_out_a/b）+ O(k) 稀疏活跃模块执行（支持 N=512/1024 稳跑 8GB 显卡）。"""
    def __init__(self, hidden_size, N=48, k=4, G=4, D=128, pd=64, pm=16, phidden=128, qr=32, out_rank=16,
                 ctx_decay=(0.5, 0.85, 0.95, 0.99), bottleneck=True, wave_query=True):
        super().__init__()
        self.H, self.D, self.G, self.pd, self.N = hidden_size, D, G, pd, N
        self.proj_in = nn.Linear(hidden_size, D, bias=False)
        self.pool = WavePool(
            N=N, k=k, G=G, D=D, d=pd, m=pm, hidden=phidden, qr=qr,
            init='spread', rule='ctx', ctx_decay=list(ctx_decay),
            bottleneck=bottleneck, wave_query=wave_query, wave_ctx=True,
        )
        self.pool.plastic = False
        self.pool.sparse_active = True                             # 仅 gather 活跃模块，算力与显存 O(k) 与 N 无关
        nn.init.normal_(self.pool.core.W2, std=0.02)
        # 每模块独立出口增量：(N, pd, out_rank) @ (N, out_rank, H)，b 零初始化，起步严格输出 0 且完全按模块隔离
        self.mod_out_a = nn.Parameter(torch.randn(N, pd, out_rank) / math.sqrt(pd))
        self.mod_out_b = nn.Parameter(torch.zeros(N, out_rank, hidden_size))
        self.state = None

    def reset_session(self):
        self.state = None

    def forward(self, h):
        """h: (b, T, H) → 每波仅 gather 活跃模块及其独立出口加权求和输出增量 (b, T, H)。"""
        b = h.shape[0]
        if self.state is None or self.state[0].shape[0] != b:
            self.state = self.pool.init_state(b, h.device)
        x = self.proj_in(F.rms_norm(h, (self.H,)).to(self.proj_in.weight.dtype))
        pool = self.pool
        if pool.sparse_active and not (pool.training and torch.is_grad_enabled() and pool.plastic):
            sp_cap = []
            pool.sparse_capture = sp_cap
            hs, _, new_st = pool(x, self.state)
            pool.sparse_capture = None
            self.state = tuple(t.detach() for t in new_st)
            delta = torch.zeros_like(h, dtype=self.mod_out_b.dtype)
            for w_idx, act_global, wt_a in sp_cap:
                h_w = F.rms_norm(hs[w_idx].to(self.mod_out_a.dtype), (self.pd,))
                a_sub = self.mod_out_a[act_global]                 # 仅取活跃模块 (M_act, pd, r)
                b_sub = self.mod_out_b[act_global]                 # (M_act, r, H)
                a_mix = torch.einsum('btm,mdr->btdr', wt_a.to(a_sub.dtype), a_sub)
                b_mix = torch.einsum('btm,mrh->btrh', wt_a.to(b_sub.dtype), b_sub)
                mid = torch.einsum('btd,btdr->btr', h_w, a_mix)
                delta = delta + torch.einsum('btr,btrh->bth', mid, b_mix)
            return delta.to(h.dtype)
        # 兼容回退路径（sparse_active=False 时）
        prev_cap = pool.capture
        local_cap = []
        pool.capture = local_cap
        hs, _, new_st = pool(x, self.state)
        pool.capture = prev_cap
        if prev_cap is not None:
            prev_cap.extend(local_cap)
        self.state = tuple(t.detach() for t in new_st)
        delta = torch.zeros_like(h, dtype=self.mod_out_b.dtype)
        for w_idx, (_, _, wt, _) in enumerate(local_cap):
            h_w = F.rms_norm(hs[w_idx].to(self.mod_out_a.dtype), (self.pd,))
            a_mix = torch.einsum('btn,ndr->btdr', wt.to(self.mod_out_a.dtype), self.mod_out_a)
            b_mix = torch.einsum('btn,nrh->btrh', wt.to(self.mod_out_b.dtype), self.mod_out_b)
            mid = torch.einsum('btd,btdr->btr', h_w, a_mix)
            delta = delta + torch.einsum('btr,btrh->bth', mid, b_mix)
        return delta.to(h.dtype)


class LLMSidecarWrapper(nn.Module):
    """冻结因果语言模型 + RMPSidecar（独立模块出口）+ NeuralHippocampus（可改口覆写槽表）。"""
    def __init__(self, backbone, attach_layer=-4, sidecar_cfg=None, hippo_cfg=None):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        layers = self._find_layers(backbone)
        self.layer_idx = attach_layer if attach_layer >= 0 else len(layers) + attach_layer
        hidden_size = getattr(backbone.config, 'hidden_size', None) or getattr(backbone.config, 'n_embd')
        self.sidecar = RMPSidecar(hidden_size, **(sidecar_cfg or {}))
        self.hippo = NeuralHippocampus(hidden_size, **(hippo_cfg or {}))
        self._last_h = None
        self.bypass = False
        self._hook = layers[self.layer_idx].register_forward_hook(self._inject_hook)

    @staticmethod
    def _find_layers(backbone):
        if hasattr(backbone, 'model') and hasattr(backbone.model, 'layers'):
            return backbone.model.layers
        if hasattr(backbone, 'transformer') and hasattr(backbone.transformer, 'h'):
            return backbone.transformer.h
        if hasattr(backbone, 'layers'):
            return backbone.layers
        raise ValueError('未识别的骨干层列表属性')

    def _inject_hook(self, module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        self._last_h = h.detach()
        if self.bypass:
            return out
        delta = self.sidecar(h) + self.hippo(h)
        h_new = h + delta
        return (h_new,) + out[1:] if isinstance(out, tuple) else h_new

    def reset_session(self):
        self.sidecar.reset_session()
        self.hippo.new_session()

    def forward(self, input_ids, **kwargs):
        return self.backbone(input_ids=input_ids, **kwargs)

    def save_memory(self, path, learner=None):
        """原子存盘（写临时文件再替换）并自动保留上一份 .bak 可恢复快照；同步保存待巩固与已确认队列。"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + '.tmp')
        bak = p.with_suffix(p.suffix + '.bak')
        payload = {
            'sidecar': self.sidecar.state_dict(),
            'hippo': self.hippo.state_dict(),
            'slot_texts': self.hippo.slot_texts,
            'slot_keys': self.hippo.slot_keys,
            'cur_session': self.hippo.cur_session,
            'clock': self.hippo.clock,
            'layer_idx': self.layer_idx,
        }
        if learner is not None:
            payload['staged'] = learner.staged
            payload['confirmed_qa'] = learner.confirmed_qa
        torch.save(payload, tmp)
        if p.exists():
            os.replace(p, bak)
        os.replace(tmp, p)

    def load_memory(self, path, map_location='cpu', learner=None):
        ck = torch.load(path, map_location=map_location, weights_only=False)
        self.sidecar.load_state_dict(ck['sidecar'])
        self.hippo.load_state_dict(ck['hippo'])
        if 'slot_texts' in ck:
            self.hippo.slot_texts = ck['slot_texts']
        if 'slot_keys' in ck:
            self.hippo.slot_keys = ck['slot_keys']
        self.hippo.cur_session = ck.get('cur_session', 0)
        self.hippo.clock = ck.get('clock', 0)
        if learner is not None:
            if 'staged' in ck:
                learner.staged = ck['staged']
            if 'confirmed_qa' in ck:
                learner.confirmed_qa = ck['confirmed_qa']


class SidecarLearner:
    """长期可用的旁路学习器：
    - 全部可训练参数均按模块排布（dim 0 == N），无全局共享出口漂移；
    - 睡眠巩固带历史已确认规则回放 + 纯冻结原脑锚定 + 验收门控（不达标自动回滚）。"""
    def __init__(self, wrapper: LLMSidecarWrapper, lr=1e-3, smf_k=8, smf_eps=1e-2, novel_ratio=1.2, max_stage=512):
        self.w = wrapper
        self.lr, self.smf_k, self.smf_eps = lr, smf_k, smf_eps
        self.novel_ratio, self.max_stage = novel_ratio, max_stage
        pool = wrapper.sidecar.pool
        pool.plastic = False
        for p in wrapper.parameters():
            p.requires_grad_(False)
        # 全部可训练参数第 0 维均为 N：严格受 SMF _masked_step 按模块隔离！
        self.pool_params = [pool.core.W1, pool.core.b1, pool.core.W2,
                            wrapper.sidecar.mod_out_a, wrapper.sidecar.mod_out_b]
        for p in self.pool_params:
            p.requires_grad_(True)
        self.opt = None                                            # 惰性创建，避免与 /sleep 的优化器同时占两份显存
        self.bg_freq = torch.zeros(pool.N, device=pool.addr.device)
        self.smf_hits = torch.zeros(pool.N, device=pool.addr.device)
        self.loss_ema = None
        self.staged = []                                           # 待巩固的新规则：[(prio, ids_cpu, prompt_len)]
        self.confirmed_qa = []                                     # 历史已成功巩固的规则回放库（防新覆旧）

    @torch.no_grad()
    def profile_bg(self, token_batches):
        pool = self.w.sidecar.pool
        cnt = torch.zeros(pool.N, device=pool.addr.device)
        tot = 0
        self.w.eval(); self.w.sidecar.reset_session()
        for ids in token_batches:
            pool.capture = []
            self.w(ids)
            caps, pool.capture = pool.capture, None
            for _, sel, _, _ in caps:
                cnt += sel.sum((0, 1))
            tot += ids.numel()
        self.bg_freq = cnt / max(tot, 1)
        self.w.sidecar.reset_session()

    def _smf_mask(self, caps):
        pool = self.w.sidecar.pool
        cur = torch.zeros(pool.N, device=pool.addr.device)
        ticks = caps[0][1].shape[0] * caps[0][1].shape[1] if caps else 1
        for _, sel, _, _ in caps:
            cur += sel.sum((0, 1))
        cur = cur / max(ticks, 1)
        score = (cur / (self.bg_freq + self.smf_eps)).masked_fill(pool.active == 0, -1e9)
        top = score.topk(min(self.smf_k, pool.N)).indices
        keep = torch.zeros(pool.N, device=pool.addr.device)
        keep[top] = 1.0
        self.smf_hits.add_(keep)
        return keep

    def step_online(self, input_ids, gate=1.0):
        """单段在线学习（兼容基准测试）。"""
        self.w.train()
        pool = self.w.sidecar.pool
        pool.capture = []
        out = self.w(input_ids)
        caps, pool.capture = pool.capture, None
        logits = out.logits if hasattr(out, 'logits') else out
        tok_loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), input_ids[:, 1:].reshape(-1), reduction='none').view(input_ids.shape[0], -1)
        loss = tok_loss.mean(); loss_val = float(loss.item())
        written = 0
        is_novel = self.loss_ema is None or loss_val > self.novel_ratio * self.loss_ema
        self.loss_ema = loss_val if self.loss_ema is None else 0.95 * self.loss_ema + 0.05 * loss_val
        if is_novel and self.w._last_h is not None:
            flat_l = tok_loss.detach().reshape(-1)
            flat_h = self.w._last_h[:, :-1].reshape(-1, self.w._last_h.shape[-1])
            k_wr = min(4, flat_l.numel())
            top_v, top_i = flat_l.topk(k_wr)
            written = self.w.hippo.write(flat_h[top_i], top_v)
            self.staged.append((loss_val, input_ids.detach().cpu(), 0))
            if len(self.staged) > self.max_stage:
                self.staged.sort(key=lambda t: t[0], reverse=True)
                self.staged = self.staged[:self.max_stage]
        if gate > 0:
            if self.opt is None:
                self.opt = torch.optim.AdamW(self.pool_params, lr=self.lr)
            self.opt.zero_grad(); loss.backward()
            keep = self._smf_mask(caps)
            _masked_step(self.opt, self.pool_params, keep)
        return {'loss': loss_val, 'written_slots': written, 'staged': len(self.staged)}

    def stage_qa(self, full_ids, prompt_len, priority=1.0):
        """只将用户明确教/确认的问答对加入待巩固队列（睡眠时仅对助手回复算损失）。"""
        if full_ids.shape[1] <= prompt_len:
            return
        self.staged.append((float(priority), full_ids.detach().cpu(), int(prompt_len)))
        if len(self.staged) > self.max_stage:
            self.staged.sort(key=lambda t: t[0], reverse=True)
            self.staged = self.staged[:self.max_stage]

    @torch.no_grad()
    def _eval_qa_loss(self, items, dev):
        if not items:
            return 0.0
        self.w.eval()
        tot, n = 0.0, 0
        for item in items:
            ids = item[1].to(dev); plen = item[2] if len(item) >= 3 else 0
            self.w.sidecar.reset_session()
            lg = self.w(ids).logits.float()
            if plen > 0 and plen < ids.shape[1]:
                tot += float(F.cross_entropy(lg[:, plen - 1:-1].reshape(-1, lg.shape[-1]), ids[:, plen:].reshape(-1), reduction='sum').item())
                n += ids.shape[1] - plen
            else:
                tot += float(F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1), reduction='sum').item())
                n += ids.shape[1] - 1
        return tot / max(n, 1)

    @torch.no_grad()
    def _eval_anchor_loss(self, batches, dev):
        if not batches:
            return 0.0
        self.w.eval()
        tot, n = 0.0, 0
        for ab in batches:
            ab = ab.to(dev)
            self.w.sidecar.reset_session()
            lg = self.w(ab).logits.float()
            tot += float(F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), ab[:, 1:].reshape(-1), reduction='sum').item())
            n += ab.shape[1] - 1
        return tot / max(n, 1)

    def sleep_consolidate(self, steps=40, lr=1e-3, anchor_batches=None, anchor_w=1.5,
                          old_tol=0.05, hist_tol=0.15, replay_w=0.5):
        """带验收回滚的睡眠巩固：
        1. 用 SMF 在待巩固新规则上挑冷门模块；
        2. 联合优化 新规则 + 历史已确认规则回放 + 纯冻结原脑锚定；
        3. 睡后验收（新规则下降 & 通用旧能力升幅 <= old_tol & 历史规则升幅 <= hist_tol），不通过自动回滚。"""
        if not self.staged:
            return {'steps': 0, 'committed': False}
        pool = self.w.sidecar.pool
        dev = pool.addr.device
        backup = copy.deepcopy(self.w.sidecar.state_dict())
        new0 = self._eval_qa_loss(self.staged, dev)
        hist0 = self._eval_qa_loss(self.confirmed_qa, dev) if self.confirmed_qa else 0.0
        anc0 = self._eval_anchor_loss(anchor_batches, dev) if anchor_batches else 0.0

        # 选出新规则最冷门的 smf_k 个模块（若已有历史规则，也允许历史命中的活跃模块在回放中微调）
        if self.smf_hits.sum() == 0:
            self.w.eval()
            with torch.no_grad():
                for item in self.staged:
                    ids = item[1].to(dev)
                    self.w.sidecar.reset_session()
                    pool.capture = []
                    self.w(ids)
                    caps, pool.capture = pool.capture, None
                    self._smf_mask(caps)
        k_act = min(self.smf_k, max(1, int((self.smf_hits > 0).sum().item())))
        sleep_keep = torch.zeros(pool.N, device=dev)
        sleep_keep[self.smf_hits.topk(k_act).indices] = 1.0
        opt = torch.optim.Adam(self.pool_params, lr=lr)
        hist_order = list(range(len(self.confirmed_qa)))
        if hist_order:
            random.shuffle(hist_order)
        self.w.train()
        for step in range(steps):
            item = self.staged[step % len(self.staged)]
            ids = item[1].to(dev); plen = item[2] if len(item) >= 3 else 0
            self.w.sidecar.reset_session()
            lg = self.w(ids).logits
            if plen > 0 and plen < ids.shape[1]:
                loss = F.cross_entropy(lg[:, plen - 1:-1].reshape(-1, lg.shape[-1]), ids[:, plen:].reshape(-1))
            else:
                loss = F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1))
            if self.confirmed_qa and replay_w > 0:
                if step > 0 and step % len(hist_order) == 0:
                    random.shuffle(hist_order)
                hi = self.confirmed_qa[hist_order[step % len(hist_order)]]
                hids = hi[1].to(dev); hplen = hi[2] if len(hi) >= 3 else 0
                self.w.sidecar.reset_session()
                hlg = self.w(hids).logits
                if hplen > 0 and hplen < hids.shape[1]:
                    hloss = F.cross_entropy(hlg[:, hplen - 1:-1].reshape(-1, hlg.shape[-1]), hids[:, hplen:].reshape(-1))
                else:
                    hloss = F.cross_entropy(hlg[:, :-1].reshape(-1, hlg.shape[-1]), hids[:, 1:].reshape(-1))
                loss = loss + replay_w * hloss
            if anchor_batches and anchor_w > 0:
                ab = anchor_batches[step % len(anchor_batches)].to(dev)
                self.w.sidecar.reset_session()
                sl = self.w(ab).logits
                with torch.no_grad():
                    self.w.bypass = True
                    try:
                        tl = self.w(ab).logits
                    finally:
                        self.w.bypass = False
                loss = loss + anchor_w * F.kl_div(
                    F.log_softmax(sl.reshape(-1, sl.shape[-1]), -1),
                    F.log_softmax(tl.reshape(-1, tl.shape[-1]), -1),
                    log_target=True, reduction='batchmean',
                )
            opt.zero_grad(); loss.backward()
            _masked_step(opt, self.pool_params, sleep_keep)

        new1 = self._eval_qa_loss(self.staged, dev)
        hist1 = self._eval_qa_loss(self.confirmed_qa, dev) if self.confirmed_qa else 0.0
        anc1 = self._eval_anchor_loss(anchor_batches, dev) if anchor_batches else 0.0
        ok = (new1 < new0) and ((anc1 - anc0) <= old_tol) and ((not self.confirmed_qa) or (hist1 - hist0) <= hist_tol)
        if not ok:
            self.w.sidecar.load_state_dict(backup)
        else:
            self.confirmed_qa.extend(self.staged)
            self.staged = []
            self.w.hippo.decay_after_sleep()
        self.smf_hits.zero_()
        self.w.sidecar.reset_session()
        return {
            'steps': steps, 'consolidated_modules': k_act, 'committed': ok,
            'new_loss': (round(new0, 3), round(new1, 3)),
            'hist_rise': round(hist1 - hist0, 4),
            'anchor_rise': round(anc1 - anc0, 4),
        }
