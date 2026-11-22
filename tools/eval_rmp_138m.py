"""138M 独立 RMP 四合一诊断评测脚本：
1. 竞争与负载均衡（各波负载熵、死模块率、地址余弦分散度、top-1/top-2 权重锐度）
2. 多领域分工涌现（P2：4 个异构英文领域 edu/story/code/chat 的来源选择性 src_sel = MI/H）
3. 12 波时间尺度层级（P5：静态/动态有效时间常数 tau 与路由驻留长度 dwell_len）
4. 抗遗忘同收益对照（P3：OOD 新任务在线学习，对比 Full FT vs Dense LoRA vs RMP+SMF 稀疏掩码）

用法：
  python tools/eval_rmp_138m.py /content/drive/MyDrive/rmp_138m/latest.pt --out /content/drive/MyDrive/rmp_138m/eval_4in1.json
"""
import argparse, copy, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pool import WaveNet

EOT_ID = 50256


def _H(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _mi(joint):
    s = joint.sum()
    if s <= 0:
        return 0.0
    j = joint / s
    return max(0.0, _H(j.sum(1)) + _H(j.sum(0)) - _H(j.ravel()))


def load_domain_tokens(enc, n_tokens=40_000):
    """流式加载 4 类异构英文领域（各 n_tokens）：edu / story / code / chat。"""
    from datasets import load_dataset

    def _collect(ds, text_fn, skip=0):
        buf = []
        for i, row in enumerate(ds):
            if i < skip:
                continue
            t = text_fn(row)
            if not t or len(t) < 20:
                continue
            buf.extend(enc.encode_ordinary(t))
            buf.append(EOT_ID)
            if len(buf) >= n_tokens:
                break
        return torch.tensor(buf[:n_tokens], dtype=torch.long)

    domains = {}
    print('[Data] Loading 4 domains via HF streaming...')
    # 1. edu (FineWeb-Edu held-out)
    ds_edu = load_dataset('HuggingFaceFW/fineweb-edu', 'sample-10BT', split='train', streaming=True)
    domains['edu'] = _collect(ds_edu, lambda r: r.get('text', ''), skip=10)

    # 2. story (TinyStories)
    ds_story = load_dataset('roneneldan/TinyStories', split='validation', streaming=True)
    domains['story'] = _collect(ds_story, lambda r: r.get('text', ''))

    # 3. code (Python instructions / code)
    try:
        ds_code = load_dataset('iamtarun/python_code_instructions_18k_alpaca', split='train', streaming=True)
        domains['code'] = _collect(ds_code, lambda r: f"# {r.get('instruction','')}\n{r.get('output','')}\n")
    except Exception:
        ds_code = load_dataset('flytech/python-codes-25k', split='train', streaming=True)
        domains['code'] = _collect(ds_code, lambda r: f"# {r.get('instruction','')}\n{r.get('output','')}\n")

    # 4. chat (SmolTalk everyday conversations -> OOD dialogue)
    try:
        ds_chat = load_dataset('HuggingFaceTB/smoltalk', 'everyday-conversations', split='train', streaming=True)
        def _fmt_chat(r):
            msgs = r.get('messages', [])
            return '\n'.join(f"{'User' if m.get('role')=='user' else 'Assistant'}: {m.get('content','')}" for m in msgs)
        domains['chat'] = _collect(ds_chat, _fmt_chat)
    except Exception:
        ds_chat = load_dataset('daily_dialog', split='validation', streaming=True)
        domains['chat'] = _collect(ds_chat, lambda r: '\n'.join(
            f"{'User' if i%2==0 else 'Assistant'}: {u}" for i, u in enumerate(r.get('dialog', []))))

    for k, v in domains.items():
        print(f'  - {k:5s}: {len(v):,} tokens')
    return domains


def make_batches(toks, B, T):
    stride = B * (T + 1)
    n = len(toks) // stride
    assert n >= 1, f'Not enough tokens ({len(toks)}) for B={B}, T={T}'
    arr = toks[:n * stride].view(n, B, T + 1)
    return [(arr[i, :, :-1], arr[i, :, 1:]) for i in range(n)]


@torch.no_grad()
def eval_loss(model, batches, device, amp_dtype):
    model.eval()
    tot, n = 0.0, 0
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=(device == 'cuda')):
            lg, _ = model(x)
            loss = F.cross_entropy(lg.float().reshape(-1, lg.size(-1)), y.reshape(-1))
        tot += float(loss.item())
        n += 1
    return tot / max(1, n)


@torch.no_grad()
def test_competition_and_specialization(model, domains, device, amp_dtype, T=512, B=4):
    """测试 1（竞争）+ 测试 2（分工）+ 测试 3（层级）。"""
    model.eval()
    wp = model.layers[1]
    prev_plastic = wp.plastic
    wp.plastic = False
    G, N = wp.G, wp.N
    nw = N // G
    dom_names = list(domains.keys())
    n_dom = len(dom_names)

    # 1. 静态参数统计：地址几何分散度 & 静态时间常数 tau
    addr = wp.addr.float().cpu()
    b_a = wp.core.b_a.float().cpu()  # (N, d)
    a_static = torch.sigmoid(b_a)
    tau_static = 1.0 / (1.0 - a_static).clamp(min=1e-6)

    addr_cos_per_wave = []
    tau_med_per_wave = []
    for w in range(G):
        sl = slice(w * nw, (w + 1) * nw)
        aw = F.normalize(addr[sl], dim=-1)
        sim = aw @ aw.t()
        off_diag = sim[~torch.eye(nw, dtype=torch.bool)]
        addr_cos_per_wave.append(round(float(off_diag.mean()), 4))
        tau_med_per_wave.append(round(float(tau_static[sl].median()), 2))

    # 2. 前向采集各领域路由分布、切换率（驻留长度）、top1 锐度
    # joint[w, mod_in_wave, dom_idx]
    joint = np.zeros((G, nw, n_dom), dtype=np.float64)
    top1_w_sum = np.zeros(G, dtype=np.float64)
    switch_cnt = np.zeros(G, dtype=np.float64)
    tok_cnt = np.zeros(G, dtype=np.float64)
    dom_losses = {}
    mod_freq_by_dom = {d: np.zeros(N, dtype=np.float64) for d in dom_names}

    for di, dname in enumerate(dom_names):
        batches = make_batches(domains[dname][:16384], B=B, T=T)
        dloss = 0.0
        for x, y in batches:
            x, y = x.to(device), y.to(device)
            wp.capture = []
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=(device == 'cuda')):
                lg, _ = model(x)
                dloss += float(F.cross_entropy(lg.float().reshape(-1, lg.size(-1)), y.reshape(-1)).item())
            caps, wp.capture = wp.capture, None
            for w, sel, wt, _ in caps:
                sl = slice(w * nw, (w + 1) * nw)
                wt_w = wt[..., sl].float().cpu().numpy()  # (B, T, nw)
                sel_w = sel[..., sl].float().cpu().numpy()
                joint[w, :, di] += wt_w.sum(axis=(0, 1))
                mod_freq_by_dom[dname][sl] += sel_w.sum(axis=(0, 1))
                top1_idx = wt_w.argmax(axis=-1)  # (B, T)
                top1_val = wt_w.max(axis=-1)
                top1_w_sum[w] += float(top1_val.sum())
                switch_cnt[w] += float((top1_idx[:, 1:] != top1_idx[:, :-1]).sum())
                tok_cnt[w] += float(top1_idx.size)
        dom_losses[dname] = round(dloss / max(1, len(batches)), 4)
        mod_freq_by_dom[dname] /= max(1.0, mod_freq_by_dom[dname].sum())

    # 3. 汇总每波指标
    wave_stats = []
    src_sels, load_ents, dwell_lens = [], [], []
    for w in range(G):
        j = joint[w]  # (nw, n_dom)
        p_mod = j.sum(1) / max(j.sum(), 1e-9)
        ent = _H(p_mod) / math.log(nw)
        dead = float((p_mod < 0.1 / nw).mean())
        p_dom = j.sum(0) / max(j.sum(), 1e-9)
        src_sel = _mi(j) / max(_H(p_dom), 1e-9)
        top1_share = top1_w_sum[w] / max(tok_cnt[w], 1.0)
        sw_rate = switch_cnt[w] / max(tok_cnt[w] * (T - 1) / T, 1.0)
        dwell = 1.0 / max(sw_rate, 1e-5)
        src_sels.append(src_sel); load_ents.append(ent); dwell_lens.append(dwell)
        wave_stats.append({
            'wave': w,
            'ctx_decay': wp.ctx_decays[w],
            'load_ent': round(ent, 3),
            'dead': round(dead, 3),
            'addr_cos': addr_cos_per_wave[w],
            'top1_wt': round(top1_share, 3),
            'src_sel': round(src_sel, 4),
            'tau_med': tau_med_per_wave[w],
            'dwell_tok': round(dwell, 2),
        })

    wp.plastic = prev_plastic
    return {
        'domain_val_loss': dom_losses,
        'mean_load_ent': round(float(np.mean(load_ents)), 4),
        'mean_addr_cos': round(float(np.mean(addr_cos_per_wave)), 4),
        'mean_src_sel': round(float(np.mean(src_sels)), 4),
        'max_src_sel': round(float(np.max(src_sels)), 4),
        'dwell_ratio_deep_vs_shallow': round(float(np.mean(dwell_lens[-3:]) / max(np.mean(dwell_lens[:3]), 1e-5)), 2),
        'tau_ratio_deep_vs_shallow': round(float(np.mean(tau_med_per_wave[-3:]) / max(np.mean(tau_med_per_wave[:3]), 1e-5)), 2),
        'waves': wave_stats,
        'mod_freq_by_dom': {k: v.tolist() for k, v in mod_freq_by_dom.items()},
    }


def _run_online_curve(base_sd, cfg, mode, train_batches, new_val_batches, old_val_dict,
                      selected_mods, device, amp_dtype, lr, steps=40, eval_every=5):
    """从同一底座出发，在 OOD 新任务上在线学习，记录 (new_drop, old_rise) 同收益曲线。"""
    model = WaveNet(**cfg).to(device)
    model.load_state_dict(base_sd)
    wp = model.layers[1]
    wp.plastic = False  # 醒时冻结路由

    # 设定可训练参数与 SMF 掩码
    for p in model.parameters():
        p.requires_grad_(False)

    smf_mask = None
    if mode == 'full_ft':
        for p in model.parameters():
            p.requires_grad_(True)
        for p in [wp.core.lora_a, wp.core.lora_b, model.layers[0].lora_a, model.layers[0].lora_b]:
            p.requires_grad_(False)
    elif mode == 'dense_lora':
        wp.core.lora_a.requires_grad_(True)
        wp.core.lora_b.requires_grad_(True)
    elif mode == 'smf_mlp':
        # 仅更新 WavePool 模块的局部 MLP (W1, b1, W2)，且梯度严格按 selected_mods 掩码
        wp.core.W1.requires_grad_(True)
        wp.core.b1.requires_grad_(True)
        wp.core.W2.requires_grad_(True)
        smf_mask = torch.zeros(wp.N, device=device)
        smf_mask[selected_mods] = 1.0
    elif mode == 'smf_lora':
        wp.core.lora_a.requires_grad_(True)
        wp.core.lora_b.requires_grad_(True)
        smf_mask = torch.zeros(wp.N, device=device)
        smf_mask[selected_mods] = 1.0
    else:
        raise ValueError(mode)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)

    # 初始基线 loss
    base_new = eval_loss(model, new_val_batches, device, amp_dtype)
    base_old = {k: eval_loss(model, v, device, amp_dtype) for k, v in old_val_dict.items()}
    curve = [{'step': 0, 'new_loss': round(base_new, 4), 'new_drop': 0.0,
              'old_rise': {k: 0.0 for k in base_old}, 'worst_old_rise': 0.0, 'mean_old_rise': 0.0}]

    model.train()
    for step in range(1, steps + 1):
        x, y = train_batches[(step - 1) % len(train_batches)]
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=(device == 'cuda')):
            lg, _ = model(x)
            loss = F.cross_entropy(lg.float().reshape(-1, lg.size(-1)), y.reshape(-1))
        loss.backward()
        if smf_mask is not None:
            for p in params:
                if p.grad is not None and p.grad.shape[0] == wp.N:
                    shape = [wp.N] + [1] * (p.grad.ndim - 1)
                    p.grad.mul_(smf_mask.view(*shape))
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if step % eval_every == 0:
            cur_new = eval_loss(model, new_val_batches, device, amp_dtype)
            cur_old = {k: eval_loss(model, v, device, amp_dtype) for k, v in old_val_dict.items()}
            rises = {k: round(cur_old[k] - base_old[k], 4) for k in base_old}
            curve.append({
                'step': step,
                'new_loss': round(cur_new, 4),
                'new_drop': round(base_new - cur_new, 4),
                'old_rise': rises,
                'worst_old_rise': round(max(rises.values()), 4),
                'mean_old_rise': round(float(np.mean(list(rises.values()))), 4),
            })
            model.train()

    del model, opt
    if device == 'cuda':
        torch.cuda.empty_cache()
    return curve


def interp_at_drop(curve, target_drop):
    """线性插值计算当 new_drop 恰好达到 target_drop 时的 worst_old_rise / mean_old_rise。"""
    for i in range(1, len(curve)):
        d0, d1 = curve[i - 1]['new_drop'], curve[i]['new_drop']
        if d1 >= target_drop and d1 > d0:
            t = (target_drop - d0) / (d1 - d0)
            w = curve[i - 1]['worst_old_rise'] + t * (curve[i]['worst_old_rise'] - curve[i - 1]['worst_old_rise'])
            m = curve[i - 1]['mean_old_rise'] + t * (curve[i]['mean_old_rise'] - curve[i - 1]['mean_old_rise'])
            return round(w, 4), round(m, 4)
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt', help='138M 检查点路径（latest.pt / best.pt）')
    ap.add_argument('--out', default='', help='输出 JSON 路径')
    ap.add_argument('--smf_k', type=int, default=2, help='SMF 每波选中的可塑模块数（默认 2/8 = 25%）')
    ap.add_argument('--steps', type=int, default=40, help='在线学习步数')
    args = ap.parse_args()

    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    amp_dtype = torch.bfloat16 if (device == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16

    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    cfg = dict(ck['cfg'])
    step = ck.get('step', -1)
    tokens_seen = ck.get('tokens_seen', -1)
    print(f'[Load] {args.ckpt} | step={step} | tokens={tokens_seen/1e6:.1f}M | device={device}')

    domains = load_domain_tokens(enc, n_tokens=40_000)

    model = WaveNet(**cfg).to(device)
    model.load_state_dict(ck['model'])
    t0 = time.time()
    diag = test_competition_and_specialization(model, domains, device, amp_dtype)
    print('\n================ 1. 竞争 & 2. 分工 & 3. 层级 ================')
    print(f"平均负载熵 (1=完美均衡): {diag['mean_load_ent']} | 平均地址余弦 (越接近0越分散): {diag['mean_addr_cos']}")
    print(f"4领域来源选择性 src_sel: 均值={diag['mean_src_sel']} | 最大波={diag['max_src_sel']}")
    print(f"层级比 (深3波 / 浅3波): 静态tau比={diag['tau_ratio_deep_vs_shallow']}x | 路由驻留长度比={diag['dwell_ratio_deep_vs_shallow']}x")
    print(f"各领域基线 loss: {diag['domain_val_loss']}")
    print('逐波明细:')
    print(f"{'Wave':>4} {'decay':>6} {'ent':>6} {'dead':>5} {'acos':>7} {'top1':>6} {'src_sel':>8} {'tau_m':>7} {'dwell':>7}")
    for ws in diag['waves']:
        print(f"{ws['wave']:>4d} {ws['ctx_decay']:>6.3f} {ws['load_ent']:>6.3f} {ws['dead']:>5.2f} "
              f"{ws['addr_cos']:>7.4f} {ws['top1_wt']:>6.3f} {ws['src_sel']:>8.4f} {ws['tau_med']:>7.2f} {ws['dwell_tok']:>7.2f}")

    # 选出 chat 相对 edu 背景最特异的每波 smf_k 个模块
    wp = model.layers[1]
    G, N, nw = wp.G, wp.N, wp.N // wp.G
    f_chat = np.array(diag['mod_freq_by_dom']['chat'])
    f_bg = np.array(diag['mod_freq_by_dom']['edu'])
    diff = f_chat - f_bg
    selected_mods = []
    for w in range(G):
        sl = slice(w * nw, (w + 1) * nw)
        top_local = np.argsort(-diff[sl])[:args.smf_k]
        selected_mods.extend((w * nw + top_local).tolist())
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()

    # 测试 4：OOD 对话 (chat) 在线学习抗遗忘对照（旧领域：edu / story / code）
    print(f'\n================ 4. 抗遗忘同收益对照 (OOD=chat, SMF选{len(selected_mods)}/{N}模块) ================')
    T_ol, B_ol = 512, 4
    chat_train = make_batches(domains['chat'][:24576], B=B_ol, T=T_ol)
    chat_val = make_batches(domains['chat'][24576:], B=B_ol, T=T_ol)
    old_vals = {
        'edu': make_batches(domains['edu'][:12288], B=B_ol, T=T_ol),
        'story': make_batches(domains['story'][:12288], B=B_ol, T=T_ol),
        'code': make_batches(domains['code'][:12288], B=B_ol, T=T_ol),
    }

    methods = [
        ('full_ft', 1e-4),
        ('dense_lora', 2e-3),
        ('smf_lora', 4e-3),
        ('smf_mlp', 3e-4),
    ]
    curves = {}
    for mname, lr in methods:
        c = _run_online_curve(ck['model'], cfg, mname, chat_train, chat_val, old_vals,
                              selected_mods, device, amp_dtype, lr=lr, steps=args.steps, eval_every=5)
        curves[mname] = c
        best_pt = max(c, key=lambda r: r['new_drop'])
        print(f"  [{mname:10s}] max_new_drop={best_pt['new_drop']:.4f} | at_max: worst_old_rise={best_pt['worst_old_rise']:+.4f}, mean_old_rise={best_pt['mean_old_rise']:+.4f} | detail={best_pt['old_rise']}")

    print('\n--- 同收益对比表（新任务 chat_val_loss 下降相同幅度时，旧领域 loss 上升幅度：越低越抗遗忘）---')
    print(f"{'Method':<12} {'@drop=0.2 (worst/mean)':<24} {'@drop=0.4 (worst/mean)':<24} {'@drop=0.6 (worst/mean)':<24}")
    summary_table = {}
    for mname in [m[0] for m in methods]:
        row_s = {}
        cols = []
        for tgt in (0.2, 0.4, 0.6):
            w, m = interp_at_drop(curves[mname], tgt)
            row_s[str(tgt)] = {'worst': w, 'mean': m}
            cols.append(f'{w:+.4f} / {m:+.4f}' if w is not None else '未达到')
        summary_table[mname] = row_s
        print(f"{mname:<12} {cols[0]:<24} {cols[1]:<24} {cols[2]:<24}")

    res = {
        'ckpt': str(args.ckpt), 'step': step, 'tokens_seen': tokens_seen,
        'elapsed_s': round(time.time() - t0, 1),
        'diag': {k: v for k, v in diag.items() if k != 'mod_freq_by_dom'},
        'smf_selected_mods': selected_mods,
        'anti_forgetting_curves': curves,
        'anti_forgetting_table': summary_table,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'\n[Done] 结果已保存 → {args.out}')


if __name__ == '__main__':
    main()
