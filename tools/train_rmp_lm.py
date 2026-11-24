"""独立放大版 RMP 语言模型（WaveNet + BPE）预训练与采样脚本（面向 Colab Pro A100 / L4）。

特点：
1. 完全脱离外部 LLM，直接放大 pool.py 的 WaveNet（Gated DeltaNet + 慢通道 + 多尺度语境路由池）；
2. 使用 GPT-2 BPE（tiktoken，vocab=50304），eos/sep_id=50256 自动在文档边界清空循环状态；
3. 流式加载 FineWeb-Edu（sample-10BT）+ 后台线程预取打包，不占磁盘、不拖慢 A100；
4. bf16 + fla 融合内核 + 按波 gradient_checkpointing + 融合 AdamW + 原子断点续训。

用法：
  python tools/train_rmp_lm.py train --preset main --out /content/drive/MyDrive/rmp_138m --tokens 2000000000
  python tools/train_rmp_lm.py sample /content/drive/MyDrive/rmp_138m/latest.pt --prompt "Once upon a time"
"""
import argparse, json, math, os, queue, sys, threading, time
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pool import WaveNet

EOT_ID = 50256
VOCAB_SIZE = 50304  # 50257 补齐到 64 的倍数，利于 Tensor Core 对齐


def _make_decays(G, lo=0.5, hi=0.992):
    """生成浅快深慢的逐波语境衰减系数（对数等距于时间常数 tau = 1/(1-a)）。"""
    if G == 1:
        return [hi]
    tau_lo, tau_hi = 1.0 / (1.0 - lo), 1.0 / (1.0 - hi)
    return [round(1.0 - 1.0 / (tau_lo * (tau_hi / tau_lo) ** (i / (G - 1))), 5) for i in range(G)]


PRESETS = {
    # 主力档（~138M 总参数 / ~61M 激活参数，每波 8 选 2）：A100 40GB / L4 24GB
    'main': dict(
        vocab=VOCAB_SIZE, sep_id=EOT_ID, emb=640, D=640,
        M=2, d=320, m=32, hidden=640, q=320,
        G=12, N=96, k=2, pd=320, pm=32, phidden=640, qr=64,
        temp=0.1, init='spread', rule='ctx', wave_query=True, wave_ctx=True,
        ctx_decay=_make_decays(12, 0.5, 0.992), beta_max=2.0,
        bottleneck=False, read_stream=True, tie_emb=True, grad_ckpt=True,
        lora_rank=4,
    ),
    # 高竞争细粒度主力档（对齐 E129 每波 16 选 2 = 12.5% 激活；~242M 总参 / ~61M 激活）：A100 40GB 推荐
    'main_fine': dict(
        vocab=VOCAB_SIZE, sep_id=EOT_ID, emb=640, D=640,
        M=2, d=320, m=32, hidden=640, q=320,
        G=12, N=192, k=2, pd=320, pm=32, phidden=640, qr=64,
        temp=0.1, init='spread', rule='ctx', wave_query=True, wave_ctx=True,
        ctx_decay=_make_decays(12, 0.5, 0.992), beta_max=2.0,
        bottleneck=False, read_stream=True, tie_emb=True, grad_ckpt=True,
        lora_rank=4,
    ),
    # 轻量档（~38M 总参数 / ~19M 激活参数，每波 8 选 2）：快速冒烟 / T4 / 低预算验证
    'small': dict(
        vocab=VOCAB_SIZE, sep_id=EOT_ID, emb=320, D=320,
        M=2, d=160, m=32, hidden=320, q=160,
        G=8, N=64, k=2, pd=160, pm=32, phidden=320, qr=32,
        temp=0.1, init='spread', rule='ctx', wave_query=True, wave_ctx=True,
        ctx_decay=_make_decays(8, 0.5, 0.99), beta_max=2.0,
        bottleneck=False, read_stream=True, tie_emb=True, grad_ckpt=True,
        lora_rank=4,
    ),
    # 高竞争细粒度轻量档（对齐 E129 每波 16 选 2；~58M 总参 / ~19M 激活）：本地 8GB 冒烟 & 断点续训验证
    'small_fine': dict(
        vocab=VOCAB_SIZE, sep_id=EOT_ID, emb=320, D=320,
        M=2, d=160, m=32, hidden=320, q=160,
        G=8, N=128, k=2, pd=160, pm=32, phidden=320, qr=32,
        temp=0.1, init='spread', rule='ctx', wave_query=True, wave_ctx=True,
        ctx_decay=_make_decays(8, 0.5, 0.99), beta_max=2.0,
        bottleneck=False, read_stream=True, tie_emb=True, grad_ckpt=True,
        lora_rank=4,
    ),
}


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    pool_p = sum(p.numel() for p in model.layers[1].core.parameters())
    wp = model.layers[1]
    frac = (wp.k + wp.shared) / (wp.N / wp.G)
    active = int(total - pool_p * (1.0 - frac))
    return total, active


def build_optimizer(model, lr, weight_decay=0.1, betas=(0.9, 0.95)):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or 'emb' in name or 'norm' in name or name.endswith('bias') or 'b_a' in name or 'b_beta' in name or 'b_gam' in name or 'ogain' in name:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [{'params': decay, 'weight_decay': weight_decay}, {'params': no_decay, 'weight_decay': 0.0}]
    use_fused = torch.cuda.is_available()
    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=1e-8, fused=use_fused)


def get_lr(step, total_steps, max_lr, warmup_steps=500, min_lr_ratio=0.1, sched='cosine'):
    min_lr = max_lr * min_lr_ratio
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    if sched == 'wsd':
        decay_start = int(total_steps * 0.85)
        if step < decay_start:
            return max_lr
        prog = min(1.0, (step - decay_start) / max(1, total_steps - decay_start))
        return max_lr - (max_lr - min_lr) * prog
    prog = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * prog))


class StreamingPacker:
    """后台线程流式拉取 HuggingFace 数据集，GPT-2 BPE 分词并打包成定长 (B, T+1) 批次（精确记录 docs_seen 支持无重复续训）。"""
    def __init__(self, dataset_name, dataset_config, batch_size, seq_len, val_tokens=200_000, seed=42, skip_docs=0):
        import tiktoken
        from datasets import load_dataset
        self.enc = tiktoken.get_encoding('gpt2')
        self.B, self.T = batch_size, seq_len
        self.dataset_name, self.dataset_config = dataset_name, dataset_config
        self.seed, self.skip_docs = seed, skip_docs
        self.docs_seen = skip_docs
        self._stop = threading.Event()
        self._q = queue.Queue(maxsize=8)

        # 先同步抽取固定验证集（最前 val_tokens 个 token，保证各次运行可比）
        ds_val = load_dataset(dataset_name, dataset_config, split='train', streaming=True)
        val_buf, val_docs = [], 0
        for row in ds_val:
            text = row.get('text', '')
            if not text:
                continue
            val_buf.extend(self.enc.encode_ordinary(text))
            val_buf.append(EOT_ID)
            val_docs += 1
            if len(val_buf) >= val_tokens:
                break
        self.val_ids = torch.tensor(val_buf[:val_tokens], dtype=torch.long)
        self.val_docs = val_docs
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        from datasets import load_dataset
        epoch = 0
        need = self.B * (self.T + 1)
        buf = []
        docs_consumed = self.skip_docs
        while not self._stop.is_set():
            ds = load_dataset(self.dataset_name, self.dataset_config, split='train', streaming=True)
            if epoch == 0:
                ds = ds.skip(self.val_docs + self.skip_docs)
            else:
                ds = ds.skip(self.val_docs).shuffle(seed=self.seed + epoch, buffer_size=10_000)
            for row in ds:
                if self._stop.is_set():
                    return
                text = row.get('text', '')
                if not text:
                    continue
                buf.extend(self.enc.encode_ordinary(text))
                buf.append(EOT_ID)
                docs_consumed += 1
                while len(buf) >= need:
                    chunk = torch.tensor(buf[:need], dtype=torch.long).view(self.B, self.T + 1)
                    buf = buf[need:]
                    while not self._stop.is_set():
                        try:
                            self._q.put((chunk, docs_consumed), timeout=0.5)
                            break
                        except queue.Full:
                            continue
            epoch += 1

    def next_batch(self, device):
        xy, d_seen = self._q.get()
        self.docs_seen = d_seen
        return xy[:, :-1].to(device, non_blocking=True), xy[:, 1:].to(device, non_blocking=True)

    def close(self):
        self._stop.set()


class LocalFilePacker:
    """本地文本文件打包器（用于本地 8GB 卡离线冒烟与精确断点续训验证，无需联网/tiktoken）。"""
    def __init__(self, path, batch_size, seq_len, val_tokens=32_768, skip_tokens=0):
        import numpy as np
        self._np = np
        raw = np.fromfile(path, dtype=np.uint8)
        self.B, self.T = batch_size, seq_len
        self.val_ids = torch.from_numpy(raw[:val_tokens].astype(np.int64))
        self.val_docs = 0
        self.train_data = raw[val_tokens:]
        self.pos = int(skip_tokens % max(1, len(self.train_data) - batch_size * (seq_len + 1)))
        self.docs_seen = int(skip_tokens)

    def next_batch(self, device):
        need = self.B * (self.T + 1)
        if self.pos + need > len(self.train_data):
            self.pos = 0
        chunk = torch.from_numpy(self.train_data[self.pos : self.pos + need].astype(self._np.int64)).view(self.B, self.T + 1)
        self.pos += need
        self.docs_seen += need
        return chunk[:, :-1].to(device, non_blocking=True), chunk[:, 1:].to(device, non_blocking=True)

    def close(self):
        pass


@torch.no_grad()
def evaluate(model, val_ids, batch_size, seq_len, device, amp_dtype, max_batches=16):
    model.eval()
    total_loss, n = 0.0, 0
    stride = batch_size * (seq_len + 1)
    for i in range(0, min(len(val_ids) - stride, max_batches * stride), stride):
        xy = val_ids[i:i + stride].view(batch_size, seq_len + 1).to(device)
        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=(device == 'cuda')):
            logits, _ = model(xy[:, :-1])
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), xy[:, 1:].reshape(-1))
        total_loss += float(loss.item())
        n += 1
    model.train()
    mean_loss = total_loss / max(1, n)
    return mean_loss, math.exp(min(mean_loss, 20.0))


@torch.no_grad()
def generate_text(model, prompt, max_new_tokens=120, temperature=0.8, top_k=50, device='cuda'):
    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
    ids = enc.encode_ordinary(prompt) or [EOT_ID]
    was_training = model.training
    prev_sparse = model.layers[1].sparse_active
    model.eval()
    model.layers[1].sparse_active = True  # 推理期开启 O(k) 稀疏执行

    x = torch.tensor([ids], dtype=torch.long, device=device)
    amp_dtype = torch.bfloat16 if (device == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16
    with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=(device == 'cuda')):
        logits, state = model(x)
    out = list(ids)
    for _ in range(max_new_tokens):
        nxt_logits = logits[0, -1, :50257].float() / max(temperature, 1e-5)
        if top_k > 0:
            v, _ = torch.topk(nxt_logits, min(top_k, nxt_logits.size(-1)))
            nxt_logits = nxt_logits.masked_fill(nxt_logits < v[-1], float('-inf'))
        probs = F.softmax(nxt_logits, dim=-1)
        nxt = int(torch.multinomial(probs, 1).item())
        out.append(nxt)
        if nxt == EOT_ID:
            break
        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=(device == 'cuda')):
            logits, state = model(torch.tensor([[nxt]], dtype=torch.long, device=device), state)

    model.layers[1].sparse_active = prev_sparse
    if was_training:
        model.train()
    return enc.decode([t for t in out if t < 50257])


def save_ckpt(path, model, opt, cfg, step, tokens_seen, best_val, docs_seen=0):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save({
        'model': model.state_dict(),
        'opt': opt.state_dict(),
        'cfg': cfg,
        'step': step,
        'tokens_seen': tokens_seen,
        'docs_seen': docs_seen,
        'best_val': best_val,
    }, tmp)
    os.replace(tmp, path)


def cmd_train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    use_bf16 = (device == 'cuda') and torch.cuda.is_bf16_supported() and (torch.cuda.get_device_capability()[0] >= 8)
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda' and not use_bf16))

    cfg = dict(PRESETS[args.preset])
    if args.no_ckpt:
        cfg['grad_ckpt'] = False
    if args.shared > 0:
        cfg['shared'] = args.shared
    model = WaveNet(**cfg).to(device)
    total_p, active_p = count_params(model)
    tokens_per_step = args.batch * args.T * args.grad_accum
    total_steps = args.steps if args.steps > 0 else max(1, args.tokens // tokens_per_step)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    opt = build_optimizer(model, args.lr, weight_decay=args.wd)

    start_step, tokens_seen, docs_seen, best_val = 0, 0, 0, float('inf')
    latest_path = out_dir / 'latest.pt'
    if args.resume and latest_path.exists():
        ck = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        start_step = int(ck['step'])
        tokens_seen = int(ck['tokens_seen'])
        docs_seen = int(ck.get('docs_seen', tokens_seen // 800))
        best_val = float(ck.get('best_val', float('inf')))
        print(f'[Resume] 从 {latest_path} 恢复：step={start_step}, tokens={tokens_seen/1e6:.2f}M, docs_seen={docs_seen}, best_val={best_val:.4f}')

    gpu_name = torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'
    print(f'[Model] preset={args.preset} | total={total_p/1e6:.2f}M | active={active_p/1e6:.2f}M | device={gpu_name} | dtype={amp_dtype}')
    print(f'[Train] batch={args.batch} x T={args.T} x accum={args.grad_accum} = {tokens_per_step:,} tok/step | total_steps={total_steps:,} ({total_steps*tokens_per_step/1e9:.2f}B tok)')

    if args.local_file:
        packer = LocalFilePacker(args.local_file, args.batch, args.T, skip_tokens=docs_seen)
        print(f'[Data] 本地离线文件 {args.local_file} 就绪，验证集 {len(packer.val_ids):,} tokens (skip={docs_seen})')
    else:
        skip_docs = docs_seen if start_step > 0 else 0
        packer = StreamingPacker(args.dataset, args.dataset_config or None, args.batch, args.T, seed=args.seed, skip_docs=skip_docs)
        print(f'[Data] {args.dataset} ({args.dataset_config}) 就绪，验证集 {len(packer.val_ids):,} tokens ({packer.val_docs} docs, skip={skip_docs})')

    t_start = time.time()
    t_log = time.time()
    loss_acc = 0.0
    model.train()

    try:
        for step in range(start_step, total_steps):
            lr = get_lr(step, total_steps, args.lr, warmup_steps=args.warmup, sched=args.sched)
            for pg in opt.param_groups:
                pg['lr'] = lr

            opt.zero_grad(set_to_none=True)
            step_loss = 0.0
            for _ in range(args.grad_accum):
                x, y = packer.next_batch(device)
                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=(device == 'cuda')):
                    logits, _ = model(x)
                    loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1))
                    loss = loss / args.grad_accum
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                step_loss += float(loss.item())

            if scaler.is_enabled():
                scaler.unscale_(opt)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                scaler.step(opt)
                scaler.update()
            else:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                opt.step()

            tokens_seen += tokens_per_step
            loss_acc += step_loss

            if (step + 1) % args.log_every == 0 or step == start_step:
                dt = max(1e-6, time.time() - t_log)
                n_steps = 1 if step == start_step else args.log_every
                tok_s = (n_steps * tokens_per_step) / dt
                avg_loss = loss_acc / n_steps
                depth, dead, _ = model.layers[1].stats()
                mem_gb = torch.cuda.max_memory_allocated() / 1e9 if device == 'cuda' else 0.0
                print(
                    f'step {step+1:>6d}/{total_steps} | tok {tokens_seen/1e6:>7.1f}M | '
                    f'loss {avg_loss:.4f} (ppl {math.exp(min(avg_loss, 20)):.1f}) | '
                    f'lr {lr:.2e} | gn {grad_norm:.2f} | {tok_s:,.0f} tok/s | '
                    f'dead {dead:.2f} | vram {mem_gb:.1f}GB'
                )
                loss_acc = 0.0
                t_log = time.time()

            if (step + 1) % args.eval_every == 0 or (step + 1) == total_steps:
                val_loss, val_ppl = evaluate(model, packer.val_ids, args.batch, args.T, device, amp_dtype)
                elapsed_h = (time.time() - t_start) / 3600
                print(f'>>> [Eval @ step {step+1}] val_loss={val_loss:.4f} | val_ppl={val_ppl:.2f} | elapsed={elapsed_h:.2f}h')
                if not args.local_file:
                    sample = generate_text(model, 'The most important discovery in science is', max_new_tokens=80, device=device)
                    print(f'>>> [Sample] {sample!r}')
                if val_loss < best_val:
                    best_val = val_loss
                    save_ckpt(out_dir / 'best.pt', model, opt, cfg, step + 1, tokens_seen, best_val, docs_seen=packer.docs_seen)
                with open(out_dir / 'metrics.jsonl', 'a', encoding='utf-8') as f:
                    f.write(json.dumps({
                        'step': step + 1, 'tokens': tokens_seen, 'docs_seen': packer.docs_seen,
                        'val_loss': round(val_loss, 4), 'val_ppl': round(val_ppl, 2), 'elapsed_h': round(elapsed_h, 3),
                    }) + '\n')

            if (step + 1) % args.save_every == 0 or (step + 1) == total_steps:
                save_ckpt(latest_path, model, opt, cfg, step + 1, tokens_seen, best_val, docs_seen=packer.docs_seen)
                print(f'[Save] 已保存断点 → {latest_path} (docs_seen={packer.docs_seen})')
    finally:
        packer.close()


def cmd_sample(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = WaveNet(**ck['cfg']).to(device)
    model.load_state_dict(ck['model'])
    text = generate_text(model, args.prompt, max_new_tokens=args.max_tokens, temperature=args.temp, top_k=args.top_k, device=device)
    print(text)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)

    tr = sub.add_parser('train')
    tr.add_argument('--preset', choices=list(PRESETS), default='main')
    tr.add_argument('--out', required=True, help='输出与断点目录（Colab 上指向 Google Drive）')
    tr.add_argument('--dataset', default='HuggingFaceFW/fineweb-edu')
    tr.add_argument('--dataset_config', default='sample-10BT')
    tr.add_argument('--batch', type=int, default=16)
    tr.add_argument('--T', type=int, default=1024)
    tr.add_argument('--grad_accum', type=int, default=4, help='总 batch_tokens = batch * T * grad_accum')
    tr.add_argument('--tokens', type=int, default=2_000_000_000, help='目标训练 token 总数（steps=0 时生效）')
    tr.add_argument('--steps', type=int, default=0, help='若 > 0 则按步数停止')
    tr.add_argument('--lr', type=float, default=6e-4)
    tr.add_argument('--wd', type=float, default=0.1)
    tr.add_argument('--warmup', type=int, default=500)
    tr.add_argument('--sched', choices=['cosine', 'wsd'], default='wsd')
    tr.add_argument('--log_every', type=int, default=20)
    tr.add_argument('--eval_every', type=int, default=500)
    tr.add_argument('--save_every', type=int, default=500)
    tr.add_argument('--seed', type=int, default=42)
    tr.add_argument('--resume', action='store_true')
    tr.add_argument('--no_ckpt', action='store_true', help='关闭按波 gradient checkpointing')
    tr.add_argument('--shared', type=int, default=0, help='每波常驻共享模块数（预训练正常训，持续学习时冻结）')
    tr.add_argument('--local_file', default='', help='本地离线文本路径（用于 8GB 本地冒烟与断点续训验证）')

    sp = sub.add_parser('sample')
    sp.add_argument('ckpt')
    sp.add_argument('--prompt', default='Once upon a time')
    sp.add_argument('--max_tokens', type=int, default=200)
    sp.add_argument('--temp', type=float, default=0.8)
    sp.add_argument('--top_k', type=int, default=50)

    args = p.parse_args()
    if args.cmd == 'train':
        cmd_train(args)
    else:
        cmd_sample(args)


if __name__ == '__main__':
    main()
