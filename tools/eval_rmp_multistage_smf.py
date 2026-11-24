"""E128：基于 E110 实证主干（纯 WaveNet + smf_mlp，无思考/海马未证实模块）的多阶段真实持续学习验证。

在真实多域文本上推进持续学习主目标：
1. 底座：在 `data/task3-train.txt`（3 个真实领域：stories / wikiHow / simplewiki）上预训练纯 `WaveNet`（`N=64, k=4, G=8`, `wave_ctx=True`, `read_stream=True`）；
2. 多阶段顺序 OOD 持续学习（零重放、纯在线流式）：
   - 阶段 1：在线学习 64KB 真实对话（`data/src-soda.txt`）；
   - 阶段 2：继续在线学习 64KB 真实学科教育（`data/src-cosmo_khan.txt`）；
3. 同一底座严格三臂对照：
   - `full_ft`：全参在线微调；
   - `smf_static_bg`：E110 既有做法（`bg_freq` 固定在预训练分布）；
   - `smf_cumulative_bg`：生命周期累积背景频率（每块以慢 EMA 更新 `bg_freq`，无任务边界标签，自然保护阶段 1 已招募模块不被阶段 2 覆写）。
4. 评测：阶段 1 后与阶段 2 后分别测 3 个旧预训练域最差遗忘、`SODA` 降损与阶段 2 后 `SODA` 留存（阶段 1 遗忘度）、`Khan` 降损。
"""
from __future__ import annotations

import argparse
import copy
import datetime
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from online import _masked_step
from pool import WaveNet


def make_decays(G: int, lo: float = 0.5, hi: float = 0.99) -> List[float]:
    if G <= 1:
        return [hi]
    tau_lo, tau_hi = 1.0 / (1.0 - lo), 1.0 / (1.0 - hi)
    return [round(1.0 - 1.0 / (tau_lo * (tau_hi / tau_lo) ** (i / (G - 1))), 5) for i in range(G)]


def detach_state(st):
    if isinstance(st, torch.Tensor):
        return st.detach()
    if isinstance(st, (tuple, list)):
        return type(st)(detach_state(x) for x in st)
    return st


@torch.no_grad()
def eval_bpc(model: WaveNet, data_bytes: bytes, device: torch.device, chunk: int = 256) -> float:
    model.eval()
    ids = torch.tensor(list(data_bytes), dtype=torch.long, device=device)
    st = None
    tot, n = 0.0, 0
    last = None
    for i in range(0, len(ids), chunk):
        blk = ids[i : i + chunk]
        seq = blk if last is None else torch.cat([last, blk])
        last = blk[-1:]
        if len(seq) < 2:
            continue
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            lg, st = model(seq[None, :-1], st)
        tot += F.cross_entropy(lg[0].float(), seq[1:], reduction="sum").item()
        n += len(seq) - 1
    return round(tot / max(1, n) / math.log(2), 4)


def pretrain_wavenet(
    tr: np.ndarray,
    device: torch.device,
    steps: int = 2500,
    batch_size: int = 16,
    seq_len: int = 256,
    lr: float = 1.5e-3,
    seed: int = 42,
) -> Tuple[WaveNet, Dict[str, object]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = WaveNet(
        vocab=256, sep_id=30, emb=128, D=128,
        M=2, d=96, m=16, hidden=192, q=64,
        G=8, N=64, k=4, pd=96, pm=16, phidden=192, qr=32,
        temp=0.1, init="spread", rule="ctx", wave_query=True, wave_ctx=True,
        ctx_decay=make_decays(8, 0.5, 0.99), beta_max=2.0,
        bottleneck=False, read_stream=True, tie_emb=True,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05, betas=(0.9, 0.95))
    max_start = len(tr) - seq_len - 2
    t0 = time.time()
    for step in range(1, steps + 1):
        warm = max(1, int(steps * 0.05))
        if step <= warm:
            cur_lr = lr * step / warm
        else:
            prog = (step - warm) / max(1, steps - warm)
            cur_lr = lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog)))
        for g in opt.param_groups:
            g["lr"] = cur_lr
        starts = rng.integers(0, max_start, size=batch_size)
        batch_t = torch.from_numpy(np.stack([tr[s : s + seq_len + 1] for s in starts]).astype(np.int64)).to(device)
        model.train()
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            lg, _ = model(batch_t[:, :-1])
            loss = F.cross_entropy(lg.float().reshape(-1, 256), batch_t[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % max(1, steps // 5) == 0 or step == steps:
            bpc = float(loss.item() / math.log(2))
            bps = (step * batch_size * seq_len) / max(1e-3, time.time() - t0)
            print(f"  [pretrain s={seed}] step {step:4d}/{steps} | bpc={bpc:.4f} ({bps/1e3:.1f}k B/s, {time.time()-t0:.1f}s)", flush=True)
    _, dead, _ = model.layers[1].stats()
    return model, {"params": sum(p.numel() for p in model.parameters()), "train_sec": round(time.time() - t0, 1), "dead": dead}


@torch.no_grad()
def calibrate_bg(model: WaveNet, bg_bytes: bytes, device: torch.device, chunk: int = 256) -> torch.Tensor:
    pool = model.layers[1]
    ids = torch.tensor(list(bg_bytes), dtype=torch.long, device=device)
    cnt = torch.zeros(pool.N, device=device)
    model.eval()
    pool.capture = []
    st = None
    try:
        for i in range(0, len(ids) - 1, chunk):
            blk = ids[i : i + chunk][None]
            if blk.shape[1] < 2:
                continue
            pool.capture.clear()
            _, st = model(blk, st)
            for _, sel, _, _ in pool.capture:
                cnt += sel.sum((0, 1))
    finally:
        pool.capture = None
    return cnt / cnt.sum().clamp(min=1.0)


def run_online_stage(
    model: WaveNet,
    mode: str,
    stream_bytes: bytes,
    bg_freq: torch.Tensor,
    device: torch.device,
    lr: float = 1e-3,
    smf_k: int = 16,
    smf_eps: float = 1e-2,
    chunk: int = 128,
    bg_ema: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pool = model.layers[1]
    core = pool.core
    pool.plastic = False
    if mode == "full_ft":
        for p in model.parameters():
            p.requires_grad_(True)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        mlp_params = None
    else:
        for p in model.parameters():
            p.requires_grad_(False)
        mlp_params = [core.W1, core.b1, core.W2]
        for p in mlp_params:
            p.requires_grad_(True)
        opt = torch.optim.AdamW(mlp_params, lr=lr)

    ids = torch.tensor(list(stream_bytes), dtype=torch.long, device=device)
    st = None
    bg = bg_freq.clone()
    used_counts = torch.zeros(pool.N, device=device)
    routed = (pool.active > 0) & (pool.is_shared == 0)

    for i in range(0, len(ids) - 1, chunk):
        blk = ids[i : min(i + chunk, len(ids))][None]
        if blk.shape[1] < 2:
            continue
        model.train()
        opt.zero_grad(set_to_none=True)
        if mode == "full_ft":
            lg, st_new = model(blk[:, :-1], st)
            loss = F.cross_entropy(lg.float().reshape(-1, 256), blk[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        else:
            pool.capture = []
            try:
                lg, st_new = model(blk[:, :-1], st)
                loss = F.cross_entropy(lg.float().reshape(-1, 256), blk[:, 1:].reshape(-1))
                loss.backward()
                cur = torch.zeros(pool.N, device=device)
                for _, sel, _, _ in pool.capture:
                    cur += sel.sum((0, 1))
            finally:
                pool.capture = None
            cur_freq = cur / cur.sum().clamp(min=1.0)
            score = (cur_freq / (bg + smf_eps)).masked_fill(~routed, -1e9)
            top_idx = score.topk(min(smf_k, int(routed.sum().item()))).indices
            keep_vec = torch.zeros(pool.N, device=device)
            keep_vec[top_idx] = 1.0
            used_counts += keep_vec
            torch.nn.utils.clip_grad_norm_(mlp_params, 1.0)
            _masked_step(opt, mlp_params, keep_vec)
            if mode == "smf_cumulative_bg":
                bg = (1.0 - bg_ema) * bg + bg_ema * cur_freq
        st = detach_state(st_new)
    return bg, used_counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--online_lr", type=float, default=1e-3)
    ap.add_argument("--smf_k", type=int, default=16)
    ap.add_argument("--stage_bytes", type=int, default=65536)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = np.fromfile(ROOT / "data/task3-train.txt", dtype=np.uint8)
    va = np.fromfile(ROOT / "data/task3-valid.txt", dtype=np.uint8)
    va_src = np.fromfile(ROOT / "data/task3-valid.src", dtype=np.uint8)
    soda = np.fromfile(ROOT / "data/src-soda.txt", dtype=np.uint8)
    khan = np.fromfile(ROOT / "data/src-cosmo_khan.txt", dtype=np.uint8)

    old_probes = {f"src{int(s)}": bytes(va[np.flatnonzero(va_src == s)[:8192]]) for s in np.unique(va_src)}
    bg_bytes = bytes(tr[-65536:])
    soda_stream = bytes(soda[: args.stage_bytes])
    soda_probe = bytes(soda[args.stage_bytes : args.stage_bytes + 8192])
    khan_stream = bytes(khan[: args.stage_bytes])
    khan_probe = bytes(khan[args.stage_bytes : args.stage_bytes + 8192])

    all_seeds = []
    for seed in args.seeds:
        print(f"\n=== Seed {seed}: Pretraining E110-style WaveNet on task3 ===", flush=True)
        base, pinfo = pretrain_wavenet(tr, device, steps=args.steps, batch_size=args.batch, seq_len=args.seq_len, lr=args.lr, seed=seed)
        bg0 = calibrate_bg(base, bg_bytes, device)
        old_0 = {k: eval_bpc(base, v, device) for k, v in old_probes.items()}
        soda_0 = eval_bpc(base, soda_probe, device)
        khan_0 = eval_bpc(base, khan_probe, device)
        print(f"  Base bpc: old={old_0} | soda={soda_0} | khan={khan_0}", flush=True)

        seed_res = {"seed": seed, "pretrain": pinfo, "base": {"old": old_0, "soda": soda_0, "khan": khan_0}, "modes": {}}
        for mode in ("full_ft", "smf_static_bg", "smf_cumulative_bg"):
            m = copy.deepcopy(base).to(device)
            # Stage 1: SODA
            bg1, u1 = run_online_stage(m, mode, soda_stream, bg0, device, lr=args.online_lr, smf_k=args.smf_k)
            old_1 = {k: eval_bpc(m, v, device) for k, v in old_probes.items()}
            soda_1 = eval_bpc(m, soda_probe, device)
            # Stage 2: Khan
            bg2, u2 = run_online_stage(m, mode, khan_stream, bg1, device, lr=args.online_lr, smf_k=args.smf_k)
            old_2 = {k: eval_bpc(m, v, device) for k, v in old_probes.items()}
            soda_2 = eval_bpc(m, soda_probe, device)
            khan_2 = eval_bpc(m, khan_probe, device)

            # 统计两阶段模块重叠度（Jaccard）
            jacc = 0.0
            if mode != "full_ft":
                s1 = u1 > 0
                s2 = u2 > 0
                jacc = round(float((s1 & s2).sum().item() / max(1, (s1 | s2).sum().item())), 4)

            s1_old_rise = {k: round(old_1[k] - old_0[k], 4) for k in old_0}
            s2_old_rise = {k: round(old_2[k] - old_0[k], 4) for k in old_0}
            rec = {
                "stage1_soda_drop": round(soda_0 - soda_1, 4),
                "stage1_worst_old_rise": max(s1_old_rise.values()),
                "stage2_khan_drop": round(khan_0 - khan_2, 4),
                "stage2_soda_forget": round(soda_2 - soda_1, 4),
                "stage2_soda_net_drop": round(soda_0 - soda_2, 4),
                "stage2_worst_old_rise": max(s2_old_rise.values()),
                "stage2_mean_old_rise": round(sum(s2_old_rise.values()) / len(s2_old_rise), 4),
                "stage12_module_jaccard": jacc,
            }
            seed_res["modes"][mode] = rec
            print(
                f"  [{mode:17s}] S1: soda_drop={rec['stage1_soda_drop']:+.4f} worst_old={rec['stage1_worst_old_rise']:+.4f} | "
                f"S2: khan_drop={rec['stage2_khan_drop']:+.4f} soda_forget={rec['stage2_soda_forget']:+.4f} "
                f"worst_old={rec['stage2_worst_old_rise']:+.4f} jacc={jacc:.3f}",
                flush=True,
            )
        all_seeds.append(seed_res)

    summary = {}
    for mode in ("full_ft", "smf_static_bg", "smf_cumulative_bg"):
        summary[mode] = {
            k: round(float(np.mean([s["modes"][mode][k] for s in all_seeds])), 4)
            for k in all_seeds[0]["modes"][mode]
        }

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = ROOT / "runs" / f"{ts}_rmp_multistage_smf.json"
    out_path.write_text(json.dumps({"args": vars(args), "summary": summary, "seeds": all_seeds}, indent=2), encoding="utf-8")
    print(f"\nSaved E128 multi-stage SMF results to {out_path}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
