"""E127：纯粹遵循《苦涩的教训》的潜空间搜索 RMP（A3）真实多域文本验证。

零手工特征、零脚本监督、零外部 LLM、零 Self-Attention，纯在真实文本（task3 三域 + OOD SODA 对话）上检验：
1. 同参数潜空间搜索增益：`lockstep_k1` (K=1) vs `latent_search_k3` (K=3，权重共享、O(1) 状态) 的三域验证 bpc；
2. 涌现自适应算力：`halt_gate` 在无难度监督下，每字节期望思考拍数 `exp_steps` 与字节熵/惊讶度的相关性；
3. 跨拍动态路由分化：第 0 / 1 / 2 拍在同一个 `WavePool` 里选中的模块 Jaccard 重叠度（检验是否按拍动态招募不同专家）；
4. 真实多域在线抗遗忘：`smf_mlp` vs `full_ft` 在线阅读 64KB OOD SODA 对话，记录 `chat_drop -> worst_old_rise` 同收益曲线。
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

from rmp_search import LatentSearchWaveNet, SearchSMFOnlineLearner


def load_task3_and_soda():
    tr = np.fromfile(ROOT / "data/task3-train.txt", dtype=np.uint8)
    va = np.fromfile(ROOT / "data/task3-valid.txt", dtype=np.uint8)
    va_src = np.fromfile(ROOT / "data/task3-valid.src", dtype=np.uint8)
    soda = np.fromfile(ROOT / "data/src-soda.txt", dtype=np.uint8)
    return tr, va, va_src, soda


def build_old_probes(va: np.ndarray, va_src: np.ndarray, probe_bytes: int = 8192) -> Dict[str, bytes]:
    probes = {}
    for s in np.unique(va_src):
        idx = np.flatnonzero(va_src == s)
        probes[f"src{int(s)}"] = bytes(va[idx[:probe_bytes]])
    return probes


@torch.no_grad()
def eval_bpc_and_adaptive(
    model: LatentSearchWaveNet,
    data_bytes: bytes,
    device: torch.device,
    chunk: int = 256,
) -> Dict[str, float]:
    model.eval()
    ids = torch.tensor(list(data_bytes), dtype=torch.long, device=device)
    st = None
    tot_nll = 0.0
    tot_n = 0
    all_steps, all_ent, all_nll = [], [], []
    last = None
    for i in range(0, len(ids), chunk):
        blk = ids[i : i + chunk]
        seq = blk if last is None else torch.cat([last, blk])
        last = blk[-1:]
        if len(seq) < 2:
            continue
        x, y = seq[None, :-1], seq[None, 1:]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            lp, st, info = model(x, state=st, return_ticks=True)
        nll = -lp[0].gather(1, y[0, :, None]).squeeze(1)  # (L,)
        probs = lp[0].exp()
        ent = -(probs * lp[0]).sum(dim=-1)                # (L,)
        tot_nll += float(nll.sum().item())
        tot_n += len(y[0])
        all_steps.append(info["exp_steps"][0].float().cpu())
        all_ent.append(ent.float().cpu())
        all_nll.append(nll.float().cpu())

    steps_t = torch.cat(all_steps)
    ent_t = torch.cat(all_ent)
    nll_t = torch.cat(all_nll)

    def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
        a0, b0 = a - a.mean(), b - b.mean()
        denom = (a0.norm() * b0.norm()).item()
        return round(float((a0 * b0).sum().item() / denom), 4) if denom > 1e-8 else 0.0

    return {
        "bpc": round(tot_nll / max(1, tot_n) / math.log(2), 4),
        "exp_steps": round(float(steps_t.mean().item()), 3),
        "steps_std": round(float(steps_t.std().item()), 4),
        "corr_entropy": _pearson(steps_t, ent_t),
        "corr_surprisal": _pearson(steps_t, nll_t),
    }


@torch.no_grad()
def measure_cross_tick_routing(
    model: LatentSearchWaveNet,
    data_bytes: bytes,
    device: torch.device,
) -> Dict[str, float]:
    """测量同一 WavePool 在不同潜空间拍（tick 0 / 1 / 2）选中模块集合的平均 Jaccard 重叠。"""
    if model.max_think <= 1:
        return {"tick01_jaccard": 1.0, "tick02_jaccard": 1.0}
    model.eval()
    pool = model.layers[1]
    ids = torch.tensor(list(data_bytes[:512]), dtype=torch.long, device=device)[None]
    pool.capture = []
    try:
        model(ids)
        caps = list(pool.capture)
    finally:
        pool.capture = None
    G = pool.G
    # caps 长度 = max_think * G；按 tick 切分，每个 tick 合并各波 sel (1, T, N)
    tick_sels = []
    for tk in range(model.max_think):
        s_tk = sum(c[1] for c in caps[tk * G : (tk + 1) * G]) > 0  # (1, T, N) bool
        tick_sels.append(s_tk)

    def _jacc(a: torch.Tensor, b: torch.Tensor) -> float:
        inter = (a & b).float().sum(-1)
        union = (a | b).float().sum(-1).clamp(min=1.0)
        return round(float((inter / union).mean().item()), 4)

    out = {"tick01_jaccard": _jacc(tick_sels[0], tick_sels[1])}
    if model.max_think >= 3:
        out["tick02_jaccard"] = _jacc(tick_sels[0], tick_sels[2])
        out["tick12_jaccard"] = _jacc(tick_sels[1], tick_sels[2])
    return out


def train_model(
    name: str,
    max_think: int,
    tr: np.ndarray,
    device: torch.device,
    steps: int,
    batch_size: int,
    seq_len: int,
    lr: float,
    ponder_w: float,
    seed: int,
) -> Tuple[LatentSearchWaveNet, Dict[str, object]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = LatentSearchWaveNet(
        vocab=256, D=128, M=2, d=96, m=16, hidden=192, q=64,
        N=48, k=4, G=4, pd=96, pm=16, phidden=192, qr=32,
        beta_max=2.0, sep_id=30, max_think=max_think,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05, betas=(0.9, 0.95))

    max_start = len(tr) - seq_len - 2
    t0 = time.time()
    last_m = {}
    for step in range(1, steps + 1):
        # 余弦学习率（含 5% warmup）
        warm = max(1, int(steps * 0.05))
        if step <= warm:
            cur_lr = lr * step / warm
        else:
            prog = (step - warm) / max(1, steps - warm)
            cur_lr = lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog)))
        for g in opt.param_groups:
            g["lr"] = cur_lr

        starts = rng.integers(0, max_start, size=batch_size)
        batch_np = np.stack([tr[s : s + seq_len + 1] for s in starts], axis=0)
        batch_t = torch.from_numpy(batch_np.astype(np.int64)).to(device, non_blocking=True)

        model.train()
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            loss, m, _ = model.compute_loss(batch_t[:, :-1], batch_t[:, 1:], ponder_w=ponder_w, prior_p=0.5)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last_m = m
        if step % max(1, steps // 5) == 0 or step == steps:
            tok_s = (step * batch_size * seq_len) / max(1e-3, time.time() - t0)
            print(
                f"  [{name} s={seed}] step {step:4d}/{steps} | bpc={m['bpc']:.4f} "
                f"steps={m['exp_steps']:.2f} ({tok_s/1e3:.1f}k B/s, {time.time()-t0:.1f}s)",
                flush=True,
            )

    depth, dead, sizes = model.layers[1].stats()
    return model, {
        "params": sum(p.numel() for p in model.parameters()),
        "train_sec": round(time.time() - t0, 1),
        "last_train": last_m,
        "pool_dead": dead,
        "pool_sizes": sizes,
    }


def run_real_smf_continual(
    base_model: LatentSearchWaveNet,
    tr: np.ndarray,
    old_probes: Dict[str, bytes],
    soda: np.ndarray,
    device: torch.device,
    seq_bytes: int = 65536,
    eval_every: int = 16384,
    chunk: int = 128,
    smf_k: int = 12,
    lr: float = 1e-3,
) -> Dict[str, object]:
    """真实 OOD 对话（SODA）在线持续学习：对比 `smf_mlp` vs `full_ft` 的 `chat_drop -> worst_old_rise` 曲线。"""
    bg_bytes = bytes(tr[-65536:])
    chat_stream = bytes(soda[:seq_bytes])
    chat_probe = bytes(soda[seq_bytes : seq_bytes + 8192])

    old_before = {k: eval_bpc_and_adaptive(base_model, v, device)["bpc"] for k, v in old_probes.items()}
    chat_before = eval_bpc_and_adaptive(base_model, chat_probe, device)["bpc"]
    out = {"old_before": old_before, "chat_before": chat_before}

    for mode in ("smf_mlp", "full_ft"):
        m = copy.deepcopy(base_model).to(device)
        if mode == "smf_mlp":
            learner = SearchSMFOnlineLearner(m, lr=lr, smf_k=smf_k)
            learner.calibrate_background(bg_bytes, chunk=256)
        else:
            m.layers[1].plastic = False
            opt = torch.optim.AdamW(m.parameters(), lr=lr)

        curve = []
        ids_all = torch.tensor(list(chat_stream), dtype=torch.long, device=device)
        st = None
        for pos in range(0, len(ids_all) - 1, eval_every):
            end = min(pos + eval_every, len(ids_all))
            for i in range(pos, end - 1, chunk):
                blk = ids_all[i : min(i + chunk, end)][None]
                if blk.shape[1] < 2:
                    continue
                if mode == "smf_mlp":
                    _, st, _ = learner.step_chunk(blk, state=st, ponder_w=0.0)
                else:
                    m.train()
                    opt.zero_grad(set_to_none=True)
                    loss, _, st_new = m.compute_loss(blk[:, :-1], blk[:, 1:], state=st, ponder_w=0.0)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                    opt.step()
                    st = _detach(st_new)

            old_now = {k: eval_bpc_and_adaptive(m, v, device)["bpc"] for k, v in old_probes.items()}
            chat_now = eval_bpc_and_adaptive(m, chat_probe, device)["bpc"]
            rises = {k: round(old_now[k] - old_before[k], 4) for k in old_before}
            curve.append({
                "bytes": end,
                "chat_drop": round(chat_before - chat_now, 4),
                "worst_old_rise": max(rises.values()),
                "mean_old_rise": round(sum(rises.values()) / len(rises), 4),
                "rises": rises,
            })
        out[mode] = curve
    return out


def _detach(st):
    if isinstance(st, torch.Tensor):
        return st.detach()
    if isinstance(st, (tuple, list)):
        return type(st)(_detach(x) for x in st)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--ponder_w", type=float, default=0.005)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    ap.add_argument("--online_bytes", type=int, default=65536)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va, va_src, soda = load_task3_and_soda()
    old_probes = build_old_probes(va, va_src, probe_bytes=8192)
    print(f"Device: {device} | task3 train={len(tr)/1e6:.1f}MB | probes={list(old_probes.keys())}", flush=True)

    arms = [("lockstep_k1", 1), ("latent_search_k3", 3)]
    results = {name: [] for name, _ in arms}
    smf_results = []

    for name, max_think in arms:
        for s in args.seeds:
            print(f"\n=== Arm: {name} (max_think={max_think}, seed={s}) ===", flush=True)
            model, info = train_model(
                name, max_think, tr, device,
                steps=args.steps, batch_size=args.batch, seq_len=args.seq_len,
                lr=args.lr, ponder_w=args.ponder_w, seed=s,
            )
            ev = {k: eval_bpc_and_adaptive(model, v, device) for k, v in old_probes.items()}
            mean_bpc = round(sum(ev[k]["bpc"] for k in ev) / len(ev), 4)
            mean_corr_ent = round(sum(ev[k]["corr_entropy"] for k in ev) / len(ev), 4)
            mean_corr_sur = round(sum(ev[k]["corr_surprisal"] for k in ev) / len(ev), 4)
            mean_steps = round(sum(ev[k]["exp_steps"] for k in ev) / len(ev), 3)
            routing = measure_cross_tick_routing(model, old_probes["src0"], device)
            rec = {
                "seed": s, **info, "mean_bpc": mean_bpc, "mean_steps": mean_steps,
                "corr_entropy": mean_corr_ent, "corr_surprisal": mean_corr_sur,
                "cross_tick_routing": routing, "per_source": ev,
            }
            results[name].append(rec)
            print(
                f"  -> VAL mean_bpc={mean_bpc:.4f} (src0={ev['src0']['bpc']:.4f} src1={ev['src1']['bpc']:.4f} src2={ev['src2']['bpc']:.4f}) | "
                f"steps={mean_steps:.2f} corr(ent)={mean_corr_ent:+.3f} corr(nll)={mean_corr_sur:+.3f} routing={routing}",
                flush=True,
            )
            if name == "latent_search_k3":
                print(f"  -> Running real-text SODA OOD continual learning (seed={s})...", flush=True)
                smf_res = run_real_smf_continual(model, tr, old_probes, soda, device, seq_bytes=args.online_bytes)
                smf_results.append({"seed": s, **smf_res})
                for mname in ("smf_mlp", "full_ft"):
                    last = smf_res[mname][-1]
                    print(
                        f"     [{mname} @64KB] chat_drop={last['chat_drop']:+.4f} | "
                        f"worst_old_rise={last['worst_old_rise']:+.4f} mean_old_rise={last['mean_old_rise']:+.4f}",
                        flush=True,
                    )

    summary = {}
    for name, recs in results.items():
        summary[name] = {
            "params": recs[0]["params"],
            "mean_bpc": round(sum(r["mean_bpc"] for r in recs) / len(recs), 4),
            "mean_steps": round(sum(r["mean_steps"] for r in recs) / len(recs), 3),
            "corr_entropy": round(sum(r["corr_entropy"] for r in recs) / len(recs), 4),
            "corr_surprisal": round(sum(r["corr_surprisal"] for r in recs) / len(recs), 4),
            "tick01_jaccard": round(sum(r["cross_tick_routing"]["tick01_jaccard"] for r in recs) / len(recs), 4),
        }

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = ROOT / "runs" / f"{ts}_rmp_search.json"
    out_path.write_text(
        json.dumps({"args": vars(args), "summary": summary, "runs": results, "smf_continual": smf_results}, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved E127 results to {out_path}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
