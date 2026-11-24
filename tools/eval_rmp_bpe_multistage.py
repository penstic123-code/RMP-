"""BPE 底座多阶段持续学习评测脚本（适配 `train_rmp_lm.py` 产出的 `latest.pt` / `best.pt`）。

上云前清单 #5：
1. 自动从检查点读取 `cfg` 与 `WaveNet` 权重（兼容 `main` / `main_fine` / `small` / `small_fine` 与 `shared > 0` 常驻共享模块冻结）；
2. 流式加载 4 个真实英文领域（`edu` / `story` 作旧域，`chat` 作阶段 1 OOD，`code` 作阶段 2 OOD），或通过 `--local_smoke` 在本地离线快速自检；
3. 同一底座严格对比 `full_ft` vs `smf_mlp`（`smf_k` 默认取路由模块总数的 12.5%–25%），报告每阶段新域 loss/ppl 降幅、旧域最差/均值遗忘、阶段 1 在阶段 2 后的净保留率与跨阶段模块 Jaccard 重叠。

用法（Colab A100）：
  python tools/eval_rmp_bpe_multistage.py /content/drive/MyDrive/rmp_main_fine/latest.pt --out runs/eval_bpe_multistage.json
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from online import _masked_step
from pool import WaveNet

EOT_ID = 50256


@torch.no_grad()
def eval_batches(model: WaveNet, batches: List[Tuple[torch.Tensor, torch.Tensor]], device: str, amp_dtype) -> Dict[str, float]:
    model.eval()
    tot, n = 0.0, 0
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device == "cuda")):
            lg, _ = model(x)
            loss = F.cross_entropy(lg.float().reshape(-1, lg.size(-1)), y.reshape(-1))
        tot += float(loss.item())
        n += 1
    mean_loss = tot / max(1, n)
    return {
        "loss": round(mean_loss, 4),
        "ppl": round(math.exp(min(mean_loss, 20.0)), 2),
        "bpt": round(mean_loss / math.log(2), 4),
    }


def make_batches(toks: torch.Tensor, B: int, T: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    stride = B * (T + 1)
    n = len(toks) // stride
    assert n >= 1, f"Not enough tokens ({len(toks)}) for B={B}, T={T}"
    arr = toks[: n * stride].view(n, B, T + 1)
    return [(arr[i, :, :-1], arr[i, :, 1:]) for i in range(n)]


@torch.no_grad()
def calibrate_bg_bpe(model: WaveNet, bg_batches: List[Tuple[torch.Tensor, torch.Tensor]], device: str, amp_dtype) -> torch.Tensor:
    pool = model.layers[1]
    cnt = torch.zeros(pool.N, device=device)
    model.eval()
    pool.capture = []
    try:
        for x, _ in bg_batches:
            pool.capture.clear()
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device == "cuda")):
                model(x.to(device))
            for _, sel, _, _ in pool.capture:
                cnt += sel.sum((0, 1))
    finally:
        pool.capture = None
    return cnt / cnt.sum().clamp(min=1.0)


def run_bpe_stage(
    model: WaveNet,
    mode: str,
    train_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    bg_freq: torch.Tensor,
    device: str,
    amp_dtype,
    lr: float = 5e-4,
    smf_k: int = 24,
    smf_eps: float = 1e-2,
) -> torch.Tensor:
    pool = model.layers[1]
    core = pool.core
    pool.plastic = False
    # 明确冻结 shared > 0 常驻共享模块：routed 掩码天然排除 is_shared == 1
    routed = (pool.active > 0) & (pool.is_shared == 0)

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

    used_counts = torch.zeros(pool.N, device=device)
    for x, y in train_batches:
        x, y = x.to(device), y.to(device)
        model.train()
        opt.zero_grad(set_to_none=True)
        if mode == "full_ft":
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device == "cuda")):
                lg, _ = model(x)
                loss = F.cross_entropy(lg.float().reshape(-1, lg.size(-1)), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        else:
            pool.capture = []
            try:
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device == "cuda")):
                    lg, _ = model(x)
                    loss = F.cross_entropy(lg.float().reshape(-1, lg.size(-1)), y.reshape(-1))
                loss.backward()
                cur = torch.zeros(pool.N, device=device)
                for _, sel, _, _ in pool.capture:
                    cur += sel.sum((0, 1))
            finally:
                pool.capture = None
            cur_freq = cur / cur.sum().clamp(min=1.0)
            score = (cur_freq / (bg_freq + smf_eps)).masked_fill(~routed, -1e9)
            k_eff = min(smf_k, int(routed.sum().item()))
            top_idx = score.topk(k_eff).indices
            keep_vec = torch.zeros(pool.N, device=device)
            keep_vec[top_idx] = 1.0
            used_counts += keep_vec
            torch.nn.utils.clip_grad_norm_(mlp_params, 1.0)
            _masked_step(opt, mlp_params, keep_vec)
    return used_counts


def load_domains_local_smoke(n_tokens: int = 8192) -> Dict[str, torch.Tensor]:
    """离线本地冒烟用：从本地 data/ 真实文件截取 4 域字节张量（验证接口与流程无 bug）。"""
    import numpy as np
    files = {
        "edu": ROOT / "data/src-cosmo_wikihow.txt",
        "story": ROOT / "data/src-cosmo_stories.txt",
        "chat": ROOT / "data/src-soda.txt",
        "code": ROOT / "data/src-cosmo_khan.txt",
    }
    return {k: torch.from_numpy(np.fromfile(p, dtype=np.uint8)[:n_tokens].astype(np.int64)) for k, p in files.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="train_rmp_lm.py 产出的 latest.pt 或 best.pt")
    ap.add_argument("--out", default="", help="输出 JSON 路径")
    ap.add_argument("--n_tokens", type=int, default=40_000)
    ap.add_argument("--B", type=int, default=4)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--smf_k", type=int, default=0, help="0=自动取 N//8 (12.5%)")
    ap.add_argument("--local_smoke", action="store_true", help="离线用本地 data/ 验证 BPE 多阶段脚本")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = dict(ck["cfg"])
    base = WaveNet(**cfg).to(device)
    base.load_state_dict(ck["model"])
    N = base.layers[1].N
    smf_k = args.smf_k if args.smf_k > 0 else max(1, N // 8)
    print(f"[Load] ckpt={args.ckpt} | N={N} G={base.layers[1].G} k={base.layers[1].k} shared={base.layers[1].shared} | smf_k={smf_k}", flush=True)

    if args.local_smoke:
        domains = load_domains_local_smoke(n_tokens=min(args.n_tokens, 8192))
    else:
        import tiktoken
        from tools.eval_rmp_138m import load_domain_tokens
        enc = tiktoken.get_encoding("gpt2")
        domains = load_domain_tokens(enc, n_tokens=args.n_tokens)

    # 每个域前 2/3 训练/校准，后 1/3 留出评测
    splits = {}
    for k, toks in domains.items():
        cut = (len(toks) * 2) // 3
        splits[k] = (make_batches(toks[:cut], args.B, args.T), make_batches(toks[cut:], args.B, args.T))

    bg_freq = calibrate_bg_bpe(base, splits["edu"][0] + splits["story"][0], device, amp_dtype)
    old_0 = {k: eval_batches(base, splits[k][1], device, amp_dtype)["loss"] for k in ("edu", "story")}
    s1_0 = eval_batches(base, splits["chat"][1], device, amp_dtype)["loss"]
    s2_0 = eval_batches(base, splits["code"][1], device, amp_dtype)["loss"]
    print(f"[Base] old={old_0} | chat={s1_0} | code={s2_0}", flush=True)

    results = {"ckpt": args.ckpt, "cfg": cfg, "smf_k": smf_k, "base": {"old": old_0, "chat": s1_0, "code": s2_0}, "modes": {}}
    for mode in ("full_ft", "smf_mlp"):
        m = copy.deepcopy(base).to(device)
        u1 = run_bpe_stage(m, mode, splits["chat"][0], bg_freq, device, amp_dtype, lr=args.lr, smf_k=smf_k)
        old_1 = {k: eval_batches(m, splits[k][1], device, amp_dtype)["loss"] for k in ("edu", "story")}
        s1_1 = eval_batches(m, splits["chat"][1], device, amp_dtype)["loss"]

        u2 = run_bpe_stage(m, mode, splits["code"][0], bg_freq, device, amp_dtype, lr=args.lr, smf_k=smf_k)
        old_2 = {k: eval_batches(m, splits[k][1], device, amp_dtype)["loss"] for k in ("edu", "story")}
        s1_2 = eval_batches(m, splits["chat"][1], device, amp_dtype)["loss"]
        s2_2 = eval_batches(m, splits["code"][1], device, amp_dtype)["loss"]

        jacc = 0.0
        if mode != "full_ft":
            b1, b2 = u1 > 0, u2 > 0
            jacc = round(float((b1 & b2).sum().item() / max(1, (b1 | b2).sum().item())), 4)
        rec = {
            "stage1_chat_drop": round(s1_0 - s1_1, 4),
            "stage1_worst_old_rise": round(max(old_1[k] - old_0[k] for k in old_0), 4),
            "stage2_code_drop": round(s2_0 - s2_2, 4),
            "stage2_chat_forget": round(s1_2 - s1_1, 4),
            "stage2_chat_net_drop": round(s1_0 - s1_2, 4),
            "stage2_worst_old_rise": round(max(old_2[k] - old_0[k] for k in old_0), 4),
            "stage12_module_jaccard": jacc,
        }
        results["modes"][mode] = rec
        print(f"  [{mode:7s}] {rec}", flush=True)

    if args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[Saved] {out_p}", flush=True)


if __name__ == "__main__":
    main()
