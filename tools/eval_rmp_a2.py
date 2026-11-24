"""E125：A2（潜空间自适应思考 RMP + 极简端到端海马读取 + 原生 SMF）可证伪四臂对照与在线持续学习实验。

严格遵守苦涩的教训与零泄露原则：
1. 零手工特征：无 k_top=9 MaxSim、无 used_mask、无显式指针对齐监督；
2. 零脚本思考监督：仅给 (记忆库, 问题) -> 最终答案 的标准下一 token 交叉熵 + PonderNet 几何先验 KL；
3. 严格零实体泄露：训练与评测的人名、地点、物品、情境 100% 互不相交（且评测含训练期未见汉字）；
4. 四臂同主干对照（GPU 批量并行训练）：
   - `lockstep_no_hippo`  (K=1, use_hippo=False)
   - `A1_latent_no_hippo` (K=3, use_hippo=False)  -> 实证 Jelassi 2024 固定状态信息论上限
   - `A2_hippo_k1`        (K=1, use_hippo=True)   -> 检验单拍无潜空间思考能否解两跳
   - `A2_hippo_k3`        (K=3, use_hippo=True)   -> 完整 A2：无脚本涌现多跳连想 + 自适应思考步数
5. 第二阶段：在训好的 `A2_hippo_k3` 上直接对比 `smf_mlp` vs `full_ft` 在线学习 OOD 新格式/新任务时的旧任务抗遗忘。
"""
from __future__ import annotations

import argparse
import copy
import datetime
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rmp_a2 import A2SMFOnlineLearner, LatentThinkHippoRMP

# 特殊 token
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
SEP_ID = 3

# 严格零重叠训练 / 评测实体池
TRAIN_PEOPLE = [
    "王叔", "赵姐", "周工", "孙哥", "吴总", "郑姨", "冯师傅", "蒋老师",
    "韩姐", "邓叔", "彭哥", "曾姨", "肖工", "田师傅", "董姐", "袁叔",
    "潘哥", "于姨", "吕工", "施师傅", "陶姐", "姜叔", "戚哥", "谢姨",
]
TRAIN_PLACES = [
    "东门便利店", "西街药房", "南区物业", "北巷菜鸟", "银杏书屋", "松林诊所",
    "梧桐咖啡", "河畔驿站", "中心快递柜", "广场超市", "桥头修车铺", "花园门卫室",
    "街角洗衣房", "社区活动室", "地铁口报亭", "图书馆前台", "健身房柜台", "邮局大厅",
]
TRAIN_ITEMS = [
    "备用钥匙", "体检报告", "维修发票", "快递包裹", "相机电池", "备用门卡",
    "宠物粮食", "租房合同", "折叠雨伞", "露营帐篷", "电脑充电器", "挂号单",
]
TRAIN_CONDS = [
    "下雨打雷", "周末出差", "加班晚归", "停水停电", "朋友聚餐", "临时外出",
    "快递滞留", "设备故障", "物业检修", "降温刮风", "忘带手机", "门禁升级",
]

EVAL_PEOPLE = ["小李", "陈阿姨", "张医生", "林师傅", "老周", "阿远", "沈姐", "顾叔"]
EVAL_PLACES = ["蓝鲸咖啡", "桂花巷驿站", "翠湖门卫", "锦绣物业", "青石书店", "枫叶诊所"]
EVAL_ITEMS = ["大门钥匙", "过敏药盒", "胶片相机", "社保卡", "车库遥控", "备忘录"]
EVAL_CONDS = ["暴雨打雷", "临时出差", "深夜加班", "聚餐过敏"]

# 固定句子模板字符 + 所有训练/评测字符构建紧凑字符词表（评测特有字符在训练中永不出现！）
TEMPLATE_CHARS = (
    "记录：现在交给保管。今天在值班。时要联系处理。"
    "我现在该找谁拿？我现在要去哪里才能拿到？拿找谁，如果又该找谁？"
    "答复：先找，再找拒答：无记录紧急联络->（）"
)


class CharVocab:
    def __init__(self):
        chars = []
        seen = set()
        for src in (
            [TEMPLATE_CHARS]
            + TRAIN_PEOPLE + TRAIN_PLACES + TRAIN_ITEMS + TRAIN_CONDS
            + EVAL_PEOPLE + EVAL_PLACES + EVAL_ITEMS + EVAL_CONDS
        ):
            for ch in src:
                if ch not in seen:
                    seen.add(ch)
                    chars.append(ch)
        self.c2i = {ch: i + 4 for i, ch in enumerate(chars)}
        self.i2c = {i + 4: ch for i, ch in enumerate(chars)}
        self.size = len(chars) + 4

    def encode(self, text: str) -> List[int]:
        return [self.c2i[ch] for ch in text]

    def decode(self, ids: Sequence[int]) -> str:
        out = []
        for i in ids:
            ii = int(i)
            if ii == EOS_ID:
                break
            if ii in self.i2c:
                out.append(self.i2c[ii])
        return "".join(out)


def _distractors(
    rng: random.Random,
    people: Sequence[str],
    places: Sequence[str],
    items: Sequence[str],
    conds: Sequence[str],
    avoid_items: set,
    avoid_people: set,
    avoid_conds: set,
    n: int = 5,
) -> List[str]:
    out = []
    ai = [x for x in items if x not in avoid_items]
    ap = [x for x in people if x not in avoid_people]
    ac = [x for x in conds if x not in avoid_conds]
    rng.shuffle(ai); rng.shuffle(ap); rng.shuffle(ac)
    for i in range(n):
        r = i % 3
        if r == 0 and ai and ap:
            out.append(f"记录：{ai[i % len(ai)]}现在交给{ap[i % len(ap)]}保管。")
        elif r == 1 and ap and places:
            out.append(f"记录：{ap[i % len(ap)]}今天在{rng.choice(places)}值班。")
        elif ac and ap:
            out.append(f"记录：{ac[i % len(ac)]}时要联系{ap[i % len(ap)]}处理。")
    return out


def sample_raw(
    rng: random.Random,
    task_type: str,
    split: str = "train",
) -> Tuple[List[str], str, str, List[str]]:
    """返回 (活跃记忆条列表, 问题, 完整标准答案, 关键答案子串列表)。无任何跳数/思考脚本！"""
    if split == "train":
        people, places, items, conds = TRAIN_PEOPLE, TRAIN_PLACES, TRAIN_ITEMS, TRAIN_CONDS
    else:
        people, places, items, conds = EVAL_PEOPLE, EVAL_PLACES, EVAL_ITEMS, EVAL_CONDS

    if task_type == "T1_single":
        it = rng.choice(items)
        pp = rng.choice(people)
        mems = [f"记录：{it}现在交给{pp}保管。"] + _distractors(rng, people, places, items, conds, {it}, {pp}, set(), n=5)
        rng.shuffle(mems)
        return mems, f"我现在该找谁拿{it}？", f"答复：{pp}", [pp]

    if task_type == "T2_two_hop":
        it = rng.choice(items)
        pp = rng.choice(people)
        pl = rng.choice(places)
        mems = [
            f"记录：{it}现在交给{pp}保管。",
            f"记录：{pp}今天在{pl}值班。",
        ] + _distractors(rng, people, places, items, conds, {it}, {pp}, set(), n=5)
        rng.shuffle(mems)
        return mems, f"我现在要去哪里才能拿到{it}？", f"答复：{pl}", [pl]

    if task_type == "T3_dual":
        it = rng.choice(items)
        cd = rng.choice(conds)
        p1, p2 = rng.sample(list(people), 2)
        mems = [
            f"记录：{it}现在交给{p1}保管。",
            f"记录：{cd}时要联系{p2}处理。",
        ] + _distractors(rng, people, places, items, conds, {it}, {p1, p2}, {cd}, n=5)
        rng.shuffle(mems)
        return mems, f"拿{it}找谁，如果{cd}又该找谁？", f"答复：先找{p1}，再找{p2}", [p1, p2]

    if task_type == "T5_ood_style":
        # OOD 新任务/新输出格式（仅用于第二阶段 SMF 在线持续学习）
        it = rng.choice(items)
        pp = rng.choice(people)
        pl = rng.choice(places)
        mems = [
            f"记录：{it}现在交给{pp}保管。",
            f"记录：{pp}今天在{pl}值班。",
        ] + _distractors(rng, people, places, items, conds, {it}, {pp}, set(), n=4)
        rng.shuffle(mems)
        return mems, f"紧急联络：{it}？", f"紧急联络：{pp}->{pl}", [pp, pl]

    # T4_abstain
    it = rng.choice(items)
    mems = _distractors(rng, people, places, items, conds, {it}, set(), set(), n=6)
    rng.shuffle(mems)
    return mems, f"我现在该找谁拿{it}？", "拒答：无记录", ["拒答：无记录"]


def collate_batch(
    samples: Sequence[Tuple[List[str], str, str, List[str]]],
    vocab: CharVocab,
    device: torch.device,
    use_hippo: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """把批量样本打包成 (mem_ids, inp_ids, tgt_ids, prompt_lens)。
    - use_hippo=True：记忆放 mem_ids，inp_ids 为 [BOS] + query + [SEP] + answer + [EOS]（零 Self-Attention 主干！）
    - use_hippo=False：记忆拼在输入前部（测试纯 O(1) 循环状态能否靠快记忆记住并召回）。
    """
    mem_seqs, inp_seqs, tgt_seqs = [], [], []
    for mems, q, ans, _ in samples:
        m_ids = []
        for m in mems:
            m_ids.extend(vocab.encode(m))
            m_ids.append(SEP_ID)
        q_ids = vocab.encode(q)
        a_ids = vocab.encode(ans) + [EOS_ID]

        if use_hippo:
            prefix = [BOS_ID] + q_ids + [SEP_ID]
        else:
            prefix = [BOS_ID] + m_ids + q_ids + [SEP_ID]
        full = prefix + a_ids
        inp = full[:-1]
        tgt = [-100] * (len(prefix) - 1) + a_ids
        mem_seqs.append(m_ids)
        inp_seqs.append(inp)
        tgt_seqs.append(tgt)

    max_m = max(len(s) for s in mem_seqs)
    max_t = max(len(s) for s in inp_seqs)
    B = len(samples)
    mem_t = torch.full((B, max_m), PAD_ID, dtype=torch.long, device=device)
    inp_t = torch.full((B, max_t), PAD_ID, dtype=torch.long, device=device)
    tgt_t = torch.full((B, max_t), -100, dtype=torch.long, device=device)
    for i in range(B):
        mem_t[i, : len(mem_seqs[i])] = torch.tensor(mem_seqs[i], dtype=torch.long, device=device)
        inp_t[i, : len(inp_seqs[i])] = torch.tensor(inp_seqs[i], dtype=torch.long, device=device)
        tgt_t[i, : len(tgt_seqs[i])] = torch.tensor(tgt_seqs[i], dtype=torch.long, device=device)
    return mem_t, inp_t, tgt_t, torch.tensor([len(s) for s in inp_seqs], device=device)


@torch.no_grad()
def evaluate_model(
    model: LatentThinkHippoRMP,
    vocab: CharVocab,
    eval_sets: Dict[str, List[Tuple[List[str], str, str, List[str]]]],
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    model.eval()
    results = {}
    for tname, samples in eval_sets.items():
        # 1) 计算标准答案 NLL 与期望思考步数
        mem_t, inp_t, tgt_t, _ = collate_batch(samples, vocab, device, use_hippo=model.use_hippo)
        hmem = model.encode_memory(mem_t) if model.use_hippo else None
        _, metrics, _ = model.compute_loss(inp_t, tgt_t, hippo_mem=hmem, ponder_w=0.0)

        # 2) 自回归贪心生成评测精确匹配率
        exact_hits = 0
        key_hits = 0
        step_sum = 0.0
        for i, (mems, q, gold_ans, gold_keys) in enumerate(samples):
            m_ids = []
            for m in mems:
                m_ids.extend(vocab.encode(m))
                m_ids.append(SEP_ID)
            q_ids = vocab.encode(q)
            prefix = ([BOS_ID] + q_ids + [SEP_ID]) if model.use_hippo else ([BOS_ID] + m_ids + q_ids + [SEP_ID])
            p_t = torch.tensor([prefix], dtype=torch.long, device=device)
            hm_i = model.encode_memory(torch.tensor([m_ids], dtype=torch.long, device=device)) if model.use_hippo else None
            gen_ids, gen_steps = model.generate(p_t, hippo_mem=hm_i, max_new_tokens=20, eos_id=EOS_ID)
            pred_txt = vocab.decode(gen_ids[0].tolist())
            if pred_txt == gold_ans:
                exact_hits += 1
            if all(k in pred_txt for k in gold_keys):
                key_hits += 1
            step_sum += float(gen_steps.mean().item())

        n = len(samples)
        results[tname] = {
            "exact": round(exact_hits / n, 4),
            "key_acc": round(key_hits / n, 4),
            "nll": round(metrics["task_loss"], 4),
            "exp_steps": round(step_sum / n, 3),
        }
    return results


@torch.no_grad()
def evaluate_dynamic_switch(
    model: LatentThinkHippoRMP,
    vocab: CharVocab,
    device: torch.device,
    n_cases: int = 24,
    seed: int = 999,
) -> float:
    """动态改口切换测试：同一问题，零权重更新，只把记忆库里的人名从 p_old 换成 p_new，要求两次全对才算过。"""
    rng = random.Random(seed)
    model.eval()
    both_ok = 0
    for _ in range(n_cases):
        it = rng.choice(EVAL_ITEMS)
        p_old, p_new = rng.sample(EVAL_PEOPLE, 2)
        dist = _distractors(rng, EVAL_PEOPLE, EVAL_PLACES, EVAL_ITEMS, EVAL_CONDS, {it}, {p_old, p_new}, set(), n=5)
        q = f"我现在该找谁拿{it}？"
        ok = True
        for pp in (p_old, p_new):
            mems = [f"记录：{it}现在交给{pp}保管。"] + list(dist)
            rng.shuffle(mems)
            m_ids = []
            for m in mems:
                m_ids.extend(vocab.encode(m))
                m_ids.append(SEP_ID)
            prefix = ([BOS_ID] + vocab.encode(q) + [SEP_ID]) if model.use_hippo else ([BOS_ID] + m_ids + vocab.encode(q) + [SEP_ID])
            p_t = torch.tensor([prefix], dtype=torch.long, device=device)
            hm = model.encode_memory(torch.tensor([m_ids], dtype=torch.long, device=device)) if model.use_hippo else None
            gen_ids, _ = model.generate(p_t, hippo_mem=hm, max_new_tokens=16, eos_id=EOS_ID)
            if vocab.decode(gen_ids[0].tolist()) != f"答复：{pp}":
                ok = False
                break
        if ok:
            both_ok += 1
    return round(both_ok / n_cases, 4)


def train_arm(
    arm_name: str,
    max_think: int,
    use_hippo: bool,
    vocab: CharVocab,
    device: torch.device,
    steps: int,
    batch_size: int,
    lr: float,
    ponder_w: float,
    seed: int,
) -> Tuple[LatentThinkHippoRMP, Dict[str, object]]:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = LatentThinkHippoRMP(
        vocab=vocab.size, D=128, M=2, d=64, m=16, hidden=128, q=64,
        N=24, k=2, G=4, pd=96, pm=16, phidden=192, qr=32,
        beta_max=2.0, pad_id=PAD_ID, sep_id=None,
        max_think=max_think, use_hippo=use_hippo, hippo_dim=96,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    task_types = ["T1_single", "T2_two_hop", "T3_dual", "T4_abstain"]
    t0 = time.time()
    last_metrics = {}
    for step in range(1, steps + 1):
        model.train()
        batch = [sample_raw(rng, task_types[(step + i) % 4], split="train") for i in range(batch_size)]
        mem_t, inp_t, tgt_t, _ = collate_batch(batch, vocab, device, use_hippo=use_hippo)
        opt.zero_grad(set_to_none=True)
        hmem = model.encode_memory(mem_t) if use_hippo else None
        loss, m, _ = model.compute_loss(inp_t, tgt_t, hippo_mem=hmem, ponder_w=ponder_w, prior_p=0.5)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last_metrics = m
        if step % max(1, steps // 4) == 0 or step == steps:
            print(
                f"  [{arm_name} s={seed}] step {step:4d}/{steps} | "
                f"loss={m['task_loss']:.3f} acc={m['tok_acc']:.3f} steps={m['exp_steps']:.2f} ({time.time()-t0:.1f}s)",
                flush=True,
            )
    n_params = sum(p.numel() for p in model.parameters())
    return model, {"params": n_params, "train_sec": round(time.time() - t0, 1), "last_train": last_metrics}


def run_smf_continual_test(
    base_model: LatentThinkHippoRMP,
    vocab: CharVocab,
    eval_sets: Dict[str, List],
    device: torch.device,
    online_steps: int = 80,
    smf_k: int = 8,
    lr: float = 1e-3,
    seed: int = 42,
) -> Dict[str, object]:
    """在训好的 A2 模型上对比 `smf_mlp` vs `full_ft` 在线学习 OOD 新格式 T5_ood_style 时的旧任务留存。"""
    rng_eval = random.Random(seed + 777)
    ood_eval = {"T5_ood_style": [sample_raw(rng_eval, "T5_ood_style", split="eval") for _ in range(24)]}
    old_before = evaluate_model(base_model, vocab, eval_sets, device)
    ood_before = evaluate_model(base_model, vocab, ood_eval, device)["T5_ood_style"]

    # 校准背景路由频率（从旧任务训练集取 8 批）
    rng_bg = random.Random(seed + 123)
    bg_batches = []
    task_types = ["T1_single", "T2_two_hop", "T3_dual", "T4_abstain"]
    for b in range(8):
        samples = [sample_raw(rng_bg, task_types[(b + i) % 4], split="train") for i in range(16)]
        mem_t, inp_t, _, _ = collate_batch(samples, vocab, device, use_hippo=base_model.use_hippo)
        with torch.no_grad():
            hmem = base_model.encode_memory(mem_t) if base_model.use_hippo else None
        bg_batches.append((inp_t, hmem))

    # 固定在线 OOD 数据流（两臂严格同数据）
    rng_stream = random.Random(seed + 555)
    ood_stream = [[sample_raw(rng_stream, "T5_ood_style", split="train") for _ in range(16)] for _ in range(online_steps)]

    out = {"old_before": old_before, "ood_before": ood_before}
    for mode in ("smf_mlp", "full_ft"):
        m = copy.deepcopy(base_model).to(device)
        if mode == "smf_mlp":
            learner = A2SMFOnlineLearner(m, lr=lr, smf_k=smf_k)
            learner.calibrate_background(bg_batches)
            for batch in ood_stream:
                mem_t, inp_t, tgt_t, _ = collate_batch(batch, vocab, device, use_hippo=m.use_hippo)
                with torch.no_grad():
                    hmem = m.encode_memory(mem_t) if m.use_hippo else None
                learner.step(inp_t, tgt_t, hippo_mem=hmem, ponder_w=0.0)
        else:
            m.layers[1].plastic = False
            opt = torch.optim.AdamW(m.parameters(), lr=lr)
            for batch in ood_stream:
                m.train()
                mem_t, inp_t, tgt_t, _ = collate_batch(batch, vocab, device, use_hippo=m.use_hippo)
                opt.zero_grad(set_to_none=True)
                hmem = m.encode_memory(mem_t) if m.use_hippo else None
                loss, _, _ = m.compute_loss(inp_t, tgt_t, hippo_mem=hmem, ponder_w=0.0)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()

        old_after = evaluate_model(m, vocab, eval_sets, device)
        ood_after = evaluate_model(m, vocab, ood_eval, device)["T5_ood_style"]
        old_nll_rise = {k: round(old_after[k]["nll"] - old_before[k]["nll"], 4) for k in old_before}
        old_exact_drop = {k: round(old_before[k]["exact"] - old_after[k]["exact"], 4) for k in old_before}
        out[mode] = {
            "ood_after": ood_after,
            "old_after": old_after,
            "worst_old_nll_rise": max(old_nll_rise.values()),
            "mean_old_nll_rise": round(sum(old_nll_rise.values()) / len(old_nll_rise), 4),
            "worst_old_exact_drop": max(old_exact_drop.values()),
            "ood_nll_drop": round(ood_before["nll"] - ood_after["nll"], 4),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--ponder_w", type=float, default=0.01)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    ap.add_argument("--eval_n", type=int, default=24)
    ap.add_argument("--online_steps", type=int, default=80)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = CharVocab()
    print(f"Device: {device} | CharVocab size: {vocab.size} | steps={args.steps} batch={args.batch}", flush=True)

    # 固定零实体泄露评测集
    rng_e = random.Random(2026)
    eval_sets = {
        t: [sample_raw(rng_e, t, split="eval") for _ in range(args.eval_n)]
        for t in ("T1_single", "T2_two_hop", "T3_dual", "T4_abstain")
    }

    arms = [
        ("lockstep_no_hippo", 1, False),
        ("A1_latent_no_hippo", 3, False),
        ("A2_hippo_k1", 1, True),
        ("A2_hippo_k3", 3, True),
    ]

    all_runs = {}
    smf_runs = []
    for arm_name, max_think, use_hippo in arms:
        all_runs[arm_name] = []
        for s in args.seeds:
            print(f"\n=== Arm: {arm_name} (max_think={max_think}, use_hippo={use_hippo}, seed={s}) ===", flush=True)
            model, info = train_arm(
                arm_name, max_think, use_hippo, vocab, device,
                steps=args.steps, batch_size=args.batch, lr=args.lr, ponder_w=args.ponder_w, seed=s,
            )
            ev = evaluate_model(model, vocab, eval_sets, device)
            sw = evaluate_dynamic_switch(model, vocab, device, n_cases=args.eval_n, seed=900 + s)
            overall = round(sum(ev[k]["exact"] for k in ev) / len(ev), 4)
            rec = {"seed": s, **info, "eval": ev, "dynamic_switch": sw, "overall_exact": overall}
            all_runs[arm_name].append(rec)
            print(
                f"  -> EVAL overall={overall:.3f} switch={sw:.3f} | "
                f"T1={ev['T1_single']['exact']:.3f}(st={ev['T1_single']['exp_steps']:.2f}) "
                f"T2_2hop={ev['T2_two_hop']['exact']:.3f}(st={ev['T2_two_hop']['exp_steps']:.2f}) "
                f"T3_dual={ev['T3_dual']['exact']:.3f}(st={ev['T3_dual']['exp_steps']:.2f}) "
                f"T4_abs={ev['T4_abstain']['exact']:.3f}(st={ev['T4_abstain']['exp_steps']:.2f})",
                flush=True,
            )
            if arm_name == "A2_hippo_k3":
                print(f"  -> Running SMF continual learning test (seed={s})...", flush=True)
                smf_res = run_smf_continual_test(model, vocab, eval_sets, device, online_steps=args.online_steps, seed=s)
                smf_runs.append({"seed": s, **smf_res})
                for mname in ("smf_mlp", "full_ft"):
                    r = smf_res[mname]
                    print(
                        f"     [{mname}] OOD exact={r['ood_after']['exact']:.3f} (nll_drop={r['ood_nll_drop']:.3f}) | "
                        f"worst_old_nll_rise={r['worst_old_nll_rise']:+.4f} mean={r['mean_old_nll_rise']:+.4f} "
                        f"worst_old_exact_drop={r['worst_old_exact_drop']:+.3f}",
                        flush=True,
                    )

    # 汇总跨种子均值
    summary = {}
    for arm_name, recs in all_runs.items():
        summary[arm_name] = {
            "params": recs[0]["params"],
            "overall_exact": round(sum(r["overall_exact"] for r in recs) / len(recs), 4),
            "dynamic_switch": round(sum(r["dynamic_switch"] for r in recs) / len(recs), 4),
            "T1_single": round(sum(r["eval"]["T1_single"]["exact"] for r in recs) / len(recs), 4),
            "T2_two_hop": round(sum(r["eval"]["T2_two_hop"]["exact"] for r in recs) / len(recs), 4),
            "T3_dual": round(sum(r["eval"]["T3_dual"]["exact"] for r in recs) / len(recs), 4),
            "T4_abstain": round(sum(r["eval"]["T4_abstain"]["exact"] for r in recs) / len(recs), 4),
            "steps_T1": round(sum(r["eval"]["T1_single"]["exp_steps"] for r in recs) / len(recs), 3),
            "steps_T2_2hop": round(sum(r["eval"]["T2_two_hop"]["exp_steps"] for r in recs) / len(recs), 3),
            "steps_T4_abs": round(sum(r["eval"]["T4_abstain"]["exp_steps"] for r in recs) / len(recs), 3),
        }

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = ROOT / "runs" / f"{ts}_rmp_a2.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "args": vars(args),
        "vocab_size": vocab.size,
        "summary": summary,
        "runs": all_runs,
        "smf_continual": smf_runs,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved results to {out_path}", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
