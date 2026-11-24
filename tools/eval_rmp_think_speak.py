"""思说解耦 RMP（Think-Speak Decoupled RMP）首轮可证伪对照实验。

严格零实体泄露 + 拔掉 Qwen 直接给 RMP 打分 + 同参量锁步对照：
1. 训练集与评测集的人名、地点、物品、情境 100% 互不相交；
2. 同架构同参量对比：
   - `lockstep_rmp`（1:1 锁步基线，max_think=0，无内部思考与句中暂停）；
   - `think_speak_rmp`（思说解耦 RMP，开口前多步思考 + 句中暂停重规划）；
   - `think_speak_no_mid`（测试期因果消融：禁止句中暂停 `allow_mid_pause=False`）；
3. 4 项核心测试（完全拔掉 Qwen，纯测 RMP 自身逐字节输出）：
   - T1 动态改口切换（同一问题只改记忆最新事实：如 小李 → 陈阿姨，零权重更新）；
   - T2 开口前两跳规划（先查人、再查人所在地点，连续 2 拍内部思考后开口）；
   - T3 句中暂停重规划（开口先说出前半句，句中 `emit_gate` 主动暂停、补查第二条记忆，再说完后半句）；
   - T4 缺证拒答（记忆库无依据时命中 NULL 槽并输出「拒答」）；
4. 可选 `--with_qwen`：接回冻结 Qwen2.5-1.5B 做纯嘴替（只看 RMP 草稿，不看记忆库与原问题），检验转写保真度。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if not os.environ.get("HF_HOME") and (ROOT / ".pytest_cache" / "hf").exists():
    os.environ["HF_HOME"] = str(ROOT / ".pytest_cache" / "hf")

import torch
import torch.nn.functional as F

from rmp_think_speak import (
    BOS_ID,
    EOS_ID,
    RMPEpisodicBank,
    SEP_ID,
    ThinkSpeakRMP,
    WorkingMemoryBuffer,
    encode_bytes,
)

# =====================================================================
# 严格零重叠的训练/评测实体池（保证评测人名/地点/物品/情境在训练期从未见过）
# =====================================================================
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


@dataclass
class Episode:
    task_type: str  # fact_single | two_hop | mid_pause | abstain
    bank: RMPEpisodicBank
    query: str
    plan: List[Tuple]  # ("THINK", mem_id_or_None) | ("SPEAK", text_chunk, is_final)
    gold_answers: List[str]
    gold_mem_ids: List[str]


def _make_distractors(
    rng: random.Random,
    people: Sequence[str],
    places: Sequence[str],
    items: Sequence[str],
    conds: Sequence[str],
    avoid_items: set,
    avoid_people: set,
    avoid_conds: set,
    n: int = 5,
) -> List[Tuple[str, str]]:
    out = []
    avail_items = [x for x in items if x not in avoid_items]
    avail_people = [x for x in people if x not in avoid_people]
    avail_conds = [x for x in conds if x not in avoid_conds]
    rng.shuffle(avail_items); rng.shuffle(avail_people); rng.shuffle(avail_conds)
    for i in range(n):
        if i % 3 == 0 and avail_items and avail_people:
            it = avail_items[i % len(avail_items)]
            pp = avail_people[i % len(avail_people)]
            out.append((f"d_{i}", f"记录：{it}现在交给{pp}保管。"))
        elif i % 3 == 1 and avail_people and places:
            pp = avail_people[i % len(avail_people)]
            pl = rng.choice(places)
            out.append((f"d_{i}", f"记录：{pp}今天在{pl}值班。"))
        elif avail_conds and avail_people:
            cd = avail_conds[i % len(avail_conds)]
            pp = avail_people[i % len(avail_people)]
            out.append((f"d_{i}", f"记录：{cd}时要联系{pp}处理。"))
    return out


def sample_episode(
    rng: random.Random,
    task_type: str,
    people: Sequence[str],
    places: Sequence[str],
    items: Sequence[str],
    conds: Sequence[str],
) -> Episode:
    bank = RMPEpisodicBank()
    if task_type == "fact_single":
        it = rng.choice(items)
        p_old, p_new = rng.sample(list(people), 2)
        bank.add("m_old", f"记录：{it}现在交给{p_old}保管。", day=1)
        bank.supersede("m_old", "m_new", f"记录：{it}现在交给{p_new}保管。", day=2)
        for did, txt in _make_distractors(rng, people, places, items, conds, {it}, {p_old, p_new}, set(), n=5):
            bank.add(did, txt, day=1)
        q = f"我现在该找谁拿{it}？"
        plan = [("THINK", "m_new"), ("SPEAK", f"答复：{p_new}", True)]
        return Episode(task_type, bank, q, plan, [p_new], ["m_new"])

    if task_type == "two_hop":
        it = rng.choice(items)
        pp = rng.choice(people)
        pl = rng.choice(places)
        bank.add("m_a", f"记录：{it}现在交给{pp}保管。", day=1)
        bank.add("m_b", f"记录：{pp}今天在{pl}值班。", day=2)
        for did, txt in _make_distractors(rng, people, places, items, conds, {it}, {pp}, set(), n=5):
            bank.add(did, txt, day=1)
        q = f"我现在要去哪里才能拿到{it}？"
        plan = [("THINK", "m_a"), ("THINK", "m_b"), ("SPEAK", f"答复：{pl}", True)]
        return Episode(task_type, bank, q, plan, [pl], ["m_a", "m_b"])

    if task_type == "mid_pause":
        it = rng.choice(items)
        cd = rng.choice(conds)
        p1, p2 = rng.sample(list(people), 2)
        bank.add("m_1", f"记录：{it}现在交给{p1}保管。", day=1)
        bank.add("m_2", f"记录：{cd}时要联系{p2}处理。", day=2)
        for did, txt in _make_distractors(rng, people, places, items, conds, {it}, {p1, p2}, {cd}, n=5):
            bank.add(did, txt, day=1)
        q = f"拿{it}找谁，如果{cd}又该找谁？"
        # 开口前只查第一条记忆，说完前半句后在逗号处主动暂停、补查第二条记忆、再说完后半句！
        plan = [
            ("THINK", "m_1"),
            ("SPEAK", f"答复：先找{p1}，", False),
            ("THINK", "m_2"),
            ("SPEAK", f"再找{p2}", True),
        ]
        return Episode(task_type, bank, q, plan, [p1, p2], ["m_1", "m_2"])

    # abstain
    it = rng.choice(items)
    for did, txt in _make_distractors(rng, people, places, items, conds, {it}, set(), set(), n=6):
        bank.add(did, txt, day=1)
    q = f"我现在该找谁拿{it}？"
    plan = [("THINK", None), ("SPEAK", "拒答：记忆不足", True)]
    return Episode("abstain", bank, q, plan, ["拒答"], [])


def compute_episode_loss(
    model: ThinkSpeakRMP,
    ep: Episode,
    lockstep: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    dev = model.emb.weight.device
    entries = ep.bank.active_entries()
    eid_to_idx = {e.event_id: i for i, e in enumerate(entries)}
    encoded_mems = model.encode_memory_bank([e.text for e in entries], device=dev) if not lockstep else []
    used_mask = torch.zeros(len(entries), dtype=torch.bool, device=dev)
    wm = WorkingMemoryBuffer(max_slots=2)

    q_ids = [BOS_ID] + encode_bytes(ep.query) + [SEP_ID]
    q_tok = torch.tensor([q_ids], dtype=torch.long, device=dev)
    st = model.init_state(1, dev)
    h_sh_s, h_dp_s, x_s, st = model.step_tokens(q_tok, st)
    h_sh, h_dp, x_cur = h_sh_s[0, -1], h_dp_s[0, -1], x_s[0, -1]
    ctx_tok_vecs = F.normalize(model.tok_match_proj(x_s[0, 1:-1]), dim=-1)

    l_mem = torch.tensor(0.0, device=dev)
    l_gate = torch.tensor(0.0, device=dev)
    l_lm = torch.tensor(0.0, device=dev)
    n_mem = n_gate = n_lm = 0

    if lockstep:
        full_text = "".join(seg[1] for seg in ep.plan if seg[0] == "SPEAK")
        tgt_ids = encode_bytes(full_text) + [EOS_ID]
        lp0 = model.speak_log_probs(h_sh, h_dp, x_cur, wm=None)
        l_lm = l_lm + F.nll_loss(lp0.unsqueeze(0), torch.tensor([tgt_ids[0]], device=dev))
        n_lm += 1
        if len(tgt_ids) > 1:
            inp = torch.tensor([tgt_ids[:-1]], dtype=torch.long, device=dev)
            hs_o, hdp_o, x_o, st = model.step_tokens(inp, st)
            lps = model.speak_log_probs(hs_o[0], hdp_o[0], x_o[0], wm=None)
            l_lm = l_lm + F.nll_loss(lps, torch.tensor(tgt_ids[1:], device=dev))
            n_lm += 1
        return l_lm / max(1, n_lm), {"mem": 0.0, "gate": 0.0, "lm": float((l_lm / max(1, n_lm)).item())}

    # 思说解耦 RMP：严格按 plan 推进内部思考步与外部说话步
    for seg_i, seg in enumerate(ep.plan):
        if seg[0] == "THINK":
            tgt_eid = seg[1]
            l_gate = l_gate + F.binary_cross_entropy_with_logits(
                model.emit_logit(h_sh, h_dp, x_cur), torch.tensor(0.0, device=dev)
            )
            n_gate += 1

            mlogits = model.memory_logits(h_dp, x_cur, ctx_tok_vecs, encoded_mems, used_mask=used_mask)
            tgt_idx = eid_to_idx[tgt_eid] if (tgt_eid is not None and tgt_eid in eid_to_idx) else len(entries)
            l_mem = l_mem + F.cross_entropy(mlogits.unsqueeze(0), torch.tensor([tgt_idx], device=dev))
            n_mem += 1

            mem_txt = None
            if tgt_eid is not None and tgt_eid in eid_to_idx:
                mi = eid_to_idx[tgt_eid]
                mem_txt = entries[mi].text
                used_mask[mi] = True
                wm.push(encoded_mems[mi])
            h_sh, h_dp, x_cur, new_toks, st = model.internal_think_step(h_dp, x_cur, st, mem_text=mem_txt)
            if mem_txt is not None:
                ctx_tok_vecs = torch.cat([ctx_tok_vecs, new_toks], dim=0)

        elif seg[0] == "SPEAK":
            _, chunk_txt, is_final = seg
            next_is_think = (seg_i + 1 < len(ep.plan)) and (ep.plan[seg_i + 1][0] == "THINK")
            chunk_ids = encode_bytes(chunk_txt)
            tgt_ids = chunk_ids + ([EOS_ID] if is_final else [])

            l_gate = l_gate + F.binary_cross_entropy_with_logits(
                model.emit_logit(h_sh, h_dp, x_cur), torch.tensor(1.0, device=dev)
            )
            n_gate += 1

            lp0 = model.speak_log_probs(h_sh, h_dp, x_cur, wm=wm)
            l_lm = l_lm + F.nll_loss(lp0.unsqueeze(0), torch.tensor([tgt_ids[0]], device=dev))
            n_lm += 1

            inp = torch.tensor([chunk_ids], dtype=torch.long, device=dev)
            hs_o, hdp_o, x_o, st = model.step_tokens(inp, st)

            if is_final:
                lps = model.speak_log_probs(hs_o[0], hdp_o[0], x_o[0], wm=wm)
                l_lm = l_lm + F.nll_loss(lps, torch.tensor(tgt_ids[1:], device=dev))
                n_lm += 1
            elif len(chunk_ids) > 1:
                lps = model.speak_log_probs(hs_o[0, :-1], hdp_o[0, :-1], x_o[0, :-1], wm=wm)
                l_lm = l_lm + F.nll_loss(lps, torch.tensor(chunk_ids[1:], device=dev))
                n_lm += 1

            gate_end = len(chunk_ids) - 1 if next_is_think else len(chunk_ids)
            if gate_end > 0:
                g_logits = model.emit_gate(torch.cat([hs_o[0, :gate_end], hdp_o[0, :gate_end], x_o[0, :gate_end]], dim=-1)).squeeze(-1)
                l_gate = l_gate + F.binary_cross_entropy_with_logits(g_logits, torch.ones_like(g_logits))
                n_gate += 1

            h_sh, h_dp, x_cur = hs_o[0, -1], hdp_o[0, -1], x_o[0, -1]

    lm_v = l_lm / max(1, n_lm)
    mem_v = l_mem / max(1, n_mem)
    gate_v = l_gate / max(1, n_gate)
    total = lm_v + 1.5 * mem_v + 1.0 * gate_v
    return total, {"mem": float(mem_v.item()), "gate": float(gate_v.item()), "lm": float(lm_v.item())}


def train_model(
    model: ThinkSpeakRMP,
    lockstep: bool,
    steps: int = 360,
    lr: float = 2.5e-3,
    accum: int = 8,
    seed: int = 42,
) -> Dict[str, float]:
    rng = random.Random(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps // accum), eta_min=lr * 0.1)
    task_cycle = ["fact_single", "two_hop", "mid_pause", "abstain"]

    model.train()
    opt.zero_grad()
    sm = sg = sl = 0.0
    n_log = 0
    for st_i in range(1, steps + 1):
        tt = task_cycle[(st_i - 1) % len(task_cycle)]
        ep = sample_episode(rng, tt, TRAIN_PEOPLE, TRAIN_PLACES, TRAIN_ITEMS, TRAIN_CONDS)
        loss, info = compute_episode_loss(model, ep, lockstep=lockstep)
        (loss / accum).backward()
        sm += info["mem"]; sg += info["gate"]; sl += info["lm"]; n_log += 1

        if st_i % accum == 0 or st_i == steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
            opt.step()
            opt.zero_grad()
            sched.step()

    return {
        "mem": round(sm / max(1, n_log), 4),
        "gate": round(sg / max(1, n_log), 4),
        "lm": round(sl / max(1, n_log), 4),
    }


@torch.no_grad()
def evaluate_unseen(
    model: ThinkSpeakRMP,
    max_think_total: int,
    allow_mid_pause: bool = True,
    seed: int = 999,
) -> Dict[str, object]:
    """在 100% 未见实体池上做 4 项严格评测（拔掉 Qwen 直接给 RMP 打分）。"""
    rng = random.Random(seed)
    model.eval()

    # T1: 动态改口切换（16 对 = 32 次问答，同一问题只改记忆最新事实，零权重更新）
    t1_before = t1_after = t1_both = t1_leak = 0
    t1_samples = []
    for i in range(16):
        it = EVAL_ITEMS[i % len(EVAL_ITEMS)]
        p1, p2 = rng.sample(EVAL_PEOPLE, 2)
        bank = RMPEpisodicBank()
        bank.add("k_v1", f"记录：{it}现在交给{p1}保管。", day=1)
        for did, txt in _make_distractors(rng, EVAL_PEOPLE, EVAL_PLACES, EVAL_ITEMS, EVAL_CONDS, {it}, {p1, p2}, set(), n=5):
            bank.add(did, txt, day=1)
        q = f"我现在该找谁拿{it}？"
        r1 = model.generate(q, bank=bank, max_think_total=max_think_total, allow_mid_pause=allow_mid_pause)

        # 仅改写记忆库里的最新事实（不改权重、不改问题！）
        bank.supersede("k_v1", "k_v2", f"记录：{it}现在交给{p2}保管。", day=2)
        r2 = model.generate(q, bank=bank, max_think_total=max_think_total, allow_mid_pause=allow_mid_pause)

        ok1 = (p1 in r1["text"]) and (p2 not in r1["text"])
        ok2 = (p2 in r2["text"]) and (p1 not in r2["text"])
        leak = (p1 in r2["text"])
        t1_before += int(ok1); t1_after += int(ok2); t1_both += int(ok1 and ok2); t1_leak += int(leak)
        if i < 3:
            t1_samples.append({"q": q, "p1": p1, "out1": r1["text"], "p2": p2, "out2": r2["text"], "both": ok1 and ok2})

    # T2: 开口前两跳规划（16 题）
    t2_rec = t2_ans = 0
    t2_samples = []
    for i in range(16):
        ep = sample_episode(rng, "two_hop", EVAL_PEOPLE, EVAL_PLACES, EVAL_ITEMS, EVAL_CONDS)
        r = model.generate(ep.query, bank=ep.bank, max_think_total=max_think_total, allow_mid_pause=allow_mid_pause)
        rec_ok = set(ep.gold_mem_ids).issubset(set(r["recalled_ids"]))
        ans_ok = ep.gold_answers[0] in r["text"]
        t2_rec += int(rec_ok); t2_ans += int(ans_ok)
        if i < 2:
            t2_samples.append({"q": ep.query, "gold": ep.gold_answers[0], "out": r["text"], "thinks": r["pre_speech_thinks"], "recalled": r["recalled_ids"]})

    # T3: 句中暂停重规划（16 题）
    t3_mid = t3_rec = t3_ans = 0
    t3_samples = []
    for i in range(16):
        ep = sample_episode(rng, "mid_pause", EVAL_PEOPLE, EVAL_PLACES, EVAL_ITEMS, EVAL_CONDS)
        r = model.generate(ep.query, bank=ep.bank, max_think_total=max_think_total, allow_mid_pause=allow_mid_pause)
        mid_ok = r["mid_speech_thinks"] >= 1
        rec_ok = set(ep.gold_mem_ids).issubset(set(r["recalled_ids"]))
        ans_ok = all(g in r["text"] for g in ep.gold_answers)
        t3_mid += int(mid_ok); t3_rec += int(rec_ok); t3_ans += int(ans_ok)
        if i < 3:
            t3_samples.append({
                "q": ep.query, "gold": ep.gold_answers, "out": r["text"],
                "pre_t": r["pre_speech_thinks"], "mid_t": r["mid_speech_thinks"], "trace": r["trace"],
            })

    # T4: 缺证拒答（16 题）
    t4_abs = 0
    for i in range(16):
        ep = sample_episode(rng, "abstain", EVAL_PEOPLE, EVAL_PLACES, EVAL_ITEMS, EVAL_CONDS)
        r = model.generate(ep.query, bank=ep.bank, max_think_total=max_think_total, allow_mid_pause=allow_mid_pause)
        if "拒答" in r["text"]:
            t4_abs += 1

    return {
        "fact_flip_before": round(t1_before / 16, 4),
        "fact_flip_after": round(t1_after / 16, 4),
        "fact_flip_both": round(t1_both / 16, 4),
        "fact_old_leak": round(t1_leak / 16, 4),
        "two_hop_recall": round(t2_rec / 16, 4),
        "two_hop_ans": round(t2_ans / 16, 4),
        "mid_pause_triggered": round(t3_mid / 16, 4),
        "mid_pause_recall": round(t3_rec / 16, 4),
        "mid_pause_ans": round(t3_ans / 16, 4),
        "abstain_acc": round(t4_abs / 16, 4),
        "samples": {"t1": t1_samples, "t2": t2_samples, "t3": t3_samples},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=640)
    ap.add_argument("--seeds", type=str, default="42,43")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--D", type=int, default=128)
    args = ap.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== 思说解耦 RMP v2（WavePool + 有界工作记忆注意力）对照实验（dev={dev}, steps={args.steps}, seeds={seeds}）===", flush=True)

    runs = []
    for sd in seeds:
        # 1. 同参量 1:1 锁步基线（max_think=0）
        torch.manual_seed(sd)
        m_lock = ThinkSpeakRMP(D=args.D, N=24, k=2, G=4, pd=96, pm=16, phidden=192, qr=32, mem_dim=96, wm_dim=96).to(dev)
        n_params = sum(p.numel() for p in m_lock.parameters())
        loss_lock = train_model(m_lock, lockstep=True, steps=args.steps, lr=args.lr, seed=sd)
        res_lock = evaluate_unseen(m_lock, max_think_total=0, seed=900 + sd)

        # 2. 思说解耦 RMP + 有界工作记忆注意力（开口前思考 + 句中暂停重规划）
        torch.manual_seed(sd)
        m_ts = ThinkSpeakRMP(D=args.D, N=24, k=2, G=4, pd=96, pm=16, phidden=192, qr=32, mem_dim=96, wm_dim=96).to(dev)
        loss_ts = train_model(m_ts, lockstep=False, steps=args.steps, lr=args.lr, seed=sd)
        res_ts = evaluate_unseen(m_ts, max_think_total=4, allow_mid_pause=True, seed=900 + sd)
        # 3. 因果消融：同一思说解耦权重，测试期禁止句中暂停
        res_nomid = evaluate_unseen(m_ts, max_think_total=4, allow_mid_pause=False, seed=900 + sd)

        print(
            f"[seed={sd}] params={n_params:,} | "
            f"lockstep(flip={res_lock['fact_flip_both']:.1%}, 2hop={res_lock['two_hop_ans']:.1%}, mid={res_lock['mid_pause_ans']:.1%}, abs={res_lock['abstain_acc']:.1%}) | "
            f"think_speak(flip={res_ts['fact_flip_both']:.1%}, 2hop={res_ts['two_hop_ans']:.1%}, mid={res_ts['mid_pause_ans']:.1%}[trig={res_ts['mid_pause_triggered']:.1%}], abs={res_ts['abstain_acc']:.1%}) | "
            f"no_mid_pause(mid={res_nomid['mid_pause_ans']:.1%})",
            flush=True,
        )
        runs.append({
            "seed": sd, "params": n_params,
            "losses": {"lockstep": loss_lock, "think_speak": loss_ts},
            "lockstep": res_lock, "think_speak": res_ts, "no_mid_pause": res_nomid,
        })

    mkeys = [
        "fact_flip_both", "fact_old_leak", "two_hop_recall", "two_hop_ans",
        "mid_pause_triggered", "mid_pause_recall", "mid_pause_ans", "abstain_acc",
    ]
    summary = {}
    print(f"\n=== 思说解耦 RMP 汇总（{len(seeds)} 种子均值，100% 未见实体，零 Qwen / 零 Attention）===")
    hdr = f"{'方法':<18} | {'动态改口↑':>8} | {'旧值漏↓':>7} | {'两跳召回':>8} | {'两跳答对':>8} | {'句中停顿':>8} | {'句中双中':>8} | {'缺证拒答':>8}"
    print(hdr); print("-" * len(hdr))
    for arm in ("lockstep", "no_mid_pause", "think_speak"):
        summary[arm] = {mk: round(sum(r[arm][mk] for r in runs) / len(runs), 4) for mk in mkeys}
        m = summary[arm]
        print(
            f"{arm:<18} | {m['fact_flip_both']:>8.1%} | {m['fact_old_leak']:>7.1%} | "
            f"{m['two_hop_recall']:>8.1%} | {m['two_hop_ans']:>8.1%} | "
            f"{m['mid_pause_triggered']:>8.1%} | {m['mid_pause_ans']:>8.1%} | {m['abstain_acc']:>8.1%}",
            flush=True,
        )

    print("\n--- 代表性未见实体样本（seed 0, think_speak）---")
    for k, arr in runs[0]["think_speak"]["samples"].items():
        print(f"  [{k}] {arr[:2]}")

    elapsed = round(time.time() - t0, 1)
    out_path = ROOT / f"runs/{time.strftime('%Y%m%d-%H%M%S')}_rmp_think_speak.json"
    out_path.write_text(
        json.dumps(
            {
                "experiment": "rmp_think_speak_v1",
                "steps": args.steps,
                "seeds": seeds,
                "D": args.D,
                "elapsed_s": elapsed,
                "summary_mean": summary,
                "seed_runs": runs,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n结果已保存：{out_path}（耗时 {elapsed}s）", flush=True)


if __name__ == "__main__":
    main()
