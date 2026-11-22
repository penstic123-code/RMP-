"""E4 实验：接回小 LLM（冻结 Qwen2.5-1.5B-Instruct）验证语言表达与上限。

严格遵循 docs/design/associative-memory.md · E4：
- 给同一个冻结 Qwen2.5-1.5B-Instruct 分别喂入三路证据：
  ① one_shot_llm（一次语义检索证据）
  ② learned_hybrid_llm（可学习循环回想证据，仅在独立主题训练集上训练）
  ③ oracle_llm（人工 Oracle 正确证据）
- 在跨事件题、已变更最新事实题、难负例（拒答/防串台）与原 6 题无关键词闲聊上实测，
  定量区分「没想起（检索缺口）」与「想起但小模型不会说（表达缺口）」。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if not os.environ.get("HF_HOME") and (ROOT / ".pytest_cache" / "hf").exists():
    os.environ["HF_HOME"] = str(ROOT / ".pytest_cache" / "hf")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from assoc_memory import (
    EpisodicEvent,
    EpisodicStore,
    FrozenTextEncoder,
    IterativeRecallNet,
    OneShotBaseline,
    SparseAssociativeGraph,
    train_iterative_recall,
)
from tools.eval_assoc_e1 import build_ayuan_eval_world, build_disjoint_train_world
from tools.test_diary_10k import NO_KEYWORD_PROMPTS


class AssocLanguageInterface:
    """独立语言接口：仅负责把记忆系统选中的 0~2 条证据翻译成自然中文回复。"""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.lm = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
        self.lm.eval()

    @torch.no_grad()
    def reply(self, query: str, events: Sequence[EpisodicEvent], max_new_tokens: int = 110) -> str:
        if events:
            ev_lines = "\n".join(f"- [Day {e.day}] {e.text}" for e in events)
            sys_prompt = (
                "你是用户的私人生活助手，请用简洁自然的中文直接回答用户问题。"
                "如果下方提供了用户过往经历片段，请结合这些真实事实回答，不要编造未提及的细节。"
            )
            user_prompt = f"【已回忆起的个人经历】\n{ev_lines}\n\n【用户问题】\n{query}"
        else:
            sys_prompt = (
                "你是用户的日常助手，请用简洁自然的中文直接回答问题。"
                "当前没有检索到相关个人记忆，请作为通用问题直接作答或如实说明不了解其私人细节，不要编造个人经历。"
            )
            user_prompt = query

        msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
        out = self.lm.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tok.eos_token_id,
        )
        return self.tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True).strip()


def any_kw(text: str, kws: Sequence[str]) -> bool:
    tl = text.lower()
    return any(k.lower() in tl for k in kws)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--budget", type=int, default=16)
    args = ap.parse_args()

    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== E4 接入冻结小 LLM 语言接口对比（llm={args.llm}, dev={dev}）===", flush=True)

    # 1. 训练并准备联想记忆系统（仅在独立 12 主题训练集上训练，零泄露）
    encoder = FrozenTextEncoder("BAAI/bge-small-zh-v1.5", device=dev)
    tr_store, tr_graph, tr_examples = build_disjoint_train_world(encoder)
    ev_store, ev_graph, eval_sets, _smoke6 = build_ayuan_eval_world(encoder)

    b_oneshot = OneShotBaseline()
    b_oneshot.calibrate(tr_store, tr_examples)

    torch.manual_seed(args.seed)
    net = IterativeRecallNet(emb_dim=encoder.dim, hidden_dim=32, low_rank=16).to(dev)
    train_iterative_recall(
        net, tr_store, tr_graph, tr_examples, epochs=args.epochs, lr=4e-3, candidate_budget=args.budget, seed=args.seed
    )
    abs_t = net.calibrate_abstain_thresh(tr_store, tr_graph, tr_examples, mode="hybrid", candidate_budget=args.budget)
    print(f"记忆系统就绪 | one_shot thresh={b_oneshot.thresh1:.2f} | learned abstain_thresh={abs_t:.2f}", flush=True)

    # 先跑完所有记忆检索并释放 BGE 显存，再加载 Qwen2.5-1.5B（确保单卡 8GB 显存充裕）
    cross_items = [
        (eval_sets["cross"][0], ["陈阿姨"]),
        (eval_sets["cross"][1], ["阿尔伯特", "Albert"]),
        (eval_sets["cross"][2], ["陈阿姨"]),
        (eval_sets["cross"][3], ["氯雷他定", "玄关"]),
        (eval_sets["cross"][5], ["弗林德斯", "暗房"]),
        (eval_sets["cross"][6], ["北海道"]),
        (eval_sets["cross"][7], ["原型", "生活费", "周末"]),
        (eval_sets["cross"][8], ["TypeScript", "前端"]),
        (eval_sets["cross"][9], ["预算", "实现"]),
        (eval_sets["cross"][10], ["书桌"]),
        (eval_sets["cross"][11], ["周五", "八点"]),
    ]
    sup_items = [
        (eval_sets["superseded"][0], ["周五", "八点"], ["周三", "七点"]),
        (eval_sets["superseded"][1], ["北海道"], ["去冲绳"]),
        (eval_sets["superseded"][2], ["燕麦拿铁", "少冰"], ["全脂"]),
        (eval_sets["superseded"][3], ["陈阿姨"], []),
    ]
    hard_neg_items = [
        (eval_sets["hard_neg"][0], ["旺财", "陈阿姨", "阿远", "林远"]),
        (eval_sets["hard_neg"][1], ["过敏", "起疹子", "氯雷他定", "阿远"]),
        (eval_sets["hard_neg"][2], ["记账", "阿远", "林远"]),
        (eval_sets["hard_neg"][3], ["老富士", "冲绳", "阿远"]),
        (eval_sets["hard_neg"][4], ["旺财", "打雷", "陈阿姨"]),
        (eval_sets["hard_neg"][5], ["沙发", "阿远", "林远"]),
    ]

    # 预先缓存所有查询的三路检索事件 ID
    recalled_cache: Dict[str, Dict[str, List[str]]] = {}
    all_qs = (
        [ex.query for ex, _ in cross_items]
        + [ex.query for ex, _, _ in sup_items]
        + [ex.query for ex, _ in hard_neg_items]
        + [q for _, q in NO_KEYWORD_PROMPTS]
    )
    gold_map = {ex.query: ex.gold_ids for ex, _ in cross_items}
    gold_map.update({ex.query: ex.gold_ids for ex, _, _ in sup_items})
    gold_map.update({ex.query: [] for ex, _ in hard_neg_items})

    for q in all_qs:
        r_one = b_oneshot.recall(ev_store, q, top_k=2).event_ids
        r_hyb = net.recall(ev_store, ev_graph, q, mode="hybrid", candidate_budget=args.budget, abstain_thresh=abs_t).event_ids
        r_ora = gold_map.get(q, r_hyb)
        recalled_cache[q] = {"one_shot": r_one, "learned_hybrid": r_hyb, "oracle": r_ora}

    # 卸载编码器到 CPU 腾出显存
    encoder.model.to("cpu")
    if dev == "cuda":
        torch.cuda.empty_cache()

    # 2. 加载冻结 Qwen2.5-1.5B-Instruct 作为统一语言接口
    lang = AssocLanguageInterface(model_name=args.llm, device=dev)

    arms = ("one_shot", "learned_hybrid", "oracle")
    arm_metrics: Dict[str, Dict[str, float]] = {}
    samples: Dict[str, List[Dict]] = {a: [] for a in arms}

    for arm in arms:
        c_ret = c_ans = 0
        for ex, kws in cross_items:
            eids = recalled_cache[ex.query][arm]
            evs = [ev_store.get(eid) for eid in eids]
            ans = lang.reply(ex.query, evs)
            ret_ok = bool(ex.target_id and ex.target_id in eids)
            ans_ok = any_kw(ans, kws)
            c_ret += int(ret_ok)
            c_ans += int(ans_ok)
            samples[arm].append({"type": "cross", "q": ex.query, "eids": eids, "ans": ans, "ret_ok": ret_ok, "ans_ok": ans_ok})

        s_ans = s_leak = 0
        for ex, pos_kws, bad_kws in sup_items:
            eids = recalled_cache[ex.query][arm]
            evs = [ev_store.get(eid) for eid in eids]
            ans = lang.reply(ex.query, evs)
            ok = any_kw(ans, pos_kws)
            leak = bool(bad_kws and any_kw(ans, bad_kws) and not ok)
            s_ans += int(ok)
            s_leak += int(leak)
            samples[arm].append({"type": "superseded", "q": ex.query, "eids": eids, "ans": ans, "ans_ok": ok, "leak": leak})

        hn_intrude = 0
        for ex, forbidden in hard_neg_items:
            eids = recalled_cache[ex.query][arm]
            evs = [ev_store.get(eid) for eid in eids]
            ans = lang.reply(ex.query, evs)
            intrude = any_kw(ans, forbidden) or len(eids) > 0
            hn_intrude += int(intrude)
            samples[arm].append({"type": "hard_neg", "q": ex.query, "eids": eids, "ans": ans, "intrude": intrude})

        arm_metrics[arm] = {
            "cross_retrieval_target": round(c_ret / len(cross_items), 4),
            "cross_llm_answer_acc": round(c_ans / len(cross_items), 4),
            "superseded_llm_acc": round(s_ans / len(sup_items), 4),
            "superseded_old_leak": round(s_leak / len(sup_items), 4),
            "hard_neg_intrusion_rate": round(hn_intrude / len(hard_neg_items), 4),
        }

    # 原 6 题日常闲聊定性对照（one_shot vs learned_hybrid）
    smoke_chat = []
    for tag, q in NO_KEYWORD_PROMPTS:
        row = {"tag": tag, "q": q}
        for arm in ("one_shot", "learned_hybrid"):
            eids = recalled_cache[q][arm]
            evs = [ev_store.get(eid) for eid in eids]
            row[arm] = {"eids": eids, "reply": lang.reply(q, evs)}
        smoke_chat.append(row)

    print("\n=== E4 接回冻结 Qwen2.5-1.5B-Instruct 核心对比（区分“没想起”与“想起但不会说”）===")
    header = f"{'路数':<16} | {'跨事件检索命中':>10} | {'跨事件LLM答对':>10} | {'已变更LLM答对':>10} | {'难负例串台↓':>9}"
    print(header)
    print("-" * len(header))
    for arm in arms:
        m = arm_metrics[arm]
        print(
            f"{arm:<16} | {m['cross_retrieval_target']:>10.1%} | {m['cross_llm_answer_acc']:>10.1%} | "
            f"{m['superseded_llm_acc']:>10.1%} | {m['hard_neg_intrusion_rate']:>9.1%}",
            flush=True,
        )

    elapsed = round(time.time() - t0, 1)
    out_path = ROOT / f"runs/{time.strftime('%Y%m%d-%H%M%S')}_assoc_e4.json"
    payload = {
        "experiment": "E4_llm_interface",
        "llm": args.llm,
        "encoder": "BAAI/bge-small-zh-v1.5",
        "seed": args.seed,
        "elapsed_s": elapsed,
        "metrics": arm_metrics,
        "samples": samples,
        "smoke6_chat": smoke_chat,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存：{out_path}（耗时 {elapsed}s）", flush=True)


if __name__ == "__main__":
    main()
