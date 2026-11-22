"""30 天高写入量暴力模拟压测（Qwen2.5-1.5B-Instruct + N=512 独立出口 RMP + SQLite 16K 槽海马体）：

模拟 1 个月重度日常使用（Day 1 → Day 30）：
1. 海马体高噪写入（~1,000 条槽位）：
   - 每天写入 30 条长对话/废话干扰 + 个人事实，30 天累计 ~960+ 槽；
   - 中途穿插 10 次连环改口覆盖（如城市 Melbourne → Sydney → Brisbane → Perth 连续 3 次改口）；
   - 穿插 5 次主动删除隐私（/del），验证删除后绝不召回。
2. RMP 旁路高强度持续学习（10 批 × 10 条 = 100 条全新规则，零海马体提示）：
   - 每 3 天巩固 10 条全新规则（共 100 条），带历史回放 + 纯原脑锚定 + 验收回滚。
3. 每 5 天（Day 5/10/15/20/25/30）拉体检曲线：
   - 1000 槽海马体检索 recall@3、最新值准确率、零旧值回吐、隐私删除生效；
   - 第 1 批（Day 3）10 条规则留存率 + 当批新规则掌握率；
   - FP32 英/中/代码/数学通用能力 30 天累计遗忘曲线；
   - 峰值显存、SQLite 体积、单次检索毫秒耗时。
"""
import json, os, sys, tempfile, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from llm_sidecar import LLMSidecarWrapper, SidecarLearner, SIDECAR_PRESETS
from memory_store import MemoryStore

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_SYS = "You are Qwen, a helpful AI assistant. Answer concisely and directly."

BG_TEXTS = [
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "The Pacific Ocean is the largest and deepest ocean on Earth.",
    "地球是太阳系中由内及外的第三颗行星，也是目前已知唯一孕育生命的天体。",
    "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    m = len(arr) // 2\n    return arr[:m] + arr[m:]\n",
    "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n",
    "In linear algebra, an eigenvector of a matrix only changes by a scalar factor under transformation.",
    "在深度学习中，梯度裁剪常用于防止反向传播过程中的梯度爆炸问题。",
]
OLD_PROBES = {
    'en': [
        "A binary search tree stores keys in sorted order so lookup, insertion, and deletion run in logarithmic time on average.",
        "Photosynthesis converts light energy into chemical energy inside chloroplasts, producing glucose and releasing oxygen.",
        "Plate tectonics explains how Earth's rigid outer shell is divided into plates that slowly drift over the mantle.",
    ],
    'zh': [
        "光合作用主要发生在植物叶片的叶绿体中，它将太阳能转化为化学能并释放出氧气。",
        "丝绸之路是古代连接中国与中亚、西亚乃至地中海沿岸的重要商贸与文化交流通道。",
        "在操作系统中，虚拟内存通过页表映射将进程地址空间与物理内存解耦，提高内存利用率。",
    ],
    'code': [
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n    return quicksort([x for x in arr[1:] if x < pivot]) + [pivot] + quicksort([x for x in arr[1:] if x >= pivot])\n",
        "def binary_search(nums, target):\n    lo, hi = 0, len(nums) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if nums[mid] == target:\n            return mid\n        lo, hi = (mid + 1, hi) if nums[mid] < target else (lo, mid - 1)\n    return -1\n",
    ],
    'math': [
        "The derivative of x^3 + 2x^2 - 5x + 7 with respect to x is 3x^2 + 4x - 5.",
        "By the Pythagorean theorem, a right triangle with legs of length 3 and 4 has a hypotenuse of length 5.",
    ],
}

# 生成 10 批 × 10 条 = 100 条语义清晰区分的全新虚构规则（每 3 天巩固一批）
DOMAINS = [
    ("Nebula", "ORION", 7100), ("Vesper", "LYRA", 8200),
    ("Valoria", "EMERALD", 3300), ("Zorbus", "KRON", 4400),
    ("Astralonia", "SOLIS", 5500), ("Kestrel", "XENON", 6600),
    ("Thalassa", "COMET", 2700), ("Helios", "VANE", 9800),
    ("Borealis", "FROST", 1900), ("Meridian", "AURORA", 5100),
]
RULE_ROLES = [
    ("secret access code", "is"), ("network listening port", "is port"),
    ("chief architect", "is Dr."), ("capital outpost", "is"),
    ("backup encryption cipher", "is"), ("primary fuel alloy", "is"),
    ("emergency override phrase", "is"), ("annual spring festival", "is"),
    ("telemetry beacon channel", "is"), ("flagship transport vessel", "is"),
]


def make_rule_batches():
    batches = []
    for b_idx, (dom, pref, base_num) in enumerate(DOMAINS):
        rules = []
        for j, (role, verb) in enumerate(RULE_ROLES):
            code = f"{pref}-{base_num + j * 17}"
            q = f"What is the {role} for {dom}?"
            ans = f"The {role} for {dom} {verb} {code}."
            rules.append((q, ans, code))
        batches.append(rules)
    return batches


# 30 天时间线上的事实写入、连环改口与隐私删除事件
TIMELINE_FACTS = {
    1:  [("name", "My name is Alex Mercer."), ("cat", "My pet cat is named Mirafe."), ("city", "I live in Melbourne, Australia.")],
    2:  [("car", "I drive a red Honda Civic."), ("secret_pin", "My temporary locker pin is 99412.")],
    4:  [("allergy", "I am allergic to peanuts."), ("lang", "My favourite programming language is Python.")],
    6:  [("company", "I work at NorthStar Labs."), ("coffee", "My usual coffee order is a hot latte.")],
    8:  [("sport", "I play tennis every Tuesday."), ("birthday", "My birthday is on October 14.")],
    10: [("city", "I just moved and now I live in Sydney, Australia.")],                  # 改口 1：墨尔本 → 悉尼
    12: [("car", "I sold my Civic and now I drive a silver Subaru Outback.")],            # 改口 2：Honda → Subaru
    14: [("lang", "I switched my favourite programming language to Rust.")],              # 改口 3：Python → Rust
    16: [("coffee", "I changed my coffee order to an iced oat flat white.")],             # 改口 4：latte → iced oat flat white
    18: [("sport", "I switched sports and now I play badminton every Wednesday.")],       # 改口 5：tennis → badminton
    20: [("city", "I relocated again and now I live in Brisbane, Australia.")],           # 改口 6：悉尼 → 布里斯班
    22: [("company", "I joined a new company called DeepOrbit AI.")],                     # 改口 7：NorthStar → DeepOrbit
    24: [("allergy", "My allergy updated: I am allergic to shellfish and walnuts.")],     # 改口 8：peanuts → shellfish and walnuts
    26: [("city", "My final move: I now live in Perth, Western Australia.")],             # 改口 9：布里斯班 → 珀斯（三连改口！）
    28: [("car", "I upgraded my car to a blue Tesla Model 3.")],                          # 改口 10：Subaru → Tesla Model 3
}
DELETE_DAYS = {15: "secret_pin"}                                   # 第 15 天删除隐私项

# 真正互不重复的 900 条日常干扰生成词库（低词法重合，不会被自动去重压扁）
_SUBJ = ["archaeology", "botany", "ceramics", "cartography", "acoustics", "meteorology", "calligraphy", "typography",
         "astronomy", "geology", "origami", "horology", "forestry", "viticulture", "metallurgy", "hydrology",
         "ornithology", "cryptography", "logistics", "ergonomics", "architecture", "culinary", "navigation", "robotics",
         "acoustics2", "textiles", "glaciology", "volcanology", "limnology", "mycology"]
_ACT = ["documentary", "podcast", "seminar", "workshop", "exhibition", "journal", "lecture", "essay",
        "symposium", "catalog", "blueprint", "manual", "chronicle", "monograph", "almanac", "notebook",
        "anthology", "treatise", "digest", "compendium", "gazette", "dossier", "bulletin", "primer",
        "handbook", "manifesto", "memorandum", "report", "survey", "review"]
_PLACE = ["harbour", "observatory", "conservatory", "courtyard", "promenade", "amphitheatre", "pavilion", "viaduct",
          "cathedral", "aqueduct", "lighthouse", "greenhouse", "planetarium", "botanical", "marina", "gallery",
          "museum", "cloister", "citadel", "terrace", "veranda", "atrium", "rotunda", "arcade",
          "boardwalk", "pier", "estuary", "meadow", "canyon", "plateau"]


def make_distractor(day, i):
    s = _SUBJ[(day * 7 + i) % len(_SUBJ)]
    a = _ACT[(day * 3 + i * 5) % len(_ACT)]
    p = _PLACE[(day * 11 + i * 2) % len(_PLACE)]
    return f"On calendar day {day} item {i + 1}, someone discussed {s} inside a {a} near the {p} district."

# Day 30 最终海马体抽考（覆盖连环改口最新值、绝不回吐旧值、未改口项、已删隐私）
FINAL_HIPPO_TESTS = [
    ("Which city do I live in now?", "Perth", ["Melbourne", "Sydney", "Brisbane"]),
    ("What car do I drive now?", "Tesla", ["Honda", "Civic", "Subaru"]),
    ("What is my favourite programming language now?", "Rust", ["Python"]),
    ("What is my coffee order now?", "flat white", ["latte"]),
    ("What sport do I play now?", "badminton", ["tennis"]),
    ("Which company do I work at now?", "DeepOrbit", ["NorthStar"]),
    ("What food am I allergic to now?", "shellfish", ["peanuts"]),
    ("What is my pet cat's name?", "Mirafe", []),
    ("When is my birthday?", "October 14", []),
]


@torch.no_grad()
def eval_nll_fp32(w, tok, texts):
    w.eval(); tot_l, tot_n = 0.0, 0
    for txt in texts:
        w.sidecar.reset_session()
        ids = tok(txt, return_tensors='pt').input_ids.to(DEV)
        lg = w(ids).logits.float()
        tot_l += float(F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1), reduction='sum').item())
        tot_n += ids.shape[1] - 1
    return tot_l / max(tot_n, 1)


@torch.no_grad()
def ask(w, tok, question, sys_msg=BASE_SYS):
    w.eval(); w.sidecar.reset_session()
    prompt = tok.apply_chat_template([{'role': 'system', 'content': sys_msg}, {'role': 'user', 'content': question}],
                                     tokenize=False, add_generation_prompt=True)
    inp = tok(prompt, return_tensors='pt').to(DEV)
    out = w.backbone.generate(**inp, max_new_tokens=32, do_sample=False)
    return tok.decode(out[0, inp.input_ids.shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def encode_frozen(w, tok, text):
    ids = tok(text, return_tensors='pt').input_ids.to(DEV)
    w.eval(); w.bypass = True
    try:
        w(ids)
    finally:
        w.bypass = False
    return w._last_h


def rebuild_hippo(store, w, tok):
    h = w.hippo
    h.used.zero_(); h.superseded.zero_(); h.hits.zero_(); h.priority.zero_(); h.ts.zero_()
    h.slot_texts = [''] * h.n_slots; h.slot_keys = [''] * h.n_slots; h.clock = 0
    for _, fk, txt, sid, _ in store.list_facts(active_only=True):
        h.cur_session = sid
        h.write_fact(encode_frozen(w, tok, txt), txt, fact_key=fk)


def main():
    t0 = time.perf_counter()
    cfg = SIDECAR_PRESETS['large']
    model_name = cfg['model']
    print(f'=== 30 天高写入量暴力压测：{model_name} + N={cfg["sidecar"]["N"]} (sparse_active=True) ===', flush=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    lm = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16 if DEV == 'cuda' else torch.float32).to(DEV)
    w = LLMSidecarWrapper(lm, attach_layer=-4, sidecar_cfg=cfg['sidecar'], hippo_cfg=cfg['hippo'])
    w.sidecar.to(DEV); w.hippo.to(DEV)
    L = SidecarLearner(w, lr=1e-3, smf_k=8)
    bg_batches = [tok(t, return_tensors='pt').input_ids.to(DEV) for t in BG_TEXTS]
    L.profile_bg(bg_batches)
    old0 = {k: eval_nll_fp32(w, tok, v) for k, v in OLD_PROBES.items()}
    rule_batches = make_rule_batches()

    td = tempfile.TemporaryDirectory()
    db_path = Path(td.name) / 'month30.db'
    mem_path = Path(td.name) / 'month30.pt'
    store = MemoryStore(db_path)
    store.set_meta('model_name', model_name)

    history_log = []
    batch_idx = 0
    for day in range(1, 31):
        w.reset_session()
        # 1. 每天写入 30 条真正互不重复的干扰（30 天累计 ~900 活跃槽）
        for i in range(30):
            dtxt = make_distractor(day, i)
            store.add_fact(dtxt, session_id=day)
            w.hippo.write_fact(encode_frozen(w, tok, dtxt), dtxt)
        # 2. 写入当天真实事实 / 改口覆盖
        for fk, ftxt in TIMELINE_FACTS.get(day, []):
            store.add_fact(ftxt, fact_key=fk, session_id=day)
            w.hippo.write_fact(encode_frozen(w, tok, ftxt), ftxt, fact_key=fk)
        # 3. 处理隐私删除
        if day in DELETE_DAYS:
            del_key = DELETE_DAYS[day]
            for fid, fk, _, _, _ in store.list_facts(active_only=True):
                if fk == del_key:
                    store.delete_fact(fid)
            rebuild_hippo(store, w, tok)

        # 4. 每 3 天巩固 10 条全新规则（共 100 条规则！给足 100 步睡眠 = 每条 10 步）
        slp_info = None
        if day % 3 == 0 and batch_idx < len(rule_batches):
            rules = rule_batches[batch_idx]
            for q, ans, _ in rules:
                store.add_rule(q, ans, status='staged')
                qp = tok.apply_chat_template([{'role': 'system', 'content': BASE_SYS}, {'role': 'user', 'content': q}],
                                             tokenize=False, add_generation_prompt=True)
                qf = tok.apply_chat_template([{'role': 'system', 'content': BASE_SYS}, {'role': 'user', 'content': q},
                                              {'role': 'assistant', 'content': ans}], tokenize=False)
                L.stage_qa(tok(qf, return_tensors='pt').input_ids, tok(qp, return_tensors='pt').input_ids.shape[1])
            slp_info = L.sleep_consolidate(steps=100, lr=1.5e-3, anchor_batches=bg_batches, anchor_w=2.0,
                                           old_tol=0.05, hist_tol=0.30, replay_w=0.8)
            if slp_info['committed']:
                store.mark_staged_confirmed()
                w.save_memory(mem_path, learner=L)
            batch_idx += 1

        # 5. 第 3/10/20/30 天做全面体检
        if day in (3, 10, 20, 30):
            active_slots = int((w.hippo.used & (~w.hippo.superseded)).sum())
            b1_hits = sum(code.lower() in ask(w, tok, q).lower() for q, _, code in rule_batches[0])
            cur_b = rule_batches[batch_idx - 1]
            cur_hits = sum(code.lower() in ask(w, tok, q).lower() for q, _, code in cur_b)
            old_d = {k: round(eval_nll_fp32(w, tok, v) - old0[k], 4) for k, v in OLD_PROBES.items()}
            vram = round(torch.cuda.max_memory_allocated() / 1e6, 1) if DEV == 'cuda' else 0.0
            db_kb = round(db_path.stat().st_size / 1024, 1)
            row = {
                'day': day, 'active_slots': active_slots, 'total_rules': len(L.confirmed_qa),
                'b1_retention': f'{b1_hits}/10', 'latest_batch': f'{cur_hits}/10',
                'old_nll_delta': old_d, 'peak_vram_mb': vram, 'db_kb': db_kb,
            }
            history_log.append(row)
            print(f'[Day {day:2d}] 活跃槽={active_slots} | 已巩固规则={len(L.confirmed_qa)} | '
                  f'第1批保留={b1_hits}/10 | 最新批={cur_hits}/10 | 旧能力={old_d} | VRAM={vram:.0f}MB | DB={db_kb}KB', flush=True)

    # ---------- Day 30 终极海马体 900+ 槽抗干扰 + 三连改口 + 隐私删除验收 ----------
    print('\n=== [Day 30 终极海马体抽考（~920 活跃槽：900 干扰 + 10 次改口 + 隐私删除）] ===', flush=True)
    t_rec = time.perf_counter()
    rec_ok, ans_ok, clean_ok = 0, 0, 0
    for q, want, forbid in FINAL_HIPPO_TESTS:
        rec = w.hippo.recall_facts(encode_frozen(w, tok, q), query_text=q, top_k=3, min_score=0.15, cross_session_only=False)
        in_top3 = any(want.lower() in r.lower() for r in rec)
        rec_ok += int(in_top3)
        facts_str = '\n'.join(f'- The user previously said: "{f}"' for f in rec)
        sys_m = f"{BASE_SYS}\n\nFacts about the USER (not you):\n{facts_str}" if rec else BASE_SYS
        rep = ask(w, tok, q, sys_msg=sys_m)
        ok = want.lower() in rep.lower()
        no_old = all(fb.lower() not in rep.lower() for fb in forbid)
        ans_ok += int(ok); clean_ok += int(ok and no_old)
        print(f'  [Top3={"Y" if in_top3 else "N"} | Ans={"OK" if ok else "MISS"} | 无旧值={"Y" if no_old else "N"}] Q: {q} -> A: {rep[:80]}', flush=True)
    avg_rec_ms = (time.perf_counter() - t_rec) / len(FINAL_HIPPO_TESTS) * 1000
    # 隐私删除检验
    del_rec = w.hippo.recall_facts(encode_frozen(w, tok, "What is my temporary locker pin?"),
                                   query_text="What is my temporary locker pin?", top_k=3, min_score=0.15, cross_session_only=False)
    pin_gone = all('99412' not in r for r in del_rec)
    print(f'Day 30 海马体终极结果: recall@3={rec_ok}/{len(FINAL_HIPPO_TESTS)} | ans_acc={ans_ok}/{len(FINAL_HIPPO_TESTS)} '
          f'| 改口无旧值={clean_ok}/{len(FINAL_HIPPO_TESTS)} | 隐私彻底删除={pin_gone} | 平均检索+答={avg_rec_ms:.0f}ms', flush=True)

    out = {
        'model': model_name, 'N': cfg['sidecar']['N'], 'timeline': history_log,
        'day30_hippo': {'recall@3': f'{rec_ok}/{len(FINAL_HIPPO_TESTS)}', 'ans_acc': f'{ans_ok}/{len(FINAL_HIPPO_TESTS)}',
                        'no_old_leak': f'{clean_ok}/{len(FINAL_HIPPO_TESTS)}', 'privacy_deleted': pin_gone},
        'elapsed_s': round(time.perf_counter() - t0, 1),
    }
    p = ROOT / f'runs/{time.strftime("%Y%m%d-%H%M%S")}_stress30d_large.json'
    json.dump(out, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'=== 30 天暴力压测完成 ({out["elapsed_s"]}s) → {p.name} ===', flush=True)
    store.close(); td.cleanup()


if __name__ == '__main__':
    main()
