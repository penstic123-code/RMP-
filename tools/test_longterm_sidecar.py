"""长期可用性完整验收套件（Qwen2.5-0.5B + 独立出口 RMP N=48 + 可改口海马体 + 原子存盘）：

1. 海马体 6/6 抗干扰 + 改口覆盖（墨尔本 → 悉尼）+ 真实子进程重启读档验收；
2. RMP 5 轮连续学习（每轮 6 条全新规则，共 30 条，全程关掉海马体文本提示）：
   - 每轮报告：睡眠提交状态、本轮新规则答对数、第 1 轮 6 条规则保留数（门槛 ≥5/6）、FP32 通用旧能力 NLL 变化、峰值显存。
"""
import argparse, os, subprocess, sys, tempfile, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from llm_sidecar import LLMSidecarWrapper, SidecarLearner, SIDECAR_PRESETS

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_SYS = "You are Qwen, a helpful AI assistant. Answer concisely and directly."

# 5 轮 × 6 条全新虚构规则（每条附 1 个未训练换问法）
ROUNDS = [
    [
        ("What is the secret code for Project Nebula?", "The secret code for Project Nebula is ORION-94.", "ORION-94", "Tell me Project Nebula's secret code."),
        ("Who is the captain of Starship Vesper?", "The captain of Starship Vesper is Commander Lyra Chen.", "Lyra Chen", "Who commands Starship Vesper?"),
        ("What color is the crystal of Valoria?", "The crystal of Valoria is emerald violet.", "emerald violet", "Describe the color of Valoria's crystal."),
        ("What port does the Zorbus daemon listen on?", "The Zorbus daemon listens on port 7741.", "7741", "Which network port is used by the Zorbus daemon?"),
        ("What is the capital city of Astralonia?", "The capital city of Astralonia is Solisgrad.", "Solisgrad", "Name the capital of Astralonia."),
        ("Which element powers the Kestrel engine?", "The Kestrel engine is powered by liquid xenon.", "xenon", "What fuel element runs the Kestrel engine?"),
    ],
    [
        ("What is the emergency override phrase for Vault 12?", "The emergency override phrase for Vault 12 is SILENT-COMET.", "SILENT-COMET", "What phrase triggers the Vault 12 emergency override?"),
        ("How many moons orbit planet Thalassa?", "Planet Thalassa has exactly five moons.", "five", "Count the moons orbiting Thalassa."),
        ("What currency is used in Republic of Verdan?", "The currency of Republic of Verdan is the silver kron.", "kron", "Name the official currency of Verdan."),
        ("Who founded the Helios Institute?", "The Helios Institute was founded by Dr. Aris Vane.", "Aris Vane", "Who is the founder of the Helios Institute?"),
        ("What is the max speed of the Falcon-9X glider?", "The max speed of the Falcon-9X glider is 620 knots.", "620", "How fast can the Falcon-9X glider fly?"),
        ("Which river flows through Old Kaldor?", "The Amber River flows through Old Kaldor.", "Amber", "Name the river passing through Old Kaldor."),
    ],
    [
        ("What is the default timeout for Chronos sync?", "The default timeout for Chronos sync is 45 seconds.", "45", "How long is the default Chronos sync timeout?"),
        ("Which mineral is mined on Asteroid Vex-4?", "Asteroid Vex-4 is mined for cobaltite.", "cobaltite", "What mineral comes from Asteroid Vex-4?"),
        ("Who painted The Glass Horizon?", "The Glass Horizon was painted by Elena Moroz.", "Elena Moroz", "Who is the artist behind The Glass Horizon?"),
        ("What is the callsign of Outpost Borealis?", "The callsign of Outpost Borealis is FROST-7.", "FROST-7", "Give the radio callsign for Outpost Borealis."),
        ("Which alloy is used in Aegis hull plating?", "Aegis hull plating is made of titanium-carbide.", "titanium-carbide", "What alloy makes up Aegis hull plating?"),
        ("What festival is celebrated in Lumina every spring?", "Lumina celebrates the Lantern Tide every spring.", "Lantern Tide", "Name Lumina's annual spring festival."),
    ],
    [
        ("What is the admin key prefix for NexusDB?", "The admin key prefix for NexusDB is NXS_ROOT.", "NXS_ROOT", "Which prefix do NexusDB admin keys use?"),
        ("How tall is Mount Zephyr?", "Mount Zephyr is 4,820 meters tall.", "4,820", "What is the elevation of Mount Zephyr?"),
        ("Who composed Symphony of Rust?", "Symphony of Rust was composed by Viktor Solberg.", "Viktor Solberg", "Name the composer of Symphony of Rust."),
        ("What is the backup frequency for Beacon Delta?", "The backup frequency for Beacon Delta is 144.8 MHz.", "144.8", "Which frequency serves as backup for Beacon Delta?"),
        ("Which tree is the national symbol of Oakhaven?", "The silver birch is the national symbol of Oakhaven.", "silver birch", "Name Oakhaven's national tree."),
        ("What is the serial prefix for Drone Mk-IV?", "The serial prefix for Drone Mk-IV is DRN-400.", "DRN-400", "How do Drone Mk-IV serial numbers begin?"),
    ],
    [
        ("What is the default cipher for MeshLink v2?", "The default cipher for MeshLink v2 is ChaCha-512.", "ChaCha-512", "Which cipher does MeshLink v2 use by default?"),
        ("Who governs the Cloud Citadel of Nyx?", "The Cloud Citadel of Nyx is governed by Regent Sera.", "Regent Sera", "Name the ruler of the Cloud Citadel of Nyx."),
        ("What is the freezing point of CryoGel-B?", "CryoGel-B freezes at minus 85 degrees Celsius.", "85", "At what temperature does CryoGel-B freeze?"),
        ("Which guild maintains the Aqueduct of Sol?", "The Stonewright Guild maintains the Aqueduct of Sol.", "Stonewright", "Who is responsible for maintaining the Aqueduct of Sol?"),
        ("What is the batch limit for PulseQueue?", "The batch limit for PulseQueue is 256 messages.", "256", "How many messages can a single PulseQueue batch hold?"),
        ("What is the flagship vessel of the Meridian Fleet?", "The flagship vessel of the Meridian Fleet is HMS Aurora.", "Aurora", "Name the Meridian Fleet's flagship."),
    ],
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
BG_TEXTS = [
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "The Pacific Ocean is the largest and deepest ocean on Earth.",
    "地球是太阳系中由内及外的第三颗行星，也是目前已知唯一孕育生命的天体。",
    "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    m = len(arr) // 2\n    return arr[:m] + arr[m:]\n",
    "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n",
    "In linear algebra, an eigenvector of a matrix only changes by a scalar factor under transformation.",
    "在深度学习中，梯度裁剪常用于防止反向传播过程中的梯度爆炸问题。",
]

USER_FACTS = [
    ("name", "My name is Alex Mercer."),
    ("cat", "My pet cat is named Mirafe and she is three years old."),
    ("city", "I live in Melbourne, Australia."),                    # 稍后会改口覆盖为 Sydney！
    ("allergy", "I am allergic to peanuts and shellfish."),
    ("lang", "My favourite programming language is Rust."),
    ("friend", "My best friend from university is named Daniel Park."),
    ("car", "I drive a silver Subaru Outback."),
    ("coffee", "My coffee order is always an iced oat flat white."),
    ("project", "I am currently building a neural memory architecture called DSPN."),
    ("birthday", "My birthday is on October 14."),
    ("sport", "I play badminton every Wednesday evening."),
    ("movie", "My favourite movie is Interstellar."),
]
DISTRACTORS = [
    "The weather looks a bit cloudy outside today, maybe it will rain later.",
    "I was reading an article about Roman architecture and aqueducts yesterday.",
    "Traffic on the highway was unusually slow this morning due to roadwork.",
    "I need to remember to buy groceries, paper towels, and laundry detergent.",
    "Classical music like Chopin nocturnes helps me focus while reading.",
    "Sometimes debugging memory leaks in C++ takes hours of careful inspection.",
    "The local library just renovated its second floor with quiet study pods.",
    "Solar panels have become much cheaper over the last decade worldwide.",
    "I watched a documentary about deep sea hydrothermal vents last night.",
    "Learning a second language gets easier when you practice speaking daily.",
    "Fresh sourdough bread smells amazing right out of the bakery oven.",
    "Public transit trains were running right on schedule all week long.",
    "Winter mornings are crisp, though the afternoon sun warms things up nicely.",
    "Organizing files into clear folders always saves time in the long run.",
    "A good cup of green tea in the afternoon keeps energy steady.",
    "Mountains near the coast often create sharp local weather changes.",
]
FACT_TESTS = [
    ("What is my pet cat's name?", "Mirafe", "Mirafe"),
    ("What food am I allergic to?", "peanut", "peanuts"),
    ("Which city do I live in?", "Sydney", "Sydney"),              # 必须答 Sydney 且不答 Melbourne！
    ("What car do I drive?", "Subaru", "Subaru"),
    ("When is my birthday?", "October 14", "October 14"),
    ("What sport do I play on Wednesday evenings?", "badminton", "badminton"),
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
    out = w.backbone.generate(**inp, max_new_tokens=36, do_sample=False)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preset', default='small', choices=list(SIDECAR_PRESETS))
    ap.add_argument('--skip_restart', action='store_true', help='跳过子进程重启（节省大底座二次加载时间）')
    a = ap.parse_args()
    cfg = SIDECAR_PRESETS[a.preset]
    t0 = time.perf_counter()
    model_name = cfg['model']
    print(f'=== 加载 [{a.preset}] {model_name} (N={cfg["sidecar"]["N"]}, {DEV}) ===', flush=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    lm = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16 if DEV == 'cuda' else torch.float32).to(DEV)
    w = LLMSidecarWrapper(lm, attach_layer=-4, sidecar_cfg=cfg['sidecar'], hippo_cfg=cfg['hippo'])
    w.sidecar.to(DEV); w.hippo.to(DEV)
    sc_p = sum(p.numel() for p in w.sidecar.parameters()) / 1e6
    print(f'Sidecar 参数: {sc_p:.1f}M | sparse_active={w.sidecar.pool.sparse_active}', flush=True)
    L = SidecarLearner(w, lr=1e-3, smf_k=8)
    bg_batches = [tok(t, return_tensors='pt').input_ids.to(DEV) for t in BG_TEXTS]
    L.profile_bg(bg_batches)

    # ---------- 1. 海马体 6/6 + 改口覆盖（墨尔本 → 悉尼）+ 子进程重启 ----------
    print('\n=== [验收 1] 海马体 6/6 抗干扰 + 改口覆盖（墨尔本→悉尼）+ 子进程重启 ===', flush=True)
    for i in range(max(len(USER_FACTS), len(DISTRACTORS))):
        if i < len(USER_FACTS):
            fk, txt = USER_FACTS[i]
            w.hippo.write_fact(encode_frozen(w, tok, txt), txt, fact_key=fk)
        if i < len(DISTRACTORS):
            w.hippo.write_fact(encode_frozen(w, tok, DISTRACTORS[i]), DISTRACTORS[i])
    # 改口：从墨尔本搬到悉尼
    upd_txt = "I live in Sydney, Australia near Darling Harbour."
    w.hippo.write_fact(encode_frozen(w, tok, upd_txt), upd_txt, fact_key='city')
    w.reset_session()
    rec_hits, ans_hits, city_clean = 0, 0, False
    for q, ans_key, fact_key in FACT_TESTS:
        rec = w.hippo.recall_facts(encode_frozen(w, tok, q), query_text=q, top_k=3, min_score=0.15, cross_session_only=False)
        in_top3 = any(fact_key.lower() in r.lower() for r in rec)
        rec_hits += int(in_top3)
        facts_str = '\n'.join(f'- The user previously said: "{f}"' for f in rec)
        sys_m = f"{BASE_SYS}\n\nFacts about the USER (not you):\n{facts_str}" if rec else BASE_SYS
        rep = ask(w, tok, q, sys_msg=sys_m)
        ok = ans_key.lower() in rep.lower()
        if ans_key == 'Sydney':
            city_clean = ok and ('melbourne' not in rep.lower())
        ans_hits += int(ok)
        print(f'  [Top3={"Y" if in_top3 else "N"} | Ans={"OK" if ok else "MISS"}] Q: {q} -> A: {rep[:85]}', flush=True)
    print(f'海马体结果: recall@3={rec_hits}/6 | ans_acc={ans_hits}/6 | 改口只答悉尼={city_clean}', flush=True)

    # 原子存盘 + 子进程真实重启验证
    if not a.skip_restart:
        with tempfile.TemporaryDirectory() as td:
            mp = os.path.join(td, 'longterm_mem.pt')
            dbp = os.path.join(td, 'longterm.db')
            from memory_store import MemoryStore
            st = MemoryStore(dbp)
            for s in range(w.hippo.n_slots):
                if w.hippo.used[s] and not w.hippo.superseded[s] and w.hippo.slot_texts[s]:
                    st.add_fact(w.hippo.slot_texts[s], fact_key=w.hippo.slot_keys[s], session_id=int(w.hippo.session_id[s]))
            st.set_meta('model_name', model_name); st.close()
            w.save_memory(mp, learner=L)
            w.save_memory(mp, learner=L)
            has_bak = os.path.exists(mp + '.bak')
            env = dict(os.environ, TRITON_CACHE_DIR='.pytest_cache\\triton', HF_HOME='.pytest_cache\\hf')
            p_out = subprocess.check_output(
                [sys.executable, str(ROOT / 'tools/chat_llm_sidecar.py'), '--preset', a.preset, '--mem', mp, '--db', dbp,
                 '--script', "Which city do I live in?||What is my pet cat's name?"],
                cwd=str(ROOT), env=env, text=True, encoding='utf-8', errors='replace',
            )
            restart_ok = ('sydney' in p_out.lower()) and ('melbourne' not in p_out.lower()) and ('mirafe' in p_out.lower())
            print(f'子进程重启验证: .bak存在={has_bak} | 重启后仍答悉尼且记猫名Mirafe={restart_ok}', flush=True)

    # ---------- 2. RMP 5 轮连续学习（独立模块出口 + 历史回放 + 睡眠验收回滚） ----------
    print('\n=== [验收 2] RMP 5 轮连续学习（30 条新规则，每轮 6 条，零海马体文本提示） ===', flush=True)
    old0 = {k: eval_nll_fp32(w, tok, v) for k, v in OLD_PROBES.items()}
    for r_idx, rules in enumerate(ROUNDS, 1):
        for q, ans, _, _ in rules:
            qp = tok.apply_chat_template([{'role': 'system', 'content': BASE_SYS}, {'role': 'user', 'content': q}],
                                         tokenize=False, add_generation_prompt=True)
            qf = tok.apply_chat_template([{'role': 'system', 'content': BASE_SYS}, {'role': 'user', 'content': q},
                                          {'role': 'assistant', 'content': ans}], tokenize=False)
            L.stage_qa(tok(qf, return_tensors='pt').input_ids, tok(qp, return_tensors='pt').input_ids.shape[1])
        slp = L.sleep_consolidate(steps=55, lr=1.5e-3, anchor_batches=bg_batches, anchor_w=2.0, old_tol=0.05, hist_tol=0.20, replay_w=0.8)
        w.reset_session()
        cur_orig = sum(key.lower() in ask(w, tok, q).lower() for q, _, key, _ in rules)
        cur_para = sum(key.lower() in ask(w, tok, pq).lower() for _, _, key, pq in rules)
        r1_orig = sum(key.lower() in ask(w, tok, q).lower() for q, _, key, _ in ROUNDS[0])
        r1_para = sum(key.lower() in ask(w, tok, pq).lower() for _, _, key, pq in ROUNDS[0])
        old_cur = {k: round(eval_nll_fp32(w, tok, v) - old0[k], 4) for k, v in OLD_PROBES.items()}
        print(f'Round {r_idx}: sleep={slp["committed"]} (new:{slp["new_loss"]}, hist_d:{slp["hist_rise"]:+.3f}, anc_d:{slp["anchor_rise"]:+.3f}) '
              f'| 本轮原/换={cur_orig}/6, {cur_para}/6 | R1保留原/换={r1_orig}/6, {r1_para}/6 | 旧能力={old_cur}', flush=True)
    peak_mb = torch.cuda.max_memory_allocated() / 1e6 if DEV == 'cuda' else 0.0
    print(f'=== 完成 | 峰值显存={peak_mb:.0f} MB | 总耗时={time.perf_counter()-t0:.1f}s ===', flush=True)


if __name__ == '__main__':
    main()
