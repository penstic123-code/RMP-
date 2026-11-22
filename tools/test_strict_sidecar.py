"""严格防作弊压力测试 v2（Qwen2.5-0.5B + RMPSidecar + NeuralHippocampus）：
1. 测试 A（纯 RMP 权重盲考，零海马体文本提示、零上下文）：
   - 8 条虚构规则：分别测 原问句（8 题）、未见过的新问法（每条 3 个换问法共 24 题）、以及关闭旁路对照（bypass=True）；
   - 旧能力保旧改用多段独立探针 + FP32 精确 NLL。
2. 测试 B（海马体长对话抗干扰，修复 bypass 缓存后）：
   - 12 条多维度用户事实 + 16 条长干扰废话 → /new 后跨会话抽考 6 题；
   - 分别报告 Top-3 召回率（recall@3）与端到端回答准确率（ans_acc）。
"""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from llm_sidecar import LLMSidecarWrapper, SidecarLearner

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_SYS = "You are Qwen, a helpful AI assistant. Answer concisely and directly."

QA_RULES = [
    ("What is the secret code for Project Nebula?", "The secret code for Project Nebula is ORION-94.", "ORION-94", [
        "Tell me Project Nebula's secret code.",
        "Which code unlocks Project Nebula?",
        "For Project Nebula, what is the secret code?",
    ]),
    ("Who is the captain of Starship Vesper?", "The captain of Starship Vesper is Commander Lyra Chen.", "Lyra Chen", [
        "Who commands Starship Vesper?",
        "Name the captain leading Starship Vesper.",
        "Which officer is in charge of Starship Vesper?",
    ]),
    ("What color is the crystal of Valoria?", "The crystal of Valoria is emerald violet.", "emerald violet", [
        "Describe the color of Valoria's crystal.",
        "What hue does the crystal of Valoria have?",
        "Which color is associated with the crystal of Valoria?",
    ]),
    ("What port does the Zorbus daemon listen on?", "The Zorbus daemon listens on port 7741.", "7741", [
        "Which network port is used by the Zorbus daemon?",
        "On what port should I connect to the Zorbus daemon?",
        "Give me the listening port number of Zorbus daemon.",
    ]),
    ("What is the capital city of Astralonia?", "The capital city of Astralonia is Solisgrad.", "Solisgrad", [
        "Name the capital of Astralonia.",
        "Which city serves as Astralonia's capital?",
        "What is Astralonia's capital city called?",
    ]),
    ("Which element powers the Kestrel engine?", "The Kestrel engine is powered by liquid xenon.", "xenon", [
        "What fuel element runs the Kestrel engine?",
        "Which chemical element drives the Kestrel engine?",
        "What does the Kestrel engine use for power?",
    ]),
    ("What is the emergency override phrase for Vault 12?", "The emergency override phrase for Vault 12 is SILENT-COMET.", "SILENT-COMET", [
        "Give the emergency override phrase to open Vault 12.",
        "What phrase triggers the Vault 12 emergency override?",
        "Tell me Vault 12's emergency override phrase.",
    ]),
    ("How many moons orbit planet Thalassa?", "Planet Thalassa has exactly five moons.", "five", [
        "What is the number of moons around planet Thalassa?",
        "Count the moons orbiting Thalassa.",
        "How many natural satellites does planet Thalassa have?",
    ]),
]

# 扩大独立旧能力探针（每类多段，统一用 FP32 算 NLL）
OLD_PROBES = {
    'en': [
        "A binary search tree stores keys in sorted order so lookup, insertion, and deletion run in logarithmic time on average.",
        "Photosynthesis converts light energy into chemical energy inside chloroplasts, producing glucose and releasing oxygen.",
        "The Industrial Revolution began in Britain during the late eighteenth century and transformed agriculture, textiles, and transport.",
        "Plate tectonics explains how Earth's rigid outer shell is divided into plates that slowly drift over the mantle.",
    ],
    'zh': [
        "光合作用主要发生在植物叶片的叶绿体中，它将太阳能转化为化学能并释放出氧气。",
        "丝绸之路是古代连接中国与中亚、西亚乃至地中海沿岸的重要商贸与文化交流通道。",
        "相对论由爱因斯坦提出，深刻改变了人类对时间、空间与引力本质的认识。",
        "在操作系统中，虚拟内存通过页表映射将进程地址空间与物理内存解耦，提高内存利用率。",
    ],
    'code': [
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n    return quicksort([x for x in arr[1:] if x < pivot]) + [pivot] + quicksort([x for x in arr[1:] if x >= pivot])\n",
        "def binary_search(nums, target):\n    lo, hi = 0, len(nums) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if nums[mid] == target:\n            return mid\n        if nums[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n",
        "class LRUCache:\n    def __init__(self, cap):\n        self.cap = cap\n        self.data = {}\n",
    ],
    'math': [
        "The derivative of x^3 + 2x^2 - 5x + 7 with respect to x is 3x^2 + 4x - 5.",
        "By the Pythagorean theorem, a right triangle with legs of length 3 and 4 has a hypotenuse of length 5.",
        "The integral of 2x from 0 to 3 equals x^2 evaluated from 0 to 3, which is 9.",
        "For any invertible square matrix A, the product of A and its inverse A^-1 equals the identity matrix I.",
    ],
}
BG_TEXTS = [
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "The Pacific Ocean is the largest and deepest ocean on Earth.",
    "地球是太阳系中由内及外的第三颗行星，也是目前已知唯一孕育生命的天体。",
    "HTTP status code 404 indicates that the requested resource could not be found on the server.",
    "In linear algebra, an eigenvector of a matrix only changes by a scalar factor under transformation.",
    "在深度学习中，梯度裁剪常用于防止反向传播过程中的梯度爆炸问题。",
]

USER_FACTS = [
    "My name is Alex Mercer.",
    "My pet cat is named Mirafe and she is three years old.",
    "I live in Sydney, Australia near Darling Harbour.",
    "I am allergic to peanuts and shellfish.",
    "My favourite programming language is Rust.",
    "My best friend from university is named Daniel Park.",
    "I drive a silver Subaru Outback.",
    "My coffee order is always an iced oat flat white.",
    "I am currently building a neural memory architecture called DSPN.",
    "My birthday is on October 14.",
    "I play badminton every Wednesday evening.",
    "My favourite movie is Interstellar.",
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
    ("Which city do I live in?", "Sydney", "Sydney"),
    ("What car do I drive?", "Subaru", "Subaru"),
    ("When is my birthday?", "October 14", "October 14"),
    ("What sport do I play on Wednesday evenings?", "badminton", "badminton"),
]


@torch.no_grad()
def eval_nll_fp32(w, tok, texts):
    """在 FP32 精度下对多段独立探针求平均 token NLL。"""
    w.eval()
    tot_loss, tot_toks = 0.0, 0
    for txt in texts:
        w.sidecar.reset_session()
        ids = tok(txt, return_tensors='pt').input_ids.to(DEV)
        lg = w(ids).logits.float()
        n = ids.shape[1] - 1
        tot_loss += float(F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1), reduction='sum').item())
        tot_toks += n
    return tot_loss / max(tot_toks, 1)


@torch.no_grad()
def ask_no_context(w, tok, question, sys_msg=BASE_SYS, bypass=False):
    w.eval(); w.sidecar.reset_session()
    prev = w.bypass; w.bypass = bypass
    try:
        prompt = tok.apply_chat_template([{'role': 'system', 'content': sys_msg}, {'role': 'user', 'content': question}],
                                         tokenize=False, add_generation_prompt=True)
        inp = tok(prompt, return_tensors='pt').to(DEV)
        out = w.backbone.generate(**inp, max_new_tokens=40, do_sample=False)
        return tok.decode(out[0, inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
    finally:
        w.bypass = prev


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
    t0 = time.perf_counter()
    model_name = 'Qwen/Qwen2.5-0.5B-Instruct'
    print(f'=== 加载 {model_name} ({DEV}) ===', flush=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    lm = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16 if DEV == 'cuda' else torch.float32).to(DEV)
    w = LLMSidecarWrapper(
        lm, attach_layer=-4,
        sidecar_cfg=dict(N=48, k=4, G=4, D=128, pd=64, pm=16, phidden=128, ctx_decay=(0.5, 0.85, 0.95, 0.99)),
        hippo_cfg=dict(n_slots=2048, d_key=64, d_val=64),
    )
    w.sidecar.to(DEV); w.hippo.to(DEV)

    # ---------- 测试 A：纯 RMP 权重 + 原问句 / 换问法泛化 / 关闭旁路对照 / FP32 扩大旧探针 ----------
    print('\n=== [测试 A] 纯 RMP 权重盲考（原问句 8 题 + 未训练换问法 24 题 + bypass 对照） ===', flush=True)
    old0 = {k: eval_nll_fp32(w, tok, v) for k, v in OLD_PROBES.items()}
    L = SidecarLearner(w, lr=2e-3, smf_k=8)
    bg_batches = [tok(t, return_tensors='pt').input_ids.to(DEV) for t in BG_TEXTS]
    L.profile_bg(bg_batches)
    for q, ans, _, _ in QA_RULES:
        qp = tok.apply_chat_template([{'role': 'system', 'content': BASE_SYS}, {'role': 'user', 'content': q}],
                                     tokenize=False, add_generation_prompt=True)
        qf = tok.apply_chat_template([{'role': 'system', 'content': BASE_SYS}, {'role': 'user', 'content': q},
                                      {'role': 'assistant', 'content': ans}], tokenize=False)
        L.stage_qa(tok(qf, return_tensors='pt').input_ids, tok(qp, return_tensors='pt').input_ids.shape[1])

    L.sleep_consolidate(steps=60, lr=1e-3, anchor_batches=bg_batches, anchor_w=1.5)
    w.reset_session()

    orig_on, orig_off, para_on, para_off = 0, 0, 0, 0
    for q, _, key, paras in QA_RULES:
        r_on = ask_no_context(w, tok, q, bypass=False)
        r_off = ask_no_context(w, tok, q, bypass=True)
        orig_on += int(key.lower() in r_on.lower())
        orig_off += int(key.lower() in r_off.lower())
        p_ok = 0
        for pq in paras:
            pr_on = ask_no_context(w, tok, pq, bypass=False)
            pr_off = ask_no_context(w, tok, pq, bypass=True)
            ok = key.lower() in pr_on.lower()
            para_on += int(ok); p_ok += int(ok)
            para_off += int(key.lower() in pr_off.lower())
        print(f'  [{key:14s}] 原问句={int(key.lower() in r_on.lower())}/1 | 换问法={p_ok}/3 | 示例答: {ask_no_context(w, tok, paras[0], bypass=False)[:70]}', flush=True)
    old1 = {k: eval_nll_fp32(w, tok, v) for k, v in OLD_PROBES.items()}
    rises = {k: round(old1[k] - old0[k], 5) for k in OLD_PROBES}
    print(f'原问句: 旁路开 {orig_on}/8 vs 旁路关(bypass) {orig_off}/8', flush=True)
    print(f'未训练换问法: 旁路开 {para_on}/24 vs 旁路关(bypass) {para_off}/24', flush=True)
    print(f'FP32 扩大旧探针 NLL 变化: {rises}', flush=True)

    # ---------- 测试 B：修好 bypass 缓存后重测海马体抗干扰（拆开报告 recall@3 与 ans_acc） ----------
    print('\n=== [测试 B] 海马体长对话抗干扰（12 事实 + 16 干扰 = 28 槽，拆开报告 recall@3 与 ans_acc） ===', flush=True)
    mixed = []
    for i in range(max(len(USER_FACTS), len(DISTRACTORS))):
        if i < len(USER_FACTS):
            mixed.append(USER_FACTS[i])
        if i < len(DISTRACTORS):
            mixed.append(DISTRACTORS[i])
    for txt in mixed:
        w.hippo.write_fact(encode_frozen(w, tok, txt), txt)
    w.reset_session()
    print(f'海马体已存槽位: {int(w.hippo.used.sum())} 条', flush=True)
    rec_hits, ans_hits = 0, 0
    for q, ans_key, fact_key in FACT_TESTS:
        rec = w.hippo.recall_facts(encode_frozen(w, tok, q), top_k=3, min_cos=0.10, cross_session_only=True)
        in_top3 = any(fact_key.lower() in r.lower() for r in rec)
        rec_hits += int(in_top3)
        facts_str = '\n'.join(f'- The user previously said: "{f}"' for f in rec)
        sys_m = f"{BASE_SYS}\n\nFacts about the USER (not you):\n{facts_str}" if rec else BASE_SYS
        rep = ask_no_context(w, tok, q, sys_msg=sys_m)
        ok = ans_key.lower() in rep.lower()
        ans_hits += int(ok)
        print(f'  [Top3={"Y" if in_top3 else "N"} | Ans={"OK" if ok else "MISS"}] Q: {q} -> A: {rep[:80]}', flush=True)
    print(f'海马体结果: recall@3 = {rec_hits}/{len(FACT_TESTS)} | ans_acc = {ans_hits}/{len(FACT_TESTS)} | 总耗时 {time.perf_counter()-t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
