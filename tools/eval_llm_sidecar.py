"""冻结 Qwen2.5-0.5B-Instruct + RMP 旁路记忆池 + 神经海马体槽表：实测持续学习、抗遗忘与跨会话召回。

用法：
  python tools/eval_llm_sidecar.py [--model Qwen/Qwen2.5-0.5B-Instruct] [--steps 12]
"""
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from llm_sidecar import LLMSidecarWrapper, SidecarLearner

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

# 背景旧知识探针（通用中英文 + 代码，与训练背景不重叠）
BG_TEXTS = [
    "Photosynthesis is the process by which green plants convert sunlight, water, and carbon dioxide into glucose and oxygen.",
    "The water cycle describes how water evaporates from the surface of the earth, rises into the atmosphere, cools and condenses into rain or snow.",
    "Python is a high-level programming language known for its clear syntax and strong ecosystem in scientific computing and web development.",
    "地球绕太阳公转一周大约需要365.25天，这就是公历每四年设置一个闰年的原因。",
    "TCP协议通过三次握手建立可靠连接，并在数据包丢失时自动重传，以确保传输完整性。",
    "在机器学习中，过拟合指模型在训练集上表现很好，但在未见过的测试数据上泛化能力下降。",
]
OLD_PROBES = {
    'en_general': "A binary search tree stores keys in sorted order so lookup, insertion, and deletion run in logarithmic time on average.",
    'zh_general': "光合作用主要发生在植物叶片的叶绿体中，它将太阳能转化为化学能并释放出氧气。",
    'code': "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",
}

# 全新领域/技能（自定义协议语法与虚构概念，Qwen2.5-0.5B 预训练从未见过）
NEW_TRAIN = [
    "[ZOR-PROTOCOL v3] @NODE=alpha7 :: ACTION=FLUSH_RING >> STATUS=ACK_99 | CHECKSUM=QX41\n"
    "Rule: When ZOR-PROTOCOL receives FLUSH_RING on alpha7, reply with ACK_99 and rotate checksum QX41 to QX42.",
    "[ZOR-PROTOCOL v3] @NODE=beta2 :: ACTION=SYNC_PULSE >> STATUS=ACK_88 | CHECKSUM=QX77\n"
    "Rule: When ZOR-PROTOCOL receives SYNC_PULSE on beta2, reply with ACK_88 and rotate checksum QX77 to QX78.",
    "[ZOR-PROTOCOL v3] @NODE=gamma9 :: ACTION=SEAL_EPOCH >> STATUS=ACK_77 | CHECKSUM=QX12\n"
    "Rule: When ZOR-PROTOCOL receives SEAL_EPOCH on gamma9, reply with ACK_77 and rotate checksum QX12 to QX13.",
    "[ZOR-PROTOCOL v3] @NODE=delta4 :: ACTION=DRAIN_PIPE >> STATUS=ACK_66 | CHECKSUM=QX55\n"
    "Rule: When ZOR-PROTOCOL receives DRAIN_PIPE on delta4, reply with ACK_66 and rotate checksum QX55 to QX56.",
]
NEW_TEST = (
    "[ZOR-PROTOCOL v3] @NODE=omega5 :: ACTION=FLUSH_RING >> STATUS=ACK_99 | CHECKSUM=QX91\n"
    "Rule: When ZOR-PROTOCOL receives FLUSH_RING on omega5, reply with ACK_99 and rotate checksum QX91 to QX92."
)


@torch.no_grad()
def eval_nll(wrapper, tok, text):
    wrapper.eval()
    wrapper.sidecar.reset_session()
    ids = tok(text, return_tensors='pt').input_ids.to(DEV)
    lg = wrapper(ids).logits
    return float(F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1)).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen2.5-0.5B-Instruct')
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--smf_k', type=int, default=8)
    ap.add_argument('--sleep_steps', type=int, default=20)
    a = ap.parse_args()

    t0 = time.perf_counter()
    print(f'Loading {a.model} on {DEV}...', flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    lm = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16 if DEV == 'cuda' else torch.float32).to(DEV)
    base_old = {k: eval_nll_raw(lm, tok, v) for k, v in OLD_PROBES.items()}
    base_new = eval_nll_raw(lm, tok, NEW_TEST)

    w = LLMSidecarWrapper(
        lm, attach_layer=-4,
        sidecar_cfg=dict(N=48, k=4, G=4, D=128, pd=64, pm=16, phidden=128, ctx_decay=(0.5, 0.85, 0.95, 0.99)),
        hippo_cfg=dict(n_slots=2048, d_key=64, d_val=64),
    )
    w.sidecar.to(DEV); w.hippo.to(DEV)
    sc_params = sum(p.numel() for p in w.sidecar.parameters()) + sum(p.numel() for p in w.hippo.parameters())
    lm_params = sum(p.numel() for p in w.backbone.parameters())
    print(f'Backbone: {lm_params/1e6:.1f}M (frozen) | Sidecar+Hippo: {sc_params/1e6:.2f}M ({sc_params/lm_params:.2%})', flush=True)

    # 1. 验证初始恒等
    init_old = {k: eval_nll(w, tok, v) for k, v in OLD_PROBES.items()}
    init_new = eval_nll(w, tok, NEW_TEST)
    max_init_diff = max(abs(init_old[k] - base_old[k]) for k in OLD_PROBES)
    print(f'Init identity max diff: {max_init_diff:.6f} | base_new NLL: {base_new:.4f}', flush=True)

    # 2. 背景频率统计（SMF）
    L = SidecarLearner(w, lr=a.lr, smf_k=a.smf_k)
    bg_batches = [tok(t, return_tensors='pt').input_ids.to(DEV) for t in BG_TEXTS]
    L.profile_bg(bg_batches)

    # 3. 白天在线学习新协议
    for ep in range(a.epochs):
        for txt in NEW_TRAIN:
            w.sidecar.reset_session()
            ids = tok(txt, return_tensors='pt').input_ids.to(DEV)
            info = L.step_online(ids)
    w.reset_session()                                              # 跨会话重置：清 RMP 状态，保留海马体槽表
    on_old = {k: eval_nll(w, tok, v) for k, v in OLD_PROBES.items()}
    on_new = eval_nll(w, tok, NEW_TEST)
    on_rises = {k: round(on_old[k] - init_old[k], 5) for k in OLD_PROBES}
    print(f'After online + reset_session: new NLL {init_new:.4f} -> {on_new:.4f} (drop {init_new - on_new:+.4f}) | '
          f'old rises={on_rises} | hippo slots={int(w.hippo.used.sum())}', flush=True)

    # 4. 晚上睡眠巩固
    slp = L.sleep_consolidate(steps=a.sleep_steps, lr=3e-4, anchor_batches=bg_batches, anchor_w=2.0)
    w.reset_session()
    slp_old = {k: eval_nll(w, tok, v) for k, v in OLD_PROBES.items()}
    slp_new = eval_nll(w, tok, NEW_TEST)
    slp_rises = {k: round(slp_old[k] - init_old[k], 5) for k in OLD_PROBES}
    peak_mb = torch.cuda.max_memory_allocated() / 1e6 if DEV == 'cuda' else 0.0
    print(f'After sleep ({slp}): new NLL {init_new:.4f} -> {slp_new:.4f} (drop {init_new - slp_new:+.4f}) | '
          f'old rises={slp_rises} | peak VRAM={peak_mb:.0f} MB | elapsed={time.perf_counter()-t0:.1f}s', flush=True)

    # 5. 保存轻量记忆权重
    mem_path = ROOT / 'runs/qwen05b_sidecar_demo.pt'
    w.save_memory(mem_path)
    out = {
        'model': a.model, 'backbone_params': lm_params, 'sidecar_params': sc_params,
        'init_identity_diff': max_init_diff, 'new_before': init_new, 'new_online': on_new, 'new_sleep': slp_new,
        'old_before': init_old, 'old_rises_online': on_rises, 'old_rises_sleep': slp_rises,
        'hippo_slots': int(w.hippo.used.sum()), 'peak_vram_mb': round(peak_mb, 1),
    }
    p = ROOT / f'runs/{time.strftime("%Y%m%d-%H%M%S")}_llm-sidecar-eval.json'
    json.dump(out, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('→', p.name, '| memory saved:', mem_path.name)


@torch.no_grad()
def eval_nll_raw(lm, tok, text):
    lm.eval()
    ids = tok(text, return_tensors='pt').input_ids.to(DEV)
    lg = lm(ids).logits
    return float(F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1)).item())


if __name__ == '__main__':
    main()
