"""长期可用交互式命令行对话：冻结开源 LLM + 独立出口 RMP 旁路 + SQLite 长期记忆库（换模型自动重建索引）。

特性（面向 1 个月–1 年+ 长期使用）：
  - SQLite（默认 runs/memory.db）自动保存所有有效事实原文、事实键与已确认教学规则，完全独立于具体 LLM 底座；
  - 换用新模型（--model）时自动检测并从 SQLite 重建海马体向量索引；
  - 默认保旧锚定集涵盖英文、中文、Python 代码与数学，防长期代码漂移；
  - 支持人工兜底管理命令：/list  /edit <id> <新内容>  /del <id>  /fact 键=内容  /teach 问题 => 答案  /confirm  /sleep  /new  /stats  /quit
"""
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llm_sidecar import LLMSidecarWrapper, SidecarLearner, SIDECAR_PRESETS
from memory_store import MemoryStore

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
BG_SEED = [
    "Photosynthesis converts sunlight into chemical energy in plants.",
    "地球绕太阳公转一周大约需要365天。",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    p = arr[0]\n    return quicksort([x for x in arr[1:] if x < p]) + [p] + quicksort([x for x in arr[1:] if x >= p])\n",
    "def binary_search(nums, target):\n    lo, hi = 0, len(nums) - 1\n    while lo <= hi:\n        m = (lo + hi) // 2\n        if nums[m] == target: return m\n        lo, hi = (m + 1, hi) if nums[m] < target else (lo, m - 1)\n    return -1\n",
    "The derivative of x^3 + 2x^2 - 5x + 7 with respect to x is 3x^2 + 4x - 5.",
]
BASE_SYS = "You are Qwen, a helpful AI assistant. Clearly distinguish between yourself (the AI assistant) and the user."


@torch.no_grad()
def encode_utterance(w: LLMSidecarWrapper, tok, text):
    ids = tok(text, return_tensors='pt').input_ids.to(DEV)
    w.eval(); w.bypass = True
    try:
        w(ids)
    finally:
        w.bypass = False
    return w._last_h


def stage_confirmed_qa(L: SidecarLearner, tok, q, ans, to_confirmed=False):
    qp = tok.apply_chat_template([{'role': 'system', 'content': BASE_SYS}, {'role': 'user', 'content': q}],
                                 tokenize=False, add_generation_prompt=True)
    qf = tok.apply_chat_template([{'role': 'system', 'content': BASE_SYS}, {'role': 'user', 'content': q},
                                  {'role': 'assistant', 'content': ans}], tokenize=False)
    item = (1.0, tok(qf, return_tensors='pt').input_ids.cpu(), int(tok(qp, return_tensors='pt').input_ids.shape[1]))
    if to_confirmed:
        L.confirmed_qa.append(item)
    else:
        L.staged.append(item)


def rebuild_hippo_from_db(store: MemoryStore, w: LLMSidecarWrapper, tok):
    """从 SQLite 全量重建海马体活跃向量索引（换底座模型或 /edit /del 后秒级完成）。"""
    h = w.hippo
    h.used.zero_(); h.superseded.zero_(); h.hits.zero_(); h.priority.zero_(); h.ts.zero_()
    h.slot_texts = [''] * h.n_slots; h.slot_keys = [''] * h.n_slots; h.clock = 0
    rows = store.list_facts(active_only=True)
    for _, fk, txt, sid, _ in rows:
        h.cur_session = sid
        h.write_fact(encode_utterance(w, tok, txt), txt, fact_key=fk)
    if rows:
        h.cur_session = max(r[3] for r in rows) + 1
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preset', default='small', choices=list(SIDECAR_PRESETS), help='small=0.5B+N48 | large=1.5B+N512 宽模块稳跑版')
    ap.add_argument('--model', default='', help='覆盖预设的底座模型名')
    ap.add_argument('--db', default=str(ROOT / 'runs/memory.db'), help='SQLite 长期记忆库路径（事实与教学规则原文）')
    ap.add_argument('--mem', default='', help='当前底座的 RMP 旁路权重路径（默认按预设区分）')
    ap.add_argument('--fresh', action='store_true', help='忽略已有权重（仅保留指定 db）')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--smf_k', type=int, default=8)
    ap.add_argument('--max_new', type=int, default=120)
    ap.add_argument('--script', default='', help='非交互自动测试脚本（用 || 分隔）')
    a = ap.parse_args()
    cfg = SIDECAR_PRESETS[a.preset]
    model_name = a.model or cfg['model']
    mem_default = ROOT / f'runs/sidecar_{a.preset}_memory.pt'
    mem_path = Path(a.mem) if a.mem else mem_default

    print(f'正在加载 [{a.preset}] {model_name} ({DEV})...', flush=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    lm = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16 if DEV == 'cuda' else torch.float32).to(DEV)
    w = LLMSidecarWrapper(lm, attach_layer=-4, sidecar_cfg=cfg['sidecar'], hippo_cfg=cfg['hippo'])
    w.sidecar.to(DEV); w.hippo.to(DEV)
    L = SidecarLearner(w, lr=a.lr, smf_k=a.smf_k)
    bg_batches = [tok(t, return_tensors='pt').input_ids.to(DEV) for t in BG_SEED]
    L.profile_bg(bg_batches)

    store = MemoryStore(a.db)
    prev_model = store.get_meta('model_name')
    model_changed = bool(prev_model and prev_model != model_name)
    if not a.fresh and not model_changed and mem_path.exists():
        w.load_memory(mem_path, map_location=DEV, learner=L)
    # 始终以 SQLite 为事实与规则的单一真源：换模型自动重建向量索引
    n_facts = rebuild_hippo_from_db(store, w, tok)
    L.staged.clear(); L.confirmed_qa.clear()
    for _, q, ans, st in store.list_rules():
        stage_confirmed_qa(L, tok, q, ans, to_confirmed=(st == 'confirmed' and not model_changed))
    store.set_meta('model_name', model_name)
    if model_changed:
        print(f'[检测到换底座：{prev_model} → {model_name}，已从 SQLite 自动重建 {n_facts} 条海马体索引，{len(L.staged)} 条规则待 /sleep 重训]', flush=True)
    elif n_facts or L.confirmed_qa or L.staged:
        print(f'已从 SQLite 恢复：活跃事实 {n_facts} 条 | 已巩固规则 {len(L.confirmed_qa)} 条 | 待巩固 {len(L.staged)} 条', flush=True)

    history, last_qa = [], None
    script_lines = [s.strip() for s in a.script.split('||') if s.strip()] if a.script else None
    if not script_lines:
        print('就绪！命令：/list | /edit id 内容 | /del id | /fact 键=内容 | /teach 问题 => 答案 | /confirm | /sleep | /new | /stats | /quit\n')

    idx_s = 0
    while True:
        if script_lines is not None:
            if idx_s >= len(script_lines):
                break
            line = script_lines[idx_s]; idx_s += 1
            print(f'你: {line}', flush=True)
        else:
            try:
                line = input('你: ').strip()
            except (EOFError, KeyboardInterrupt):
                break
        if not line:
            continue
        if line == '/quit':
            break
        if line == '/new':
            history.clear(); last_qa = None; w.reset_session()
            print(f'[新会话 #{w.hippo.cur_session}：对话清空，SQLite 保留 {len(store.list_facts())} 条活跃事实]\n', flush=True)
            continue
        if line == '/list':
            fs = store.list_facts(); rs = store.list_rules()
            print('[活跃事实]', flush=True)
            for fid, fk, txt, sid, _ in fs:
                kstr = f'[{fk}] ' if fk else ''
                print(f'  #{fid} {kstr}{txt}', flush=True)
            if rs:
                print('[教学规则]', flush=True)
                for rid, q, ans, st in rs:
                    print(f'  R#{rid} ({st}) {q} => {ans}', flush=True)
            print('', flush=True); continue
        if line.startswith('/edit '):
            parts = line[6:].strip().split(maxsplit=1)
            if len(parts) < 2 or not parts[0].isdigit():
                print('[用法] /edit <事实id> <新内容>\n', flush=True); continue
            ok = store.edit_fact(int(parts[0]), parts[1])
            if ok:
                rebuild_hippo_from_db(store, w, tok); w.save_memory(mem_path, learner=L)
                print(f'[已修改事实 #{parts[0]} 并重建索引]\n', flush=True)
            else:
                print(f'[未找到活跃事实 #{parts[0]}]\n', flush=True)
            continue
        if line.startswith('/del '):
            fid = line[5:].strip()
            if not fid.isdigit():
                print('[用法] /del <事实id>\n', flush=True); continue
            ok = store.delete_fact(int(fid))
            if ok:
                rebuild_hippo_from_db(store, w, tok); w.save_memory(mem_path, learner=L)
                print(f'[已删除事实 #{fid} 并重建索引]\n', flush=True)
            else:
                print(f'[未找到活跃事实 #{fid}]\n', flush=True)
            continue
        if line.startswith('/fact '):
            body = line[6:].strip()
            fk, txt = (body.split('=', 1) if '=' in body else ('', body))
            store.add_fact(txt.strip(), fact_key=fk.strip(), session_id=w.hippo.cur_session)
            rebuild_hippo_from_db(store, w, tok); w.save_memory(mem_path, learner=L)
            print(f'[已写入/覆盖事实 key="{fk.strip()}" 到 SQLite 并更新索引]\n', flush=True)
            continue
        if line.startswith('/teach '):
            body = line[7:].strip()
            if '=>' not in body:
                print('[用法] /teach 问题 => 标准答案\n', flush=True); continue
            q, ans = [s.strip() for s in body.split('=>', 1)]
            store.add_rule(q, ans, status='staged')
            stage_confirmed_qa(L, tok, q, ans)
            w.save_memory(mem_path, learner=L)
            print(f'[已写入 SQLite 待巩固规则（当前 {len(L.staged)} 条）]\n', flush=True)
            continue
        if line == '/confirm':
            if last_qa is None:
                print('[没有可确认的上一轮问答]\n', flush=True); continue
            store.add_rule(*last_qa, status='staged')
            stage_confirmed_qa(L, tok, *last_qa)
            w.save_memory(mem_path, learner=L)
            print(f'[已确认上一轮回答并写入 SQLite（待巩固 {len(L.staged)} 条）]\n', flush=True)
            continue
        if line == '/sleep':
            res = L.sleep_consolidate(steps=45, lr=1.2e-3, anchor_batches=bg_batches, anchor_w=2.0)
            if res.get('committed'):
                store.mark_staged_confirmed()
                w.save_memory(mem_path, learner=L)
            print(f'[睡眠巩固] {res}\n', flush=True)
            continue
        if line == '/stats':
            vram = torch.cuda.memory_allocated() / 1e6 if DEV == 'cuda' else 0.0
            print(f'[统计] 底座={model_name} (preset={a.preset}, N={w.sidecar.N}) | SQLite 活跃事实 {len(store.list_facts())} 条 | '
                  f'待巩固 {len(L.staged)} / 已确认 {len(L.confirmed_qa)} | 显存 {vram:.0f} MB\n', flush=True)
            continue

        # 正常对话：召回 → 自动写入非问句事实到 SQLite（若自动触发改口覆盖同步更新 SQLite）→ 生成回复
        h_u = encode_utterance(w, tok, line)
        recalled = w.hippo.recall_facts(h_u, query_text=line, top_k=3, min_score=0.15, cross_session_only=False)
        before_sup = set(i for i in range(w.hippo.n_slots) if w.hippo.superseded[i])
        written = w.hippo.write_fact(h_u, line)
        if written:
            new_sup = [w.hippo.slot_texts[i] for i in range(w.hippo.n_slots) if w.hippo.superseded[i] and i not in before_sup]
            store.add_fact(line, session_id=w.hippo.cur_session, supersede_texts=new_sup)
            w.save_memory(mem_path, learner=L)

        sys_msg = BASE_SYS
        if recalled:
            facts_str = '\n'.join(f'- The user previously said: "{f}"' for f in recalled)
            sys_msg = (f"{BASE_SYS}\n\nFacts the USER previously told you about THEMSELVES "
                       f"(these belong to the USER, NOT to you Qwen):\n{facts_str}")
        msgs = [{'role': 'system', 'content': sys_msg}] + history + [{'role': 'user', 'content': line}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok(prompt, return_tensors='pt').to(DEV)
        w.eval(); w.sidecar.reset_session()
        with torch.no_grad():
            out = w.backbone.generate(**inp, max_new_tokens=a.max_new, do_sample=False)
        reply = tok.decode(out[0, inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
        history.append({'role': 'user', 'content': line})
        history.append({'role': 'assistant', 'content': reply})
        last_qa = (line, reply)

        rec_tag = f', recalled={len(recalled)}' if recalled else ''
        print(f'模型: {reply}')
        print(f'   [+db={written}{rec_tag}, staged={len(L.staged)}, confirmed={len(L.confirmed_qa)}]\n', flush=True)
    store.close()


if __name__ == '__main__':
    main()
