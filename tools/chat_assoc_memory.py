"""端到端可交互长期联想记忆助手 CLI（纯通用神经回想核 v2 + SQLite 持久化 + 冻结 Qwen2.5-1.5B）。

支持命令：
- 直接输入自然中文问题/闲聊：自动执行循环回想（沿图 + 跳图双路 + 拒答门控），并用冻结 1.5B LLM 生成自然回复；
- /add <文本>：即时写入一条新经历并动态更新稀疏联想图；
- /supersede <旧事件ID> <新文本>：按事件 ID 精确覆盖旧安排（旧事件置 superseded，0 旧值泄露）；
- /delete <事件ID>：按事件 ID 精确删除隐私记忆；
- /list：列出当前所有活跃事件；
- /confirm：将上一轮验证有用的多跳回想路径巩固硬化成稀疏图边并落盘 SQLite；
- /quit：保存并退出。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if not os.environ.get("HF_HOME") and (ROOT / ".pytest_cache" / "hf").exists():
    os.environ["HF_HOME"] = str(ROOT / ".pytest_cache" / "hf")

import torch

from assoc_memory import EpisodicEvent
from assoc_neural_v2 import (
    GeneralNeuralRecallNet,
    NeuralEpisodicStore,
    PureNeuralGraph,
    TokenLevelEncoder,
    train_neural_v2,
)
from assoc_store_sqlite import SQLiteAssocPersistence
from tools.eval_assoc_e1 import build_disjoint_train_world
from tools.eval_assoc_e4 import AssocLanguageInterface
from tools.eval_assoc_v2 import calibrate_v2_thresh, convert_store_to_neural
from tools.eval_assoc_v2_lora import LoRATokenEncoder, train_end_to_end_lora


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=str, default="runs/assoc_memory_chat.db")
    ap.add_argument("--ckpt", type=str, default="runs/assoc_v2_lora_ckpt.pt")
    ap.add_argument("--llm", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--no_llm", action="store_true", help="仅展示回想到的原文证据，不加载 LLM")
    ap.add_argument("--epochs", type=int, default=24)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[初始化] 正在加载 BGE 顶 2 层 LoRA + 纯通用 v2 回想核（dev={dev}）...", flush=True)
    enc = LoRATokenEncoder("BAAI/bge-small-zh-v1.5", device=dev, r=8)
    torch.manual_seed(42)
    net = GeneralNeuralRecallNet(d_model=32, n_heads=4, n_layers=2).to(dev)

    ckpt_path = ROOT / args.ckpt
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=dev)
        net.load_state_dict(ck["net"])
        for p, v in zip(enc.lora_params, ck["lora"]):
            p.data.copy_(v.to(dev))
        abs_t = float(ck["abstain_thresh"])
        print(f"[已加载检查点] {ckpt_path} (abstain_thresh={abs_t:.2f})", flush=True)
    else:
        tr_s0, _, tr_examples = build_disjoint_train_world(enc)
        tr_raw = [(e.event_id, e.day, e.source, e.text, e.status) for e in tr_s0.all_events(include_deleted=True)]
        train_end_to_end_lora(enc, net, tr_raw, tr_examples, epochs=args.epochs, seed=42)
        tr_store = convert_store_to_neural(tr_s0, enc)
        tr_graph = PureNeuralGraph(tr_store, max_neighbors=16)
        abs_t = calibrate_v2_thresh(net, tr_store, tr_graph, tr_examples, budget=16)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"net": net.state_dict(), "lora": [p.detach().cpu() for p in enc.lora_params], "abstain_thresh": abs_t},
            ckpt_path,
        )
        print(f"[训练并保存检查点] {ckpt_path} (abstain_thresh={abs_t:.2f})", flush=True)

    # 加载或新建 SQLite 事件记忆库
    db = SQLiteAssocPersistence(args.db)
    rows = db.conn.execute("SELECT event_id, text, day, source, status, supersedes, superseded_by FROM episodic_events ORDER BY rowid ASC").fetchall()
    store = NeuralEpisodicStore(enc)
    for eid, text, day, source, status, supersedes, superseded_by in rows:
        ev = store.add_event(text, day=day, source=source, event_id=eid, status=status)
        ev.supersedes = supersedes
        ev.superseded_by = superseded_by
    graph = PureNeuralGraph(store, max_neighbors=16)
    seen_pairs = set()
    for src_id, dst_id, w in db.conn.execute("SELECT src_id, dst_id, weight FROM consolidated_edges").fetchall():
        pair = tuple(sorted((src_id, dst_id)))
        if pair in seen_pairs or w <= 0:
            continue
        if (
            src_id in store
            and dst_id in store
            and store.get(src_id).status == "active"
            and store.get(dst_id).status == "active"
        ):
            seen_pairs.add(pair)
            graph.reinforce_path([src_id, dst_id], delta=float(w))

    lang = None if args.no_llm else AssocLanguageInterface(model_name=args.llm, device=dev)
    day_counter = max((e.day for e in store.all_events(include_deleted=True)), default=0)
    last_recalled: List[str] = []

    print(f"[就绪] 当前活跃事件数={len(store.active_events())} | abstain_thresh={abs_t:.2f}")
    print("命令：/add <内容> | /supersede <旧ID> <新内容> | /delete <ID> | /list | /confirm | /quit\n")

    def save_all():
        # 将 NeuralEpisodicStore 落盘至 SQLite
        import time as _t
        now = _t.time()
        with db.conn:
            for e in store.all_events(include_deleted=True):
                db.conn.execute(
                    """INSERT INTO episodic_events(event_id, text, day, source, status, supersedes, superseded_by, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(event_id) DO UPDATE SET
                           text=excluded.text, status=excluded.status,
                           supersedes=excluded.supersedes, superseded_by=excluded.superseded_by,
                           updated_at=excluded.updated_at""",
                    (e.event_id, e.text, e.day, e.source, e.status, e.supersedes, e.superseded_by, now, now),
                )
            for (u, v), w in graph.consolidated.items():
                db.conn.execute(
                    """INSERT INTO consolidated_edges(src_id, dst_id, weight, updated_at) VALUES (?, ?, ?, ?)
                       ON CONFLICT(src_id, dst_id) DO UPDATE SET weight=excluded.weight, updated_at=excluded.updated_at""",
                    (u, v, float(w), now),
                )

    while True:
        try:
            line = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line == "/quit":
            save_all()
            db.close()
            print("已保存并退出。")
            break
        if line == "/list":
            for e in store.active_events():
                print(f"  [{e.event_id} | Day {e.day}] {e.text}")
            continue
        if line.startswith("/add "):
            day_counter += 1
            ev = store.add_event(line[5:].strip(), day=day_counter, source="user_cli")
            graph.rebuild()
            save_all()
            print(f"  [已记住] {ev.event_id} (Day {ev.day})")
            continue
        if line.startswith("/supersede "):
            parts = line[11:].strip().split(" ", 1)
            if len(parts) < 2 or parts[0] not in store:
                print("  用法：/supersede <旧事件ID> <新文本>")
                continue
            day_counter += 1
            ev = store.supersede_event(parts[0], parts[1].strip(), day=day_counter, source="user_cli")
            graph.rebuild()
            save_all()
            print(f"  [已覆写] {parts[0]} -> {ev.event_id} (Day {ev.day})")
            continue
        if line.startswith("/delete "):
            eid = line[8:].strip()
            ok = store.delete_event(eid)
            graph.rebuild()
            save_all()
            print(f"  [删除{'成功' if ok else '失败'}] {eid}")
            continue
        if line == "/confirm":
            if len(last_recalled) >= 2:
                graph.reinforce_path(last_recalled, delta=0.8)
                save_all()
                print(f"  [已巩固路径] {' <-> '.join(last_recalled)}")
            else:
                print("  上一轮无多跳路径可巩固。")
            continue

        res = net.recall(store, graph, line, mode="hybrid", max_rounds=3, candidate_budget=16, abstain_thresh=abs_t)
        last_recalled = list(res["event_ids"])
        if res["abstained"] or not last_recalled:
            print(f"  [记忆系统] 主动停机/无相关个人记忆 (p_valid={res['p_valid']:.2f})")
            evs_for_llm = []
        else:
            print(f"  [记忆系统] 唤起 {last_recalled} (p_valid={res['p_valid']:.2f}, via_jump={res['used_jump']}):")
            evs_for_llm = [store.get(eid) for eid in last_recalled]
            for e in evs_for_llm:
                print(f"    - [{e.event_id} | Day {e.day}] {e.text}")

        if lang is not None:
            reply = lang.reply(line, evs_for_llm)
            print(f"助手> {reply}\n")


if __name__ == "__main__":
    main()
