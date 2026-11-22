"""联想记忆海马体 SQLite 持久化层（与 assoc_memory.py 无缝对接，WAL 事务安全）。

支持：
- 事件原文、日期、来源、状态（active / superseded / deleted）、覆盖链（supersedes / superseded_by）持久落盘；
- 巩固强化后的图边权重（consolidated_edges）持久落盘；
- 重启后一键加载恢复 EpisodicStore + SparseAssociativeGraph，且保留所有巩固边权重。
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Optional, Tuple

from assoc_memory import EpisodicEvent, EpisodicStore, SparseAssociativeGraph


class SQLiteAssocPersistence:
    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        with self.conn:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS episodic_events (
                    event_id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    day INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    supersedes TEXT,
                    superseded_by TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS consolidated_edges (
                    src_id TEXT NOT NULL,
                    dst_id TEXT NOT NULL,
                    weight REAL NOT NULL DEFAULT 0.0,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (src_id, dst_id)
                );
                """
            )

    def save_store_and_graph(self, store: EpisodicStore, graph: SparseAssociativeGraph) -> None:
        """将当前事件库与巩固强化边原子写入 SQLite。"""
        now = time.time()
        with self.conn:
            for ev in store.all_events(include_deleted=True):
                self.conn.execute(
                    """
                    INSERT INTO episodic_events
                        (event_id, text, day, source, status, supersedes, superseded_by, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(event_id) DO UPDATE SET
                        text=excluded.text,
                        day=excluded.day,
                        source=excluded.source,
                        status=excluded.status,
                        supersedes=excluded.supersedes,
                        superseded_by=excluded.superseded_by,
                        updated_at=excluded.updated_at
                    """,
                    (
                        ev.event_id,
                        ev.text,
                        ev.day,
                        ev.source,
                        ev.status,
                        ev.supersedes,
                        ev.superseded_by,
                        now,
                        now,
                    ),
                )
            for src_id, nbrs in graph.adj.items():
                for dst_id, edge in nbrs.items():
                    if edge.consolidated_weight > 0.0:
                        self.conn.execute(
                            """
                            INSERT INTO consolidated_edges (src_id, dst_id, weight, updated_at)
                            VALUES (?, ?, ?, ?)
                            ON CONFLICT(src_id, dst_id) DO UPDATE SET
                                weight=excluded.weight,
                                updated_at=excluded.updated_at
                            """,
                            (src_id, dst_id, float(edge.consolidated_weight), now),
                        )

    def load_store_and_graph(
        self,
        encoder,
        max_neighbors: int = 16,
    ) -> Tuple[EpisodicStore, SparseAssociativeGraph]:
        """从 SQLite 恢复完整 EpisodicStore 与 SparseAssociativeGraph（含巩固边权重）。"""
        store = EpisodicStore(encoder)
        rows = self.conn.execute(
            """
            SELECT event_id, text, day, source, status, supersedes, superseded_by
            FROM episodic_events
            ORDER BY rowid ASC
            """
        ).fetchall()
        for eid, text, day, source, status, supersedes, superseded_by in rows:
            ev = store.add_event(
                text=text,
                day=day,
                source=source,
                event_id=eid,
                status=status,
                supersedes=None,  # 直接按库中持久状态恢复，避免重复覆写
            )
            ev.supersedes = supersedes
            ev.superseded_by = superseded_by

        graph = SparseAssociativeGraph(store, max_neighbors=max_neighbors)
        edge_rows = self.conn.execute("SELECT src_id, dst_id, weight FROM consolidated_edges").fetchall()
        seen_pairs = set()
        for src_id, dst_id, weight in edge_rows:
            pair = tuple(sorted((src_id, dst_id)))
            if pair in seen_pairs or weight <= 0.0:
                continue
            if (
                src_id in store
                and dst_id in store
                and store.get(src_id).status == "active"
                and store.get(dst_id).status == "active"
            ):
                seen_pairs.add(pair)
                graph.reinforce_path([src_id, dst_id], delta=float(weight))
        return store, graph

    def close(self) -> None:
        self.conn.close()
