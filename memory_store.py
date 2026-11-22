"""SQLite 长期记忆库（面向 1 个月–1 年+ 长期可用，与具体 LLM 底座解耦）。

表结构：
- facts：用户事实原文、可选事实键、是否有效（改口覆盖或删除时置 active=0，保留历史审计）、时间戳；
- rules：用户通过 /teach 或 /confirm 确认的教学问答原文（status='staged' | 'confirmed'），换 LLM 底座后可一键重训睡眠；
- meta：记录上次使用的底座模型名，换底座时自动触发海马体向量索引重建。
"""
import sqlite3, time
from pathlib import Path


class MemoryStore:
    def __init__(self, db_path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.execute('PRAGMA journal_mode=WAL')               # WAL 事务安全，断电/崩溃不坏库
        self._init_schema()

    def _init_schema(self):
        with self.conn:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fact_key TEXT DEFAULT '',
                    text TEXT NOT NULL,
                    active INTEGER DEFAULT 1,
                    session_id INTEGER DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    status TEXT DEFAULT 'staged',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
            """)

    def get_meta(self, key, default=''):
        row = self.conn.execute('SELECT value FROM meta WHERE key=?', (key,)).fetchone()
        return row[0] if row else default

    def set_meta(self, key, value):
        with self.conn:
            self.conn.execute('INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)', (key, str(value)))

    def add_fact(self, text, fact_key='', session_id=0, supersede_texts=()):
        """原子写入新事实；若给定 fact_key 或匹配到需覆盖的旧文本，自动把旧记录置 active=0。"""
        now = time.time()
        fk = fact_key.strip().lower()
        with self.conn:
            if fk:
                self.conn.execute('UPDATE facts SET active=0, updated_at=? WHERE active=1 AND fact_key=?', (now, fk))
            for old_t in supersede_texts:
                self.conn.execute('UPDATE facts SET active=0, updated_at=? WHERE active=1 AND text=?', (now, old_t))
            cur = self.conn.execute(
                'INSERT INTO facts(fact_key, text, active, session_id, created_at, updated_at) VALUES (?, ?, 1, ?, ?, ?)',
                (fk, text.strip(), int(session_id), now, now),
            )
            return cur.lastrowid

    def list_facts(self, active_only=True):
        q = 'SELECT id, fact_key, text, session_id, updated_at FROM facts'
        q += ' WHERE active=1 ORDER BY id ASC' if active_only else ' ORDER BY id ASC'
        return self.conn.execute(q).fetchall()

    def edit_fact(self, fact_id, new_text):
        now = time.time()
        with self.conn:
            cur = self.conn.execute('UPDATE facts SET text=?, updated_at=? WHERE id=? AND active=1', (new_text.strip(), now, int(fact_id)))
            return cur.rowcount > 0

    def delete_fact(self, fact_id):
        now = time.time()
        with self.conn:
            cur = self.conn.execute('UPDATE facts SET active=0, updated_at=? WHERE id=? AND active=1', (now, int(fact_id)))
            return cur.rowcount > 0

    def add_rule(self, question, answer, status='staged'):
        now = time.time()
        with self.conn:
            cur = self.conn.execute(
                'INSERT INTO rules(question, answer, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)',
                (question.strip(), answer.strip(), status, now, now),
            )
            return cur.lastrowid

    def list_rules(self, status=None):
        if status:
            return self.conn.execute('SELECT id, question, answer, status FROM rules WHERE status=? ORDER BY id ASC', (status,)).fetchall()
        return self.conn.execute('SELECT id, question, answer, status FROM rules ORDER BY id ASC').fetchall()

    def mark_staged_confirmed(self):
        now = time.time()
        with self.conn:
            self.conn.execute("UPDATE rules SET status='confirmed', updated_at=? WHERE status='staged'", (now,))

    def delete_rule(self, rule_id):
        with self.conn:
            cur = self.conn.execute('DELETE FROM rules WHERE id=?', (int(rule_id),))
            return cur.rowcount > 0

    def close(self):
        self.conn.close()
