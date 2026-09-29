"""SQLite 持久化。

四张表：
  sessions          —— 一次学习会话的汇总（关闭时写入）
  events            —— 值得记住的瞬间（延误太久的分心、里程碑、离席…）
  behavior_samples  —— 每 N 秒一条的行为快照，用于画曲线
  milestones        —— 达成的专注里程碑

设计约束：
  * 单连接 + 线程锁。Qt 定时器和将来可能的导出线程都可能碰它，
    sqlite 对象默认不能跨线程；
  * WAL 模式，避免写的时候读被阻塞；
  * **写失败绝不能让主循环崩掉**。所有公开方法都吞异常并记日志 ——
    磁盘满 / 文件锁 / 权限问题都不该让学习监控挂掉。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at          REAL NOT NULL,
    ended_at            REAL,
    total_seconds       REAL DEFAULT 0,
    focused_seconds     REAL DEFAULT 0,
    distracted_seconds  REAL DEFAULT 0,
    away_seconds        REAL DEFAULT 0,
    uncertain_seconds   REAL DEFAULT 0,
    focus_score         REAL DEFAULT 0,
    transitions         INTEGER DEFAULT 0,
    note                TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER,
    ts          REAL NOT NULL,
    kind        TEXT NOT NULL,
    behavior    TEXT DEFAULT '',
    state       TEXT DEFAULT '',
    duration    REAL DEFAULT 0,
    detail      TEXT DEFAULT '',
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS behavior_samples (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER,
    ts          REAL NOT NULL,
    behavior    TEXT DEFAULT '',
    state       TEXT DEFAULT '',
    confidence  REAL DEFAULT 0,
    scores      TEXT DEFAULT '',
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS milestones (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER,
    ts          REAL NOT NULL,
    minutes     INTEGER NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_events_session   ON events(session_id, ts);
CREATE INDEX IF NOT EXISTS idx_samples_session  ON behavior_samples(session_id, ts);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at);
"""


@dataclass
class EventRecord:
    id: int
    ts: float
    kind: str
    behavior: str
    state: str
    duration: float
    detail: str

    @property
    def local_time(self) -> str:
        return time.strftime("%H:%M:%S", time.localtime(self.ts))


class Database:
    """SQLite 封装（失败即降级，不影响主流程）。"""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.enabled = True
        self.last_error = ""

        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._session_id: int | None = None

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._connect()
        except Exception as exc:
            self.enabled = False
            self.last_error = str(exc)
            print(f"[Database] 不可用，降级为纯内存模式：{exc}")

    # ------------------------------------------------------------------

    def _connect(self) -> None:
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def _execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor | None:
        """所有写操作的唯一出口。异常一律吞掉并降级。"""
        if not self.enabled or self._conn is None:
            return None

        try:
            with self._lock:
                cursor = self._conn.execute(sql, params)
                self._conn.commit()
                return cursor
        except Exception as exc:
            self.last_error = str(exc)
            print(f"[Database] 写入失败（已忽略）：{exc}")
            return None

    def _query(self, sql: str, params: tuple = ()) -> list:
        if not self.enabled or self._conn is None:
            return []

        try:
            with self._lock:
                cursor = self._conn.execute(sql, params)
                return cursor.fetchall()
        except Exception as exc:
            self.last_error = str(exc)
            print(f"[Database] 查询失败：{exc}")
            return []

    # ------------------------------------------------------------------
    # 会话
    # ------------------------------------------------------------------

    def start_session(self, started_at: float | None = None) -> int | None:
        started_at = time.time() if started_at is None else started_at
        cursor = self._execute(
            "INSERT INTO sessions (started_at) VALUES (?)", (started_at,)
        )

        if cursor is None:
            return None

        self._session_id = cursor.lastrowid
        return self._session_id

    @property
    def session_id(self) -> int | None:
        return self._session_id

    def finish_session(self, stats) -> None:
        if self._session_id is None:
            return

        self._execute(
            """
            UPDATE sessions SET
                ended_at = ?, total_seconds = ?, focused_seconds = ?,
                distracted_seconds = ?, away_seconds = ?, uncertain_seconds = ?,
                focus_score = ?, transitions = ?
            WHERE id = ?
            """,
            (
                time.time(),
                float(getattr(stats, "total_seconds", 0.0)),
                float(getattr(stats, "focused_seconds", 0.0)),
                float(getattr(stats, "distracted_seconds", 0.0)),
                float(getattr(stats, "away_seconds", 0.0)),
                float(getattr(stats, "uncertain_seconds", 0.0)),
                float(getattr(stats, "focus_score", 0.0)),
                int(getattr(stats, "transitions", 0)),
                self._session_id,
            ),
        )

    # ------------------------------------------------------------------
    # 事件
    # ------------------------------------------------------------------

    def add_event(
        self,
        kind: str,
        behavior: str = "",
        state: str = "",
        duration: float = 0.0,
        detail: str = "",
        ts: float | None = None,
    ) -> None:
        self._execute(
            """
            INSERT INTO events (session_id, ts, kind, behavior, state, duration, detail)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self._session_id,
                time.time() if ts is None else ts,
                kind,
                behavior,
                state,
                float(duration),
                detail,
            ),
        )

    def recent_events(self, limit: int = 50) -> list:
        rows = self._query(
            "SELECT id, ts, kind, behavior, state, duration, detail "
            "FROM events ORDER BY ts DESC LIMIT ?",
            (int(limit),),
        )
        return [EventRecord(*row) for row in rows]

    def events_for_session(self, session_id: int) -> list:
        rows = self._query(
            "SELECT id, ts, kind, behavior, state, duration, detail "
            "FROM events WHERE session_id = ? ORDER BY ts ASC",
            (int(session_id),),
        )
        return [EventRecord(*row) for row in rows]

    # ------------------------------------------------------------------
    # 行为采样
    # ------------------------------------------------------------------

    def add_sample(
        self,
        behavior: str,
        state: str,
        confidence: float = 0.0,
        scores: dict | None = None,
        ts: float | None = None,
    ) -> None:
        payload = ""

        if scores:
            try:
                payload = json.dumps(
                    {str(k): round(float(v), 3) for k, v in scores.items()},
                    ensure_ascii=False,
                )
            except (TypeError, ValueError):
                payload = ""

        self._execute(
            """
            INSERT INTO behavior_samples
                (session_id, ts, behavior, state, confidence, scores)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                self._session_id,
                time.time() if ts is None else ts,
                behavior,
                state,
                float(confidence),
                payload,
            ),
        )

    def samples_for_session(self, session_id: int) -> list:
        return self._query(
            "SELECT ts, behavior, state, confidence, scores "
            "FROM behavior_samples WHERE session_id = ? ORDER BY ts ASC",
            (int(session_id),),
        )

    # ------------------------------------------------------------------
    # 里程碑
    # ------------------------------------------------------------------

    def add_milestone(self, minutes: int, ts: float | None = None) -> None:
        self._execute(
            "INSERT INTO milestones (session_id, ts, minutes) VALUES (?, ?, ?)",
            (self._session_id, time.time() if ts is None else ts, int(minutes)),
        )

    # ------------------------------------------------------------------
    # 汇总查询（Dashboard 用）
    # ------------------------------------------------------------------

    def recent_sessions(self, limit: int = 20) -> list:
        return self._query(
            """
            SELECT id, started_at, ended_at, total_seconds, focused_seconds,
                   distracted_seconds, away_seconds, focus_score, transitions
            FROM sessions WHERE ended_at IS NOT NULL
            ORDER BY started_at DESC LIMIT ?
            """,
            (int(limit),),
        )

    def totals(self) -> dict:
        rows = self._query(
            """
            SELECT COUNT(*), COALESCE(SUM(total_seconds),0),
                   COALESCE(SUM(focused_seconds),0),
                   COALESCE(AVG(focus_score),0)
            FROM sessions WHERE ended_at IS NOT NULL
            """
        )

        if not rows:
            return {"sessions": 0, "total_seconds": 0.0,
                    "focused_seconds": 0.0, "avg_score": 0.0}

        count, total, focused, avg_score = rows[0]
        return {
            "sessions": int(count),
            "total_seconds": float(total),
            "focused_seconds": float(focused),
            "avg_score": float(avg_score),
        }

    def daily_focus(self, days: int = 7) -> list:
        """最近 N 天每天的有效专注秒数（时间戳按本地日聚合）。"""
        return self._query(
            """
            SELECT date(started_at, 'unixepoch', 'localtime') AS day,
                   COALESCE(SUM(focused_seconds), 0),
                   COALESCE(AVG(focus_score), 0)
            FROM sessions
            WHERE ended_at IS NOT NULL
              AND started_at >= ?
            GROUP BY day ORDER BY day ASC
            """,
            (time.time() - days * 86400,),
        )

    def behavior_distribution(self, limit_sessions: int = 20) -> dict:
        """最近若干会话的行为时长分布（从样本表统计）。"""
        rows = self._query(
            """
            SELECT behavior, COUNT(*) FROM behavior_samples
            WHERE session_id IN (
                SELECT id FROM sessions ORDER BY started_at DESC LIMIT ?
            )
            GROUP BY behavior ORDER BY COUNT(*) DESC
            """,
            (int(limit_sessions),),
        )
        return {row[0]: int(row[1]) for row in rows if row[0]}

    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._conn is not None:
            try:
                with self._lock:
                    self._conn.commit()
                    self._conn.close()
            except Exception:
                pass
            self._conn = None
