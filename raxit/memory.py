"""Persistent memory and conversation history, backed by SQLite.

Two separate stores with different lifetimes:

* `messages` is the raw conversation transcript, replayed to the model so a
  conversation survives a process restart (Termux gets killed a lot).
* `facts` is long-term memory the agent curates itself via the `remember` and
  `recall` tools — things worth knowing next week, not next turn.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Iterator

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    session   TEXT NOT NULL,
    role      TEXT NOT NULL,
    content   TEXT NOT NULL,          -- JSON-encoded content blocks
    created   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session, id);

CREATE TABLE IF NOT EXISTS facts (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    key       TEXT NOT NULL UNIQUE,
    value     TEXT NOT NULL,
    tags      TEXT NOT NULL DEFAULT '',
    updated   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    kind      TEXT NOT NULL,
    detail    TEXT NOT NULL,
    created   REAL NOT NULL
);
"""


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


# --- conversation transcript -------------------------------------------------


def append_message(session: str, role: str, content: Any) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO messages (session, role, content, created) VALUES (?,?,?,?)",
            (session, role, json.dumps(content, default=_blockify), time.time()),
        )


def load_messages(session: str, limit: int = 200) -> list[dict[str, Any]]:
    """Return the tail of a session as Messages-API `messages` entries.

    The tail is taken by id and then re-ordered, so the newest `limit` turns
    are kept rather than the oldest.
    """
    with connect() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session=? ORDER BY id DESC LIMIT ?",
            (session, limit),
        ).fetchall()
    msgs = [{"role": r["role"], "content": json.loads(r["content"])} for r in reversed(rows)]
    return _trim_to_valid_start(msgs)


def clear_session(session: str) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM messages WHERE session=?", (session,))


def _trim_to_valid_start(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop leading turns until the history starts on a `user` message.

    A tail slice can begin mid-exchange — on an assistant turn, or on a user
    turn holding `tool_result` blocks whose matching `tool_use` was cut off.
    Either shape is a 400 from the API, so skip forward to the first plain
    user turn.
    """
    for i, m in enumerate(msgs):
        if m["role"] != "user":
            continue
        content = m["content"]
        if isinstance(content, str):
            return msgs[i:]
        if not any(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in content
        ):
            return msgs[i:]
    return []


def _blockify(obj: Any) -> Any:
    """Serialize SDK content-block objects (Pydantic models) to plain dicts."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(exclude_none=True)
    return str(obj)


# --- long-term facts ---------------------------------------------------------


def remember(key: str, value: str, tags: str = "") -> None:
    with connect() as conn:
        conn.execute(
            """INSERT INTO facts (key, value, tags, updated) VALUES (?,?,?,?)
               ON CONFLICT(key) DO UPDATE SET value=excluded.value,
                                              tags=excluded.tags,
                                              updated=excluded.updated""",
            (key, value, tags, time.time()),
        )


def recall(query: str = "", limit: int = 25) -> list[dict[str, str]]:
    like = f"%{query}%"
    with connect() as conn:
        if query:
            rows = conn.execute(
                """SELECT key, value, tags FROM facts
                   WHERE key LIKE ? OR value LIKE ? OR tags LIKE ?
                   ORDER BY updated DESC LIMIT ?""",
                (like, like, like, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT key, value, tags FROM facts ORDER BY updated DESC LIMIT ?",
                (limit,),
            ).fetchall()
    return [dict(r) for r in rows]


def forget(key: str) -> bool:
    with connect() as conn:
        cur = conn.execute("DELETE FROM facts WHERE key=?", (key,))
        return cur.rowcount > 0


# --- activity log ------------------------------------------------------------


def log_event(kind: str, detail: str) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO events (kind, detail, created) VALUES (?,?,?)",
            (kind, detail, time.time()),
        )


def recent_events(limit: int = 50) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT kind, detail, created FROM events ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
