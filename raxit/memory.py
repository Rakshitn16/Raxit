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


def append_message(session: str, message: dict[str, Any]) -> None:
    """Persist one OpenAI-format message (`user`, `assistant` or `tool`)."""
    with connect() as conn:
        conn.execute(
            "INSERT INTO messages (session, role, content, created) VALUES (?,?,?,?)",
            (session, message["role"], json.dumps(message), time.time()),
        )


def load_messages(session: str, limit: int = 120) -> list[dict[str, Any]]:
    """Return the tail of a session, trimmed to a shape the API will accept.

    The newest `limit` messages are kept rather than the oldest, then repaired
    at both ends — see `_repair`.
    """
    with connect() as conn:
        rows = conn.execute(
            "SELECT content FROM messages WHERE session=? ORDER BY id DESC LIMIT ?",
            (session, limit),
        ).fetchall()
    return _repair([json.loads(r["content"]) for r in reversed(rows)])


def clear_session(session: str) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM messages WHERE session=?", (session,))


def _repair(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Make a message-list slice replayable.

    Two ways a stored history goes invalid, both rejected by the API:

    * The head is cut mid-tool-round, so the slice opens on a `tool` message
      whose originating `assistant` turn was left behind. Fix by skipping
      forward to the first `user` message.
    * The tail is an `assistant` turn with `tool_calls` that never got their
      `tool` replies — what a crash mid-loop leaves behind. Fix by dropping
      the dangling turn.
    """
    start = next((i for i, m in enumerate(msgs) if m.get("role") == "user"), None)
    if start is None:
        return []
    msgs = msgs[start:]

    while msgs:
        last = msgs[-1]
        if last.get("role") != "assistant" or not last.get("tool_calls"):
            break
        answered = {
            m.get("tool_call_id") for m in msgs if m.get("role") == "tool"
        }
        if all(c["id"] in answered for c in last["tool_calls"]):
            break
        msgs.pop()

    return msgs


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
    """Search stored facts, ranked by how many query words match.

    Matching is per-word rather than on the whole phrase, and treats `_`, `-`
    and spaces as the same character. The model writes keys like
    `favourite_drink` but later searches for "favourite drink", and a plain
    substring match silently returns nothing for that — a memory that quietly
    fails to recall is worse than no memory at all.
    """
    with connect() as conn:
        rows = conn.execute(
            "SELECT key, value, tags, updated FROM facts ORDER BY updated DESC"
        ).fetchall()

    words = [w for w in _normalize(query).split() if len(w) > 1]
    if not words:
        return [_fact(r) for r in rows[:limit]]

    scored = []
    for row in rows:
        haystack = _normalize(f"{row['key']} {row['value']} {row['tags']}")
        hits = sum(1 for w in words if w in haystack)
        if hits:
            scored.append((hits, row["updated"], row))

    scored.sort(key=lambda s: (-s[0], -s[1]))
    return [_fact(row) for _, _, row in scored[:limit]]


def _normalize(text: str) -> str:
    return text.lower().replace("_", " ").replace("-", " ").replace(",", " ")


def _fact(row: Any) -> dict[str, str]:
    return {"key": row["key"], "value": row["value"], "tags": row["tags"]}


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
