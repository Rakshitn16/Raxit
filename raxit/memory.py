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
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

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

    A tool round is an `assistant` turn carrying `tool_calls` plus one `tool`
    message per call. The API rejects the history unless every round is
    complete, and a stored transcript breaks that in three ways:

    * The head is cut mid-round, so the slice opens on a `tool` message whose
      `assistant` turn was left behind — the ordinary cost of keeping only a
      tail. Fixed by skipping forward to the first `user` message.
    * A round was only partly answered, because results are written one at a
      time and Termux killed the process between two of them.
    * A round was not answered at all, which is where a crash mid-loop lands.

    The last two are dropped wholesale — the assistant turn and any orphaned
    replies with it. A partial round matters more than it looks: it sits in
    the middle of the transcript rather than at the end, so it is reloaded on
    every subsequent turn and keeps failing until it falls out of the window.
    """
    start = next((i for i, m in enumerate(msgs) if m.get("role") == "user"), None)
    if start is None:
        return []
    msgs = msgs[start:]

    answered = {m.get("tool_call_id") for m in msgs if m.get("role") == "tool"}
    orphaned: set[str] = set()
    kept: list[dict[str, Any]] = []

    for message in msgs:
        calls = message.get("tool_calls") if message.get("role") == "assistant" else None
        if calls and not all(c["id"] in answered for c in calls):
            orphaned.update(c["id"] for c in calls)
            continue
        kept.append(message)

    # A `tool` message whose round was just dropped, or one that never had a
    # round to begin with, has nothing left to attach to.
    live = _called(kept) - orphaned
    return [
        m for m in kept if m.get("role") != "tool" or m.get("tool_call_id") in live
    ]


def _called(msgs: list[dict[str, Any]]) -> set[str]:
    return {
        call["id"]
        for m in msgs
        if m.get("role") == "assistant"
        for call in m.get("tool_calls") or []
    }


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
