"""Memory, shell, scheduling and web tools."""

from __future__ import annotations

import datetime as dt
import shlex
import subprocess
import zoneinfo

import httpx

from .. import memory
from ..config import settings
from .registry import obj, opt, run_blocking, tool


@tool(
    "remember",
    "Store a durable fact about the user, their preferences, or their world. "
    "Use this for things worth knowing next week, not for conversational "
    "context — the transcript already carries that.",
    obj(
        key={"type": "string", "description": "Short stable identifier, e.g. 'home_wifi'."},
        value={"type": "string"},
        tags={"type": "string", "description": "Comma-separated, for later filtering."},
    ),
)
def remember(key: str, value: str, tags: str = "") -> str:
    memory.remember(key, value, tags)
    return f"Remembered {key}."


@tool(
    "recall",
    "Search stored facts. Call this before asking the user something you may "
    "already have been told.",
    opt(query={"type": "string", "description": "Substring match; omit for everything."}),
)
def recall(query: str = "") -> str:
    facts = memory.recall(query)
    if not facts:
        return "Nothing stored matching that."
    return "\n".join(f"- {f['key']}: {f['value']}" for f in facts)


@tool(
    "forget",
    "Delete a stored fact by key.",
    obj(key={"type": "string"}),
)
def forget(key: str) -> str:
    return f"Forgot {key}." if memory.forget(key) else f"No fact named {key}."


@tool(
    "now",
    "Current local date and time. Call this rather than guessing — you have no "
    "reliable sense of the present moment.",
    opt(),
)
def now() -> str:
    tz = zoneinfo.ZoneInfo(settings.timezone)
    return dt.datetime.now(tz).strftime("%A %d %B %Y, %H:%M %Z")


@tool(
    "shell",
    "Run a shell command on the tablet and return its output. Commands outside "
    "the allowlist are refused rather than queued, so prefer a dedicated tool "
    "when one exists.",
    obj(command={"type": "string"}),
)
async def shell(command: str) -> str:
    try:
        parts = shlex.split(command)
    except ValueError as exc:
        return f"Could not parse command: {exc}"
    if not parts:
        return "Empty command."
    if parts[0] not in settings.shell_allowlist:
        return (
            f"'{parts[0]}' is not on the unattended allowlist. Ask the user to "
            "run it themselves, or to add it to SHELL_ALLOWLIST in config.py."
        )

    def go() -> str:
        proc = subprocess.run(parts, capture_output=True, text=True, timeout=60)
        out = (proc.stdout + proc.stderr).strip()
        return out[:4000] or f"(no output, exit {proc.returncode})"

    return await run_blocking(go)


@tool(
    "fetch_url",
    "Fetch a URL and return its body as text. Use for APIs and plain pages; it "
    "does not execute JavaScript, so single-page apps will come back empty.",
    obj(url={"type": "string"}),
)
async def fetch_url(url: str) -> str:
    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        resp = await client.get(url, headers={"User-Agent": "Raxit/1.0"})
        resp.raise_for_status()
        return resp.text[:20000]


@tool(
    "log",
    "Record a note in the activity log, visible to the user in the web UI. Use "
    "it when a routine does something worth an audit trail.",
    obj(kind={"type": "string"}, detail={"type": "string"}),
)
def log(kind: str, detail: str) -> str:
    memory.log_event(kind, detail)
    return "Logged."
