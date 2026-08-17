"""Configuration for the Raxit agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("RAXIT_DATA_DIR", ROOT / "data"))
ROUTINES_DIR = Path(os.getenv("RAXIT_ROUTINES_DIR", ROOT / "routines"))
WEB_DIR = ROOT / "web"

DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "raxit.db"

# Claude Opus 5 is the brain. The tablet's Helio G99 cannot run a local model
# fast enough for conversation, so inference is remote and execution is local.
MODEL = os.getenv("RAXIT_MODEL", "claude-opus-5")

# Effort trades intelligence against latency and cost. `low` is snappy enough
# for "set a timer"; routines that plan multi-step work override this per call.
EFFORT = os.getenv("RAXIT_EFFORT", "medium")

MAX_TOKENS = int(os.getenv("RAXIT_MAX_TOKENS", "8192"))

# Cap the agentic loop so a confused tool never spins forever on battery.
MAX_TOOL_ITERATIONS = int(os.getenv("RAXIT_MAX_TOOL_ITERATIONS", "16"))

HOST = os.getenv("RAXIT_HOST", "127.0.0.1")
PORT = int(os.getenv("RAXIT_PORT", "8788"))

# Shell commands the agent may run unattended. Anything not listed needs the
# user to approve it in the UI first. Claude decides *what* to run; this decides
# what is allowed to run without a human in the loop.
SHELL_ALLOWLIST = {
    "termux-battery-status",
    "termux-wifi-connectioninfo",
    "termux-telephony-deviceinfo",
    "termux-notification-list",
    "df",
    "free",
    "uptime",
    "date",
    "ls",
    "cat",
}

OWNER_NAME = os.getenv("RAXIT_OWNER_NAME", "the user")
TIMEZONE = os.getenv("RAXIT_TIMEZONE", "Asia/Kolkata")


@dataclass
class Settings:
    model: str = MODEL
    effort: str = EFFORT
    max_tokens: int = MAX_TOKENS
    max_tool_iterations: int = MAX_TOOL_ITERATIONS
    owner_name: str = OWNER_NAME
    timezone: str = TIMEZONE
    shell_allowlist: set[str] = field(default_factory=lambda: set(SHELL_ALLOWLIST))


settings = Settings()
