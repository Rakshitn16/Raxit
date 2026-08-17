"""Configuration for the Raxit agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = Path(os.getenv("RAXIT_DATA_DIR", ROOT / "data"))
ROUTINES_DIR = Path(os.getenv("RAXIT_ROUTINES_DIR", ROOT / "routines"))
WEB_DIR = ROOT / "web"

DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "raxit.db"

# NVIDIA NIM speaks the OpenAI wire format, so the `openai` SDK works against
# it unchanged. Point BASE_URL elsewhere (Groq, OpenRouter, a local vLLM) and
# nothing else in the codebase needs to know.
BASE_URL = os.getenv("RAXIT_BASE_URL", "https://integrate.api.nvidia.com/v1")
API_KEY = os.getenv("NVIDIA_API_KEY") or os.getenv("RAXIT_API_KEY", "")

# Verified to handle multi-round tool calling on NVIDIA's free tier.
MODEL = os.getenv("RAXIT_MODEL", "nvidia/nemotron-3-super-120b-a12b")

# Nemotron exposes its scratchpad as a separate `reasoning_content` stream.
# Measured on this hardware: ~10.3s with thinking vs ~4.1s without, for the
# same single-tool question and the same answer. Interactive use is the common
# path and 10s is a long time to wait for a spoken reply, so this defaults off
# and individual routines opt back in via `thinking: true`.
ENABLE_THINKING = os.getenv("RAXIT_THINKING", "false").lower() in ("1", "true", "yes")
REASONING_BUDGET = int(os.getenv("RAXIT_REASONING_BUDGET", "4096"))

TEMPERATURE = float(os.getenv("RAXIT_TEMPERATURE", "0.6"))
TOP_P = float(os.getenv("RAXIT_TOP_P", "0.95"))
MAX_TOKENS = int(os.getenv("RAXIT_MAX_TOKENS", "4096"))

# The free tier allows ~40 requests/minute and one turn costs one request per
# tool round, so a runaway loop burns the quota fast as well as the battery.
MAX_TOOL_ITERATIONS = int(os.getenv("RAXIT_MAX_TOOL_ITERATIONS", "8"))
MAX_RETRIES = int(os.getenv("RAXIT_MAX_RETRIES", "4"))

HOST = os.getenv("RAXIT_HOST", "127.0.0.1")
PORT = int(os.getenv("RAXIT_PORT", "8788"))

# Shell commands the agent may run unattended. Anything else is refused with an
# explanation. The model decides *what* to run; this decides what may run
# without a human.
SHELL_ALLOWLIST = {
    "termux-battery-status",
    "termux-wifi-connectioninfo",
    "termux-telephony-deviceinfo",
    "termux-notification-list",
    "df", "free", "uptime", "date", "ls", "cat",
}

OWNER_NAME = os.getenv("RAXIT_OWNER_NAME", "the user")
TIMEZONE = os.getenv("RAXIT_TIMEZONE", "Asia/Kolkata")


@dataclass
class Settings:
    base_url: str = BASE_URL
    api_key: str = API_KEY
    model: str = MODEL
    enable_thinking: bool = ENABLE_THINKING
    reasoning_budget: int = REASONING_BUDGET
    temperature: float = TEMPERATURE
    top_p: float = TOP_P
    max_tokens: int = MAX_TOKENS
    max_tool_iterations: int = MAX_TOOL_ITERATIONS
    max_retries: int = MAX_RETRIES
    owner_name: str = OWNER_NAME
    timezone: str = TIMEZONE
    shell_allowlist: set[str] = field(default_factory=lambda: set(SHELL_ALLOWLIST))


settings = Settings()
