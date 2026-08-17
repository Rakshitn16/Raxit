"""Timezone resolution has no system fallback on Android.

Android ships no IANA timezone database — there is no /usr/share/zoneinfo,
nothing at all for `zoneinfo` to fall back to. Every platform this project
was developed and tested on has one, so `zoneinfo.ZoneInfo(...)` silently
works everywhere except the one platform Raxit actually runs on, and the
failure only shows up as the scheduler crashing on startup on a tablet.

`tzdata` is a pure-data pip package that plugs the same gap `zoneinfo` uses
the system database for. These tests force that scenario here, on a machine
that does have a system database, by clearing `zoneinfo.TZPATH` — the same
condition Termux is in permanently — so a regression is caught in CI rather
than on a device an hour from being handed to somebody.
"""

from __future__ import annotations

import zoneinfo
from pathlib import Path

import pytest

from raxit.agent import Agent
from raxit.config import settings
from raxit.routines import RoutineRunner


@pytest.fixture
def no_system_tzdata():
    """Simulate Android: no OS timezone files, only the `tzdata` pip package."""
    original = zoneinfo.TZPATH
    zoneinfo.reset_tzpath(to=())
    try:
        yield
    finally:
        zoneinfo.reset_tzpath(to=original)


def test_the_configured_timezone_resolves_with_no_system_tzdata(no_system_tzdata):
    # This is the bug itself, isolated: without the `tzdata` package, this
    # raises ZoneInfoNotFoundError on any platform with no OS database —
    # Android among them.
    zoneinfo.ZoneInfo(settings.timezone)


def test_the_scheduler_starts_with_no_system_tzdata(no_system_tzdata):
    """The actual crash: AsyncIOScheduler resolves the timezone in its
    constructor, before the server can serve a single request."""
    RoutineRunner(Agent())


def test_the_now_tool_works_with_no_system_tzdata(no_system_tzdata):
    from raxit.tools.system import now

    assert now()


def test_tzdata_is_a_declared_dependency():
    """The fix, pinned so it can't quietly fall out of requirements.txt again
    — this project only reads it because a human read a traceback once."""
    requirements = Path(__file__).resolve().parent.parent / "requirements.txt"
    text = requirements.read_text().lower()
    assert "tzdata" in text
