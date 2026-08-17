"""The non-Android tools: memory, time, shell, HTTP.

The shell allowlist gets the most attention here. It is the boundary between
"the agent read the battery" and "the agent ran an arbitrary command on a
device holding someone's SMS", and it runs unattended.
"""

from __future__ import annotations

import datetime as dt
import re

import pytest

from raxit import memory
from raxit.config import settings
from raxit.tools import system

# --- memory tools ------------------------------------------------------------


def test_remember_confirms_what_it_stored():
    assert system.remember("home_wifi", "Airtel_5G") == "Remembered home_wifi."
    assert memory.recall("home_wifi")[0]["value"] == "Airtel_5G"


def test_recall_formats_facts_as_lines_the_model_can_read():
    system.remember("a", "1")
    system.remember("b", "2")
    assert set(system.recall().splitlines()) == {"- a: 1", "- b: 2"}


def test_recall_says_so_plainly_when_nothing_matches():
    """An empty string would read to the model as a broken tool; a sentence
    tells it to ask the user instead."""
    assert system.recall("anything") == "Nothing stored matching that."


def test_forget_distinguishes_deleting_from_never_having_had_it():
    system.remember("temp", "v")
    assert system.forget("temp") == "Forgot temp."
    assert system.forget("temp") == "No fact named temp."


def test_log_writes_to_the_activity_feed():
    assert system.log("note", "did a thing") == "Logged."
    assert memory.recent_events()[0]["detail"] == "did a thing"


# --- time --------------------------------------------------------------------


def test_now_reports_the_configured_timezone_not_the_hosts():
    """The tablet's process timezone is whatever Termux inherited; the answer
    the user wants is their own local time."""
    text = system.now()
    expected = dt.datetime.now(
        __import__("zoneinfo").ZoneInfo(settings.timezone)
    ).strftime("%A %d %B %Y")
    assert text.startswith(expected)


def test_now_is_human_readable_rather_than_an_iso_timestamp():
    """It is spoken aloud as often as it is read, and nobody wants to hear
    "2026-08-18T00:04" out of a speaker.

    Asserted on the shape rather than by hunting for a stray "T" — the
    obvious version of this test matches Tuesday and Thursday and so fails
    two days in seven.
    """
    text = system.now()

    assert text.split()[0].endswith("day"), "should open with a weekday name"
    assert re.search(r"\d{1,2}:\d{2}", text), "should carry a clock time"
    assert not re.search(r"\d{4}-\d{2}-\d{2}", text), "that is an ISO date"


# --- shell -------------------------------------------------------------------


async def test_an_allowlisted_command_runs_and_returns_its_output():
    assert await system.shell("date")


async def test_a_command_outside_the_allowlist_is_refused_with_a_reason(monkeypatch):
    """Refusal has to explain itself, or the model treats it as a transient
    failure and burns the round budget retrying."""
    result = await system.shell("rm -rf /")
    assert "not on the unattended allowlist" in result
    assert "SHELL_ALLOWLIST" in result


async def test_the_refusal_names_the_binary_not_the_whole_command():
    assert "'curl'" in await system.shell("curl http://example.com")


async def test_a_chained_command_cannot_smuggle_a_second_binary_past_the_gate():
    """`date ; rm -rf ~` passes the allowlist check, since it starts with an
    allowed binary. It is still safe, because the command is executed as an
    argv list rather than handed to a shell: `;` and `rm` arrive as arguments
    to `date`, which rejects them. Pinned, because switching to shell=True
    would turn this exact string into a working deletion."""
    result = await system.shell("date ; rm -rf ~")

    assert "extra operand" in result or "invalid" in result.lower()
    assert "date:" in result  # date complained, rather than rm running


async def test_an_empty_command_is_rejected():
    assert await system.shell("   ") == "Empty command."


async def test_an_unbalanced_quote_is_reported_rather_than_raised():
    result = await system.shell("cat 'unclosed")
    assert "Could not parse command" in result


async def test_output_is_truncated_so_one_command_cannot_flood_the_context(
    monkeypatch, tmp_path
):
    """`cat` is allowlisted, and a large file replayed into the transcript
    would blow the context window for every subsequent turn."""
    big = tmp_path / "big.txt"
    big.write_text("x" * 20000)
    monkeypatch.setattr(settings, "shell_allowlist", {"cat"})

    result = await system.shell(f"cat {big}")

    assert len(result) <= 4000


async def test_a_command_with_no_output_still_says_something(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "shell_allowlist", {"true"})
    result = await system.shell("true")
    assert "no output" in result


async def test_stderr_is_returned_too(monkeypatch):
    """A command that failed is more useful to the model than the silence of
    a discarded stderr."""
    monkeypatch.setattr(settings, "shell_allowlist", {"ls"})
    result = await system.shell("ls /definitely/not/here")
    assert "No such file" in result or "cannot access" in result


def test_the_allowlist_holds_no_way_to_write_or_fetch():
    """Read-only by construction: the model decides what to run, this decides
    what may run with nobody watching."""
    forbidden = {"rm", "mv", "cp", "curl", "wget", "sh", "bash", "python", "pkg"}
    assert not (settings.shell_allowlist & forbidden)


# --- http --------------------------------------------------------------------


async def test_fetch_url_returns_the_body(monkeypatch):
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["user-agent"] == "Raxit/1.0"
        return httpx.Response(200, text="the body")

    _patch_httpx(monkeypatch, handler)
    assert await system.fetch_url("http://example.com") == "the body"


async def test_fetch_url_truncates_a_large_page(monkeypatch):
    import httpx

    _patch_httpx(monkeypatch, lambda r: httpx.Response(200, text="y" * 50000))
    assert len(await system.fetch_url("http://example.com")) == 20000


async def test_fetch_url_raises_on_an_http_error(monkeypatch):
    """Surfaced to the model as an error tool result rather than a page of
    HTML saying 404, which it would otherwise try to summarise."""
    import httpx

    _patch_httpx(monkeypatch, lambda r: httpx.Response(404, text="nope"))
    with pytest.raises(httpx.HTTPStatusError):
        await system.fetch_url("http://example.com")


async def test_fetch_url_follows_redirects(monkeypatch):
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/":
            return httpx.Response(302, headers={"location": "/final"})
        return httpx.Response(200, text="arrived")

    _patch_httpx(monkeypatch, handler)
    assert await system.fetch_url("http://example.com/") == "arrived"


def _patch_httpx(monkeypatch, handler) -> None:
    import httpx

    original = httpx.AsyncClient

    def build(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return original(*args, **kwargs)

    monkeypatch.setattr(system.httpx, "AsyncClient", build)
