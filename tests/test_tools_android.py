"""The termux-api wrappers, with the binaries faked.

CI has no Android, so these check the two things that are testable off-device
and still break in practice: the exact command each tool builds, and what the
model is told when the Termux:API bridge is not there — the single most
common setup failure, since the CLI package installs fine without the app.
"""

from __future__ import annotations

import json

import pytest

from raxit.tools import android
from raxit.tools.android import TermuxUnavailable


@pytest.fixture
def termux(monkeypatch: pytest.MonkeyPatch):
    """Record the commands the tools would have run, and script their output."""

    class Fake:
        def __init__(self) -> None:
            self.commands: list[tuple[list[str], str | None]] = []
            self.output = ""

        def returns(self, value) -> Fake:
            self.output = value if isinstance(value, str) else json.dumps(value)
            return self

        @property
        def args(self) -> list[str]:
            return self.commands[-1][0]

        @property
        def stdin(self) -> str | None:
            return self.commands[-1][1]

    fake = Fake()

    def run(args, stdin=None, timeout=android.TIMEOUT):
        fake.commands.append((args, stdin))
        return fake.output

    monkeypatch.setattr(android, "_run", run)
    return fake


# --- the bridge --------------------------------------------------------------


async def test_a_missing_binary_says_how_to_install_it(monkeypatch):
    """`pkg install termux-api` gives you the binaries; without the F-Droid
    app they are stubs. Anyone who installs only one hits this, so the error
    has to name both halves."""
    monkeypatch.setattr(android.shutil, "which", lambda name: None)

    with pytest.raises(TermuxUnavailable, match="F-Droid"):
        android._run(["termux-battery-status"])


async def test_a_missing_binary_names_the_command_that_was_missing(monkeypatch):
    monkeypatch.setattr(android.shutil, "which", lambda name: None)

    with pytest.raises(TermuxUnavailable, match="termux-tts-speak"):
        android._run(["termux-tts-speak"])


def test_a_real_binary_is_run_and_its_output_returned():
    """The wrappers are all faked below this line, so run one command for real
    to prove the subprocess plumbing itself works."""
    assert android._run(["echo", "hello"]) == "hello"


def test_stdin_reaches_the_binary():
    # `speak`, `send_sms` and `clipboard` all pass their payload this way.
    assert android._run(["cat"], stdin="piped in") == "piped in"


def test_a_nonzero_exit_is_raised_with_whatever_the_binary_said(monkeypatch):
    monkeypatch.setattr(android.shutil, "which", lambda name: "/usr/bin/x")
    monkeypatch.setattr(
        android.subprocess,
        "run",
        lambda *a, **k: type(
            "P", (), {"returncode": 1, "stdout": "", "stderr": "denied"}
        )(),
    )

    with pytest.raises(RuntimeError, match="denied"):
        android._run(["termux-sms-list"])


def test_empty_json_output_becomes_an_empty_object(monkeypatch):
    """`termux-notification-list` prints nothing at all when there is nothing
    to list, and `json.loads("")` would raise."""
    monkeypatch.setattr(android, "_run", lambda *a, **k: "")
    assert android._json(["termux-notification-list"]) == {}


# --- speech ------------------------------------------------------------------


async def test_speak_sends_the_text_on_stdin(termux):
    """Passing it as an argument would break on any text containing a quote —
    which is most sentences the model produces."""
    result = await android.speak("Good morning.")

    assert termux.stdin == "Good morning."
    assert termux.args[0] == "termux-tts-speak"
    assert "Good morning." in result


async def test_speak_passes_pitch_and_rate(termux):
    await android.speak("hi", pitch=1.5, rate=0.8)

    assert "-p" in termux.args and "1.5" in termux.args
    assert "-r" in termux.args and "0.8" in termux.args


async def test_speak_truncates_its_confirmation(termux):
    """The confirmation goes back to the model as a tool result; echoing a
    whole spoken paragraph doubles it in the transcript."""
    result = await android.speak("x" * 500)
    assert len(result) < 120


async def test_listen_returns_what_was_heard(termux):
    termux.returns("turn the torch on")
    assert await android.listen() == "turn the torch on"


async def test_listen_reports_silence_rather_than_an_empty_string(termux):
    termux.returns("")
    assert await android.listen() == "(nothing heard)"


# --- notifications -----------------------------------------------------------


async def test_notify_passes_title_content_and_priority(termux):
    await android.notify("Battery low", "12% left", priority="high")

    assert termux.args[:1] == ["termux-notification"]
    assert "--title" in termux.args and "Battery low" in termux.args
    assert "--priority" in termux.args and "high" in termux.args


async def test_notify_reuses_an_id_per_title_so_repeats_replace_each_other(termux):
    """A routine firing hourly would otherwise stack a fresh notification each
    time and bury the status bar."""
    await android.notify("Battery low", "12%")
    first = termux.args[termux.args.index("--id") + 1]
    await android.notify("Battery low", "9%")
    second = termux.args[termux.args.index("--id") + 1]

    assert first == second


async def test_notify_defaults_to_ordinary_priority(termux):
    await android.notify("t", "c")
    assert termux.args[termux.args.index("--priority") + 1] == "default"


async def test_read_notifications_summarises_each_one(termux):
    termux.returns(
        [
            {"packageName": "com.whatsapp", "title": "Amma", "content": "call me"},
            {"packageName": "com.android.calendar", "title": "Standup", "content": "9am"},
        ]
    )

    result = await android.read_notifications()

    assert "com.whatsapp" in result and "call me" in result
    assert len(result.splitlines()) == 2


async def test_read_notifications_says_so_when_there_are_none(termux):
    termux.returns([])
    assert await android.read_notifications() == "No notifications."


async def test_read_notifications_tolerates_missing_fields(termux):
    """Not every app sets a title; a KeyError here would surface as a tool
    failure for a perfectly normal status bar."""
    termux.returns([{"packageName": "com.x"}])
    assert "com.x" in await android.read_notifications()


# --- device state ------------------------------------------------------------


async def test_battery_reports_the_numbers_that_matter(termux):
    termux.returns(
        {"percentage": 83, "status": "CHARGING", "temperature": 31.2, "health": "GOOD"}
    )

    result = await android.battery()

    assert "83%" in result and "CHARGING" in result and "GOOD" in result


async def test_location_defaults_to_the_network_provider(termux):
    """GPS takes tens of seconds and drains the battery; network is right for
    "where am I roughly" which is what gets asked."""
    termux.returns({"latitude": 12.9, "longitude": 77.6, "accuracy": 20})

    await android.location()

    assert termux.args[-1] == "network"


async def test_location_honours_an_explicit_provider(termux):
    termux.returns({"latitude": 1, "longitude": 2, "accuracy": 3})
    await android.location(provider="gps")
    assert termux.args[-1] == "gps"


async def test_location_reports_unavailability_rather_than_empty_coordinates(termux):
    termux.returns("")
    assert await android.location() == "Location unavailable."


async def test_clipboard_reads_by_default(termux):
    termux.returns("copied text")
    assert await android.clipboard() == "copied text"
    assert termux.args == ["termux-clipboard-get"]


async def test_clipboard_writes_through_stdin(termux):
    await android.clipboard(action="set", text="new value")

    assert termux.args == ["termux-clipboard-set"]
    assert termux.stdin == "new value"


async def test_an_empty_clipboard_says_so(termux):
    termux.returns("")
    assert await android.clipboard() == "(clipboard empty)"


async def test_torch_maps_the_boolean_to_the_binarys_words(termux):
    await android.torch(True)
    assert termux.args == ["termux-torch", "on"]
    await android.torch(False)
    assert termux.args == ["termux-torch", "off"]


async def test_toast_shows_at_the_top_where_it_is_not_under_a_thumb(termux):
    await android.toast("done")

    assert termux.args == ["termux-toast", "-g", "top", "done"]


async def test_vibrate_defaults_to_a_short_buzz_and_forces_it(termux):
    """`-f` vibrates even in silent mode, which is the mode a tablet spends
    most of its life in."""
    await android.vibrate()

    assert "400" in termux.args and "-f" in termux.args


async def test_take_photo_expands_a_home_relative_path(termux):
    result = await android.take_photo("~/shot.jpg")

    assert "~" not in termux.args[-1]
    assert termux.args[-1].endswith("/shot.jpg")
    assert "~" not in result


async def test_take_photo_defaults_to_the_rear_camera(termux):
    await android.take_photo()
    assert termux.args[termux.args.index("-c") + 1] == "0"


# --- communication -----------------------------------------------------------


async def test_send_sms_puts_the_body_on_stdin_and_the_number_in_the_flag(termux):
    await android.send_sms("+919812345678", "on my way")

    assert termux.args == ["termux-sms-send", "-n", "+919812345678"]
    assert termux.stdin == "on my way"


async def test_read_sms_summarises_the_inbox(termux):
    termux.returns([{"number": "+91", "received": "10:00", "body": "hello"}])

    result = await android.read_sms(limit=5)

    assert "hello" in result
    assert "5" in termux.args and "inbox" in termux.args


async def test_read_sms_says_so_when_the_inbox_is_empty(termux):
    termux.returns([])
    assert await android.read_sms() == "No messages."


async def test_call_dials_the_number(termux):
    assert "+91" in await android.call("+91")
    assert termux.args == ["termux-telephony-call", "+91"]


async def test_open_url_hands_off_to_the_default_app(termux):
    await android.open_url("spotify:track:123")
    assert termux.args == ["termux-open-url", "spotify:track:123"]
