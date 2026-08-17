"""The grouped Termux:API tools and app control.

Same approach as the android wrappers: fake the binaries, assert the exact
command each tool builds. The grouped tools get extra attention because one
tool now stands in for several binaries, so a wrong `kind` silently runs the
wrong command rather than failing to dispatch.
"""

from __future__ import annotations

import json

import pytest

from raxit import tools
from raxit.tools import apps, device
from raxit.tools.apps import AndroidToolMissing


@pytest.fixture
def termux(monkeypatch: pytest.MonkeyPatch):
    """Record commands the device tools would have run; script their output."""

    class Fake:
        def __init__(self) -> None:
            self.commands: list[list[str]] = []
            self.output = ""

        def returns(self, value) -> Fake:
            self.output = value if isinstance(value, str) else json.dumps(value)
            return self

        @property
        def args(self) -> list[str]:
            return self.commands[-1]

    fake = Fake()

    def run(args, stdin=None, timeout=30):
        fake.commands.append(args)
        return fake.output

    monkeypatch.setattr(device, "_run", run)
    monkeypatch.setattr(device, "_json", lambda args: json.loads(run(args) or "null"))
    return fake


@pytest.fixture
def android_cli(monkeypatch: pytest.MonkeyPatch):
    """Same, for the `am`/`pm`/`monkey` binaries apps.py shells out to."""

    class Fake:
        def __init__(self) -> None:
            self.commands: list[list[str]] = []
            self.output = ""

        @property
        def args(self) -> list[str]:
            return self.commands[-1]

    fake = Fake()

    def run(args, timeout=30):
        fake.commands.append(args)
        return fake.output

    monkeypatch.setattr(apps, "_run", run)
    return fake


# --- contacts ----------------------------------------------------------------


async def test_contacts_returns_names_with_numbers(termux):
    """The tool that makes "call Amma" work without knowing a number."""
    termux.returns([{"name": "Amma", "number": "+919812345678"}])

    result = await device.contacts()

    assert "Amma" in result and "+919812345678" in result


async def test_contacts_filters_by_name_case_insensitively(termux):
    termux.returns(
        [{"name": "Amma", "number": "+91"}, {"name": "Ravi", "number": "+92"}]
    )

    result = await device.contacts(match="amma")

    assert "Amma" in result and "Ravi" not in result


async def test_contacts_says_so_when_nobody_matches(termux):
    """An empty string would read to the model as a broken tool and invite it
    to guess at a number, which is the one thing it must not do."""
    termux.returns([{"name": "Ravi", "number": "+92"}])

    assert "No contact matching" in await device.contacts(match="zzz")


async def test_contacts_handles_an_empty_address_book(termux):
    termux.returns([])
    assert await device.contacts() == "Address book empty."


# --- call log ----------------------------------------------------------------


async def test_call_log_summarises_recent_calls(termux):
    termux.returns(
        [{"type": "INCOMING", "name": "Amma", "date": "2026-08-17", "duration": "00:02"}]
    )

    result = await device.call_log()

    assert "INCOMING" in result and "Amma" in result


async def test_call_log_falls_back_to_the_number_when_there_is_no_name(termux):
    termux.returns([{"type": "MISSED", "phone_number": "+919812345678"}])
    assert "+919812345678" in await device.call_log()


async def test_call_log_passes_the_limit_through(termux):
    termux.returns([])
    await device.call_log(limit=25)
    assert "25" in termux.args


# --- sensors -----------------------------------------------------------------


async def test_sensor_takes_a_single_reading_rather_than_streaming(termux):
    """Without `-n 1` termux-sensor streams until killed, which would hang the
    turn until the tool timeout rather than answering."""
    termux.returns({"light": {"values": [12.0]}})

    await device.sensor("light")

    assert termux.args[:3] == ["termux-sensor", "-s", "light"]
    assert "-n" in termux.args and "1" in termux.args


async def test_sensor_returns_raw_text_when_the_output_is_not_json(termux):
    termux.returns("no such sensor")
    assert "no such sensor" in await device.sensor("light")


def test_the_sensor_enum_shown_to_the_model_matches_the_supported_list():
    schema = tools.REGISTRY["sensor"].input_schema
    assert schema["properties"]["kind"]["enum"] == device.SENSORS


# --- grouped device info -----------------------------------------------------


@pytest.mark.parametrize("kind, binary", sorted(
    (k, v[0]) for k, v in device.INFO_COMMANDS.items()
))
async def test_each_info_kind_runs_its_own_binary(termux, kind: str, binary: str):
    """One tool standing in for five binaries only works if the mapping is
    right — a wrong `kind` here runs the wrong command and reports it as
    fact, rather than failing loudly."""
    termux.returns("{}")

    await device.device_info(kind)

    assert termux.args == [binary]


def test_the_device_info_enum_covers_every_command_it_can_run():
    """The enum is the model's only view of what `kind` accepts, so a command
    added to the mapping without reaching the schema is unreachable."""
    schema = tools.REGISTRY["device_info"].input_schema
    assert set(schema["properties"]["kind"]["enum"]) == set(device.INFO_COMMANDS)


async def test_device_info_truncates_a_long_report(termux):
    termux.returns("x" * 5000)
    assert len(await device.device_info("wifi")) <= 1500


# --- settings ----------------------------------------------------------------


async def test_reading_volume_takes_no_stream(termux):
    termux.returns('[{"stream":"music","volume":7}]')
    result = await device.device_setting("volume")
    assert termux.args == ["termux-volume"]
    assert "music" in result


async def test_setting_volume_names_the_stream(termux):
    await device.device_setting("volume", stream="ring", value=5)
    assert termux.args == ["termux-volume", "ring", "5"]


async def test_setting_brightness(termux):
    await device.device_setting("brightness", value=200)
    assert termux.args == ["termux-brightness", "200"]


async def test_reading_brightness_explains_that_it_cannot(termux):
    """Android exposes no read path. Saying so beats returning a wrong number
    or a bare failure the model will retry."""
    assert "write-only" in await device.device_setting("brightness")


# --- dialogs -----------------------------------------------------------------


async def test_ask_user_returns_the_typed_answer(termux):
    termux.returns({"code": 0, "text": "half past six"})
    assert await device.ask_user("When?") == "half past six"


async def test_a_dismissed_dialog_is_reported_as_dismissed(termux):
    """A negative code means they swiped it away. Reporting that as an empty
    answer would let the agent treat silence as consent."""
    termux.returns({"code": -2, "text": ""})
    assert "Dismissed" in await device.ask_user("Send it?")


async def test_ask_user_passes_the_dialog_kind(termux):
    termux.returns({"code": 0, "text": "yes"})
    await device.ask_user("Sure?", kind="confirm")
    assert "confirm" in termux.args


async def test_dismiss_notification_clears_by_id(termux):
    await device.dismiss_notification("raxit-123")
    assert termux.args == ["termux-notification-remove", "raxit-123"]


# --- app control -------------------------------------------------------------


async def test_list_apps_strips_the_package_prefix(android_cli):
    android_cli.output = "package:com.spotify.music\npackage:com.whatsapp"

    result = await apps.list_apps()

    assert "com.spotify.music" in result
    assert "package:" not in result


async def test_list_apps_filters(android_cli):
    android_cli.output = "package:com.spotify.music\npackage:com.whatsapp"
    assert await apps.list_apps(match="spot") == "com.spotify.music"


async def test_list_apps_says_so_when_nothing_matches(android_cli):
    android_cli.output = "package:com.whatsapp"
    assert "No installed package matches" in await apps.list_apps(match="zzz")


async def test_launching_uses_monkey_because_am_needs_an_activity_name(
    android_cli,
):
    """`am start` needs package/activity, and nothing on the device reports
    the activity conveniently. `monkey` launches from the package alone."""
    await apps.launch_app("com.spotify.music")

    assert android_cli.args[:3] == ["monkey", "-p", "com.spotify.music"]
    assert "android.intent.category.LAUNCHER" in android_cli.args


async def test_share_text_sends_a_plain_text_intent(android_cli):
    await apps.share_text("the note", title="From Raxit")

    assert "android.intent.action.SEND" in android_cli.args
    assert "text/plain" in android_cli.args
    assert "the note" in android_cli.args


async def test_send_intent_builds_only_the_flags_it_was_given(android_cli):
    await apps.send_intent("android.settings.WIFI_SETTINGS")

    assert android_cli.args == [
        "am", "start", "-a", "android.settings.WIFI_SETTINGS",
    ]


async def test_send_intent_carries_data_package_and_one_extra(android_cli):
    await apps.send_intent(
        "android.intent.action.VIEW",
        data="https://example.com",
        package="com.android.chrome",
        extra_key="k",
        extra_value="v",
    )

    args = android_cli.args
    assert args[args.index("-d") + 1] == "https://example.com"
    assert args[args.index("-p") + 1] == "com.android.chrome"
    assert args[args.index("--es") + 1 : args.index("--es") + 3] == ["k", "v"]


def test_a_missing_android_binary_explains_itself(monkeypatch):
    """`am` and `pm` come from Android, not Termux, and are absent on some
    ROMs — so the failure has to name the fix rather than read as a crash."""
    monkeypatch.setattr(apps.shutil, "which", lambda name: None)

    with pytest.raises(AndroidToolMissing, match="termux-am"):
        apps._run(["am", "start"])


def test_a_real_binary_runs_through_the_apps_helper():
    assert apps._run(["echo", "launched"]) == "launched"


def test_a_failing_android_binary_raises_with_its_output(monkeypatch):
    monkeypatch.setattr(apps.shutil, "which", lambda name: "/system/bin/am")
    monkeypatch.setattr(
        apps.subprocess,
        "run",
        lambda *a, **k: type(
            "P", (), {"returncode": 1, "stdout": "", "stderr": "no such package"}
        )(),
    )

    with pytest.raises(RuntimeError, match="no such package"):
        apps._run(["am", "start"])
