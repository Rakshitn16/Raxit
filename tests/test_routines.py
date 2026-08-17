"""Routines: YAML parsing, scheduling, and what happens when one fails.

A routine that fails at 7am fails with nobody watching, so the recovery
behaviour — a broken file skipped rather than taking the others down with it,
an exception logged rather than killing the scheduler — is the point.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from raxit import memory
from raxit.agent import Agent, Event
from raxit.routines import Routine, RoutineRunner, load_all

VALID = """
name: morning_brief
cron: "0 7 * * *"
prompt: |
  Check the battery and read out anything overnight.
"""


def write(directory: Path, filename: str, body: str) -> Path:
    path = directory / filename
    path.write_text(body)
    return path


# --- parsing -----------------------------------------------------------------


def test_a_valid_routine_parses(tmp_path: Path):
    routine = Routine.from_file(write(tmp_path, "morning.yaml", VALID))
    assert routine.name == "morning_brief"
    assert routine.cron == "0 7 * * *"
    assert "Check the battery" in routine.prompt


def test_a_routine_defaults_to_enabled_thinking_and_the_shared_session(
    tmp_path: Path,
):
    """Thinking defaults on for routines and off for interactive chat: a
    routine plans several steps and nobody is waiting on the latency."""
    routine = Routine.from_file(write(tmp_path, "m.yaml", VALID))
    assert (routine.enabled, routine.thinking, routine.session) == (
        True, True, "routines",
    )


def test_overrides_are_read_from_the_file(tmp_path: Path):
    body = VALID + "enabled: false\nthinking: false\nsession: kitchen\n"
    routine = Routine.from_file(write(tmp_path, "m.yaml", body))
    assert (routine.enabled, routine.thinking, routine.session) == (
        False, False, "kitchen",
    )


@pytest.mark.parametrize(
    "body, missing",
    [
        ("cron: '0 7 * * *'\nprompt: x\n", "name"),
        ("name: n\nprompt: x\n", "cron"),
        ("name: n\ncron: '0 7 * * *'\n", "prompt"),
        ("name: n\n", "cron, prompt"),
    ],
)
def test_a_missing_field_is_named_in_the_error(tmp_path: Path, body: str, missing: str):
    with pytest.raises(ValueError, match=missing):
        Routine.from_file(write(tmp_path, "broken.yaml", body))


def test_an_empty_file_is_rejected_rather_than_parsed_as_nothing(tmp_path: Path):
    with pytest.raises(ValueError):
        Routine.from_file(write(tmp_path, "empty.yaml", ""))


# --- loading a directory -----------------------------------------------------


def test_both_yaml_extensions_are_picked_up(tmp_path: Path):
    write(tmp_path, "a.yaml", VALID)
    write(tmp_path, "b.yml", VALID.replace("morning_brief", "evening"))
    assert {r.name for r in load_all(tmp_path)} == {"morning_brief", "evening"}


def test_a_broken_file_is_skipped_without_taking_the_others_with_it(
    tmp_path: Path, caplog
):
    """One malformed YAML file must not silently disable every routine on the
    tablet — including the ones that were working yesterday."""
    write(tmp_path, "good.yaml", VALID)
    write(tmp_path, "broken.yaml", "name: only_a_name\n")

    routines = load_all(tmp_path)

    assert [r.name for r in routines] == ["morning_brief"]
    assert "broken.yaml" in caplog.text


def test_unparseable_yaml_is_skipped_too(tmp_path: Path):
    write(tmp_path, "good.yaml", VALID)
    write(tmp_path, "bad.yaml", "name: [unclosed\n")
    assert [r.name for r in load_all(tmp_path)] == ["morning_brief"]


def test_non_yaml_files_are_ignored(tmp_path: Path):
    write(tmp_path, "notes.txt", "not a routine")
    assert load_all(tmp_path) == []


def test_an_empty_directory_loads_nothing(tmp_path: Path):
    assert load_all(tmp_path) == []


# --- the shipped routines ----------------------------------------------------


def test_the_routines_in_the_repo_actually_parse():
    """These ship as working examples; a typo in one is the first thing a new
    user sees fail."""
    from raxit.config import ROUTINES_DIR

    routines = load_all(ROUTINES_DIR)
    assert {r.name for r in routines} == {"morning_brief", "battery_watch"}


def test_the_shipped_cron_expressions_are_valid():
    from apscheduler.triggers.cron import CronTrigger

    from raxit.config import ROUTINES_DIR

    for routine in load_all(ROUTINES_DIR):
        CronTrigger.from_crontab(routine.cron)


# --- scheduling --------------------------------------------------------------


@pytest.fixture
def runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RoutineRunner:
    monkeypatch.setattr("raxit.routines.ROUTINES_DIR", tmp_path)
    return RoutineRunner(Agent())


def test_loading_schedules_the_enabled_routines(runner, tmp_path: Path):
    write(tmp_path, "a.yaml", VALID)
    write(
        tmp_path, "b.yaml", VALID.replace("morning_brief", "evening") + "enabled: false\n"
    )

    runner.load()

    assert set(runner.routines) == {"morning_brief", "evening"}
    assert [j.id for j in runner.scheduler.get_jobs()] == ["morning_brief"]


def test_a_reload_replaces_the_schedule_rather_than_appending_to_it(
    runner, tmp_path: Path
):
    """Reload is exposed in the UI and hit repeatedly while editing a routine.
    Stacking duplicate jobs would fire the same brief several times."""
    path = write(tmp_path, "a.yaml", VALID)
    runner.load()
    path.write_text(VALID.replace("morning_brief", "renamed"))

    runner.load()

    assert set(runner.routines) == {"renamed"}
    assert [j.id for j in runner.scheduler.get_jobs()] == ["renamed"]


def test_a_removed_file_is_unscheduled(runner, tmp_path: Path):
    path = write(tmp_path, "a.yaml", VALID)
    runner.load()
    path.unlink()

    runner.load()

    assert runner.routines == {}
    assert runner.scheduler.get_jobs() == []


def test_a_routine_does_not_stack_up_behind_itself(runner, tmp_path: Path):
    """A tablet asleep at 7am wakes with a backlog; without coalescing it
    would fire every missed run in a burst and spend the whole rate limit."""
    write(tmp_path, "a.yaml", VALID)

    runner.load()

    job = runner.scheduler.get_jobs()[0]
    assert job.max_instances == 1
    assert job.coalesce is True
    assert job.misfire_grace_time == 600


def test_describe_reports_each_routine_for_the_ui(runner, tmp_path: Path):
    write(tmp_path, "a.yaml", VALID)
    runner.load()

    described = runner.describe()

    assert described[0]["name"] == "morning_brief"
    assert described[0]["cron"] == "0 7 * * *"
    assert described[0]["enabled"] is True


def test_describe_works_before_the_scheduler_has_started(runner, tmp_path: Path):
    """Jobs added before `start()` are pending and carry no next_run_time at
    all, so reading it naively raises during startup — when the UI first
    calls /api/status."""
    write(tmp_path, "a.yaml", VALID)
    runner.load()

    assert runner.describe()[0]["next_run"] is None


def test_describe_includes_disabled_routines_so_they_can_be_seen(
    runner, tmp_path: Path
):
    write(tmp_path, "a.yaml", VALID + "enabled: false\n")
    runner.load()

    assert runner.describe()[0]["enabled"] is False
    assert runner.describe()[0]["next_run"] is None


def test_stopping_a_scheduler_that_never_started_is_harmless(runner):
    runner.stop()


async def test_starting_loads_the_routines_and_runs_the_scheduler(
    runner, tmp_path: Path
):
    """Async because AsyncIOScheduler binds to the running loop when it
    starts — which is why the server starts it from its lifespan hook rather
    than at import."""
    write(tmp_path, "a.yaml", VALID)

    runner.start()
    try:
        assert runner.scheduler.running
        assert set(runner.routines) == {"morning_brief"}
        assert runner.describe()[0]["next_run"] is not None
    finally:
        runner.stop()

    # `shutdown(wait=False)` hands the teardown back to the loop rather than
    # blocking, which is why the server's shutdown hook is async.
    await asyncio.sleep(0)
    assert not runner.scheduler.running


# --- firing ------------------------------------------------------------------


class FakeAgent:
    """Stands in for the agent so a routine can be fired without a provider."""

    def __init__(self, events: list[Event] | Exception) -> None:
        self.events = events
        self.seen: list[dict] = []

    async def run(self, session, prompt, *, thinking=None, unattended=False, **kw):
        self.seen.append(
            {
                "session": session,
                "prompt": prompt,
                "thinking": thinking,
                "unattended": unattended,
            }
        )
        if isinstance(self.events, Exception):
            raise self.events
        for event in self.events:
            yield event


def fired(tmp_path: Path, monkeypatch, agent: FakeAgent, body: str = VALID):
    monkeypatch.setattr("raxit.routines.ROUTINES_DIR", tmp_path)
    write(tmp_path, "a.yaml", body)
    runner = RoutineRunner(agent)
    runner.load()
    return runner


async def test_firing_returns_the_agents_closing_text(tmp_path, monkeypatch):
    agent = FakeAgent([Event("done", {"text": "Battery is 83%."})])
    runner = fired(tmp_path, monkeypatch, agent)

    assert await runner.fire("morning_brief") == "Battery is 83%."


async def test_a_routine_runs_unattended_with_its_own_thinking_setting(
    tmp_path, monkeypatch
):
    """Unattended is what makes the agent decide rather than ask — there is
    nobody there to answer a clarifying question at 7am."""
    agent = FakeAgent([Event("done", {"text": "ok"})])
    runner = fired(tmp_path, monkeypatch, agent, VALID + "thinking: false\n")

    await runner.fire("morning_brief")

    assert agent.seen[0]["unattended"] is True
    assert agent.seen[0]["thinking"] is False
    assert agent.seen[0]["session"] == "routines"


async def test_a_firing_is_bracketed_in_the_audit_log(tmp_path, monkeypatch):
    agent = FakeAgent([Event("done", {"text": "Battery is 83%."})])
    runner = fired(tmp_path, monkeypatch, agent)

    await runner.fire("morning_brief")

    kinds = [e["kind"] for e in memory.recent_events()]
    assert "routine_start" in kinds and "routine_done" in kinds


async def test_an_error_event_is_logged_and_the_routine_still_returns(
    tmp_path, monkeypatch
):
    agent = FakeAgent(
        [Event("error", {"message": "rate limited"}), Event("done", {"text": "partial"})]
    )
    runner = fired(tmp_path, monkeypatch, agent)

    assert await runner.fire("morning_brief") == "partial"
    assert any(e["kind"] == "routine_error" for e in memory.recent_events())


async def test_a_crash_is_caught_so_the_scheduler_survives_it(tmp_path, monkeypatch):
    """An exception escaping into APScheduler kills the job. On a tablet
    nobody is looking at the log, so the routine would just stop happening."""
    runner = fired(tmp_path, monkeypatch, FakeAgent(RuntimeError("no network")))

    result = await runner.fire("morning_brief")

    assert "no network" in result
    assert any(e["kind"] == "routine_error" for e in memory.recent_events())


async def test_firing_an_unknown_routine_raises(tmp_path, monkeypatch):
    runner = fired(tmp_path, monkeypatch, FakeAgent([]))

    with pytest.raises(KeyError):
        await runner.fire("nope")
