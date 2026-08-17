"""Agent mode: a goal pursued across several turns, unsupervised.

The behaviour worth pinning is all about stopping. A mission that will not
stop spends the free tier and the battery; one that stops early looks exactly
like one that succeeded. Both failures are silent, because by definition
nobody is watching.
"""

from __future__ import annotations

import asyncio

import pytest

from raxit import memory, missions
from raxit.agent import Event
from raxit.missions import DONE_MARKER, Mission, MissionRunner


@pytest.fixture(autouse=True)
def mission_table(fresh_db):
    missions.init()


class FakeAgent:
    """Replies with a scripted answer per step, recording the prompts."""

    def __init__(self, replies: list[str] | Exception) -> None:
        self.replies = replies
        self.prompts: list[str] = []
        self.kwargs: list[dict] = []
        self.started = asyncio.Event()
        self.release: asyncio.Event | None = None

    async def run(self, session, prompt, *, unattended=False, thinking=None, **kw):
        self.prompts.append(prompt)
        self.kwargs.append(
            {"session": session, "unattended": unattended, "thinking": thinking}
        )
        self.started.set()
        if self.release is not None:
            await self.release.wait()
        if isinstance(self.replies, Exception):
            raise self.replies
        index = len(self.prompts) - 1
        reply = (
            self.replies[index] if index < len(self.replies) else self.replies[-1]
        )
        if reply.startswith("!error "):
            yield Event("error", {"message": reply.removeprefix("!error ")})
            return
        yield Event("done", {"text": reply})


async def drive(replies, goal="tidy up", max_steps=3) -> tuple[Mission, FakeAgent]:
    agent = FakeAgent(replies)
    runner = MissionRunner(agent)
    mission = runner.start(goal, max_steps)
    await asyncio.gather(*runner.tasks.values())
    return missions.load(mission.id), agent


# --- stopping ----------------------------------------------------------------


async def test_a_mission_stops_when_the_agent_says_it_is_done():
    mission, agent = await drive([f"Bins are out. {DONE_MARKER}"])

    assert mission.status == "done"
    assert mission.steps == 1
    assert len(agent.prompts) == 1


async def test_the_marker_is_stripped_from_the_reported_result():
    """It is protocol, not content — it should never be spoken aloud or shown
    as the mission's answer."""
    mission, _ = await drive([f"All finished.\n{DONE_MARKER}"])

    assert mission.result == "All finished."
    assert DONE_MARKER not in mission.result


async def test_a_mission_keeps_going_until_the_marker_appears():
    mission, agent = await drive(
        ["step one done", "step two done", f"and now it is finished {DONE_MARKER}"]
    )

    assert mission.status == "done"
    assert mission.steps == 3
    assert len(agent.prompts) == 3


async def test_a_mission_that_never_finishes_runs_out_rather_than_forever():
    """The backstop. One turn costs one request per tool round, so a goal the
    agent will not let go of spends the whole rate limit unattended."""
    mission, agent = await drive(["still working"] * 5, max_steps=3)

    assert mission.status == "exhausted"
    assert mission.steps == 3
    assert len(agent.prompts) == 3


async def test_the_marker_is_matched_case_insensitively():
    mission, _ = await drive(["done here. mission complete"])
    assert mission.status == "done"


# --- the prompts -------------------------------------------------------------


async def test_the_first_step_carries_the_goal_and_the_budget():
    _, agent = await drive(["working", f"done {DONE_MARKER}"], goal="empty the bins")

    assert "empty the bins" in agent.prompts[0]
    assert "3 steps" in agent.prompts[0]


async def test_later_steps_do_not_repeat_the_goal_because_the_transcript_has_it():
    """Each step is a turn in one session, so the agent can see its own
    previous work. Restating the goal every time would waste the window it
    needs for that."""
    _, agent = await drive(["a", "b", f"c {DONE_MARKER}"], goal="empty the bins")

    assert "empty the bins" not in agent.prompts[1]
    assert "step 2" in agent.prompts[1]


async def test_the_final_step_asks_for_a_report_rather_than_more_work():
    _, agent = await drive(["a", "b", "c"], max_steps=3)

    assert "final step" in agent.prompts[-1]


async def test_every_step_shares_one_session_of_its_own():
    """A long autonomous run in the default session would push a person's
    conversation out of the 120-message window."""
    mission, agent = await drive(["a", f"b {DONE_MARKER}"])

    sessions = {k["session"] for k in agent.kwargs}
    assert sessions == {missions.session_for(mission.id)}
    assert "mission" in sessions.pop()


async def test_a_mission_runs_unattended_and_with_reasoning_on():
    """Unattended because nobody is there to approve an SMS; reasoning on
    because the latency costs nothing when nobody is waiting."""
    _, agent = await drive([f"done {DONE_MARKER}"])

    assert agent.kwargs[0]["unattended"] is True
    assert agent.kwargs[0]["thinking"] is True


# --- failure and cancellation ------------------------------------------------


async def test_an_agent_error_fails_the_mission_rather_than_looping_on_it():
    mission, _ = await drive(["!error rate limited"])

    assert mission.status == "failed"
    assert "rate limited" in mission.result


async def test_a_crash_is_recorded_instead_of_escaping_into_the_task():
    """Nothing awaits this task, so an exception would be swallowed by asyncio
    and the mission would sit at "running" forever."""
    agent = FakeAgent(RuntimeError("no network"))
    runner = MissionRunner(agent)
    mission = runner.start("something", 2)
    await asyncio.gather(*runner.tasks.values())

    stored = missions.load(mission.id)
    assert stored.status == "failed"
    assert "no network" in stored.result


async def test_a_running_mission_can_be_cancelled():
    agent = FakeAgent(["working"] * 5)
    agent.release = asyncio.Event()
    runner = MissionRunner(agent)
    mission = runner.start("long job", 5)

    await agent.started.wait()
    assert runner.cancel(mission.id) is True
    agent.release.set()
    await asyncio.gather(*runner.tasks.values(), return_exceptions=True)

    assert missions.load(mission.id).status == "cancelled"


async def test_cancelling_something_that_is_not_running_reports_so():
    runner = MissionRunner(FakeAgent([]))
    assert runner.cancel(999) is False


async def test_finished_missions_stop_being_tracked():
    agent = FakeAgent([f"done {DONE_MARKER}"])
    runner = MissionRunner(agent)
    runner.start("quick", 2)
    await asyncio.gather(*runner.tasks.values())

    assert runner.tasks == {}


# --- persistence -------------------------------------------------------------


async def test_a_mission_survives_being_reloaded():
    """Termux kills long-running processes, and a mission is exactly the thing
    that will be in flight when it happens."""
    mission, _ = await drive([f"done {DONE_MARKER}"], goal="water the plants")

    reloaded = missions.load(mission.id)
    assert reloaded.goal == "water the plants"
    assert reloaded.status == "done"


async def test_each_step_is_logged_as_it_happens():
    mission, _ = await drive(["found the bins", f"took them out {DONE_MARKER}"])

    assert len(mission.log) == 2
    assert "found the bins" in mission.log[0]


async def test_missions_are_listed_newest_first():
    await drive([f"a {DONE_MARKER}"], goal="first")
    await drive([f"b {DONE_MARKER}"], goal="second")

    assert [m.goal for m in missions.recent()] == ["second", "first"]


async def test_start_and_finish_reach_the_audit_log():
    await drive([f"done {DONE_MARKER}"])

    kinds = {e["kind"] for e in memory.recent_events()}
    assert {"mission_start", "mission_done"} <= kinds


async def test_running_out_of_steps_is_audited_distinctly_from_finishing():
    await drive(["nope"] * 3, max_steps=2)

    kinds = {e["kind"] for e in memory.recent_events()}
    assert "mission_exhausted" in kinds and "mission_done" not in kinds


def test_loading_a_mission_that_does_not_exist_returns_nothing():
    assert missions.load(4242) is None


def test_a_fresh_mission_is_not_finished():
    assert missions.create("x", 3).finished is False
