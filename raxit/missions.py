"""Agent mode: a goal pursued across many turns, without being driven.

A chat turn is bounded — the agent gets a handful of tool rounds and then has
to answer, which is right when somebody is waiting for a reply. A mission is
the other shape: "keep working on this until it is done", where done is
something the agent decides rather than the clock.

The loop is deliberately dull. Each step is one ordinary agent turn in a
session of its own, so the transcript accumulates and the next step sees what
the last one did — no separate scratchpad, no bespoke state machine. What
makes it a mission rather than a conversation is that the prompts come from
here instead of a person, and that it stops on its own.

Three things end a mission, and all three are recorded:

* the agent says so, by ending a reply with MISSION COMPLETE
* the step budget runs out, which is the backstop against a goal that cannot
  be finished and an agent too willing to keep trying
* somebody cancels it

Missions run unattended, so `send_sms` and `call` are refused inside one for
the same reason they are refused in a routine. `ask_user` is the exception
worth knowing about: it puts a dialog on the tablet, so a mission *can* get a
human decision when it genuinely needs one — as long as somebody is near the
device to answer.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from . import memory
from .agent import Agent

log = logging.getLogger("raxit.missions")

DONE_MARKER = "MISSION COMPLETE"

SCHEMA = """
CREATE TABLE IF NOT EXISTS missions (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    goal      TEXT NOT NULL,
    status    TEXT NOT NULL,          -- running | done | exhausted | failed | cancelled
    steps     INTEGER NOT NULL DEFAULT 0,
    max_steps INTEGER NOT NULL,
    result    TEXT NOT NULL DEFAULT '',
    log       TEXT NOT NULL DEFAULT '[]',
    created   REAL NOT NULL,
    updated   REAL NOT NULL
);
"""

FIRST_STEP = """\
You are working on a goal on your own, with nobody watching the conversation.

GOAL: {goal}

Work on it now. Use your tools — do the thing rather than describing a plan \
for it. You have {budget} steps in total; this is step 1, and each step is one \
reply from you, so do as much per step as you sensibly can.

When the goal is genuinely finished, end your reply with the single line \
{marker}. Do not write that line while work remains — it stops the mission, \
and a mission stopped early looks identical to one that succeeded.

If the goal turns out to be impossible or wrong, say why plainly, then end \
with {marker} rather than burning the remaining steps.\
"""

NEXT_STEP = """\
Continue the goal. This is step {step} of {budget}.

Look at what you have already done above, then take the next step — do not \
repeat work that is already finished, and do not restate the plan.

End your reply with {marker} when the goal is complete.\
"""

LAST_STEP = """\
This is your final step ({budget} of {budget}). Finish what you can now and \
then report, plainly, what is done and what is not. End with {marker}.\
"""


@dataclass
class Mission:
    id: int
    goal: str
    status: str
    steps: int
    max_steps: int
    result: str = ""
    log: list[str] = field(default_factory=list)

    @property
    def finished(self) -> bool:
        return self.status != "running"

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "goal": self.goal,
            "status": self.status,
            "steps": self.steps,
            "max_steps": self.max_steps,
            "result": self.result,
            "log": self.log,
        }


def init() -> None:
    with memory.connect() as conn:
        conn.executescript(SCHEMA)


def create(goal: str, max_steps: int = 6) -> Mission:
    now = time.time()
    with memory.connect() as conn:
        cur = conn.execute(
            "INSERT INTO missions (goal, status, max_steps, created, updated)"
            " VALUES (?,?,?,?,?)",
            (goal, "running", max_steps, now, now),
        )
        mission_id = int(cur.lastrowid or 0)
    memory.log_event("mission_start", f"{mission_id}: {goal[:150]}")
    return Mission(mission_id, goal, "running", 0, max_steps)


def load(mission_id: int) -> Mission | None:
    with memory.connect() as conn:
        row = conn.execute(
            "SELECT * FROM missions WHERE id=?", (mission_id,)
        ).fetchone()
    return _from_row(row) if row else None


def recent(limit: int = 20) -> list[Mission]:
    with memory.connect() as conn:
        rows = conn.execute(
            "SELECT * FROM missions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_from_row(r) for r in rows]


def _from_row(row: Any) -> Mission:
    return Mission(
        id=row["id"],
        goal=row["goal"],
        status=row["status"],
        steps=row["steps"],
        max_steps=row["max_steps"],
        result=row["result"],
        log=json.loads(row["log"] or "[]"),
    )


def _save(mission: Mission) -> None:
    with memory.connect() as conn:
        conn.execute(
            "UPDATE missions SET status=?, steps=?, result=?, log=?, updated=?"
            " WHERE id=?",
            (
                mission.status,
                mission.steps,
                mission.result,
                json.dumps(mission.log[-40:]),
                time.time(),
                mission.id,
            ),
        )


def session_for(mission_id: int) -> str:
    """Missions get their own transcript, so a long autonomous run does not
    push a person's conversation out of the 120-message window."""
    return f"mission:{mission_id}"


class MissionRunner:
    """Starts missions and keeps track of the ones in flight."""

    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self.tasks: dict[int, asyncio.Task[None]] = {}

    def start(self, goal: str, max_steps: int = 6) -> Mission:
        mission = create(goal, max_steps)
        task = asyncio.create_task(self._drive(mission.id))
        self.tasks[mission.id] = task
        task.add_done_callback(lambda _: self.tasks.pop(mission.id, None))
        return mission

    def cancel(self, mission_id: int) -> bool:
        task = self.tasks.get(mission_id)
        if task is None:
            return False
        task.cancel()
        return True

    async def _drive(self, mission_id: int) -> None:
        mission = load(mission_id)
        if mission is None:  # pragma: no cover - only if the row vanished
            return
        session = session_for(mission_id)
        try:
            while mission.steps < mission.max_steps:
                step = mission.steps + 1
                reply = await self._one_step(mission, session, step)
                mission.steps = step
                mission.log.append(f"step {step}: {reply[:300]}")
                mission.result = reply

                if DONE_MARKER in reply.upper():
                    mission.result = _strip_marker(reply)
                    mission.status = "done"
                    _save(mission)
                    memory.log_event("mission_done", f"{mission_id}: {step} steps")
                    return
                _save(mission)

            mission.status = "exhausted"
            _save(mission)
            memory.log_event(
                "mission_exhausted", f"{mission_id}: {mission.max_steps} steps"
            )
        except asyncio.CancelledError:
            mission.status = "cancelled"
            _save(mission)
            memory.log_event("mission_cancelled", str(mission_id))
            raise
        except Exception as exc:
            log.exception("mission %s failed", mission_id)
            mission.status = "failed"
            mission.result = f"{type(exc).__name__}: {exc}"
            _save(mission)
            memory.log_event("mission_failed", f"{mission_id}: {exc}")

    async def _one_step(self, mission: Mission, session: str, step: int) -> str:
        prompt = _prompt_for(mission, step)
        final = ""
        async for event in self.agent.run(
            session,
            prompt,
            unattended=True,
            # A mission plans; the extra latency costs nothing when nobody is
            # waiting on the reply.
            thinking=True,
        ):
            if event.type == "done":
                final = event.data["text"]
            elif event.type == "error":
                raise RuntimeError(event.data["message"])
        return final


def _prompt_for(mission: Mission, step: int) -> str:
    if step == 1:
        return FIRST_STEP.format(
            goal=mission.goal, budget=mission.max_steps, marker=DONE_MARKER
        )
    if step >= mission.max_steps:
        return LAST_STEP.format(budget=mission.max_steps, marker=DONE_MARKER)
    return NEXT_STEP.format(
        step=step, budget=mission.max_steps, marker=DONE_MARKER
    )


def _strip_marker(text: str) -> str:
    # The marker is protocol, not content — it should never be spoken aloud
    # or shown as the mission's answer.
    lines = [ln for ln in text.splitlines() if DONE_MARKER not in ln.upper()]
    return "\n".join(lines).strip()
