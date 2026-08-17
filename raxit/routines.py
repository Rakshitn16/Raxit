"""Routines: things Raxit does on a schedule, without being asked.

A routine is a YAML file in `routines/` holding a cron expression and a prompt.
When it fires, the prompt is handed to the agent in unattended mode — the agent
plans and picks its own tools, so a routine reads like an instruction to a
person rather than a script.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from . import memory
from .agent import Agent
from .config import ROUTINES_DIR, settings

log = logging.getLogger("raxit.routines")


@dataclass
class Routine:
    name: str
    prompt: str
    cron: str
    enabled: bool = True
    # Nemotron's reasoning pass: worth it for a routine that plans several
    # steps, wasted latency on one that just reads a sensor.
    thinking: bool = True
    session: str = "routines"

    @classmethod
    def from_file(cls, path: Path) -> Routine:
        raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
        missing = {"name", "prompt", "cron"} - raw.keys()
        if missing:
            raise ValueError(f"{path.name} is missing: {', '.join(sorted(missing))}")
        return cls(
            name=raw["name"],
            prompt=raw["prompt"],
            cron=raw["cron"],
            enabled=raw.get("enabled", True),
            thinking=raw.get("thinking", True),
            session=raw.get("session", "routines"),
        )


def load_all(directory: Path | None = None) -> list[Routine]:
    # Resolved per call rather than bound as a default, so the directory can
    # be repointed after import.
    directory = ROUTINES_DIR if directory is None else directory
    routines: list[Routine] = []
    for path in sorted(directory.glob("*.y*ml")):
        try:
            routines.append(Routine.from_file(path))
        except Exception as exc:
            log.error("skipping %s: %s", path.name, exc)
    return routines


class RoutineRunner:
    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self.scheduler = AsyncIOScheduler(timezone=settings.timezone)
        self.routines: dict[str, Routine] = {}

    def load(self) -> None:
        """(Re)read the routines directory and rebuild the schedule."""
        self.scheduler.remove_all_jobs()
        self.routines.clear()
        for routine in load_all():
            self.routines[routine.name] = routine
            if not routine.enabled:
                continue
            self.scheduler.add_job(
                self.fire,
                CronTrigger.from_crontab(routine.cron, timezone=settings.timezone),
                args=[routine.name],
                id=routine.name,
                # A slow routine shouldn't stack up behind itself, and a tablet
                # that was asleep at 7am should not fire five catch-up runs.
                max_instances=1,
                coalesce=True,
                misfire_grace_time=600,
            )
        log.info("loaded %d routines (%d scheduled)", len(self.routines),
                 len(self.scheduler.get_jobs()))

    async def fire(self, name: str) -> str:
        """Run one routine to completion and return its closing text."""
        routine = self.routines.get(name)
        if routine is None:
            raise KeyError(name)

        memory.log_event("routine_start", name)
        final = ""
        try:
            async for event in self.agent.run(
                routine.session,
                routine.prompt,
                thinking=routine.thinking,
                unattended=True,
            ):
                if event.type == "done":
                    final = event.data["text"]
                elif event.type == "error":
                    memory.log_event("routine_error", f"{name}: {event.data['message']}")
        except Exception as exc:
            log.exception("routine %s failed", name)
            memory.log_event("routine_error", f"{name}: {exc}")
            return f"failed: {exc}"

        memory.log_event("routine_done", f"{name}: {final[:200]}")
        return final

    def start(self) -> None:
        self.load()
        self.scheduler.start()

    def stop(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)

    def describe(self) -> list[dict[str, Any]]:
        # Jobs added before the scheduler starts are "pending" and carry no
        # next_run_time attribute at all, so read it defensively.
        jobs = {j.id: j for j in self.scheduler.get_jobs()}
        return [
            {
                "name": r.name,
                "cron": r.cron,
                "enabled": r.enabled,
                "prompt": r.prompt,
                "next_run": _next_run(jobs.get(r.name)),
            }
            for r in self.routines.values()
        ]


def _next_run(job: Any) -> str | None:
    when = getattr(job, "next_run_time", None)
    return when.isoformat() if when else None


async def _demo() -> None:  # pragma: no cover - manual smoke test
    runner = RoutineRunner(Agent())
    runner.load()
    for entry in runner.describe():
        print(entry)


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(_demo())
