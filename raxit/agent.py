"""The agent loop: the model plans, the tablet acts.

A hand-written tool loop rather than a framework, because two things have to
happen inline: every step is streamed to the web UI as it occurs, and dangerous
tools are intercepted *before* they execute.

The wire format is OpenAI's (NVIDIA NIM speaks it), which shapes the loop:
a tool round is an `assistant` message carrying `tool_calls`, followed by one
`tool` message per call. Those must stay adjacent and complete or the API
rejects the history.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable

from . import llm, memory, tools
from .config import settings

log = logging.getLogger("raxit.agent")

SYSTEM_PROMPT = """\
You are Raxit, a voice-and-text assistant living on {owner}'s Samsung Galaxy \
Tab A9+. You run as a background process on the tablet itself, which is what \
lets you act on it rather than only talk about it.

You have real control of the device: speech, notifications, SMS, calls, \
clipboard, camera, torch, location, and a restricted shell. Prefer doing the \
thing over describing how to do it. When a request maps onto a tool, call it.

Ground yourself before acting. You have no innate sense of the current time, \
the device's state, or what you were told last week — call `now`, `battery` \
and `recall` rather than guessing at any of them.

You persist across sessions. When you learn something durable about {owner} — \
a preference, a schedule, a name, a recurring annoyance — store it with \
`remember` so the next conversation starts warmer than this one did.

You often run unattended, fired by a routine at 7am with nobody watching. \
Nobody can answer a clarifying question in that mode, so make the reasonable \
call and deliver the result through `notify` or `speak`. Save questions for \
when someone is actually in the conversation with you.

When a tool fails, report the failure and carry on with the rest of the \
request. Do not improvise a workaround through the shell — a tool that is \
unavailable stays unavailable, and hunting for a back door burns the request \
budget without answering the question. One retry is reasonable if the error \
reads as transient; a second is not.

Keep spoken replies to a sentence or two — they are heard, not skimmed, and a \
paragraph read aloud is punishing. Written replies can be longer when the \
substance earns it. Lead with the answer; put the reasoning after it.

Deliver what was asked at the scope intended. If you think the request is \
mistaken or a better approach exists, say so in a sentence and carry on with \
what was asked — don't quietly widen it or substitute your own plan.
"""


@dataclass
class Event:
    """One thing that happened, streamed to whoever is watching."""

    type: str  # text | thinking | tool_result | done | error
    data: dict[str, Any]


class Agent:
    def __init__(self) -> None:
        self.system = SYSTEM_PROMPT.format(owner=settings.owner_name)
        # Tools the user has blanket-approved for this process lifetime.
        self.approved: set[str] = set()

    async def run(
        self,
        session: str,
        user_input: str,
        *,
        unattended: bool = False,
        thinking: bool | None = None,
        approve: Callable[[str, dict[str, Any]], Any] | None = None,
    ) -> AsyncIterator[Event]:
        """Drive one exchange to completion, yielding events as they occur.

        `approve` is consulted before any tool marked dangerous. A refusal
        becomes an error tool result, which the model reads and routes around
        rather than retrying blindly.
        """
        memory.append_message(session, user_input_message(user_input))
        history = memory.load_messages(session)
        messages = [{"role": "system", "content": self.system}, *history]

        for _ in range(settings.max_tool_iterations):
            collector = llm.StreamCollector()
            try:
                async for delta in llm.stream_deltas(
                    messages, tools.definitions(), collector, thinking=thinking
                ):
                    if "reasoning" in delta:
                        yield Event("thinking", {"text": delta["reasoning"]})
                    if "text" in delta:
                        yield Event("text", {"text": delta["text"]})
            except Exception as exc:
                log.exception("completion failed")
                yield Event("error", {"message": str(exc)})
                return

            assistant = collector.assistant_message()
            memory.append_message(session, assistant)
            messages.append(assistant)

            calls = assistant.get("tool_calls")
            if not calls:
                yield Event("done", {"text": llm.strip_reasoning(collector.text)})
                return

            results = await self._execute(calls, unattended=unattended, approve=approve)
            for call, result in zip(calls, results):
                yield Event(
                    "tool_result",
                    {
                        "name": call["function"]["name"],
                        "input": call["function"]["arguments"],
                        "output": result["content"],
                        "is_error": result.pop("_error", False),
                    },
                )
                memory.append_message(session, result)
                messages.append(result)

        # Out of tool rounds. Rather than dropping everything on the floor,
        # make one final pass with no tools offered, so the user still gets an
        # answer built from whatever the loop did manage to gather.
        async for event in self._forced_answer(session, messages, thinking):
            yield event

    async def _forced_answer(
        self,
        session: str,
        messages: list[dict[str, Any]],
        thinking: bool | None,
    ) -> AsyncIterator[Event]:
        yield Event(
            "notice",
            {
                "message": f"Hit the {settings.max_tool_iterations}-round tool "
                "limit; summarising what I have."
            },
        )
        messages = [
            *messages,
            {
                "role": "user",
                "content": (
                    "You have run out of tool calls for this turn. Answer now "
                    "using only what you already gathered, and say plainly "
                    "which parts you could not complete."
                ),
            },
        ]
        collector = llm.StreamCollector()
        try:
            async for delta in llm.stream_deltas(
                messages, None, collector, thinking=thinking
            ):
                if "text" in delta:
                    yield Event("text", {"text": delta["text"]})
        except Exception as exc:
            yield Event("error", {"message": str(exc)})
            return

        final = collector.assistant_message()
        memory.append_message(session, final)
        yield Event("done", {"text": llm.strip_reasoning(collector.text)})

    async def _execute(
        self,
        calls: list[dict[str, Any]],
        *,
        unattended: bool,
        approve: Callable[[str, dict[str, Any]], Any] | None,
    ) -> list[dict[str, Any]]:
        """Run every tool call from one assistant turn, concurrently.

        Nemotron usually emits calls one at a time, but the format permits
        several per turn and each needs its own `tool` reply, so gather rather
        than assume a single call.
        """

        async def one(call: dict[str, Any]) -> dict[str, Any]:
            name = call["function"]["name"]
            call_id = call["id"]

            try:
                payload = json.loads(call["function"]["arguments"] or "{}")
            except json.JSONDecodeError as exc:
                return _tool_result(
                    call_id, f"Could not parse arguments as JSON: {exc}", error=True
                )
            if not isinstance(payload, dict):
                return _tool_result(
                    call_id, "Arguments must be a JSON object.", error=True
                )

            entry = tools.REGISTRY.get(name)
            if entry is None:
                return _tool_result(call_id, f"No such tool: {name}", error=True)

            if entry.dangerous and name not in self.approved:
                allowed = False
                if approve is not None:
                    allowed = await _maybe_await(approve(name, payload))
                if not allowed:
                    reason = (
                        "blocked: this tool needs a human and nobody is watching"
                        if unattended
                        else "declined by the user"
                    )
                    return _tool_result(call_id, f"{name} {reason}.", error=True)

            try:
                output = await tools.invoke(name, payload)
                memory.log_event("tool", f"{name} {json.dumps(payload)[:200]}")
                return _tool_result(call_id, output)
            except Exception as exc:  # surfaced to the model, not swallowed
                memory.log_event("tool_error", f"{name}: {exc}")
                return _tool_result(call_id, f"{type(exc).__name__}: {exc}", error=True)

        return list(await asyncio.gather(*(one(c) for c in calls)))


def user_input_message(text: str) -> dict[str, Any]:
    return {"role": "user", "content": text}


def _tool_result(call_id: str, content: str, *, error: bool = False) -> dict[str, Any]:
    # `_error` is a local display flag, popped before the message is replayed.
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": content,
        "_error": error,
    }


async def _maybe_await(value: Any) -> Any:
    if asyncio.iscoroutine(value):
        return await value
    return bool(value)
