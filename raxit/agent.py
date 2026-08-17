"""The agent loop: Claude plans, the tablet acts.

This is a hand-written tool loop rather than the SDK's tool runner, because
every step has to be streamed to the web UI as it happens and dangerous tools
have to be intercepted *before* they execute. The runner's per-turn hooks could
cover the second half, but the loop stays explicit so the event stream and the
approval gate live in one readable place.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable

import anthropic

from . import memory, tools
from .config import settings

SYSTEM_PROMPT = """\
You are Raxit, a voice-and-text assistant living on {owner}'s Samsung Galaxy \
Tab A9+. You run as a background process on the tablet itself, which is what \
lets you act on it rather than only talk about it.

You have real control of the device: speech, notifications, SMS, calls, \
clipboard, camera, torch, location, and a restricted shell. Prefer doing the \
thing over describing how to do it. When a request maps onto a tool, call it.

Ground yourself before acting. You have no innate sense of the current time, \
the device's state, or what you were told last week — call `now`, `battery`, \
and `recall` rather than guessing at any of them.

You persist across sessions. When you learn something durable about {owner} — \
a preference, a schedule, a name, a recurring annoyance — store it with \
`remember` so the next conversation starts warmer than this one did.

You often run unattended, fired by a routine at 7am with nobody watching. In \
that mode nobody can answer a clarifying question, so make the reasonable call \
and deliver the result through `notify` or `speak`. Save questions for when \
someone is actually in the conversation with you.

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

    type: str  # text | thinking | tool_use | tool_result | done | error
    data: dict[str, Any]


class ApprovalRequired(Exception):
    """Raised when a dangerous tool needs a human before it runs."""

    def __init__(self, tool_name: str, payload: dict[str, Any]) -> None:
        super().__init__(f"{tool_name} requires approval")
        self.tool_name = tool_name
        self.payload = payload


class Agent:
    def __init__(self, client: anthropic.AsyncAnthropic | None = None) -> None:
        self.client = client or anthropic.AsyncAnthropic()
        self.system = SYSTEM_PROMPT.format(owner=settings.owner_name)
        # Tool names the user has pre-approved for this process lifetime.
        self.approved: set[str] = set()

    async def run(
        self,
        session: str,
        user_input: str,
        *,
        effort: str | None = None,
        unattended: bool = False,
        approve: Callable[[str, dict[str, Any]], bool] | None = None,
    ) -> AsyncIterator[Event]:
        """Drive one exchange to completion, yielding events as they occur.

        `approve` is consulted before any tool marked dangerous. Returning
        False turns into an error tool_result, which Claude reads and routes
        around rather than retrying blindly.
        """
        memory.append_message(session, "user", user_input)
        messages = memory.load_messages(session)

        for _ in range(settings.max_tool_iterations):
            async with self.client.messages.stream(
                model=settings.model,
                max_tokens=settings.max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": self.system,
                        # The system prompt and tool schemas are byte-stable
                        # across every turn, so cache them once and read them
                        # back for a tenth of the price on each subsequent call.
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                thinking={"type": "adaptive", "display": "summarized"},
                output_config={"effort": effort or settings.effort},
                tools=tools.definitions(),
                messages=messages,
            ) as stream:
                async for chunk in stream:
                    if chunk.type != "content_block_delta":
                        continue
                    if chunk.delta.type == "text_delta":
                        yield Event("text", {"text": chunk.delta.text})
                    elif chunk.delta.type == "thinking_delta":
                        yield Event("thinking", {"text": chunk.delta.thinking})

                response = await stream.get_final_message()

            memory.append_message(session, "assistant", response.content)
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "refusal":
                yield Event("error", {"message": "The model declined this request."})
                return

            calls = [b for b in response.content if b.type == "tool_use"]
            if not calls:
                yield Event("done", {"text": _text_of(response)})
                return

            results = await self._execute(calls, unattended=unattended, approve=approve)
            for call, result in zip(calls, results):
                yield Event(
                    "tool_result",
                    {
                        "name": call.name,
                        "input": call.input,
                        "output": result["content"],
                        "is_error": result.get("is_error", False),
                    },
                )

            memory.append_message(session, "user", results)
            messages.append({"role": "user", "content": results})

        yield Event(
            "error",
            {"message": f"Stopped after {settings.max_tool_iterations} tool rounds."},
        )

    async def _execute(
        self,
        calls: list[Any],
        *,
        unattended: bool,
        approve: Callable[[str, dict[str, Any]], bool] | None,
    ) -> list[dict[str, Any]]:
        """Run every tool call in one assistant turn, concurrently.

        Claude may request several tools at once; the API requires all of their
        results back in a single user message, so gather rather than serialize.
        """

        async def one(call: Any) -> dict[str, Any]:
            entry = tools.REGISTRY.get(call.name)
            if entry is None:
                return _result(call.id, f"No such tool: {call.name}", error=True)

            if entry.dangerous and call.name not in self.approved:
                allowed = False
                if approve is not None:
                    allowed = await _maybe_await(approve(call.name, dict(call.input)))
                if not allowed:
                    reason = (
                        "declined by the user"
                        if not unattended
                        else "blocked: this tool needs a human and nobody is watching"
                    )
                    return _result(call.id, f"{call.name} {reason}.", error=True)

            try:
                output = await tools.invoke(call.name, dict(call.input))
                memory.log_event("tool", f"{call.name} {json.dumps(dict(call.input))[:200]}")
                return _result(call.id, output)
            except Exception as exc:  # surfaced to Claude, not swallowed
                memory.log_event("tool_error", f"{call.name}: {exc}")
                return _result(call.id, f"{type(exc).__name__}: {exc}", error=True)

        return list(await asyncio.gather(*(one(c) for c in calls)))


def _result(tool_use_id: str, content: str, *, error: bool = False) -> dict[str, Any]:
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
    }
    if error:
        block["is_error"] = True
    return block


def _text_of(response: Any) -> str:
    return "".join(b.text for b in response.content if b.type == "text").strip()


async def _maybe_await(value: Any) -> Any:
    if asyncio.iscoroutine(value):
        return await value
    return value
