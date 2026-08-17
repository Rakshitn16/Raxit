"""The agent loop, driven by scripted completions.

Faking at the chunk level rather than stubbing `stream_deltas` means these
tests run the real streaming reassembly, the real message shapes, and the
real approval gate — the parts that break when the wire format shifts.
"""

from __future__ import annotations

import asyncio

import pytest

from conftest import empty_chunk, one_tool_call, reasoning_chunk, text_chunk, tool_chunk
from raxit import llm, memory, tools
from raxit.agent import SYSTEM_PROMPT, Agent
from raxit.config import settings
from raxit.tools.registry import obj, opt, tool


@pytest.fixture
def sandbox_tools(monkeypatch: pytest.MonkeyPatch):
    """Replace the real device tools with harmless fakes.

    Nothing here should reach `termux-*`; a test that shells out to a missing
    binary is testing the error path by accident.
    """
    registry: dict = {}
    monkeypatch.setattr(tools.registry, "REGISTRY", registry)
    monkeypatch.setattr(tools, "REGISTRY", registry)

    seen: list[tuple[str, dict]] = []

    @tool("battery", "Read the battery.", opt())
    def battery() -> str:
        seen.append(("battery", {}))
        return "83%, charging"

    @tool("send_sms", "Send an SMS.", obj(number={"type": "string"}), dangerous=True)
    def send_sms(number: str) -> str:
        seen.append(("send_sms", {"number": number}))
        return f"SMS sent to {number}."

    @tool("boom", "Fail.", opt())
    def boom() -> str:
        raise RuntimeError("termux-api not installed")

    @tool("slow", "Take a while.", opt())
    async def slow() -> str:
        await asyncio.sleep(0.05)
        seen.append(("slow", {}))
        return "slept"

    return seen


async def collect(agent: Agent, *args, **kwargs) -> list:
    return [event async for event in agent.run(*args, **kwargs)]


def kinds(events: list) -> list[str]:
    return [e.type for e in events]


def only(events: list, kind: str) -> list:
    return [e for e in events if e.type == kind]


# --- plain answers -----------------------------------------------------------


async def test_a_tool_less_answer_streams_text_then_finishes(scripted, sandbox_tools):
    scripted.add([text_chunk("83"), text_chunk("% left")]).install()

    events = await collect(Agent(), "s", "battery?")

    assert kinds(events) == ["text", "text", "done"]
    assert events[-1].data["text"] == "83% left"


async def test_reasoning_is_streamed_separately_from_the_answer(scripted, sandbox_tools):
    """Nemotron's scratchpad arrives on its own field. It goes to the UI's
    thinking pane, never into the spoken reply."""
    scripted.add(
        [reasoning_chunk("let me check"), text_chunk("83%")]
    ).install()

    events = await collect(Agent(), "s", "battery?")

    assert only(events, "thinking")[0].data["text"] == "let me check"
    assert only(events, "done")[0].data["text"] == "83%"


async def test_inline_think_blocks_are_stripped_from_the_final_answer(
    scripted, sandbox_tools
):
    """Models other than Nemotron inline their reasoning in the content
    stream. Speaking that aloud would read the scratchpad to the user."""
    scripted.add([text_chunk("<think>hmm</think>"), text_chunk("83%")]).install()

    assert only(await collect(Agent(), "s", "q"), "done")[0].data["text"] == "83%"


async def test_chunks_with_no_choices_are_ignored(scripted, sandbox_tools):
    scripted.add([empty_chunk(), text_chunk("hi"), empty_chunk()]).install()

    assert only(await collect(Agent(), "s", "q"), "done")[0].data["text"] == "hi"


# --- tool rounds -------------------------------------------------------------


async def test_a_tool_round_runs_the_tool_then_answers(scripted, sandbox_tools):
    scripted.add(one_tool_call("battery", "{}")).add([text_chunk("83%")]).install()

    events = await collect(Agent(), "s", "battery?")

    result = only(events, "tool_result")[0]
    assert result.data["name"] == "battery"
    assert result.data["output"] == "83%, charging"
    assert result.data["is_error"] is False
    assert only(events, "done")[0].data["text"] == "83%"
    assert sandbox_tools == [("battery", {})]


async def test_fragmented_tool_arguments_are_reassembled(scripted, sandbox_tools):
    """`arguments` arrives as JSON string fragments that are invalid on their
    own — the payload only exists once they are concatenated."""
    scripted.add(one_tool_call("send_sms", '{"number": "+919812345678"}'))
    scripted.add([text_chunk("sent")]).install()

    agent = Agent()
    await collect(agent, "s", "text him", approve=lambda name, payload: True)

    assert sandbox_tools == [("send_sms", {"number": "+919812345678"})]


async def test_several_tool_calls_in_one_turn_each_get_a_reply(
    scripted, sandbox_tools
):
    """The format permits several calls per assistant turn, and the API
    rejects the history unless every one has its own `tool` message."""
    scripted.add(
        [
            tool_chunk(0, call_id="c1", name="battery", arguments="{}"),
            tool_chunk(1, call_id="c2", name="slow", arguments="{}"),
        ]
    ).add([text_chunk("both done")]).install()

    events = await collect(Agent(), "s", "q")

    assert [e.data["name"] for e in only(events, "tool_result")] == ["battery", "slow"]
    replayed = scripted.calls[1]["messages"]
    tool_ids = [m["tool_call_id"] for m in replayed if m["role"] == "tool"]
    assert tool_ids == ["c1", "c2"]


async def test_tool_calls_in_one_turn_run_concurrently(scripted, sandbox_tools):
    scripted.add(
        [
            tool_chunk(0, call_id="c1", name="slow", arguments="{}"),
            tool_chunk(1, call_id="c2", name="slow", arguments="{}"),
            tool_chunk(2, call_id="c3", name="slow", arguments="{}"),
        ]
    ).add([text_chunk("done")]).install()

    loop = asyncio.get_running_loop()
    start = loop.time()
    await collect(Agent(), "s", "q")
    elapsed = loop.time() - start

    # Three 50ms tools: serial would be 150ms.
    assert elapsed < 0.12


async def test_a_failing_tool_comes_back_as_an_error_the_model_can_read(
    scripted, sandbox_tools
):
    scripted.add(one_tool_call("boom", "{}")).add([text_chunk("that failed")]).install()

    events = await collect(Agent(), "s", "q")

    result = only(events, "tool_result")[0]
    assert result.data["is_error"] is True
    assert "termux-api not installed" in result.data["output"]
    # And the failure is replayed to the model rather than hidden from it.
    replayed = scripted.calls[1]["messages"]
    assert "termux-api not installed" in replayed[-1]["content"]


async def test_an_unknown_tool_is_reported_rather_than_crashing_the_turn(
    scripted, sandbox_tools
):
    scripted.add(one_tool_call("teleport", "{}")).add([text_chunk("no")]).install()

    events = await collect(Agent(), "s", "q")

    result = only(events, "tool_result")[0]
    assert result.data["is_error"] is True
    assert "No such tool: teleport" in result.data["output"]


async def test_unparseable_arguments_are_reported_not_raised(scripted, sandbox_tools):
    """A truncated stream leaves `arguments` as half a JSON object. The model
    can recover from being told so; it cannot recover from a 500."""
    scripted.add(one_tool_call("battery", '{"broken": ')).add(
        [text_chunk("retrying")]
    ).install()

    result = only(await collect(Agent(), "s", "q"), "tool_result")[0]
    assert result.data["is_error"] is True
    assert "Could not parse arguments" in result.data["output"]
    assert sandbox_tools == []


async def test_non_object_arguments_are_rejected(scripted, sandbox_tools):
    scripted.add(one_tool_call("battery", "[1, 2]")).add([text_chunk("ok")]).install()

    result = only(await collect(Agent(), "s", "q"), "tool_result")[0]
    assert "must be a JSON object" in result.data["output"]


async def test_the_display_only_error_flag_is_not_replayed_to_the_model(
    scripted, sandbox_tools
):
    """`_error` drives the UI's red border. Sending it to the API would be an
    unrecognised key on a `tool` message."""
    scripted.add(one_tool_call("boom", "{}")).add([text_chunk("ok")]).install()

    await collect(Agent(), "s", "q")

    replayed = [m for m in scripted.calls[1]["messages"] if m["role"] == "tool"]
    assert replayed and all("_error" not in m for m in replayed)
    stored = [m for m in memory.load_messages("s") if m["role"] == "tool"]
    assert stored and all("_error" not in m for m in stored)


# --- the approval gate -------------------------------------------------------


async def test_a_dangerous_tool_asks_before_it_runs(scripted, sandbox_tools):
    scripted.add(one_tool_call("send_sms", '{"number": "+91"}'))
    scripted.add([text_chunk("sent")]).install()
    asked: list[tuple[str, dict]] = []

    async def approve(name: str, payload: dict) -> bool:
        asked.append((name, payload))
        return True

    await collect(Agent(), "s", "text him", approve=approve)

    assert asked == [("send_sms", {"number": "+91"})]
    assert sandbox_tools == [("send_sms", {"number": "+91"})]


async def test_a_refusal_becomes_an_error_result_rather_than_a_dead_turn(
    scripted, sandbox_tools
):
    """The model needs to hear "no" as a tool result so it can say so in
    words. Aborting the turn would leave the user staring at nothing."""
    scripted.add(one_tool_call("send_sms", '{"number": "+91"}'))
    scripted.add([text_chunk("I did not send it.")]).install()

    events = await collect(
        Agent(), "s", "text him", approve=lambda name, payload: False
    )

    result = only(events, "tool_result")[0]
    assert result.data["is_error"] is True
    assert "declined by the user" in result.data["output"]
    assert sandbox_tools == []
    assert only(events, "done")[0].data["text"] == "I did not send it."


async def test_a_dangerous_tool_with_no_approver_is_refused(scripted, sandbox_tools):
    """`/api/chat` has no channel to ask on. Defaulting to "allow" there would
    let a curl one-liner send SMS with nobody in the loop."""
    scripted.add(one_tool_call("send_sms", '{"number": "+91"}'))
    scripted.add([text_chunk("blocked")]).install()

    events = await collect(Agent(), "s", "text him")

    assert only(events, "tool_result")[0].data["is_error"] is True
    assert sandbox_tools == []


async def test_unattended_refusal_says_why_nobody_answered(scripted, sandbox_tools):
    """Wording matters: the model should report that it could not ask, not
    that the user said no — a 7am routine has no user to have said it."""
    scripted.add(one_tool_call("send_sms", '{"number": "+91"}'))
    scripted.add([text_chunk("blocked")]).install()

    events = await collect(Agent(), "s", "text him", unattended=True)

    assert "nobody is watching" in only(events, "tool_result")[0].data["output"]


async def test_a_blanket_approval_skips_the_prompt(scripted, sandbox_tools):
    scripted.add(one_tool_call("send_sms", '{"number": "+91"}'))
    scripted.add([text_chunk("sent")]).install()
    agent = Agent()
    agent.approved.add("send_sms")
    asked: list = []

    await collect(
        agent, "s", "text him", approve=lambda n, p: asked.append(n) or False
    )

    assert asked == []
    assert sandbox_tools == [("send_sms", {"number": "+91"})]


async def test_safe_tools_are_never_gated(scripted, sandbox_tools):
    scripted.add(one_tool_call("battery", "{}")).add([text_chunk("83%")]).install()
    asked: list = []

    await collect(Agent(), "s", "q", approve=lambda n, p: asked.append(n) or True)

    assert asked == []


# --- the round limit ---------------------------------------------------------


async def test_running_out_of_rounds_still_produces_an_answer(
    scripted, sandbox_tools, monkeypatch
):
    """Observed in the wild: the model burned every round improvising shell
    workarounds after one tool failed, and the turn returned nothing at all.
    Better to answer from what was gathered and say what is missing."""
    monkeypatch.setattr(settings, "max_tool_iterations", 2)
    scripted.add(one_tool_call("battery", "{}", "c1"))
    scripted.add(one_tool_call("battery", "{}", "c2"))
    scripted.add([text_chunk("Battery is 83%; I could not finish the rest.")])
    scripted.install()

    events = await collect(Agent(), "s", "q")

    assert only(events, "notice")[0].data["message"].startswith("Hit the 2-round")
    assert only(events, "done")[0].data["text"].startswith("Battery is 83%")


async def test_the_final_pass_offers_no_tools_so_it_cannot_loop_again(
    scripted, sandbox_tools, monkeypatch
):
    monkeypatch.setattr(settings, "max_tool_iterations", 1)
    scripted.add(one_tool_call("battery", "{}")).add([text_chunk("done")]).install()

    await collect(Agent(), "s", "q")

    assert scripted.calls[0]["tools"], "the normal round should offer tools"
    assert not scripted.calls[-1]["tools"], "the forced answer must not"


async def test_the_final_pass_tells_the_model_it_is_out_of_calls(
    scripted, sandbox_tools, monkeypatch
):
    monkeypatch.setattr(settings, "max_tool_iterations", 1)
    scripted.add(one_tool_call("battery", "{}")).add([text_chunk("done")]).install()

    await collect(Agent(), "s", "q")

    assert "run out of tool calls" in scripted.calls[-1]["messages"][-1]["content"]


# --- failures ----------------------------------------------------------------


async def test_a_provider_failure_ends_the_turn_with_an_error_event(
    scripted, sandbox_tools, monkeypatch
):
    async def explode(*args, **kwargs):
        raise llm.RateLimited("429 after retries")

    monkeypatch.setattr(llm, "stream_completion", explode)

    events = await collect(Agent(), "s", "q")

    assert kinds(events) == ["error"]
    assert "429 after retries" in events[0].data["message"]


async def test_a_provider_failure_during_the_final_pass_is_reported(
    scripted, sandbox_tools, monkeypatch
):
    monkeypatch.setattr(settings, "max_tool_iterations", 1)
    scripted.add(one_tool_call("battery", "{}")).install()
    real = llm.stream_completion
    calls = {"n": 0}

    async def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("connection reset")
        return await real(*args, **kwargs)

    monkeypatch.setattr(llm, "stream_completion", flaky)

    events = await collect(Agent(), "s", "q")

    assert kinds(events)[-1] == "error"
    assert "connection reset" in events[-1].data["message"]


async def test_the_stream_is_closed_even_when_the_turn_ends_early(
    scripted, sandbox_tools
):
    """An abandoned response body leaks a pooled connection. On a process that
    runs for weeks on a tablet, those accumulate until requests stall."""
    scripted.add([text_chunk("hi")]).install()

    await collect(Agent(), "s", "q")

    assert all(stream.closed for stream in scripted.streams)


# --- persistence -------------------------------------------------------------


async def test_the_whole_exchange_is_persisted_in_replayable_order(
    scripted, sandbox_tools
):
    scripted.add(one_tool_call("battery", "{}", "c1")).add(
        [text_chunk("83%")]
    ).install()

    await collect(Agent(), "s", "battery?")

    stored = memory.load_messages("s")
    assert [m["role"] for m in stored] == ["user", "assistant", "tool", "assistant"]
    assert stored[1]["tool_calls"][0]["id"] == "c1"
    assert stored[2]["tool_call_id"] == "c1"


async def test_an_earlier_turn_is_replayed_into_the_next_one(scripted, sandbox_tools):
    scripted.add([text_chunk("83%")]).add([text_chunk("still 83%")]).install()
    agent = Agent()

    await collect(agent, "s", "battery?")
    await collect(agent, "s", "and now?")

    replayed = scripted.calls[1]["messages"]
    assert [m["role"] for m in replayed] == [
        "system", "user", "assistant", "user",
    ]
    assert replayed[1]["content"] == "battery?"


async def test_other_sessions_are_not_replayed(scripted, sandbox_tools):
    scripted.add([text_chunk("a")]).add([text_chunk("b")]).install()
    agent = Agent()

    await collect(agent, "kitchen", "secret")
    await collect(agent, "study", "hello")

    contents = [m.get("content") for m in scripted.calls[1]["messages"]]
    assert "secret" not in contents


def test_the_system_prompt_forbids_answering_device_questions_from_memory():
    """This paragraph is load-bearing, not decoration.

    Measured against the live endpoint: a smaller model a few turns into a
    conversation stopped calling tools and answered from nothing — a battery
    percentage, a clock time months off, and an "I've stored that" for
    something it never stored. Deleting these sentences is how that behaviour
    comes back, so pin the rule rather than the wording.
    """
    prompt = SYSTEM_PROMPT.lower()
    assert "every turn, not just the first" in prompt
    assert "saying you did it is not doing it" in prompt


async def test_the_system_prompt_leads_and_names_the_owner(scripted, sandbox_tools):
    scripted.add([text_chunk("hi")]).install()

    await collect(Agent(), "s", "q")

    system = scripted.calls[0]["messages"][0]
    assert system["role"] == "system"
    assert settings.owner_name in system["content"]


async def test_tool_use_is_written_to_the_audit_log(scripted, sandbox_tools):
    scripted.add(one_tool_call("battery", "{}")).add([text_chunk("83%")]).install()

    await collect(Agent(), "s", "q")

    assert any(e["kind"] == "tool" for e in memory.recent_events())


async def test_a_tool_failure_is_written_to_the_audit_log(scripted, sandbox_tools):
    scripted.add(one_tool_call("boom", "{}")).add([text_chunk("ok")]).install()

    await collect(Agent(), "s", "q")

    assert any(e["kind"] == "tool_error" for e in memory.recent_events())


async def test_a_declined_tool_is_not_logged_as_having_run(scripted, sandbox_tools):
    scripted.add(one_tool_call("send_sms", '{"number": "+91"}'))
    scripted.add([text_chunk("ok")]).install()

    await collect(Agent(), "s", "q", approve=lambda n, p: False)

    assert not any(e["kind"] == "tool" for e in memory.recent_events())


# --- thinking ----------------------------------------------------------------


async def test_a_routine_waits_out_a_rate_limit_but_a_person_is_not_made_to(
    scripted, sandbox_tools
):
    """`patient` is the agent telling the provider layer whether anyone is
    listening — the one thing only the caller knows."""
    scripted.add([text_chunk("a")]).add([text_chunk("b")]).install()

    await collect(Agent(), "s", "q", unattended=True)
    await collect(Agent(), "s", "q")

    assert scripted.calls[0]["kwargs"]["patient"] is True
    assert scripted.calls[1]["kwargs"]["patient"] is False


async def test_the_forced_answer_inherits_the_turns_patience(
    scripted, sandbox_tools, monkeypatch
):
    monkeypatch.setattr(settings, "max_tool_iterations", 1)
    scripted.add(one_tool_call("battery", "{}")).add([text_chunk("done")]).install()

    await collect(Agent(), "s", "q", unattended=True)

    assert scripted.calls[-1]["kwargs"]["patient"] is True


async def test_a_routines_thinking_choice_reaches_the_provider(
    scripted, sandbox_tools
):
    """Routines opt into the reasoning pass individually — it roughly doubles
    latency, which is worth it for a morning brief and not for a sensor read."""
    scripted.add([text_chunk("hi")]).install()

    await collect(Agent(), "s", "q", thinking=True)

    assert scripted.calls[0]["kwargs"]["thinking"] is True
