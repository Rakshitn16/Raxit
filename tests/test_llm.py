"""Stream reassembly and provider-error handling.

`StreamCollector` is the piece most likely to break when a provider changes
its chunking, and its failures are silent — a tool call assembled wrong just
looks like the model refusing to use tools.
"""

from __future__ import annotations

import re

import openai
import pytest

from conftest import empty_chunk, reasoning_chunk, text_chunk, tool_chunk
from raxit import llm
from raxit.config import settings
from raxit.llm import RateLimited, StreamCollector, strip_reasoning


def collect(chunks: list) -> StreamCollector:
    collector = StreamCollector()
    for chunk in chunks:
        collector.add(chunk)
    return collector


# --- text --------------------------------------------------------------------


def test_text_chunks_accumulate_and_are_yielded_for_live_display():
    collector = StreamCollector()
    assert collector.add(text_chunk("Hel")) == {"text": "Hel"}
    assert collector.add(text_chunk("lo")) == {"text": "lo"}
    assert collector.text == "Hello"


def test_a_chunk_with_no_choices_yields_nothing():
    assert StreamCollector().add(empty_chunk()) == {}


def test_an_empty_content_delta_is_not_reported_as_a_change():
    assert StreamCollector().add(text_chunk("")) == {}


# --- reasoning ---------------------------------------------------------------


def test_reasoning_accumulates_on_its_own_field():
    collector = collect([reasoning_chunk("think"), text_chunk("answer")])
    assert collector.reasoning == "think"
    assert collector.text == "answer"


def test_reasoning_is_not_replayed_to_the_model():
    """It is display-only. Echoing it back would spend context on every
    subsequent turn re-reading the last turn's scratchpad."""
    collector = collect([reasoning_chunk("long scratchpad"), text_chunk("83%")])
    message = collector.assistant_message()
    assert message == {"role": "assistant", "content": "83%"}


def test_strip_reasoning_removes_inline_think_blocks():
    assert strip_reasoning("<think>hmm</think>Answer") == "Answer"
    assert strip_reasoning("<THINK>a\nb</THINK>\n\nAnswer") == "Answer"
    assert strip_reasoning("<think>one</think>A<think>two</think>B") == "AB"


def test_strip_reasoning_leaves_ordinary_text_alone():
    assert strip_reasoning("  Answer  ") == "Answer"
    assert strip_reasoning("") == ""
    assert strip_reasoning(None) == ""


def test_an_unclosed_think_block_is_left_intact():
    # Better a visible scratchpad than an answer silently swallowed whole.
    assert strip_reasoning("<think>truncated") == "<think>truncated"


# --- tool calls --------------------------------------------------------------


def test_a_tool_call_is_reassembled_from_its_fragments():
    """The first chunk carries id and name; `arguments` then arrives as JSON
    string fragments that parse only once joined."""
    collector = collect(
        [
            tool_chunk(0, call_id="call_1", name="send_sms", arguments=""),
            tool_chunk(0, arguments='{"number":'),
            tool_chunk(0, arguments=' "+91", "mes'),
            tool_chunk(0, arguments='sage": "hi"}'),
        ]
    )
    call = collector.tool_calls()[0]
    assert call["id"] == "call_1"
    assert call["type"] == "function"
    assert call["function"]["name"] == "send_sms"
    assert call["function"]["arguments"] == '{"number": "+91", "message": "hi"}'


def test_interleaved_calls_are_kept_apart_by_index():
    """Two calls in one turn arrive interleaved. Appending by arrival order
    would splice one call's arguments into the other's."""
    collector = collect(
        [
            tool_chunk(0, call_id="c1", name="battery", arguments="{"),
            tool_chunk(1, call_id="c2", name="notify", arguments='{"title":'),
            tool_chunk(0, arguments="}"),
            tool_chunk(1, arguments=' "hi"}'),
        ]
    )
    calls = collector.tool_calls()
    assert [c["id"] for c in calls] == ["c1", "c2"]
    assert calls[0]["function"]["arguments"] == "{}"
    assert calls[1]["function"]["arguments"] == '{"title": "hi"}'


def test_calls_come_back_in_index_order_whatever_order_they_streamed_in():
    collector = collect(
        [
            tool_chunk(2, call_id="c3", name="c", arguments="{}"),
            tool_chunk(0, call_id="c1", name="a", arguments="{}"),
            tool_chunk(1, call_id="c2", name="b", arguments="{}"),
        ]
    )
    assert [c["function"]["name"] for c in collector.tool_calls()] == ["a", "b", "c"]


def test_a_call_with_no_arguments_gets_an_empty_object():
    """A bare `""` is not valid JSON; the agent parses this string directly."""
    collector = collect([tool_chunk(0, call_id="c1", name="battery", arguments="")])
    assert collector.tool_calls()[0]["function"]["arguments"] == "{}"


def test_a_call_that_never_received_an_id_still_gets_one():
    """Every `tool` reply must carry a matching id, so a missing one has to be
    synthesised rather than left blank."""
    collector = collect([tool_chunk(0, name="battery", arguments="{}")])
    assert collector.tool_calls()[0]["id"] == "call_0"


def test_a_fragment_that_never_named_a_tool_is_dropped():
    # A truncated stream can leave a slot with arguments and no name; there is
    # nothing to dispatch it to.
    collector = collect([tool_chunk(0, arguments='{"a": 1}')])
    assert collector.tool_calls() == []


def test_an_assistant_turn_with_tool_calls_keeps_both_parts():
    collector = collect(
        [
            text_chunk("Checking."),
            tool_chunk(0, call_id="c1", name="battery", arguments="{}"),
        ]
    )
    message = collector.assistant_message()
    assert message["content"] == "Checking."
    assert message["tool_calls"][0]["id"] == "c1"


def test_content_is_null_rather_than_empty_when_the_turn_is_only_a_tool_call():
    message = collect(
        [tool_chunk(0, call_id="c1", name="battery", arguments="{}")]
    ).assistant_message()
    assert message["content"] is None


def test_a_turn_with_no_tool_calls_omits_the_key_entirely():
    assert "tool_calls" not in collect([text_chunk("hi")]).assistant_message()


# --- request shaping ---------------------------------------------------------


class FakeCompletions:
    def __init__(self, outcomes: list) -> None:
        self.outcomes = outcomes
        self.seen: list[dict] = []

    async def create(self, **kwargs):
        self.seen.append(kwargs)
        outcome = self.outcomes.pop(0) if self.outcomes else "ok"
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch):
    def install(outcomes: list) -> FakeCompletions:
        completions = FakeCompletions(outcomes)
        client = type(
            "C", (), {"chat": type("Chat", (), {"completions": completions})()}
        )()
        monkeypatch.setattr(llm, "client", lambda: client)
        monkeypatch.setattr(llm.asyncio, "sleep", _no_sleep)
        return completions

    return install


async def _no_sleep(delay: float) -> None:
    return None


def rate_limit_error() -> openai.RateLimitError:
    return openai.RateLimitError("429", response=_response(429), body=None)


def _response(status: int):
    import httpx

    return httpx.Response(status, request=httpx.Request("POST", "http://x"))


async def test_a_request_carries_the_configured_model_and_sampling(fake_client):
    completions = fake_client(["stream"])

    await llm.stream_completion([{"role": "user", "content": "hi"}])

    sent = completions.seen[0]
    assert sent["model"] == settings.model
    assert sent["stream"] is True
    assert sent["temperature"] == settings.temperature


async def test_tools_are_only_sent_when_there_are_some(fake_client):
    completions = fake_client(["a", "b"])

    await llm.stream_completion([], tools=[{"type": "function"}])
    await llm.stream_completion([], tools=None)

    assert completions.seen[0]["tool_choice"] == "auto"
    assert "tools" not in completions.seen[1]


async def test_an_empty_tool_list_is_treated_as_no_tools(fake_client):
    """The forced final pass passes `None`, but an empty list would otherwise
    send `tools: []`, which some endpoints reject."""
    completions = fake_client(["a"])

    await llm.stream_completion([], tools=[])

    assert "tools" not in completions.seen[0]


async def test_the_per_call_thinking_flag_overrides_the_global_default(fake_client):
    completions = fake_client(["a", "b"])

    await llm.stream_completion([], thinking=True)
    await llm.stream_completion([], thinking=False)

    kwargs = completions.seen[0]["extra_body"]["chat_template_kwargs"]
    assert kwargs["enable_thinking"] is True
    assert completions.seen[1]["extra_body"]["chat_template_kwargs"][
        "enable_thinking"
    ] is False


async def test_without_an_override_the_configured_default_is_used(fake_client):
    completions = fake_client(["a"])

    await llm.stream_completion([])

    assert completions.seen[0]["extra_body"]["chat_template_kwargs"][
        "enable_thinking"
    ] is settings.enable_thinking


# --- retries -----------------------------------------------------------------


async def test_a_rate_limit_is_retried_and_then_succeeds(fake_client):
    completions = fake_client([rate_limit_error(), "stream"])

    assert await llm.stream_completion([]) == "stream"
    assert len(completions.seen) == 2


async def test_persistent_rate_limiting_raises_something_actionable(
    fake_client, monkeypatch
):
    """The free tier's ceiling is the failure a user will actually hit, so the
    message has to say what to do rather than surface a bare 429."""
    monkeypatch.setattr(settings, "max_retries_interactive", 1)
    fake_client([rate_limit_error(), rate_limit_error()])

    with pytest.raises(RateLimited, match="rate limiting this key"):
        await llm.stream_completion([])


# --- patience ----------------------------------------------------------------


async def test_an_interactive_turn_gives_up_sooner_than_a_routine(
    fake_client, monkeypatch
):
    """Somebody holding the tablet should be told to try again, not left
    listening to silence. A 7am routine has nobody to tell, so it waits."""
    monkeypatch.setattr(settings, "max_retries_interactive", 2)
    monkeypatch.setattr(settings, "max_retries", 5)

    impatient = fake_client([rate_limit_error()] * 9)
    with pytest.raises(RateLimited):
        await llm.stream_completion([])
    assert len(impatient.seen) == 3  # the first try plus two retries

    patient = fake_client([rate_limit_error()] * 9)
    with pytest.raises(RateLimited):
        await llm.stream_completion([], patient=True)
    assert len(patient.seen) == 6


async def test_the_interactive_ladder_stays_within_a_few_seconds(monkeypatch):
    """Measured on a live turn, the old ladder spent ~31s before giving up.
    Whatever the numbers become, an interactive wait has to stay short enough
    that a voice reply is still worth waiting for."""
    monkeypatch.setattr(llm.random, "uniform", lambda a, b: b)
    total = sum(
        llm._backoff(attempt, patient=False, asked=None)
        for attempt in range(settings.max_retries_interactive)
    )
    assert total <= 8


async def test_a_routine_is_allowed_to_wait_much_longer(monkeypatch):
    monkeypatch.setattr(llm.random, "uniform", lambda a, b: b)
    total = sum(
        llm._backoff(attempt, patient=True, asked=None)
        for attempt in range(settings.max_retries)
    )
    assert total >= 20


async def test_no_delay_is_taken_after_the_final_attempt(fake_client, monkeypatch):
    """Sleeping after the last try is pure dead time — the answer is already
    going to be an error."""
    slept: list[float] = []

    async def record(delay: float) -> None:
        slept.append(delay)

    monkeypatch.setattr(settings, "max_retries_interactive", 2)
    fake_client([rate_limit_error()] * 5)
    monkeypatch.setattr(llm.asyncio, "sleep", record)

    with pytest.raises(RateLimited):
        await llm.stream_completion([])

    assert len(slept) == 2  # not 3


async def test_a_retry_after_header_is_honoured_when_one_is_sent(monkeypatch):
    """NVIDIA sends none today — a real 429 came back with no retry or
    rate-limit headers at all — but RAXIT_BASE_URL points at other
    OpenAI-compatible endpoints, and most of those do."""
    import httpx

    response = httpx.Response(
        429, headers={"retry-after": "3"}, request=httpx.Request("POST", "http://x")
    )
    exc = openai.RateLimitError("429", response=response, body=None)

    assert llm.retry_after(exc) == 3.0
    assert llm._backoff(0, patient=False, asked=3.0) == 3.0


async def test_an_absent_or_junk_retry_after_falls_back_to_the_ladder():
    assert llm.retry_after(rate_limit_error()) is None
    assert llm.retry_after(RuntimeError("no response at all")) is None


async def test_an_endpoint_asking_for_an_absurd_wait_is_capped(monkeypatch):
    """A misbehaving proxy answering `Retry-After: 3600` must not park an
    interactive turn for an hour."""
    assert llm._backoff(0, patient=False, asked=3600.0) <= 6
    assert llm._backoff(0, patient=True, asked=3600.0) <= 30


async def test_a_dropped_connection_is_retried(fake_client):
    import httpx

    error = openai.APIConnectionError(request=httpx.Request("POST", "http://x"))
    completions = fake_client([error, "stream"])

    assert await llm.stream_completion([]) == "stream"
    assert len(completions.seen) == 2


async def test_a_persistent_connection_failure_gives_up_with_the_cause(
    fake_client, monkeypatch
):
    import httpx

    monkeypatch.setattr(settings, "max_retries_interactive", 1)
    error = openai.APIConnectionError(request=httpx.Request("POST", "http://x"))
    fake_client([error, error])

    with pytest.raises(RuntimeError, match="failed after retries"):
        await llm.stream_completion([])


async def test_an_unexpected_error_is_not_retried(fake_client):
    """Retrying a 400 just spends the rate-limit budget on the same rejection."""
    completions = fake_client(
        [openai.BadRequestError("bad", response=_response(400), body=None)]
    )

    with pytest.raises(openai.BadRequestError):
        await llm.stream_completion([])
    assert len(completions.seen) == 1


# --- the shared client -------------------------------------------------------


def test_a_missing_key_says_where_to_get_a_free_one(monkeypatch):
    monkeypatch.setattr(llm, "_client", None)
    monkeypatch.setattr(settings, "api_key", "")

    with pytest.raises(RuntimeError, match=re.escape("build.nvidia.com")):
        llm.client()


def test_the_client_is_shared_across_calls(monkeypatch):
    """One httpx pool for the process. Building one per request leaks a pool
    each time and forfeits connection reuse, which on a tablet radio is the
    slowest part of a turn."""
    monkeypatch.setattr(llm, "_client", None)
    try:
        assert llm.client() is llm.client()
    finally:
        llm._client = None


async def test_aclose_releases_the_client(monkeypatch):
    closed = {"n": 0}

    class Fake:
        async def close(self) -> None:
            closed["n"] += 1

    monkeypatch.setattr(llm, "_client", Fake())
    await llm.aclose()

    assert closed["n"] == 1
    assert llm._client is None


async def test_aclose_is_safe_when_nothing_was_opened(monkeypatch):
    monkeypatch.setattr(llm, "_client", None)
    await llm.aclose()


# --- stream lifecycle --------------------------------------------------------


async def test_stream_deltas_closes_the_response_after_a_full_read(monkeypatch):
    stream = _fake_stream(monkeypatch, [text_chunk("a"), text_chunk("b")])

    collector = StreamCollector()
    async for _ in llm.stream_deltas([], None, collector):
        pass

    assert stream.closed
    assert collector.text == "ab"


async def test_stream_deltas_closes_the_response_when_the_consumer_stops_early(
    monkeypatch,
):
    """A client closing the tab mid-answer abandons the response body. Left to
    the garbage collector that connection stays pooled for an unbounded time,
    which on a process running for weeks is a leak rather than a blip — so the
    agent drives this through `aclosing`, and the close must be prompt."""
    from contextlib import aclosing

    stream = _fake_stream(monkeypatch, [text_chunk("a"), text_chunk("b")])

    collector = StreamCollector()
    async with aclosing(llm.stream_deltas([], None, collector)) as deltas:
        async for _ in deltas:
            break

    assert stream.closed


def _fake_stream(monkeypatch, chunks: list):
    from conftest import FakeStream

    stream = FakeStream(chunks)

    async def fake(*args, **kwargs):
        return stream

    monkeypatch.setattr(llm, "stream_completion", fake)
    return stream
