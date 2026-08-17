"""LLM provider: NVIDIA NIM via its OpenAI-compatible endpoint.

NVIDIA hosts 100+ models at `integrate.api.nvidia.com/v1` and speaks the OpenAI
wire format, so the official `openai` SDK reaches it with nothing but a
`base_url` change. A free NVIDIA Developer account gets ~40 requests/minute —
the constraint that shapes this file, because one conversational turn costs one
request *per tool round*.

Everything provider-specific lives here. Swapping to Groq, OpenRouter, Together
or a local vLLM is a base_url and a model id.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
from collections.abc import AsyncIterator
from typing import Any

import httpx
import openai

from .config import settings

log = logging.getLogger("raxit.llm")

# Nemotron returns its scratchpad in a dedicated `reasoning_content` field, but
# some other NIM models inline it as <think>...</think>. Strip that as a
# fallback so raw reasoning is never spoken aloud or shown as the answer.
THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


class RateLimited(RuntimeError):
    """The free tier's requests-per-minute ceiling was hit."""


def strip_reasoning(text: str) -> str:
    return THINK_BLOCK.sub("", text or "").strip()


_client: openai.AsyncOpenAI | None = None


def client() -> openai.AsyncOpenAI:
    """The shared client.

    One instance for the process, not one per request: each `AsyncOpenAI`
    carries its own httpx connection pool, so constructing one per call leaks
    a pool every time and forfeits connection reuse — which matters on a
    tablet's flaky radio, where a fresh TLS handshake per request is the
    slowest part of a turn.
    """
    global _client
    if _client is None:
        if not settings.api_key:
            raise RuntimeError(
                "No API key. Get one free at https://build.nvidia.com (no card "
                "required), then set NVIDIA_API_KEY=nvapi-... in .env"
            )
        _client = openai.AsyncOpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=httpx.Timeout(180.0, connect=15.0),
            max_retries=0,  # handled below, with rate-limit awareness
        )
    return _client


async def aclose() -> None:
    """Release the connection pool. Called from the server's shutdown hook."""
    global _client
    if _client is not None:
        await _client.close()
        _client = None


def _extra_body() -> dict[str, Any]:
    """Nemotron-specific reasoning controls.

    NIM ignores unknown `chat_template_kwargs` rather than erroring, so this
    stays harmless on models that don't implement thinking.
    """
    return {
        "chat_template_kwargs": {"enable_thinking": settings.enable_thinking},
        "reasoning_budget": settings.reasoning_budget,
    }


def retry_after(exc: Exception) -> float | None:
    """Seconds the endpoint asked us to wait, if it said.

    NVIDIA does not send `Retry-After` today — a measured 429 came back with
    no retry or rate-limit headers at all and a two-field body. This reads it
    anyway because it costs three lines and every other OpenAI-compatible
    endpoint you might point `RAXIT_BASE_URL` at does send it.
    """
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    try:
        return max(0.0, float(headers.get("retry-after", "")))
    except (TypeError, ValueError):
        return None


def _backoff(attempt: int, *, patient: bool, asked: float | None) -> float:
    if asked is not None:
        return min(asked, 30.0 if patient else 6.0)
    return min(2**attempt + random.uniform(0, 1), 20.0 if patient else 5.0)


async def stream_completion(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    thinking: bool | None = None,
    patient: bool = False,
) -> Any:
    """Start one streamed chat completion, retrying on 429 and transient 5xx.

    `patient` picks how long to keep trying, and the caller knows which is
    right: a routine firing at 7am should wait out a rate limit, while
    somebody holding the tablet should not listen to half a minute of silence
    before being told to try again. Interactive turns therefore give up after
    a few seconds and say so; unattended ones keep going.

    The 429 is worth taking seriously — a burst of 30 requests to the free
    tier had 11 rejected — and one conversational turn costs one request per
    tool round, so a multi-round answer trips it on its own.
    """
    kwargs: dict[str, Any] = {
        "model": settings.model,
        "messages": messages,
        "temperature": settings.temperature if temperature is None else temperature,
        "top_p": settings.top_p,
        "max_tokens": settings.max_tokens if max_tokens is None else max_tokens,
        "stream": True,
        "extra_body": _extra_body(),
    }
    if thinking is not None:
        kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": thinking},
            "reasoning_budget": settings.reasoning_budget,
        }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    retries = settings.max_retries if patient else settings.max_retries_interactive
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return await client().chat.completions.create(**kwargs)
        except openai.RateLimitError as exc:
            last = exc
            if attempt == retries:
                break
            delay = _backoff(attempt, patient=patient, asked=retry_after(exc))
            log.warning("rate limited; retrying in %.1fs", delay)
            await asyncio.sleep(delay)
        except (openai.APIConnectionError, openai.InternalServerError) as exc:
            last = exc
            if attempt == retries:
                break
            await asyncio.sleep(_backoff(attempt, patient=patient, asked=None))

    if isinstance(last, openai.RateLimitError):
        raise RateLimited(
            "NVIDIA's free tier is rate limiting this key. One turn costs one "
            "request per tool round, so a couple of multi-step questions in a "
            "row will do it. Wait a few seconds and ask again, or request a "
            "limit increase on your NVIDIA developer account."
        ) from last
    raise RuntimeError(f"LLM request failed after retries: {last}") from last


async def stream_deltas(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    collector: StreamCollector,
    *,
    thinking: bool | None = None,
    patient: bool = False,
) -> AsyncIterator[dict[str, str]]:
    """Run a completion, folding chunks into `collector` and yielding deltas.

    Closing the stream is the point of this wrapper: a consumer that stops
    early — a tab closed mid-answer — would otherwise abandon the response
    body, holding its pooled connection until the garbage collector got to
    it. Drive this through `contextlib.aclosing` so the close is prompt
    rather than eventual.
    """
    stream = await stream_completion(
        messages, tools, thinking=thinking, patient=patient
    )
    try:
        async for chunk in stream:
            delta = collector.add(chunk)
            if delta:
                yield delta
    finally:
        await stream.close()


class StreamCollector:
    """Accumulates one streamed completion into a usable message.

    Tool calls are the fiddly part: the API dribbles them out across many
    chunks — the first carries `id` and function name, then `arguments` arrives
    as JSON *string fragments* that are meaningless until concatenated. Chunks
    are keyed by `index` and several calls can interleave, so buffer per index
    and assemble only at the end.
    """

    def __init__(self) -> None:
        self.text = ""
        self.reasoning = ""
        self._calls: dict[int, dict[str, Any]] = {}

    def add(self, chunk: Any) -> dict[str, str]:
        """Fold in one chunk; return the deltas for live display."""
        if not chunk.choices:
            return {}
        delta = chunk.choices[0].delta
        out: dict[str, str] = {}

        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            self.reasoning += reasoning
            out["reasoning"] = reasoning

        if delta.content:
            self.text += delta.content
            out["text"] = delta.content

        for call in delta.tool_calls or []:
            slot = self._calls.setdefault(
                call.index, {"id": "", "name": "", "arguments": ""}
            )
            if call.id:
                slot["id"] = call.id
            if call.function and call.function.name:
                slot["name"] = call.function.name
            if call.function and call.function.arguments:
                slot["arguments"] += call.function.arguments
        return out

    def tool_calls(self) -> list[dict[str, Any]]:
        return [
            {
                "id": call["id"] or f"call_{index}",
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": call["arguments"] or "{}",
                },
            }
            for index, call in sorted(self._calls.items())
            if call["name"]
        ]

    def assistant_message(self) -> dict[str, Any]:
        """The assistant turn, in the shape the API wants replayed to it.

        `reasoning_content` is deliberately dropped: it is display-only, and
        echoing it back wastes context on every subsequent turn.
        """
        message: dict[str, Any] = {
            "role": "assistant",
            "content": strip_reasoning(self.text) or None,
        }
        calls = self.tool_calls()
        if calls:
            message["tool_calls"] = calls
        return message


async def list_models() -> list[str]:
    """Model ids available on the endpoint — useful when a default 404s."""
    async with httpx.AsyncClient(timeout=30) as http:
        resp = await http.get(f"{settings.base_url}/models")
        resp.raise_for_status()
        return sorted(m["id"] for m in resp.json()["data"])
