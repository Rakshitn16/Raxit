"""Shared fixtures.

Two things need isolating before `raxit` is imported at all: the SQLite file
(a test run must not touch the tablet's real memory) and the API key (the
config module reads `.env` at import, and a developer with a real key in
their working copy must not have tests fire live requests against NVIDIA).
"""

from __future__ import annotations

import copy
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

# Must precede the first `raxit` import: config.py reads the environment and
# creates DATA_DIR at module scope.
os.environ.setdefault("RAXIT_DATA_DIR", "/tmp/raxit-tests")
os.environ["NVIDIA_API_KEY"] = "nvapi-test-key-not-real"
os.environ["RAXIT_API_KEY"] = "nvapi-test-key-not-real"
os.environ["RAXIT_TIMEZONE"] = "Asia/Kolkata"
os.environ["RAXIT_OWNER_NAME"] = "the tester"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from raxit import memory


@pytest.fixture(autouse=True)
def fresh_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Give every test its own database.

    `memory.connect` reads the module-level DB_PATH on each call, so pointing
    that at a tmp file is enough — no need to reimport the module.
    """
    db = tmp_path / "raxit.db"
    monkeypatch.setattr(memory, "DB_PATH", db)
    memory.init()
    yield db


# --- fake completion streams -------------------------------------------------
#
# The agent reads chunks through `StreamCollector`, so faking at the chunk
# level exercises the real reassembly path rather than stubbing it out. These
# builders produce the duck-typed shape the OpenAI SDK yields.


def text_chunk(text: str) -> SimpleNamespace:
    return _chunk(content=text)


def reasoning_chunk(text: str) -> SimpleNamespace:
    return _chunk(reasoning_content=text)


def tool_chunk(
    index: int,
    *,
    call_id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
) -> SimpleNamespace:
    """One tool-call fragment.

    The API dribbles a single call across many chunks — id and name first,
    then `arguments` as JSON string fragments — so tests build them the same
    way to prove the reassembly works.
    """
    function = SimpleNamespace(name=name, arguments=arguments)
    call = SimpleNamespace(index=index, id=call_id, function=function)
    return _chunk(tool_calls=[call])


def _chunk(
    content: str | None = None,
    reasoning_content: str | None = None,
    tool_calls: list[Any] | None = None,
) -> SimpleNamespace:
    delta = SimpleNamespace(
        content=content,
        reasoning_content=reasoning_content,
        tool_calls=tool_calls,
    )
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


def empty_chunk() -> SimpleNamespace:
    """A chunk with no choices — NIM sends these around usage reporting."""
    return SimpleNamespace(choices=[])


class FakeStream:
    """Stands in for the SDK's async streaming response."""

    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks
        self.closed = False

    def __aiter__(self) -> FakeStream:
        self._iter = iter(self._chunks)
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration from None

    async def close(self) -> None:
        self.closed = True


def one_tool_call(name: str, arguments: str, call_id: str = "call_abc") -> list[Any]:
    """Chunks for an assistant turn that calls exactly one tool."""
    return [
        tool_chunk(0, call_id=call_id, name=name, arguments=""),
        *[tool_chunk(0, arguments=frag) for frag in _fragments(arguments)],
    ]


def _fragments(text: str, size: int = 4) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Queue up successive completions for one agent turn.

    Each entry is the chunk list for one round; `stream_completion` hands them
    out in order, so a test can script "call this tool, then answer".
    """

    class Scripted:
        def __init__(self) -> None:
            self.rounds: list[list[Any]] = []
            self.calls: list[dict[str, Any]] = []
            self.streams: list[FakeStream] = []

        def add(self, chunks: list[Any]) -> Scripted:
            self.rounds.append(chunks)
            return self

        def install(self) -> Scripted:
            from raxit import llm

            async def fake(
                messages: list[dict[str, Any]],
                tools: list[dict[str, Any]] | None = None,
                **kwargs: Any,
            ) -> FakeStream:
                # Snapshot: the agent keeps appending to this same list as the
                # turn proceeds, so holding the reference would show a test
                # the *end* state of every round rather than what was sent.
                self.calls.append(
                    {
                        "messages": copy.deepcopy(messages),
                        "tools": tools,
                        "kwargs": kwargs,
                    }
                )
                chunks = self.rounds.pop(0) if self.rounds else [text_chunk("done")]
                stream = FakeStream(chunks)
                self.streams.append(stream)
                return stream

            monkeypatch.setattr(llm, "stream_completion", fake)
            return self

    return Scripted()
