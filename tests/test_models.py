"""`python -m raxit.models` — the escape hatch for when a model id 404s.

Free-tier catalogues change without notice, so this is the first thing a user
reaches for when the configured default stops existing. It has to work when
nothing else does.
"""

from __future__ import annotations

import httpx
import pytest

from raxit import llm, models
from raxit.config import settings

CATALOG = ["meta/llama-3.3-70b-instruct", "nvidia/nemotron-3-super-120b-a12b"]


@pytest.fixture
def endpoint(monkeypatch: pytest.MonkeyPatch):
    """Serve a fake /models listing."""

    def install(ids: list[str] | int) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if isinstance(ids, int):
                return httpx.Response(ids)
            return httpx.Response(200, json={"data": [{"id": i} for i in ids]})

        original = httpx.AsyncClient

        def build(*args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            return original(*args, **kwargs)

        monkeypatch.setattr(llm.httpx, "AsyncClient", build)

    return install


async def test_models_are_listed_sorted(endpoint):
    endpoint(["z-ai/glm-5.2", "meta/llama-3.3-70b-instruct"])
    assert await llm.list_models() == [
        "meta/llama-3.3-70b-instruct",
        "z-ai/glm-5.2",
    ]


async def test_an_unreachable_endpoint_raises_rather_than_printing_nothing(endpoint):
    endpoint(500)
    with pytest.raises(httpx.HTTPStatusError):
        await llm.list_models()


async def test_the_cli_prints_the_whole_catalog(endpoint, monkeypatch, capsys):
    endpoint(CATALOG)
    monkeypatch.setattr(models.sys, "argv", ["models"])

    await models.main()

    out = capsys.readouterr().out
    assert "2 models" in out
    assert all(model_id in out for model_id in CATALOG)


async def test_the_cli_filters_on_a_substring(endpoint, monkeypatch, capsys):
    endpoint(CATALOG)
    monkeypatch.setattr(models.sys, "argv", ["models", "LLAMA"])

    await models.main()

    out = capsys.readouterr().out
    assert "1 matching" in out
    assert "nemotron" not in out


async def test_the_cli_marks_the_configured_model(endpoint, monkeypatch, capsys):
    """The point of running this is usually "is the thing in my .env real" —
    so the answer has to be visible at a glance."""
    endpoint(CATALOG)
    monkeypatch.setattr(models.sys, "argv", ["models"])
    monkeypatch.setattr(settings, "model", CATALOG[0])

    await models.main()

    marked = [
        line for line in capsys.readouterr().out.splitlines() if line.startswith("  *")
    ]
    assert marked and CATALOG[0] in marked[0]


async def test_the_cli_says_so_when_a_filter_matches_nothing(
    endpoint, monkeypatch, capsys
):
    endpoint(CATALOG)
    monkeypatch.setattr(models.sys, "argv", ["models", "gpt-4"])

    await models.main()

    assert "(nothing matched)" in capsys.readouterr().out
