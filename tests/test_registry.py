"""The registry's whole claim is that a tool's schema and its implementation
cannot drift apart, because both come from one declaration. These tests hold
it to that: every schema is checked against the real function signature.
"""

from __future__ import annotations

import inspect

import pytest

from raxit import tools
from raxit.tools.registry import Tool, invoke, obj, opt, tool

ALL_TOOLS = sorted(tools.REGISTRY)


def test_expected_tools_are_registered():
    # Importing `raxit.tools` must pull in every module; a tool defined in a
    # file nobody imports is invisible to the model and fails silently.
    assert len(ALL_TOOLS) == 22
    assert {"speak", "listen", "battery", "remember", "shell"} <= set(ALL_TOOLS)


@pytest.mark.parametrize("name", ALL_TOOLS)
def test_schema_is_a_well_formed_object_schema(name: str):
    schema = tools.REGISTRY[name].input_schema
    assert schema["type"] == "object"
    assert set(schema["required"]) <= set(schema["properties"])
    for prop, spec in schema["properties"].items():
        assert "type" in spec, f"{name}.{prop} has no type"


@pytest.mark.parametrize("name", ALL_TOOLS)
def test_schema_properties_match_the_function_signature(name: str):
    """Every declared property must be a real parameter, and vice versa.

    This is the drift check. A schema advertising a parameter the function
    does not accept turns into a TypeError at call time — after the model has
    already spent a request deciding to use it.
    """
    entry = tools.REGISTRY[name]
    params = inspect.signature(entry.fn).parameters
    assert set(entry.input_schema["properties"]) == set(params), name


@pytest.mark.parametrize("name", ALL_TOOLS)
def test_required_params_have_no_default_and_optional_ones_do(name: str):
    entry = tools.REGISTRY[name]
    params = inspect.signature(entry.fn).parameters
    required = set(entry.input_schema["required"])
    for param, spec in params.items():
        has_default = spec.default is not inspect.Parameter.empty
        if param in required:
            assert not has_default, f"{name}.{param} is required but has a default"
        else:
            assert has_default, f"{name}.{param} is optional but has no default"


def test_only_outbound_communication_is_gated():
    """Dangerous means "cannot be undone once it leaves the device".

    Pinned deliberately: silently un-gating `send_sms` would let an unattended
    routine text somebody at 7am with nobody watching.
    """
    gated = {n for n, t in tools.REGISTRY.items() if t.dangerous}
    assert gated == {"send_sms", "call"}


@pytest.mark.parametrize("name", ALL_TOOLS)
def test_descriptions_are_sentences_not_restated_names(name: str):
    # The description is the only thing the model sees when choosing between
    # 22 tools, so a placeholder or a bare echo of the name is a real defect.
    description = tools.REGISTRY[name].description
    assert description[0].isupper() and description.rstrip().endswith(".")
    assert len(description.split()) >= 4, name
    assert description.lower().strip(". ") != name.replace("_", " ")


def test_definition_is_the_openai_function_shape():
    entry = tools.REGISTRY["battery"]
    definition = entry.definition()
    assert definition["type"] == "function"
    assert definition["function"]["name"] == "battery"
    assert definition["function"]["parameters"] is entry.input_schema
    assert "description" in definition["function"]


def test_definitions_covers_every_tool():
    assert len(tools.definitions()) == len(tools.REGISTRY)


# --- dispatch ----------------------------------------------------------------


@pytest.fixture
def sandbox_registry(monkeypatch: pytest.MonkeyPatch):
    """A registry containing only what a test puts in it."""
    monkeypatch.setattr(tools.registry, "REGISTRY", {})
    return tools.registry.REGISTRY


async def test_invoke_runs_a_sync_tool(sandbox_registry):
    @tool("sync_tool", "x", obj(value={"type": "string"}))
    def _(value: str) -> str:
        return f"got {value}"

    assert await invoke("sync_tool", {"value": "hi"}) == "got hi"


async def test_invoke_awaits_an_async_tool(sandbox_registry):
    @tool("async_tool", "x", opt())
    async def _() -> str:
        return "awaited"

    assert await invoke("async_tool", {}) == "awaited"


async def test_invoke_stringifies_non_string_returns(sandbox_registry):
    """Tool results are replayed to the model as text, so anything else has to
    be coerced rather than shipped as a dict the API will reject."""

    @tool("number_tool", "x", opt())
    def _() -> int:
        return 42

    result = await invoke("number_tool", {})
    assert result == "42" and isinstance(result, str)


async def test_invoke_rejects_an_unknown_tool(sandbox_registry):
    with pytest.raises(KeyError, match="nope"):
        await invoke("nope", {})


async def test_invoke_propagates_tool_failures(sandbox_registry):
    """The agent turns exceptions into error tool results; the registry must
    not swallow them first."""

    @tool("boom", "x", opt())
    def _() -> str:
        raise RuntimeError("device on fire")

    with pytest.raises(RuntimeError, match="device on fire"):
        await invoke("boom", {})


async def test_invoke_rejects_arguments_the_tool_does_not_take(sandbox_registry):
    @tool("strict", "x", obj(a={"type": "string"}))
    def _(a: str) -> str:
        return a

    with pytest.raises(TypeError):
        await invoke("strict", {"a": "1", "b": "2"})


def test_tool_decorator_returns_the_original_function(sandbox_registry):
    def original() -> str:
        return "x"

    decorated = tool("t", "x", opt())(original)
    assert decorated is original
    assert sandbox_registry["t"].fn is original


def test_obj_requires_everything_and_opt_requires_nothing():
    assert obj(a={"type": "string"}, b={"type": "integer"})["required"] == ["a", "b"]
    assert opt(a={"type": "string"})["required"] == []
    assert opt()["properties"] == {}


def test_tool_defaults_to_safe():
    assert Tool("n", "d", opt(), lambda: "", ).dangerous is False
