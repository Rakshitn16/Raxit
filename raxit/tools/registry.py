"""Tool registry.

A tool is a Python callable plus a JSON Schema. `@tool` registers both, so
adding a capability to Raxit means writing one function — the schema sent to
Claude and the dispatch table stay in sync automatically.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    fn: Callable[..., Any]
    # Tools that change the world outside the tablet (send an SMS, place a
    # call) are gated behind user confirmation in the UI.
    dangerous: bool = False

    def definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


REGISTRY: dict[str, Tool] = {}


def tool(
    name: str,
    description: str,
    schema: dict[str, Any],
    *,
    dangerous: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        REGISTRY[name] = Tool(name, description, schema, fn, dangerous)
        return fn

    return decorate


def definitions() -> list[dict[str, Any]]:
    return [t.definition() for t in REGISTRY.values()]


async def invoke(name: str, payload: dict[str, Any]) -> str:
    """Run a tool and return its result as text.

    Failures are returned rather than raised: the caller marks them
    `is_error` so Claude can read the message and try another approach.
    """
    entry = REGISTRY.get(name)
    if entry is None:
        raise KeyError(f"unknown tool: {name}")

    result = entry.fn(**payload)
    if inspect.isawaitable(result):
        result = await result
    return result if isinstance(result, str) else str(result)


def obj(**properties: dict[str, Any]) -> dict[str, Any]:
    """Shorthand for an object schema where every property is required."""
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
    }


def opt(**properties: dict[str, Any]) -> dict[str, Any]:
    """Object schema with no required properties."""
    return {"type": "object", "properties": properties, "required": []}


def run_blocking(fn: Callable[..., Any], *args: Any) -> Awaitable[Any]:
    """Push a blocking call onto the default executor.

    `termux-*` binaries take a few hundred milliseconds each; running them
    inline would stall the event loop and freeze the web UI mid-response.
    """
    return asyncio.get_running_loop().run_in_executor(None, fn, *args)
