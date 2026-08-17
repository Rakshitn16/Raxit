"""Importing this package registers every tool.

`android` and `system` are imported for their `@tool` side effects; the
registry is what the rest of the codebase talks to.
"""

from . import android, system  # noqa: F401  (registration side effects)
from .registry import REGISTRY, definitions, invoke

__all__ = ["REGISTRY", "definitions", "invoke"]
