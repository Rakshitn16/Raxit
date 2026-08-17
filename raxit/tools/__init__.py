"""Importing this package registers every tool.

The submodules are imported for their `@tool` side effects; the registry is
what the rest of the codebase talks to. A tool defined in a module nobody
imports is invisible to the model and fails silently, so a test asserts the
registry's size rather than trusting this line.
"""

from . import android, apps, device, system  # noqa: F401  (registration side effects)
from .registry import REGISTRY, definitions, invoke

__all__ = ["REGISTRY", "definitions", "invoke"]
