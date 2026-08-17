"""List models available on the configured endpoint: `python -m raxit.models`.

Handy when a model id 404s, or when hunting for a cheaper/faster one — the
catalog on NVIDIA's free tier changes more often than any README can track.
"""

from __future__ import annotations

import asyncio
import sys

from . import llm
from .config import settings


async def main() -> None:
    needle = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    ids = await llm.list_models()
    shown = [i for i in ids if needle in i.lower()]

    print(f"{settings.base_url} — {len(ids)} models"
          + (f", {len(shown)} matching {needle!r}" if needle else ""))
    for model_id in shown:
        print(f"  {'* ' if model_id == settings.model else '  '}{model_id}")
    if needle and not shown:
        print("  (nothing matched)")


if __name__ == "__main__":
    asyncio.run(main())
