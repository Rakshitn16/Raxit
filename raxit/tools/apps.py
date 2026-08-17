"""App control: launching apps and driving them through Android intents.

What is and is not possible from Termux without root, because the boundary
decides the shape of this file:

* Launching an app, handing it a deep link, sharing content into it, and
  asking the package manager what exists — all fine. `am`, `pm` and `monkey`
  are ordinary binaries and Termux is allowed to call them.
* Tapping a button inside another app is not. Injecting touch events needs
  INJECT_EVENTS, a signature-level permission Termux cannot hold. Anything
  claiming otherwise wants root, or adb from a PC.

So this drives apps from the outside, through the same intent system apps use
to talk to each other. That covers more than it sounds: most apps publish
deep links that jump straight to a screen — a chat, a route, a track — which
is usually the thing a tap was for anyway.
"""

from __future__ import annotations

import shutil
import subprocess

from .registry import mix, obj, opt, run_blocking, tool

TIMEOUT = 30


class AndroidToolMissing(RuntimeError):
    pass


def _run(args: list[str], timeout: int = TIMEOUT) -> str:
    if shutil.which(args[0]) is None:
        raise AndroidToolMissing(
            f"`{args[0]}` is not available in this Termux. It comes from "
            "Android itself; on some ROMs it needs `pkg install termux-am`."
        )
    proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    out = (proc.stdout + proc.stderr).strip()
    if proc.returncode != 0:
        raise RuntimeError(f"{args[0]} failed ({proc.returncode}): {out[:300]}")
    return out


@tool(
    "list_apps",
    "List packages installed on the tablet. Use this to find an exact "
    "package name before launching something — what a person calls an app "
    "is rarely its package id.",
    opt(
        match={
            "type": "string",
            "description": "Case-insensitive substring, e.g. 'spotify'. Omit "
            "for everything, which is long.",
        },
    ),
)
async def list_apps(match: str = "") -> str:
    def go() -> str:
        out = _run(["pm", "list", "packages"])
        names = sorted(
            line.removeprefix("package:").strip()
            for line in out.splitlines()
            if line.startswith("package:")
        )
        if match:
            names = [n for n in names if match.lower() in n.lower()]
        if not names:
            return f"No installed package matches {match!r}."
        head = names[:60]
        more = f"\n({len(names) - len(head)} more)" if len(names) > len(head) else ""
        return "\n".join(head) + more

    return await run_blocking(go)


@tool(
    "launch_app",
    "Open an app by package name, as if its icon were tapped. Find the name "
    "with `list_apps`. To land on a specific screen rather than the app's "
    "front page, use `open_url` with a deep link instead.",
    obj(
        package={
            "type": "string",
            "description": "Package id, e.g. 'com.spotify.music'.",
        }
    ),
)
async def launch_app(package: str) -> str:
    def go() -> str:
        # `monkey` launches from the package alone. `am start` would need the
        # activity name, which nothing on the device reports conveniently.
        _run(["monkey", "-p", package, "-c", "android.intent.category.LAUNCHER", "1"])
        return f"Launched {package}."

    return await run_blocking(go)


@tool(
    "share_text",
    "Push text into Android's share sheet so it can go to any app that "
    "accepts text. Use when the destination has no dedicated tool here.",
    mix(
        {"text": {"type": "string"}},
        title={"type": "string", "description": "Subject shown on the sheet."},
    ),
)
async def share_text(text: str, title: str = "Shared by Raxit") -> str:
    def go() -> str:
        _run(
            [
                "am", "start",
                "-a", "android.intent.action.SEND",
                "-t", "text/plain",
                "--es", "android.intent.extra.TEXT", text,
                "--es", "android.intent.extra.SUBJECT", title,
            ]
        )
        return f"Offered {len(text)} characters to the share sheet."

    return await run_blocking(go)


@tool(
    "send_intent",
    "Fire an arbitrary Android intent — the escape hatch for driving an app "
    "nothing else here covers, including opening a specific Settings screen. "
    "Prefer `open_url` or `launch_app` when either would do; this is easy to "
    "get wrong and fails silently when an action name is misspelled.",
    mix(
        {
            "action": {
                "type": "string",
                "description": "e.g. android.settings.WIFI_SETTINGS, "
                "android.intent.action.VIEW",
            }
        },
        data={"type": "string", "description": "URI passed as -d."},
        package={"type": "string", "description": "Restrict to this package."},
        extra_key={"type": "string", "description": "Key of one string extra."},
        extra_value={"type": "string", "description": "Value of that extra."},
    ),
)
async def send_intent(
    action: str,
    data: str = "",
    package: str = "",
    extra_key: str = "",
    extra_value: str = "",
) -> str:
    def go() -> str:
        args = ["am", "start", "-a", action]
        if data:
            args += ["-d", data]
        if package:
            args += ["-p", package]
        if extra_key:
            args += ["--es", extra_key, extra_value]
        _run(args)
        return f"Sent {action}."

    return await run_blocking(go)
