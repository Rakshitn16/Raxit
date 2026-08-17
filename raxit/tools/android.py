"""Device control via the Termux:API binaries.

Everything here shells out to a `termux-*` command, which is the only way to
reach Android's APIs from Termux. Install the companion app *and* the package:

    pkg install termux-api        # the CLI binaries
    # plus the Termux:API app from F-Droid — the binaries are stubs without it

Each wrapper returns a short text string, because that string goes straight
back to the model as a tool result.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any

from .registry import obj, opt, run_blocking, tool

TIMEOUT = 30


class TermuxUnavailable(RuntimeError):
    pass


def _run(args: list[str], stdin: str | None = None, timeout: int = TIMEOUT) -> str:
    if shutil.which(args[0]) is None:
        raise TermuxUnavailable(
            f"{args[0]} not found. Run `pkg install termux-api` and install the "
            "Termux:API app from F-Droid."
        )
    proc = subprocess.run(
        args,
        input=stdin,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"{args[0]} failed ({proc.returncode}): {detail}")
    return proc.stdout.strip()


def _json(args: list[str]) -> Any:
    out = _run(args)
    return json.loads(out) if out else {}


# --- speech ------------------------------------------------------------------


@tool(
    "speak",
    "Speak text aloud through the tablet's speaker. Use this whenever the user "
    "is interacting by voice, or for anything they need to hear without looking "
    "at the screen.",
    obj(
        text={"type": "string", "description": "What to say."},
        pitch={"type": "number", "description": "0.5-2.0, default 1.0."},
        rate={"type": "number", "description": "0.5-2.0, default 1.0."},
    ),
)
async def speak(text: str, pitch: float = 1.0, rate: float = 1.0) -> str:
    def go() -> str:
        _run(
            [
                "termux-tts-speak",
                "-p", str(pitch),
                "-r", str(rate),
                "-s", "NOTIFICATION",
            ],
            stdin=text,
            timeout=120,
        )
        return f"Spoke: {text[:80]}"

    return await run_blocking(go)


@tool(
    "listen",
    "Capture one utterance from the microphone and return it as text. Blocks "
    "until the speaker stops or the timeout elapses.",
    opt(),
)
async def listen() -> str:
    def go() -> str:
        heard = _run(["termux-speech-to-text"], timeout=60)
        return heard or "(nothing heard)"

    return await run_blocking(go)


# --- notifications -----------------------------------------------------------


@tool(
    "notify",
    "Post an Android notification. Use for anything the user should see later "
    "rather than right now — a routine's result, a reminder, a finished task.",
    obj(
        title={"type": "string"},
        content={"type": "string"},
        priority={
            "type": "string",
            "enum": ["min", "low", "default", "high", "max"],
            "description": "Use 'high' only for things worth interrupting for.",
        },
    ),
)
async def notify(title: str, content: str, priority: str = "default") -> str:
    def go() -> str:
        _run(
            [
                "termux-notification",
                "--title", title,
                "--content", content,
                "--priority", priority,
                "--id", f"raxit-{abs(hash(title)) % 100000}",
            ]
        )
        return f"Notified: {title}"

    return await run_blocking(go)


@tool(
    "read_notifications",
    "List the notifications currently in the status bar. Useful for answering "
    "'what did I miss' without the user unlocking the tablet.",
    opt(),
)
async def read_notifications() -> str:
    def go() -> str:
        items = _json(["termux-notification-list"])
        if not items:
            return "No notifications."
        lines = [
            f"- [{n.get('packageName', '?')}] {n.get('title', '')}: {n.get('content', '')}"
            for n in items
        ]
        return "\n".join(lines)

    return await run_blocking(go)


# --- device state ------------------------------------------------------------


@tool(
    "battery",
    "Read battery percentage, charging state, temperature and health.",
    opt(),
)
async def battery() -> str:
    def go() -> str:
        b = _json(["termux-battery-status"])
        return (
            f"{b.get('percentage')}%, {b.get('status')}, "
            f"{b.get('temperature')}C, health {b.get('health')}"
        )

    return await run_blocking(go)


@tool(
    "location",
    "Get the tablet's current location. GPS is slow and battery-hungry; prefer "
    "'network' unless precision actually matters.",
    opt(
        provider={"type": "string", "enum": ["gps", "network", "passive"]},
    ),
)
async def location(provider: str = "network") -> str:
    def go() -> str:
        loc = _json(["termux-location", "-p", provider])
        if not loc:
            return "Location unavailable."
        return (
            f"lat {loc.get('latitude')}, lon {loc.get('longitude')}, "
            f"+/-{loc.get('accuracy')}m via {provider}"
        )

    return await run_blocking(go)


@tool(
    "clipboard",
    "Read or write the Android clipboard.",
    opt(
        action={"type": "string", "enum": ["get", "set"]},
        text={"type": "string", "description": "Required when action is 'set'."},
    ),
)
async def clipboard(action: str = "get", text: str = "") -> str:
    def go() -> str:
        if action == "set":
            _run(["termux-clipboard-set"], stdin=text)
            return "Clipboard set."
        return _run(["termux-clipboard-get"]) or "(clipboard empty)"

    return await run_blocking(go)


@tool(
    "toast",
    "Flash a short on-screen toast message. Good for a quick acknowledgement "
    "that does not deserve a notification.",
    obj(text={"type": "string"}),
)
async def toast(text: str) -> str:
    def go() -> str:
        _run(["termux-toast", "-g", "top", text])
        return "Toast shown."

    return await run_blocking(go)


@tool(
    "vibrate",
    "Vibrate the tablet for a number of milliseconds.",
    opt(duration_ms={"type": "integer", "description": "Default 400."}),
)
async def vibrate(duration_ms: int = 400) -> str:
    def go() -> str:
        _run(["termux-vibrate", "-d", str(duration_ms), "-f"])
        return f"Vibrated {duration_ms}ms."

    return await run_blocking(go)


@tool(
    "torch",
    "Turn the camera flashlight on or off.",
    obj(on={"type": "boolean"}),
)
async def torch(on: bool) -> str:
    def go() -> str:
        _run(["termux-torch", "on" if on else "off"])
        return f"Torch {'on' if on else 'off'}."

    return await run_blocking(go)


@tool(
    "take_photo",
    "Capture a still from a camera and save it to a file path.",
    opt(
        path={"type": "string", "description": "Output path, default ~/raxit-photo.jpg."},
        camera={"type": "string", "description": "Camera id: '0' rear, '1' front."},
    ),
)
async def take_photo(path: str = "~/raxit-photo.jpg", camera: str = "0") -> str:
    import os

    target = os.path.expanduser(path)

    def go() -> str:
        _run(["termux-camera-photo", "-c", camera, target], timeout=60)
        return f"Saved photo to {target}"

    return await run_blocking(go)


# --- communication (gated) ---------------------------------------------------


@tool(
    "send_sms",
    "Send an SMS. This leaves the device and cannot be recalled, so state the "
    "recipient and the full message text when you propose it.",
    obj(
        number={"type": "string", "description": "Recipient phone number."},
        message={"type": "string"},
    ),
    dangerous=True,
)
async def send_sms(number: str, message: str) -> str:
    def go() -> str:
        _run(["termux-sms-send", "-n", number], stdin=message)
        return f"SMS sent to {number}."

    return await run_blocking(go)


@tool(
    "read_sms",
    "Read recent SMS messages from the inbox.",
    opt(limit={"type": "integer", "description": "Default 10."}),
)
async def read_sms(limit: int = 10) -> str:
    def go() -> str:
        msgs = _json(["termux-sms-list", "-l", str(limit), "-t", "inbox"])
        if not msgs:
            return "No messages."
        return "\n".join(
            f"- {m.get('number')} ({m.get('received')}): {m.get('body')}" for m in msgs
        )

    return await run_blocking(go)


@tool(
    "call",
    "Place a phone call. Dials immediately once invoked.",
    obj(number={"type": "string"}),
    dangerous=True,
)
async def call(number: str) -> str:
    def go() -> str:
        _run(["termux-telephony-call", number])
        return f"Calling {number}."

    return await run_blocking(go)


@tool(
    "open_url",
    "Open a URL or deep link in the tablet's default handler app. Also works "
    "for app deep links such as 'spotify:' or 'geo:' URIs.",
    obj(url={"type": "string"}),
)
async def open_url(url: str) -> str:
    def go() -> str:
        _run(["termux-open-url", url])
        return f"Opened {url}"

    return await run_blocking(go)
