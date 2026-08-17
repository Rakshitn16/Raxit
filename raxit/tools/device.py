"""The rest of the Termux:API surface, deliberately grouped.

Termux:API exposes about twenty-five more commands than Raxit wrapped, and
the obvious move — one tool each — is the wrong one. Measured against the
live endpoint in this project: a model that picked the right tool 6/6 with a
handful loaded dropped to 2/6 once twenty-two were in front of it. Tool
choice degrades with tool count, so capability added carelessly subtracts.

So related commands share one tool behind a `kind` enum. `device_info`
covers five binaries and costs the model one decision instead of five. The
enum values are still discoverable — they are in the schema — but they are
read only after the tool has been chosen, which is the cheap half.
"""

from __future__ import annotations

import json
from typing import Any

from .android import _json, _run
from .registry import mix, obj, opt, run_blocking, tool

# termux-sensor names vary by hardware; these are the ones a Tab A9+ has.
SENSORS = ["light", "proximity", "accelerometer", "gyroscope", "magnetometer"]

INFO_COMMANDS = {
    "wifi": ["termux-wifi-connectioninfo"],
    "wifi_scan": ["termux-wifi-scaninfo"],
    "telephony": ["termux-telephony-deviceinfo"],
    "cell": ["termux-telephony-cellinfo"],
    "camera": ["termux-camera-info"],
    "audio": ["termux-audio-info"],
    "voices": ["termux-tts-engines"],
}


@tool(
    "contacts",
    "Look somebody up in the tablet's address book and get their number. "
    "Call this before `call` or `send_sms` whenever the request names a "
    "person rather than a number — never guess at somebody's number.",
    opt(
        match={
            "type": "string",
            "description": "Case-insensitive name fragment. Omit for all.",
        },
    ),
)
async def contacts(match: str = "") -> str:
    def go() -> str:
        entries = _json(["termux-contact-list"]) or []
        if match:
            entries = [
                e for e in entries if match.lower() in str(e.get("name", "")).lower()
            ]
        if not entries:
            return f"No contact matching {match!r}." if match else "Address book empty."
        lines = [f"- {e.get('name', '?')}: {e.get('number', '?')}" for e in entries[:40]]
        return "\n".join(lines)

    return await run_blocking(go)


@tool(
    "call_log",
    "Recent calls: who, which direction, when, and how long. Use it for "
    "'did she ring back' and 'who called me' questions.",
    opt(limit={"type": "integer", "description": "How many entries, default 10."}),
)
async def call_log(limit: int = 10) -> str:
    def go() -> str:
        entries = _json(["termux-call-log", "-l", str(limit)]) or []
        if not entries:
            return "No calls in the log."
        return "\n".join(
            f"- {e.get('type', '?')} {e.get('name') or e.get('phone_number', '?')}"
            f" at {e.get('date', '?')} ({e.get('duration', '?')})"
            for e in entries
        )

    return await run_blocking(go)


@tool(
    "sensor",
    "Read one of the tablet's physical sensors once. Use `light` to tell "
    "whether the room is dark, `proximity` for whether it is face-down or "
    "covered, `accelerometer` for orientation and movement.",
    obj(kind={"type": "string", "enum": SENSORS}),
)
async def sensor(kind: str) -> str:
    def go() -> str:
        # -n 1 takes a single reading; without it termux-sensor streams until
        # killed, which would hang the turn until the tool timeout.
        raw = _run(["termux-sensor", "-s", kind, "-n", "1"], timeout=25)
        try:
            data: Any = json.loads(raw)
        except json.JSONDecodeError:
            return raw[:500] or f"No reading from {kind}."
        return f"{kind}: {json.dumps(data)[:400]}"

    return await run_blocking(go)


@tool(
    "media",
    "Control the tablet's own media player: play a file, pause, resume, "
    "stop, or report what is playing. This is Termux's player — to control "
    "Spotify or YouTube, open a deep link with `open_url` instead.",
    mix(
        {
            "action": {
                "type": "string",
                "enum": ["play", "pause", "resume", "stop", "info"],
            }
        },
        path={"type": "string", "description": "Audio file, required for 'play'."},
    ),
)
async def media(action: str, path: str = "") -> str:
    import os

    def go() -> str:
        if action == "play":
            if not path:
                return "A file path is required to play something."
            _run(["termux-media-player", "play", os.path.expanduser(path)])
            return f"Playing {path}."
        verb = {
            "pause": "pause", "resume": "play", "stop": "stop", "info": "info",
        }[action]
        return _run(["termux-media-player", verb]) or f"Media {action}."

    return await run_blocking(go)


@tool(
    "device_info",
    "Report a piece of the tablet's own state: which wifi it is on, nearby "
    "networks, SIM and signal, cameras, audio configuration, or the "
    "installed text-to-speech voices.",
    obj(kind={"type": "string", "enum": sorted(INFO_COMMANDS)}),
)
async def device_info(kind: str) -> str:
    def go() -> str:
        out = _run(INFO_COMMANDS[kind])
        return out[:1500] if out else f"Nothing reported for {kind}."

    return await run_blocking(go)


@tool(
    "device_setting",
    "Read or change a device setting. Omit `value` to read the current one. "
    "Volume streams are named: music, notification, ring, alarm, call.",
    mix(
        {"name": {"type": "string", "enum": ["volume", "brightness"]}},
        stream={
            "type": "string",
            "description": "Which volume stream; only used when name is 'volume'.",
        },
        value={
            "type": "integer",
            "description": "New level. Volume is per-stream and capped by "
            "Android; brightness is 0-255. Omit to read.",
        },
    ),
)
async def device_setting(
    name: str, stream: str = "music", value: int | None = None
) -> str:
    def go() -> str:
        if name == "volume":
            if value is None:
                return _run(["termux-volume"])[:800]
            _run(["termux-volume", stream, str(value)])
            return f"{stream} volume set to {value}."
        if value is None:
            return "Brightness is write-only on Android; pass a value to set it."
        _run(["termux-brightness", str(value)])
        return f"Brightness set to {value}."

    return await run_blocking(go)


@tool(
    "ask_user",
    "Put a dialog on the tablet's screen and wait for an answer. Use it when "
    "you genuinely need a decision from somebody standing at the device and "
    "no other channel is open — it blocks until they respond or it times "
    "out, so never call it from a routine.",
    mix(
        {"title": {"type": "string", "description": "The question."}},
        kind={
            "type": "string",
            "enum": ["text", "confirm", "date", "time"],
            "description": "Default 'text'.",
        },
    ),
)
async def ask_user(title: str, kind: str = "text") -> str:
    def go() -> str:
        answer = _json(["termux-dialog", kind, "-t", title])
        if not answer:
            return "No answer."
        if answer.get("code", 0) < 0:
            return "Dismissed without answering."
        return str(answer.get("text") or answer.get("index") or answer)

    return await run_blocking(go)


@tool(
    "dismiss_notification",
    "Clear a notification this agent posted, by the id `notify` reported.",
    obj(notification_id={"type": "string"}),
)
async def dismiss_notification(notification_id: str) -> str:
    def go() -> str:
        _run(["termux-notification-remove", notification_id])
        return f"Cleared {notification_id}."

    return await run_blocking(go)
