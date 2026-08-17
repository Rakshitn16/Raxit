# Raxit

A Jarvis-style agent that runs **on** a Samsung Galaxy Tab A9+ (or any Android
device with Termux), rather than talking to one from a server somewhere.

Claude does the thinking; the tablet does the acting. The agent process lives in
Termux on the device, so its tools reach real Android APIs — it can speak,
listen, notify, text, call, read the clipboard, take a photo, check the battery,
and run shell commands, all on hardware you're holding.

```
   voice / web UI  ─────▶  agent loop (on the tablet)  ─────▶  Claude Opus 5
                                    │                              │
                                    │◀──── tool calls ─────────────┘
                                    ▼
                          termux-api ──▶ TTS · STT · SMS · calls · camera
                          sqlite     ──▶ memory + transcript + audit log
                          cron       ──▶ routines (unattended)
```

## Why this shape

The Tab A9+ runs a Helio G99. It cannot host a model worth talking to — a 3B
quantized model gets a handful of tokens per second and forgets what you said.
So inference is remote and *execution* is local, which is the half that actually
needs to be on the device.

## Setup

Install **Termux** and **Termux:API** from F-Droid. The Play Store builds are
stale and the API bridge won't connect.

```bash
git clone https://github.com/rakshitn16/raxit && cd raxit
./scripts/install-termux.sh
# put ANTHROPIC_API_KEY in .env
./scripts/run.sh
```

Open `http://127.0.0.1:8788` in the tablet's browser. Add it to the home screen
for a full-screen app.

## Talking to it

- **Web UI** — text, plus the mic button for voice.
- **Voice, hands-free** — ask it to `listen`, or wire a Tasker/Bixby shortcut to
  `POST /api/chat`.
- **Scripts** —

  ```bash
  curl -s localhost:8788/api/chat \
    -H 'content-type: application/json' \
    -d '{"message":"how much battery is left?"}'
  ```

## Routines

A routine is a YAML file in `routines/` — a cron expression and a prompt. When
it fires, the agent gets the prompt in unattended mode and picks its own tools,
so routines read like instructions to a person, not scripts:

```yaml
name: morning_brief
cron: "0 7 * * *"
effort: low
prompt: |
  Check the date and battery, recall today's commitments, and read out any
  overnight notifications worth knowing about. Speak it in under 30 seconds,
  then post the same thing as a notification.
```

Drop the file in, hit **reload** in the UI (or `POST /api/routines/reload`), and
it's live. `POST /api/routines/{name}/run` fires one immediately for testing.

## Tools

| | |
|---|---|
| **Voice** | `speak`, `listen` |
| **Attention** | `notify`, `read_notifications`, `toast`, `vibrate` |
| **Sensors** | `battery`, `location`, `take_photo`, `torch` |
| **Comms** | `send_sms`\*, `read_sms`, `call`\*, `open_url` |
| **Memory** | `remember`, `recall`, `forget` |
| **System** | `shell`, `fetch_url`, `now`, `log`, `clipboard` |

\* Gated: these leave the device and can't be recalled, so the UI asks before
they run. In unattended mode they're refused outright rather than guessed at.

Adding a tool is one decorated function in `raxit/tools/` — the JSON schema sent
to Claude and the dispatch table are generated from the same declaration, so
they can't drift.

## Safety

- `shell` runs only allowlisted binaries (`raxit/config.py`); anything else is
  refused with an explanation rather than queued.
- SMS and calls require explicit per-call approval.
- The server binds to `127.0.0.1` and has **no authentication**. Setting
  `RAXIT_HOST=0.0.0.0` exposes full device control to your LAN — only do it on a
  network you trust, and put a reverse proxy with auth in front if you care.
- Everything the agent does lands in the `events` table and the UI's activity
  feed.

## Cost

Prompt caching is on for the system prompt and tool schemas, so repeat turns
read the stable prefix at ~10% of input price. Routines default to `effort:
low`, which is the difference between a few cents a day and a few dollars. Set
`RAXIT_MODEL=claude-sonnet-5` if the tablet is doing mostly device chores.

## Layout

```
raxit/
  agent.py       tool loop, streaming, approval gating
  server.py      FastAPI + WebSocket + REST
  routines.py    YAML routines on a cron scheduler
  memory.py      SQLite transcript, facts, audit log
  tools/
    registry.py  @tool decorator → schema + dispatch
    android.py   termux-api wrappers
    system.py    memory, shell, http, time
routines/        your scheduled prompts
web/index.html   the UI
```
