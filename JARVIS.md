# JARVIS — Build Specification

A voice-first, agentic assistant that runs **on** a Samsung Galaxy Tab A9+, with
real control of the device and the apps on it.

This document is the build brief. Hand it to Claude Code and work through the
phases in order. Every phase has acceptance criteria; do not start the next one
until the current one passes on the actual tablet.

---

## 0. How to use this document

**For Claude Code:** Build in the phase order given in §11. Each phase is
independently useful — Phase 1 alone is a working voice assistant. Do not
scaffold all of it at once; a half-built Phase 4 is worse than a finished
Phase 2.

**Non-negotiables, learned the hard way** (§14 has the full list):
- Read §14 *before* writing `requirements.txt`. Android breaks four common
  Python assumptions and each one costs an hour if rediscovered.
- Read §9 before adding the twentieth tool. Tool count is not free.
- Verify against the live API, not against mocks alone. Mocks agreed with
  every wrong assumption made while building this spec's predecessor.

---

## 1. The device, honestly

| | |
|---|---|
| Tablet | Samsung Galaxy Tab A9+ |
| SoC | Snapdragon 695 (2×2.2 GHz A78 + 6×1.8 GHz A55, Adreno 619) |
| RAM | 8 GB |
| Free storage | ~30 GB |
| OS | Android 13/14 → **wireless debugging available** (this matters, §6) |
| Runtime | Termux + Ubuntu via proot, Ollama, code-server |

**What this hardware can and cannot do:**

- ✅ Run the agent process, SQLite, a web server, audio I/O, wake-word
  detection — all comfortably. This is a coordination workload, not a compute
  one.
- ✅ Run local **embeddings** (a single forward pass, ~50 ms).
- ⚠️ Run a 3–4B quantized LLM at roughly **3–8 tokens/sec**. Fine for
  classification (a few tokens out). Not fine for an agentic tool loop, which
  needs hundreds of tokens per turn across several turns.
- ❌ Run a model good enough to drive 30 tools reliably. Not close.

**Therefore: inference is remote, execution is local.** That split is the whole
architecture. The tablet is the hands; the cloud is the planning cortex; a small
local model is the spinal reflex.

---

## 2. Architecture

```
  ┌─────────────┐
  │  wake word  │  openWakeWord "hey_jarvis" — always-on, local, ~2% CPU
  └──────┬──────┘
         │ wakes
  ┌──────▼──────┐
  │     STT     │  local whisper.cpp (offline) ─or─ Groq Whisper (fast)
  └──────┬──────┘
         │ text
  ┌──────▼──────────────────────────────────┐
  │              ROUTER                     │  local 3B model, ~300ms
  │  trivial? → answer locally, offline     │
  │  otherwise → cloud agent                │
  └──────┬──────────────────────────────────┘
         │
  ┌──────▼──────────────────────────────────┐
  │           AGENT LOOP (cloud)            │  Claude Opus 5
  │   plan → call tool → observe → repeat   │  tool search + deferred loading
  └──────┬──────────────────────────────────┘
         │ tool calls
  ┌──────▼──────────────────────────────────────────────────────┐
  │                      TOOL LAYER                             │
  │                                                             │
  │  termux-api ──▶ TTS, STT, SMS, calls, camera, sensors,      │
  │                 clipboard, notifications, contacts          │
  │  intents    ──▶ launch apps, deep links, share sheet        │
  │  local adb  ──▶ TAP, SWIPE, TYPE, SCREENSHOT, read UI tree  │  ← §6
  │  vision     ──▶ screenshot → Claude → coordinates → tap     │
  │  sqlite     ──▶ transcript + facts + vectors + audit log    │
  │  cron       ──▶ routines (unattended)                       │
  └─────────────────────────────────────────────────────────────┘
```

---

## 3. The brain: which model, where

### 3.1 Cloud (the agent loop)

**Use `claude-opus-5`.** Not a preference — a measured requirement. The
predecessor project benchmarked two models on identical tool-calling tasks with
the full tool registry loaded:

| | fresh question | ten turns into a conversation |
|---|---|---|
| large model | 6/6 correct tool | **6/6** |
| small fast model | 6/6 correct tool | **2/6** |

The small model looked perfect one question at a time, then quietly stopped
calling tools mid-conversation and *invented* the answers — a fabricated battery
percentage, a clock time months off, "I've stored that" for something never
stored, and "I'll send that SMS" with no SMS sent. **For an assistant whose job
is reporting the state of the device in your hand, confident fabrication is the
worst possible failure.** Pay for the good model.

| Model | ID | Context | $/1M in | $/1M out | Use for |
|---|---|---|---|---|---|
| Claude Opus 5 | `claude-opus-5` | 1M | $5 | $25 | **The agent loop.** Default. |
| Claude Sonnet 5 | `claude-sonnet-5` | 1M | $3 | $15 | High-volume routines if cost bites |
| Claude Haiku 4.5 | `claude-haiku-4-5` | 200K | $1 | $5 | Classification, summarising notifications |

Key API settings:

```python
client.messages.create(
    model="claude-opus-5",
    max_tokens=16000,
    thinking={"type": "adaptive"},          # on by default for Opus 5
    output_config={"effort": "low"},         # "low" for voice; "high" for missions
    cache_control={"type": "ephemeral"},     # cache system prompt + tool defs
    system=SYSTEM_PROMPT,
    tools=TOOLS,
    messages=history,
)
```

- **`effort: "low"` for interactive voice.** A spoken reply that takes 12
  seconds is a failed reply. Use `"high"`/`"xhigh"` only for autonomous missions
  where nobody is waiting.
- **Prompt caching is not optional here.** The system prompt plus ~30 tool
  schemas is a large, byte-identical prefix on every single turn. Caching cuts
  it to ~10% cost. Keep the prefix stable — no timestamps, no UUIDs, sort your
  JSON, keep tool order deterministic.
- Verify it is working: `response.usage.cache_read_input_tokens` must be
  non-zero on the second turn. If it's zero, something in the prefix is
  changing per request.

### 3.2 Local (Ollama — the reflex layer)

Do **not** put Ollama in the agent loop. Give it the three jobs it is genuinely
good at on this hardware:

```bash
ollama pull qwen3:4b            # router + offline fallback  (~2.5 GB)
ollama pull nomic-embed-text    # semantic memory embeddings  (~270 MB)
```

1. **Router.** Classify each utterance: `trivial | device | complex | offline`.
   "What time is it" and "torch on" never need to leave the tablet. Target
   <500 ms. This is a few output tokens — well within budget.
2. **Embeddings** for semantic memory (§8). Fast, local, private.
3. **Offline fallback.** No network → degrade to local-only. The assistant
   should still set timers, toggle the torch, and answer from memory on a plane.

Note: Ollama under proot-Ubuntu carries overhead. If it's slow, run Ollama
directly in Termux instead of inside proot.

### 3.3 Speech

| Stage | Primary | Fallback | Notes |
|---|---|---|---|
| Wake word | **openWakeWord**, pretrained `hey_jarvis` model | Porcupine (free key, more accurate) | ONNX, CPU-light, runs continuously |
| STT | Groq Whisper API (fast, cheap) | `whisper.cpp` base.en local | `termux-speech-to-text` is the zero-effort option but needs network and is mediocre |
| TTS | **Piper** (local neural, ARM builds exist) | `termux-tts-speak` | Piper is a large quality jump and stays offline |

**Battery policy for the wake word** — this is the "don't destroy my device"
control:
- Always-on listening while **charging**.
- On battery: wake word active only while the screen is on, *or* bind to a
  hardware/Bixby button, *or* a home-screen widget.
- Make it one config flag. Measure actual drain over a day before committing.

---

## 4. What "max potential" actually means here

Ranked by how much they change the experience:

1. **Real UI control via local ADB** (§6) — the difference between an assistant
   that *talks about* your apps and one that *uses* them.
2. **Wake word** — hands-free is the whole point of a Jarvis.
3. **Semantic memory** — it remembers you without you managing it.
4. **Autonomous missions** — give it a goal, walk away.
5. **Vision on the screen** — screenshot → model → tap. Handles apps that have
   no deep links.

---

## 5. Agent core

A hand-written tool loop, not a framework, because two things must happen inline:
every step streams to the UI as it occurs, and dangerous tools are intercepted
*before* they execute.

```
user text
  → append to transcript
  → loop (max N rounds):
       stream completion with tools
       if no tool calls: done, speak/return answer
       for each tool call:
           if dangerous and not approved: ask, or refuse if unattended
           execute (concurrently — one turn may contain several calls)
           append tool result (errors included, as text the model can read)
  → out of rounds: one final pass with NO tools, forcing an answer from what
    was gathered
```

**Rules that matter:**

- **A failed tool returns its error as a tool result**, never an exception that
  kills the turn. The model routes around a failure it can read; it cannot route
  around a 500.
- **Never end a turn empty.** Running out of tool rounds must force a final
  tool-less answer summarising what was gathered. The alternative — observed in
  the predecessor — is the model burning every round improvising, then returning
  nothing at all.
- **Parallel tool calls**: one assistant turn may contain several. Execute them
  concurrently and return **all** results in a single user message. Splitting
  them trains the model to stop parallelising.
- **Approval gating**: SMS, calls, payments, anything irreversible. In the UI,
  ask. Unattended (routines, missions), refuse and say why — "blocked, nobody is
  watching" reads differently to the model than "the user declined", and it
  should.
- **Audit everything** to a SQLite `events` table. On-device, reviewable.

### 5.1 Missions (agent mode)

A chat turn is bounded. A mission is "keep working until done":

- Goal + step budget, persisted in SQLite so it survives Termux being killed.
- Each step is one ordinary agent turn in a session of its own, so the
  transcript accumulates and step *n+1* can see what step *n* did. No bespoke
  state machine.
- Three things end it: the agent declares completion, the budget runs out, or
  it's cancelled. **All three must be recorded.** A mission that stops early
  looks identical to one that succeeded unless you write down which happened.
- Use the API's **task budgets** so the model paces itself rather than being
  cut off mid-thought:

```python
client.beta.messages.stream(
    model="claude-opus-5", max_tokens=64000,
    output_config={"effort": "high",
                   "task_budget": {"type": "tokens", "total": 64000}},
    betas=["task-budgets-2026-03-13"],
    ...
)
```

---

## 6. Device & app control — three tiers

### Tier 0 — Termux:API (no setup beyond the app)

Speech, notifications, SMS, calls, camera, torch, clipboard, contacts, call log,
sensors, wifi/telephony info, volume, brightness, dialogs, media player. ~25
commands. This is the bread and butter.

### Tier 1 — Intents & deep links (no setup at all)

Launch any app by package, hand any app a URI, push into the share sheet, open
Settings screens. Cheap and surprisingly powerful — a deep link usually lands on
exactly the screen a tap was for:

```
https://wa.me/<number>?text=<msg>   WhatsApp chat, prefilled
google.navigation:q=<address>       turn-by-turn starts
spotify:track:<id>                  plays
tel:, mailto:, geo:, intent://...   the usual
am start -a android.settings.WIFI_SETTINGS
```

Launch by package with `monkey -p <pkg> -c android.intent.category.LAUNCHER 1`
— `am start` needs an activity name that nothing reports conveniently.

### Tier 2 — Local ADB over wireless debugging ★ the unlock

**Android 11+ can run `adb` against itself — no root, no PC.** This is what
turns "control apps" from marketing into fact.

Setup (one time, plus a re-connect after each reboot):

```bash
pkg install android-tools
# Settings → Developer options → Wireless debugging → ON
# → "Pair device with pairing code" gives HOST:PORT + a 6-digit code
adb pair localhost:<pair_port>        # enter the code
adb connect localhost:<connect_port>  # different port from the pairing one
adb devices                           # must show "device", not "unauthorized"
```

What that buys:

| Capability | Command |
|---|---|
| Tap | `input tap X Y` |
| Swipe / scroll | `input swipe X1 Y1 X2 Y2 <ms>` |
| Type text | `input text "..."` |
| Hardware keys | `input keyevent KEYCODE_BACK` / `HOME` / `ENTER` |
| **Read the screen** | `uiautomator dump /sdcard/ui.xml` → parse for clickable nodes + bounds |
| **Screenshot** | `screencap -p /sdcard/s.png` |
| App state | `dumpsys window \| grep mCurrentFocus` |

**Two ways to drive the UI, and you want both:**

1. **Accessibility-tree first** (cheap, reliable, deterministic).
   `uiautomator dump` returns every visible node with text, content-desc,
   `clickable`, and pixel `bounds`. Find the node whose text matches, compute
   its centre, tap it. No vision model, no tokens, no ambiguity. **This should
   be the default path.**
2. **Vision fallback** (expensive, general). `screencap` → send the PNG to
   `claude-opus-5` as an image → ask for the coordinates to tap. Use only when
   the tree is unhelpful (canvas apps, games, custom-rendered UI).

**Known sharp edges:**
- The **connect port changes on every reboot**; the pairing survives. Write a
  reconnect helper that discovers the port via mDNS
  (`_adb-tls-connect._tcp`) and re-runs `adb connect`. Expect to run it after
  each reboot; wire it into the startup script.
- Wireless debugging can switch itself off after a reboot on some builds. Detect
  and tell the user rather than silently losing Tier 2.
- Everything here **must be gated and audited.** A model that can tap anything
  can tap "Confirm payment". Screen-driving tools go behind the same approval
  gate as SMS, plus a per-app allowlist.

### Tier 3 — Root

Not recommended. Voids the warranty, breaks Samsung Knox and banking apps
permanently, and buys little over Tier 2. Skip it.

### What is genuinely impossible

- Injecting touches **without** ADB or root (`INJECT_EVENTS` is signature-level).
- Reading other apps' private storage.
- Anything in a banking app with screenshot protection — `screencap` returns
  black by design.

---

## 7. What to build the server as

- **FastAPI + uvicorn**, WebSocket for streaming chat, REST for scripts and
  Tasker.
- Bind `127.0.0.1` by default.
- **Add token auth from day one.** The predecessor shipped without it, and
  `RAXIT_HOST=0.0.0.0` then meant "full device control, no password, to anyone
  on the wifi". A shared secret in a header is twenty lines. Do it in Phase 1,
  not "later".
- A single-page web UI, added to the home screen for a full-screen app.

---

## 8. Memory — three tiers

```
working    │ the transcript, per session, replayed to the model
episodic   │ an events/audit log: what it did, when, and whether it worked
semantic   │ curated facts + embeddings, searchable by meaning
```

**SQLite for all three.** One file, no server, survives process death (Termux
kills long-running processes constantly — design for it).

**Semantic recall is the part worth getting right.** The predecessor matched
facts by word overlap, which meant `favourite_drink` was found by "what do I
like to drink" but *not* by "what's my caffeine situation" — no shared word.
A memory that silently fails to recall is worse than no memory, because the
user stops trusting it.

Fix: embed every fact with `nomic-embed-text` on write, cosine-search on read,
and keep the keyword match as a union (hybrid retrieval). Both are local, fast,
and private.

**Transcript repair matters more than it sounds.** You keep a tail of the last N
messages. That slice can open mid-tool-round, or contain a round where only some
tool calls got their replies (the process died between two writes). The API
rejects both. Write a repair pass that drops incomplete rounds *anywhere* in the
slice, not just at the end — a broken round in the middle gets reloaded every
turn and breaks the session permanently.

---

## 9. Tool discipline — read this before adding tools

**Tool-calling accuracy degrades as the tool list grows.** Measured on the
predecessor: the same model scored 6/6 with a handful of tools loaded and 2/6
with twenty-two. Twenty-five new capabilities added naively would have made the
assistant *worse*.

Three mitigations, use all of them:

1. **Group related commands behind one tool with a `kind` enum.** One
   `device_info(kind)` covering five binaries costs the model one decision
   instead of five. Same for sensors, media, settings.
2. **Use the API's tool search with deferred loading** — this is the real
   answer at scale, and it is native:

```python
tools = [
    {"type": "tool_search_tool_bm25_20251119", "name": "tool_search_tool_bm25"},
    {"name": "battery", ..., "defer_loading": True},
    {"name": "tap",     ..., "defer_loading": True},
    # ... 40 more, all deferred
    {"name": "speak",   ...},   # keep a few hot ones non-deferred
]
```

Deferred tools aren't in context until the model searches for them. This is how
you get to 50+ tools without the collapse. **At least one tool must be
non-deferred and the search tool itself must never be deferred** — otherwise the
API returns 400.

3. **Write descriptions for a stranger.** The description is the *only* thing
   the model sees when choosing. "Read the battery" is worse than "Read battery
   percentage, charging state, temperature and health." Say when to use it and
   when not to.

**Generate the schema and the dispatch entry from one declaration** (a `@tool`
decorator) so they cannot drift. A schema advertising a parameter the function
doesn't accept becomes a `TypeError` *after* the model has already spent a
request deciding to use it. Add a test that checks every schema against the real
function signature.

**One more, learned by benchmarking wrong:** benchmark tool choice with the
**real, full registry**. With four tools loaded, the weak model looked fine. The
failure only appeared at twenty-two. And test in a *continuing conversation*,
not a fresh session per question — the degradation lives in the conversation,
not the question.

---

## 10. MCP — where it fits

MCP is worth adding, but **not** by bolting on every server you can find (§9).

- **For Claude Code, on the tablet:** MCP servers make *your development* better
  — filesystem, git, fetch. Unrelated to the runtime.
- **For Jarvis at runtime:** use the MCP connector for genuinely external
  systems — your calendar, notes, home automation, a music service. Requires
  both halves or it 400s:

```python
mcp_servers=[{"type": "url", "url": "...", "name": "calendar"}],
tools=[{"type": "mcp_toolset", "mcp_server_name": "calendar"}],
betas=["mcp-client-2025-11-20"],
```

- Also declare the server tools you want: `web_search_20260209` and
  `web_fetch_20260209` give it live information without you writing a scraper.

Add MCP in Phase 5, not before. Device control first.

---

## 11. Build phases

Each phase ends with acceptance criteria that must pass **on the tablet**.

### Phase 1 — Skeleton that talks (day 1)
Agent loop, tool registry with the `@tool` decorator, SQLite transcript,
FastAPI + WebSocket, web UI, token auth, `speak`/`listen`/`battery`/`now`.

✅ *Accept:* Ask "how much battery?" in the browser and hear the correct,
tool-sourced answer spoken aloud. Kill and restart the process; the
conversation is still there.

### Phase 2 — The device (day 2)
Full Termux:API surface, grouped per §9. Contacts, notifications, SMS/calls
behind the approval gate, sensors, clipboard, media, camera.

✅ *Accept:* "Text Amma that I'm running late" resolves the name from contacts,
asks for approval, and sends only after you allow it. Denying it produces a
spoken explanation, not silence.

### Phase 3 — Voice, hands-free (day 3)
openWakeWord `hey_jarvis`, STT, Piper TTS, the battery policy, barge-in
(interrupt playback when you start talking).

✅ *Accept:* Say "Hey Jarvis" across the room with the screen off; it wakes,
listens, acts, and speaks. Overnight battery drain measured and acceptable.

### Phase 4 — App control ★ (days 4–5)
Intents and deep links. Then local ADB: pair, connect, reconnect helper.
`tap` / `swipe` / `type` / `screenshot` / `read_screen` via `uiautomator`,
approval-gated with a per-app allowlist. Vision fallback last.

✅ *Accept:* "Open WhatsApp and message Ravi that I'll be 10 minutes late" works
end to end. Then reboot the tablet and confirm the reconnect helper restores
Tier 2 without you looking up a port.

### Phase 5 — Memory & missions (days 6–7)
Embeddings, hybrid retrieval, transcript repair. Missions with budgets,
persistence, and cancellation. Routines on cron.

✅ *Accept:* Tell it a preference on Monday, ask for it on Wednesday *in
different words*, and it recalls. Give it a 5-step mission; it finishes, or runs
out and says exactly what it did and didn't do.

### Phase 6 — Polish
MCP connectors, web search, proactive notifications, Tasker/Bixby button
binding, cost dashboard, a `/health` endpoint.

---

## 12. Repo layout

```
jarvis/
  agent.py          tool loop, streaming, approval gating
  llm.py            the ONLY provider-aware file — swap endpoints here
  router.py         local Ollama classification + offline fallback
  memory.py         transcript, facts, embeddings, audit log
  missions.py       goal loop, budgets, persistence
  routines.py       YAML cron routines
  server.py         FastAPI + WebSocket + REST + auth
  voice/
    wake.py         openWakeWord
    stt.py          whisper / Groq
    tts.py          Piper / termux-tts
  tools/
    registry.py     @tool decorator → schema + dispatch, one declaration
    termux.py       termux-api wrappers
    apps.py         intents, deep links, launch, share
    screen.py       ADB: tap, swipe, type, screencap, uiautomator
    system.py       memory, shell, http, time
routines/           your scheduled prompts, in English
web/index.html      the UI
scripts/
  install.sh        one-time setup
  run.sh            start (takes a wake lock)
  adb-reconnect.sh  post-reboot Tier 2 restore
tests/              pytest — no device, no network needed
```

Keep `llm.py` as the single provider-aware file. Everything else should not know
which model is answering.

---

## 13. Setup

```bash
# Termux and Termux:API from F-Droid — NOT the Play Store builds.
pkg update -y
pkg install -y python termux-api git rust binutils android-tools libexpat openssl
termux-setup-storage

python -m venv .venv && . .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

termux-wake-lock          # or routines silently stop firing
./scripts/run.sh
```

`requirements.txt` — note every line has a reason (§14):

```
anthropic>=1.0            # the SDK; NOT an OpenAI-compatible shim
fastapi
uvicorn                   # plain — NOT uvicorn[standard], see §14
websockets                # uvicorn has no ws implementation without it
pydantic
httpx
apscheduler
PyYAML
python-dotenv
tzdata                    # Android ships no timezone database, see §14
openwakeword
piper-tts
numpy
```

---

## 14. Android/Termux gotchas — hard-won, do not rediscover

Each of these cost real time on the predecessor. They are not hypothetical.

1. **`tzdata` — the one that stops the server booting at all.**
   Android ships no IANA timezone database, so `zoneinfo.ZoneInfo("Asia/Kolkata")`
   raises `ZoneInfoNotFoundError`. APScheduler resolves the timezone in its
   constructor, at import, so the process dies before binding the port — `curl`
   just says connection refused. `pip install tzdata`. Pure data, no build.

2. **Never `uvicorn[standard]`.** The extra pulls `uvloop`, `httptools` and
   `watchfiles` — none publish wheels Termux can use, so pip builds from source
   and `watchfiles` wants a Rust toolchain. Use plain `uvicorn` **plus an
   explicit `websockets`**, or the chat socket's upgrade is rejected outright.

3. **Rust is unavoidable, so install it up front.** `pydantic-core` and (if you
   use the OpenAI SDK) `jiter` are Rust extensions with no Android wheels.
   `pkg install rust binutils` *before* pip. If maturin picks the wrong target,
   `export CARGO_BUILD_TARGET=aarch64-linux-android`. Expect 10–30 minutes of
   compiling; silence is not a hang.

4. **Termux:API is two things.** The `termux-api` *package* gives you the
   binaries; the Termux:API *app* from F-Droid makes them work. With only one,
   every device tool fails at runtime with no obvious cause. Verify with
   `termux-battery-status` and check it at install time.

5. **There is no `/tmp`.** Writes to `/tmp/x` fail silently-ish
   (`curl: (23) client returned ERROR on write`). Use `$HOME` or `$PREFIX/tmp`.

6. **Activate the venv in every new Termux session.** A second tab has no
   `(.venv)` and `pip install` there goes to the system Python while your app
   can't see it. Put the activation in `run.sh` and in `.bashrc`.

7. **Wake lock or your routines stop.** `termux-wake-lock` in `run.sh`, always.
   Android suspends the process otherwise and nothing fires.

8. **"Address already in use" means an old instance is alive** in another tab.
   `pkill -f jarvis.server` before starting. Have `run.sh` do it.

9. **TTS goes to the notification stream by default.** `termux-tts-speak -s
   NOTIFICATION` is silent if notification volume is down — which it often is,
   independently of the volume rocker. Use `-s MUSIC` for spoken replies, and if
   it's silent, check the *stream*, not the engine.

10. **Termux ships Python 3.14.** Test against it. Wheels for 3.14 on Android
    are scarce; that's why §14.3 matters.

11. **`.env` edits in `nano` can silently not save**, and a placeholder key gets
    sent as a literal bearer token → `403 Authorization failed`. Prefer
    `sed -i` and verify with a hash, not by eyeballing.

---

## 15. Cost

At `effort: "low"` with prompt caching on, a typical tool-using turn is roughly
2–6k cached input + ~1k fresh + ~500 output. Ballpark **$0.02–0.05 per
interaction**, i.e. a few dollars a month for personal use. Missions at
`effort: "high"` cost meaningfully more — that's what the budget cap is for.

Three levers, in order of effect: **prompt caching** (biggest), **effort**
(next), **route trivial requests to the local model so they cost nothing at
all** (free, and also works offline).

---

## 16. Safety

- Approval gate on everything irreversible: SMS, calls, payments, **and every
  screen-driving tool**. A model that can tap can tap "Confirm".
- Per-app allowlist for Tier 2 control. Banking and payment apps denied by
  default, no exceptions and no override flag.
- Shell restricted to an allowlist of read-only binaries. Execute as an argv
  list, never `shell=True` — otherwise `date; rm -rf ~` becomes a working
  deletion.
- Token auth on the server before it ever binds to anything but localhost.
- Everything the agent does goes to the audit log, reviewable in the UI.
- Memory lives only on the tablet. Only prompts go to the model provider — say
  so plainly in the README, since the thing has your SMS and contacts.

---

## 17. Definition of done

You say **"Hey Jarvis"** across the room with the tablet face-down and charging.
It wakes, hears you, decides whether it needs the cloud, uses the right tools,
does the thing — including tapping through an app that has no deep link — tells
you in a sentence, remembers what matters for next week, and it all still works
after a reboot without you opening a terminal.
