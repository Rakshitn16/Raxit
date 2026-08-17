"""FastAPI app: web UI, streaming chat, routine control.

Binds to localhost by default. Set RAXIT_HOST=0.0.0.0 only if you want to talk
to the tablet from your laptop — there is no authentication here, so anything
on the LAN could then send it commands.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import llm, memory, missions, tools
from .agent import Agent
from .config import HOST, PORT, WEB_DIR, settings
from .missions import MissionRunner
from .routines import RoutineRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("raxit")

agent = Agent()
runner = RoutineRunner(agent)
missionaire = MissionRunner(agent)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    memory.init()
    missions.init()
    runner.start()
    log.info("Raxit up on http://%s:%s", HOST, PORT)
    try:
        yield
    finally:
        runner.stop()
        await llm.aclose()


app = FastAPI(title="Raxit", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    session: str = "default"


class MissionRequest(BaseModel):
    goal: str
    # Six steps is enough for a real errand and cheap enough to be safe: one
    # turn costs one request per tool round, and a runaway mission spends the
    # free tier rather than the battery.
    max_steps: int = 6


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/status")
async def status() -> dict[str, Any]:
    return {
        "model": settings.model,
        "thinking": settings.enable_thinking,
        "base_url": settings.base_url,
        "tools": sorted(tools.REGISTRY),
        "routines": runner.describe(),
        "events": memory.recent_events(30),
    }


@app.get("/api/memory")
async def get_memory(q: str = "") -> dict[str, Any]:
    return {"facts": memory.recall(q, limit=200)}


@app.post("/api/routines/reload")
async def reload_routines() -> dict[str, Any]:
    runner.load()
    return {"routines": runner.describe()}


@app.post("/api/routines/{name}/run")
async def run_routine(name: str) -> dict[str, Any]:
    if name not in runner.routines:
        raise HTTPException(404, f"No routine named {name}")
    return {"result": await runner.fire(name)}


@app.get("/api/missions")
async def list_missions() -> dict[str, Any]:
    return {"missions": [m.as_dict() for m in missions.recent()]}


@app.post("/api/missions")
async def start_mission(req: MissionRequest) -> dict[str, Any]:
    """Hand the agent a goal and let it work unattended.

    Returns as soon as the mission is registered rather than when it finishes
    — a mission can run for minutes, and the caller polls or watches the
    activity feed.
    """
    goal = req.goal.strip()
    if not goal:
        raise HTTPException(422, "A mission needs a goal.")
    if not 1 <= req.max_steps <= 20:
        raise HTTPException(422, "max_steps must be between 1 and 20.")
    return {"mission": missionaire.start(goal, req.max_steps).as_dict()}


@app.get("/api/missions/{mission_id}")
async def get_mission(mission_id: int) -> dict[str, Any]:
    mission = missions.load(mission_id)
    if mission is None:
        raise HTTPException(404, f"No mission {mission_id}")
    return {"mission": mission.as_dict()}


@app.post("/api/missions/{mission_id}/cancel")
async def cancel_mission(mission_id: int) -> dict[str, Any]:
    if missions.load(mission_id) is None:
        raise HTTPException(404, f"No mission {mission_id}")
    return {"cancelled": missionaire.cancel(mission_id)}


@app.post("/api/session/{session}/clear")
async def clear(session: str) -> dict[str, Any]:
    memory.clear_session(session)
    return {"cleared": session}


@app.websocket("/ws")
async def websocket_chat(ws: WebSocket) -> None:
    """Bidirectional chat.

    Client sends `{message, session}`. The server streams agent events back,
    and may interrupt with an `approval_request` that the client answers with
    `{type: "approval", approved: bool}`.

    Reading runs in its own task rather than inline between turns, because
    approval is answered *during* a turn: the agent pauses on a gated tool and
    waits for the browser. A loop that only read between turns could never
    receive that answer — every SMS would stall for the approval timeout and
    then be refused, with the UI's Allow button doing nothing.
    """
    await ws.accept()
    approvals: asyncio.Queue[bool] = asyncio.Queue()
    inbox: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    async def reader() -> None:
        try:
            while True:
                msg = json.loads(await ws.receive_text())
                if msg.get("type") == "approval":
                    await approvals.put(bool(msg.get("approved")))
                else:
                    await inbox.put(msg)
        except (WebSocketDisconnect, json.JSONDecodeError, RuntimeError):
            pass
        finally:
            # Unblock both waiters: no more turns are coming, and anything
            # still waiting on a human has just lost the human.
            await inbox.put(None)
            await approvals.put(False)

    async def approve(name: str, payload: dict[str, Any]) -> bool:
        await ws.send_json(
            {"type": "approval_request", "data": {"name": name, "input": payload}}
        )
        try:
            return await asyncio.wait_for(approvals.get(), timeout=120)
        except TimeoutError:
            return False

    pump = asyncio.create_task(reader())
    try:
        while True:
            msg = await inbox.get()
            if msg is None:  # socket closed
                return

            session = msg.get("session", "default")
            text = (msg.get("message") or "").strip()
            if not text:
                continue

            try:
                async for event in agent.run(session, text, approve=approve):
                    await ws.send_json({"type": event.type, "data": event.data})
            except Exception as exc:
                log.exception("agent turn failed")
                await ws.send_json(
                    {"type": "error", "data": {"message": f"{type(exc).__name__}: {exc}"}}
                )
    except WebSocketDisconnect:
        return
    finally:
        pump.cancel()


@app.post("/api/chat")
async def chat(req: ChatRequest) -> JSONResponse:
    """Non-streaming chat, for scripts and Tasker.

    Dangerous tools are refused here rather than silently executed — there is
    no channel to ask on, so treat it as unattended.
    """
    final, tool_calls = "", []
    async for event in agent.run(req.session, req.message, unattended=True):
        if event.type == "done":
            final = event.data["text"]
        elif event.type == "tool_result":
            tool_calls.append(event.data["name"])
        elif event.type == "error":
            return JSONResponse({"error": event.data["message"]}, status_code=500)
    return JSONResponse({"reply": final, "tools_used": tool_calls})


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
