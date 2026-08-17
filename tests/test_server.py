"""The HTTP and WebSocket surface.

The websocket tests are the interesting ones: approval is a round trip
mid-turn — the server pauses a tool call, asks the browser, and resumes on
the answer — and that handshake is what stands between the model and an SMS.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from raxit import memory, server
from raxit.agent import Event


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Any:
    """A test client with the scheduler stubbed out.

    Starting APScheduler inside the test client leaves a live timer thread
    behind; the routine tests cover scheduling on its own.
    """
    monkeypatch.setattr(server.runner, "start", lambda: None)
    monkeypatch.setattr(server.runner, "stop", lambda: None)
    with TestClient(server.app) as test_client:
        yield test_client


@pytest.fixture
def fake_agent(monkeypatch: pytest.MonkeyPatch):
    """Drive the server with scripted agent events instead of a provider."""

    class Fake:
        def __init__(self) -> None:
            self.events: list[Event] = [Event("done", {"text": "ok"})]
            self.seen: list[dict] = []
            self.approve_with: dict | None = None

        async def run(
            self, session: str, message: str, *, unattended: bool = False, **kwargs
        ) -> AsyncIterator[Event]:
            self.seen.append(
                {"session": session, "message": message, "unattended": unattended}
            )
            approve = kwargs.get("approve")
            if self.approve_with is not None and approve is not None:
                allowed = await approve(*self.approve_with["args"])
                self.approve_with["allowed"] = allowed
            for event in self.events:
                yield event

    fake = Fake()
    monkeypatch.setattr(server, "agent", fake)
    return fake


# --- status ------------------------------------------------------------------


def test_status_describes_the_running_configuration(client):
    body = client.get("/api/status").json()

    assert body["model"] and body["base_url"]
    assert len(body["tools"]) == 34
    assert isinstance(body["routines"], list)


def test_status_lists_tools_alphabetically(client):
    tools = client.get("/api/status").json()["tools"]
    assert tools == sorted(tools)


def test_status_carries_the_recent_activity_feed(client):
    memory.log_event("tool", "battery")
    assert client.get("/api/status").json()["events"][0]["detail"] == "battery"


def test_the_web_ui_is_served_at_the_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "<html" in response.text.lower()


# --- memory ------------------------------------------------------------------


def test_memory_returns_stored_facts(client):
    memory.remember("home_wifi", "Airtel_5G")
    assert client.get("/api/memory").json()["facts"][0]["key"] == "home_wifi"


def test_memory_can_be_searched(client):
    memory.remember("home_wifi", "Airtel_5G")
    memory.remember("favourite_drink", "coffee")

    facts = client.get("/api/memory", params={"q": "drink"}).json()["facts"]

    assert [f["key"] for f in facts] == ["favourite_drink"]


def test_memory_is_empty_before_anything_is_remembered(client):
    assert client.get("/api/memory").json()["facts"] == []


# --- routines ----------------------------------------------------------------


def test_reloading_returns_the_new_schedule(client, monkeypatch):
    monkeypatch.setattr(server.runner, "load", lambda: None)
    monkeypatch.setattr(server.runner, "describe", lambda: [{"name": "morning_brief"}])

    body = client.post("/api/routines/reload").json()

    assert body["routines"] == [{"name": "morning_brief"}]


def test_running_a_routine_returns_its_result(client, monkeypatch):
    async def fire(name: str) -> str:
        return f"ran {name}"

    monkeypatch.setattr(server.runner, "routines", {"morning_brief": object()})
    monkeypatch.setattr(server.runner, "fire", fire)

    assert client.post("/api/routines/morning_brief/run").json() == {
        "result": "ran morning_brief"
    }


def test_running_an_unknown_routine_is_a_404_not_a_500(client, monkeypatch):
    monkeypatch.setattr(server.runner, "routines", {})

    response = client.post("/api/routines/nope/run")

    assert response.status_code == 404
    assert "nope" in response.json()["detail"]


# --- sessions ----------------------------------------------------------------


def test_clearing_a_session_drops_its_transcript(client):
    memory.append_message("kitchen", {"role": "user", "content": "hi"})

    assert client.post("/api/session/kitchen/clear").json() == {"cleared": "kitchen"}
    assert memory.load_messages("kitchen") == []


def test_clearing_one_session_leaves_the_others(client):
    memory.append_message("a", {"role": "user", "content": "keep me"})
    memory.append_message("b", {"role": "user", "content": "drop me"})

    client.post("/api/session/b/clear")

    assert len(memory.load_messages("a")) == 1


# --- REST chat ---------------------------------------------------------------


def test_chat_returns_the_reply_and_the_tools_it_used(client, fake_agent):
    fake_agent.events = [
        Event("tool_result", {"name": "battery", "output": "83%"}),
        Event("done", {"text": "83% and charging."}),
    ]

    body = client.post("/api/chat", json={"message": "battery?"}).json()

    assert body == {"reply": "83% and charging.", "tools_used": ["battery"]}


def test_chat_runs_unattended_because_there_is_no_one_to_ask(client, fake_agent):
    """A curl one-liner has no channel for an approval prompt, so dangerous
    tools must be refused rather than silently allowed."""
    client.post("/api/chat", json={"message": "text him"})

    assert fake_agent.seen[0]["unattended"] is True


def test_chat_defaults_to_the_shared_session(client, fake_agent):
    client.post("/api/chat", json={"message": "hi"})
    assert fake_agent.seen[0]["session"] == "default"


def test_chat_honours_an_explicit_session(client, fake_agent):
    client.post("/api/chat", json={"message": "hi", "session": "tasker"})
    assert fake_agent.seen[0]["session"] == "tasker"


def test_chat_surfaces_an_agent_error_as_a_500(client, fake_agent):
    fake_agent.events = [Event("error", {"message": "rate limited"})]

    response = client.post("/api/chat", json={"message": "hi"})

    assert response.status_code == 500
    assert response.json()["error"] == "rate limited"


def test_chat_requires_a_message(client, fake_agent):
    assert client.post("/api/chat", json={}).status_code == 422


# --- websocket chat ----------------------------------------------------------


def test_the_websocket_streams_every_event_in_order(client, fake_agent):
    fake_agent.events = [
        Event("text", {"text": "83"}),
        Event("text", {"text": "%"}),
        Event("done", {"text": "83%"}),
    ]

    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"message": "battery?"}))
        received = [ws.receive_json() for _ in range(3)]

    assert [event["type"] for event in received] == ["text", "text", "done"]
    assert received[-1]["data"]["text"] == "83%"


def test_a_dangerous_tool_pauses_the_turn_and_asks_the_browser(client, fake_agent):
    """The approval round trip: the server sends `approval_request`, the turn
    blocks, and the client's answer resumes it."""
    fake_agent.approve_with = {"args": ("send_sms", {"number": "+91"})}
    fake_agent.events = [Event("done", {"text": "sent"})]

    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"message": "text him"}))
        request = ws.receive_json()
        assert request["type"] == "approval_request"
        assert request["data"] == {"name": "send_sms", "input": {"number": "+91"}}
        ws.send_text(json.dumps({"type": "approval", "approved": True}))
        assert ws.receive_json()["type"] == "done"

    assert fake_agent.approve_with["allowed"] is True


def test_a_refusal_comes_back_as_false(client, fake_agent):
    fake_agent.approve_with = {"args": ("call", {"number": "+91"})}

    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"message": "call him"}))
        ws.receive_json()
        ws.send_text(json.dumps({"type": "approval", "approved": False}))
        ws.receive_json()

    assert fake_agent.approve_with["allowed"] is False


def test_a_missing_approved_field_counts_as_a_refusal(client, fake_agent):
    """Anything other than an explicit yes is a no — this gate exists to stop
    an SMS leaving the device."""
    fake_agent.approve_with = {"args": ("send_sms", {})}

    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"message": "text him"}))
        ws.receive_json()
        ws.send_text(json.dumps({"type": "approval"}))
        ws.receive_json()

    assert fake_agent.approve_with["allowed"] is False


def test_an_empty_message_is_ignored_rather_than_sent(client, fake_agent):
    """The mic returns an empty string when it hears nothing, and an empty
    turn still costs a request against a 40-per-minute budget."""
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"message": "   "}))
        ws.send_text(json.dumps({"message": "real question"}))
        ws.receive_json()

    assert [turn["message"] for turn in fake_agent.seen] == ["real question"]


def test_the_socket_stays_open_for_a_second_turn(client, fake_agent):
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"message": "first"}))
        ws.receive_json()
        ws.send_text(json.dumps({"message": "second"}))
        ws.receive_json()

    assert [turn["message"] for turn in fake_agent.seen] == ["first", "second"]


def test_a_turn_that_raises_is_reported_without_dropping_the_socket(
    client, fake_agent
):
    async def explode(*args, **kwargs):
        raise RuntimeError("provider down")
        yield  # pragma: no cover - makes this an async generator

    fake_agent.run = explode

    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"message": "hi"}))
        received = ws.receive_json()

    assert received["type"] == "error"
    assert "provider down" in received["data"]["message"]


def test_the_websocket_defaults_to_the_shared_session(client, fake_agent):
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"message": "hi"}))
        ws.receive_json()

    assert fake_agent.seen[0]["session"] == "default"
