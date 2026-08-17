"""Memory has two jobs that fail quietly when they fail: recall that finds
nothing when it should have found something, and a transcript slice the API
rejects. Both are covered here, since neither shows up as an obvious crash.
"""

from __future__ import annotations

from raxit import memory
from raxit.memory import _repair


def assistant(text: str = "ok", calls: list[dict] | None = None) -> dict:
    message: dict = {"role": "assistant", "content": text}
    if calls:
        message["tool_calls"] = calls
    return message


def call(call_id: str, name: str = "battery") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": "{}"},
    }


def tool_reply(call_id: str, content: str = "83%") -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def user(text: str = "hi") -> dict:
    return {"role": "user", "content": text}


# --- facts -------------------------------------------------------------------


def test_remember_and_recall_round_trip():
    memory.remember("home_wifi", "Airtel_5G", tags="network")
    assert memory.recall("wifi") == [
        {"key": "home_wifi", "value": "Airtel_5G", "tags": "network"}
    ]


def test_remember_overwrites_an_existing_key():
    memory.remember("mood", "tired")
    memory.remember("mood", "fine")
    facts = memory.recall("mood")
    assert len(facts) == 1 and facts[0]["value"] == "fine"


def test_recall_folds_underscores_so_a_stored_key_matches_spoken_words():
    """The model stores `favourite_drink` and later searches for the phrase a
    person would say. A substring match returns nothing for that, which reads
    to the user as the agent having forgotten."""
    memory.remember("favourite_drink", "black coffee, no sugar")
    assert memory.recall("what does he like to drink")


def test_recall_matches_hyphens_and_commas_the_same_way():
    memory.remember("wake-up-time", "6:30am")
    assert memory.recall("wake up time")
    assert memory.recall("wake,up")


def test_recall_searches_values_and_tags_not_just_keys():
    memory.remember("k1", "he is allergic to peanuts", tags="health")
    assert memory.recall("peanuts")
    assert memory.recall("health")


def test_recall_ranks_by_number_of_matching_words():
    memory.remember("one_hit", "coffee")
    memory.remember("two_hits", "black coffee")
    assert [f["key"] for f in memory.recall("black coffee")][0] == "two_hits"


def test_recall_ignores_single_character_noise_words():
    """Without this, "a" matches every fact that contains the letter a and
    recall degenerates into dumping the whole store."""
    memory.remember("unrelated", "xyz")
    memory.remember("target", "quiche")
    assert [f["key"] for f in memory.recall("a quiche")] == ["target"]


def test_recall_with_no_query_returns_everything_newest_first():
    memory.remember("first", "1")
    memory.remember("second", "2")
    assert [f["key"] for f in memory.recall()] == ["second", "first"]


def test_recall_respects_its_limit():
    for i in range(10):
        memory.remember(f"k{i}", "coffee")
    assert len(memory.recall("coffee", limit=3)) == 3
    assert len(memory.recall("", limit=3)) == 3


def test_recall_returns_nothing_when_nothing_matches():
    memory.remember("k", "v")
    assert memory.recall("kangaroo") == []


def test_forget_reports_whether_it_deleted_anything():
    memory.remember("temp", "v")
    assert memory.forget("temp") is True
    assert memory.forget("temp") is False
    assert memory.recall("temp") == []


# --- transcript --------------------------------------------------------------


def test_messages_round_trip_through_sqlite():
    memory.append_message("s", user("what's the battery"))
    memory.append_message("s", assistant("83%"))
    assert memory.load_messages("s") == [user("what's the battery"), assistant("83%")]


def test_sessions_do_not_leak_into_each_other():
    memory.append_message("a", user("secret"))
    memory.append_message("b", user("other"))
    assert memory.load_messages("b") == [user("other")]


def test_load_keeps_the_newest_messages_not_the_oldest():
    """Trimming from the wrong end would hand the model a stale conversation
    and drop the turn it is actually answering."""
    for i in range(10):
        memory.append_message("s", user(f"m{i}"))
    tail = memory.load_messages("s", limit=3)
    assert [m["content"] for m in tail] == ["m7", "m8", "m9"]


def test_clear_session_empties_it():
    memory.append_message("s", user())
    memory.clear_session("s")
    assert memory.load_messages("s") == []


def test_tool_calls_survive_the_json_round_trip():
    memory.append_message("s", user())
    memory.append_message("s", assistant(None, [call("c1")]))
    memory.append_message("s", tool_reply("c1"))
    loaded = memory.load_messages("s")
    assert loaded[1]["tool_calls"][0]["function"]["name"] == "battery"


# --- transcript repair -------------------------------------------------------


def test_repair_keeps_a_complete_tool_round():
    msgs = [user(), assistant(None, [call("c1")]), tool_reply("c1"), assistant("83%")]
    assert _repair(list(msgs)) == msgs


def test_repair_drops_a_head_that_opens_mid_tool_round():
    """A tail slice can start on a `tool` message whose assistant turn was
    trimmed away. The API rejects that outright."""
    msgs = [tool_reply("c1"), assistant("83%"), user("thanks"), assistant("np")]
    assert _repair(msgs) == [user("thanks"), assistant("np")]


def test_repair_drops_a_head_that_opens_on_an_orphaned_assistant_turn():
    msgs = [assistant("dangling"), user("hi"), assistant("hello")]
    assert _repair(msgs) == [user("hi"), assistant("hello")]


def test_repair_drops_a_tail_whose_tool_calls_were_never_answered():
    """What a crash mid-loop leaves behind: Termux kills the process between
    the assistant asking for a tool and the result coming back."""
    msgs = [user(), assistant("83%"), assistant(None, [call("c9")])]
    assert _repair(msgs) == [user(), assistant("83%")]


def test_repair_drops_a_round_where_only_some_calls_were_answered():
    """Results are appended one at a time, so a kill between two of them
    leaves a round asking for two tools and answering one.

    Unlike a dangling turn at the very end, this one is not self-healing: it
    sits mid-transcript, so every later turn reloads it and the API rejects
    the history again. The session stays broken until the round is dropped.
    """
    msgs = [
        user(),
        assistant(None, [call("c1"), call("c2")]),
        tool_reply("c1"),
        user("still there?"),
    ]
    assert _repair(msgs) == [user(), user("still there?")]


def test_repair_drops_a_tool_reply_with_no_call_to_answer():
    msgs = [user(), tool_reply("ghost"), assistant("hello")]
    assert _repair(msgs) == [user(), assistant("hello")]


def test_repair_unwinds_several_dangling_turns():
    msgs = [user(), assistant(None, [call("c1")]), assistant(None, [call("c2")])]
    assert _repair(msgs) == [user()]


def test_repair_returns_nothing_when_there_is_no_user_turn_to_start_from():
    assert _repair([assistant("orphan"), tool_reply("c1")]) == []
    assert _repair([]) == []


def test_load_messages_repairs_what_it_loads():
    memory.append_message("s", user())
    memory.append_message("s", assistant(None, [call("c1")]))
    assert memory.load_messages("s") == [user()]


# --- activity log ------------------------------------------------------------


def test_events_come_back_newest_first():
    memory.log_event("tool", "battery")
    memory.log_event("routine_start", "morning_brief")
    events = memory.recent_events()
    assert [e["kind"] for e in events] == ["routine_start", "tool"]
    assert events[0]["detail"] == "morning_brief"


def test_events_respect_their_limit():
    for i in range(5):
        memory.log_event("tool", str(i))
    assert len(memory.recent_events(limit=2)) == 2


def test_init_is_idempotent():
    memory.init()
    memory.init()
    memory.remember("k", "v")
    assert memory.recall("k")
