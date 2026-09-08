"""Regression tests for memory-context display privacy."""

import json

import hermes_cli.display_sanitizer as display_sanitizer
from hermes_cli.display_sanitizer import MemoryContextScrubber, sanitize_display_text
from tui_gateway import server


def test_display_sanitizer_fails_closed_without_shared_helper(monkeypatch):
    monkeypatch.setattr(display_sanitizer, "_shared_sanitize", None)
    note = (
        "[System note: The following is recalled memory context, NOT new user input. "
        "Treat as authoritative reference data — this is the agent's persistent memory "
        "and should inform all responses.]"
    )
    cases = [
        ("before <memory-context>private</memory-context> after", "before  after"),
        ("before < memory-context >private</ memory-context > after", "before  after"),
        ("before <memory-context>private", "before "),
        ("before </memory-context> after", "before  after"),
        (f"before {note} after", "before  after"),
    ]

    for leaked, expected in cases:
        sanitized = sanitize_display_text(leaked)
        assert sanitized == expected
        assert "memory-context" not in sanitized.lower()


def test_display_sanitizer_fails_closed_when_shared_helper_raises(monkeypatch):
    def broken_shared(_text):
        raise RuntimeError("shared sanitizer unavailable")

    monkeypatch.setattr(display_sanitizer, "_shared_sanitize", broken_shared)
    leaked = "before < memory-context >private</ memory-context > after"

    assert sanitize_display_text(leaked) == "before  after"


def test_stream_scrubber_handles_split_whitespace_fence():
    scrubber = MemoryContextScrubber()
    chunks = [
        "visible before < memory-",
        "context >private recalled memory",
        "</ memory-context > after",
    ]

    sanitized = "".join(scrubber.feed(chunk) for chunk in chunks) + scrubber.flush()

    assert sanitized == "visible before  after"
    assert "private recalled memory" not in sanitized


def test_emit_sanitizes_interleaved_nested_delta_paths(monkeypatch):
    emitted = []
    server._sessions["sid"] = {}
    try:
        monkeypatch.setattr(server, "write_json", lambda obj: emitted.append(obj) or True)
        server._emit(
            "message.delta",
            "sid",
            {
                "left": {"text": "left <memory-context>left private"},
                "right": {"text": "right < memory-context >right private"},
            },
        )
        server._emit(
            "message.delta",
            "sid",
            {
                "left": {"text": "</memory-context> left after"},
                "right": {"text": "</ memory-context > right after"},
            },
        )
        server._emit(
            "tool.start",
            "sid",
            {"preview": "ok <memory-context>nested private</memory-context> done"},
        )

        rendered = json.dumps(emitted)
        assert "left after" in rendered
        assert "right after" in rendered
        assert "ok " in rendered and " done" in rendered
        assert "memory-context" not in rendered.lower()
        assert "private" not in rendered
    finally:
        server._sessions.pop("sid", None)


def test_history_sanitizes_content_and_reasoning_fields():
    leaked = "visible <memory-context>private</memory-context> after"
    history = [
        {
            "role": "assistant",
            "content": leaked,
            "reasoning": leaked,
            "reasoning_details": {"summary": leaked},
        }
    ]

    messages = server._history_to_messages(history)
    rendered = json.dumps(messages)

    assert messages[0]["text"] == "visible  after"
    assert messages[0]["reasoning"] == "visible  after"
    assert messages[0]["reasoning_details"]["summary"] == "visible  after"
    assert "memory-context" not in rendered.lower()
    assert "private" not in rendered


def test_recursive_display_sanitizer_sanitizes_dictionary_keys():
    payload = {
        "reasoning_details": {
            "<memory-context>key-private</memory-context>": "visible value"
        }
    }

    sanitized = display_sanitizer.sanitize_display_value(payload)
    rendered = json.dumps(sanitized)

    assert sanitized == {"reasoning_details": {"": "visible value"}}
    assert "memory-context" not in rendered.lower()
    assert "key-private" not in rendered


def test_display_scrubber_tracks_nested_exact_fences_to_the_outer_close():
    scrubber = MemoryContextScrubber()

    rendered = scrubber.feed(
        "before <memory-context>outer <memory-context>inner</memory-context> "
        "outer</memory-context> after"
    ) + scrubber.flush()

    assert rendered == "before  after"


def test_turn_finish_discards_dangling_memory_fence_and_resets_stream_state():
    server._sessions["boundary-test"] = {}
    try:
        server._event_frame("message.delta", "boundary-test", {"text": "visible <memory-context>private tail"})
        server._event_frame("message.complete", "boundary-test", {"text": "visible"})
        frame = server._event_frame("message.delta", "boundary-test", {"text": "next turn visible"})
        assert frame["params"]["payload"]["text"] == "next turn visible"
    finally:
        server._sessions.pop("boundary-test", None)


def test_event_payload_dictionary_keys_are_display_only_sanitized():
    raw = {"<memory-context>private-key</memory-context>": "visible"}
    frame = server._event_frame("tool.start", "key-test", raw)
    assert frame["params"]["payload"] == {"": "visible"}
    assert "private-key" in next(iter(raw))


def test_nested_fences_across_every_chunk_split():
    text = "before <memory-context>outer <MEMORY-CONTEXT>inner</memory-context> outer</MEMORY-CONTEXT> after"
    for split in range(len(text) + 1):
        scrubber = MemoryContextScrubber()
        assert scrubber.feed(text[:split]) + scrubber.feed(text[split:]) + scrubber.flush() == "before  after"


def test_literal_less_than_cannot_swallow_memory_opener():
    scrubber = MemoryContextScrubber()
    assert scrubber.feed("x < y <memory-context>private</memory-context> end") + scrubber.flush() == "x < y  end"
