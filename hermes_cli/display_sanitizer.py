"""Fail-closed sanitizers for user-visible Hermes text and payloads."""

from __future__ import annotations

from typing import Any


class MemoryContextScrubber:
    """Strip memory-context spans and recalled-memory notes across chunks."""

    def __init__(self) -> None:
        self._inside = False
        self._buffer = ""

    @staticmethod
    def _tag_kind(token: str) -> str | None:
        if not token.startswith("<") or not token.endswith(">"):
            return None
        inner = token[1:-1].strip().lower()
        closing = inner.startswith("/")
        if closing:
            inner = inner[1:].strip()
        if inner != "memory-context":
            return None
        return "close" if closing else "open"

    @staticmethod
    def _is_memory_note(token: str) -> bool:
        lowered = " ".join(token.lower().split())
        return (
            lowered.startswith("[system note:")
            and "recalled memory context" in lowered
            and "not new user input" in lowered
        )

    @staticmethod
    def _looks_sensitive_prefix(token: str) -> bool:
        compact = "".join(token.lower().split())
        open_tag = "<memory-context>"
        close_tag = "</memory-context>"
        return (
            open_tag.startswith(compact)
            or close_tag.startswith(compact)
            or compact.startswith("<memory-context")
            or compact.startswith("</memory-context")
            or compact.startswith("[systemnote:")
        )

    def feed(self, text: str) -> str:
        buffer = self._buffer + str(text or "")
        self._buffer = ""
        visible: list[str] = []

        while buffer:
            if self._inside:
                token_start = buffer.find("<")
                if token_start < 0:
                    return "".join(visible)
                buffer = buffer[token_start:]
            else:
                starts = [index for index in (buffer.find("<"), buffer.find("[")) if index >= 0]
                if not starts:
                    visible.append(buffer)
                    break
                token_start = min(starts)
                visible.append(buffer[:token_start])
                buffer = buffer[token_start:]

            if buffer.startswith("<"):
                token_end = buffer.find(">")
                if token_end < 0:
                    self._buffer = buffer
                    break
                token = buffer[: token_end + 1]
                kind = self._tag_kind(token)
                if self._inside:
                    if kind == "close":
                        self._inside = False
                elif kind == "open":
                    self._inside = True
                elif kind != "close":
                    visible.append(token)
                buffer = buffer[token_end + 1 :]
                continue

            note_end = buffer.find("]")
            if note_end < 0:
                self._buffer = buffer
                break
            token = buffer[: note_end + 1]
            if not self._is_memory_note(token):
                visible.append(token)
            buffer = buffer[note_end + 1 :]

        return "".join(visible)

    def flush(self) -> str:
        tail, self._buffer = self._buffer, ""
        if self._inside or self._looks_sensitive_prefix(tail):
            self._inside = False
            return ""
        return tail


def _strict_sanitize(text: str) -> str:
    scrubber = MemoryContextScrubber()
    return scrubber.feed(text) + scrubber.flush()


try:
    from agent.memory_manager import sanitize_context as _shared_sanitize
except Exception:  # pragma: no cover - broken optional memory integration
    _shared_sanitize = None


def sanitize_display_text(text: Any) -> str:
    """Return fail-closed text safe for TUI/Desktop display surfaces."""
    if text is None:
        return ""
    safe = _strict_sanitize(str(text))
    if _shared_sanitize is None:
        return safe
    try:
        return _shared_sanitize(safe)
    except Exception:
        return safe


def sanitize_display_value(value: Any) -> Any:
    """Recursively sanitize strings in a JSON-compatible display payload."""
    if isinstance(value, str):
        return sanitize_display_text(value)
    if isinstance(value, list):
        return [sanitize_display_value(item) for item in value]
    if isinstance(value, dict):
        return {
            sanitize_display_text(key) if isinstance(key, str) else key: sanitize_display_value(item)
            for key, item in value.items()
        }
    return value
