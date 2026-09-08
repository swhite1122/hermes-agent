"""Regression test for Desktop REST transcript display privacy."""

from fastapi import FastAPI
from starlette.testclient import TestClient

from hermes_cli.web_routers import sessions


def display_client():
    # Test the real router without starting unrelated cron/DB maintenance threads.
    app = FastAPI()
    app.include_router(sessions.list_router)
    app.include_router(sessions.manage_router)
    return TestClient(app)


def test_session_messages_endpoint_sanitizes_nested_display_fields(monkeypatch):
    leaked = "visible < memory-context >private</ memory-context > after"

    class FakeDB:
        def resolve_session_id(self, _session_id):
            return "session-1"

        def resolve_resume_session_id(self, session_id):
            return session_id

        def get_messages(
            self,
            _session_id,
            *,
            limit=None,
            offset=0,
            latest=False,
            include_compacted=False,
        ):
            assert limit == 500
            assert latest is True
            return [
                {
                    "role": "assistant",
                    "content": leaked,
                    "reasoning": leaked,
                    "reasoning_details": {"summary": leaked},
                }
            ]

        def close(self):
            return None

    monkeypatch.setattr(
        sessions,
        "_open_session_db_for_profile",
        lambda _profile, **_kwargs: FakeDB(),
    )

    with display_client() as client:
        response = client.get("/api/sessions/session-1/messages")

    assert response.status_code == 200
    rendered = response.text
    assert "visible" in rendered and " after" in rendered
    assert "memory-context" not in rendered.lower()
    assert "private" not in rendered


def test_session_detail_and_list_sanitize_titles_and_metadata(monkeypatch):
    leaked = "visible <memory-context>private</memory-context> after"

    class FakeDB:
        def get_session(self, session_id):
            assert session_id == "session-1"
            return {"id": session_id, "title": leaked, "metadata": {"nested": leaked}}

        def resolve_session_id(self, _session_id):
            return "session-1"

        def list_sessions_rich(self, **_kwargs):
            return [{"id": "session-1", "title": leaked, "metadata": {"nested": leaked}}]

        def session_count(self, **_kwargs):
            return 1

        def close(self):
            return None

    monkeypatch.setattr(
        sessions,
        "_open_session_db_for_profile",
        lambda _profile, **_kwargs: FakeDB(),
    )
    monkeypatch.setattr(sessions, "_maybe_auto_archive_for_profile", lambda _profile: None)

    with display_client() as client:
        detail = client.get("/api/sessions/session-1")
        listing = client.get("/api/sessions?limit=20")

    assert detail.status_code == listing.status_code == 200
    rendered = detail.text + listing.text
    assert "visible" in rendered and " after" in rendered
    assert "memory-context" not in rendered.lower()
    assert "private" not in rendered
