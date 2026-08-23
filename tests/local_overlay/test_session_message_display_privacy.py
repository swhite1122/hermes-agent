"""Regression test for Desktop REST transcript display privacy."""

from starlette.testclient import TestClient

from hermes_cli import web_server


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
        web_server,
        "_open_session_db_for_profile",
        lambda _profile, **_kwargs: FakeDB(),
    )

    with TestClient(web_server.app) as client:
        client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
        response = client.get("/api/sessions/session-1/messages")

    assert response.status_code == 200
    rendered = response.text
    assert "visible" in rendered and " after" in rendered
    assert "memory-context" not in rendered.lower()
    assert "private" not in rendered
