"""Markdown handling tests for PhotonAdapter.

Markdown is on by default (the sidecar sends it via spectrum-ts'
``markdown()`` builder and iMessage renders it); ``PHOTON_MARKDOWN=false``
reverts to the stripped-plain-text path.
"""
from __future__ import annotations

from pathlib import Path
import json
import subprocess
from typing import Any, Dict, List, Tuple

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.photon import adapter as photon_adapter
from plugins.platforms.photon.adapter import PhotonAdapter

_MD = "**bold** and `code`"


def _make_adapter(monkeypatch: pytest.MonkeyPatch) -> PhotonAdapter:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    cfg = PlatformConfig(enabled=True, token="", extra={})
    return PhotonAdapter(cfg)


def _capture_sidecar(adapter: PhotonAdapter) -> List[Tuple[str, Dict[str, Any]]]:
    calls: List[Tuple[str, Dict[str, Any]]] = []

    async def _fake_call(path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        calls.append((path, body))
        return {"ok": True, "messageId": "msg-123"}

    adapter._sidecar_call = _fake_call  # type: ignore[assignment]
    return calls


def test_format_message_passthrough_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PHOTON_MARKDOWN", raising=False)
    adapter = _make_adapter(monkeypatch)
    assert adapter.format_message(_MD) == _MD


def test_format_message_strips_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHOTON_MARKDOWN", "false")
    adapter = _make_adapter(monkeypatch)
    assert adapter.format_message(_MD) == "bold and code"


def test_supports_code_blocks_mirrors_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PHOTON_MARKDOWN", raising=False)
    assert _make_adapter(monkeypatch).supports_code_blocks is True
    monkeypatch.setenv("PHOTON_MARKDOWN", "false")
    assert _make_adapter(monkeypatch).supports_code_blocks is False


@pytest.mark.asyncio
async def test_sidecar_send_includes_markdown_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PHOTON_MARKDOWN", raising=False)
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    await adapter.send("+15551234567", _MD)

    path, body = calls[0]
    assert path == "/send"
    assert body["format"] == "markdown"
    assert body["text"] == _MD  # passed through unstripped


@pytest.mark.asyncio
async def test_sidecar_send_omits_format_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Old-sidecar compat: the key is absent, not "text", when disabled."""
    monkeypatch.setenv("PHOTON_MARKDOWN", "false")
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    await adapter.send("+15551234567", _MD)

    _, body = calls[0]
    assert "format" not in body
    assert body["text"] == "bold and code"


@pytest.mark.asyncio
async def test_standalone_send_includes_markdown_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PHOTON_MARKDOWN", raising=False)
    monkeypatch.setenv("PHOTON_SIDECAR_TOKEN", "tok")

    posted: List[Tuple[str, Dict[str, Any]]] = []

    class _Resp:
        status_code = 200

        @staticmethod
        def json() -> Dict[str, Any]:
            return {"ok": True, "messageId": "m-9"}

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url: str, json: Dict[str, Any], headers=None):
            posted.append((url, json))
            return _Resp()

    monkeypatch.setattr(photon_adapter.httpx, "AsyncClient", _FakeClient)

    cfg = PlatformConfig(enabled=True, token="", extra={})
    result = await photon_adapter._standalone_send(cfg, "+15551234567", _MD)

    assert result.get("success") is True
    assert posted[0][1]["format"] == "markdown"


def test_sidecar_helper_retries_linked_markdown_as_plain_text() -> None:
    helper = (
        Path(photon_adapter.__file__).resolve().parent
        / "sidecar"
        / "send-text-fallback.mjs"
    )
    script = f"""
        import {{ sendTextWithMarkdownFallback }} from {json.dumps(helper.as_uri())};
        const calls = [];
        const space = {{
          send: async (builder) => {{
            calls.push(builder);
            if (calls.length === 1) {{
              throw new Error('[upstream] enable_data_detection is not supported by the IMAgentKit send path');
            }}
            return {{ id: 'msg-ok' }};
          }},
        }};
        const result = await sendTextWithMarkdownFallback(
          space,
          'See https://example.com',
          'markdown',
          {{
            spectrumText: (value) => ({{ type: 'text', value }}),
            spectrumMarkdown: (value) => ({{ type: 'markdown', value }}),
            log: () => undefined,
          }},
        );
        console.log(JSON.stringify({{ result, calls }}));
    """
    completed = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["result"] == {"id": "msg-ok"}
    assert [call["type"] for call in payload["calls"]] == ["markdown", "text"]


def test_sidecar_helper_rethrows_unrelated_markdown_errors() -> None:
    helper = (
        Path(photon_adapter.__file__).resolve().parent
        / "sidecar"
        / "send-text-fallback.mjs"
    )
    script = f"""
        import {{ sendTextWithMarkdownFallback }} from {json.dumps(helper.as_uri())};
        const space = {{
          send: async () => {{ throw new Error('unrelated failure'); }},
        }};
        try {{
          await sendTextWithMarkdownFallback(
            space,
            'hello',
            'markdown',
            {{
              spectrumText: (value) => ({{ type: 'text', value }}),
              spectrumMarkdown: (value) => ({{ type: 'markdown', value }}),
              log: () => undefined,
            }},
          );
        }} catch (error) {{
          console.log(error.message);
        }}
    """
    completed = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        text=True,
        capture_output=True,
        check=True,
    )
    assert completed.stdout.strip() == "unrelated failure"


def test_sidecar_token_persists_across_adapter_instances(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("PHOTON_SIDECAR_TOKEN", raising=False)
    adapter_one = _make_adapter(monkeypatch)
    adapter_two = _make_adapter(monkeypatch)

    assert adapter_one._sidecar_token == adapter_two._sidecar_token
    token_path = tmp_path / "photon-sidecar-token"
    assert token_path.read_text(encoding="utf-8").strip() == adapter_one._sidecar_token
    if hasattr(token_path.stat(), "st_mode"):
        assert token_path.stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_sidecar_call_tries_loopback_when_adapter_client_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("PHOTON_SIDECAR_TOKEN", raising=False)
    adapter = _make_adapter(monkeypatch)
    adapter._http_client = None
    posted: List[Tuple[str, Dict[str, Any]]] = []

    class _Resp:
        status_code = 200
        text = '{"ok": true}'

        @staticmethod
        def json() -> Dict[str, Any]:
            return {"ok": True, "messageId": "msg-after-reconnect"}

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url: str, json: Dict[str, Any], headers=None):
            posted.append((url, json))
            assert headers is not None
            assert headers["X-Hermes-Sidecar-Token"] == adapter._sidecar_token
            return _Resp()

    monkeypatch.setattr(photon_adapter.httpx, "AsyncClient", _FakeClient)

    result = await adapter._sidecar_call("/send", {"spaceId": "any;-;+155****4567", "text": "ok"})

    assert result["messageId"] == "msg-after-reconnect"
    assert posted[0][0].endswith(":8789/send")
