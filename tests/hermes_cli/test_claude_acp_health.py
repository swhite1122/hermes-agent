"""Safe health/canary contracts for the Claude Agent ACP provider."""

from __future__ import annotations

from types import SimpleNamespace


def test_probe_reports_installed_but_logged_out_without_spawning(monkeypatch):
    from hermes_cli import claude_acp_health as health

    monkeypatch.setattr(
        health,
        "get_external_process_provider_status",
        lambda _provider: {
            "configured": True,
            "logged_in": False,
            "resolved_command": "/stable/claude-agent-acp",
            "error": "Claude login is not active.",
        },
    )
    monkeypatch.setattr(
        health,
        "ClaudeACPClient",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not spawn")),
    )
    monkeypatch.setattr(
        health,
        "_package_versions",
        lambda _command: (health.CLAUDE_ACP_VERSION, health.CLAUDE_SDK_VERSION),
    )

    result = health.probe_claude_acp(run_canary=True)

    assert result["ok"] is False
    assert result["installed"] is True
    assert result["logged_in"] is False
    assert result["models"] == []
    assert result["canaries"] == {}
    assert "login" in result["error"].lower()


def test_probe_discovers_models_and_runs_opus_sonnet_canaries(monkeypatch, tmp_path):
    from hermes_cli import claude_acp_health as health

    calls: list[str] = []

    class FakeCompletions:
        def create(self, *, model, messages, timeout):
            calls.append(model)
            return SimpleNamespace(
                model=model,
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="HERMES_CLAUDE_ACP_OK")
                    )
                ],
            )

    class FakeClient:
        def __init__(self, **kwargs):
            assert kwargs["acp_command"] == "/stable/claude-agent-acp"
            self.chat = SimpleNamespace(completions=FakeCompletions())

        def discover_models(self, *, timeout_seconds):
            assert timeout_seconds == 30.0
            return [
                "claude-opus-5",
                "claude-sonnet-5",
            ]

    monkeypatch.setattr(
        health,
        "get_external_process_provider_status",
        lambda _provider: {
            "configured": True,
            "logged_in": True,
            "resolved_command": "/stable/claude-agent-acp",
        },
    )
    monkeypatch.setattr(health, "ClaudeACPClient", FakeClient)
    monkeypatch.setattr(
        health,
        "_package_versions",
        lambda _command: (health.CLAUDE_ACP_VERSION, health.CLAUDE_SDK_VERSION),
    )

    result = health.probe_claude_acp(run_canary=True, cwd=tmp_path)

    assert result == {
        "ok": True,
        "installed": True,
        "logged_in": True,
        "adapter_command": "/stable/claude-agent-acp",
        "adapter_version": "0.62.0",
        "sdk_version": "0.3.219",
        "models": [
            "claude-opus-5",
            "claude-sonnet-5",
        ],
        "canaries": {
            "claude-opus-5": True,
            "claude-sonnet-5": True,
        },
        "error": "",
    }
    assert calls == ["claude-opus-5", "claude-sonnet-5"]
    assert "token" not in str(result).lower()
    assert "credential" not in str(result).lower()


def test_probe_rejects_wrong_adapter_or_sdk_version_before_spawning(monkeypatch):
    from hermes_cli import claude_acp_health as health

    monkeypatch.setattr(
        health,
        "get_external_process_provider_status",
        lambda _provider: {
            "configured": True,
            "logged_in": True,
            "resolved_command": "/stable/claude-agent-acp",
        },
    )
    monkeypatch.setattr(health, "_package_versions", lambda _command: ("0.62.0", "0.3.999"))
    monkeypatch.setattr(
        health,
        "ClaudeACPClient",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not spawn")),
    )

    result = health.probe_claude_acp(run_canary=True)

    assert result["ok"] is False
    assert result["adapter_version"] == "0.62.0"
    assert result["sdk_version"] == "0.3.999"
    assert "version" in result["error"].lower()
