"""Focused contracts for the Claude Agent ACP provider."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest


_FAKE_ACP = r'''
import json
import os
import sys
import time

mode = sys.argv[1]
log_path = sys.argv[2]
model_options = [
    {"value": "default", "name": "Default", "description": "Opus 5 with 1M context"},
    {"value": "opus[1m]", "name": "Opus", "description": "Opus 5 with 1M context"},
    {"value": "claude-fable-5[1m]", "name": "Fable", "description": "Fable 5"},
    {"value": "sonnet", "name": "Sonnet", "description": "Sonnet 5"},
]
selectors = {
    "claude-opus-5": "opus[1m]",
    "claude-fable-5": "claude-fable-5[1m]",
    "claude-sonnet-5": "sonnet",
}
actual_models = {value: key for key, value in selectors.items()}
actual_models["default"] = "claude-opus-5"
current = "default"

def emit(payload):
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()

def record(request):
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"pid": os.getpid(), "method": request.get("method"), "params": request.get("params")}) + "\n")

def config_options(selected):
    return [{
        "id": "model",
        "name": "Model",
        "category": "model",
        "type": "select",
        "currentValue": selected,
        "options": model_options,
    }]

for line in sys.stdin:
    request = json.loads(line)
    record(request)
    request_id = request.get("id")
    method = request.get("method")
    params = request.get("params") or {}
    if mode == "stderr_exit":
        sys.stderr.write("ANTHROPIC_API_KEY=sk-ant-api03-" + "A" * 48 + "\n")
        sys.stderr.flush()
        raise SystemExit(3)
    if method == "initialize":
        emit({"jsonrpc": "2.0", "id": request_id, "result": {"protocolVersion": 1, "agentCapabilities": {}}})
    elif method == "session/new":
        if mode == "hang_session":
            time.sleep(60)
        emit({"jsonrpc": "2.0", "id": request_id, "result": {"sessionId": "session-1", "configOptions": config_options(current)}})
    elif method == "session/set_config_option":
        if params["value"] not in selectors.values():
            raise ValueError(f"unexpected raw selector: {params['value']}")
        current = params["value"]
        emit({"jsonrpc": "2.0", "id": request_id, "result": {"configOptions": config_options(current)}})
    elif method == "session/prompt":
        if mode == "hang_prompt":
            time.sleep(60)
        if mode == "active_slow_prompt":
            for _ in range(6):
                emit({"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": "session-1", "update": {"sessionUpdate": "agent_thought_chunk", "content": {"type": "text", "text": "."}}}})
                time.sleep(0.05)
        if mode == "fallback_model":
            current = "claude-fable-5[1m]"
            emit({"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": "session-1", "update": {"sessionUpdate": "config_option_update", "configOptions": config_options(current)}}})
        actual = "claude-opus-5" if mode == "silent_fallback" else actual_models[current]
        if mode != "missing_actual_model":
            emit({"jsonrpc": "2.0", "method": "_claude/sdkMessage", "params": {"sessionId": "session-1", "message": {"type": "assistant", "message": {"model": actual}}}})
        emit({"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": "session-1", "update": {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "HERMES_CLAUDE_ACP_OK"}}}})
        usage = {
            "canonicalModel": actual,
            "contextWindow": 1000000,
            "inputTokens": 11,
            "outputTokens": 7,
            "cacheReadInputTokens": 5,
            "cacheCreationInputTokens": 3,
            "costUSD": 0.25,
        }
        if mode != "missing_provider":
            usage["provider"] = "bedrock" if mode == "non_first_party" else "firstParty"
        emit({"jsonrpc": "2.0", "method": "_claude/sdkMessage", "params": {"sessionId": "session-1", "message": {"type": "result", "modelUsage": {current: usage}}}})
        emit({"jsonrpc": "2.0", "id": request_id, "result": {"stopReason": "end_turn"}})
'''


def _write_fake_acp(tmp_path: Path) -> tuple[Path, Path]:
    script = tmp_path / "fake claude agent acp.py"
    log = tmp_path / "requests.jsonl"
    script.write_text(_FAKE_ACP)
    return script, log


def _client(tmp_path: Path, mode: str = "normal"):
    from agent.claude_acp_client import ClaudeACPClient

    script, log = _write_fake_acp(tmp_path)
    client = ClaudeACPClient(
        acp_command=sys.executable,
        acp_args=[str(script), mode, str(log)],
        acp_cwd=str(tmp_path),
    )
    return client, log


def _requests(log: Path) -> list[dict]:
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]


def _wait_for_method(log: Path, method: str, timeout: float = 5.0) -> list[dict]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        requests = _requests(log)
        if any(item["method"] == method for item in requests):
            return requests
        time.sleep(0.02)
    raise AssertionError(f"Timed out waiting for {method}")


def _assert_pid_exited(pid: int, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.02)
    raise AssertionError(f"ACP process {pid} is still alive")


def test_provider_registration_and_runtime_resolution(monkeypatch):
    from providers import get_provider_profile
    from hermes_cli import runtime_provider as rp
    from hermes_cli.auth import PROVIDER_REGISTRY
    from hermes_cli.providers import get_provider

    profile = get_provider_profile("claude-acp")
    assert profile is not None
    assert profile.name == "claude-acp"
    assert profile.auth_type == "external_process"
    assert profile.base_url == "acp://claude"
    assert PROVIDER_REGISTRY["claude-acp"].name == "Claude Agent ACP"
    assert get_provider("claude-acp").auth_type == "external_process"

    monkeypatch.setattr(
        rp,
        "resolve_external_process_provider_credentials",
        lambda provider: {
            "provider": provider,
            "api_key": "claude-acp",
            "base_url": "acp://claude",
            "command": "/usr/local/bin/claude-agent-acp",
            "args": [],
            "source": "process",
        },
    )
    resolved = rp.resolve_runtime_provider(requested="claude-acp")
    assert resolved["provider"] == "claude-acp"
    assert resolved["base_url"] == "acp://claude"
    assert resolved["model"] if "model" in resolved else True
    assert resolved["command"].endswith("claude-agent-acp")


def test_provider_status_uses_stable_adapter_and_real_claude_login(monkeypatch, tmp_path):
    from hermes_cli import auth

    stable = (
        tmp_path
        / "providers"
        / "claude-acp"
        / "node_modules"
        / ".bin"
        / "claude-agent-acp"
    )
    stable.parent.mkdir(parents=True)
    stable.write_text("#!/bin/sh\n")
    stable.chmod(0o755)

    monkeypatch.setattr(auth, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(auth.shutil, "which", lambda _command: None)
    monkeypatch.setattr(
        auth.subprocess,
        "run",
        lambda *args, **kwargs: type(
            "Result",
            (),
            {"returncode": 0, "stdout": '{"loggedIn": false}', "stderr": ""},
        )(),
    )

    status = auth.get_external_process_provider_status("claude-acp")

    assert status["configured"] is True
    assert status["logged_in"] is False
    assert status["resolved_command"] == str(stable)
    assert "not logged in" in status["error"].lower()


def test_claude_auth_probe_failure_does_not_hide_installed_provider(monkeypatch, tmp_path):
    from hermes_cli import auth

    monkeypatch.setattr(auth, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(auth.shutil, "which", lambda _command: "/usr/bin/claude-agent-acp")

    def timeout(*args, **kwargs):
        raise auth.subprocess.TimeoutExpired(cmd="claude", timeout=5)

    monkeypatch.setattr(auth.subprocess, "run", timeout)

    status = auth.get_external_process_provider_status("claude-acp")

    assert status["configured"] is True
    assert status["logged_in"] is False
    assert "timed out" in status["error"].lower()


def test_provider_status_fails_closed_for_non_executable_stable_adapter(monkeypatch, tmp_path):
    from hermes_cli import auth

    stable = (
        tmp_path
        / "providers"
        / "claude-acp"
        / "node_modules"
        / ".bin"
        / "claude-agent-acp"
    )
    stable.parent.mkdir(parents=True)
    stable.write_text("not executable\n")
    stable.chmod(0o644)

    monkeypatch.setattr(auth, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(
        auth.shutil,
        "which",
        lambda _command: "/untrusted/path/claude-agent-acp",
    )
    monkeypatch.setattr(
        auth.subprocess,
        "run",
        lambda *args, **kwargs: type(
            "Result",
            (),
            {"returncode": 0, "stdout": '{"loggedIn": true}', "stderr": ""},
        )(),
    )

    status = auth.get_external_process_provider_status("claude-acp")

    assert status["installed"] is True
    assert status["configured"] is False
    assert status["logged_in"] is False
    assert "not executable" in status["error"].lower()


def test_subprocess_invocation_uses_argument_array_without_shell(monkeypatch, tmp_path):
    from agent.claude_acp_client import ClaudeACPClient

    captured = {}

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        raise FileNotFoundError("missing")

    client = ClaudeACPClient(
        acp_command="claude-agent-acp;touch",
        acp_args=["/tmp/SHOULD_NOT_EXIST"],
        acp_cwd=str(tmp_path),
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with patch("agent.copilot_acp_client.subprocess.Popen", side_effect=fake_popen):
        with pytest.raises(RuntimeError, match="Could not start Claude Agent ACP"):
            client.discover_models(timeout_seconds=0.1)

    assert captured["argv"] == ["claude-agent-acp;touch", "/tmp/SHOULD_NOT_EXIST"]
    assert captured["kwargs"].get("shell", False) is False


def test_model_config_discovery_exposes_only_approved_ids_and_keeps_raw_selectors(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client, _ = _client(tmp_path)
    assert client.discover_models(timeout_seconds=2) == [
        "claude-opus-5",
        "claude-sonnet-5",
    ]
    assert "claude-fable-5[1m]" in client.last_raw_advertised_models
    assert "claude-fable-5" not in client.last_model_metadata


def test_fable_is_rejected_before_prompt(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client, log = _client(tmp_path)
    with pytest.raises(ValueError, match="claude-fable-5.*not advertised"):
        client.chat.completions.create(
            model="claude-fable-5",
            messages=[{"role": "user", "content": "hello"}],
            timeout=2,
        )
    assert "session/prompt" not in [item["method"] for item in _requests(log)]


def test_session_disables_claude_transcript_persistence(tmp_path):
    client, _ = _client(tmp_path)
    options = client._session_new_params()["_meta"]["claudeCode"]["options"]
    assert options["persistSession"] is False


def test_session_ignores_user_project_local_settings_and_uses_backend_prompt(tmp_path):
    from agent.claude_acp_client import CLAUDE_ACP_SYSTEM_PROMPT

    client, _ = _client(tmp_path)
    params = client._session_new_params()
    options = params["_meta"]["claudeCode"]["options"]
    assert options["settingSources"] == []
    assert options["tools"] == []
    assert params["_meta"]["systemPrompt"] == CLAUDE_ACP_SYSTEM_PROMPT


def test_opus_selector_and_context_metadata_are_preserved(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client, _ = _client(tmp_path)
    client.discover_models(timeout_seconds=2)
    assert client.last_model_metadata["claude-opus-5"] == {
        "selector": "opus[1m]",
        "context_window": 1_000_000,
    }

    from agent.model_metadata import get_model_context_length

    assert get_model_context_length(
        "claude-opus-5",
        base_url="acp://claude",
        provider="claude-acp",
    ) == 1_000_000


def test_exact_selected_model_is_sent_before_prompt(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client, log = _client(tmp_path)
    response = client.chat.completions.create(
        model="claude-sonnet-5",
        messages=[{"role": "user", "content": "Reply exactly HERMES_CLAUDE_ACP_OK"}],
        timeout=2,
    )

    requests = _requests(log)
    methods = [item["method"] for item in requests]
    assert methods.index("session/set_config_option") < methods.index("session/prompt")
    selected = next(item for item in requests if item["method"] == "session/set_config_option")
    assert selected["params"] == {
        "sessionId": "session-1",
        "configId": "model",
        "value": "sonnet",
    }
    assert response.choices[0].message.content == "HERMES_CLAUDE_ACP_OK"
    assert response.model == "claude-sonnet-5"
    assert client.last_confirmed_model == "claude-sonnet-5"
    assert client.last_picker_confirmed_model == "claude-sonnet-5"
    assert client.last_actual_model == "claude-sonnet-5"
    assert client.last_serving_provider == "firstParty"


def test_model_usage_is_mapped_to_openai_usage(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client, _ = _client(tmp_path)
    try:
        response = client.chat.completions.create(
            model="claude-opus-5",
            messages=[{"role": "user", "content": "hello"}],
            timeout=2,
        )
    finally:
        client.close()

    assert response.usage.prompt_tokens == 19
    assert response.usage.completion_tokens == 7
    assert response.usage.total_tokens == 26
    assert response.usage.prompt_tokens_details.cached_tokens == 5
    assert response.usage.prompt_tokens_details.cache_write_tokens == 3
    assert response.usage.cache_creation_input_tokens == 3
    assert response.usage.cost_usd == pytest.approx(0.25)

    from agent.usage_pricing import normalize_usage

    canonical = normalize_usage(
        response.usage,
        provider="claude-acp",
        api_mode="chat_completions",
    )
    assert canonical.input_tokens == 11
    assert canonical.cache_read_tokens == 5
    assert canonical.cache_write_tokens == 3
    assert canonical.output_tokens == 7


def test_reuses_in_memory_session_and_sends_only_appended_context(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client, log = _client(tmp_path)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object"},
            },
        }
    ]
    first_messages = [{"role": "user", "content": "inspect UNIQUE_ORIGINAL"}]
    second_messages = first_messages + [
        {"role": "assistant", "content": "HERMES_CLAUDE_ACP_OK"},
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "read_file",
            "content": "UNIQUE_TOOL_RESULT",
        },
    ]

    try:
        client.chat.completions.create(
            model="claude-opus-5",
            messages=first_messages,
            tools=tools,
            timeout=2,
        )
        client.chat.completions.create(
            model="claude-opus-5",
            messages=second_messages,
            tools=tools,
            timeout=2,
        )
        requests = _requests(log)
    finally:
        client.close()

    assert len({item["pid"] for item in requests}) == 1
    assert [item["method"] for item in requests].count("initialize") == 1
    assert [item["method"] for item in requests].count("session/new") == 1
    prompts = [
        item["params"]["prompt"][0]["text"]
        for item in requests
        if item["method"] == "session/prompt"
    ]
    assert len(prompts) == 2
    assert "UNIQUE_ORIGINAL" in prompts[0]
    assert "Available tools" in prompts[0]
    assert "UNIQUE_TOOL_RESULT" in prompts[1]
    assert "UNIQUE_ORIGINAL" not in prompts[1]
    assert "Available tools" not in prompts[1]


def test_rewritten_history_restarts_in_memory_session(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client, log = _client(tmp_path)
    try:
        client.chat.completions.create(
            model="claude-opus-5",
            messages=[{"role": "user", "content": "original history"}],
            timeout=2,
        )
        client.chat.completions.create(
            model="claude-opus-5",
            messages=[
                {"role": "system", "content": "compressed summary"},
                {"role": "user", "content": "continue after compression"},
            ],
            timeout=2,
        )
        requests = _requests(log)
    finally:
        client.close()

    assert len({item["pid"] for item in requests}) == 2
    assert [item["method"] for item in requests].count("initialize") == 2
    assert [item["method"] for item in requests].count("session/new") == 2
    prompts = [
        item["params"]["prompt"][0]["text"]
        for item in requests
        if item["method"] == "session/prompt"
    ]
    assert "compressed summary" in prompts[1]
    assert "original history" not in prompts[1]


def test_persistent_acp_requests_are_serialized(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client, _ = _client(tmp_path)
    state_lock = threading.Lock()
    active = 0
    max_active = 0

    def fake_completion(**kwargs):
        nonlocal active, max_active
        del kwargs
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with state_lock:
            active -= 1
        return "ok"

    monkeypatch.setattr(client, "_create_chat_completion", fake_completion)
    threads = [
        threading.Thread(target=client.chat.completions.create, kwargs={})
        for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert max_active == 1


def test_per_turn_call_budget_falls_back_but_resets_for_next_user_turn(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("HERMES_CLAUDE_ACP_MAX_CALLS_PER_TURN", "2")
    client, log = _client(tmp_path)
    messages = [{"role": "user", "content": "first user turn"}]

    try:
        client.chat.completions.create(
            model="claude-opus-5", messages=messages, timeout=2
        )
        messages = messages + [
            {"role": "assistant", "content": "HERMES_CLAUDE_ACP_OK"},
            {"role": "tool", "content": "tool result one"},
        ]
        client.chat.completions.create(
            model="claude-opus-5", messages=messages, timeout=2
        )
        messages = messages + [
            {"role": "assistant", "content": "HERMES_CLAUDE_ACP_OK"},
            {"role": "tool", "content": "tool result two"},
        ]
        with pytest.raises(RuntimeError, match="per-turn call budget exhausted"):
            client.chat.completions.create(
                model="claude-opus-5", messages=messages, timeout=2
            )

        next_turn = messages + [
            {"role": "user", "content": "second user turn"}
        ]
        client.chat.completions.create(
            model="claude-opus-5", messages=next_turn, timeout=2
        )
        prompt_count = [item["method"] for item in _requests(log)].count(
            "session/prompt"
        )
    finally:
        client.close()

    assert prompt_count == 3


def test_claude_acp_session_limit_is_non_retryable_and_fallback_worthy():
    from agent.error_classifier import FailoverReason, classify_api_error

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 7, 25, 15, 55, tzinfo=tz)

    with patch("agent.error_classifier.datetime", FixedDatetime):
        result = classify_api_error(
            RuntimeError(
                "You've hit your session limit · resets 6pm (America/New_York)"
            ),
            provider="claude-acp",
            model="claude-opus-5",
        )

    assert result.reason == FailoverReason.session_limit
    assert result.retryable is False
    assert result.should_fallback is True
    assert result.error_context["retry_after_seconds"] == pytest.approx(7500.0)


def test_claude_acp_local_turn_budget_is_non_retryable_without_rate_cooldown():
    from agent.error_classifier import FailoverReason, classify_api_error

    result = classify_api_error(
        RuntimeError(
            "Claude Agent ACP per-turn call budget exhausted (24 calls)."
        ),
        provider="claude-acp",
        model="claude-opus-5",
    )

    assert result.reason == FailoverReason.turn_budget
    assert result.retryable is False
    assert result.should_fallback is True
    assert "retry_after_seconds" not in result.error_context


def test_unavailable_model_fails_without_prompt_or_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client, log = _client(tmp_path)
    with pytest.raises(ValueError, match="claude-does-not-exist.*not advertised"):
        client.chat.completions.create(
            model="claude-does-not-exist",
            messages=[{"role": "user", "content": "hello"}],
            timeout=2,
        )
    assert "session/prompt" not in [item["method"] for item in _requests(log)]


def test_api_key_billing_guard_fails_before_spawn(monkeypatch, tmp_path):
    from agent.claude_acp_client import ClaudeACPClient

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-test-billing-key")
    client = ClaudeACPClient(acp_cwd=str(tmp_path))
    with patch("agent.copilot_acp_client.subprocess.Popen") as popen:
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY.*API-key billing.*unset"):
            client.discover_models(timeout_seconds=0.1)
    popen.assert_not_called()


def test_child_environment_omits_api_keys_and_oauth_tokens(monkeypatch, tmp_path):
    from agent.claude_acp_client import ClaudeACPClient

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "should-not-pass")
    monkeypatch.setenv("ANTHROPIC_TOKEN", "should-not-pass")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "should-not-pass")
    env = ClaudeACPClient(acp_cwd=str(tmp_path))._build_subprocess_env()
    assert "ANTHROPIC_API_KEY" not in env
    assert "ANTHROPIC_AUTH_TOKEN" not in env
    assert "ANTHROPIC_TOKEN" not in env
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in env


def test_timeout_terminates_and_reaps_process(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client, log = _client(tmp_path, "hang_session")
    with pytest.raises(TimeoutError):
        client.discover_models(timeout_seconds=0.15)
    requests = _requests(log)
    pid = requests[0]["pid"]
    _assert_pid_exited(pid)
    assert client._active_process is None
    assert client.is_closed is True


def test_request_timeout_policy_caps_control_calls_and_detects_idle_prompts():
    from agent.copilot_acp_client import _request_timeout_policy

    assert _request_timeout_policy("initialize", 1800) == (30.0, 30.0)
    assert _request_timeout_policy("session/new", 1800) == (30.0, 30.0)
    assert _request_timeout_policy("session/prompt", 1800) == (1800.0, 300.0)
    assert _request_timeout_policy("session/prompt", 120) == (120.0, 120.0)


def test_prompt_activity_resets_idle_timeout_without_extending_absolute_cap(tmp_path, monkeypatch):
    from agent import copilot_acp_client

    original_policy = copilot_acp_client._request_timeout_policy

    def short_idle(method, timeout_seconds):
        if method == "session/prompt":
            return 1.0, 0.1
        return original_policy(method, timeout_seconds)

    monkeypatch.setattr(copilot_acp_client, "_request_timeout_policy", short_idle)
    client, _log = _client(tmp_path, "active_slow_prompt")

    response = client.chat.completions.create(
        model="claude-opus-5",
        messages=[{"role": "user", "content": "hello"}],
        timeout=1.0,
    )

    assert response.choices[0].message.content == "HERMES_CLAUDE_ACP_OK"


@pytest.mark.parametrize(
    "message,expected_reason",
    [
        (
            "Claude Agent ACP actual-model telemetry was missing from the raw SDK events.",
            "model_not_found",
        ),
        (
            "Claude Agent ACP requested 'claude-sonnet-5' but the SDK answered with 'claude-opus-5'.",
            "model_not_found",
        ),
        (
            "Claude Agent ACP provider telemetry was missing for the actual answering model.",
            "model_not_found",
        ),
        (
            "Claude Agent ACP reported serving provider 'bedrock'. Only firstParty is allowed.",
            "model_not_found",
        ),
    ],
)
def test_integrity_failures_are_non_retryable_and_fallback_worthy(message, expected_reason):
    from agent.error_classifier import classify_api_error

    result = classify_api_error(
        RuntimeError(message),
        provider="claude-acp",
        model="claude-sonnet-5",
    )

    assert result.reason.value == expected_reason
    assert result.retryable is False
    assert result.should_fallback is True


def test_claude_acp_timeout_falls_back_without_a_second_long_wait():
    from agent.error_classifier import FailoverReason, classify_api_error

    result = classify_api_error(
        TimeoutError("Timed out waiting for Claude Agent ACP response to session/prompt."),
        provider="claude-acp",
        model="claude-opus-5",
    )

    assert result.reason == FailoverReason.timeout
    assert result.retryable is False
    assert result.should_fallback is True


def test_claude_acp_process_exit_gets_one_fast_retry():
    from agent.error_classifier import FailoverReason, classify_api_error

    result = classify_api_error(
        RuntimeError("Claude Agent ACP process exited before responding to session/new."),
        provider="claude-acp",
        model="claude-opus-5",
    )

    assert result.reason == FailoverReason.timeout
    assert result.retryable is True
    assert result.should_fallback is False


@pytest.mark.parametrize(
    "message",
    [
        "Claude Agent ACP session/new failed: Not logged in. Run /login.",
        "Claude Agent ACP process exited early: Claude authentication has expired.",
        "Claude Agent ACP process exited early: Login required before starting a session.",
        "Claude Agent ACP refused to start because ANTHROPIC_API_KEY is set.",
    ],
)
def test_claude_acp_login_failures_fall_back_without_retry(message):
    from agent.error_classifier import FailoverReason, classify_api_error

    result = classify_api_error(
        RuntimeError(message),
        provider="claude-acp",
        model="claude-opus-5",
    )

    assert result.reason == FailoverReason.auth_permanent
    assert result.retryable is False
    assert result.should_fallback is True


def test_cancellation_terminates_and_reaps_process(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client, log = _client(tmp_path, "hang_prompt")
    outcome: list[BaseException] = []

    def run() -> None:
        try:
            client.chat.completions.create(
                model="claude-sonnet-5",
                messages=[{"role": "user", "content": "wait"}],
                timeout=30,
            )
        except BaseException as exc:
            outcome.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    requests = _wait_for_method(log, "session/prompt")
    pid = requests[0]["pid"]
    client.close()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert outcome and "cancel" in str(outcome[0]).lower()
    _assert_pid_exited(pid)
    assert client._active_process is None


def test_stderr_tokens_are_redacted(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client, _ = _client(tmp_path, "stderr_exit")
    with pytest.raises(RuntimeError) as exc_info:
        client.discover_models(timeout_seconds=2)
    message = str(exc_info.value)
    assert "sk-ant-api03-" not in message
    assert "AAAA" not in message
    assert "redact" in message.lower() or "***" in message


def test_config_update_records_and_rejects_actual_model_change(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client, _ = _client(tmp_path, "fallback_model")
    with pytest.raises(
        RuntimeError,
        match="requested 'claude-sonnet-5'.*answered with 'claude-fable-5'.*rejected",
    ):
        client.chat.completions.create(
            model="claude-sonnet-5",
            messages=[{"role": "user", "content": "hello"}],
            timeout=2,
        )
    assert client.last_confirmed_model == "claude-fable-5"


def test_actual_model_mismatch_is_rejected_without_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client, _ = _client(tmp_path, "silent_fallback")
    with pytest.raises(
        RuntimeError,
        match="requested 'claude-sonnet-5'.*answered with 'claude-opus-5'.*rejected",
    ):
        client.chat.completions.create(
            model="claude-sonnet-5",
            messages=[{"role": "user", "content": "hello"}],
            timeout=2,
        )
    assert client.last_confirmed_model == "claude-opus-5"


def test_missing_actual_model_telemetry_is_rejected(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client, _ = _client(tmp_path, "missing_actual_model")
    with pytest.raises(RuntimeError, match="actual-model telemetry.*missing.*rejected"):
        client.chat.completions.create(
            model="claude-sonnet-5",
            messages=[{"role": "user", "content": "hello"}],
            timeout=2,
        )
    assert client.last_picker_confirmed_model == "claude-sonnet-5"
    assert client.last_actual_model == ""


def test_non_first_party_provider_is_rejected(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client, _ = _client(tmp_path, "non_first_party")
    with pytest.raises(RuntimeError, match="provider 'bedrock'.*firstParty.*rejected"):
        client.chat.completions.create(
            model="claude-sonnet-5",
            messages=[{"role": "user", "content": "hello"}],
            timeout=2,
        )
    assert client.last_serving_provider == "bedrock"


def test_missing_provider_telemetry_is_rejected(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client, _ = _client(tmp_path, "missing_provider")
    with pytest.raises(RuntimeError, match="provider telemetry.*missing.*rejected"):
        client.chat.completions.create(
            model="claude-sonnet-5",
            messages=[{"role": "user", "content": "hello"}],
            timeout=2,
        )
    assert client.last_serving_provider == ""


def test_model_normalization_preserves_claude_acp_ids():
    from hermes_cli.model_normalize import normalize_model_for_provider

    assert normalize_model_for_provider("claude-sonnet-5", "claude-acp") == "claude-sonnet-5"
    assert normalize_model_for_provider("claude-acp/claude-opus-5", "claude-acp") == "claude-opus-5"
