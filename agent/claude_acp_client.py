"""First-class Claude Agent ACP provider backed by the Claude subscription store.

The OpenAI-compatible surface is inherited from :mod:`agent.copilot_acp_client`,
including JSON-RPC transport, permission/file safety, timeouts, cancellation,
redaction, and child-process cleanup.  This subclass adds Claude-specific model
discovery/selection and a fail-closed API-billing guard.
"""

from __future__ import annotations

import os
import re
import copy
import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agent.copilot_acp_client import (
    CopilotACPClient,
    _format_messages_as_prompt,
    _resolve_home_dir,
)
from tools.environments.local import hermes_subprocess_env

logger = logging.getLogger(__name__)

CLAUDE_ACP_MARKER_BASE_URL = "acp://claude"
DEFAULT_CLAUDE_ACP_COMMAND = "claude-agent-acp"
CLAUDE_ACP_SYSTEM_PROMPT = (
    "You are the language-model backend for Hermes Agent. Follow the conversation "
    "and instructions supplied by Hermes and return only the requested assistant "
    "response. Do not act as an autonomous coding agent, inspect files, run commands, "
    "or invoke tools; Hermes owns all tool use and execution."
)

# These variables would bypass the existing Claude subscription credential store
# or change where the SDK sends traffic.  Claude ACP intentionally uses only the
# credential files reachable through HOME/CLAUDE_CONFIG_DIR.
_CLAUDE_ROUTING_ENV_VARS = {
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_CUSTOM_HEADERS",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
}

_CONTEXT_HINT_RE = re.compile(r"(?:\[\d+m\]|-\d+m)$", re.IGNORECASE)
_CONTEXT_WINDOW_RE = re.compile(
    r"(?:\[|\b)(\d+(?:\.\d+)?)\s*([mk])(?:\]|\b)", re.IGNORECASE
)
_FAMILY_VERSION_RE = re.compile(
    r"\b(opus|sonnet|fable|haiku)\s+(\d+(?:[.-]\d+)?)\b",
    re.IGNORECASE,
)
_APPROVED_MODELS = frozenset({"claude-opus-5", "claude-sonnet-5"})


def _canonical_selector(option: dict[str, Any]) -> str:
    """Derive the SDK-stable selector from a live ACP picker row."""
    value = str(option.get("value") or "").strip()
    without_hint = _CONTEXT_HINT_RE.sub("", value)
    if without_hint.startswith("claude-"):
        return without_hint
    label_text = " ".join(
        str(option.get(key) or "") for key in ("name", "description")
    )
    match = _FAMILY_VERSION_RE.search(label_text)
    if match:
        family = match.group(1).lower()
        version = match.group(2).replace(".", "-")
        return f"claude-{family}-{version}"
    return value


def _canonical_options(model_option: dict[str, Any] | None) -> list[str]:
    values: list[str] = []
    for option in _model_rows(model_option):
        canonical = _canonical_selector(option)
        if canonical in _APPROVED_MODELS and canonical not in values:
            values.append(canonical)
    return values


def _context_window_from_option(option: dict[str, Any]) -> int | None:
    text = " ".join(
        str(option.get(key) or "")
        for key in ("value", "name", "description", "resolvedModel")
    )
    match = _CONTEXT_WINDOW_RE.search(text)
    if not match:
        return None
    amount = float(match.group(1))
    multiplier = 1_000_000 if match.group(2).lower() == "m" else 1_000
    value = int(amount * multiplier)
    return value if value > 0 else None


def _model_rows(model_option: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(model_option, dict):
        return []
    rows: list[dict[str, Any]] = []
    for item in model_option.get("options") or []:
        if not isinstance(item, dict):
            continue
        children = item.get("options")
        if isinstance(children, list):
            rows.extend(child for child in children if isinstance(child, dict))
        else:
            rows.append(item)
    return rows


class ClaudeACPClient(CopilotACPClient):
    """OpenAI-compatible facade over ``claude-agent-acp`` stdio JSON-RPC."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        acp_command: str | None = None,
        acp_args: list[str] | None = None,
        acp_cwd: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        resolved_command = acp_command or command or DEFAULT_CLAUDE_ACP_COMMAND
        resolved_args = list(acp_args if acp_args is not None else (args or []))
        super().__init__(
            api_key=api_key or "claude-acp",
            base_url=base_url or CLAUDE_ACP_MARKER_BASE_URL,
            default_headers=default_headers,
            acp_command=resolved_command,
            acp_args=resolved_args,
            acp_cwd=acp_cwd,
            **kwargs,
        )
        self.last_raw_advertised_models: list[str] = []
        self.last_model_metadata: dict[str, dict[str, Any]] = {}
        self.last_serving_provider: str = ""
        self._reuse_acp_session = True
        self._last_hermes_messages: list[dict[str, Any]] = []
        self._last_tools_fingerprint = ""
        self._last_request_model = ""
        self._turn_anchor = ""
        self._turn_call_count = 0

    def close(self) -> None:
        super().close()
        self._last_hermes_messages = []
        self._last_tools_fingerprint = ""
        self._last_request_model = ""

    @staticmethod
    def _latest_user_anchor(messages: list[dict[str, Any]]) -> str:
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if isinstance(message, dict) and message.get("role") == "user":
                payload = json.dumps(message, sort_keys=True, default=str)
                digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
                return f"{index}:{digest}"
        return "no-user"

    def _enforce_turn_call_budget(self, messages: list[dict[str, Any]]) -> None:
        try:
            limit = int(os.getenv("HERMES_CLAUDE_ACP_MAX_CALLS_PER_TURN", "24"))
        except (TypeError, ValueError):
            limit = 24
        limit = max(1, min(250, limit))
        anchor = self._latest_user_anchor(messages)
        if anchor != self._turn_anchor:
            self._turn_anchor = anchor
            self._turn_call_count = 0
        if self._turn_call_count >= limit:
            raise RuntimeError(
                "Claude Agent ACP per-turn call budget exhausted "
                f"({limit} calls). Switching providers protects the Claude "
                "subscription window; the budget resets on the next user message."
            )
        self._turn_call_count += 1
        logger.info(
            "Claude ACP turn budget call=%d limit=%d anchor=%s",
            self._turn_call_count,
            limit,
            anchor[:16],
        )

    def _create_chat_completion(self, **kwargs: Any) -> Any:
        messages = kwargs.get("messages") or []
        if not isinstance(messages, list):
            messages = []
        self._enforce_turn_call_budget(messages)
        snapshot = copy.deepcopy(messages)
        tools_fingerprint = json.dumps(
            kwargs.get("tools") or [], sort_keys=True, default=str
        )
        request_model = str(kwargs.get("model") or "")
        try:
            result = super()._create_chat_completion(**kwargs)
        except Exception:
            raise
        if self._persistent_acp_request is not None:
            self._last_hermes_messages = snapshot
            self._last_tools_fingerprint = tools_fingerprint
            self._last_request_model = request_model
        return result

    def _format_request_prompt(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None,
        tools: list[dict[str, Any]] | None,
        tool_choice: Any,
    ) -> str:
        tools_fingerprint = json.dumps(tools or [], sort_keys=True, default=str)
        previous = self._last_hermes_messages
        can_append = (
            self._persistent_acp_request is not None
            and str(model or "") == self._last_request_model
            and tools_fingerprint == self._last_tools_fingerprint
            and len(messages) >= len(previous)
            and messages[: len(previous)] == previous
        )
        if can_append:
            appended = [
                message
                for message in messages[len(previous) :]
                if isinstance(message, dict) and message.get("role") != "assistant"
            ]
            if appended:
                logger.info(
                    "Claude ACP prompt mode=delta appended_messages=%d "
                    "skipped_assistant_messages=%d",
                    len(appended),
                    len(messages[len(previous) :]) - len(appended),
                )
                return _format_messages_as_prompt(appended)

        if self._persistent_acp_request is not None:
            self.close()
        logger.info(
            "Claude ACP prompt mode=full messages=%d tools=%d",
            len(messages),
            len(tools or []),
        )
        return super()._format_request_prompt(
            messages,
            model=model,
            tools=tools,
            tool_choice=tool_choice,
        )

    @property
    def _provider_label(self) -> str:
        return "Claude Agent ACP"

    def _command_start_error(self) -> str:
        return (
            f"Could not start Claude Agent ACP command '{self._acp_command}'. "
            "Install @agentclientprotocol/claude-agent-acp@0.62.0 in the stable "
            "Hermes provider directory or ensure `claude-agent-acp` is on PATH."
        )

    def _build_subprocess_env(self) -> dict[str, str]:
        if os.environ.get("ANTHROPIC_API_KEY", "").strip():
            raise RuntimeError(
                "Claude Agent ACP refused to start because ANTHROPIC_API_KEY is set. "
                "That variable can select Anthropic API-key billing instead of your "
                "Claude subscription. Please unset ANTHROPIC_API_KEY for this Hermes process "
                "and retry; Claude ACP will use the existing Claude credential store."
            )

        # Start from Hermes' credential-safe environment and then remove every
        # known Claude API/routing override explicitly. HOME remains available so
        # the official SDK can read Claude Code's credential store itself; no OAuth
        # token is copied into Hermes config or the child environment.
        env = hermes_subprocess_env(inherit_credentials=False)
        for name in _CLAUDE_ROUTING_ENV_VARS:
            env.pop(name, None)
        env["HOME"] = _resolve_home_dir()
        from hermes_constants import apply_subprocess_home_env

        apply_subprocess_home_env(env)
        return env

    def _session_new_params(self) -> dict[str, Any]:
        return {
            "cwd": self._acp_cwd,
            "mcpServers": [],
            # Hermes owns tool execution. Disabling Claude Code's native tool set
            # prevents an ACP turn from bypassing Hermes permission and audit paths.
            "_meta": {
                "systemPrompt": CLAUDE_ACP_SYSTEM_PROMPT,
                "claudeCode": {
                    "emitRawSDKMessages": True,
                    "options": {
                        "tools": [],
                        "persistSession": False,
                        "settingSources": [],
                    }
                }
            },
        }

    def _model_selection_values(
        self, model_option: dict[str, Any] | None
    ) -> list[str]:
        rows = _model_rows(model_option)
        self.last_raw_advertised_models = [
            str(option.get("value") or "").strip()
            for option in rows
            if str(option.get("value") or "").strip()
        ]
        self.last_model_metadata = {}
        for option in rows:
            selector = str(option.get("value") or "").strip()
            canonical = _canonical_selector(option)
            if not selector or canonical not in _APPROVED_MODELS:
                continue
            metadata: dict[str, Any] = {"selector": selector}
            context_window = _context_window_from_option(option)
            if context_window:
                metadata["context_window"] = context_window
            existing = self.last_model_metadata.get(canonical)
            if existing is None or existing.get("selector") == "default":
                self.last_model_metadata[canonical] = metadata

        from agent.model_metadata import save_context_length

        for canonical, metadata in self.last_model_metadata.items():
            context_window = metadata.get("context_window")
            if isinstance(context_window, int) and context_window > 0:
                save_context_length(
                    canonical,
                    CLAUDE_ACP_MARKER_BASE_URL,
                    context_window,
                )
        return _canonical_options(model_option)

    def _confirmed_model_from_options(self, config_options: Any) -> str:
        if not isinstance(config_options, list):
            return ""
        for config in config_options:
            if not isinstance(config, dict):
                continue
            if config.get("id") != "model" and config.get("category") != "model":
                continue
            current = str(config.get("currentValue") or "").strip()
            for option in _model_rows(config):
                if isinstance(option, dict) and option.get("value") == current:
                    return _canonical_selector(option)
            return current
        return ""

    def _raw_model_selector_from_options(self, config_options: Any) -> str:
        if not isinstance(config_options, list):
            return ""
        for config in config_options:
            if not isinstance(config, dict):
                continue
            if config.get("id") == "model" or config.get("category") == "model":
                return str(config.get("currentValue") or "").strip()
        return ""

    def _model_selector_for_request(self, requested_model: str) -> str:
        metadata = self.last_model_metadata.get(requested_model) or {}
        return str(metadata.get("selector") or requested_model).strip()

    def _stable_answering_model(self, actual_model: str) -> str:
        return _CONTEXT_HINT_RE.sub("", actual_model.strip())

    def _handle_extension_notification(
        self, message: dict[str, Any], model_state: dict[str, Any] | None
    ) -> bool:
        if message.get("method") != "_claude/sdkMessage":
            return False
        if model_state is None:
            return True
        raw = (message.get("params") or {}).get("message") or {}
        actual = ""
        if (
            isinstance(raw, dict)
            and raw.get("type") == "assistant"
            and raw.get("parent_tool_use_id") in (None, "")
        ):
            actual = str((raw.get("message") or {}).get("model") or "").strip()
        elif isinstance(raw, dict) and raw.get("type") == "stream_event":
            event = raw.get("event") or {}
            if (
                raw.get("parent_tool_use_id") in (None, "")
                and isinstance(event, dict)
                and event.get("type") == "message_start"
            ):
                actual = str(
                    (event.get("message") or {}).get("model") or ""
                ).strip()
        if actual and actual != "<synthetic>":
            model_state["actual_model"] = actual
        if isinstance(raw, dict) and raw.get("type") == "result":
            model_usage = raw.get("modelUsage")
            if isinstance(model_usage, dict):
                model_state["model_usage"] = model_usage
        return True

    def _validate_answering_model(
        self, requested_model: str | None, confirmed_model: str
    ) -> None:
        requested = _CONTEXT_HINT_RE.sub("", str(requested_model or "").strip())
        actual = _CONTEXT_HINT_RE.sub("", confirmed_model.strip())
        if not actual:
            raise RuntimeError(
                "Claude Agent ACP actual-model telemetry was missing from the raw "
                "SDK assistant/message_start events. The completed response was rejected; "
                "the picker-confirmed model is not accepted as proof of the answering model."
            )
        if requested and actual and requested != actual:
            raise RuntimeError(
                f"Claude Agent ACP requested '{requested}' but the SDK answered "
                f"with '{actual}'. The fallback response was rejected; choose an "
                "advertised model that can answer this request."
            )

    @staticmethod
    def _matching_model_usage(
        actual_model: str, model_usage: Any
    ) -> dict[str, Any] | None:
        if not isinstance(model_usage, dict):
            return None
        actual = _CONTEXT_HINT_RE.sub("", actual_model.strip())
        for key, usage in model_usage.items():
            if not isinstance(usage, dict):
                continue
            canonical = _CONTEXT_HINT_RE.sub(
                "", str(usage.get("canonicalModel") or "").strip()
            )
            usage_key = _CONTEXT_HINT_RE.sub("", str(key).strip())
            if actual and actual in {canonical, usage_key}:
                return usage
        return None

    def _validate_serving_provider(
        self, actual_model: str, model_state: dict[str, Any]
    ) -> None:
        usage = self._matching_model_usage(
            actual_model,
            model_state.get("model_usage"),
        )
        provider = str((usage or {}).get("provider") or "").strip()
        self.last_serving_provider = provider
        if not provider:
            raise RuntimeError(
                "Claude Agent ACP provider telemetry was missing for the actual answering "
                "model. The completed response was rejected because firstParty routing "
                "could not be proven."
            )
        if provider != "firstParty":
            raise RuntimeError(
                f"Claude Agent ACP reported serving provider '{provider}'. Only "
                "firstParty Claude Max/Pro subscription routing is allowed; the completed "
                "response was rejected."
            )

        context_window = (usage or {}).get("contextWindow")
        stable_model = _CONTEXT_HINT_RE.sub("", actual_model.strip())
        if isinstance(context_window, int) and context_window > 0 and stable_model:
            from agent.model_metadata import save_context_length

            save_context_length(
                stable_model,
                CLAUDE_ACP_MARKER_BASE_URL,
                context_window,
            )

    def _matching_response_usage(
        self, actual_model: str, model_state: dict[str, Any]
    ) -> dict[str, Any] | None:
        return self._matching_model_usage(
            actual_model, model_state.get("model_usage")
        )

    def _completion_usage(self, result: SimpleNamespace) -> SimpleNamespace:
        raw = getattr(result, "model_usage", None) or {}

        def _number(*names: str) -> int:
            for name in names:
                value = raw.get(name)
                if isinstance(value, (int, float)) and value >= 0:
                    return int(value)
            return 0

        cache_read = _number("cacheReadInputTokens", "cache_read_input_tokens")
        cache_creation = _number(
            "cacheCreationInputTokens", "cache_creation_input_tokens"
        )
        prompt = (
            _number("inputTokens", "input_tokens") + cache_read + cache_creation
        )
        output = _number("outputTokens", "output_tokens")
        try:
            cost_usd = float(raw.get("costUSD", raw.get("cost_usd", 0.0)) or 0.0)
        except (TypeError, ValueError):
            cost_usd = 0.0
        logger.info(
            "Claude ACP usage prompt_tokens=%d output_tokens=%d "
            "cache_read_tokens=%d cache_creation_tokens=%d cost_usd=%.6f",
            prompt,
            output,
            cache_read,
            cache_creation,
            cost_usd,
        )
        return SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=output,
            total_tokens=prompt + output,
            prompt_tokens_details=SimpleNamespace(
                cached_tokens=cache_read,
                cache_write_tokens=cache_creation,
            ),
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=cache_creation,
            cost_usd=cost_usd,
        )

    def discover_models(self, *, timeout_seconds: float = 30.0) -> list[str]:
        """Return exact model ids advertised by the ACP session model selector."""
        result = self._run_acp_session(
            prompt_text=None,
            timeout_seconds=timeout_seconds,
            require_model_config=True,
        )
        return list(result.advertised_models)

    def _execute_prompt(
        self,
        prompt_text: str,
        *,
        model: str | None,
        timeout_seconds: float,
    ) -> SimpleNamespace:
        requested_model = str(model or "").strip()
        if not requested_model:
            raise ValueError(
                "Claude Agent ACP requires an explicit model selected from its "
                "advertised ACP session model config option."
            )
        return self._run_acp_session(
            prompt_text=prompt_text,
            timeout_seconds=timeout_seconds,
            requested_model=requested_model,
            require_model_config=True,
        )


__all__ = [
    "CLAUDE_ACP_SYSTEM_PROMPT",
    "CLAUDE_ACP_MARKER_BASE_URL",
    "DEFAULT_CLAUDE_ACP_COMMAND",
    "ClaudeACPClient",
]
