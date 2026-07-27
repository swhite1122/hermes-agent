"""Safe health probe for the Claude Agent ACP provider."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from agent.claude_acp_client import ClaudeACPClient
from hermes_cli.auth import get_external_process_provider_status

CLAUDE_ACP_CANARY_MARKER = "HERMES_CLAUDE_ACP_OK"
CLAUDE_ACP_CANARY_MODELS = ("claude-opus-5", "claude-sonnet-5")
CLAUDE_ACP_VERSION = "0.62.0"
CLAUDE_SDK_VERSION = "0.3.219"


def _safe_error(prefix: str, exc: BaseException) -> str:
    return f"{prefix}: {type(exc).__name__}"


def _message_content(response: Any) -> str:
    try:
        choice = response.choices[0]
        message = choice.message
        return str(getattr(message, "content", "") or "").strip()
    except Exception:
        return ""


def _package_versions(adapter_command: str) -> tuple[str, str]:
    """Read installed package versions without executing package code."""
    if not adapter_command:
        return "", ""
    command_path = Path(adapter_command).expanduser()
    candidates = [command_path]
    try:
        candidates.append(command_path.resolve())
    except OSError:
        pass

    node_modules = None
    for candidate in candidates:
        for parent in (candidate.parent, *candidate.parents):
            if parent.name == "node_modules":
                node_modules = parent
                break
        if node_modules is not None:
            break
    if node_modules is None and command_path.parent.name == ".bin":
        node_modules = command_path.parent.parent
    if node_modules is None:
        return "", ""

    def read_version(path: Path) -> str:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            return ""
        return str(payload.get("version") or "") if isinstance(payload, dict) else ""

    adapter_version = read_version(
        node_modules / "@agentclientprotocol" / "claude-agent-acp" / "package.json"
    )
    sdk_version = read_version(
        node_modules / "@anthropic-ai" / "claude-agent-sdk" / "package.json"
    )
    return adapter_version, sdk_version


def probe_claude_acp(
    run_canary: bool = False,
    cwd: str | Path | None = None,
) -> dict[str, Any]:
    """Return a credential-safe Claude ACP health snapshot."""
    status = get_external_process_provider_status("claude-acp")
    adapter_command = str(
        status.get("resolved_command") or status.get("command") or ""
    ).strip()
    installed = bool(status.get("installed") or status.get("configured"))
    logged_in = bool(status.get("logged_in"))
    adapter_version, sdk_version = _package_versions(adapter_command)

    result: dict[str, Any] = {
        "ok": False,
        "installed": installed,
        "logged_in": logged_in,
        "adapter_command": adapter_command,
        "adapter_version": adapter_version,
        "sdk_version": sdk_version,
        "models": [],
        "canaries": {},
        "error": "",
    }

    if not installed:
        result["error"] = str(
            status.get("error") or "Claude Agent ACP adapter is not installed."
        )
        return result
    if adapter_version != CLAUDE_ACP_VERSION or sdk_version != CLAUDE_SDK_VERSION:
        result["error"] = (
            "Claude ACP package version mismatch "
            f"(found {adapter_version or 'unknown'} / {sdk_version or 'unknown'}, "
            f"expected {CLAUDE_ACP_VERSION} / {CLAUDE_SDK_VERSION})."
        )
        return result
    if not logged_in:
        result["error"] = str(
            status.get("error") or "Claude CLI is not logged in."
        )
        return result

    client_kwargs: dict[str, Any] = {}
    if adapter_command:
        client_kwargs["acp_command"] = adapter_command
    if cwd is not None:
        client_kwargs["acp_cwd"] = str(Path(cwd).resolve())

    client: ClaudeACPClient | None = None
    try:
        try:
            active_client = ClaudeACPClient(**client_kwargs)
            client = active_client
            models = active_client.discover_models(timeout_seconds=30.0)
        except Exception as exc:
            result["error"] = _safe_error("Claude ACP model discovery failed", exc)
            return result

        result["models"] = list(models)
        canaries: dict[str, bool] = {}
        if run_canary:
            for model in CLAUDE_ACP_CANARY_MODELS:
                try:
                    response = active_client.chat.completions.create(
                        model=model,
                        messages=[
                            {
                                "role": "user",
                                "content": f"Reply exactly {CLAUDE_ACP_CANARY_MARKER}",
                            }
                        ],
                        timeout=120,
                    )
                    canaries[model] = (
                        _message_content(response) == CLAUDE_ACP_CANARY_MARKER
                    )
                except Exception:
                    canaries[model] = False
            result["canaries"] = canaries

        result["ok"] = bool(
            result["installed"]
            and result["logged_in"]
            and result["adapter_version"] == CLAUDE_ACP_VERSION
            and result["sdk_version"] == CLAUDE_SDK_VERSION
            and result["models"]
            and (not run_canary or all(canaries.values()))
        )
        if run_canary and not result["ok"] and not result["error"]:
            result["error"] = "Claude ACP canary failed."
        return result
    finally:
        if client is not None:
            client.close()


def _print_human(result: dict[str, Any]) -> None:
    status = "ok" if result.get("ok") else "failed"
    print(f"Claude ACP health: {status}")
    print(f"installed: {bool(result.get('installed'))}")
    print(f"logged_in: {bool(result.get('logged_in'))}")
    if result.get("adapter_command"):
        print(f"adapter_command: {result['adapter_command']}")
    print(f"adapter_version: {result.get('adapter_version') or 'unknown'}")
    print(f"sdk_version: {result.get('sdk_version') or 'unknown'}")
    models = result.get("models") or []
    print("models: " + (", ".join(models) if models else "none"))
    canaries = result.get("canaries") or {}
    for model in CLAUDE_ACP_CANARY_MODELS:
        if model in canaries:
            print(f"canary {model}: {'ok' if canaries[model] else 'failed'}")
    if result.get("error"):
        print(f"error: {result['error']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m hermes_cli.claude_acp_health",
        description="Probe Claude Agent ACP installation, login, discovery, and canaries.",
    )
    parser.add_argument("--canary", action="store_true", help="Run Opus/Sonnet canaries.")
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    args = parser.parse_args(argv)

    result = probe_claude_acp(run_canary=args.canary)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        _print_human(result)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
