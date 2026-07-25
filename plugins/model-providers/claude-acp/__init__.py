"""Claude Agent ACP provider profile."""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class ClaudeACPProfile(ProviderProfile):
    """Claude Agent SDK through a local ACP subprocess and subscription login."""

    def create_client(self, **client_kwargs: Any) -> Any:
        from agent.claude_acp_client import ClaudeACPClient

        return ClaudeACPClient(**client_kwargs)

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> list[str] | None:
        del api_key, base_url
        from agent.claude_acp_client import ClaudeACPClient

        return ClaudeACPClient().discover_models(timeout_seconds=timeout)


claude_acp = ClaudeACPProfile(
    name="claude-acp",
    aliases=("claude-agent-acp",),
    display_name="Claude Agent ACP",
    description="Claude Max/Pro subscription via the official Claude Agent SDK ACP adapter",
    api_mode="chat_completions",
    env_vars=(),
    base_url="acp://claude",
    auth_type="external_process",
    process_command="claude-agent-acp",
    process_args=(),
)

register_provider(claude_acp)
