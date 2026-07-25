"""Claude Agent ACP provider profile."""

from providers import register_provider
from providers.base import ProviderProfile


class ClaudeACPProfile(ProviderProfile):
    """Claude Agent SDK through a local ACP subprocess and subscription login."""

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
)

register_provider(claude_acp)
