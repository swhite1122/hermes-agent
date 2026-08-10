from types import SimpleNamespace

from hermes_cli.moa_config import normalize_moa_config


def test_moa_max_iterations_is_a_positive_optional_cap():
    assert normalize_moa_config({"max_iterations": 6})["max_iterations"] == 6
    assert normalize_moa_config({"max_iterations": 0})["max_iterations"] is None
    assert normalize_moa_config({"max_iterations": "bad"})["max_iterations"] is None


def test_moa_turn_budget_caps_global_agent_budget():
    from agent import conversation_loop

    resolver = getattr(conversation_loop, "_resolve_turn_max_iterations", None)
    assert callable(resolver), "MoA needs a per-turn iteration-budget resolver"

    moa_agent = SimpleNamespace(
        max_iterations=250,
        provider="moa",
        client=SimpleNamespace(max_iterations=6),
    )
    normal_agent = SimpleNamespace(
        max_iterations=250,
        provider="openai-codex",
        client=SimpleNamespace(max_iterations=6),
    )

    assert resolver(moa_agent) == 6
    assert resolver(normal_agent) == 250
