"""Regression tests for single-owner cron scheduling across profiles."""

import threading

import pytest


def test_default_config_cron_enabled_is_true():
    """Cron is enabled by default; specialist profiles opt out explicitly."""
    from hermes_cli.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["cron"]["enabled"] is True


@pytest.mark.parametrize("value", [False, "false", "0", "no", "off", "disabled"])
def test_cron_enabled_false_values(value):
    from cron.scheduler_provider import _cron_enabled

    assert _cron_enabled({"cron": {"enabled": value}}) is False


@pytest.mark.parametrize("value", [True, "true", "1", "yes", "on", None])
def test_cron_enabled_true_values(value):
    from cron.scheduler_provider import _cron_enabled

    cfg = {"cron": {}}
    if value is not None:
        cfg["cron"]["enabled"] = value

    assert _cron_enabled(cfg) is True


def test_resolve_disabled_config_returns_noop_scheduler(monkeypatch):
    """cron.enabled=false lets a profile gateway chat without ticking shared cron."""
    import hermes_cli.config as cfg
    from cron import scheduler_provider as sp

    monkeypatch.setattr(cfg, "load_config", lambda: {"cron": {"enabled": False, "provider": ""}})
    provider = sp.resolve_cron_scheduler()
    assert provider.name == "disabled"
    assert provider.start(threading.Event()) is None


def test_disabled_scheduler_fire_due_never_claims_job(monkeypatch):
    import cron.jobs
    from cron.scheduler_provider import DisabledCronScheduler

    def fail_if_called(_job_id):
        raise AssertionError("disabled cron must not claim jobs")

    monkeypatch.setattr(cron.jobs, "claim_job_for_fire", fail_if_called)

    assert DisabledCronScheduler().fire_due("job-123") is False