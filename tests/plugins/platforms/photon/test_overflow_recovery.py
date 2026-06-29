"""Photon adapter resilience to transient Spectrum/Envoy upstream overflow.

Covers the three behaviors that let the adapter ride through a Photon
"reset reason: overflow" event instead of degrading delivery and silently
dying (issue #50185):

  1. ``_is_retryable_error`` classifies the Envoy/sidecar overflow strings as
     retryable so ``_send_with_retry`` actually engages its backoff loop.
  2. ``send_typing`` / ``stop_typing`` are no-ops for Photon because typing is
     cosmetic and the live Photon upstream has repeatedly failed on setTyping.
  3. ``_supervise_sidecar`` detects an unexpected sidecar exit and raises a
     ``retryable=True`` fatal so the gateway reconnect watcher revives the
     platform — instead of returning silently and leaving ``_inbound_loop``
     spinning against a dead port.
  4. ``_monitor_sidecar_health`` promotes degraded upstream stream health
     reported by ``/healthz`` into the same retryable reconnect path.

No Node sidecar is spawned and no ports are bound.
"""
from __future__ import annotations

from typing import Any, Dict

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.photon.adapter import PhotonAdapter


def _make_adapter(monkeypatch: pytest.MonkeyPatch) -> PhotonAdapter:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    cfg = PlatformConfig(enabled=True, token="", extra={})
    return PhotonAdapter(cfg)


# -- Gap 1: retryable classification of overflow errors ---------------------

@pytest.mark.parametrize(
    "error",
    [
        "UNAVAILABLE: internal sidecar error",
        "upstream connect error or disconnect/reset before headers",
        "reset reason: overflow",
        # Case-insensitive: real strings arrive with mixed case.
        "Internal Sidecar Error",
    ],
)
def test_overflow_strings_classified_retryable(error: str) -> None:
    assert PhotonAdapter._is_retryable_error(error) is True


def test_unrelated_error_not_retryable() -> None:
    # A genuine permanent failure must NOT be retried.
    assert PhotonAdapter._is_retryable_error("400 bad request: invalid spaceId") is False
    assert PhotonAdapter._is_retryable_error(None) is False


def test_base_network_patterns_still_match() -> None:
    # The override delegates to the base classifier first, so generic
    # network strings keep working.
    assert PhotonAdapter._is_retryable_error("ConnectError: connection refused") is True


def test_photon_rate_limit_retry_after_uses_cooldown_floor() -> None:
    retry_after = PhotonAdapter._rate_limit_retry_after_seconds(
        "SpectrumCloudError: Rate limited by ip_per_minute "
        "(max 300 per 60s, scope=ip). Retry after 60s."
    )

    assert retry_after is not None
    # A plain 60s retry-after was not enough on the live VPS: each reconnect
    # attempt immediately burned into the same Photon Cloud IP limit again.
    assert retry_after >= 900


def test_non_rate_limit_errors_have_no_retry_after() -> None:
    assert PhotonAdapter._rate_limit_retry_after_seconds("connection refused") is None


# -- Gap 2: typing indicators disabled --------------------------------------

@pytest.mark.asyncio
async def test_send_typing_is_noop_for_reliability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    calls: list[Dict[str, Any]] = []

    async def _fake_call(path: str, payload: Dict[str, Any]) -> Any:
        calls.append(payload)
        return {"ok": True}

    monkeypatch.setattr(adapter, "_sidecar_call", _fake_call)

    await adapter.send_typing("chat-1")

    assert calls == []


@pytest.mark.asyncio
async def test_stop_typing_is_noop_for_reliability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    calls: list[Dict[str, Any]] = []

    async def _fake_call(path: str, payload: Dict[str, Any]) -> Any:
        calls.append(payload)
        return {"ok": True}

    monkeypatch.setattr(adapter, "_sidecar_call", _fake_call)

    await adapter.stop_typing("chat-1")

    assert calls == []


@pytest.mark.asyncio
async def test_send_with_retry_waits_through_transient_photon_outage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    attempts = 0
    sleeps: list[float] = []

    async def _fake_send(**kwargs: Any):
        nonlocal attempts
        attempts += 1
        if attempts < 4:
            from gateway.platforms.base import SendResult

            return SendResult(
                success=False,
                error="Photon sidecar /send returned 500: internal sidecar error",
                retryable=True,
            )
        from gateway.platforms.base import SendResult

        return SendResult(success=True, message_id="msg-ok")

    async def _fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(adapter, "send", _fake_send)
    monkeypatch.setattr("plugins.platforms.photon.adapter.asyncio.sleep", _fake_sleep)

    result = await adapter._send_with_retry("chat-1", "hello")

    assert result.success is True
    assert attempts == 4
    assert sleeps == [3.0, 6.0, 12.0]


# -- Gap 3: sidecar crash detection -----------------------------------------

class _EofStdout:
    """A proc.stdout whose readline() reports immediate EOF (dead sidecar)."""

    def readline(self) -> bytes:
        return b""


class _DeadProc:
    """Minimal subprocess.Popen stand-in for a sidecar that has exited."""

    def __init__(self, exit_code: int = 1) -> None:
        self.stdout = _EofStdout()
        self.stdin = None
        self._exit_code = exit_code

    def poll(self) -> int:
        return self._exit_code


@pytest.mark.asyncio
async def test_unexpected_sidecar_exit_raises_retryable_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    # Simulate a live session whose sidecar then dies underneath it.
    adapter._inbound_running = True

    notified: list[bool] = []

    async def _fake_notify() -> None:
        notified.append(True)

    monkeypatch.setattr(adapter, "_notify_fatal_error", _fake_notify)

    await adapter._supervise_sidecar(_DeadProc(exit_code=137))  # type: ignore[arg-type]

    assert adapter.has_fatal_error is True
    assert adapter.fatal_error_code == "SIDECAR_CRASHED"
    # retryable=True routes the platform into the reconnect watcher rather
    # than crashing the whole gateway.
    assert adapter.fatal_error_retryable is True
    assert adapter._running is False
    assert notified == [True]


@pytest.mark.asyncio
async def test_clean_shutdown_does_not_raise_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    # disconnect() sets _inbound_running = False before stopping the sidecar,
    # so the detection block must NOT fire on a clean shutdown.
    adapter._inbound_running = False

    notified: list[bool] = []

    async def _fake_notify() -> None:
        notified.append(True)

    monkeypatch.setattr(adapter, "_notify_fatal_error", _fake_notify)

    await adapter._supervise_sidecar(_DeadProc(exit_code=0))  # type: ignore[arg-type]

    assert adapter.has_fatal_error is False
    assert notified == []


@pytest.mark.asyncio
async def test_degraded_stream_health_raises_retryable_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    adapter._inbound_running = True
    adapter._sidecar_health_interval = 0.0

    async def _fake_call(path: str, payload: Dict[str, Any]) -> Any:
        assert path == "/healthz"
        return {
            "ok": True,
            "stream": {
                "ok": False,
                "state": "degraded",
                "degradedForMs": 120000,
                "lastIssue": "[spectrum.stream] stream interrupted; reconnecting",
            },
        }

    notified: list[bool] = []

    async def _fake_notify() -> None:
        notified.append(True)
        adapter._inbound_running = False

    monkeypatch.setattr(adapter, "_sidecar_call", _fake_call)
    monkeypatch.setattr(adapter, "_notify_fatal_error", _fake_notify)

    await adapter._monitor_sidecar_health()

    assert adapter.has_fatal_error is True
    assert adapter.fatal_error_code == "UPSTREAM_STREAM_DEGRADED"
    assert adapter.fatal_error_retryable is True
    assert notified == [True]


@pytest.mark.asyncio
async def test_short_stream_degradation_waits_for_sidecar_recovery_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    adapter._inbound_running = True
    adapter._sidecar_health_interval = 0.0

    async def _fake_call(path: str, payload: Dict[str, Any]) -> Any:
        assert path == "/healthz"
        adapter._inbound_running = False
        return {
            "ok": True,
            "stream": {
                "ok": False,
                "state": "degraded",
                "degradedForMs": 10934,
                "restartAfterMs": 90000,
                "lastIssue": "[spectrum.stream] stream interrupted; reconnecting",
            },
        }

    notified: list[bool] = []

    async def _fake_notify() -> None:
        notified.append(True)

    monkeypatch.setattr(adapter, "_sidecar_call", _fake_call)
    monkeypatch.setattr(adapter, "_notify_fatal_error", _fake_notify)

    await adapter._monitor_sidecar_health()

    assert adapter.has_fatal_error is False
    assert notified == []
