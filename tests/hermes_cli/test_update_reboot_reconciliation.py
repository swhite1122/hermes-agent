"""A reboot retires pre-boot modules, not same-boot update obligations."""
from datetime import datetime, timezone

import pytest

from hermes_cli import update_cmd_fleet as fleet
from hermes_cli import update_receipt


@pytest.mark.parametrize(
    "finished,boot,state,sha,marker,expected",
    [
        (100, 200, "current", "new", False, False),
        (200, 100, "current", "new", False, True),
        (100, 200, "stale", "old", False, True),
        (100, 200, "current", "new", True, True),
        (None, 200, "current", "new", False, True),
        ("invalid", 200, "current", "new", False, True),
    ],
)
def test_reboot_reconciles_only_finished_old_obligations(
    monkeypatch, tmp_path, finished, boot, state, sha, marker, expected
):
    import psutil
    from hermes_cli import update_cmd

    stamp = datetime.fromtimestamp(finished, timezone.utc).isoformat() if isinstance(finished, int) else finished
    receipt = {
        "finished_at": stamp,
        "plan": {"runtimes": [
            {"kind": "gateway", "profile": "default", "code_sha": "old"},
            {"kind": "dashboard", "profile": "default"},
        ]},
        "fleet": [{"profile": "default", "state": "stale", "code_sha": "old"}],
    }
    monkeypatch.setattr(update_receipt, "read_latest_receipt", lambda: receipt)
    monkeypatch.setattr(update_receipt, "collect_fleet_versions", lambda: [
        {"profile": "default", "state": state, "code_sha": sha}
    ])
    monkeypatch.setattr(update_cmd, "_current_checkout_sha", lambda: "new")
    monkeypatch.setattr(psutil, "boot_time", lambda: boot)
    pending = tmp_path / "pending"
    if marker:
        pending.touch()
    monkeypatch.setattr(fleet, "_fleet_restart_pending_marker_path", lambda: pending)
    assert fleet._pending_fleet_restart_needed() is expected
    assert receipt["fleet"][0]["code_sha"] == "old"
