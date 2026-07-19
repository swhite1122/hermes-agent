"""Regression test for VPS host/container detection."""

import builtins
import os

import hermes_constants
from hermes_constants import is_container


def test_host_with_docker_child_mount_is_not_container(monkeypatch, tmp_path):
    monkeypatch.setattr(hermes_constants, "_container_detected", None)
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    monkeypatch.setattr(os.path, "exists", lambda _path: False)
    cgroup = tmp_path / "cgroup"
    cgroup.write_text("0::/\n")
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(
        "29 1 8:1 / / rw,relatime shared:1 - ext4 /dev/sda1 rw\n"
        "322 29 0:48 / /var/lib/docker/rootfs/overlayfs/abc rw,relatime "
        "- overlay overlay rw,lowerdir=/var/lib/containerd/snapshots/93/fs\n"
    )
    real_open = builtins.open

    def fake_open(path, *args, **kwargs):
        if path == "/proc/1/cgroup":
            return real_open(cgroup, *args, **kwargs)
        if path == "/proc/self/mountinfo":
            return real_open(mountinfo, *args, **kwargs)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)

    assert is_container() is False
