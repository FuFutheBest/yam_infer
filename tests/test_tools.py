"""free_hw (who holds the robot) and home_arms (home from any state), with fake workers."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from yam_infer.hw.arms import PROJECT_ROOT, SIDES, load_ports, port_open, spawn_worker
from yam_infer.tools import free_hw

from .test_arms_fake_worker import hw_config  # noqa: F401 - fixture (config on free ports)


def _wait_port(port: int, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while not port_open(port):
        assert time.monotonic() < deadline, f"port {port} not ready"
        time.sleep(0.2)


def test_python_module_parses_interpreter_options() -> None:
    proc = subprocess.Popen(
        [sys.executable, "-u", "-X", "utf8", "-m", "http.server", "0"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(0.5)
        assert free_hw.python_module(proc.pid) == "http.server"
        assert not free_hw.is_ours(proc.pid)
        assert free_hw.python_module(os.getppid()) != "yam_infer.run"
    finally:
        proc.kill()
        proc.wait()


def test_scan_reports_worker_on_rpc_port(hw_config: Path, tmp_path: Path) -> None:  # noqa: F811
    port = load_ports(hw_config)["left"]
    proc = spawn_worker("left", hw_config, tmp_path / "w.log", fake=True)
    try:
        _wait_port(port)
        res = free_hw.scan(hw_config)
        assert proc.pid in res.holders
        assert any(r.startswith(f"RPC 端口 {port}") for r in res.holders[proc.pid])
        assert free_hw.is_ours(proc.pid, "yam_infer.hw.follower.worker")
    finally:
        proc.terminate()
        proc.wait(timeout=10)


@pytest.mark.parametrize("orphan", [None, "right"])
def test_home_arms_from_any_state(hw_config: Path, tmp_path: Path, orphan) -> None:  # noqa: F811
    """No worker at all, or one orphaned worker: both arms get homed, nothing is left running."""
    ports = load_ports(hw_config)
    procs = []
    if orphan:
        procs.append(spawn_worker(orphan, hw_config, tmp_path / "orphan.log", fake=True))
        _wait_port(ports[orphan])
    rest = tmp_path / "rest_pose.yaml"
    rest.write_text("left: [0.1, 0, 0, 0, 0, 0, 1]\nright: [-0.1, 0, 0, 0, 0, 0, 1]\n")
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "yam_infer.tools.home_arms",
            "--fake",
            "--yes",
            "--hw-config",
            str(hw_config),
            "--rest-pose",
            str(rest),
            "--record-dir",
            str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        input="",
    )
    assert "归位完成" in out.stdout, out.stdout + out.stderr
    for side in SIDES:
        assert f"{side}: 已到位" in out.stdout
        assert not port_open(ports[side])
    for p in procs:
        assert p.wait(timeout=10) is not None
