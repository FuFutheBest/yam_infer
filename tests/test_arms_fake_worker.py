"""FollowerArms <-> real worker processes over portal RPC, with simulated arms (--fake).

Exercises the whole production path except CAN: process start, port wait, session phases,
heartbeat (keeps the 150 ms lease alive while we idle), fault recovery, shutdown.
"""

from __future__ import annotations

import socket
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from yam_infer.hw.arms import DEFAULT_HW_CONFIG, FollowerArms


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture()
def hw_config(tmp_path: Path) -> Path:
    """Project hardware config on free ports (so tests never collide with a running robot)."""
    cfg = yaml.safe_load(DEFAULT_HW_CONFIG.read_text())
    for side in ("left", "right"):
        for role in ("leader", "follower"):
            cfg["arms"][side][role]["rpc_port"] = _free_port()
    path = tmp_path / "hardware.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_fake_workers_session_heartbeat_and_close(hw_config: Path, tmp_path: Path) -> None:
    arms = FollowerArms(hw_config, log_dir=tmp_path, fake=True, start_timeout=60)
    try:
        state = arms.get_state()
        assert state.shape == (14,)
        arms.begin()
        arms.submit(state, "ALIGNING")
        arms.activate()
        target = state.copy()
        target[7] += 0.2  # right shoulder
        arms.submit(target, "ACTIVE")
        time.sleep(0.6)  # > 4 leases: only the heartbeat keeps the workers out of FAULT
        assert arms.heartbeat_error is None
        np.testing.assert_allclose(arms.get_state()[7], target[7], atol=0.02)
    finally:
        arms.close()
    assert all(p.poll() is not None for p in arms.procs.values())


def test_refuses_to_attach_to_a_foreign_worker(hw_config: Path, tmp_path: Path) -> None:
    port = yaml.safe_load(hw_config.read_text())["arms"]["left"]["follower"]["rpc_port"]
    with socket.create_server(("127.0.0.1", port)):
        with pytest.raises(RuntimeError, match="already in use"):
            FollowerArms(hw_config, log_dir=tmp_path, fake=True)
