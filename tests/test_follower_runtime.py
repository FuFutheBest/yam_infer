"""Worker-local watchdog (yam_infer.hw.follower.runtime), no hardware.

Ported from SuperInference tests/devices/test_i2rt_teleop_watchdog.py @0dbfa5d (follower cases).
"""

from __future__ import annotations

import threading
import time
from dataclasses import replace

import numpy as np
import pytest
from portal import packlib

from yam_infer.hw.follower.config import ControlConfig
from yam_infer.hw.follower.runtime import (
    ALIGNING,
    FAULT,
    SAFE_HOLD,
    WorkerControlRuntime,
)


class FakeClock:
    def __init__(self) -> None:
        self.now_ns = 1_000_000_000

    def __call__(self) -> int:
        return self.now_ns

    def advance(self, seconds: float) -> None:
        self.now_ns += int(seconds * 1e9)


class FakeEndpoint:
    def __init__(self, *, follow_gripper: bool = True) -> None:
        self.follow_gripper = follow_gripper
        self.state = np.arange(7, dtype=np.float64)
        self.commands: list[np.ndarray] = []
        self.safe_states: list[np.ndarray] = []
        self.read_error: Exception | None = None

    def get_info(self):
        if self.read_error is not None:
            raise self.read_error
        return self.state.copy()

    def command_joint_pos(self, command) -> None:
        value = np.asarray(command, dtype=np.float64).copy()
        self.commands.append(value)
        self.state[:6] = value[:6]
        if self.follow_gripper:
            self.state[-1] = value[-1]

    def enter_safe_state(self, state) -> None:
        self.safe_states.append(np.asarray(state, dtype=np.float64).copy())

    def set_feedback_enabled(self, enabled, kp, kd) -> None:
        pass


class BlockingCommandEndpoint(FakeEndpoint):
    """Command call emulating a stuck hardware API."""

    def __init__(self) -> None:
        super().__init__()
        self.command_started = threading.Event()
        self.release_command = threading.Event()

    def command_joint_pos(self, command) -> None:
        self.command_started.set()
        self.release_command.wait(timeout=2.0)
        super().command_joint_pos(command)


class BlockingStartupEndpoint(FakeEndpoint):
    """First feedback read waits for explicit release."""

    def __init__(self) -> None:
        super().__init__()
        self.read_started = threading.Event()
        self.release_read = threading.Event()

    def get_info(self):
        self.read_started.set()
        self.release_read.wait(timeout=2.0)
        return super().get_info()


def _runtime() -> tuple[WorkerControlRuntime, FakeEndpoint, FakeClock]:
    endpoint = FakeEndpoint()
    clock = FakeClock()
    control = replace(
        ControlConfig(),
        command_frame_max_age_s=0.05,
        command_lease_timeout_s=0.10,
        feedback_timeout_s=0.05,
        following_error_fault_threshold=10.0,
    )
    runtime = WorkerControlRuntime(
        endpoint, role="follower", side="left", control=control, clock_ns=clock
    )
    runtime.run_once()
    return runtime, endpoint, clock


def _frame(clock: FakeClock, sequence: int, phase: str, size: int = 7):
    return {
        "session_id": "session",
        "sequence": sequence,
        "issued_at_ns": clock(),
        "phase": phase,
        "joint_position": np.linspace(0.0, 0.6, size),
    }


def _portal_roundtrip(value):
    wire = b"".join(bytes(part) for part in packlib.pack(value))
    return packlib.unpack(wire)


def test_runtime_starts_by_holding_observed_follower_position() -> None:
    runtime, endpoint, _ = _runtime()
    assert runtime.get_status()["phase"] == SAFE_HOLD
    np.testing.assert_allclose(endpoint.safe_states, [np.arange(7)])


def test_runtime_start_can_be_cancelled_while_hardware_read_blocks() -> None:
    endpoint = BlockingStartupEndpoint()
    runtime = WorkerControlRuntime(endpoint, role="follower", side="left", control=ControlConfig())
    cancel_event = threading.Event()
    result: list[bool] = []
    start_thread = threading.Thread(
        target=lambda: result.append(runtime.start(timeout_s=1.0, cancel_event=cancel_event))
    )
    start_thread.start()
    assert endpoint.read_started.wait(timeout=1.0)
    cancel_event.set()
    runtime.request_stop()
    start_thread.join(timeout=0.2)
    assert not start_thread.is_alive()
    assert result == [False]
    endpoint.release_read.set()
    assert runtime.close()


def test_runtime_applies_only_current_ordered_session_frames() -> None:
    runtime, endpoint, clock = _runtime()
    runtime.begin_control_session("session")
    runtime.submit_control_frame(_frame(clock, 0, ALIGNING))
    runtime.run_once()
    np.testing.assert_allclose(endpoint.commands[-1], np.linspace(0.0, 0.6, 7))
    assert runtime.get_status()["last_applied_sequence"] == 0

    with pytest.raises(ValueError, match="sequence must increase"):
        runtime.submit_control_frame(_frame(clock, 0, ALIGNING))
    stale = _frame(clock, 1, ALIGNING)
    clock.advance(0.06)
    with pytest.raises(ValueError, match="stale"):
        runtime.submit_control_frame(stale)


def test_rpc_contract_survives_portal_transport() -> None:
    runtime, endpoint, clock = _runtime()
    runtime.begin_control_session("session")
    runtime.submit_control_frame(_portal_roundtrip(_frame(clock, 0, ALIGNING)))
    runtime.run_once()
    runtime.run_once()
    sample = _portal_roundtrip(runtime.get_sample())
    np.testing.assert_allclose(endpoint.commands[-1], np.linspace(0.0, 0.6, 7))
    np.testing.assert_allclose(sample["state"], endpoint.state)
    assert sample["phase"] == ALIGNING
    assert int(sample["last_applied_sequence"]) == 0


def test_activate_requires_an_applied_alignment_frame() -> None:
    runtime, _, clock = _runtime()
    runtime.begin_control_session("session")
    with pytest.raises(RuntimeError, match="before an alignment frame"):
        runtime.set_phase("session", "ACTIVE")
    runtime.submit_control_frame(_frame(clock, 0, ALIGNING))
    runtime.run_once()
    assert runtime.set_phase("session", "ACTIVE")["phase"] == "ACTIVE"


def test_command_lease_expiry_latches_fault_and_holds_current_position() -> None:
    runtime, endpoint, clock = _runtime()
    runtime.begin_control_session("session")
    runtime.submit_control_frame(_frame(clock, 0, ALIGNING))
    runtime.run_once()
    safe_count = len(endpoint.safe_states)

    clock.advance(0.11)
    runtime.run_once()

    status = runtime.get_status()
    assert status["phase"] == FAULT
    assert status["fault_reason"] == "command lease expired"
    assert len(endpoint.safe_states) == safe_count + 1
    with pytest.raises(RuntimeError, match="faulted"):
        runtime.begin_control_session("new-session")
    runtime.reset_fault()
    assert runtime.get_status()["phase"] == SAFE_HOLD


def test_feedback_timeout_latches_fault() -> None:
    runtime, endpoint, clock = _runtime()
    endpoint.read_error = RuntimeError("CAN timeout")
    clock.advance(0.06)
    runtime.run_once()
    assert runtime.get_status()["phase"] == FAULT
    assert "feedback read failed" in runtime.get_status()["fault_reason"]


def test_follower_watchdog_ignores_gripper_tracking_error() -> None:
    endpoint = FakeEndpoint(follow_gripper=False)
    clock = FakeClock()
    control = replace(
        ControlConfig(), following_error_fault_threshold=0.1, following_error_fault_samples=1
    )
    runtime = WorkerControlRuntime(
        endpoint, role="follower", side="left", control=control, clock_ns=clock
    )
    runtime.run_once()
    runtime.begin_control_session("session")
    command = endpoint.state.copy()
    command[-1] = 0.0
    runtime.submit_control_frame(
        {
            "session_id": "session",
            "sequence": 0,
            "issued_at_ns": clock(),
            "phase": ALIGNING,
            "joint_position": command,
        }
    )
    runtime.run_once()
    assert endpoint.state[-1] == 6.0
    assert runtime.get_status()["phase"] == ALIGNING
    assert runtime.get_status()["fault_reason"] is None


def test_follower_watchdog_reports_side_joint_and_phase() -> None:
    runtime, endpoint, clock = _runtime()
    runtime.control = replace(
        runtime.control, following_error_fault_threshold=0.1, following_error_fault_samples=1
    )
    runtime.begin_control_session("session")
    command = endpoint.state.copy()
    command[3] += 0.2
    runtime.submit_control_frame(
        {
            "session_id": "session",
            "sequence": 0,
            "issued_at_ns": clock(),
            "phase": ALIGNING,
            "joint_position": command,
        }
    )
    runtime.run_once()
    assert runtime.get_status()["fault_reason"] == (
        "left follower following error 0.200 rad at joint 4 exceeds 0.100 rad during ALIGNING"
    )


def test_threaded_watchdog_remains_responsive_when_hardware_command_blocks() -> None:
    endpoint = BlockingCommandEndpoint()
    control = replace(
        ControlConfig(),
        command_frame_max_age_s=0.20,
        command_lease_timeout_s=0.20,
        feedback_timeout_s=0.05,
        watchdog_check_interval_s=0.005,
    )
    runtime = WorkerControlRuntime(endpoint, role="follower", side="left", control=control)
    runtime.start(timeout_s=1.0)
    try:
        runtime.begin_control_session("session")
        runtime.submit_control_frame(
            {
                "session_id": "session",
                "sequence": 0,
                "issued_at_ns": time.monotonic_ns(),
                "phase": ALIGNING,
                "joint_position": np.linspace(0.0, 0.6, 7),
            }
        )
        assert endpoint.command_started.wait(timeout=1.0)
        started = time.monotonic()
        status = runtime.get_status()
        assert time.monotonic() - started < 0.05
        assert status["io_inflight"] is True
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and runtime.get_status()["phase"] != FAULT:
            time.sleep(0.005)
        status = runtime.get_status()
        assert status["phase"] == FAULT
        assert "hardware I/O stalled" in status["fault_reason"]
    finally:
        endpoint.release_command.set()
        assert runtime.close()
