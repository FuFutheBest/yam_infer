"""Seven-value follower endpoint used by the follower worker.

Vendored from SuperInference-feat-i2rt_NEDF3 @0dbfa5d
``devices/robots/i2rt_teleop/endpoints.py`` (see VENDORED.md).
Local changes: follower only (leader endpoint removed), only the integrated
``i2rt_can`` gripper backend is supported (the LKTech serial gripper is not used
on this robot), plus ``FakeFollowerEndpoint`` for hardware-free tests.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from .config import FollowerConfig, GripperConfig

ARM_DOF = 6
ENDPOINT_DOF = 7


def _vector(value: Any, size: int, field: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.size != size:
        raise ValueError(f"{field} must contain exactly {size} values")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{field} must contain only finite values")
    return result


class CompositeFollowerEndpoint:
    """A YAM arm with its integrated I2RT gripper behind a seven-value API."""

    def __init__(self, arm: Any, gripper_config: GripperConfig) -> None:
        if gripper_config.backend != "i2rt_can":
            raise ValueError(
                f"gripper backend {gripper_config.backend!r} is not supported here "
                "(only the integrated 'i2rt_can' gripper)"
            )
        self.arm = arm
        self.gripper_config = gripper_config
        self._closed = False

    def start(self) -> None:
        pass

    def get_info(self) -> np.ndarray:
        observations = self.arm.get_observations()
        joints = _vector(observations["joint_pos"], ARM_DOF, "follower joints")
        gripper = _vector(observations.get("gripper_pos"), 1, "I2RT gripper feedback")[0]
        return np.concatenate([joints, [float(np.clip(gripper, 0.0, 1.0))]])

    def command_joint_pos(self, joint_pos: Any) -> None:
        command = _vector(joint_pos, ENDPOINT_DOF, "follower command")
        self.arm.command_joint_pos(command.copy())

    def update_kp_kd(self, kp: Any, kd: Any) -> None:
        self.arm.update_kp_kd(
            _vector(kp, ENDPOINT_DOF, "follower kp"),
            _vector(kd, ENDPOINT_DOF, "follower kd"),
        )

    def set_feedback_enabled(self, enabled: bool, kp: Any, kd: Any) -> None:
        del enabled, kp, kd

    def enter_safe_state(self, state: Any) -> None:
        self.command_joint_pos(_vector(state, ENDPOINT_DOF, "follower state"))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.arm.close()


def create_follower_endpoint(config: FollowerConfig) -> CompositeFollowerEndpoint:
    """Construct a physical follower endpoint (opens the CAN bus, motors on)."""

    # vendored SDK in third_party/i2rt, put on PYTHONPATH by FollowerArms when it starts the worker
    from i2rt.robots.get_robot import get_yam_robot  # pyright: ignore[reportMissingImports]
    from i2rt.robots.utils import GripperType  # pyright: ignore[reportMissingImports]

    if config.gripper.backend != "i2rt_can":
        raise ValueError(f"gripper backend {config.gripper.backend!r} is not supported here")
    gripper_type = GripperType.from_string_name(config.gripper.i2rt_type)
    arm = get_yam_robot(channel=config.can_channel, gripper_type=gripper_type)
    endpoint = CompositeFollowerEndpoint(arm=arm, gripper_config=config.gripper)
    try:
        endpoint.start()
    except Exception:
        endpoint.close()
        raise
    return endpoint


class FakeFollowerEndpoint:
    """Hardware-free stand-in: the state follows the last command with a ~130 ms lag."""

    def __init__(self, initial: Any | None = None, tau_s: float = 0.13) -> None:
        start = (
            np.zeros(ENDPOINT_DOF) if initial is None else _vector(initial, ENDPOINT_DOF, "initial")
        )
        start = start.copy()
        if initial is None:
            start[-1] = 1.0  # gripper open
        self._state = start
        self._target = start.copy()
        self._tau = tau_s
        self._t = time.monotonic()
        self._lock = threading.Lock()
        self.commands = 0

    def _advance(self) -> None:
        now = time.monotonic()
        a = 1.0 - np.exp(-(now - self._t) / self._tau)
        self._state = self._state + (self._target - self._state) * a
        self._t = now

    def get_info(self) -> np.ndarray:
        with self._lock:
            self._advance()
            return self._state.copy()

    def command_joint_pos(self, joint_pos: Any) -> None:
        with self._lock:
            self._advance()
            self._target = _vector(joint_pos, ENDPOINT_DOF, "follower command").copy()
            self.commands += 1

    def update_kp_kd(self, kp: Any, kd: Any) -> None:
        del kp, kd

    def set_feedback_enabled(self, enabled: bool, kp: Any, kd: Any) -> None:
        del enabled, kp, kd

    def enter_safe_state(self, state: Any) -> None:
        self.command_joint_pos(state)

    def close(self) -> None:
        pass
