"""Follower arms through SuperInference's own watchdog-protected worker processes.

We do NOT talk to CAN directly.  We start the same worker the teleoperation uses
(`devices.robots.i2rt_teleop.worker --role follower`, inside the SuperInference venv)
and play the role of the teleop coordinator over its portal RPC:

    begin_control_session ─► ALIGNING frames ─► set_phase(ACTIVE) ─► ACTIVE frames @100 Hz ─► end_control_session
                                                                                (= safe hold at current pose)
Worker safety kept intact: stale-frame rejection (50 ms), command lease (150 ms),
following-error fault (1 rad), SDK joint limits.
Leader arms are not started (no human input during inference).
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np

SIDES = ("left", "right")
ALIGNING, ACTIVE = "ALIGNING", "ACTIVE"


def _die_with_parent() -> None:
    """Child-side: ask Linux to SIGTERM the worker when our process dies (prctl PR_SET_PDEATHSIG).
    A worker's SIGTERM handler releases the arm (zero torque) - same as a normal close().
    """
    try:
        import ctypes
        import signal

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(
            1, signal.SIGTERM
        )  # 1 = PR_SET_PDEATHSIG
    except (OSError, AttributeError):
        pass


def _load_ports(teleop_config: Path) -> dict[str, int]:
    import yaml

    try:
        cfg = yaml.safe_load(teleop_config.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"cannot read teleop config {teleop_config}: {exc}") from exc
    return {s: int(cfg["arms"][s]["follower"]["rpc_port"]) for s in SIDES}


class FollowerArms:
    def __init__(
        self,
        si_root: Path,
        si_python: Path,
        teleop_config: Path,
        rpc_timeout: float = 0.1,
        start_timeout: float = 30.0,
        log_dir: Path | None = None,
    ):
        import portal

        self.si_root = si_root
        self.ports = _load_ports(teleop_config)
        self.timeout = rpc_timeout
        self.session: str | None = None
        self.seq = -1
        self.procs: dict[str, subprocess.Popen] = {}
        self.log_dir = log_dir
        self.clients: dict = {}
        # heartbeat: the worker FAULTs ("command lease expired") if no frame arrives for 150 ms,
        # so while we wait (Enter key, pause, scene reset) the last target is re-sent every ~20 ms.
        self._rpc_lock = threading.RLock()
        self._phase = ALIGNING
        self._last_target: np.ndarray | None = None
        self._last_submit = 0.0
        self.heartbeat_error: str | None = None
        self._hb_stop = threading.Event()
        self._hb_thread: threading.Thread | None = None
        try:
            self._start(si_root, si_python, teleop_config, start_timeout, portal)
        except BaseException:
            self.close()  # never leave a half-started worker holding an arm / CAN bus
            raise

    def _start(
        self,
        si_root: Path,
        si_python: Path,
        teleop_config: Path,
        start_timeout: float,
        portal,
    ) -> None:
        log_dir = self.log_dir
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = str(si_root) + os.pathsep + env.get("PYTHONPATH", "")
        for side in SIDES:
            out = None
            if log_dir:
                try:
                    out = open(
                        log_dir / f"worker_{side}.log", "w"
                    )  # noqa: SIM115 - owned by the child
                except OSError as exc:
                    print(f"warning: cannot create worker log: {exc}", file=sys.stderr)
            self.procs[side] = subprocess.Popen(
                [
                    str(si_python),
                    "-m",
                    "devices.robots.i2rt_teleop.worker",
                    "--config",
                    str(teleop_config),
                    "--side",
                    side,
                    "--role",
                    "follower",
                ],
                cwd=si_root,
                env=env,
                start_new_session=True,
                stdout=out,
                stderr=subprocess.STDOUT if out else None,
                preexec_fn=_die_with_parent,  # no orphan workers even if this program is killed
            )
        # portal.Client blocks forever inside call() while the server is down, so first wait for the
        # TCP port with a plain socket while watching the worker processes (a dead worker = clear error).
        deadline = time.monotonic() + start_timeout
        pending = set(SIDES)
        while pending:
            self._check_procs()
            for s in list(pending):
                try:
                    with socket.create_connection(
                        ("127.0.0.1", self.ports[s]), timeout=0.2
                    ):
                        pending.discard(s)
                except OSError:
                    pass
            if pending and time.monotonic() > deadline:
                raise TimeoutError(
                    f"follower workers not ready after {start_timeout:.0f}s: {sorted(pending)} "
                    f"(left: {self._worker_error('left')}; right: {self._worker_error('right')})"
                )
            if pending:
                time.sleep(0.2)
        self.clients = {s: portal.Client(f"127.0.0.1:{self.ports[s]}") for s in SIDES}
        for s in SIDES:
            self._call(s, "ping", timeout=2.0)

    # ------------------------------------------------------------------ rpc
    def _check_procs(self) -> None:
        for s, p in self.procs.items():
            if p.poll() is not None:
                raise RuntimeError(
                    f"{s} follower worker exited with code {p.returncode}: {self._worker_error(s)}"
                )

    def _worker_error(self, side: str) -> str:
        """Last error line of a worker log (ANSI colours stripped)."""
        if self.log_dir is None:
            return "see worker output"
        path = self.log_dir / f"worker_{side}.log"
        try:
            lines = re.sub(
                r"\x1b\[[0-9;]*m", "", path.read_text(errors="replace")
            ).splitlines()
        except OSError:
            return f"see {path}"
        errors = [
            ln.strip() for ln in lines if "Error:" in ln or "failed" in ln.lower()
        ]
        return (errors[-1] if errors else "no error line") + f"  (full log: {path})"

    def _call(self, side: str, method: str, *args, timeout: float | None = None):
        with self._rpc_lock:
            return getattr(self.clients[side], method)(*args).result(
                timeout=timeout or self.timeout
            )

    def _heartbeat_loop(self) -> None:
        while not self._hb_stop.wait(0.01):
            with self._rpc_lock:
                if self.session is None or self._last_target is None:
                    continue
                if time.monotonic() - self._last_submit < 0.03:
                    continue
                try:
                    self._send(self._last_target, self._phase)
                except (
                    Exception
                ) as exc:  # noqa: BLE001 - surfaced through get_state / heartbeat_error
                    self.heartbeat_error = str(exc)
                    self._last_target = None  # stop hammering a faulted worker

    def _start_heartbeat(self) -> None:
        if self._hb_thread is None:
            self._hb_thread = threading.Thread(
                target=self._heartbeat_loop, daemon=True, name="arm-heartbeat"
            )
            self._hb_thread.start()

    def get_state(self) -> np.ndarray:
        """(14,) follower state [left 6 joints, left grip, right 6 joints, right grip]."""
        with self._rpc_lock:
            futures = {s: self.clients[s].get_sample() for s in SIDES}
            results = {s: futures[s].result(timeout=self.timeout) for s in SIDES}
        parts = []
        for s in SIDES:
            r = results[s]
            if not isinstance(r, dict):
                raise RuntimeError(f"{s} follower returned an invalid sample: {r!r}")
            if r.get("phase") == "FAULT":
                raise RuntimeError(f"{s} follower FAULT: {r.get('fault_reason')}")
            parts.append(np.asarray(r["state"], dtype=np.float64).reshape(7))
        return np.concatenate(parts)

    def begin(self) -> None:
        with self._rpc_lock:
            self.session = uuid.uuid4().hex
            self.seq = -1
            self._phase = ALIGNING
            self._last_target = None
            self.heartbeat_error = None
            for s in SIDES:
                self._call(s, "begin_control_session", self.session, timeout=1.0)
        self._start_heartbeat()

    def activate(self) -> None:
        with self._rpc_lock:
            for s in SIDES:
                self._call(s, "set_phase", self.session, ACTIVE, timeout=1.0)
            self._phase = ACTIVE

    def recover(self) -> np.ndarray:
        """After a worker FAULT: reset, open a new session and hold the current pose. Returns the state."""
        with self._rpc_lock:
            self.session = None
            self._last_target = None
            self.reset_fault()
            state = self.get_state()
            self.begin()
            self.submit(state, ALIGNING)
            self.activate()
            return state

    def submit(self, target14: np.ndarray, phase: str) -> None:
        with self._rpc_lock:
            self._send(target14, phase)
            self._phase = phase
            self._last_target = np.asarray(target14, dtype=np.float64).copy()

    def _send(self, target14: np.ndarray, phase: str) -> None:
        if self.session is None:
            raise RuntimeError("no control session")
        self._last_submit = time.monotonic()
        self.seq += 1
        issued = time.monotonic_ns()
        futures = [
            self.clients[s].submit_control_frame(
                {
                    "session_id": self.session,
                    "sequence": self.seq,
                    "issued_at_ns": issued,
                    "phase": phase,
                    "joint_position": np.asarray(
                        target14[i * 7 : (i + 1) * 7], dtype=np.float64
                    ),
                }
            )
            for i, s in enumerate(SIDES)
        ]
        for f in futures:
            f.result(timeout=self.timeout)

    def end(self) -> None:
        """Leave control: workers hold the current pose (safe hold, motors stay powered)."""
        with self._rpc_lock:
            self._last_target = None
        if self.session is None:
            return
        for s in SIDES:
            try:
                self._call(s, "end_control_session", self.session, timeout=1.0)
            except Exception as exc:  # noqa: BLE001
                print(f"warning: end session {s}: {exc}", file=sys.stderr)
        self.session = None

    def reset_fault(self) -> None:
        for s in SIDES:
            self._call(s, "reset_fault", timeout=1.0)

    def close(self) -> None:
        """Stops the workers => SDK sets ZERO TORQUE (arms go limp). Park first!"""
        self.end()
        self._hb_stop.set()
        for p in self.procs.values():
            if p.poll() is None:
                p.terminate()
        for p in self.procs.values():
            try:
                p.wait(timeout=8)
            except subprocess.TimeoutExpired:
                p.kill()
        for c in self.clients.values():
            # portal's close can hang if the server is already gone -> bounded, in a daemon thread
            t = threading.Thread(target=c.close, kwargs={"timeout": 1.0}, daemon=True)
            t.start()
            t.join(timeout=2.0)
        self.clients = {}


class SimArms:
    """Dry-run stand-in: first-order lag towards the commanded target (~130 ms like the real follower)."""

    def __init__(self, initial: np.ndarray):
        self.state = np.asarray(initial, dtype=np.float64).copy()
        self.target = self.state.copy()
        self.t = time.monotonic()
        self.session: str | None = None

    def get_state(self) -> np.ndarray:
        now = time.monotonic()
        a = 1 - np.exp(-(now - self.t) / 0.13)
        self.state += (self.target - self.state) * a
        self.t = now
        return self.state.copy()

    def begin(self) -> None:
        self.session = "sim"

    def activate(self) -> None:
        pass

    def submit(self, target14: np.ndarray, phase: str) -> None:
        self.get_state()
        self.target = np.asarray(target14, dtype=np.float64).copy()

    def end(self) -> None:
        self.session = None

    def reset_fault(self) -> None:
        pass

    def recover(self) -> np.ndarray:
        self.begin()
        return self.get_state()

    @property
    def heartbeat_error(self) -> None:
        return None

    def close(self) -> None:
        pass
