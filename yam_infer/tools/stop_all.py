"""Bring every running I2RT follower worker home, then stop all related processes.

    python -m yam_infer.tools.stop_all [--record-dir runs_infer] [--no-home]

For each follower worker still alive (RPC ports from the teleop config): reset a FAULT, take control,
move slowly to the rest pose saved by the last yam_infer.run (runs_infer/rest_pose.json, else all
joints 0 / gripper open), then terminate it (motors off).  Finally stops leftover yam_infer.run
processes.  A side whose worker is dead is skipped (nothing to move).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import threading
import time
import uuid
from pathlib import Path

import numpy as np

SIDES = ("left", "right")


WORKER = r"python[0-9.]* -m devices\.robots\.i2rt_teleop\.worker"
# only inference processes started from THIS project's venv (another project on the robot also has a
# "yam_inference" package with its own run/worker modules - never touch those)
PROJECT_VENV = Path(__file__).resolve().parents[2] / ".venv"
INFER = re.escape(str(PROJECT_VENV)) + r"/bin/python[0-9.]* -m yam_infer\.run"


def _pids(pattern: str) -> list[int]:
    """Only real python module processes (never shells whose command line merely mentions the name)."""
    out = subprocess.run(
        ["pgrep", "-f", pattern], capture_output=True, text=True
    ).stdout.split()
    return [int(p) for p in out if int(p) not in (os.getpid(), os.getppid())]


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.3):
            return True
    except OSError:
        return False


def _call(client, method, *args, timeout=1.0):
    return getattr(client, method)(*args).result(timeout=timeout)


def home_side(port: int, side: str, target7: np.ndarray, speed: float) -> None:
    import portal

    client = portal.Client(f"127.0.0.1:{port}")
    try:
        sample = _call(client, "get_sample")
        if sample.get("phase") == "FAULT":
            print(f"  {side}: FAULT ({sample.get('fault_reason')}) -> reset")
            _call(client, "reset_fault")
            time.sleep(0.1)
        start = np.asarray(_call(client, "get_sample")["state"], dtype=np.float64)
        session = uuid.uuid4().hex
        _call(client, "begin_control_session", session)
        seq = [-1]
        lock = threading.Lock()
        current = [start.copy()]
        phase = ["ALIGNING"]
        stop = threading.Event()

        def send():
            with lock:
                seq[0] += 1
                _call(
                    client,
                    "submit_control_frame",
                    {
                        "session_id": session,
                        "sequence": seq[0],
                        "issued_at_ns": time.monotonic_ns(),
                        "phase": phase[0],
                        "joint_position": current[0],
                    },
                    timeout=0.2,
                )

        def heartbeat():  # keep the 150 ms command lease alive
            while not stop.wait(0.02):
                try:
                    send()
                except Exception:  # noqa: BLE001
                    return

        send()
        threading.Thread(target=heartbeat, daemon=True).start()
        time.sleep(0.1)
        _call(client, "set_phase", session, "ACTIVE")
        phase[0] = "ACTIVE"
        dist = float(np.abs(target7[:6] - start[:6]).max())
        duration = max(2.0, 1.5 * dist / speed)
        print(f"  {side}: moving home ({dist:.2f} rad, {duration:.1f}s)")
        n = int(duration * 100)
        for i in range(1, n + 1):
            a = i / n
            a = a * a * (3 - 2 * a)
            with lock:
                current[0] = start + (target7 - start) * a
            send()
            time.sleep(0.01)
        time.sleep(0.5)
        stop.set()
        _call(client, "end_control_session", session)
    finally:
        t = threading.Thread(target=client.close, kwargs={"timeout": 1.0}, daemon=True)
        t.start()
        t.join(2.0)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--record-dir", default="runs_infer")
    p.add_argument("--teleop-config", default="")
    p.add_argument(
        "--no-home", action="store_true", help="just stop everything (arms drop!)"
    )
    p.add_argument("--speed", type=float, default=0.4, help="rad/s peak")
    args = p.parse_args()

    for pid in _pids(INFER):
        print(f"stopping yam_infer.run pid {pid}")
        os.kill(pid, signal.SIGINT)
    time.sleep(1.0)

    ports = {"left": 1234, "right": 1235}
    if args.teleop_config:
        import yaml

        cfg = yaml.safe_load(Path(args.teleop_config).read_text())
        ports = {s: int(cfg["arms"][s]["follower"]["rpc_port"]) for s in SIDES}
    rest_file = Path(args.record_dir) / "rest_pose.json"
    try:
        rest = np.asarray(json.loads(rest_file.read_text()), dtype=np.float64)
        print(f"rest pose from {rest_file}")
    except (OSError, ValueError):
        rest = np.array([0, 0, 0, 0, 0, 0, 1.0] * 2, dtype=np.float64)
        print("no saved rest pose -> joints 0, gripper open")

    if not args.no_home:
        for i, side in enumerate(SIDES):
            if not _port_open(ports[side]):
                print(f"  {side}: no worker running")
                continue
            try:
                home_side(ports[side], side, rest[i * 7 : (i + 1) * 7], args.speed)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"  {side}: could not move home ({exc}) -> it will be released where it is"
                )

    pids = _pids(WORKER)
    for pid in pids:
        print(f"stopping worker pid {pid} (motors off)")
        os.kill(pid, signal.SIGTERM)
    deadline = time.time() + 8
    while time.time() < deadline and any(_pids(WORKER)):
        time.sleep(0.2)
    for pid in _pids(WORKER):
        os.kill(pid, signal.SIGKILL)
    print(
        "all stopped"
        if not (_pids(WORKER) + _pids(INFER))
        else "WARNING: still running"
    )


if __name__ == "__main__":
    main()
