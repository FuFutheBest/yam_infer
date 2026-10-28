"""Bring both follower arms to the rest pose, whatever state the workers are in, then power off.

    python -m yam_infer.tools.home_arms                 # configs/rest_pose.yaml, ask before moving
    python -m yam_infer.tools.home_arms --target zero   # all joints 0, grippers open
    python -m yam_infer.tools.home_arms --keep-on       # stay powered, holding the pose

Per arm:

| state of its worker                         | what happens                                        |
|---------------------------------------------|-----------------------------------------------------|
| our yam_infer.run still running             | SIGINT first -> it parks itself and exits           |
| a worker is serving (ours or SuperInference) | RPC: reset FAULT -> session -> slow move home       |
| worker busy (another program's session),    | free_hw lists the blockers and asks to stop them,   |
| hung, or no worker at all                   | then OUR worker is started (holds where the arm is) |
|                                             | and moves it home                                   |

Afterwards our workers are stopped (motors off AT the rest pose); a SuperInference worker
that we homed through RPC is left running (it keeps holding the pose).
"""

from __future__ import annotations

import argparse
import os
import signal
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from yam_infer.hw.arms import DEFAULT_HW_CONFIG, SIDES, load_ports, port_open, spawn_worker
from yam_infer.tools import free_hw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REST_POSE = PROJECT_ROOT / "configs" / "rest_pose.yaml"
ZERO_POSE = np.array([0, 0, 0, 0, 0, 0, 1.0] * 2, dtype=np.float64)


class SessionBusy(RuntimeError):
    pass


def _call(client, method, *args, timeout=1.0):
    return getattr(client, method)(*args).result(timeout=timeout)


def home_side(port: int, side: str, target7: np.ndarray, speed: float) -> None:
    """Drive one worker (our RPC contract) slowly to target7, then end the session (it holds)."""
    import portal

    client = portal.Client(f"127.0.0.1:{port}")
    stop = threading.Event()
    try:
        sample = _call(client, "get_sample", timeout=2.0)
        if sample.get("phase") == "FAULT":
            print(f"  {side}: FAULT（{sample.get('fault_reason')}）-> 复位")
            _call(client, "reset_fault")
            time.sleep(0.1)
        start = np.asarray(_call(client, "get_sample")["state"], dtype=np.float64)
        # NEVER send a call the worker would reject: a portal server re-raises method errors and
        # the worker process dies (arm drops). get_status() cannot fail.
        status = _call(client, "get_status")
        if status.get("session_id"):
            raise SessionBusy(f"{side} 臂正被另一个程序控制（会话 {status['session_id'][:8]}…）")
        session = uuid.uuid4().hex
        _call(client, "begin_control_session", session)
        seq = [-1]
        lock = threading.Lock()
        current = [start.copy()]
        phase = ["ALIGNING"]

        def send():
            with lock:
                seq[0] += 1
                frame = {
                    "session_id": session,
                    "sequence": seq[0],
                    "issued_at_ns": time.monotonic_ns(),
                    "phase": phase[0],
                    "joint_position": current[0],
                }
                _call(client, "submit_control_frame", frame, timeout=0.2)

        def heartbeat():  # keep the 150 ms command lease alive
            while not stop.wait(0.02):
                try:
                    send()
                except Exception:  # noqa: BLE001 - surfaced by the main loop's own send()
                    return

        send()
        threading.Thread(target=heartbeat, daemon=True).start()
        deadline = time.monotonic() + 1.0  # ACTIVE only after the worker applied a frame
        while int(_call(client, "get_sample").get("last_applied_sequence", -1)) < 0:
            if time.monotonic() > deadline:
                raise TimeoutError(f"{side}: worker applied no alignment frame")
            time.sleep(0.005)
        _call(client, "set_phase", session, "ACTIVE")
        phase[0] = "ACTIVE"
        dist = float(np.abs(target7[:6] - start[:6]).max())
        duration = max(2.0, 1.5 * dist / speed)
        print(f"  {side}: 归位中（最大 {dist:.2f} rad，{duration:.1f} s）")
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
        print(f"  {side}: 已到位")
    finally:
        stop.set()
        t = threading.Thread(target=client.close, kwargs={"timeout": 1.0}, daemon=True)
        t.start()
        t.join(2.0)


def _listener_pid(port: int, hw_config: Path) -> int | None:
    res = free_hw.scan(hw_config, ports_only=True)
    for pid, reasons in res.holders.items():
        if any(r.startswith(f"RPC 端口 {port}") for r in reasons):
            return pid
    return None


def _wait_port(port: int, proc, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while not port_open(port):
        if proc.poll() is not None:
            raise RuntimeError(
                f"worker 退出（code {proc.returncode}），见 runs_infer/home_worker_*.log"
            )
        if time.monotonic() > deadline:
            raise TimeoutError(f"worker 在 {timeout:.0f}s 内没有就绪")
        time.sleep(0.2)


def _stop_our_runs() -> None:
    for pid in _our_runs():
        print(f"推理程序 pid {pid} 仍在运行 -> SIGINT（它会自己归位并退出，最多等 45 s）")
        os.kill(pid, signal.SIGINT)
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline and Path(f"/proc/{pid}").exists():
            time.sleep(0.3)


def _our_runs() -> list[int]:
    """Running `python -m yam_infer.run` processes (this project's inference program)."""
    try:
        names = os.listdir("/proc")
    except OSError:
        return []
    return [
        int(n)
        for n in names
        if n.isdigit() and int(n) != os.getpid() and free_hw.is_ours(int(n), "yam_infer.run")
    ]


def load_rest_pose(path: Path) -> np.ndarray:
    """configs/rest_pose.yaml -> (14,) [left 6 joints, left grip, right 6 joints, right grip]."""
    import yaml

    try:
        cfg = yaml.safe_load(Path(path).read_text()) or {}
        pose = np.asarray([*cfg["left"], *cfg["right"]], dtype=np.float64)
    except (OSError, yaml.YAMLError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"读不了休息姿态 {path}: {exc}（或用 --target zero）") from exc
    if pose.shape != (14,) or not np.all(np.isfinite(pose)):
        raise SystemExit(f"{path}: left/right 各需 7 个数（6 关节 + 夹爪）")
    return pose


def load_target(kind: str, rest_file: Path) -> np.ndarray:
    if kind == "zero":
        print("目标：关节全 0、夹爪张开")
        return ZERO_POSE.copy()
    print(f"目标：休息姿态（{rest_file}）")
    return load_rest_pose(rest_file)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--target", choices=["rest", "zero"], default="rest")
    p.add_argument("--rest-pose", type=Path, default=DEFAULT_REST_POSE, help="--target rest pose")
    p.add_argument(
        "--record-dir", type=Path, default=PROJECT_ROOT / "runs_infer", help="worker logs"
    )
    p.add_argument("--hw-config", type=Path, default=DEFAULT_HW_CONFIG)
    p.add_argument("--speed", type=float, default=0.4, help="peak joint speed rad/s")
    p.add_argument("--keep-on", action="store_true", help="leave our workers running (holding)")
    p.add_argument("--yes", "-y", action="store_true", help="do not ask before moving")
    p.add_argument("--fake", action="store_true", help="start simulated workers (tests)")
    args = p.parse_args()

    target = load_target(args.target, args.rest_pose)
    print("  左:", np.round(target[:7], 2), "\n  右:", np.round(target[7:], 2))
    if not args.yes and free_hw.ask("确认移动？手臂会以低速运动，请确保周围无障碍 [y/N]: ") != "y":
        return
    ports = load_ports(args.hw_config)
    log_dir = args.record_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    _stop_our_runs()
    ours: dict[str, object] = {}  # side -> Popen of workers we started
    done: dict[str, int | None] = {}  # side -> pid of the worker that homed it

    def try_home(side: str) -> None:
        i, port = SIDES.index(side), ports[side]
        try:
            home_side(port, side, target[i * 7 : (i + 1) * 7], args.speed)
            done[side] = _listener_pid(port, args.hw_config)
        except SessionBusy as exc:
            print(f"  {exc}")
        except Exception as exc:  # noqa: BLE001 - hung / dead worker -> take over below
            print(f"  {side}: 通过现有 worker 归位失败（{type(exc).__name__}: {exc}）")

    try:
        print("\n[1] 尝试通过正在运行的 worker 归位")
        for side in SIDES:
            if port_open(ports[side]):
                try_home(side)
            else:
                print(f"  {side}: 没有 worker")
        if len(done) < len(SIDES):
            print("\n[2] 释放硬件，由本项目接管剩下的手臂")
            keep = {pid for pid in done.values() if pid}
            if free_hw.free(args.hw_config, ports_only=args.fake, ignore_pids=keep):
                for side in SIDES:
                    pid = done.get(side)
                    if side in done and pid and _listener_pid(ports[side], args.hw_config) == pid:
                        continue  # its worker survived -> still homed
                    done.pop(side, None)  # e.g. stopping the WebUI dropped this arm too
                    if not port_open(ports[side]):
                        print(f"  {side}: 启动本项目 worker（先保持手臂当前位置）")
                        log = log_dir / f"home_worker_{side}.log"
                        ours[side] = spawn_worker(side, args.hw_config, log, args.fake)
                        _wait_port(ports[side], ours[side])
                    try_home(side)
            else:
                print("硬件仍被占用，无法归位剩下的手臂。")
        missing = set(SIDES) - set(done)
        print("\n归位完成" if not missing else f"\n未能归位: {sorted(missing)}")
    finally:
        _finish(ours, ports, args.keep_on, args.hw_config)


def _finish(ours: dict, ports: dict[str, int], keep_on: bool, hw_config: Path) -> None:
    # also stop our pre-existing workers (e.g. left behind by a crashed run)
    for side in SIDES:
        pid = _listener_pid(ports[side], hw_config) if side not in ours else None
        if pid and free_hw.is_ours(pid, "yam_infer.hw.follower.worker"):
            ours[side] = pid
        elif pid:
            print(f"  {side}: 由其他程序的 worker（pid {pid}）保持位置，未停止它")
    if keep_on:
        if ours:
            print("--keep-on：本项目 worker 保持通电，按 Ctrl+C 结束（结束即断电）")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
    for side, proc in ours.items():
        pid = proc if isinstance(proc, int) else proc.pid
        print(f"  {side}: 停止本项目 worker pid {pid}（电机断力）")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        if not _wait_exit(proc, 8.0):
            print(f"  {side}: 8 s 内未退出 -> SIGKILL")
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def _wait_exit(proc, timeout: float) -> bool:
    """proc: our Popen child (reaped via poll) or a foreign pid."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if isinstance(proc, int):
            if not free_hw._alive(proc):
                return True
        elif proc.poll() is not None:
            return True
        time.sleep(0.2)
    return False


if __name__ == "__main__":
    main()
