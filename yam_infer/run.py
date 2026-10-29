"""Run a trained policy on the I2RT YAM follower arms.

    python -m yam_infer.run --ckpt model.pt --task put_blue_block_in_blue_cup

Dry run without any hardware (simulated arm + frames from a recorded episode):
    python -m yam_infer.run --ckpt model.pt --task put_banana_in_bowl --dry-run EPISODE_DIR
    ... --dry-run EPISODE_DIR --fake-arms   # same, but through the real worker processes + RPC

Keys (type + Enter):  Enter = start / pause-resume   r = new episode   q = park & quit
Threads:
    policy thread  ~30 Hz : cameras + follower state -> model -> chunk -> scheduler
    control loop   100 Hz : scheduler.target_at(now) -> safety filter -> submit to workers -> recorder
"""

from __future__ import annotations

import argparse
import queue
import signal
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

from yam_infer.control.scheduler import ActionScheduler, SafetyFilter
from yam_infer.control.state_history import StateHistory
from yam_infer.data.recorder import RunRecorder
from yam_infer.hw.arms import DEFAULT_HW_CONFIG, FollowerArms

CONTROL_HZ = 100.0
DEFAULT_RUN_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "run.yaml"


def smooth_move(
    arms,
    start: np.ndarray,
    goal: np.ndarray,
    phase: str,
    max_speed: float = 0.5,
    min_time: float = 2.0,
) -> None:
    """Cubic smoothstep from start to goal (joints limited to max_speed rad/s peak)."""
    dist = float(np.max(np.abs(goal - start)[np.r_[0:6, 7:13]]))
    duration = max(min_time, 1.5 * dist / max_speed)
    n = int(duration * CONTROL_HZ)
    print(f"  moving {dist:.2f} rad over {duration:.1f}s ({phase})")
    t_next = time.monotonic()
    for i in range(1, n + 1):
        a = i / n
        a = a * a * (3 - 2 * a)
        arms.submit(start + (goal - start) * a, phase)
        t_next += 1.0 / CONTROL_HZ
        time.sleep(max(0.0, t_next - time.monotonic()))


class KeyReader:
    def __init__(self):
        self.q: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        for line in sys.stdin:
            self.q.put(line.strip().lower())

    def get(self, block: bool = False, timeout: float | None = None) -> str | None:
        try:
            return self.q.get(block=block, timeout=timeout)
        except queue.Empty:
            return None


def run_episode(
    args, policy, cams, arms, keys: KeyReader, ep_index: int, run_meta: dict | None = None
) -> str:
    """One policy episode. Returns 'next' or 'quit'."""
    scheduler = ActionScheduler(
        policy.fps,
        policy.chunk_size,
        args.ensemble_m,
        mode=args.exec_mode,
        horizon=args.exec_horizon,
        blend_s=args.blend_s,
    )
    safety = SafetyFilter(
        policy.a_min,
        policy.a_max,
        args.margin,
        args.max_joint_speed,
    )
    stop = threading.Event()
    paused = threading.Event()
    shared: dict = {"error": None}
    stats = {"policy_steps": 0}
    # follower states of the last ~1 s (written by the control loop) -> the joints are taken AT the
    # image time, as in training (see control/state_history.py)
    history = StateHistory(seconds=1.0, rate_hz=CONTROL_HZ)

    def policy_loop(recorder: RunRecorder):
        period = 1.0 / policy.fps
        t_next = time.monotonic()
        while not stop.is_set():
            try:
                if not paused.is_set() and scheduler.needs_new_chunk(time.monotonic()):
                    frames, cam = cams.grab()  # cam[name] = (arrival time, frame counter)
                    t_img = float(np.mean([cam[c][0] for c in policy.cameras]))  # observation time
                    qpos = history.at(t_img)  # joints at that moment (interpolated)
                    t_infer = time.monotonic()
                    chunk, _images = policy.predict(frames, qpos, args.task)
                    t_done = time.monotonic()
                    latency = (t_done - t_infer) * 1000
                    # the chunk starts at the observation time (training: actions a(t)...a(t+49)
                    # for the sample at t), so image age + inference latency are skipped over
                    scheduler.add(t_img, chunk)
                    recorder.log_policy(
                        t_img,
                        qpos,
                        chunk,
                        cam_t=[cam[c][0] for c in policy.cameras],
                        cam_seq=[cam[c][1] for c in policy.cameras],
                        t_infer=t_infer,
                        t_done=t_done,
                        raw_out=policy.last_raw,
                    )
                    stats["policy_steps"] += 1
                    if (
                        stats["policy_steps"]
                        % (int(policy.fps * 2) if args.exec_mode == "ensemble" else 3)
                        == 0
                    ):
                        print(
                            f"  policy step {stats['policy_steps']}  latency {latency:.0f} ms",
                            flush=True,
                        )
            except Exception as exc:  # noqa: BLE001
                shared["error"] = f"policy thread: {exc}"
                traceback.print_exc()
                stop.set()
                return
            t_next += period
            delay = t_next - time.monotonic()
            if delay < -period:  # model slower than the camera: don't try to catch up
                t_next = time.monotonic()
            time.sleep(max(0.0, delay))

    print(f"\n=== episode {ep_index}: press Enter to START the policy, q to quit ===")
    while True:
        k = keys.get(block=True)
        if k == "q":
            return "quit"
        if k == "":
            break

    recorder = RunRecorder(
        Path(args.record_dir),
        args.task,
        {
            **(run_meta or {}),
            "ckpt": str(args.ckpt),
            "episode_index": ep_index,
            "policy": policy.info(),
            "settings": {k: v for k, v in vars(args).items() if k != "ckpt"},
        },
        fps=policy.fps,
    )
    cams.set_sink(recorder.camera_sink)  # every camera frame from now on -> data/*.mcap
    hold = arms.get_state()
    history.add(time.monotonic(), hold)
    safety.reset(hold)
    thread = threading.Thread(target=policy_loop, args=(recorder,), daemon=True)
    thread.start()
    t_start = time.monotonic()
    end_reason = "unknown"
    t_next = time.monotonic()
    last_target = hold.copy()
    recorder.event("start", t_start, max_seconds=args.max_seconds)
    print("  RUNNING  (Enter = pause/resume, r = end episode, q = quit)")
    try:
        while not stop.is_set():
            now = time.monotonic()
            qpos = arms.get_state()
            t_read = time.monotonic()  # stamped after the read, like SI's recorder
            history.add(t_read, qpos)
            raw = None if paused.is_set() else scheduler.target_at(now)
            target = last_target if raw is None else raw
            if args.freeze_left:
                target = target.copy()
                target[0:7] = hold[0:7]
            safe = safety(target, 1.0 / CONTROL_HZ)
            arms.submit(safe, "ACTIVE")
            last_target = safe
            recorder.log_command(t_read, raw, safe, qpos, getattr(arms, "last_sample_t", None))

            k = keys.get()
            if k == "":
                if paused.is_set():
                    scheduler.clear()
                    paused.clear()
                    recorder.event("resume")
                    print("  resumed")
                else:
                    paused.set()
                    recorder.event("pause")
                    print("  PAUSED (holding pose) - Enter to resume")
            elif k in ("r", "q"):
                end_reason = "operator next" if k == "r" else "operator quit"
                break
            if now - t_start > args.max_seconds:
                end_reason = "max_seconds"
                break
            t_next += 1.0 / CONTROL_HZ
            delay = t_next - time.monotonic()
            if delay < -0.05:
                t_next = time.monotonic()
            time.sleep(max(0.0, delay))
        else:
            end_reason = shared["error"] or "stopped"
    except KeyboardInterrupt:
        end_reason = "ctrl-c"
    except Exception as exc:  # noqa: BLE001
        end_reason = f"control loop: {exc}"
        traceback.print_exc()
    finally:
        stop.set()
        thread.join(timeout=2)
        cams.set_sink(None)
        recorder.event("end", reason=end_reason)
        log_dir = getattr(arms, "log_dir", None)
        out = recorder.close(
            worker_logs=sorted(Path(log_dir).glob("worker_*.log")) if log_dir else None,
            end_reason=end_reason,
            duration_s=time.monotonic() - t_start,
            policy_steps=stats["policy_steps"],
            safety_clamped=safety.clamped,
            safety_rate_limited=safety.rate_limited,
        )
        print(f"  episode ended: {end_reason}\n  recorded -> {out}")
        if out is not None and out.parent.name == "quarantine":
            print("  !! recording incomplete -> quarantined, see quarantine_reason.json")
    if end_reason.startswith(("control loop", "policy thread")) or end_reason in (
        "ctrl-c",
        "operator quit",
    ):
        return "quit"
    return "next"


def park_and_release(arms, start_pose: np.ndarray, rest_pose: np.ndarray | None, park: str) -> None:
    """Return home, then cut torque.

    park="home"  : current -> demo start pose -> REST pose (where the arms were lying when the program
                   started, i.e. a pose that is stable without torque) -> zero torque. Nothing drops.
    park="start" : current -> demo start pose -> zero torque (arms fall from there!).
    A FAULTed worker is reset and re-taken before moving.
    """
    if park != "none":
        for attempt in range(2):
            try:
                if attempt == 1 or arms.session is None:
                    print("  worker not in control -> reset fault and re-open a session")
                    arms.recover()
                print("returning to the demo start pose ...")
                smooth_move(arms, arms.get_state(), start_pose, "ACTIVE")
                if park == "home" and rest_pose is not None:
                    print("returning to the rest pose (same path as at startup) ...")
                    smooth_move(arms, arms.get_state(), rest_pose, "ACTIVE")
                    time.sleep(0.5)  # settle before releasing
                break
            except Exception as exc:  # noqa: BLE001
                print(f"park attempt {attempt + 1} failed: {exc}")
    print("stopping workers -> motors off" + ("" if park == "home" else "  (!!! arms may drop)"))
    arms.close()


def install_stop_signals() -> None:
    """SIGINT (Ctrl+C) and SIGTERM -> KeyboardInterrupt once -> the normal park-and-exit path.

    Installed explicitly: a process started in the background by a non-interactive shell
    inherits SIGINT=ignored and would otherwise never stop.  Repeated signals while parking
    are ignored so the slow move home is not cut short (that would drop the arms).
    """
    count = [0]

    def handler(signum: int, _frame) -> None:
        count[0] += 1
        if count[0] == 1:
            raise KeyboardInterrupt
        print(f"  ({signal.Signals(signum).name} ignored: already stopping, parking the arms)")

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def build_run_meta(args) -> dict:
    """Facts needed to reproduce a recording: model checksum, code version, config files."""
    import hashlib
    import platform
    import subprocess

    meta: dict = {"host": platform.node()}
    try:
        h = hashlib.sha256()
        with open(args.ckpt, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        meta["ckpt_sha256"] = h.hexdigest()
    except OSError as exc:
        meta["ckpt_sha256"] = f"unavailable: {exc}"
    root = Path(__file__).resolve().parents[1]
    try:
        out = subprocess.run(["git", "-C", str(root), "describe", "--always", "--dirty"],
                             capture_output=True, text=True, timeout=5)
        meta["code_version"] = out.stdout.strip() or "unknown (no git checkout)"
    except (OSError, subprocess.TimeoutExpired):
        meta["code_version"] = "unknown"
    configs = {}
    for path in (args.hw_config, args.run_config):
        try:
            configs[str(path)] = Path(path).read_text()
        except OSError as exc:
            configs[str(path)] = f"unreadable: {exc}"
    meta["configs"] = configs
    return meta


RUN_CONFIG_SCHEMA = {  # section -> key -> (type, check, description of the valid range)
    "execution": {
        "mode": (str, lambda v: v in ("chunk", "ensemble"), "chunk or ensemble"),
        "horizon": (int, lambda v: 1 <= v <= 50, "1..50 steps"),
        "blend_s": (float, lambda v: 0 <= v <= 2, "0..2 s"),
        "ensemble_m": (float, lambda v: -5 <= v <= 5, "-5..5"),
    },
    "episode": {
        "max_seconds": (float, lambda v: v > 0, "> 0 s"),
    },
}


def load_run_config(run_config: Path) -> dict[str, dict]:
    """configs/run.yaml -> {section: {key: value}}; every key required, unknown keys rejected."""
    import yaml

    try:
        cfg = yaml.safe_load(Path(run_config).read_text()) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read {run_config}: {exc}") from exc
    if not isinstance(cfg, dict):
        raise ValueError(f"{run_config}: top level must be a mapping")
    unknown = sorted(set(cfg) - set(RUN_CONFIG_SCHEMA))
    if unknown:
        raise ValueError(f"{run_config}: unknown section(s) {unknown}")
    out: dict[str, dict] = {}
    for section, keys in RUN_CONFIG_SCHEMA.items():
        raw = cfg.get(section)
        if not isinstance(raw, dict):
            raise ValueError(f"{run_config}: missing section '{section}'")
        extra = sorted(set(raw) - set(keys))
        if extra:
            raise ValueError(f"{run_config}: unknown key(s) {section}.{extra}")
        out[section] = {}
        for key, (typ, ok, rng) in keys.items():
            if key not in raw:
                raise ValueError(f"{run_config}: missing {section}.{key}")
            value = raw[key]
            if typ is float and isinstance(value, int) and not isinstance(value, bool):
                value = float(value)
            if not isinstance(value, typ) or isinstance(value, bool) or not ok(value):
                raise ValueError(f"{run_config}: {section}.{key} = {raw[key]!r}, expected {rng}")
            out[section][key] = value
    return out


def main() -> None:
    install_stop_signals()
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--ckpt", required=True)
    p.add_argument(
        "--task",
        required=True,
        choices=["put_banana_in_bowl", "put_blue_block_in_blue_cup"],
    )
    p.add_argument(
        "--hw-config",
        type=Path,
        default=DEFAULT_HW_CONFIG,
        help="arms (CAN, RPC ports, watchdog) + camera serials",
    )
    p.add_argument("--left-serial", help="override cameras.left_wrist of the hw config")
    p.add_argument("--right-serial", help="override cameras.right_wrist of the hw config")
    p.add_argument(
        "--remote",
        default="",
        help="host:port of `python -m yam_infer.policy.server` (GPU elsewhere)",
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--no-jpeg-roundtrip", action="store_true")
    p.add_argument(
        "--goto-start",
        choices=["task_mean", "none"],
        default="task_mean",
        help="move to the mean start pose of the demos before each episode",
    )
    p.add_argument(
        "--park",
        choices=["home", "start", "none"],
        default="home",
        help="home: back to the rest pose the arms had at program start, then motors off (default)",
    )
    p.add_argument(
        "--exec-mode",
        choices=["chunk", "ensemble"],
        default=None,
        help="default: execution.mode of --run-config. chunk: execute --exec-horizon steps of each plan, then re-plan; "
        "ensemble: re-plan every step + ACT temporal ensembling",
    )
    p.add_argument(
        "--exec-horizon",
        type=int,
        default=None,
        help="chunk mode: steps (30/s) per plan; default: execution.horizon of --run-config",
    )
    p.add_argument(
        "--blend-s",
        type=float,
        default=None,
        help="chunk mode: cross-fade between plans [s]; default: execution.blend_s of --run-config",
    )
    p.add_argument(
        "--ensemble-m",
        type=float,
        default=None,
        help="ensemble mode: >0 older plans dominate, <0 newest; default: execution.ensemble_m of --run-config",
    )
    p.add_argument("--margin", type=float, default=0.15, help="rad allowed outside the demo range")
    p.add_argument("--max-joint-speed", type=float, default=1.5, help="rad/s safety limit")
    p.add_argument(
        "--move-left",
        dest="freeze_left",
        action="store_false",
        help="let the model move the left arm (default: left arm held still, as in the demos)",
    )
    p.add_argument(
        "--run-config",
        type=Path,
        default=DEFAULT_RUN_CONFIG,
        help="run settings: execution.*, episode.max_seconds",
    )
    p.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="per episode; default: episode.max_seconds of --run-config",
    )
    p.add_argument("--record-dir", default="runs_infer")
    p.add_argument(
        "--dry-run",
        default="",
        help="episode dir: no hardware, simulated arm + recorded frames",
    )
    p.add_argument(
        "--fake-arms",
        action="store_true",
        help="with --dry-run: simulate the arms inside the real worker processes (tests RPC path)",
    )
    args = p.parse_args()
    if args.fake_arms and not args.dry_run:
        p.error("--fake-arms needs --dry-run (no fake arms next to real cameras)")
    try:
        run_cfg = load_run_config(args.run_config)
    except ValueError as exc:
        p.error(str(exc))
    ex = run_cfg["execution"]
    for arg, key in (("exec_mode", "mode"), ("exec_horizon", "horizon"), ("ensemble_m", "ensemble_m")):
        if getattr(args, arg) is None:
            setattr(args, arg, ex[key])
    if args.blend_s is None:
        args.blend_s = ex["blend_s"]
    if args.max_seconds is None:
        args.max_seconds = run_cfg["episode"]["max_seconds"]
    if not 1 <= args.exec_horizon <= 50:
        p.error(f"--exec-horizon must be 1..50, got {args.exec_horizon}")
    if args.exec_mode == "chunk":  # only the parameters that apply to the mode
        used = (f"horizon={args.exec_horizon} steps (new plan every {args.exec_horizon / 30:.2f} s)"
                f", blend={args.blend_s} s  [ensemble_m unused]")
    else:
        used = f"ensemble_m={args.ensemble_m} (new plan every frame)  [horizon, blend unused]"
    print(f"execution: {args.exec_mode}: {used}  episode limit: {args.max_seconds:.0f} s")

    if args.remote:
        from yam_infer.policy.server import RemotePolicy

        policy = RemotePolicy(args.remote, jpeg_roundtrip=not args.no_jpeg_roundtrip)
    else:
        from yam_infer.policy.policy import Policy

        policy = Policy(args.ckpt, args.device, jpeg_roundtrip=not args.no_jpeg_roundtrip)
    info = policy.info()
    run_meta = build_run_meta(args)
    print(
        f"policy: tasks={info['tasks']} chunk={info['chunk_size']} fps={info['fps']} device={info['device']}"
    )
    start_pose = np.asarray(info["start_pose"][args.task])

    cams = arms = keys = None
    rest_pose = None
    try:
        if args.dry_run:
            from yam_infer.hw.arms import SimArms
            from yam_infer.hw.cameras import ReplayCameraRig

            cams = ReplayCameraRig(args.dry_run)
            if args.fake_arms:
                log_dir = Path(args.record_dir)
                log_dir.mkdir(parents=True, exist_ok=True)
                arms = FollowerArms(args.hw_config, log_dir=log_dir, fake=True)
            else:
                arms = SimArms(start_pose)
        else:
            from yam_infer.hw.cameras import CameraRig, resolve_serials

            serials = resolve_serials(args.hw_config, args.left_serial, args.right_serial)
            print(f"cameras: {serials}")
            cams = CameraRig(serials)
            log_dir = Path(args.record_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            arms = FollowerArms(args.hw_config, log_dir=log_dir)
        cams.wait_ready()
        keys = KeyReader()
        rest_pose = (
            arms.get_state()
        )  # unpowered resting pose -> where we return before cutting torque
        print("follower state (rest pose):", np.round(rest_pose, 2))

        arms.begin()  # ALIGNING phase
        cur = arms.get_state()
        arms.submit(cur, "ALIGNING")
        if args.goto_start == "task_mean":
            print("moving to the demo start pose ... (Ctrl+C to abort)")
            smooth_move(arms, cur, start_pose, "ALIGNING")
        arms.activate()
        ep = 0
        while (
            True
        ):  # between episodes the heartbeat keeps re-sending the last target (no lease timeout)
            if run_episode(args, policy, cams, arms, keys, ep, run_meta) == "quit":
                break
            ep += 1
            print("reset the scene. moving back to the start pose ...")
            cur = arms.get_state()
            if args.goto_start == "task_mean":
                smooth_move(arms, cur, start_pose, "ACTIVE")
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        if arms is not None:
            park_and_release(arms, start_pose, rest_pose, args.park)
        if cams is not None:
            cams.close()


if __name__ == "__main__":
    main()
