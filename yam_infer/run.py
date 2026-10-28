"""Run a trained policy on the I2RT YAM follower arms.

    python -m yam_infer.run --ckpt model.pt --task put_blue_block_in_blue_cup \
        --si-root ~/SuperInference-feat-i2rt_NEDF3

Dry run without any hardware (simulated arm + frames from a recorded episode):
    python -m yam_infer.run --ckpt model.pt --task put_banana_in_bowl --dry-run EPISODE_DIR

Keys (type + Enter):  Enter = start / pause-resume   r = new episode   q = park & quit
Threads:
    policy thread  ~30 Hz : cameras + follower state -> model -> chunk -> scheduler
    control loop   100 Hz : scheduler.target_at(now) -> safety filter -> submit to workers -> recorder
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

from yam_infer.recorder import RunRecorder
from yam_infer.scheduler import ActionScheduler, SafetyFilter

CONTROL_HZ = 100.0


def smooth_move(arms, start: np.ndarray, goal: np.ndarray, phase: str, max_speed: float = 0.5,
                min_time: float = 2.0) -> None:
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


def run_episode(args, policy, cams, arms, keys: KeyReader, ep_index: int) -> str:
    """One policy episode. Returns 'next' or 'quit'."""
    scheduler = ActionScheduler(policy.fps, policy.chunk_size, args.ensemble_m, mode=args.exec_mode,
                                horizon=args.exec_horizon)
    safety = SafetyFilter(policy.a_min, policy.a_max, args.margin, args.max_joint_speed,
                          max_grip_squeeze=args.max_grip_squeeze)
    recorder = RunRecorder(Path(args.record_dir), args.task, {
        "ckpt": str(args.ckpt), "episode_index": ep_index, "policy": policy.info(),
        "settings": {k: v for k, v in vars(args).items() if k != "ckpt"},
    }, fps=policy.fps)
    stop = threading.Event()
    paused = threading.Event()
    state_lock = threading.Lock()
    shared: dict = {"qpos": arms.get_state(), "error": None}
    stats = {"policy_steps": 0}

    def policy_loop():
        period = 1.0 / policy.fps
        t_next = time.monotonic()
        while not stop.is_set():
            try:
                if not paused.is_set() and scheduler.needs_new_chunk(time.monotonic()):
                    frames, _t_cam = cams.grab()
                    with state_lock:
                        qpos = shared["qpos"].copy()
                    t_obs = time.monotonic()
                    chunk, images = policy.predict(frames, qpos, args.task)
                    latency = (time.monotonic() - t_obs) * 1000
                    scheduler.add(t_obs, chunk)
                    recorder.log_policy(t_obs, qpos, chunk, latency, images)
                    stats["policy_steps"] += 1
                    if stats["policy_steps"] % (int(policy.fps * 2) if args.exec_mode == "ensemble" else 3) == 0:
                        print(f"  policy step {stats['policy_steps']}  latency {latency:.0f} ms", flush=True)
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
            recorder.close(end_reason="quit before start")
            return "quit"
        if k == "":
            break

    hold = arms.get_state()
    stall_ref, stall_t = None, time.monotonic()
    safety.reset(hold)
    thread = threading.Thread(target=policy_loop, daemon=True)
    thread.start()
    t_start = time.monotonic()
    end_reason = "unknown"
    t_next = time.monotonic()
    last_target = hold.copy()
    print("  RUNNING  (Enter = pause/resume, r = end episode, q = quit)")
    try:
        while not stop.is_set():
            now = time.monotonic()
            qpos = arms.get_state()
            with state_lock:
                shared["qpos"] = qpos
            raw = None if paused.is_set() else scheduler.target_at(now)
            target = last_target if raw is None else raw
            if args.freeze_left:
                target = target.copy()
                target[0:7] = hold[0:7]
            safe = safety(target, 1.0 / CONTROL_HZ, qpos)
            arms.submit(safe, "ACTIVE")
            last_target = safe
            recorder.log_command(now, raw, safe, qpos)

            k = keys.get()
            if k == "":
                if paused.is_set():
                    scheduler.clear()
                    paused.clear()
                    print("  resumed")
                else:
                    paused.set()
                    print("  PAUSED (holding pose) - Enter to resume")
            elif k in ("r", "q"):
                end_reason = "operator next" if k == "r" else "operator quit"
                break
            # stall detector: right arm has not moved for stall_seconds -> stop (saves the gripper motor)
            if stall_ref is None or paused.is_set() or np.abs(qpos[7:13] - stall_ref).max() > args.stall_rad:
                stall_ref, stall_t = qpos[7:13].copy(), now
            elif args.stall_seconds > 0 and not paused.is_set() and now - stall_t > args.stall_seconds:
                end_reason = f"stalled: right arm moved < {args.stall_rad} rad in {args.stall_seconds:.0f}s"
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
        out = recorder.close(end_reason=end_reason, duration_s=time.monotonic() - t_start,
                             policy_steps=stats["policy_steps"], safety_clamped=safety.clamped,
                             safety_rate_limited=safety.rate_limited, grip_squeeze_limited=safety.squeeze_limited)
        print(f"  episode ended: {end_reason}\n  recorded -> {out}")
    if end_reason.startswith(("control loop", "policy thread")) or end_reason in ("ctrl-c", "operator quit"):
        return "quit"
    return "next"


def save_rest_pose(record_dir: Path, rest_pose: np.ndarray) -> None:
    """Used by `yam_infer.tools.stop_all` to bring orphaned arms home."""
    try:
        (record_dir / "rest_pose.json").write_text(json.dumps(np.asarray(rest_pose).tolist()))
    except OSError as exc:
        print(f"warning: could not save rest pose: {exc}")


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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--task", required=True, choices=["put_banana_in_bowl", "put_blue_block_in_blue_cup"])
    p.add_argument("--si-root", type=Path, default=Path.home() / "SuperInference-feat-i2rt_NEDF3")
    p.add_argument("--si-python", type=Path, default=None, help="default: <si-root>/.venv/bin/python")
    p.add_argument("--teleop-config", type=Path, default=None,
                   help="default: <si-root>/configs/i2rt_d405_yam_teleop.yaml")
    p.add_argument("--left-serial")
    p.add_argument("--right-serial")
    p.add_argument("--remote", default="", help="host:port of `python -m yam_infer.server` (GPU elsewhere)")
    p.add_argument("--device", default="auto")
    p.add_argument("--no-jpeg-roundtrip", action="store_true")
    p.add_argument("--goto-start", choices=["task_mean", "none"], default="task_mean",
                   help="move to the mean start pose of the demos before each episode")
    p.add_argument("--park", choices=["home", "start", "none"], default="home",
                   help="home: back to the rest pose the arms had at program start, then motors off (default)")
    p.add_argument("--exec-mode", choices=["chunk", "ensemble"], default="chunk",
                   help="chunk: execute --exec-horizon steps of each prediction, then re-plan (default); "
                        "ensemble: re-plan every step + ACT temporal ensembling")
    p.add_argument("--exec-horizon", type=int, default=30, help="chunk mode: steps (at 30 Hz) per prediction")
    p.add_argument("--ensemble-m", type=float, default=0.01,
                   help="ensemble mode: >0 older predictions dominate (ACT paper 0.01); <0 newest dominates")
    p.add_argument("--margin", type=float, default=0.15, help="rad allowed outside the demo range")
    p.add_argument("--max-joint-speed", type=float, default=1.5, help="rad/s safety limit")
    p.add_argument("--move-left", dest="freeze_left", action="store_false",
                   help="let the model move the left arm (default: left arm held still, as in the demos)")
    p.add_argument("--max-seconds", type=float, default=40.0, help="per episode (demos last 12-25 s)")
    p.add_argument("--stall-seconds", type=float, default=10.0, help="end episode if the right arm is still; 0=off")
    p.add_argument("--stall-rad", type=float, default=0.08)
    p.add_argument("--max-grip-squeeze", type=float, default=0.1,
                   help="gripper target at most this much narrower than measured (protects the gripper motor)")
    p.add_argument("--record-dir", default="runs_infer")
    p.add_argument("--dry-run", default="", help="episode dir: no hardware, simulated arm + recorded frames")
    args = p.parse_args()

    if args.remote:
        from yam_infer.server import RemotePolicy

        policy = RemotePolicy(args.remote, jpeg_roundtrip=not args.no_jpeg_roundtrip)
    else:
        from yam_infer.policy import Policy

        policy = Policy(args.ckpt, args.device, jpeg_roundtrip=not args.no_jpeg_roundtrip)
    info = policy.info()
    print(f"policy: tasks={info['tasks']} chunk={info['chunk_size']} fps={info['fps']} device={info['device']}")
    start_pose = np.asarray(info["start_pose"][args.task])

    cams = arms = keys = None
    rest_pose = None
    try:
        if args.dry_run:
            from yam_infer.cameras import ReplayCameraRig
            from yam_infer.robot import SimArms

            cams = ReplayCameraRig(args.dry_run)
            arms = SimArms(start_pose)
        else:
            from yam_infer.cameras import CameraRig, resolve_serials
            from yam_infer.robot import FollowerArms

            si_root = args.si_root.expanduser().resolve()
            serials = resolve_serials(si_root, args.left_serial, args.right_serial)
            print(f"cameras: {serials}")
            cams = CameraRig(serials)
            log_dir = Path(args.record_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            arms = FollowerArms(
                si_root, args.si_python or si_root / ".venv" / "bin" / "python",
                args.teleop_config or si_root / "configs" / "i2rt_d405_yam_teleop.yaml", log_dir=log_dir,
            )
        cams.wait_ready()
        keys = KeyReader()
        rest_pose = arms.get_state()  # unpowered resting pose -> where we return before cutting torque
        print("follower state (rest pose):", np.round(rest_pose, 2))
        if not args.dry_run:
            save_rest_pose(Path(args.record_dir), rest_pose)

        arms.begin()  # ALIGNING phase
        cur = arms.get_state()
        arms.submit(cur, "ALIGNING")
        if args.goto_start == "task_mean":
            print("moving to the demo start pose ... (Ctrl+C to abort)")
            smooth_move(arms, cur, start_pose, "ALIGNING")
        arms.activate()
        ep = 0
        while True:  # between episodes the heartbeat keeps re-sending the last target (no lease timeout)
            if run_episode(args, policy, cams, arms, keys, ep) == "quit":
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
