"""Record one inference episode (from Enter to its end) for review / replay.

runs_infer/<YYYYmmdd_HHMMSS>_<task>/        (named ..._inprogress while recording)
    metadata.json      NEDF3 metadata (task, camera mapping, state layout)
    data/*.mcap        RAW STREAMS, NEDF3-compatible (see data/nedf3_writer.py):
                         every camera frame (JPEG q90, header.seq = frame counter,
                         header.stamp = RealSense hardware time), every follower reading (~100 Hz)
    policy.npz         MODEL INPUT/OUTPUT, one row per inference:
                         t          observation time (mean camera arrival time)  [s, monotonic]
                         cam_t      arrival time of each camera frame used        (n_cam,)
                         cam_seq    header.seq of those frames in data/*.mcap     (n_cam,)
                         qpos       follower state interpolated at t (model input) (14,)
                         raw_out    network output, normalised (NaN for --remote)  (K,14)
                         chunk      absolute joint targets, step k is at t + k/fps (K,14)
                         t_infer / t_done   inference start / end; latency_ms = (t_done-t_infer)*1000
                       camera order = meta.json policy.cameras; images are NOT stored twice:
                       model input = resize(decode(JPEG of cam_seq))  (tools/replay_policy.py)
    command.npz        CONTROL LOOP, 100 Hz: t (after reading the arms), qpos, target_raw (model),
                         target_safe (sent), t_sample (worker sampling time per arm, left/right)
    events.jsonl       start / pause / resume / end / errors, with times
    episode_stats.json message counts per topic, dropped frames, writer error, image age
    meta.json          checkpoint (+sha256), code version, settings, config files, clock mapping
    worker_*.log       copies of the worker logs (first episode of a run: whole start-up)

Times: monotonic seconds everywhere in npz/jsonl; MCAP log_time = wall-clock ns obtained as
monotonic + one offset measured at start (meta.json clock) -> no jumps if NTP adjusts the clock.
Finalisation like SI's recorder: an episode with missing streams or a writer error goes to
runs_infer/quarantine/ (+ quarantine_reason.json); a crashed run's *_inprogress dir is moved there
at the next start; an episode that never started (quit before Enter) leaves nothing behind.

Review:  python -m yam_infer.tools.review runs_infer/<run> [--video]
Replay:  python -m yam_infer.tools.replay_policy runs_infer/<run> --ckpt <model.pt>
"""

from __future__ import annotations

import json
import queue
import shutil
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from yam_infer.data.nedf3_writer import (
    CAMERA_TOPICS,
    CONTRACT_VERSION,
    HAND_TOPICS,
    JOINT_TOPICS,
    Nedf3Writer,
    encode_jpeg_rgb,
)

MAX_QUEUED_FRAMES = 120  # ~2 s of both cameras; beyond that frames are dropped (and counted)


def recover_incomplete(root: Path) -> None:
    """Move episodes left *_inprogress by a crashed run into root/quarantine/."""
    for d in Path(root).glob("*_inprogress"):
        if d.is_dir():
            _quarantine(
                d,
                d.name.removesuffix("_inprogress") + "_interrupted",
                {"reason": "program exited before the episode was finalised"},
            )


def _quarantine(src: Path, name: str, reason: dict) -> Path:
    qdir = src.parent / "quarantine"
    qdir.mkdir(exist_ok=True)
    dst = qdir / name
    if dst.exists():
        dst = qdir / f"{name}_{uuid.uuid4().hex[:8]}"
    src.rename(dst)
    (dst / "quarantine_reason.json").write_text(json.dumps(reason, indent=1, ensure_ascii=False))
    return dst


class RunRecorder:
    def __init__(self, root: Path, task: str, meta: dict, fps: float):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        recover_incomplete(root)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.name = f"{stamp}_{task}"
        self.dir = root / f"{self.name}_inprogress"
        self.dir.mkdir()
        self.fps = fps
        # one fixed monotonic -> wall mapping for the whole episode
        self.mono0_ns, self.wall0_ns = time.monotonic_ns(), time.time_ns()
        self.meta = {
            **meta,
            "task": task,
            "started": stamp,
            "format": "yam_infer-recording-v2",
            "clock": {"monotonic_ns": self.mono0_ns, "wall_ns": self.wall0_ns},
        }
        self._save_meta()
        cameras = list(meta.get("policy", {}).get("cameras", ["right_wrist", "left_wrist"]))
        self.nedf = Nedf3Writer(
            self.dir,
            {
                "hardware_config_name": "i2rt_dual_bilateral",
                "task_name": task,
                "task_description": task,
                "collection_mode": "policy_inference",
                "robot_vendor_model": "I2RT_YAM",
                "end_effector_vendor_model": "I2RT_gripper",
                "camera_info": {"camera_0": "right_wrist", "camera_1": "left_wrist"},
                "data_format": "NEDF3",
                "main_camera": "camera_0",
                "robot_type": "dual_arm",
                "end_effector": "gripper",
                "extra": {
                    "contract_version": CONTRACT_VERSION,
                    "producer": "yam_infer",
                    "state_layout": "left_joint0..5,left_gripper,right_joint0..5,right_gripper",
                    "role": "follower only (no leader during inference)",
                    "camera_color_space": {"jpeg": "rgb", "opencv_encoder_input": "bgr"},
                    "camera_header": "seq = frame counter, stamp = RealSense hardware time [ns]",
                    "policy_cameras": cameras,
                },
                "record_topic_list": [
                    "nmx/nedf/metadata",
                    *CAMERA_TOPICS.values(),
                    *JOINT_TOPICS.values(),
                    *HAND_TOPICS.values(),
                ],
            },
            self.wall0_ns,
        )
        self.policy: dict[str, list] = {
            k: []
            for k in (
                "t",
                "cam_t",
                "cam_seq",
                "qpos",
                "raw_out",
                "chunk",
                "t_infer",
                "t_done",
                "latency_ms",
            )
        }
        self.command: dict[str, list] = {
            k: [] for k in ("t", "target_raw", "target_safe", "qpos", "t_sample")
        }
        self._lock = threading.Lock()
        self._q: queue.Queue = queue.Queue()
        self._frames_queued = 0
        self.dropped_frames = 0
        self.writer_error: str | None = None
        try:
            self._events = open(self.dir / "events.jsonl", "a", encoding="utf-8")  # noqa: SIM115
        except OSError as exc:
            self.nedf.close()
            raise RuntimeError(f"cannot create events.jsonl in {self.dir}: {exc}") from exc
        self._thread = threading.Thread(target=self._writer_loop, daemon=True, name="recorder")
        self._thread.start()

    # ------------------------------------------------------------------ helpers
    def wall_ns(self, t_mono: float) -> int:
        return self.wall0_ns + int(round(t_mono * 1e9)) - self.mono0_ns

    def _save_meta(self) -> None:
        try:
            (self.dir / "meta.json").write_text(
                json.dumps(self.meta, indent=1, ensure_ascii=False, default=str)
            )
        except OSError as exc:
            print(f"warning: could not write meta.json: {exc}")

    def _writer_loop(self) -> None:
        while True:
            item = self._q.get()
            try:
                if item is None:
                    return
                if self.writer_error is not None:
                    continue
                kind, wall, payload = item
                if kind == "cam":
                    name, rgb, seq, hw_ns = payload
                    self.nedf.write_camera(name, wall, encode_jpeg_rgb(rgb), rgb.shape, seq, hw_ns)
                else:
                    self.nedf.write_state(wall, payload)
            except Exception as exc:  # noqa: BLE001 - recorded, episode goes to quarantine
                self.writer_error = repr(exc)
            finally:
                if item is not None and item[0] == "cam":
                    with self._lock:
                        self._frames_queued -= 1

    # ------------------------------------------------------------------ inputs
    def camera_sink(self, name: str, t: float, rgb: np.ndarray, seq: int, hw_ns: int) -> None:
        """Called from the camera threads for EVERY frame while the episode runs."""
        with self._lock:
            if self._frames_queued >= MAX_QUEUED_FRAMES:
                self.dropped_frames += 1
                return
            self._frames_queued += 1
        self._q.put(("cam", self.wall_ns(t), (name, rgb, seq, hw_ns)))

    def event(self, name: str, t: float | None = None, **info) -> None:
        t = time.monotonic() if t is None else t
        line = json.dumps(
            {"t": t, "wall_ns": self.wall_ns(t), "event": name, **info},
            ensure_ascii=False,
            default=str,
        )
        with self._lock:
            self._events.write(line + "\n")
            self._events.flush()

    def log_policy(
        self,
        t: float,
        qpos: np.ndarray,
        chunk: np.ndarray,
        *,
        cam_t,
        cam_seq,
        t_infer: float,
        t_done: float,
        raw_out: np.ndarray | None = None,
    ) -> None:
        with self._lock:
            self.policy["t"].append(t)
            self.policy["cam_t"].append(np.asarray(cam_t, np.float64))
            self.policy["cam_seq"].append(np.asarray(cam_seq, np.int64))
            self.policy["qpos"].append(np.asarray(qpos, np.float32))
            self.policy["raw_out"].append(
                np.full(np.shape(chunk), np.nan, np.float32)
                if raw_out is None
                else np.asarray(raw_out, np.float32)
            )
            self.policy["chunk"].append(np.asarray(chunk, np.float32))
            self.policy["t_infer"].append(t_infer)
            self.policy["t_done"].append(t_done)
            self.policy["latency_ms"].append((t_done - t_infer) * 1000)

    def log_command(
        self,
        t: float,
        raw: np.ndarray | None,
        safe: np.ndarray,
        qpos: np.ndarray,
        t_sample: np.ndarray | None = None,
    ) -> None:
        """One control step: the reading taken at t (also written to the raw MCAP) + targets."""
        qpos = np.asarray(qpos, np.float64)
        with self._lock:
            self.command["t"].append(t)
            self.command["target_raw"].append(
                np.full(14, np.nan) if raw is None else np.asarray(raw, np.float32)
            )
            self.command["target_safe"].append(np.asarray(safe, np.float32))
            self.command["qpos"].append(qpos.astype(np.float32))
            self.command["t_sample"].append(
                np.full(2, np.nan) if t_sample is None else np.asarray(t_sample, np.float64)
            )
        self._q.put(("state", self.wall_ns(t), qpos.copy()))

    # ------------------------------------------------------------------ finish
    def close(self, worker_logs: list[Path] | None = None, **extra) -> Path | None:
        """Flush everything, then finalise / quarantine / remove the episode directory."""
        self._q.put(None)
        self._thread.join(timeout=30)
        if self._thread.is_alive():
            self.writer_error = self.writer_error or "writer thread did not finish in 30 s"
        try:
            self.nedf.close()
        except Exception as exc:  # noqa: BLE001
            self.writer_error = self.writer_error or f"closing mcap: {exc!r}"
        with self._lock:
            self._events.close()
            for name, buf in (("policy", self.policy), ("command", self.command)):
                if buf["t"]:
                    try:
                        with open(self.dir / f"{name}.npz", "wb") as fh:
                            np.savez_compressed(
                                fh, allow_pickle=False, **{k: np.asarray(v) for k, v in buf.items()}
                            )
                    except OSError as exc:
                        self.writer_error = self.writer_error or f"{name}.npz: {exc}"
        for log in worker_logs or []:
            try:
                shutil.copy2(log, self.dir / log.name)
            except OSError:
                pass
        counts = dict(self.nedf.counts)
        age = np.asarray(self.policy["t_infer"]) - np.asarray(self.policy["t"])
        stats = {
            "topic_counts": counts,
            "policy_steps": len(self.policy["t"]),
            "control_steps": len(self.command["t"]),
            "dropped_camera_frames": self.dropped_frames,
            "writer_error": self.writer_error,
            "image_age_ms": None
            if not len(age)
            else {"median": float(np.median(age) * 1000), "max": float(age.max() * 1000)},
        }
        (self.dir / "episode_stats.json").write_text(json.dumps(stats, indent=1))
        self.meta.update(extra)
        self.meta["ended"] = time.strftime("%Y%m%d_%H%M%S")
        self._save_meta()

        if not self.policy["t"] and not self.command["t"]:  # e.g. quit before Enter
            try:
                shutil.rmtree(self.dir)
            except OSError as exc:
                print(f"warning: could not remove empty episode {self.dir}: {exc}")
            return None
        missing = [t for t, n in counts.items() if n == 0]
        if missing or self.writer_error:
            return _quarantine(
                self.dir, self.name, {"missing_topics": missing, "writer_error": self.writer_error}
            )
        final = self.dir.parent / self.name
        if final.exists():
            final = self.dir.parent / f"{self.name}_{uuid.uuid4().hex[:4]}"
        self.dir.rename(final)
        self.dir = final
        return final
