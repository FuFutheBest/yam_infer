"""Record everything about one inference episode so it can be reviewed afterwards.

runs_infer/<YYYYmmdd_HHMMSS>_<task>/
    meta.json        checkpoint, task, settings, safety counters, end reason
    policy.npz       per policy step (≈30 Hz): t, qpos(14), chunk(K,14), latency_ms
    command.npz      per command step (100 Hz): t, target_raw(14), target_safe(14), qpos(14)
    frames.mp4       model-input images (right | left side by side) at policy rate
Review with:  python -m yam_infer.tools.review runs_infer/<run>
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np


class RunRecorder:
    def __init__(self, root: Path, task: str, meta: dict, fps: float):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.dir = Path(root) / f"{stamp}_{task}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta = {**meta, "task": task, "started": stamp}
        self.policy: dict[str, list] = {
            "t": [],
            "qpos": [],
            "chunk": [],
            "latency_ms": [],
        }
        self.command: dict[str, list] = {
            "t": [],
            "target_raw": [],
            "target_safe": [],
            "qpos": [],
        }
        self.fps = fps
        self._video: cv2.VideoWriter | None = None
        self._lock = threading.Lock()
        self._save_meta()

    def _save_meta(self) -> None:
        try:
            (self.dir / "meta.json").write_text(
                json.dumps(self.meta, indent=1, ensure_ascii=False, default=str)
            )
        except OSError as exc:
            print(f"warning: could not write meta.json: {exc}")

    def log_policy(
        self,
        t: float,
        qpos: np.ndarray,
        chunk: np.ndarray,
        latency_ms: float,
        images: np.ndarray,
    ):
        with self._lock:
            self.policy["t"].append(t)
            self.policy["qpos"].append(np.asarray(qpos, np.float32))
            self.policy["chunk"].append(np.asarray(chunk, np.float32))
            self.policy["latency_ms"].append(latency_ms)
        frame = np.concatenate(list(images), axis=1)[
            :, :, ::-1
        ]  # (H, n*W, 3) BGR for OpenCV
        if self._video is None:
            h, w = frame.shape[:2]
            self._video = cv2.VideoWriter(
                str(self.dir / "frames.mp4"),
                cv2.VideoWriter.fourcc(*"mp4v"),
                self.fps,
                (w, h),
            )
        self._video.write(np.ascontiguousarray(frame))

    def log_command(
        self,
        t: float,
        raw: np.ndarray | None,
        safe: np.ndarray,
        qpos: np.ndarray | None,
    ):
        with self._lock:
            self.command["t"].append(t)
            self.command["target_raw"].append(
                np.full(14, np.nan) if raw is None else np.asarray(raw, np.float32)
            )
            self.command["target_safe"].append(np.asarray(safe, np.float32))
            self.command["qpos"].append(
                np.full(14, np.nan) if qpos is None else np.asarray(qpos, np.float32)
            )

    def close(self, **extra) -> Path:
        self.meta.update(extra)
        self.meta["ended"] = time.strftime("%Y%m%d_%H%M%S")
        self._save_meta()
        if self._video is not None:
            self._video.release()
        with self._lock:
            for name, buf in (("policy", self.policy), ("command", self.command)):
                if buf["t"]:
                    arrays: dict[str, np.ndarray] = {
                        k: np.asarray(v) for k, v in buf.items()
                    }
                    try:
                        with open(self.dir / f"{name}.npz", "wb") as fh:
                            np.savez_compressed(fh, allow_pickle=False, **arrays)
                    except OSError as exc:
                        print(f"ERROR: could not save {name}.npz: {exc}")
        return self.dir
