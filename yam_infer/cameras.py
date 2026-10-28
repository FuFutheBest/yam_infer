"""Wrist cameras: two RealSense D405, 640x480 RGB @ 30 fps — same stream settings as the recorder
(SuperInference devices/realsense.py: rs.format.rgb8, no crop).  A background thread keeps the latest frame.

Serial resolution order: CLI argument > env SI_I2RT_{LEFT,RIGHT}_RS_SERIAL >
SuperInference .superinference/local/i2rt_hardware_bindings.json (saved by the WebUI).
"""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np


def resolve_serials(si_root: Path, left: str | None, right: str | None) -> dict[str, str]:
    left = left or os.environ.get("SI_I2RT_LEFT_RS_SERIAL")
    right = right or os.environ.get("SI_I2RT_RIGHT_RS_SERIAL")
    if not (left and right):
        path = si_root / ".superinference" / "local" / "i2rt_hardware_bindings.json"
        try:
            profiles = json.loads(path.read_text()).get("profiles", {})
        except (OSError, json.JSONDecodeError):
            profiles = {}
        for name in ("yam_d405_2cam", "yam_d405_3cam"):
            cams = profiles.get(name, {}).get("cameras", {})
            left = left or cams.get("left", {}).get("identity")
            right = right or cams.get("right", {}).get("identity")
    if not (left and right):
        raise RuntimeError("camera serials unknown: pass --left-serial/--right-serial or set SI_I2RT_*_RS_SERIAL")
    return {"left_wrist": str(left), "right_wrist": str(right)}


class RealSenseCamera:
    def __init__(self, serial: str, width: int = 640, height: int = 480, fps: int = 30):
        # compiled extension without type stubs -> treat as dynamic
        rs: Any = importlib.import_module("pyrealsense2")
        self.serial = serial
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        self.pipeline.start(config)
        for _ in range(10):  # let auto-exposure settle
            self.pipeline.wait_for_frames()
        self._frame: np.ndarray | None = None
        self._t = 0.0
        self._count = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name=f"cam-{serial}")
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                frames = self.pipeline.wait_for_frames(1000)
            except RuntimeError:
                continue
            color = frames.get_color_frame()
            if not color:
                continue
            img = np.asanyarray(color.get_data()).copy()
            with self._lock:
                self._frame, self._t = img, time.monotonic()
                self._count += 1

    def latest(self) -> tuple[float, np.ndarray | None]:
        with self._lock:
            return self._t, self._frame

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        with contextlib.suppress(RuntimeError):
            self.pipeline.stop()


class CameraRig:
    def __init__(self, serials: dict[str, str]):
        self.cams: dict[str, RealSenseCamera] = {}
        try:
            for name, serial in serials.items():
                self.cams[name] = RealSenseCamera(serial)
        except Exception:
            self.close()
            raise

    def wait_ready(self, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        while any(c.latest()[1] is None for c in self.cams.values()):
            if time.monotonic() > deadline:
                raise TimeoutError("cameras produced no frames")
            time.sleep(0.05)

    def grab(self, max_age: float = 0.2) -> tuple[dict[str, np.ndarray], float]:
        """Latest frame of every camera; raises if one is older than max_age seconds."""
        now = time.monotonic()
        frames, oldest = {}, now
        for name, cam in self.cams.items():
            t, f = cam.latest()
            if f is None or now - t > max_age:
                raise RuntimeError(f"camera {name} stale ({now - t:.2f}s)")
            frames[name] = f
            oldest = min(oldest, t)
        return frames, oldest

    def close(self) -> None:
        for c in self.cams.values():
            c.close()


class ReplayCameraRig:
    """Dry-run helper: serves frames from a recorded NEDF3 episode (no hardware)."""

    def __init__(self, episode_dir: str):
        from yam_infer.nedf3 import decode_jpeg_rgb, read_episode

        ep = read_episode(episode_dir)
        self._decode = decode_jpeg_rgb
        self.frames = {name: ep.cameras[name][1] for name in ("right_wrist", "left_wrist")}
        self.n = min(len(v) for v in self.frames.values())
        self.t0 = time.monotonic()

    def wait_ready(self, timeout: float = 0) -> None:
        self.t0 = time.monotonic()

    def grab(self, max_age: float = 0.2) -> tuple[dict[str, np.ndarray], float]:
        i = min(int((time.monotonic() - self.t0) * 30), self.n - 1)
        return {k: self._decode(v[i]) for k, v in self.frames.items()}, time.monotonic()

    def close(self) -> None:
        pass
