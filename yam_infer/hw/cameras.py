"""Wrist cameras: two RealSense D405, 640x480 RGB @ 30 fps — same stream settings as the recorder
(SuperInference devices/realsense.py: rs.format.rgb8, no crop).  A background thread keeps the latest frame.

Serial resolution order: CLI argument > env YAM_{LEFT,RIGHT}_CAM_SERIAL >
`cameras:` section of configs/hardware.yaml.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

# sink(camera name, arrival time [monotonic s], rgb frame, frame counter, hardware stamp [ns])
FrameSink = Callable[[str, float, np.ndarray, int, int], None]


def resolve_serials(
    hw_config: Path, left: str | None = None, right: str | None = None
) -> dict[str, str]:
    """CLI > env YAM_{LEFT,RIGHT}_CAM_SERIAL > `cameras:` section of the hardware config."""
    import yaml

    left = left or os.environ.get("YAM_LEFT_CAM_SERIAL")
    right = right or os.environ.get("YAM_RIGHT_CAM_SERIAL")
    if not (left and right):
        try:
            cams = (yaml.safe_load(Path(hw_config).read_text()) or {}).get("cameras") or {}
        except (OSError, yaml.YAMLError):
            cams = {}
        left = left or cams.get("left_wrist")
        right = right or cams.get("right_wrist")
    if not (left and right):
        raise RuntimeError(
            f"camera serials unknown: set cameras.left_wrist/right_wrist in {hw_config}, "
            "pass --left-serial/--right-serial or set YAM_{LEFT,RIGHT}_CAM_SERIAL"
        )
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
        self._count = 0  # frame counter = header.seq of this camera's recorded frames
        self._lock = threading.Lock()
        self.name = serial
        self.sink: FrameSink | None = None
        self._hw_ns = 0
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
            t = time.monotonic()  # arrival time (same definition as SI's recorder)
            try:
                hw_ns = int(color.get_timestamp() * 1e6)  # RealSense hardware clock [ms -> ns]
            except RuntimeError:
                hw_ns = 0
            with self._lock:
                self._count += 1
                self._frame, self._t, self._hw_ns, seq = img, t, hw_ns, self._count
                sink = self.sink
            if sink is not None:
                sink(self.name, t, img, seq, hw_ns)

    def set_sink(self, sink: FrameSink | None) -> None:
        """Attach a per-frame callback.  The current latest frame is handed over at once: it is
        what the next grab() returns, so it must be in the recording too."""
        with self._lock:
            self.sink = sink
            frame, t, seq, hw_ns = self._frame, self._t, self._count, self._hw_ns
        if sink is not None and frame is not None:
            sink(self.name, t, frame, seq, hw_ns)

    def latest(self) -> tuple[float, np.ndarray | None, int]:
        """(arrival time, frame, frame counter)."""
        with self._lock:
            return self._t, self._frame, self._count

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
                self.cams[name].name = name
        except Exception:
            self.close()
            raise

    def wait_ready(self, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        while any(c.latest()[1] is None for c in self.cams.values()):
            if time.monotonic() > deadline:
                raise TimeoutError("cameras produced no frames")
            time.sleep(0.05)

    def set_sink(self, sink: FrameSink | None) -> None:
        """sink(name, t, rgb, seq, hw_ns) is called for every new frame (None = stop)."""
        for cam in self.cams.values():
            cam.set_sink(sink)

    def grab(self, max_age: float = 0.2) -> tuple[dict[str, np.ndarray], dict[str, tuple[float, int]]]:
        """Latest frame of every camera + (arrival time, frame counter).  Times are
        time.monotonic() (same clock as the follower state history).  Raises if a frame is older
        than max_age seconds."""
        now = time.monotonic()
        frames, info = {}, {}
        for name, cam in self.cams.items():
            t, f, seq = cam.latest()
            if f is None or now - t > max_age:
                raise RuntimeError(f"camera {name} stale ({now - t:.2f}s)")
            frames[name], info[name] = f, (t, seq)
        return frames, info

    def close(self) -> None:
        for c in self.cams.values():
            c.close()


class ReplayCameraRig:
    """Dry-run helper: plays a recorded NEDF3 episode as two 30 Hz cameras (no hardware).

    A background thread behaves like the RealSense threads: new frame every 1/30 s, arrival time,
    frame counter, sink callback for the recorder."""

    def __init__(self, episode_dir: str, fps: float = 30.0):
        from yam_infer.data.nedf3 import decode_jpeg_rgb, read_episode

        ep = read_episode(episode_dir)
        self._decode = decode_jpeg_rgb
        self.jpegs = {name: ep.cameras[name][1] for name in ("right_wrist", "left_wrist")}
        self.n = min(len(v) for v in self.jpegs.values())
        self.fps = fps
        self.sink: FrameSink | None = None
        self._latest: dict[str, tuple[float, np.ndarray, int]] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        t0 = time.monotonic()
        i = 0
        while not self._stop.is_set():
            k = min(i, self.n - 1)  # hold the last frame at the end of the recording
            frames = {name: self._decode(v[k]) for name, v in self.jpegs.items()}
            t = time.monotonic()
            with self._lock:
                for name, rgb in frames.items():
                    self._latest[name] = (t, rgb, i + 1)
                sink = self.sink
            if sink is not None:
                for name, rgb in frames.items():
                    sink(name, t, rgb, i + 1, 0)
            i += 1
            self._stop.wait(max(0.0, t0 + i / self.fps - time.monotonic()))

    def wait_ready(self, timeout: float = 10.0) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True, name="cam-replay")
            self._thread.start()
        deadline = time.monotonic() + timeout
        while len(self._latest) < len(self.jpegs):
            if time.monotonic() > deadline:
                raise TimeoutError("replay produced no frames")
            time.sleep(0.01)

    def set_sink(self, sink: FrameSink | None) -> None:
        with self._lock:
            self.sink = sink
            latest = dict(self._latest)
        if sink is not None:  # current frames = what the next grab() returns -> record them too
            for name, (t, rgb, seq) in latest.items():
                sink(name, t, rgb, seq, 0)

    def grab(self, max_age: float = 0.2) -> tuple[dict[str, np.ndarray], dict[str, tuple[float, int]]]:
        with self._lock:
            latest = dict(self._latest)
        return {k: v[1] for k, v in latest.items()}, {k: (v[0], v[2]) for k, v in latest.items()}

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
