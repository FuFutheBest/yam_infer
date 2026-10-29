"""Recent follower states, so an image can be paired with the joints AT the image's time.

Training (opus_inference_train data conversion) built each sample on a 30 Hz grid: joints
linearly interpolated at the grid time t, camera = the frame whose timestamp is nearest to t.
At inference the image is the input that is "late" (it arrived 0-33 ms ago), so we keep the last
~1 s of follower states (written by the 100 Hz control loop) and interpolate them at the image
time instead of pairing the image with the newest reading.
All times are time.monotonic() seconds (same clock as the camera threads).
"""

from __future__ import annotations

import threading
from collections import deque

import numpy as np


class StateHistory:
    def __init__(self, seconds: float = 1.0, rate_hz: float = 100.0):
        self._buf: deque[tuple[float, np.ndarray]] = deque(maxlen=max(2, int(seconds * rate_hz)))
        self._lock = threading.Lock()

    def add(self, t: float, state: np.ndarray) -> None:
        with self._lock:
            if self._buf and t <= self._buf[-1][0]:
                return  # keep times strictly increasing
            self._buf.append((t, np.asarray(state, dtype=np.float64).copy()))

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()

    def at(self, t: float) -> np.ndarray:
        """State at time t: linear interpolation between the two neighbouring readings
        (clamped to the oldest / newest reading outside the stored range)."""
        with self._lock:
            if not self._buf:
                raise RuntimeError("state history is empty")
            times = np.fromiter((s[0] for s in self._buf), dtype=np.float64, count=len(self._buf))
            states = [s[1] for s in self._buf]
        if t <= times[0]:
            return states[0].copy()
        if t >= times[-1]:
            return states[-1].copy()
        i = int(np.searchsorted(times, t))  # times[i-1] < t <= times[i]
        a = (t - times[i - 1]) / (times[i] - times[i - 1])
        return states[i - 1] + (states[i] - states[i - 1]) * a
