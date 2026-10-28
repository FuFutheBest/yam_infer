"""Turn 30 Hz action chunks into a smooth 100 Hz joint-target stream.

Two execution modes:

* "chunk" (default): COMMIT to a chunk for `horizon` steps before using a new one; the switch is
  cross-faded over `blend_s`.  Robust against "procrastination": the model often predicts
  "pause a moment, then move" (humans pause in the demos); re-planning every step then keeps
  executing only the pause and the arm stalls forever (seen on the real robot, 2026-10-27).
* "ensemble": ACT temporal ensembling, described below.

Each chunk predicted from an observation taken at time t_obs covers times
t_obs + k/fps (k = 0..K-1).  At any query time we interpolate every chunk that
covers that time and average them (ACT "temporal ensembling") with weights
w_i = exp(-m * i), i = 0 for the OLDEST chunk (ACT convention):
    m > 0  -> old predictions dominate (ACT paper: 0.01, very smooth but laggy)
    m = 0  -> plain average
    m < 0  -> newest predictions dominate (-0.3 ≈ last ~3 predictions)
run.py default: --ensemble-m 0.01
"""

from __future__ import annotations

import threading
from collections import deque

import numpy as np


class ActionScheduler:
    def __init__(
        self,
        fps: float,
        chunk_size: int,
        ensemble_m: float = 0.01,
        max_chunks: int | None = None,
        mode: str = "chunk",
        horizon: int = 30,
        blend_s: float = 0.1,
    ):
        if mode not in ("chunk", "ensemble"):
            raise ValueError(f"unknown mode {mode}")
        self.dt = 1.0 / fps
        self.k = chunk_size
        self.m = ensemble_m
        self.mode = mode
        self.horizon = min(horizon, chunk_size)
        self.blend_s = blend_s
        self._chunks: deque[tuple[float, np.ndarray]] = deque(maxlen=max_chunks or chunk_size)
        self._lock = threading.Lock()

    def add(self, t_obs: float, actions: np.ndarray) -> None:
        with self._lock:
            self._chunks.append((t_obs, np.asarray(actions, dtype=np.float64)))

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()

    def needs_new_chunk(self, t: float) -> bool:
        """Chunk mode: only predict when the committed chunk is used up (ensemble: always)."""
        if self.mode == "ensemble":
            return True
        with self._lock:
            if not self._chunks:
                return True
            t0 = self._chunks[-1][0]
        return (t - t0) / self.dt >= self.horizon - 1

    def _interp(self, t0: float, acts: np.ndarray, t: float) -> np.ndarray | None:
        f = (t - t0) / self.dt
        if f < 0:
            return None
        f = min(f, self.k - 1)  # hold the last action if a chunk is overrun
        lo = int(np.floor(f))
        hi = min(lo + 1, self.k - 1)
        return acts[lo] + (acts[hi] - acts[lo]) * (f - lo)

    def target_at(self, t: float) -> np.ndarray | None:
        """Target at time t (seconds, same clock as t_obs) or None if no chunk covers t."""
        with self._lock:
            chunks = list(self._chunks)
        if self.mode == "chunk":
            if not chunks:
                return None
            t_new, a_new = chunks[-1]
            new = self._interp(t_new, a_new, t)
            if len(chunks) == 1 or new is None:
                return (
                    new
                    if new is not None
                    else (self._interp(*chunks[-2], t) if len(chunks) > 1 else None)
                )
            alpha = (t - t_new) / self.blend_s
            if alpha >= 1:
                return new
            old = self._interp(*chunks[-2], t)
            return new if old is None else old + (new - old) * alpha
        values, weights = [], []
        for i, (t0, acts) in enumerate(chunks):  # i = 0 is the oldest chunk
            f = (t - t0) / self.dt
            if f < 0 or f > self.k - 1:
                continue
            lo = int(np.floor(f))
            hi = min(lo + 1, self.k - 1)
            a = acts[lo] + (acts[hi] - acts[lo]) * (f - lo)
            values.append(a)
            weights.append(np.exp(-self.m * i))  # ACT convention: oldest chunk gets weight 1
        if not values:
            return None
        w = np.asarray(weights)
        return (np.stack(values) * w[:, None]).sum(0) / w.sum()


class SafetyFilter:
    """Clamp targets to the range seen in the demonstrations (+margin) and limit the speed."""

    def __init__(
        self,
        a_min,
        a_max,
        margin_rad: float = 0.15,
        max_joint_speed: float = 1.5,
        max_gripper_speed: float = 3.0,
        max_grip_squeeze: float = 0.1,
    ):
        # gripper target may be at most `max_grip_squeeze` narrower than the measured opening.
        # Holding an object with a large squeeze for a long time overheated the right gripper motor
        # (DM4310 "motor over temperature", 2026-10-27). Demos squeeze ~0.19 but only for ~6 s.
        self.max_grip_squeeze = max_grip_squeeze
        self.squeeze_limited = 0
        self.lo = np.asarray(a_min, dtype=np.float64).copy()
        self.hi = np.asarray(a_max, dtype=np.float64).copy()
        joints = np.r_[0:6, 7:13]
        self.lo[joints] -= margin_rad
        self.hi[joints] += margin_rad
        self.lo[[6, 13]] = 0.0
        self.hi[[6, 13]] = 1.0
        self.speed = np.full(14, max_joint_speed)
        self.speed[[6, 13]] = max_gripper_speed
        self.last: np.ndarray | None = None
        self.clamped = 0
        self.rate_limited = 0

    def reset(self, current: np.ndarray) -> None:
        self.last = np.asarray(current, dtype=np.float64).copy()

    def __call__(
        self, target: np.ndarray, dt: float, current: np.ndarray | None = None
    ) -> np.ndarray:
        x = np.clip(target, self.lo, self.hi)
        if not np.allclose(x, target):
            self.clamped += 1
        if current is not None:
            for g in (6, 13):
                floor = current[g] - self.max_grip_squeeze
                if x[g] < floor:
                    x[g] = floor
                    self.squeeze_limited += 1
        if self.last is not None:
            step = self.speed * max(dt, 1e-3)
            y = np.clip(x, self.last - step, self.last + step)
            if not np.allclose(x, y):
                self.rate_limited += 1
            x = y
        self.last = x
        return x
