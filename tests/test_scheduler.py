"""Action scheduler + safety filter (pure numpy)."""

from __future__ import annotations

import numpy as np

from yam_infer.control.scheduler import ActionScheduler, SafetyFilter


def _at(s: ActionScheduler, t: float) -> float:
    target = s.target_at(t)
    assert target is not None
    return float(target[0])


def _chunk(k: int = 50, start: float = 0.0, step: float = 0.01) -> np.ndarray:
    return np.tile((start + step * np.arange(k))[:, None], (1, 14))


def test_chunk_mode_commits_for_horizon_then_asks_for_a_new_chunk() -> None:
    s = ActionScheduler(fps=30, chunk_size=50, mode="chunk", horizon=30)
    assert s.needs_new_chunk(0.0)
    s.add(0.0, _chunk())
    assert not s.needs_new_chunk(0.5)
    assert s.needs_new_chunk(29 / 30)
    np.testing.assert_allclose(_at(s, 10 / 30), 0.10, atol=1e-9)
    np.testing.assert_allclose(_at(s, 10.5 / 30), 0.105, atol=1e-9)  # interpolated


def test_chunk_switch_is_cross_faded() -> None:
    s = ActionScheduler(fps=30, chunk_size=50, mode="chunk", horizon=30, blend_s=0.1)
    s.add(0.0, _chunk(start=0.0))
    s.add(1.0, _chunk(start=1.0))
    old, new = 0.0 + 0.01 * 30 * 1.05, 1.0 + 0.01 * 30 * 0.05
    mid = _at(s, 1.05)
    assert min(old, new) < mid < max(old, new)
    np.testing.assert_allclose(_at(s, 1.2), 1.0 + 0.01 * 6, atol=1e-9)


def test_safety_filter_range_and_speed() -> None:
    f = SafetyFilter(np.zeros(14), np.ones(14), margin_rad=0.1, max_joint_speed=1.0)
    cur = np.full(14, 0.5)
    f.reset(cur)
    out = f(np.full(14, 5.0), dt=0.01)
    assert np.all(out[np.r_[0:6, 7:13]] <= 0.5 + 0.01 + 1e-9)  # rate limited
    assert f.clamped == 1 and f.rate_limited == 1
    f.reset(cur)
    target = cur.copy()
    target[13] = 0.0  # close the right gripper fully: follows the model (only speed-limited)
    out = f(target, dt=1.0)
    assert out[13] == 0.0
