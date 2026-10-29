"""Pairing an image with the follower state at the image time (control/state_history.py)."""

from __future__ import annotations

import numpy as np
import pytest

from yam_infer.control.state_history import StateHistory


def _filled(n: int = 100, dt: float = 0.01) -> StateHistory:
    h = StateHistory(seconds=1.0, rate_hz=100)
    for i in range(n):  # joint value == time -> easy to check interpolation
        h.add(i * dt, np.full(14, i * dt))
    return h


def test_interpolates_between_neighbouring_readings() -> None:
    h = _filled()
    np.testing.assert_allclose(h.at(0.505), 0.505)
    np.testing.assert_allclose(h.at(0.5), 0.5)


def test_clamps_outside_the_stored_range() -> None:
    h = _filled()
    np.testing.assert_allclose(h.at(-1.0), 0.0)
    np.testing.assert_allclose(h.at(5.0), 0.99)


def test_keeps_only_the_last_second_and_increasing_times() -> None:
    h = _filled(n=250)
    np.testing.assert_allclose(h.at(0.0), 1.5)  # oldest kept reading (250 - 100) * 10 ms
    h.add(1.0, np.zeros(14))  # out-of-order sample is ignored
    np.testing.assert_allclose(h.at(10.0), 2.49)


def test_empty_history_raises() -> None:
    with pytest.raises(RuntimeError):
        StateHistory().at(0.0)
