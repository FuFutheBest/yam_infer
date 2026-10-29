"""Recorder: NEDF3-compatible raw streams, frame references, finalisation / quarantine."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from yam_infer.data.nedf3 import decode_jpeg_rgb, read_episode, state14
from yam_infer.data.nedf3_writer import encode_jpeg_rgb
from yam_infer.data.recorder import RunRecorder

CAMS = ["right_wrist", "left_wrist"]


def _recorder(root: Path) -> RunRecorder:
    return RunRecorder(root, "put_banana_in_bowl", {"policy": {"cameras": CAMS}}, fps=30)


def _fill(rec: RunRecorder, frames: bool = True) -> np.ndarray:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
    t0 = time.monotonic()
    for i in range(30):
        t = t0 + i / 100
        q = np.full(14, i / 100.0)
        rec.log_command(t, q + 0.1, q + 0.05, q, np.array([t - 0.003, t - 0.002]))
        if frames and i % 3 == 0:
            for c in CAMS:
                rec.camera_sink(c, t, img, i // 3 + 1, 123_000_000 + i)
    rec.log_policy(
        t0,
        np.zeros(14),
        np.zeros((50, 14)),
        cam_t=[t0, t0],
        cam_seq=[1, 1],
        t_infer=t0 + 0.02,
        t_done=t0 + 0.03,
        raw_out=np.ones((50, 14)),
    )
    rec.event("start", t0)
    return img


def test_episode_is_nedf3_readable_and_frames_are_referenced(tmp_path: Path) -> None:
    rec = _recorder(tmp_path)
    img = _fill(rec)
    out = rec.close(end_reason="test")
    assert out is not None and out == tmp_path / rec.name and out.is_dir()
    assert not list(tmp_path.glob("*_inprogress"))

    ep = read_episode(out)  # the same reader the dry-run / training side uses
    assert ep.task == "put_banana_in_bowl"
    assert len(ep.cameras["right_wrist"][1]) == 10
    assert len(ep.joints["follower"]["right"][0]) == 30
    st = state14(ep, "follower", ep.joints["follower"]["left"][0])
    np.testing.assert_allclose(st[:, 0], np.arange(30) / 100, atol=1e-9)
    # stored JPEG == the policy's own round-trip encoding of the frame
    assert ep.cameras["right_wrist"][1][0] == encode_jpeg_rgb(img)
    assert decode_jpeg_rgb(ep.cameras["right_wrist"][1][0]).shape == img.shape

    pol = np.load(out / "policy.npz")
    assert pol["cam_seq"].tolist() == [[1, 1]] and abs(pol["latency_ms"][0] - 10.0) < 1e-6
    stats = json.loads((out / "episode_stats.json").read_text())
    assert stats["dropped_camera_frames"] == 0 and stats["writer_error"] is None
    assert json.loads((out / "events.jsonl").read_text().splitlines()[0])["event"] == "start"


def test_missing_stream_goes_to_quarantine(tmp_path: Path) -> None:
    rec = _recorder(tmp_path)
    _fill(rec, frames=False)
    out = rec.close(end_reason="test")
    assert out is not None and out.parent == tmp_path / "quarantine"
    reason = json.loads((out / "quarantine_reason.json").read_text())
    assert "nmx/hal/camera/camera_0/rgbd" in reason["missing_topics"]


def test_empty_episode_leaves_nothing_and_crashed_one_is_recovered(tmp_path: Path) -> None:
    rec = _recorder(tmp_path)
    assert rec.close(end_reason="quit before start") is None
    assert not any(p for p in tmp_path.iterdir() if p.name != "quarantine")

    crashed = tmp_path / "20260101_000000_put_banana_in_bowl_inprogress"
    crashed.mkdir()
    rec = _recorder(tmp_path)  # next start moves it away
    assert (tmp_path / "quarantine" / "20260101_000000_put_banana_in_bowl_interrupted").is_dir()
    rec.close()
