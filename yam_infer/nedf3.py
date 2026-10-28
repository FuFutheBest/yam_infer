"""Read one NEDF3 / MCAP episode recorded by SuperInference I2RT DM3.

Only depends on `mcap`, `mcap-protobuf-support`, `numpy` and `opencv` (schemas are embedded in the file).

Layout of 14-D vectors everywhere in this project (same as the recorder's SHM contract):
    [0:6]  left joint0..5 (rad)   [6]  left gripper (0 closed .. 1 open)
    [7:13] right joint0..5 (rad)  [13] right gripper
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SIDES = ("left", "right")
ROLES = ("leader", "follower")
CAMERAS = {"right_wrist": "nmx/hal/camera/camera_0/rgbd", "left_wrist": "nmx/hal/camera/camera_1/rgbd"}


@dataclass
class RawEpisode:
    path: Path
    task: str
    task_description: str
    # role -> side -> (timestamps_ns[N], values[N, 6])
    joints: dict = field(default_factory=dict)
    # role -> side -> (timestamps_ns[N], values[N])
    grippers: dict = field(default_factory=dict)
    # camera name -> (timestamps_ns[N], list[jpeg bytes])
    cameras: dict = field(default_factory=dict)


def _mcap_files(episode_dir: Path) -> list[Path]:
    index = episode_dir / "data" / "mcap_index.json"
    if index.exists():
        try:
            entries = json.loads(index.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"bad mcap index {index}: {exc}") from exc
        return [episode_dir / "data" / e["filename"] for e in entries]
    return sorted((episode_dir / "data").glob("*.mcap"))


def read_episode(episode_dir: str | Path, load_images: bool = True) -> RawEpisode:
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    episode_dir = Path(episode_dir)
    try:
        meta = json.loads((episode_dir / "metadata.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"bad metadata in {episode_dir}: {exc}") from exc
    ep = RawEpisode(episode_dir, meta.get("task_name", ""), meta.get("task_description", ""))

    joint_buf = {(r, s): ([], []) for r in ROLES for s in SIDES}
    grip_buf = {(r, s): ([], []) for r in ROLES for s in SIDES}
    cam_buf = {name: ([], []) for name in CAMERAS}
    topic_map = {}
    for r in ROLES:
        for s in SIDES:
            topic_map[f"nmx/hal/i2rt/{r}/{s}/joint"] = ("joint", (r, s))
            topic_map[f"nmx/hal/i2rt/{r}/{s}/hand_state"] = ("grip", (r, s))
    if load_images:
        for name, topic in CAMERAS.items():
            topic_map[topic] = ("cam", name)

    for f in _mcap_files(episode_dir):
        if not f.is_file():
            raise FileNotFoundError(f"missing MCAP shard {f}")
        try:
            fh = open(f, "rb")  # noqa: SIM115
        except OSError as exc:
            raise RuntimeError(f"cannot open MCAP shard {f}: {exc}") from exc
        with fh:
            reader = make_reader(fh, decoder_factories=[DecoderFactory()])
            for _schema, channel, message, decoded in reader.iter_decoded_messages(
                topics=list(topic_map)
            ):
                kind, key = topic_map[channel.topic]
                t = int(message.log_time)
                if kind == "joint":
                    pos = {j.name: j.pos for j in decoded.joints}
                    joint_buf[key][0].append(t)
                    joint_buf[key][1].append([pos[f"joint{i}"] for i in range(6)])
                elif kind == "grip":
                    grip_buf[key][0].append(t)
                    grip_buf[key][1].append(decoded.joints[0].pos)
                else:
                    cam_buf[key][0].append(t)
                    cam_buf[key][1].append(bytes(decoded.rgb.data))

    def _sorted(ts, vals, as_array=True):
        order = np.argsort(np.asarray(ts, dtype=np.int64), kind="stable")
        ts = np.asarray(ts, dtype=np.int64)[order]
        if as_array:
            return ts, np.asarray(vals, dtype=np.float64)[order]
        return ts, [vals[i] for i in order]

    for (r, s), (ts, vals) in joint_buf.items():
        ep.joints.setdefault(r, {})[s] = _sorted(ts, vals)
    for (r, s), (ts, vals) in grip_buf.items():
        ep.grippers.setdefault(r, {})[s] = _sorted(ts, vals)
    for name, (ts, vals) in cam_buf.items():
        if ts:
            ep.cameras[name] = _sorted(ts, vals, as_array=False)
    return ep


def interp(ts_src: np.ndarray, values: np.ndarray, ts_query: np.ndarray) -> np.ndarray:
    """Linear interpolation of (N,) or (N,D) values; clamps outside the recorded range."""
    x = (ts_src - ts_src[0]).astype(np.float64)
    q = (ts_query - ts_src[0]).astype(np.float64)
    if values.ndim == 1:
        return np.interp(q, x, values)
    return np.stack([np.interp(q, x, values[:, d]) for d in range(values.shape[1])], axis=1)


def state14(ep: RawEpisode, role: str, ts_query: np.ndarray) -> np.ndarray:
    """Return the (T,14) state of `role` ('leader' or 'follower') sampled at ts_query."""
    parts = []
    for s in SIDES:
        jt, jv = ep.joints[role][s]
        gt, gv = ep.grippers[role][s]
        parts.append(interp(jt, jv, ts_query))
        parts.append(interp(gt, gv, ts_query)[:, None])
    return np.concatenate(parts, axis=1)


def decode_jpeg_rgb(data: bytes) -> np.ndarray:
    """JPEG bytes -> RGB uint8 (H,W,3). The recorder stored true RGB colors (v2 contract)."""
    import cv2

    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("could not decode JPEG frame")
    return np.ascontiguousarray(np.asarray(bgr)[:, :, ::-1])
