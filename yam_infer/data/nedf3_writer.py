"""Write NEDF3-compatible MCAP files (same topics / protobuf schemas as SuperInference's I2RT recorder).

Schemas come from yam_infer/data/nedf3_schemas/*.desc (copied out of an SI recording by
scripts/extract_nedf3_schemas.py), so no private nmx_msg / dm3_core package is needed.  Recordings
written here can be read by yam_infer.data.nedf3.read_episode, the training-side converter and
SI's viewers.

Differences to SI's recorder (all inside the same schema):
    camera  rgb.header.seq   = our per-camera frame counter (policy.npz `cam_seq` points at it)
            rgb.header.stamp = RealSense hardware timestamp [ns, camera clock] (SI: always 0)
            rgb.header.frame_id = camera name
    joints / gripper at the control rate (~100 Hz; SI: arm 90 Hz, gripper 30 Hz)
    no leader topics (no human input during inference)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from mcap.writer import CompressionType, Writer

SCHEMA_DIR = Path(__file__).resolve().parent / "nedf3_schemas"
METADATA_TOPIC = "nmx/nedf/metadata"
CAMERA_TOPICS = {  # SI contract: camera_0 = right wrist, camera_1 = left wrist
    "right_wrist": "nmx/hal/camera/camera_0/rgbd",
    "left_wrist": "nmx/hal/camera/camera_1/rgbd",
}
JOINT_TOPICS = {s: f"nmx/hal/i2rt/follower/{s}/joint" for s in ("left", "right")}
HAND_TOPICS = {s: f"nmx/hal/i2rt/follower/{s}/hand_state" for s in ("left", "right")}
CONTRACT_VERSION = "i2rt-nedf3-v2"  # the SI contract these files follow
JPEG_QUALITY = 90


def encode_jpeg_rgb(rgb: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    """RGB uint8 -> JPEG bytes, exactly like SI's recorder (RGB->BGR, OpenCV, q90).

    The policy's JPEG round-trip uses this same function, so the recorded JPEG of a frame is
    byte-identical to what the model saw before resizing (replay_policy relies on that).
    """
    ok, buf = cv2.imencode(
        ".jpg", np.ascontiguousarray(rgb[:, :, ::-1]), [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    )
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def _message_classes() -> dict[str, tuple[Any, bytes]]:
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

    classes: dict[str, tuple[Any, bytes]] = {}
    for name in ("nmx.msg.Metadata", "nmx.msg.ArmState", "nmx.msg.HandState", "nmx.msg.RGBD"):
        data = (SCHEMA_DIR / f"{name}.desc").read_bytes()
        pool = descriptor_pool.DescriptorPool()
        for fd in descriptor_pb2.FileDescriptorSet.FromString(data).file:
            pool.Add(fd)
        classes[name] = (message_factory.GetMessageClass(pool.FindMessageTypeByName(name)), data)
    return classes


class Nedf3Writer:
    """One episode: <dir>/metadata.json + <dir>/data/data_<start_ns>_0.mcap. Not thread-safe."""

    def __init__(self, episode_dir: Path, metadata: dict, start_wall_ns: int):
        self.dir = Path(episode_dir)
        (self.dir / "data").mkdir(parents=True, exist_ok=True)
        self.metadata = metadata
        (self.dir / "metadata.json").write_text(json.dumps(metadata, indent=1, ensure_ascii=False))
        self._cls = _message_classes()
        path = self.dir / "data" / f"data_{start_wall_ns}_0.mcap"
        try:
            self._fh = open(path, "wb")  # noqa: SIM115 - closed in close()
        except OSError as exc:
            raise RuntimeError(f"cannot create {path}: {exc}") from exc
        self._w = Writer(self._fh, chunk_size=256 * 1024, compression=CompressionType.LZ4)
        self._w.start(profile="", library="yam_infer")
        sid = {
            name: self._w.register_schema(name=name, encoding="protobuf", data=data)
            for name, (_cls, data) in self._cls.items()
        }
        self._ch: dict[str, int] = {}
        for topic, schema in (
            [(METADATA_TOPIC, "nmx.msg.Metadata")]
            + [(t, "nmx.msg.RGBD") for t in CAMERA_TOPICS.values()]
            + [(t, "nmx.msg.ArmState") for t in JOINT_TOPICS.values()]
            + [(t, "nmx.msg.HandState") for t in HAND_TOPICS.values()]
        ):
            self._ch[topic] = self._w.register_channel(
                topic=topic, message_encoding="protobuf", schema_id=sid[schema]
            )
        self.counts = dict.fromkeys(self._ch, 0)
        meta_msg = self._cls["nmx.msg.Metadata"][0](
            version=CONTRACT_VERSION, metadata=json.dumps(metadata, sort_keys=True).encode()
        )
        self._put(METADATA_TOPIC, start_wall_ns, meta_msg)

    def _put(self, topic: str, wall_ns: int, msg: Any) -> None:
        n = self.counts[topic]
        self._w.add_message(
            self._ch[topic],
            log_time=wall_ns,
            publish_time=wall_ns,
            data=msg.SerializeToString(),
            sequence=n,
        )
        self.counts[topic] = n + 1

    def write_camera(
        self,
        name: str,
        wall_ns: int,
        jpeg: bytes,
        shape: tuple[int, ...],
        seq: int,
        hw_stamp_ns: int = 0,
    ) -> None:
        msg = self._cls["nmx.msg.RGBD"][0]()
        msg.rgb.header.seq = int(seq)
        msg.rgb.header.stamp = max(0, int(hw_stamp_ns))
        msg.rgb.header.frame_id = name
        msg.rgb.rows, msg.rgb.cols, msg.rgb.channels = int(shape[0]), int(shape[1]), 3
        msg.rgb.type = 1  # ImageDataType.ENCODED (as SI writes it)
        msg.rgb.encoded_format = "jpeg"
        msg.rgb.data = jpeg
        msg.depth.type = 1
        msg.depth_scale = 1.0
        self._put(CAMERA_TOPICS[name], wall_ns, msg)

    def write_state(self, wall_ns: int, state14: np.ndarray) -> None:
        """One follower reading [left 6 joints, left grip, right 6 joints, right grip]."""
        for i, side in enumerate(("left", "right")):
            part = state14[i * 7 : (i + 1) * 7]
            arm = self._cls["nmx.msg.ArmState"][0]()
            for j in range(6):
                joint = arm.joints.add()
                joint.name, joint.pos = f"joint{j}", float(part[j])
            self._put(JOINT_TOPICS[side], wall_ns, arm)
            hand = self._cls["nmx.msg.HandState"][0]()
            joint = hand.joints.add()
            joint.name, joint.pos = "gripper", float(part[6])
            self._put(HAND_TOPICS[side], wall_ns, hand)

    def close(self) -> None:
        try:
            self._w.finish()
        finally:
            self._fh.close()
