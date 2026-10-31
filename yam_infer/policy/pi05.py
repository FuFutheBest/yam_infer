"""pi0.5 models (yam_train pi05 branch): the network runs in openpi's own environment as a policy server,
this process is the client.

    model server (openpi venv, GPU):   python -m yam_train.pi05.cli serve --model models/pi05_yam --port 8000
    robot:                             configs/models.yaml  format: pi05, server: <host>:8000

Why a server: pi0.5 is a 3.3B-parameter JAX model with its own pinned dependencies (jax 0.5.3,
numpy<2 ...) that cannot live in this project's environment (numpy 2.2.6 for i2rt).  This is openpi's own
deployment pattern (openpi docs/remote_inference.md); the server may run on the robot PC (localhost)
or on another GPU machine.

Protocol = openpi's websocket policy server (msgpack + numpy arrays), see _msgpack_* below:
    on connect      server -> metadata (yam_model.json of the exported model: fps, cameras, prompts,
                    action_min/max, start_pose, action_horizon, ...)
    per inference   client -> {"images": {cam: uint8 (224,224,3)}, "state": float32 (14,), "prompt": str}
                    server -> {"actions": (50, 14) absolute targets, "policy_timing": ..., "server_timing": ...}

Image preprocessing = training (yam_train.pi05.to_lerobot): recorded JPEG (here: JPEG q90 round-trip
with the recorder's encoder) -> RGB -> resize_with_pad 224x224 (openpi_client.image_tools, PIL bilinear).
The server applies the delta->absolute decoding (joints + current state), so the chunk is absolute.
"""

from __future__ import annotations

import contextlib
import time

import cv2
import numpy as np

from yam_infer.data.nedf3_writer import encode_jpeg_rgb

FORMAT = "yam_train/pi05-v1"


# ---- openpi_client.msgpack_numpy (openpi @215abfb, Apache-2.0), verbatim logic; see VENDORED.md
def _pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def _pack(obj) -> bytes:
    import msgpack

    data = msgpack.packb(obj, default=_pack_array)
    if not isinstance(data, bytes):
        raise TypeError("msgpack.packb returned no bytes")
    return data


def _unpack(data: bytes):
    import msgpack

    return msgpack.unpackb(data, object_hook=_unpack_array)


# ---- openpi_client.image_tools.resize_with_pad (openpi @215abfb, Apache-2.0), same arithmetic
def resize_with_pad(rgb: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize keeping the aspect ratio (PIL bilinear), pad with black to (height, width)."""
    from PIL import Image

    if rgb.shape[:2] == (height, width):
        return rgb
    image = Image.fromarray(rgb)
    cur_width, cur_height = image.size
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized = image.resize((resized_width, resized_height), resample=Image.Resampling.BILINEAR)
    canvas = Image.new(resized.mode, (width, height), 0)
    canvas.paste(resized, (max(0, int((width - resized_width) / 2)), max(0, int((height - resized_height) / 2))))
    return np.asarray(canvas)


def preprocess_frame_pi05(rgb: np.ndarray, size_hw: tuple[int, int], jpeg_roundtrip: bool = True) -> np.ndarray:
    """D405 RGB uint8 (480,640,3) -> model image uint8 (224,224,3), exactly like yam_train.pi05.to_lerobot."""
    if jpeg_roundtrip:  # the training frames are the recorder's JPEGs, decoded with OpenCV
        decoded = cv2.imdecode(np.frombuffer(encode_jpeg_rgb(rgb), np.uint8), cv2.IMREAD_COLOR)
        if decoded is not None:
            rgb = np.asarray(decoded)[:, :, ::-1]
    return np.ascontiguousarray(resize_with_pad(np.ascontiguousarray(rgb), *size_hw))


class Pi05Policy:
    """Same interface as yam_infer.policy.policy.Policy; the model runs in an openpi policy server."""

    def __init__(self, address: str, jpeg_roundtrip: bool = True, connect_timeout_s: float = 120.0):
        import websockets.sync.client

        self._pack, self._unpack = _pack, _unpack
        uri = address if address.startswith("ws") else f"ws://{address}"
        deadline = time.monotonic() + connect_timeout_s
        while True:  # the server needs ~1 min to load the model after start
            try:
                conn = websockets.sync.client.connect(uri, compression=None, max_size=None, open_timeout=10)
                break
            except (ConnectionRefusedError, TimeoutError, OSError) as exc:
                if time.monotonic() > deadline:
                    raise ConnectionError(f"no pi05 policy server at {uri}: {exc}") from exc
                print(f"waiting for the pi05 policy server at {uri} ...", flush=True)
                time.sleep(3)
        self._stack = contextlib.ExitStack()  # websockets wants the connection used as a context manager
        self._ws = self._stack.enter_context(conn)
        first = self._ws.recv()
        if isinstance(first, str):
            raise ConnectionError(f"{uri} did not send model metadata: {first[:200]}")
        meta = self._unpack(first)
        if not isinstance(meta, dict) or meta.get("format") != FORMAT:
            got = meta.get("format") if isinstance(meta, dict) else type(meta).__name__
            raise ValueError(f"{uri} serves {got!r}, expected {FORMAT!r}")
        self.meta = meta
        self.address = address
        self.format = FORMAT
        self.jpeg_roundtrip = jpeg_roundtrip
        self.prompts: dict[str, str] = dict(meta["prompts"])
        self.tasks = list(self.prompts)
        self.cameras = list(meta["cameras"])
        self.image_size = tuple(meta["image_hw"])
        self.fps = float(meta["fps"])
        self.chunk_size = int(meta["action_horizon"])
        self.start_pose = {k: np.asarray(v) for k, v in meta["start_pose"].items()}
        self.a_min, self.a_max = np.asarray(meta["action_min"]), np.asarray(meta["action_max"])
        # the network predicts joint deltas (+ absolute grippers); the server returns absolute targets
        self.action_mode = "delta"
        self.action_shift = 0  # chunk step 0 = leader at the observation time (same as training)
        self.step = meta.get("step")
        self.last_raw: np.ndarray | None = None
        self.last_timing: dict = {}

    def info(self) -> dict:
        return {
            "format": self.format,
            "tasks": self.tasks,
            "prompts": self.prompts,
            "cameras": self.cameras,
            "image_size": list(self.image_size),
            "fps": self.fps,
            "chunk_size": self.chunk_size,
            "train_step": self.step,
            "device": f"pi05 server {self.address}",
            "action_mode": self.action_mode,
            "action_shift": self.action_shift,
            "start_pose": {k: v.tolist() for k, v in self.start_pose.items()},
            "action_min": self.a_min.tolist(),
            "action_max": self.a_max.tolist(),
            "server_model": {k: self.meta.get(k) for k in ("config", "experiment", "step", "episodes")},
        }

    def prepare(self, frames: dict[str, np.ndarray]) -> np.ndarray:
        return np.stack([preprocess_frame_pi05(frames[c], self.image_size, self.jpeg_roundtrip)
                         for c in self.cameras])

    def predict_prepared(self, images: np.ndarray, qpos: np.ndarray, task: str) -> np.ndarray:
        if task not in self.prompts:
            raise ValueError(f"this model knows the tasks {self.tasks}, not {task!r}")
        obs = {
            "images": {c: images[i] for i, c in enumerate(self.cameras)},
            "state": np.asarray(qpos, dtype=np.float32),
            "prompt": self.prompts[task],
        }
        self._ws.send(self._pack(obs))
        reply = self._ws.recv()
        if isinstance(reply, str):  # the server sends its traceback as text, then closes
            raise RuntimeError(f"pi05 policy server error:\n{reply}")
        out = self._unpack(reply)
        actions = np.asarray(out["actions"], dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != 14:
            raise RuntimeError(f"pi05 server returned actions of shape {actions.shape}, expected (H, 14)")
        self.last_timing = {**out.get("policy_timing", {}), **out.get("server_timing", {})}
        return actions

    def predict(self, frames: dict[str, np.ndarray], qpos: np.ndarray, task: str):
        images = self.prepare(frames)
        return self.predict_prepared(images, qpos, task), images

    def close(self) -> None:
        self._stack.close()
