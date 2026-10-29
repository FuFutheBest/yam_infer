"""Load a trained checkpoint and turn (images, qpos, task) into an action chunk in real units.

Two checkpoint formats (detected from the file):

| format                 | trained by                     | network              | tasks | images  | chunk |
|------------------------|--------------------------------|----------------------|-------|---------|-------|
| act_v2 (no `format`)   | opus_inference_train (old)     | policy/act.py        | many  | 240x320 | 50    |
| `yam_train/act_ref-v1` | yam_train (tonyzhaozh/act ref) | policy/detr (source) | one   | 480x640 | 100   |

Preprocessing MUST match training:
    D405 RGB 480x640 -> JPEG q90 round-trip (the recorder's encoder) -> INTER_AREA resize to the
    model's image_size (act_ref: no resize, the source uses the recorded frames as they are)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from yam_infer.data.nedf3_writer import encode_jpeg_rgb


def preprocess_frame(
    rgb: np.ndarray, size_hw: tuple[int, int], jpeg_roundtrip: bool = True
) -> np.ndarray:
    """RGB uint8 (480,640,3) -> RGB uint8 (H,W,3) exactly like the training pipeline.

    The JPEG round-trip uses the recorder's own encoder, so decode(recorded JPEG) == the image
    the model saw before resizing (tools/replay_policy.py calls this with jpeg_roundtrip=False).
    """
    if jpeg_roundtrip:  # the training images went through the recorder's JPEG q90 encoder
        decoded = cv2.imdecode(np.frombuffer(encode_jpeg_rgb(rgb), np.uint8), cv2.IMREAD_COLOR)
        if decoded is not None:
            rgb = np.asarray(decoded)[:, :, ::-1]
    h, w = size_hw
    if rgb.shape[:2] == (h, w):
        return np.ascontiguousarray(rgb)
    return cv2.resize(np.ascontiguousarray(rgb), (w, h), interpolation=cv2.INTER_AREA)


class Policy:
    def __init__(self, ckpt_path: str | Path, device: str = "auto", jpeg_roundtrip: bool = True):
        ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.format = str(ck.get("format", "act_v2"))
        if self.format == "act_v2":
            self._load_act_v2(ck)
        elif self.format == "yam_train/act_ref-v1":
            self._load_act_ref(ck)
        else:
            raise ValueError(f"{ckpt_path}: unknown checkpoint format {self.format!r}")
        self.net.load_state_dict(ck["model"])
        self.net.to(self.device).eval()
        self.last_raw: np.ndarray | None = None  # set by every predict
        self.jpeg_roundtrip = jpeg_roundtrip

    # ------------------------------------------------------------------ loaders
    def _load_act_v2(self, ck: dict) -> None:
        from yam_infer.policy.act import ACT, ACTConfig

        cfg = ACTConfig(**{**ck["config"], "pretrained_backbone": False})
        self.net = ACT(cfg)
        self.tasks: list[str] = list(ck["tasks"])
        self.cameras: list[str] = list(ck["cameras"])  # ["right_wrist", "left_wrist"]
        self.image_size = tuple(ck["image_size"])
        self.fps = float(ck["fps"])
        self.chunk_size = int(cfg.chunk_size)
        self.start_pose = {k: np.asarray(v) for k, v in ck["start_pose"].items()}
        st = ck["stats"]
        self.q_mean, self.q_std = np.asarray(st["qpos"]["mean"]), np.asarray(st["qpos"]["std"])
        self.action_mode = ck.get("action_mode", "abs")
        norm = ck.get("action_norm", st["action"])  # stats of the network output (abs or delta)
        self.a_mean, self.a_std = np.asarray(norm["mean"]), np.asarray(norm["std"])
        # absolute leader range seen in the demos -> used by the safety filter
        self.a_min, self.a_max = np.asarray(st["action"]["min"]), np.asarray(st["action"]["max"])
        self.action_shift = 0
        self.step = ck.get("step")

    def _load_act_ref(self, ck: dict) -> None:
        from yam_infer.policy.act_ref import ACTRefPolicy

        self.cameras = list(ck["camera_names"])
        self.net = ACTRefPolicy(ck["config"]["policy"], self.cameras)
        self.tasks = [str(ck["task"])]
        self.image_size = tuple(ck["image_size"])
        self.fps = float(ck["fps"])
        self.chunk_size = int(ck["chunk_size"])
        self.start_pose = {self.tasks[0]: np.asarray(ck["start_pose"])}
        st = {k: np.asarray(v, dtype=np.float64) for k, v in ck["stats"].items()}
        self.q_mean, self.q_std = st["qpos_mean"], st["qpos_std"]
        self.a_mean, self.a_std = st["action_mean"], st["action_std"]
        self.a_min, self.a_max = st["leader_min"], st["leader_max"]
        self.action_mode = ck.get("action_mode", "abs")
        # chunk step 0 = leader(t - shift) for the observation at t (source real-robot branch)
        self.action_shift = int(ck.get("action_shift", 0))
        self.step = ck.get("epoch")

    # ------------------------------------------------------------------ api
    def info(self) -> dict:
        return {
            "format": self.format,
            "tasks": self.tasks,
            "cameras": self.cameras,
            "image_size": list(self.image_size),
            "fps": self.fps,
            "chunk_size": self.chunk_size,
            "train_step": self.step,
            "device": str(self.device),
            "action_mode": self.action_mode,
            "action_shift": self.action_shift,
            "start_pose": {k: v.tolist() for k, v in self.start_pose.items()},
            "action_min": self.a_min.tolist(),
            "action_max": self.a_max.tolist(),
        }

    def prepare(self, frames: dict[str, np.ndarray]) -> np.ndarray:
        """Raw camera frames -> (n_cam, H, W, 3) uint8 model input."""
        return np.stack(
            [
                preprocess_frame(frames[c], self.image_size, self.jpeg_roundtrip)
                for c in self.cameras
            ]
        )

    @torch.no_grad()
    def predict_prepared(self, images: np.ndarray, qpos: np.ndarray, task: str) -> np.ndarray:
        """images (n_cam,H,W,3) uint8, qpos (14,) -> actions (K,14) in rad / gripper 0..1."""
        if task not in self.tasks:
            raise ValueError(f"this model was trained for {self.tasks}, not {task!r}")
        img = torch.from_numpy(images).to(self.device).permute(0, 3, 1, 2).float().div_(255.0)[None]
        q = torch.from_numpy(((qpos - self.q_mean) / self.q_std).astype(np.float32)).to(
            self.device
        )[None]
        if self.format == "act_v2":
            t = torch.tensor([self.tasks.index(task)], device=self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                pred, _, _ = self.net(img, q, t)
        else:  # act_ref: float32 like the source
            pred = self.net(q, img)
        raw = pred[0].float().cpu().numpy()  # normalised network output (K, 14)
        self.last_raw = raw
        return self.to_absolute(raw, qpos)

    def to_absolute(self, pred_norm: np.ndarray, qpos: np.ndarray) -> np.ndarray:
        """normalised network output (..., K, 14) -> absolute joint targets."""
        act = pred_norm * self.a_std + self.a_mean
        if self.action_mode == "delta":
            act = act + np.asarray(qpos)[..., None, :]
        return act

    def predict(
        self, frames: dict[str, np.ndarray], qpos: np.ndarray, task: str
    ) -> tuple[np.ndarray, np.ndarray]:
        images = self.prepare(frames)
        return self.predict_prepared(images, qpos, task), images
