# pyright: reportMissingImports=false
# (openpi exists only in the openpi venv this module runs in, not in this project's .venv)
"""Data mapping YAM robot <-> pi0.5 model (openpi DataTransformFn).

Copied from yam_train/pi05/yam_policy.py (yam_train pi05 @54588b2), serving part only; must stay
identical to what the model was trained with (see VENDORED.md).

Robot side (what yam_infer.policy.pi05 sends):
    images/right_wrist   uint8 (224, 224, 3)  right wrist D405, resize_with_pad from 480x640
    images/left_wrist    uint8 (224, 224, 3)  left wrist D405 (an overview of the table)
    state                float32 (14,)        follower [left j0..5, left grip, right j0..5, right grip]
    prompt               str

pi0.5 image slots: left wrist camera -> base_0_rgb, right wrist -> right_wrist_0_rgb,
left_wrist_0_rgb empty (black, mask False).
"""

from __future__ import annotations

import dataclasses

import numpy as np
from openpi import transforms

MODEL_IMAGE_HW = (224, 224)
CAMERAS = ("right_wrist", "left_wrist")


def _mask(*segments: int) -> tuple[bool, ...]:
    """openpi transforms.make_bool_mask: positive n -> n x True, negative n -> |n| x False."""
    out: list[bool] = []
    for n in segments:
        out.extend([n > 0] * abs(n))
    return tuple(out)


# which action dims the network predicts relative to the current state (delta), per training config:
#   pi05_yam*  (block_bowl + blue_cup): right arm joints delta, left joints + grippers absolute
#   pi05_yam3* (+ bimanual tasks):      both arms' joints delta, grippers absolute
DELTA_MASK = _mask(-7, 6, -1)
DELTA_MASK_BOTH = _mask(6, -1, 6, -1)


def delta_mask_for(meta: dict) -> tuple[bool, ...]:
    """Delta mask of an exported model: yam_model.json `delta_joints` (newer exports) or by config name."""
    if "delta_joints" in meta:
        joints = set(meta["delta_joints"])
        return tuple(i in joints for i in range(14))
    name = meta["config"]
    if name.startswith("pi05_yam3"):
        return DELTA_MASK_BOTH
    if name.startswith("pi05_yam"):
        return DELTA_MASK
    raise ValueError(f"unknown pi05 config {name!r} and yam_model.json has no delta_joints")


def _hwc_uint8(img) -> np.ndarray:
    img = np.asarray(img)
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).clip(0, 255).astype(np.uint8)
    if img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
        img = np.transpose(img, (1, 2, 0))
    return img


@dataclasses.dataclass(frozen=True)
class YamInputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        images = data["images"]
        missing = [c for c in CAMERAS if c not in images]
        if missing:
            raise ValueError(f"YAM pi0.5 needs images {CAMERAS}, missing {missing}")
        base = _hwc_uint8(images["left_wrist"])
        right = _hwc_uint8(images["right_wrist"])
        inputs = {
            "state": np.asarray(data["state"], dtype=np.float32),
            "image": {
                "base_0_rgb": base,
                "left_wrist_0_rgb": np.zeros_like(base),
                "right_wrist_0_rgb": right,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            prompt = data["prompt"]
            inputs["prompt"] = prompt.decode() if isinstance(prompt, bytes) else prompt
        return inputs


@dataclasses.dataclass(frozen=True)
class YamOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :14])}  # model: 32-D padded; robot: 14


def make_example(prompt: str) -> dict:
    """A random robot-side observation (warm-up)."""
    h, w = MODEL_IMAGE_HW
    return {
        "images": {c: np.random.randint(256, size=(h, w, 3), dtype=np.uint8) for c in CAMERAS},
        "state": np.random.rand(14).astype(np.float32),
        "prompt": prompt,
    }
