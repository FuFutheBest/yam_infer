"""Inference side of yam_train's `act_ref` models (= tonyzhaozh/act ACTPolicy, one task per model).

Checkpoint `format: yam_train/act_ref-v1` (yam_train/train.py):
    model        ACTPolicy.state_dict()  (keys "model.*")
    config       the training YAML (policy: section = ACT source argument names)
    stats        qpos_mean/std, action_mean/std, leader_min/max
    task, camera_names, image_size [480, 640], fps 30, chunk_size 100, action_mode, action_shift,
    start_pose
Network code: yam_infer/policy/detr = yam_train/detr (tonyzhaozh/act, see VENDORED.md), unmodified.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager

import torch
from torch import nn
from torchvision import transforms

FORMAT = "yam_train/act_ref-v1"

# act/detr/main.py::get_args_parser defaults (same table as yam_train/policy.py)
SOURCE_DEFAULTS = {
    "lr": 1e-4,
    "lr_backbone": 1e-5,
    "weight_decay": 1e-4,
    "backbone": "resnet18",
    "dilation": False,
    "position_embedding": "sine",
    "enc_layers": 4,
    "dec_layers": 6,
    "dim_feedforward": 2048,
    "hidden_dim": 256,
    "dropout": 0.1,
    "nheads": 8,
    "num_queries": 400,
    "pre_norm": False,
    "masks": False,
}


def make_args(policy_cfg: dict, camera_names: list[str]) -> argparse.Namespace:
    """yam_train/policy.py::make_args: `chunk_size` -> `num_queries`, `kl_weight` is training only."""
    args = dict(SOURCE_DEFAULTS)
    for k, v in policy_cfg.items():
        if k == "chunk_size":
            args["num_queries"] = v
        elif k != "kl_weight":
            args[k] = v
    args["camera_names"] = list(camera_names)
    return argparse.Namespace(**args)


@contextmanager
def _no_imagenet_download():
    """detr/models/backbone.py builds resnet18 with pretrained=is_main_process(), i.e. downloads
    ImageNet weights.  At inference every weight comes from the checkpoint right after, so the
    download is useless and may fail on the robot -> pretend we are not the main process while
    building.  (The vendored file itself stays identical to the source.)"""
    from yam_infer.policy.detr.models import backbone

    original = backbone.is_main_process
    backbone.is_main_process = lambda: False
    try:
        yield
    finally:
        backbone.is_main_process = original


class ACTRefPolicy(nn.Module):
    """ACTPolicy, inference branch only: image (B, n_cam, 3, H, W) in 0..1, qpos normalised -> a_hat."""

    def __init__(self, policy_cfg: dict, camera_names: list[str]):
        super().__init__()
        import warnings

        from yam_infer.policy.detr.models import build_ACT_model

        with _no_imagenet_download(), warnings.catch_warnings():
            # torchvision deprecates the source's `pretrained=` argument; harmless here
            warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
            self.model = build_ACT_model(make_args(policy_cfg, camera_names))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    @torch.no_grad()
    def forward(self, qpos: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        a_hat, _, _ = self.model(qpos, self.normalize(image), None)  # no actions -> z = 0
        return a_hat
