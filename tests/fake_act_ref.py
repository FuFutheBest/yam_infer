"""Build a small random act_ref checkpoint with yam_train's exact layout (for tests / dry runs)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from yam_infer.policy.act_ref import FORMAT, ACTRefPolicy


def make_act_ref_ckpt(
    path: Path,
    task: str = "put_banana_in_bowl",
    action_mode: str = "delta",
    chunk: int = 100,
    small: bool = True,
) -> Path:
    policy_cfg = {
        "chunk_size": chunk,
        "kl_weight": 10,
        "hidden_dim": 64 if small else 512,
        "dim_feedforward": 128 if small else 3200,
        "enc_layers": 1 if small else 4,
        "dec_layers": 1 if small else 7,
        "nheads": 8,
        "dropout": 0.1,
        "pre_norm": False,
        "backbone": "resnet18",
        "position_embedding": "sine",
        "dilation": False,
        "masks": False,
        "lr": 1e-5,
        "lr_backbone": 1e-5,
        "weight_decay": 1e-4,
    }
    cams = ["right_wrist", "left_wrist"]
    torch.manual_seed(0)
    net = ACTRefPolicy(policy_cfg, cams)
    z = np.zeros(14, np.float32)
    stats = {
        "action_mean": z.tolist(),
        "action_std": (z + 0.1).tolist(),
        "qpos_mean": z.tolist(),
        "qpos_std": (z + 1).tolist(),
        "leader_min": (z - 3).tolist(),
        "leader_max": (z + 3).tolist(),
    }
    torch.save(
        {
            "format": FORMAT,
            "config": {"policy": policy_cfg, "data": {"task": task}},
            "stats": stats,
            "task": task,
            "camera_names": cams,
            "image_size": [480, 640],
            "fps": 30,
            "chunk_size": chunk,
            "action_mode": action_mode,
            "action_shift": 1,
            "start_pose": z.tolist(),
            "model": net.state_dict(),
            "epoch": 0,
        },
        path,
    )
    return path


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--task", default="put_banana_in_bowl")
    p.add_argument("--full", action="store_true", help="real act_ref sizes (512 wide, 4+7 layers)")
    a = p.parse_args()
    print(make_act_ref_ckpt(a.out, task=a.task, small=not a.full))
