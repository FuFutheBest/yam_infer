"""Strip a training checkpoint down to what inference needs (drops optimizer state, ~3x smaller).

python -m yam_infer.tools.export CKPT_IN MODEL_OUT
"""

from __future__ import annotations

import sys

import torch

KEEP = (
    "model",
    "config",
    "stats",
    "tasks",
    "cameras",
    "image_size",
    "fps",
    "start_pose",
    "state_layout",
    "action_mode",
    "action_norm",
    "step",
    "args",
)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    out = {k: ck[k] for k in KEEP if k in ck}
    torch.save(out, sys.argv[2])
    print(
        f"exported step {out.get('step')} ({out.get('action_mode', 'abs')} actions) -> {sys.argv[2]}"
    )


if __name__ == "__main__":
    main()
