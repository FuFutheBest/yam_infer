#!/usr/bin/env bash
# One-time setup on the robot PC. Run from the yam_infer directory.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
uv venv .venv --python /usr/bin/python3.10
uv pip install --python .venv/bin/python torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -r requirements.txt \
  --index-url https://mirrors.aliyun.com/pypi/simple/
.venv/bin/python -c "import torch, pyrealsense2, portal; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
