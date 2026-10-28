#!/usr/bin/env bash
# One-time setup on the robot PC (Ubuntu 22.04, Python 3.10, NVIDIA RTX 50xx). Run from anywhere.
#   bash scripts/setup_robot.sh          create .venv and install everything
#   bash scripts/setup_robot.sh --dev    + pytest / ruff
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
MIRROR="${PIP_MIRROR:-https://mirrors.aliyun.com/pypi/simple/}"
[ -x .venv/bin/python ] || uv venv .venv --python /usr/bin/python3.10
# RTX 50xx needs CUDA 12.8 wheels
uv pip install --python .venv/bin/python torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -r requirements.txt --index-url "$MIRROR"
[ "${1:-}" = "--dev" ] && uv pip install --python .venv/bin/python pytest ruff --index-url "$MIRROR"
# make the vendored i2rt SDK importable from this venv (the worker launcher also sets PYTHONPATH)
SITE=$(.venv/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
echo "$PWD/third_party/i2rt" >"$SITE/yam_infer_i2rt.pth"
.venv/bin/python scripts/check_env.py
