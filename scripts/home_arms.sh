#!/usr/bin/env bash
# Bring both follower arms to the rest pose whatever state the workers are in, then power off.
#   bash scripts/home_arms.sh                 configs/rest_pose.yaml (asks before moving)
#   bash scripts/home_arms.sh --target zero   joints 0, grippers open
#   bash scripts/home_arms.sh --keep-on       stay powered (Ctrl+C = motors off)
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
[ -x .venv/bin/python ] || { echo "错误：没有 .venv，先执行 bash scripts/setup_robot.sh"; exit 1; }
exec .venv/bin/python -m yam_infer.tools.home_arms "$@"
