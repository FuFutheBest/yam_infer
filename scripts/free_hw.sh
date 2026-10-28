#!/usr/bin/env bash
# List programs blocking the robot (CAN buses, cameras, RPC ports, CAN links down) and
# interactively stop them.  Exit 0 = hardware free.
#   bash scripts/free_hw.sh            ask before stopping anything
#   bash scripts/free_hw.sh --check    list only (exit 1 if busy)
#   bash scripts/free_hw.sh --sudo     also inspect other users' processes
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
[ -x .venv/bin/python ] || { echo "错误：没有 .venv，先执行 bash scripts/setup_robot.sh"; exit 1; }
exec .venv/bin/python -m yam_infer.tools.free_hw "$@"
