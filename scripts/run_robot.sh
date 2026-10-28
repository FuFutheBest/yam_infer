#!/usr/bin/env bash
# Usage: scripts/run_robot.sh <banana|block> [extra args for yam_infer.run]
# Before starting, free_hw lists whatever holds the CAN buses / cameras / RPC ports (e.g. the
# SuperInference WebUI) and asks whether to stop it; CAN links that are down get reset (sudo).
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
[ -f robot.env ] && . ./robot.env # machine-specific: CKPT, HW_CONFIG
CKPT="${CKPT:-models/act_v2_all.pt}"
HW_CONFIG="${HW_CONFIG:-configs/hardware.yaml}"
case "${1:-}" in
banana) TASK=put_banana_in_bowl ;;
block) TASK=put_blue_block_in_blue_cup ;;
*)
  echo "usage: $0 banana|block [args]"
  exit 1
  ;;
esac
shift
[ -x .venv/bin/python ] || { echo "错误：没有 .venv，先执行 bash scripts/setup_robot.sh"; exit 1; }
[ -f "$CKPT" ] || { echo "错误：找不到模型 $CKPT"; exit 1; }

if ! .venv/bin/python -m yam_infer.tools.free_hw --hw-config "$HW_CONFIG"; then
  echo "硬件仍被占用，未启动。"
  exit 1
fi
exec .venv/bin/python -m yam_infer.run --ckpt "$CKPT" --task "$TASK" --hw-config "$HW_CONFIG" "$@"
