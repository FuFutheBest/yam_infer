#!/usr/bin/env bash
# Usage: scripts/run_robot.sh [extra args for yam_infer.run]
#   the model, its task and how to read it: configs/models.yaml (edit that to switch)
# Before starting, free_hw lists whatever holds the CAN buses / cameras / RPC ports (e.g. the
# SuperInference WebUI) and asks whether to stop it; CAN links that are down get reset (sudo).
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
[ -f robot.env ] && . ./robot.env # machine-specific: HW_CONFIG, optional CKPT override
HW_CONFIG="${HW_CONFIG:-configs/hardware.yaml}"
[ -x .venv/bin/python ] || { echo "错误：没有 .venv，先执行 bash scripts/setup_robot.sh"; exit 1; }
CKPT_ARGS=()
[ -n "${CKPT:-}" ] && CKPT_ARGS=(--ckpt "$CKPT") # robot.env / env override of models.yaml

if ! .venv/bin/python -m yam_infer.tools.free_hw --hw-config "$HW_CONFIG"; then
  echo "硬件仍被占用，未启动。"
  exit 1
fi
exec .venv/bin/python -m yam_infer.run "${CKPT_ARGS[@]}" --hw-config "$HW_CONFIG" "$@"
