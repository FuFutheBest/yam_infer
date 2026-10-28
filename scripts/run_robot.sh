#!/usr/bin/env bash
# Usage: scripts/run_robot.sh <banana|block> [extra args for yampilot.run]
# Stop the data-collection WebUI (start.sh) first: it owns the CAN buses and cameras.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
[ -f robot.env ] && . ./robot.env          # machine-specific: SI_ROOT, CKPT
SI_ROOT="${SI_ROOT:-$HOME/SuperInference-feat-i2rt_NEDF3}"
CKPT="${CKPT:-models/act_v2_all.pt}"
case "${1:-}" in
  banana) TASK=put_banana_in_bowl ;;
  block)  TASK=put_blue_block_in_blue_cup ;;
  *) echo "usage: $0 banana|block [args]"; exit 1 ;;
esac
shift
if pgrep -f "python[0-9.]* -m (incubator\.nedf3\.I2RT_DM3\.web|devices\.robots\.i2rt_teleop)" >/dev/null; then
  echo "错误：采集 WebUI 或机械臂 worker 仍在运行（占用 CAN 和相机）："
  pgrep -af "python[0-9.]* -m (incubator\.nedf3\.I2RT_DM3\.web|devices\.robots\.i2rt_teleop)"
  echo "WebUI 请在它的终端 Ctrl+C；残留 worker 执行:  bash scripts/stop_all.sh  （会先把手臂慢慢归位再断电）"
  exit 1
fi
down=""
for i in can0 can1 can2 can3; do
  ip -br link show "$i" 2>/dev/null | grep -q "UP" || down="$down $i"
done
if [ -n "$down" ]; then
  echo "CAN 未就绪:$down -> 重置（需要 sudo 密码）"
  (cd "$SI_ROOT" && bash third_party/i2rt/scripts/reset_all_can.sh)
fi
export SI_I2RT_YAM_GRIPPER_TYPE=crank_4310
exec .venv/bin/python -m yampilot.run --ckpt "$CKPT" --task "$TASK" --si-root "$SI_ROOT" "$@"
