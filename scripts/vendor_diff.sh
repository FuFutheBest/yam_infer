#!/usr/bin/env bash
# Show how the vendored code differs from a SuperInference checkout (see VENDORED.md).
#   bash scripts/vendor_diff.sh ~/Dev/MVIG/SuperInference-feat-i2rt_NEDF3
set -uo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
SI="${1:?usage: $0 <SuperInference checkout>}"
T="$SI/devices/robots/i2rt_teleop"
echo "== SI commit: $(git -C "$SI" rev-parse HEAD 2>/dev/null || echo unknown)"
for f in runtime config; do
  if diff -q "$T/$f.py" "yam_infer/hw/follower/$f.py" >/dev/null; then
    echo "== $f.py: identical"
  else
    echo "== $f.py: DIFFERS"; diff -u "$T/$f.py" "yam_infer/hw/follower/$f.py"
  fi
done
echo "== endpoints.py / worker.py are adapted copies; upstream changes since vendoring:"
git -C "$SI" log --oneline 0dbfa5d..HEAD -- devices/robots/i2rt_teleop third_party/i2rt 2>/dev/null || true
echo "== i2rt (only files we kept):"
diff -rq "$SI/third_party/i2rt/i2rt" third_party/i2rt/i2rt -x __pycache__ 2>/dev/null | grep -v "^Only in $SI" || echo "identical"
