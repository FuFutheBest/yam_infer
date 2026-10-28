#!/usr/bin/env bash
# Bring any still-powered follower arm home, then stop every inference / worker process.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
exec .venv/bin/python -m yam_infer.tools.stop_all "$@"
