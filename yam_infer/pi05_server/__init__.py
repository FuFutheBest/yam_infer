"""pi0.5 policy server (runs in the openpi virtualenv, NOT in this project's .venv).

    $OPENPI_DIR/.venv/bin/python -m yam_infer.pi05_server --model models/pi05_yam --port 8100

Normally you do not start it yourself: `yam_infer.run` (scripts/run_robot.sh) starts it when
configs/models.yaml has a pi05 block with `model_dir:` and nothing listens on `server:` yet
(see yam_infer/policy/pi05_launch.py).

Self-contained: needs only openpi (@215abfb) and the exported model directory
(params/, assets/<repo_id>/norm_stats.json, yam_model.json).  The YAM data mapping is copied from
yam_train (pi05 branch), see VENDORED.md.  Nothing here may import the rest of yam_infer (other
numpy / dependencies).
"""
