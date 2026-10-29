"""configs/run.yaml loading: defaults, validation, command-line override."""

from __future__ import annotations

from pathlib import Path

import pytest

from yam_infer.config import DEFAULT_RUN_CONFIG, load_run_config


def test_project_config_loads() -> None:
    cfg = load_run_config(DEFAULT_RUN_CONFIG)
    assert cfg["execution"] == {"mode": "chunk", "horizon": 30, "blend_s": 0.1, "ensemble_m": 0.01}
    assert cfg["episode"]["max_seconds"] == 40.0


@pytest.mark.parametrize(
    "patch, message",
    [
        (("\n  mode: chunk", "\n  mode: chunky"), "execution.mode"),
        (("\n  horizon: 30", "\n  horizon: 0"), "execution.horizon"),
        (("\n  horizon: 30", "\n  horizon: 30.5"), "execution.horizon"),
        (("\n  blend_s: 0.1", "\n  blend_s: -1"), "execution.blend_s"),
        (("max_seconds: 40.0", "max_seconds: 0"), "episode.max_seconds"),
        (("\n  horizon: 30", "\n  horizon: 30\n  horizn: 20"), "unknown key"),
        (("  ensemble_m: 0.01\n", ""), "missing execution.ensemble_m"),
    ],
)
def test_bad_values_are_rejected(tmp_path: Path, patch, message) -> None:
    text = DEFAULT_RUN_CONFIG.read_text()
    assert patch[0] in text
    bad = tmp_path / "run.yaml"
    bad.write_text(text.replace(patch[0], patch[1], 1))
    with pytest.raises(ValueError, match=message):
        load_run_config(bad)
