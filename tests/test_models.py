"""configs/models.yaml + both checkpoint formats (act_ref built offline, no ImageNet download)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from yam_infer.config import DEFAULT_MODELS_CONFIG, check_model_matches, load_model_config
from yam_infer.policy.policy import Policy, preprocess_frame

from .fake_act_ref import make_act_ref_ckpt


def test_project_models_config() -> None:
    m = load_model_config(DEFAULT_MODELS_CONFIG)
    assert (m.format, m.task, m.action_mode) == ("act_v2", "put_blue_block_in_blue_cup", "delta")
    assert m.ckpt.is_absolute() and m.ckpt.name == "act_v2_all.pt"


def test_commented_act_ref_block_is_valid(tmp_path: Path) -> None:
    """Switching to the prepared act_ref block = uncomment it (and comment the old one)."""
    lines = DEFAULT_MODELS_CONFIG.read_text().splitlines()
    start = lines.index("# ---- new yam_train act_ref model (one task per file) ----")
    act_ref = "\n".join(ln[2:] for ln in lines[start + 1 :] if ln.startswith("# "))
    f = tmp_path / "models.yaml"
    f.write_text(act_ref + "\n")
    m = load_model_config(f)
    assert (m.format, m.task, m.action_mode) == ("act_ref", "put_banana_in_bowl", "delta")


GOOD = "model:\n  ckpt: m.pt\n  format: act_v2\n  task: t\n  action_mode: delta\n"


@pytest.mark.parametrize(
    "body, message",
    [
        (GOOD + "model2: {}\n", "exactly one section"),
        (GOOD.replace("format: act_v2", "format: act_v3"), "model.format"),
        (GOOD.replace("action_mode: delta", "action_mode: rel"), "model.action_mode"),
        (GOOD.replace("  task: t\n", ""), "model.task is required"),
        (GOOD + "  alias: b\n", "unknown key"),
        (GOOD + "  execution: {mode: ensemble}\n", "set execution in run.yaml"),
    ],
)
def test_bad_models_config(tmp_path: Path, body: str, message: str) -> None:
    f = tmp_path / "models.yaml"
    f.write_text(body)
    with pytest.raises(ValueError, match=message):
        load_model_config(f)


def test_file_must_match_the_config(tmp_path: Path) -> None:
    f = tmp_path / "models.yaml"
    f.write_text(GOOD.replace("task: t", "task: put_banana_in_bowl"))
    cfg = load_model_config(f)
    check_model_matches(cfg, "act_v2", ["put_banana_in_bowl", "x"], "delta")  # ok
    with pytest.raises(ValueError, match="action_mode"):
        check_model_matches(cfg, "act_v2", ["put_banana_in_bowl"], "abs")
    with pytest.raises(ValueError, match="task"):
        check_model_matches(cfg, "act_v2", ["x"], "delta")
    with pytest.raises(ValueError, match="format"):
        check_model_matches(cfg, "yam_train/act_ref-v1", ["put_banana_in_bowl"], "delta")


@pytest.fixture(scope="module")
def act_ref_ckpt(tmp_path_factory) -> Path:
    return make_act_ref_ckpt(tmp_path_factory.mktemp("m") / "act_ref.pt")


def test_act_ref_loads_offline_and_predicts(act_ref_ckpt: Path, monkeypatch) -> None:
    import torchvision.models._api as api

    def no_network(*_a, **_k):
        raise AssertionError("tried to download ImageNet weights")

    monkeypatch.setattr(api, "load_state_dict_from_url", no_network)
    pol = Policy(act_ref_ckpt, "cpu")
    assert pol.format == "yam_train/act_ref-v1" and pol.tasks == ["put_banana_in_bowl"]
    assert (pol.chunk_size, pol.image_size, pol.action_shift) == (100, (480, 640), 1)
    q = np.full(14, 0.3)
    out = pol.predict_prepared(np.zeros((2, 480, 640, 3), np.uint8), q, "put_banana_in_bowl")
    assert out.shape == (100, 14)
    assert pol.last_raw is not None
    np.testing.assert_allclose(out, pol.last_raw * 0.1 + q, atol=1e-6)  # delta: + current state
    with pytest.raises(ValueError, match="trained for"):
        pol.predict_prepared(np.zeros((2, 480, 640, 3), np.uint8), q, "put_blue_block_in_blue_cup")


def test_preprocess_keeps_full_resolution_for_act_ref() -> None:
    rgb = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    assert preprocess_frame(rgb, (480, 640)).shape == (480, 640, 3)
    np.testing.assert_array_equal(preprocess_frame(rgb, (480, 640), jpeg_roundtrip=False), rgb)
    assert preprocess_frame(rgb, (240, 320)).shape == (240, 320, 3)  # act_v2
