"""pi05 models: models.yaml block, image preprocessing, and the client against a fake openpi policy server
(same websocket + msgpack protocol as openpi's WebsocketPolicyServer)."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

from yam_infer.config import check_model_matches, load_model_config
from yam_infer.policy.pi05 import (
    FORMAT,
    Pi05Policy,
    _pack,
    _unpack,
    preprocess_frame_pi05,
    resize_with_pad,
)

PI05 = "model:\n  server: 127.0.0.1:8000\n  format: pi05\n  task: put_block_in_bowl\n  action_mode: delta\n"


def test_models_config_pi05(tmp_path: Path) -> None:
    f = tmp_path / "models.yaml"
    f.write_text(PI05)
    m = load_model_config(f)
    assert (m.format, m.server, m.ckpt, m.task) == (
        "pi05",
        "127.0.0.1:8000",
        None,
        "put_block_in_bowl",
    )
    check_model_matches(m, FORMAT, ["put_block_in_bowl", "x"], "delta")
    with pytest.raises(ValueError, match="format"):
        check_model_matches(m, "yam_train/act_ref-v1", ["put_block_in_bowl"], "delta")
    for body, msg in (
        (PI05.replace("127.0.0.1:8000", "nohost"), "host:port"),
        (PI05 + "  ckpt: m.pt\n", "unknown key"),
        (PI05.replace("  server: 127.0.0.1:8000\n", ""), "model.server is required"),
    ):
        f.write_text(body)
        with pytest.raises(ValueError, match=msg):
            load_model_config(f)


def test_pi05_block_of_models_yaml_is_valid(tmp_path: Path) -> None:
    """The pi05 block of configs/models.yaml, active or commented out."""
    from yam_infer.config import DEFAULT_MODELS_CONFIG, PROJECT_ROOT

    lines = DEFAULT_MODELS_CONFIG.read_text().splitlines()
    start = lines.index(
        "# ---- yam_train pi05 model (one model, both tasks; runs in a policy server) ----"
    )
    block = []
    for ln in lines[start + 1 :]:
        if ln.startswith("# ") and not ln.startswith("# ----"):
            block.append(ln[2:])
        elif ln.strip() and not ln.startswith("#"):
            block.append(ln)
        else:
            break
    f = tmp_path / "models.yaml"
    f.write_text("\n".join(block) + "\n")
    m = load_model_config(f)
    assert m.format == "pi05"
    assert m.model_dir == PROJECT_ROOT / "models/pi05_yam"


def test_models_config_pi05_model_dir(tmp_path: Path) -> None:
    from yam_infer.config import PROJECT_ROOT

    f = tmp_path / "models.yaml"
    f.write_text(PI05 + "  model_dir: models/pi05_yam\n")
    assert load_model_config(f).model_dir == PROJECT_ROOT / "models/pi05_yam"
    f.write_text(PI05)
    assert load_model_config(f).model_dir is None
    f.write_text(PI05.replace("127.0.0.1", "10.0.0.5") + "  model_dir: models/pi05_yam\n")
    with pytest.raises(ValueError, match="local"):
        load_model_config(f)


def test_resize_with_pad_matches_openpi() -> None:
    """openpi_client.image_tools.resize_with_pad: 480x640 -> 168x224 content, 28 px black bands."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    out = resize_with_pad(img, 224, 224)
    assert out.shape == (224, 224, 3) and out.dtype == np.uint8
    assert not out[:28].any() and not out[-28:].any() and out[28:196].any()
    from PIL import Image

    ref = np.asarray(Image.fromarray(img).resize((224, 168), resample=Image.Resampling.BILINEAR))
    np.testing.assert_array_equal(out[28:196], ref)
    assert preprocess_frame_pi05(img, (224, 224)).shape == (224, 224, 3)


META = {
    "format": FORMAT,
    "config": "pi05_yam",
    "step": 14999,
    "fps": 30,
    "action_horizon": 50,
    "cameras": ["right_wrist", "left_wrist"],
    "image_hw": [224, 224],
    "prompts": {"put_block_in_bowl": "put the block in the bowl"},
    "start_pose": {"put_block_in_bowl": [0.0] * 14},
    "action_min": [-1.0] * 14,
    "action_max": [1.0] * 14,
}


@pytest.fixture
def fake_server():
    from websockets.sync.server import serve

    seen: list[dict] = []

    def handler(ws):
        ws.send(_pack(META))
        for msg in ws:
            obs = _unpack(msg)
            seen.append(obs)
            if obs["prompt"] == "boom":
                ws.send("Traceback: boom")
                return
            actions = (
                np.tile(obs["state"].astype(np.float32), (50, 1))
                + np.arange(50, dtype=np.float32)[:, None]
            )
            ws.send(_pack({"actions": actions, "policy_timing": {"infer_ms": 12.0}}))

    with serve(handler, "127.0.0.1", 0) as server:
        port = server.socket.getsockname()[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        yield f"127.0.0.1:{port}", seen
        server.shutdown()


def test_client_against_fake_server(fake_server) -> None:
    address, seen = fake_server
    pol = Pi05Policy(address, connect_timeout_s=5)
    assert (pol.format, pol.tasks, pol.chunk_size, pol.fps, pol.image_size) == (
        FORMAT,
        ["put_block_in_bowl"],
        50,
        30.0,
        (224, 224),
    )
    assert pol.info()["start_pose"]["put_block_in_bowl"] == [0.0] * 14
    frames = {c: np.full((480, 640, 3), 100, np.uint8) for c in ("right_wrist", "left_wrist")}
    qpos = np.linspace(0, 1, 14)
    chunk, images = pol.predict(frames, qpos, "put_block_in_bowl")
    assert chunk.shape == (50, 14) and np.allclose(chunk[3], qpos + 3, atol=1e-6)
    assert images.shape == (2, 224, 224, 3)
    obs = seen[-1]
    assert obs["prompt"] == "put the block in the bowl" and obs["state"].dtype == np.float32
    assert set(obs["images"]) == {"right_wrist", "left_wrist"} and obs["images"][
        "left_wrist"
    ].shape == (224, 224, 3)
    assert pol.last_timing["infer_ms"] == 12.0
    with pytest.raises(ValueError, match="knows the tasks"):
        pol.predict(frames, qpos, "put_banana_in_bowl")
    pol.prompts["put_block_in_bowl"] = "boom"
    with pytest.raises(RuntimeError, match="server error"):
        pol.predict(frames, qpos, "put_block_in_bowl")


def test_no_server() -> None:
    with pytest.raises(ConnectionError, match="no pi05 policy server"):
        Pi05Policy("127.0.0.1:9", connect_timeout_s=0)
