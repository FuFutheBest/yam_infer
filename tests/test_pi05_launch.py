"""Auto-start of the pi0.5 policy server (yam_infer.policy.pi05_launch) with a fake server process,
plus the server package's import isolation (no openpi here)."""

from __future__ import annotations

import ast
import json
import socket
import sys
from pathlib import Path

import numpy as np
import pytest

from yam_infer.config import PROJECT_ROOT, ModelConfig
from yam_infer.policy.pi05 import FORMAT, Pi05Policy
from yam_infer.policy.pi05_launch import (
    PolicyServer,
    check_same_model,
    find_openpi_python,
    port_open,
)

META = {
    "format": FORMAT,
    "config": "pi05_yam",
    "experiment": "run1",
    "step": 14999,
    "repo_id": "yam/pi05_yam_all",
    "fps": 30,
    "action_horizon": 50,
    "cameras": ["right_wrist", "left_wrist"],
    "image_hw": [224, 224],
    "prompts": {"put_block_in_bowl": "put the block in the bowl"},
    "start_pose": {"put_block_in_bowl": [0.0] * 14},
    "action_min": [-1.0] * 14,
    "action_max": [1.0] * 14,
}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def model_dir(tmp_path: Path, **extra) -> Path:
    d = tmp_path / "pi05_yam"
    d.mkdir()
    (d / "yam_model.json").write_text(json.dumps({**META, **extra}))
    return d


def test_start_wait_connect_stop(tmp_path: Path) -> None:
    d, port = model_dir(tmp_path), free_port()
    server = PolicyServer.start(
        d,
        "127.0.0.1",
        port,
        tmp_path / "logs",
        python=Path(sys.executable),
        module="tests.fake_pi05_server",
    )
    try:
        assert not port_open("127.0.0.1", port)  # still "loading"
        server.wait_ready(30)
        pol = Pi05Policy(f"127.0.0.1:{port}", connect_timeout_s=5)
        check_same_model(pol.meta, d)
        chunk = pol.predict_prepared(
            np.zeros((2, 224, 224, 3), np.uint8), np.ones(14), "put_block_in_bowl"
        )
        assert chunk.shape == (50, 14)
        pol.close()
        with pytest.raises(ValueError, match="another model"):
            check_same_model({**pol.meta, "step": 1}, d)
    finally:
        server.stop()
    assert server.proc.poll() is not None and not port_open("127.0.0.1", port)
    assert "serving" in (tmp_path / "logs/pi05_server.log").read_text()


def test_server_crash_reports_log(tmp_path: Path) -> None:
    server = PolicyServer.start(
        model_dir(tmp_path, crash=True),
        "127.0.0.1",
        free_port(),
        tmp_path,
        python=Path(sys.executable),
        module="tests.fake_pi05_server",
    )
    with pytest.raises(RuntimeError, match="fake load failure"):
        server.wait_ready(30)


def test_ensure(tmp_path: Path, monkeypatch) -> None:
    port = free_port()
    cfg = ModelConfig(None, "pi05", "put_block_in_bowl", "delta", server=f"127.0.0.1:{port}")
    assert PolicyServer.ensure(cfg, tmp_path) is None  # no model_dir: never starts anything
    # the real server module under this project's python (no openpi): it must fail cleanly
    monkeypatch.setenv("OPENPI_PY", sys.executable)
    d = model_dir(tmp_path)
    cfg = ModelConfig(
        None, "pi05", "put_block_in_bowl", "delta", server=f"127.0.0.1:{port}", model_dir=d
    )
    with pytest.raises(RuntimeError, match="exited"):
        PolicyServer.ensure(cfg, tmp_path / "logs", ready_timeout_s=60)
    # a server already answers on the port (like openpi's /healthz): use it, start nothing
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Healthz(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.end_headers()

        def log_message(self, format: str, *args) -> None:  # noqa: A002 (base signature)
            pass

    httpd = HTTPServer(("127.0.0.1", port), Healthz)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        assert PolicyServer.ensure(cfg, tmp_path) is None
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_find_openpi_python(tmp_path: Path) -> None:
    py = tmp_path / "openpi/.venv/bin/python"
    py.parent.mkdir(parents=True)
    py.write_text("#!/bin/sh\n")
    py.chmod(0o755)
    assert find_openpi_python({"OPENPI_DIR": str(tmp_path / "openpi")}) == py
    assert find_openpi_python({"OPENPI_PY": str(py)}) == py
    with pytest.raises(FileNotFoundError, match="OPENPI_DIR"):
        find_openpi_python({"OPENPI_DIR": str(tmp_path / "nothing")})


def test_server_package_imports_only_openpi_and_itself() -> None:
    """yam_infer.pi05_server runs in the openpi venv: no imports of the rest of yam_infer."""
    # openpi and its own dependencies (all in the openpi venv)
    allowed = {"openpi", "numpy", "jax", "orbax", "flax", "yam_infer.pi05_server"}
    stdlib = set(sys.stdlib_module_names) | {"__future__"}
    for f in (PROJECT_ROOT / "yam_infer/pi05_server").glob("*.py"):
        for node in ast.walk(ast.parse(f.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for n in names:
                ok = n.split(".")[0] in stdlib or any(
                    n == a or n.startswith(a + ".") for a in allowed
                )
                assert ok, f"{f.name} imports {n}"
