"""Start the pi0.5 policy server automatically (models.yaml pi05 block with `model_dir:`).

    yam_infer.run --> PolicyServer.ensure(cfg) --> nothing listens on server: host:port?
                          yes: start `<openpi python> -m yam_infer.pi05_server --model DIR`
                               (own process group, log: <record_dir>/pi05_server.log),
                               wait until the port opens (model loaded + warmed up, ~1-2 min)
                          no:  use the server that is already running (check_same_model checks
                               that it serves model_dir)
    exit (normal, Ctrl+C, error) --> stop() the server we started (SIGTERM, then SIGKILL)

The openpi venv python: $OPENPI_PY, else $OPENPI_DIR/.venv/bin/python, else the first existing of
DEFAULT_OPENPI_DIRS (<project>/../openpi, the robot's ~/fufu/astra_inference_train/openpi).  GPU memory: XLA_PYTHON_CLIENT_MEM_FRACTION defaults
to 0.5 here (JAX would grab 75%; other GPU users may share the card); set it in robot.env to change.
"""

from __future__ import annotations

import atexit
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from yam_infer.config import PROJECT_ROOT, ModelConfig

LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")
DEFAULT_MEM_FRACTION = "0.5"
# openpi checkouts (openpi @215abfb with its uv .venv) tried when OPENPI_DIR / OPENPI_PY are unset
DEFAULT_OPENPI_DIRS = (
    PROJECT_ROOT.parent / "openpi",
    Path.home() / "fufu/astra_inference_train/openpi",  # robot PC
)


def split_address(address: str) -> tuple[str, int]:
    host, _, port = address.removeprefix("ws://").rpartition(":")
    return host, int(port)


def port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def find_openpi_python(env: dict[str, str] | None = None) -> Path:
    env = dict(os.environ) if env is None else env
    if env.get("OPENPI_PY"):
        candidates = [Path(env["OPENPI_PY"])]
    elif env.get("OPENPI_DIR"):
        candidates = [Path(env["OPENPI_DIR"]).expanduser() / ".venv/bin/python"]
    else:
        candidates = [d / ".venv/bin/python" for d in DEFAULT_OPENPI_DIRS]
    for c in candidates:
        try:
            if c.is_file() and os.access(c, os.X_OK):
                return c
        except OSError:  # unreadable parent directory etc. = not usable
            continue
    raise FileNotFoundError(
        f"no openpi python ({', '.join(map(str, candidates))}): set OPENPI_DIR (openpi checkout with "
        "its uv .venv, openpi @215abfb) or OPENPI_PY in robot.env"
    )


def read_model_meta(model_dir: Path) -> dict:
    try:
        return json.loads((model_dir / "yam_model.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise FileNotFoundError(
            f"{model_dir} is not an exported pi05 model (yam_model.json): {exc}"
        )


def check_same_model(served: dict, model_dir: Path) -> None:
    """The server already running on the port must serve model_dir (not some other model)."""
    want = read_model_meta(model_dir)
    keys = ("config", "experiment", "step", "repo_id")
    diff = {k: (served.get(k), want.get(k)) for k in keys if served.get(k) != want.get(k)}
    if diff:
        raise ValueError(
            f"the policy server already running serves another model than {model_dir} "
            f"(served vs. models.yaml: {diff}); stop it or change models.yaml server:"
        )


class PolicyServer:
    """A pi05_server process started by us; stop() also runs at interpreter exit."""

    def __init__(self, proc: subprocess.Popen, log_path: Path, host: str, port: int):
        self.proc, self.log_path, self.host, self.port = proc, log_path, host, port
        atexit.register(self.stop)

    @classmethod
    def ensure(
        cls, cfg: ModelConfig, log_dir: Path, ready_timeout_s: float = 900.0
    ) -> PolicyServer | None:
        """Start the server for cfg.model_dir unless one already listens; None = not started by us."""
        if cfg.model_dir is None:
            return None
        host, port = split_address(cfg.server)
        if port_open(host, port):
            print(f"pi05: using the policy server already running at {host}:{port}")
            return None
        read_model_meta(cfg.model_dir)
        server = cls.start(cfg.model_dir, host, port, log_dir)
        server.wait_ready(ready_timeout_s)
        return server

    @classmethod
    def start(
        cls,
        model_dir: Path,
        host: str,
        port: int,
        log_dir: Path,
        python: Path | None = None,
        module: str = "yam_infer.pi05_server",
    ) -> PolicyServer:
        python = python or find_openpi_python()
        env = dict(os.environ)
        for k in ("VIRTUAL_ENV", "PYTHONHOME"):  # this project's venv must not leak in
            env.pop(k, None)
        env["PYTHONPATH"] = str(PROJECT_ROOT)  # only yam_infer.pi05_server is imported from here
        env["PYTHONNOUSERSITE"] = "1"
        env["HF_HUB_OFFLINE"] = "1"
        if "XLA_PYTHON_CLIENT_PREALLOCATE" not in env:
            env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", DEFAULT_MEM_FRACTION)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "pi05_server.log"
        cmd = [str(python), "-u", "-m", module, "--model", str(model_dir)]
        cmd += ["--host", host, "--port", str(port)]
        print(f"pi05: starting the policy server ({model_dir.name}, log {log_path}) ...")
        try:
            with open(log_path, "w") as log:
                log.write(f"$ {' '.join(cmd)}\n")
                log.flush()
                proc = subprocess.Popen(
                    cmd,
                    cwd=PROJECT_ROOT,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,  # Ctrl+C goes to run.py only; we stop it ourselves
                )
        except OSError as exc:
            raise RuntimeError(f"cannot start the pi05 policy server ({cmd[0]}): {exc}") from exc
        return cls(proc, log_path, host, port)

    def log_tail(self, n: int = 20) -> str:
        try:
            return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-n:])
        except OSError:
            return ""

    def wait_ready(self, timeout_s: float) -> None:
        """Wait until the port opens (= model loaded and compiled)."""
        t0 = time.monotonic()
        last_print = t0
        while not port_open(self.host, self.port):
            if self.proc.poll() is not None:
                tail = self.log_tail()
                self.stop()
                raise RuntimeError(
                    f"the pi05 policy server exited (code {self.proc.returncode}); "
                    f"{self.log_path}:\n{tail}"
                )
            now = time.monotonic()
            if now - t0 > timeout_s:
                tail = self.log_tail()
                self.stop()
                raise TimeoutError(
                    f"the pi05 policy server was not ready after {timeout_s:.0f} s; "
                    f"{self.log_path}:\n{tail}"
                )
            if now - last_print >= 10:
                last = self.log_tail(1).strip()
                print(f"pi05: loading the model ... {now - t0:.0f} s  | {last[-100:]}", flush=True)
                last_print = now
            time.sleep(1)
        print(f"pi05: policy server ready after {time.monotonic() - t0:.0f} s")

    def stop(self, grace_s: float = 15.0) -> None:
        if self.proc.poll() is not None:
            return
        print("pi05: stopping the policy server ...", file=sys.stderr)
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
            self.proc.wait(grace_s)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait()
        except ProcessLookupError:
            pass
