"""RPC worker owning ONE follower arm (CAN bus + motors) with a local watchdog.

    python -m yam_infer.hw.follower.worker --config configs/hardware.yaml --side left
    python -m yam_infer.hw.follower.worker --config configs/hardware.yaml --side left --fake

Started by ``yam_infer.hw.arms.FollowerArms`` (one process per side); you normally never run it
by hand.  ``--fake`` serves a simulated arm (no CAN, no i2rt) for hardware-free tests.

Vendored from SuperInference-feat-i2rt_NEDF3 @0dbfa5d ``devices/robots/i2rt_teleop/worker.py``
(see VENDORED.md).  Local changes: follower role only, stdlib logging instead of SI's loguru
logger (no logs/ directory side effect), ``--fake``.  RPC contract and shutdown order unchanged.
"""

from __future__ import annotations

import argparse
import logging
import signal
import threading
from pathlib import Path
from typing import Any

import portal

from .config import SIDES, load_teleoperation_config
from .endpoints import FakeFollowerEndpoint, create_follower_endpoint
from .runtime import WorkerControlRuntime

LOGGER = logging.getLogger("yam_infer.worker")


def serve_worker(config_path: Path, side: str, fake: bool = False) -> None:
    config = load_teleoperation_config(config_path)
    arm_config = config.arms[side]
    port = arm_config.follower.rpc_port
    endpoint = None
    server = None
    runtime = None
    stage = "constructing the hardware endpoint"
    stop_requested = threading.Event()

    def request_stop(_signum: int, _frame: Any) -> None:
        LOGGER.info("Stopping %s follower worker", side)
        stop_requested.set()
        if runtime is not None and (server is None or not server.running):
            runtime.request_stop()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        LOGGER.info("worker %s/follower%s: %s", side, " (FAKE)" if fake else "", stage)
        endpoint = FakeFollowerEndpoint() if fake else create_follower_endpoint(arm_config.follower)
        if stop_requested.is_set():
            return

        runtime = WorkerControlRuntime(
            endpoint,
            role="follower",
            side=side,
            control=config.control,
        )
        if stop_requested.is_set():
            return
        stage = "initializing the safe-hold control loop"
        LOGGER.info("worker %s/follower: %s", side, stage)
        if not runtime.start(
            config.control.startup_timeout_s,
            cancel_event=stop_requested,
        ):
            return
        if stop_requested.is_set():
            return
        stage = f"starting RPC server on port {port}"
        server = portal.Server(port)
        if stop_requested.is_set():
            return
        server.bind("ping", runtime.ping)
        server.bind("get_sample", runtime.get_sample)
        server.bind("get_status", runtime.get_status)
        server.bind("begin_control_session", runtime.begin_control_session)
        server.bind("set_phase", runtime.set_phase)
        server.bind("submit_control_frame", runtime.submit_control_frame)
        server.bind("end_control_session", runtime.end_control_session)
        server.bind("reset_fault", runtime.reset_fault)
        if stop_requested.is_set():
            return
        LOGGER.info("worker %s/follower: ready on 127.0.0.1:%s", side, port)
        stage = "serving RPC requests"
        server.start(block=False)
        while server.running:
            if stop_requested.wait(0.1):
                break
    except KeyboardInterrupt:
        LOGGER.info("worker %s/follower interrupted", side)
    except Exception as exc:
        LOGGER.exception(
            "worker %s/follower failed while %s: %s: %s", side, stage, type(exc).__name__, exc
        )
        raise
    finally:
        if server is not None and server.running:
            server.close()
        if runtime is not None:
            # runtime.close() commands the safe hold first; endpoint.close() releases the motors
            if runtime.close() and endpoint is not None:
                endpoint.close()
            else:
                LOGGER.error("Skipping concurrent endpoint close for blocked %s worker", side)
        elif endpoint is not None:
            endpoint.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--side", required=True, choices=SIDES)
    parser.add_argument("--fake", action="store_true", help="simulated arm, no hardware")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    serve_worker(args.config.resolve(), args.side, fake=args.fake)


if __name__ == "__main__":
    main()
