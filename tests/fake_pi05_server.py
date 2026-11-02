"""Stand-in for `python -m yam_infer.pi05_server` (same command line), started by test_pi05_launch:
"loads" for a moment, then serves yam_model.json as metadata and state + step as actions."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from websockets.sync.server import serve

from yam_infer.policy.pi05 import _pack, _unpack


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    a = p.parse_args()
    meta = json.loads((Path(a.model) / "yam_model.json").read_text())
    if meta.get("crash"):
        raise SystemExit("fake load failure")
    print("loading ...", flush=True)
    time.sleep(1.5)  # the port must stay closed until the model is ready

    def handler(ws):
        ws.send(_pack(meta))
        for msg in ws:
            obs = _unpack(msg)
            actions = np.tile(obs["state"].astype(np.float32), (meta["action_horizon"], 1))
            ws.send(_pack({"actions": actions}))

    with serve(handler, a.host, a.port) as server:
        print("serving", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
