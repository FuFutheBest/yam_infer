# pyright: reportMissingImports=false
# (openpi exists only in the openpi venv this module runs in, not in this project's .venv)
"""Serve an exported pi0.5 model with openpi's websocket policy server (openpi venv).

    python -m yam_infer.pi05_server --model models/pi05_yam [--host 127.0.0.1] [--port 8000]

Same as `yam_train.pi05.cli serve` (yam_train pi05 @54588b2): the openpi TrainConfig is rebuilt from
the model's yam_model.json (only the parts serving uses: model shape, data transforms, norm stats),
the network is warmed up (JIT compile), then the port opens.  The client (yam_infer.policy.pi05)
gets yam_model.json as metadata on connect.  The port opening = ready (yam_infer waits for it).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import pathlib
import time

log = logging.getLogger("yam_infer.pi05_server")


def make_train_config(model_dir: pathlib.Path, meta: dict):
    """The openpi TrainConfig the model was trained with, as far as serving needs it."""
    from openpi import transforms as _transforms
    from openpi.models import pi0_config
    from openpi.training import config as _config

    from yam_infer.pi05_server.yam_policy import YamInputs, YamOutputs, delta_mask_for

    mask = delta_mask_for(meta)

    @dataclasses.dataclass(frozen=True)
    class YamServeDataConfig(_config.DataConfigFactory):
        @staticmethod
        def _yam_transforms():
            return _transforms.Group(inputs=[YamInputs()], outputs=[YamOutputs()]).push(
                inputs=[_transforms.DeltaActions(mask)],
                outputs=[_transforms.AbsoluteActions(mask)],
            )

        def create(self, assets_dirs, model_config) -> _config.DataConfig:
            return dataclasses.replace(
                self.create_base_config(assets_dirs, model_config),
                data_transforms=self._yam_transforms(),
                model_transforms=_config.ModelTransformFactory()(model_config),
                action_sequence_keys=("actions",),
            )

    return _config.TrainConfig(
        name=meta["config"],
        model=pi0_config.Pi0Config(pi05=True, action_horizon=int(meta["action_horizon"])),
        data=YamServeDataConfig(
            repo_id=meta["repo_id"],
            assets=_config.AssetsConfig(assets_dir=str(model_dir / "assets")),
        ),
        policy_metadata=meta,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, force=True, format="%(asctime)s %(levelname)s %(message)s"
    )
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--model", required=True, help="exported model directory (yam_model.json, params/, assets/)"
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()

    model_dir = pathlib.Path(a.model).resolve()
    try:
        meta = json.loads((model_dir / "yam_model.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"{model_dir} is not an exported pi05 model (yam_model.json): {exc}"
        ) from exc
    for part in ("params", f"assets/{meta['repo_id']}/norm_stats.json"):
        if not (model_dir / part).exists():
            raise SystemExit(f"{model_dir} has no {part}")

    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server

    from yam_infer.pi05_server.yam_policy import make_example

    log.info("loading %s (config %s, step %s) ...", model_dir, meta["config"], meta.get("step"))
    policy = policy_config.create_trained_policy(make_train_config(model_dir, meta), model_dir)
    prompt = next(iter(meta["prompts"].values()))
    for i in range(2):  # the first inference compiles the network: do it before the robot connects
        t0 = time.perf_counter()
        policy.infer(make_example(prompt))
        log.info("warm-up inference %d: %.0f ms", i, (time.perf_counter() - t0) * 1000)
    log.info("serving %s on %s:%d", model_dir.name, a.host, a.port)
    websocket_policy_server.WebsocketPolicyServer(
        policy, host=a.host, port=a.port, metadata=meta
    ).serve_forever()


if __name__ == "__main__":
    main()
