"""Re-run the model offline on a recorded episode and compare with what it output live.

    python -m yam_infer.tools.replay_policy runs_infer/<run> --ckpt models/act_v2_all.pt [--device cpu]

For every row of policy.npz: take the camera frames with the recorded frame counters (cam_seq) from
data/*.mcap, rebuild the model input (decode JPEG -> resize, no second JPEG round-trip: the stored
JPEG IS the round-tripped image), feed it with the recorded qpos, and compare the output chunk.
Exact match on CPU (float32).  The live run on a GPU uses bf16, so expect differences of
~1e-3 rad there; use --device cuda on the robot to compare like with like.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from yam_infer.data.nedf3_writer import CAMERA_TOPICS


def load_frames(run: Path, wanted: dict[str, set[int]]) -> dict[str, dict[int, bytes]]:
    """camera name -> {header.seq: JPEG bytes} for the requested frame counters."""
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    topic_to_cam = {t: c for c, t in CAMERA_TOPICS.items() if c in wanted}
    out: dict[str, dict[int, bytes]] = {c: {} for c in wanted}
    for f in sorted((run / "data").glob("*.mcap")):
        try:
            fh = open(f, "rb")  # noqa: SIM115
        except OSError as exc:
            raise SystemExit(f"cannot open {f}: {exc}") from exc
        with fh:
            reader = make_reader(fh, decoder_factories=[DecoderFactory()])
            for _s, ch, _m, msg in reader.iter_decoded_messages(topics=list(topic_to_cam)):
                cam = topic_to_cam[ch.topic]
                if msg.rgb.header.seq in wanted[cam]:
                    out[cam][msg.rgb.header.seq] = bytes(msg.rgb.data)
    return out


def model_inputs(run: Path, cameras: list[str], size_hw, cam_seq: np.ndarray) -> np.ndarray:
    """(N, n_cam, H, W, 3) uint8 model inputs rebuilt from the recording."""
    from yam_infer.data.nedf3 import decode_jpeg_rgb
    from yam_infer.policy.policy import preprocess_frame

    wanted = {c: set(int(x) for x in cam_seq[:, i]) for i, c in enumerate(cameras)}
    jpegs = load_frames(run, wanted)
    imgs = []
    for row in cam_seq:
        per_cam = []
        for i, c in enumerate(cameras):
            data = jpegs[c].get(int(row[i]))
            if data is None:
                raise SystemExit(f"frame {c} #{int(row[i])} not in the recording (dropped?)")
            per_cam.append(preprocess_frame(decode_jpeg_rgb(data), size_hw, jpeg_roundtrip=False))
        imgs.append(np.stack(per_cam))
    return np.stack(imgs)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("run", type=Path)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    try:
        meta = json.loads((args.run / "meta.json").read_text())
        pol = np.load(args.run / "policy.npz")
    except (OSError, ValueError) as exc:
        raise SystemExit(f"incomplete run dir {args.run}: {exc}") from exc
    if "cam_seq" not in pol:
        raise SystemExit("old recording format (no frame references) - cannot replay")
    from yam_infer.policy.policy import Policy

    policy = Policy(args.ckpt, args.device)
    imgs = model_inputs(args.run, policy.cameras, policy.image_size, pol["cam_seq"])
    task = meta["task"]
    diffs = []
    for k in range(len(pol["t"])):
        out = policy.predict_prepared(imgs[k], pol["qpos"][k].astype(np.float64), task)
        diffs.append(np.abs(out - pol["chunk"][k]).max())
    d = np.asarray(diffs)
    print(
        json.dumps(
            {
                "policy_steps": len(d),
                "max_abs_diff": float(d.max()),
                "median_abs_diff": float(np.median(d)),
                "recorded_device": meta.get("policy", {}).get("device"),
                "replay_device": str(policy.device),
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
