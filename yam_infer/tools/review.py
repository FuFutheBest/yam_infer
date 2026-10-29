"""Review one recorded inference run (see yam_infer.data.recorder).

    python -m yam_infer.tools.review runs_infer/<run> [--video]

Reads both recording formats:
    new (v2)  data/*.mcap (every camera frame) + policy.npz with cam_seq + episode_stats.json
    old (v1)  frames.mp4 (model input at policy rate) + policy.npz / command.npz
Writes into the run dir:
    review.png        commanded vs actual joints, policy latency (+ image age for v2)
    review.mp4        (with --video) v2: both cameras in real time, frames used by the model
                      outlined, predicted right-gripper chunk below; v1: the old model-input video
Prints a short health summary (latency, safety clamps, tracking error, v2: dropped frames, image age).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NAMES = [f"L{i}" for i in range(6)] + ["L grip"] + [f"R{i}" for i in range(6)] + ["R grip"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--video", action="store_true")
    args = p.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        meta = json.loads((args.run / "meta.json").read_text())
        pol = np.load(args.run / "policy.npz")
        cmd = np.load(args.run / "command.npz")
    except (OSError, ValueError) as exc:
        raise SystemExit(f"incomplete run dir {args.run}: {exc}") from exc
    v2 = "cam_seq" in pol and (args.run / "data").is_dir()
    stats = {}
    if v2:
        try:
            stats = json.loads((args.run / "episode_stats.json").read_text())
        except (OSError, ValueError):
            stats = {}

    t0 = cmd["t"][0]
    tc = cmd["t"] - t0
    tp = pol["t"] - t0
    lat = pol["latency_ms"]
    age = (pol["t_infer"] - pol["t"]) * 1000 if "t_infer" in pol else np.zeros(0)
    track = np.nanmean(np.abs(cmd["target_safe"][:, 7:13] - cmd["qpos"][:, 7:13]))
    modified = np.abs(cmd["target_safe"] - cmd["target_raw"]) > 1e-4
    groups = {
        "right_arm": slice(7, 13),
        "right_gripper": slice(13, 14),
        "left_all": slice(0, 7),
    }
    clipped = {k: float(np.nanmean(modified[:, g])) for k, g in groups.items()}
    print(
        json.dumps(
            {
                "task": meta.get("task"),
                "end_reason": meta.get("end_reason"),
                "duration_s": meta.get("duration_s"),
                "policy_steps": int(len(tp)),
                "policy_hz": float(len(tp) / max(tp[-1], 1e-6)) if len(tp) else 0,
                "latency_ms_p50": float(np.median(lat)) if len(lat) else None,
                "latency_ms_p95": float(np.percentile(lat, 95)) if len(lat) else None,
                "safety_clamped": meta.get("safety_clamped"),
                "safety_rate_limited": meta.get("safety_rate_limited"),
                "frac_commands_changed_by_safety_or_freeze": clipped,
                "mean_tracking_error_right_arm_rad": float(track),
                "format": "v2 (NEDF3 + frame references)" if v2 else "v1 (frames.mp4)",
                **({
                    "image_age_ms_p50": float(np.median(age)) if len(age) else None,
                    "image_age_ms_max": float(age.max()) if len(age) else None,
                    "dropped_camera_frames": stats.get("dropped_camera_frames"),
                    "writer_error": stats.get("writer_error"),
                    "camera_frames": {k.split("/")[3]: v for k, v in stats.get("topic_counts", {}).items()
                                      if "/camera/" in k},
                } if v2 else {}),
            },
            indent=1,
        )
    )

    fig, axes = plt.subplots(3, 6, figsize=(24, 10), sharex=True)
    for j, ax in enumerate(axes.flat[:14]):
        ax.plot(tc, cmd["target_raw"][:, j], "r-", lw=0.8, label="model target")
        ax.plot(tc, cmd["target_safe"][:, j], "g--", lw=0.8, label="sent (after safety)")
        ax.plot(tc, cmd["qpos"][:, j], "b-", lw=0.8, label="follower actual")
        ax.set_title(NAMES[j])
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=7)
    axes.flat[14].plot(tp, lat, ".-")
    axes.flat[14].set_title("policy latency (ms)")
    axes.flat[15].plot(tp[1:], np.diff(tp) * 1000, ".")
    axes.flat[15].set_title("policy period (ms)")
    if len(age) and v2:
        axes.flat[16].plot(tp, age, ".-")
        axes.flat[16].set_title("image age at inference (ms)")
    for ax in axes.flat[17 if (len(age) and v2) else 16 :]:
        ax.axis("off")
    fig.suptitle(f"{args.run.name}  end: {meta.get('end_reason')}")
    fig.tight_layout()
    fig.savefig(args.run / "review.png", dpi=80)
    print(f"-> {args.run / 'review.png'}")

    if args.video and v2:
        video_v2(args.run, meta, pol, t0)
    elif args.video and (args.run / "frames.mp4").exists():
        import cv2

        cap = cv2.VideoCapture(str(args.run / "frames.mp4"))
        writer = None
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok or i >= len(tp):
                break
            h, w = frame.shape[:2]
            panel = np.full((120, w, 3), 255, np.uint8)
            chunk = pol["chunk"][i]
            for k in range(len(chunk) - 1):  # predicted right gripper (0 closed..1 open)
                x0, x1 = int(k * w / len(chunk)), int((k + 1) * w / len(chunk))
                y0, y1 = int(110 - 100 * chunk[k, 13]), int(110 - 100 * chunk[k + 1, 13])
                cv2.line(panel, (x0, y0), (x1, y1), (0, 0, 255), 2)
            cv2.putText(
                panel,
                f"t={tp[i]:.1f}s  lat={lat[i]:.0f}ms  predicted R-gripper (next {len(chunk)} steps)",
                (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
            )
            out = np.concatenate([frame, panel], 0)
            if writer is None:
                writer = cv2.VideoWriter(
                    str(args.run / "review.mp4"),
                    cv2.VideoWriter.fourcc(*"mp4v"),
                    meta.get("policy", {}).get("fps", 30),
                    (out.shape[1], out.shape[0]),
                )
            writer.write(out)
            i += 1
        if writer is not None:
            writer.release()
            print(f"-> {args.run / 'review.mp4'}")


def _gripper_panel(cv2, w: int, chunk: np.ndarray, text: str) -> np.ndarray:
    panel = np.full((120, w, 3), 255, np.uint8)
    for k in range(len(chunk) - 1):  # predicted right gripper (0 closed..1 open)
        x0, x1 = int(k * w / len(chunk)), int((k + 1) * w / len(chunk))
        y0, y1 = int(110 - 100 * chunk[k, 13]), int(110 - 100 * chunk[k + 1, 13])
        cv2.line(panel, (x0, y0), (x1, y1), (0, 0, 255), 2)
    cv2.putText(panel, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return panel


def video_v2(run: Path, meta: dict, pol, t0: float) -> None:
    """Real-time review video from the raw MCAP streams (30 fps, both wrist cameras)."""
    import cv2
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    from yam_infer.data.nedf3_writer import CAMERA_TOPICS

    clock = meta.get("clock", {})
    off = (clock.get("wall_ns", 0) - clock.get("monotonic_ns", 0)) / 1e9
    cams = list(meta.get("policy", {}).get("cameras", ["right_wrist", "left_wrist"]))
    topic_to_cam = {CAMERA_TOPICS[c]: c for c in cams}
    frames: dict[str, list[tuple[float, int, bytes]]] = {c: [] for c in cams}
    for f in sorted((run / "data").glob("*.mcap")):
        try:
            fh = open(f, "rb")  # noqa: SIM115
        except OSError as exc:
            raise SystemExit(f"cannot open {f}: {exc}") from exc
        with fh:
            reader = make_reader(fh, decoder_factories=[DecoderFactory()])
            for _s, ch, m, msg in reader.iter_decoded_messages(topics=list(topic_to_cam)):
                frames[topic_to_cam[ch.topic]].append(
                    (m.log_time / 1e9 - off, int(msg.rgb.header.seq), bytes(msg.rgb.data)))
    if not all(frames.values()):
        print("no camera frames in data/*.mcap -> no video")
        return
    used = {c: set(int(x) for x in pol["cam_seq"][:, i]) for i, c in enumerate(cams)}
    times = {c: np.asarray([x[0] for x in v]) for c, v in frames.items()}
    ref = cams[0]
    writer = None
    for t, seq, data in frames[ref]:
        tiles = []
        for c in cams:
            j = int(np.argmin(np.abs(times[c] - t)))
            ct, cseq, cdata = frames[c][j]
            decoded = cv2.imdecode(np.frombuffer(cdata, np.uint8), cv2.IMREAD_COLOR)
            if decoded is None:
                continue
            img = cv2.resize(np.asarray(decoded), (320, 240))
            if cseq in used[c]:  # this exact frame went into the model
                cv2.rectangle(img, (1, 1), (318, 238), (0, 255, 0), 3)
            cv2.putText(img, f"{c} #{cseq}", (5, 232), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            tiles.append(img)
        frame = np.concatenate(tiles, axis=1)
        k = int(np.searchsorted(pol["t"], t, side="right")) - 1  # latest prediction at t
        text = f"t={t - t0:.2f}s"
        if k >= 0:
            text += f"  plan #{k} from t={pol['t'][k] - t0:.2f}s  lat={pol['latency_ms'][k]:.0f}ms"
            panel = _gripper_panel(cv2, frame.shape[1], pol["chunk"][k], text + "  R-gripper plan")
        else:
            panel = _gripper_panel(cv2, frame.shape[1], np.zeros((2, 14)), text + "  (no plan yet)")
        out = np.concatenate([frame, panel], 0)
        if writer is None:
            writer = cv2.VideoWriter(str(run / "review.mp4"), cv2.VideoWriter.fourcc(*"mp4v"), 30,
                                     (out.shape[1], out.shape[0]))
        writer.write(out)
    if writer is not None:
        writer.release()
        print(f"-> {run / 'review.mp4'}  (real time; green frame = model input)")


if __name__ == "__main__":
    main()
