"""Review one recorded inference run (see yampilot.recorder).

    python -m yampilot.tools.review runs_infer/<run> [--video]

Writes into the run dir:
    review.png        commanded vs actual joints, policy latency, gripper
    review.mp4        (with --video) camera frames + predicted right-gripper chunk overlay
Prints a short health summary (latency, safety clamps, tracking error).
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

    t0 = cmd["t"][0]
    tc = cmd["t"] - t0
    tp = pol["t"] - t0
    lat = pol["latency_ms"]
    track = np.nanmean(np.abs(cmd["target_safe"][:, 7:13] - cmd["qpos"][:, 7:13]))
    modified = np.abs(cmd["target_safe"] - cmd["target_raw"]) > 1e-4
    groups = {"right_arm": slice(7, 13), "right_gripper": slice(13, 14), "left_all(frozen by default)": slice(0, 7)}
    clipped = {k: float(np.nanmean(modified[:, g])) for k, g in groups.items()}
    print(json.dumps({
        "task": meta.get("task"), "end_reason": meta.get("end_reason"), "duration_s": meta.get("duration_s"),
        "policy_steps": int(len(tp)), "policy_hz": float(len(tp) / max(tp[-1], 1e-6)) if len(tp) else 0,
        "latency_ms_p50": float(np.median(lat)) if len(lat) else None,
        "latency_ms_p95": float(np.percentile(lat, 95)) if len(lat) else None,
        "safety_clamped": meta.get("safety_clamped"), "safety_rate_limited": meta.get("safety_rate_limited"),
        "frac_commands_changed_by_safety_or_freeze": clipped,
        "mean_tracking_error_right_arm_rad": float(track),
    }, indent=1))

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
    for ax in axes.flat[16:]:
        ax.axis("off")
    fig.suptitle(f"{args.run.name}  end: {meta.get('end_reason')}")
    fig.tight_layout()
    fig.savefig(args.run / "review.png", dpi=80)
    print(f"-> {args.run / 'review.png'}")

    if args.video and (args.run / "frames.mp4").exists():
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
            cv2.putText(panel, f"t={tp[i]:.1f}s  lat={lat[i]:.0f}ms  predicted R-gripper (next {len(chunk)} steps)",
                        (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
            out = np.concatenate([frame, panel], 0)
            if writer is None:
                writer = cv2.VideoWriter(str(args.run / "review.mp4"), cv2.VideoWriter.fourcc(*"mp4v"),
                                         meta.get("policy", {}).get("fps", 30), (out.shape[1], out.shape[0]))
            writer.write(out)
            i += 1
        if writer is not None:
            writer.release()
            print(f"-> {args.run / 'review.mp4'}")


if __name__ == "__main__":
    main()
