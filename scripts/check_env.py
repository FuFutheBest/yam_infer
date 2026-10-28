"""Environment self-check (no hardware is touched).  .venv/bin/python scripts/check_env.py"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "third_party" / "i2rt")]

ok = True


def check(label: str, fn) -> None:
    global ok
    try:
        print(f"  ok   {label}: {fn()}")
    except Exception as exc:  # noqa: BLE001 - report everything
        ok = False
        print(f"  FAIL {label}: {type(exc).__name__}: {exc}")


def _torch() -> str:
    import torch

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA (CPU only)"
    return f"{torch.__version__}, {gpu}"


def _versions(*mods: str) -> str:
    return ", ".join(f"{m} {getattr(importlib.import_module(m), '__version__', '?')}" for m in mods)


def _i2rt_model() -> str:
    import mujoco
    from i2rt.robots.utils import (  # pyright: ignore[reportMissingImports] (third_party/i2rt)
        ArmType,
        GripperType,
        combine_arm_and_gripper_xml,
    )

    path = combine_arm_and_gripper_xml(ArmType.YAM, GripperType.CRANK_4310)
    mj_model = getattr(mujoco, "MjModel")  # compiled extension without type stubs
    return f"YAM + crank_4310 MuJoCo model nq={mj_model.from_xml_path(path).nq}"


def _hw_config() -> str:
    from yam_infer.hw.arms import DEFAULT_HW_CONFIG, load_ports
    from yam_infer.hw.cameras import resolve_serials

    return f"ports {load_ports(DEFAULT_HW_CONFIG)}, cameras {resolve_serials(DEFAULT_HW_CONFIG)}"


def _models() -> str:
    found = sorted(p.name for p in (ROOT / "models").glob("*.pt"))
    if not found:
        raise FileNotFoundError("no models/*.pt (copy them from the user PC)")
    return ", ".join(found)


print(f"python {sys.version.split()[0]} ({sys.executable})")
check("torch", _torch)
check("numpy / cv2", lambda: _versions("numpy", "cv2"))
check("pyrealsense2 / portal", lambda: _versions("pyrealsense2", "portal"))
check("i2rt deps", lambda: _versions("can", "mujoco", "pydantic", "tyro", "dm_env"))
check("i2rt SDK", _i2rt_model)
check("hardware config", _hw_config)
check("models", _models)
check("worker module", lambda: importlib.import_module("yam_infer.hw.follower.worker").__name__)
print("ALL OK" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
