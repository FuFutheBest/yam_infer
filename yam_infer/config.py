"""configs/run.yaml (run settings) and configs/models.yaml (task -> model) loaders.

Both are strict: missing keys, unknown keys or out-of-range values raise ValueError, so a typo
never silently falls back to a default.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_CONFIG = PROJECT_ROOT / "configs" / "run.yaml"
DEFAULT_MODELS_CONFIG = PROJECT_ROOT / "configs" / "models.yaml"

# section -> key -> (type, check, description of the valid range)
EXECUTION_SCHEMA = {
    "mode": (str, lambda v: v in ("chunk", "ensemble"), "chunk or ensemble"),
    # upper bound = the model's chunk length, checked once the model is loaded
    "horizon": (int, lambda v: 1 <= v <= 400, "1..chunk_size steps"),
    "blend_s": (float, lambda v: 0 <= v <= 2, "0..2 s"),
    "ensemble_m": (float, lambda v: -5 <= v <= 5, "-5..5"),
}
RUN_CONFIG_SCHEMA = {
    "execution": EXECUTION_SCHEMA,
    "episode": {"max_seconds": (float, lambda v: v > 0, "> 0 s")},
}


def _read_yaml(path: Path) -> dict:
    import yaml

    try:
        cfg = yaml.safe_load(Path(path).read_text()) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return cfg


def _check(path: Path, where: str, raw: dict, schema: dict, required: bool) -> dict:
    extra = sorted(set(raw) - set(schema))
    if extra:
        raise ValueError(f"{path}: unknown key(s) {where}.{extra}")
    out = {}
    for key, (typ, ok, rng) in schema.items():
        if key not in raw:
            if required:
                raise ValueError(f"{path}: missing {where}.{key}")
            continue
        value = raw[key]
        if typ is float and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        if not isinstance(value, typ) or isinstance(value, bool) or not ok(value):
            raise ValueError(f"{path}: {where}.{key} = {raw[key]!r}, expected {rng}")
        out[key] = value
    return out


def load_run_config(path: Path = DEFAULT_RUN_CONFIG) -> dict[str, dict]:
    """configs/run.yaml -> {section: {key: value}}; every key required, unknown keys rejected."""
    cfg = _read_yaml(path)
    unknown = sorted(set(cfg) - set(RUN_CONFIG_SCHEMA))
    if unknown:
        raise ValueError(f"{path}: unknown section(s) {unknown}")
    out = {}
    for section, schema in RUN_CONFIG_SCHEMA.items():
        if not isinstance(cfg.get(section), dict):
            raise ValueError(f"{path}: missing section '{section}'")
        out[section] = _check(path, section, cfg[section], schema, required=True)
    return out


FORMATS = ("act_v2", "act_ref")
FORMAT_IDS = {"act_v2": "act_v2", "act_ref": "yam_train/act_ref-v1"}  # value in the checkpoint
ACTION_MODES = ("delta", "abs")


@dataclass(frozen=True)
class ModelConfig:
    """configs/models.yaml `model:` = the ONE model run_robot.sh runs, and how to read it."""

    ckpt: Path
    format: str  # act_v2 | act_ref
    task: str
    action_mode: str  # delta | abs
    compensate_action_shift: bool = False


def load_model_config(path: Path = DEFAULT_MODELS_CONFIG) -> ModelConfig:
    """configs/models.yaml -> ModelConfig.  A relative ckpt path is relative to the project root."""
    cfg = _read_yaml(path)
    if set(cfg) != {"model"} or not isinstance(cfg["model"], dict):
        raise ValueError(f"{path}: expected exactly one section 'model' (comment out the others)")
    raw = cfg["model"]
    required = ("ckpt", "format", "task", "action_mode")
    allowed = {*required, "compensate_action_shift"}
    extra = sorted(set(raw) - allowed)
    if "execution" in extra:
        raise ValueError(f"{path}: model.execution is not used any more - set execution in run.yaml")
    if extra:
        raise ValueError(f"{path}: unknown key(s) model.{extra}")
    for key in required:
        if not isinstance(raw.get(key), str) or not raw[key]:
            raise ValueError(f"{path}: model.{key} is required (a string)")
    if raw["format"] not in FORMATS:
        raise ValueError(f"{path}: model.format = {raw['format']!r}, expected one of {FORMATS}")
    if raw["action_mode"] not in ACTION_MODES:
        raise ValueError(
            f"{path}: model.action_mode = {raw['action_mode']!r}, expected one of {ACTION_MODES}"
        )
    comp = raw.get("compensate_action_shift", False)
    if not isinstance(comp, bool):
        raise ValueError(f"{path}: model.compensate_action_shift must be true/false")
    ckpt = Path(raw["ckpt"])
    return ModelConfig(
        ckpt=ckpt if ckpt.is_absolute() else PROJECT_ROOT / ckpt,
        format=raw["format"],
        task=raw["task"],
        action_mode=raw["action_mode"],
        compensate_action_shift=comp,
    )


def check_model_matches(cfg: ModelConfig, fmt: str, tasks: list[str], action_mode: str) -> None:
    """The file must be what models.yaml says it is (wrong action_mode = wrong targets!)."""
    problems = []
    if FORMAT_IDS[cfg.format] != fmt:
        problems.append(f"format: file is {fmt!r}, models.yaml says {cfg.format!r}")
    if cfg.task not in tasks:
        problems.append(f"task: file was trained for {tasks}, models.yaml says {cfg.task!r}")
    if action_mode != cfg.action_mode:
        problems.append(
            f"action_mode: file was trained with {action_mode!r}, models.yaml says {cfg.action_mode!r}"
        )
    if problems:
        raise ValueError(f"{cfg.ckpt.name} does not match configs/models.yaml: " + "; ".join(problems))
