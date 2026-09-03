#!/usr/bin/env python3
"""Validate and run SG2 ACT training from a small YAML recipe.

The recipe is the handoff between a LeRobot dataset and the policy runtime.  It
records exact state/action ordering, camera layout, and timing next to the ACT
training options. The launcher writes a concise ``cyclo_policy.yaml`` beside
every complete checkpoint and keeps the full resolved recipe at the run root.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
HOST_WORKSPACE = REPO_ROOT / "docker" / "workspace"
NOISE_TRAINER = Path(__file__).with_name("lerobot_train_with_state_noise.py")
STAGED_NOISE_TRAINER = HOST_WORKSPACE / "training" / NOISE_TRAINER.name
CONTAINER_NOISE_TRAINER = Path("/workspace/training") / NOISE_TRAINER.name

ALLOWED_NORMALIZATION = {"IDENTITY", "MEAN_STD", "MIN_MAX", "QUANTILES", "QUANTILE10"}
ALLOWED_MIXED_PRECISION = {"no", "fp16", "bf16"}
ALLOWED_INACTIVE_MODES = {"hold_current"}
ALLOWED_ORIENTATIONS = {"landscape", "portrait"}
ALLOWED_OPTIMIZERS = {"adam", "adamw", "sgd"}
ALLOWED_SCHEDULERS = {
    "none",
    "constant_with_warmup",
    "cosine_annealing_with_warmup",
    "cosine_decay_with_warmup",
    "diffuser",
}
ALLOWED_NOISE_SPACES = {"normalized", "raw"}
ALLOWED_RESET_PROFILES = {"deterministic", "randomized_evaluation"}


class RecipeError(ValueError):
    """Raised when a recipe or its dataset violates the training contract."""


@dataclass(frozen=True)
class ResolvedRecipe:
    source: Path
    payload: dict[str, Any]
    dataset_host_root: Path
    output_host_root: Path
    state_features: tuple[str, ...]
    action_features: tuple[str, ...]
    camera_inputs: tuple[str, ...]
    noise_indices: tuple[int, ...]
    noise_std: tuple[float, ...]


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RecipeError(f"{where} must be a mapping")
    return value


def _list(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise RecipeError(f"{where} must be a list")
    return value


def _string_list(value: Any, where: str, *, allow_empty: bool = False) -> list[str]:
    values = _list(value, where)
    if not all(isinstance(item, str) and item for item in values):
        raise RecipeError(f"{where} must contain non-empty strings")
    if not allow_empty and not values:
        raise RecipeError(f"{where} must not be empty")
    if len(values) != len(set(values)):
        raise RecipeError(f"{where} contains duplicates")
    return values


def _positive_int(value: Any, where: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecipeError(f"{where} must be an integer")
    lower = 0 if allow_zero else 1
    if value < lower:
        qualifier = "non-negative" if allow_zero else "positive"
        raise RecipeError(f"{where} must be {qualifier}")
    return value


def _finite_number(value: Any, where: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecipeError(f"{where} must be numeric")
    result = float(value)
    if not (-float("inf") < result < float("inf")):
        raise RecipeError(f"{where} must be finite")
    if minimum is not None and result < minimum:
        raise RecipeError(f"{where} must be >= {minimum}")
    return result


def _container_path_to_host(path: str) -> Path:
    container_path = Path(path)
    try:
        relative = container_path.relative_to("/workspace")
    except ValueError as exc:
        raise RecipeError(
            f"container path must be below /workspace so it is visible in the policy container: {path}"
        ) from exc
    return HOST_WORKSPACE / relative


def load_recipe(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise RecipeError(f"recipe does not exist: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RecipeError("recipe root must be a mapping")
    return payload


def _resolve_component_features(policy_io: dict[str, Any], field: str) -> tuple[str, ...]:
    components = _mapping(policy_io.get("components"), "policy_io.components")
    selected = _string_list(policy_io.get(field), f"policy_io.{field}")
    unknown = sorted(set(selected) - set(components))
    if unknown:
        raise RecipeError(f"policy_io.{field} contains unknown components: {unknown}")

    resolved: list[str] = []
    for component in selected:
        names = _string_list(components[component], f"policy_io.components.{component}")
        resolved.extend(names)
    if len(resolved) != len(set(resolved)):
        raise RecipeError(f"policy_io.{field} resolves to duplicate feature names")
    return tuple(resolved)


def _validate_policy_io(
    recipe: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    task = _mapping(recipe.get("task"), "task")
    if not isinstance(task.get("id"), str) or not task["id"]:
        raise RecipeError("task.id must be a non-empty string")
    robot_type = task.get("robot_type")
    if not isinstance(robot_type, str) or not robot_type:
        raise RecipeError("task.robot_type must be a non-empty string")

    policy_io = _mapping(recipe.get("policy_io"), "policy_io")
    if _finite_number(policy_io.get("fps"), "policy_io.fps") != 15.0:
        raise RecipeError("SG2 policy_io.fps must be 15")

    components = _mapping(policy_io.get("components"), "policy_io.components")
    all_features: list[str] = []
    for component, names in components.items():
        if not isinstance(component, str) or not component:
            raise RecipeError("policy_io component names must be non-empty strings")
        all_features.extend(_string_list(names, f"policy_io.components.{component}"))
    if len(all_features) != len(set(all_features)):
        raise RecipeError("policy_io.components reuse a feature name")

    state_features = _resolve_component_features(policy_io, "state_components")
    action_features = _resolve_component_features(policy_io, "action_components")

    inactive = _mapping(policy_io.get("inactive_actions", {}), "policy_io.inactive_actions")
    active_components = set(_string_list(policy_io["action_components"], "policy_io.action_components"))
    required_inactive = set(components) - active_components
    if set(inactive) != required_inactive:
        missing = sorted(required_inactive - set(inactive))
        extra = sorted(set(inactive) - required_inactive)
        raise RecipeError(
            "policy_io.inactive_actions must cover exactly the components not controlled by the policy "
            f"(missing={missing}, extra={extra})"
        )
    bad_modes = {name: mode for name, mode in inactive.items() if mode not in ALLOWED_INACTIVE_MODES}
    if bad_modes:
        raise RecipeError(f"unsupported inactive action modes: {bad_modes}")

    cameras = _mapping(policy_io.get("cameras"), "policy_io.cameras")
    if not cameras:
        raise RecipeError("policy_io.cameras must not be empty")
    camera_inputs = tuple(_string_list(policy_io.get("camera_inputs"), "policy_io.camera_inputs"))
    unknown_camera_inputs = sorted(set(camera_inputs) - set(cameras))
    if unknown_camera_inputs:
        raise RecipeError(
            f"policy_io.camera_inputs contains unknown cameras: {unknown_camera_inputs}"
        )
    camera_keys: list[str] = []
    camera_sources: list[str] = []
    for name, camera_raw in cameras.items():
        if not isinstance(name, str) or not name:
            raise RecipeError("policy_io camera names must be non-empty strings")
        where = f"policy_io.cameras.{name}"
        camera = _mapping(camera_raw, where)
        key = camera.get("key")
        if not isinstance(key, str) or not key.startswith("observation.images."):
            raise RecipeError(f"{where}.key must be an observation.images.* key")
        camera_keys.append(key)
        source = camera.get("source")
        if not isinstance(source, str) or not source:
            raise RecipeError(f"{where}.source must be a non-empty robot camera name")
        camera_sources.append(source)
        _positive_int(camera.get("width"), f"{where}.width")
        _positive_int(camera.get("height"), f"{where}.height")
        orientation = camera.get("orientation")
        if orientation not in ALLOWED_ORIENTATIONS:
            raise RecipeError(
                f"{where}.orientation must be one of {sorted(ALLOWED_ORIENTATIONS)}"
            )
        if camera.get("upright") is not True:
            raise RecipeError(f"{where}.upright must be true")
    if len(camera_keys) != len(set(camera_keys)):
        raise RecipeError("policy_io.cameras contains duplicate keys")
    if len(camera_sources) != len(set(camera_sources)):
        raise RecipeError("policy_io.cameras contains duplicate sources")
    return state_features, action_features, camera_inputs


def _selected_cameras(recipe: dict[str, Any]) -> list[dict[str, Any]]:
    policy_io = recipe["policy_io"]
    return [policy_io["cameras"][name] for name in policy_io["camera_inputs"]]


def _validate_policy(recipe: dict[str, Any]) -> None:
    policy = _mapping(recipe.get("policy"), "policy")
    if policy.get("type") != "act":
        raise RecipeError("policy.type must be act")
    if _positive_int(policy.get("n_obs_steps"), "policy.n_obs_steps") != 1:
        raise RecipeError("this ACT implementation supports policy.n_obs_steps=1 only")
    chunk_size = _positive_int(policy.get("chunk_size"), "policy.chunk_size")
    action_steps = _positive_int(policy.get("n_action_steps"), "policy.n_action_steps")
    if action_steps > chunk_size:
        raise RecipeError("policy.n_action_steps must not exceed policy.chunk_size")
    dim_model = _positive_int(policy.get("dim_model"), "policy.dim_model")
    n_heads = _positive_int(policy.get("n_heads"), "policy.n_heads")
    if dim_model % n_heads:
        raise RecipeError("policy.dim_model must be divisible by policy.n_heads")
    for field in (
        "dim_feedforward",
        "n_encoder_layers",
        "n_decoder_layers",
        "latent_dim",
        "n_vae_encoder_layers",
    ):
        _positive_int(policy.get(field), f"policy.{field}")
    _finite_number(policy.get("dropout"), "policy.dropout", minimum=0.0)
    _finite_number(policy.get("kl_weight"), "policy.kl_weight", minimum=0.0)
    temporal = policy.get("temporal_ensemble_coeff")
    if temporal is not None:
        _finite_number(temporal, "policy.temporal_ensemble_coeff", minimum=0.0)
        if action_steps != 1:
            raise RecipeError("temporal ensembling requires policy.n_action_steps=1")
    normalization = _mapping(policy.get("normalization"), "policy.normalization")
    if set(normalization) != {"VISUAL", "STATE", "ACTION"}:
        raise RecipeError("policy.normalization must define VISUAL, STATE, and ACTION")
    invalid = {key: value for key, value in normalization.items() if value not in ALLOWED_NORMALIZATION}
    if invalid:
        raise RecipeError(f"unsupported normalization modes: {invalid}")


def _validate_optimization(recipe: dict[str, Any]) -> None:
    optimizer = _mapping(recipe.get("optimizer"), "optimizer")
    mode = optimizer.get("mode")
    if mode not in {"policy_preset", "custom"}:
        raise RecipeError("optimizer.mode must be policy_preset or custom")

    preset = _mapping(optimizer.get("policy_preset"), "optimizer.policy_preset")
    for field in ("lr", "lr_backbone", "weight_decay"):
        _finite_number(preset.get(field), f"optimizer.policy_preset.{field}", minimum=0.0)

    custom = _mapping(optimizer.get("custom"), "optimizer.custom")
    optimizer_type = custom.get("type")
    if optimizer_type not in ALLOWED_OPTIMIZERS:
        raise RecipeError(f"optimizer.custom.type must be one of {sorted(ALLOWED_OPTIMIZERS)}")
    for field in ("lr", "weight_decay", "grad_clip_norm"):
        _finite_number(custom.get(field), f"optimizer.custom.{field}", minimum=0.0)
    if optimizer_type in {"adam", "adamw"}:
        betas = _list(custom.get("betas"), "optimizer.custom.betas")
        if len(betas) != 2:
            raise RecipeError("optimizer.custom.betas must contain two values")
        for index, beta in enumerate(betas):
            number = _finite_number(beta, f"optimizer.custom.betas[{index}]", minimum=0.0)
            if number >= 1.0:
                raise RecipeError(f"optimizer.custom.betas[{index}] must be < 1")
        _finite_number(custom.get("eps"), "optimizer.custom.eps", minimum=0.0)
    if optimizer_type == "sgd":
        for field in ("momentum", "dampening"):
            _finite_number(custom.get(field, 0.0), f"optimizer.custom.{field}", minimum=0.0)
        if not isinstance(custom.get("nesterov", False), bool):
            raise RecipeError("optimizer.custom.nesterov must be boolean")

    scheduler = _mapping(recipe.get("scheduler"), "scheduler")
    scheduler_type = scheduler.get("type")
    if scheduler_type not in ALLOWED_SCHEDULERS:
        raise RecipeError(f"scheduler.type must be one of {sorted(ALLOWED_SCHEDULERS)}")
    warmup = _positive_int(
        scheduler.get("num_warmup_steps", 0), "scheduler.num_warmup_steps", allow_zero=True
    )
    if mode == "policy_preset" and scheduler_type != "none":
        raise RecipeError("ACT policy preset has no scheduler; set optimizer.mode=custom")
    if mode == "custom" and scheduler_type == "none":
        raise RecipeError("explicit LeRobot optimizer mode requires a scheduler")
    if scheduler_type == "cosine_decay_with_warmup":
        _positive_int(scheduler.get("num_decay_steps"), "scheduler.num_decay_steps")
        _finite_number(scheduler.get("peak_lr"), "scheduler.peak_lr", minimum=0.0)
        _finite_number(scheduler.get("decay_lr"), "scheduler.decay_lr", minimum=0.0)
    if scheduler_type == "diffuser":
        name = scheduler.get("name")
        if not isinstance(name, str) or not name:
            raise RecipeError("scheduler.name is required for the diffuser scheduler")
    del warmup


def _validate_training(recipe: dict[str, Any]) -> None:
    dataset = _mapping(recipe.get("dataset"), "dataset")
    for field in ("repo_id", "root"):
        if not isinstance(dataset.get(field), str) or not dataset[field]:
            raise RecipeError(f"dataset.{field} must be a non-empty string")
    _container_path_to_host(dataset["root"])
    eval_split = _finite_number(dataset.get("eval_split"), "dataset.eval_split", minimum=0.0)
    if eval_split >= 1.0:
        raise RecipeError("dataset.eval_split must be < 1")

    training = _mapping(recipe.get("training"), "training")
    for field in ("steps", "batch_size", "num_workers", "prefetch_factor", "save_freq", "log_freq"):
        _positive_int(training.get(field), f"training.{field}")
    for field in ("env_eval_freq", "eval_steps", "max_eval_samples"):
        _positive_int(training.get(field), f"training.{field}", allow_zero=True)
    if training["eval_steps"] > 0 and eval_split == 0.0:
        raise RecipeError("training.eval_steps > 0 requires dataset.eval_split > 0")
    _finite_number(training.get("tolerance_s"), "training.tolerance_s", minimum=0.0)
    if training.get("mixed_precision") not in ALLOWED_MIXED_PRECISION:
        raise RecipeError(
            f"training.mixed_precision must be one of {sorted(ALLOWED_MIXED_PRECISION)}"
        )
    if not isinstance(training.get("seed"), int) or isinstance(training.get("seed"), bool):
        raise RecipeError("training.seed must be an integer")
    for field in ("cudnn_deterministic", "persistent_workers", "save_checkpoint", "resume"):
        if not isinstance(training.get(field), bool):
            raise RecipeError(f"training.{field} must be boolean")
    if training["resume"]:
        resume_config = training.get("resume_config")
        if not isinstance(resume_config, str) or not resume_config:
            raise RecipeError("training.resume_config is required when training.resume=true")
    output = training.get("output_dir")
    if not isinstance(output, str) or not output:
        raise RecipeError("training.output_dir must be a non-empty string")
    _container_path_to_host(output)


def _validate_simulation(recipe: dict[str, Any]) -> None:
    simulation = _mapping(recipe.get("simulation"), "simulation")
    for field in ("environment", "randomized_environment"):
        if not isinstance(simulation.get(field), str) or not simulation[field]:
            raise RecipeError(f"simulation.{field} must be a non-empty registered Gym ID")
    if simulation.get("default_reset") not in ALLOWED_RESET_PROFILES:
        raise RecipeError(
            f"simulation.default_reset must be one of {sorted(ALLOWED_RESET_PROFILES)}"
        )


def _resolve_noise(
    recipe: dict[str, Any], state_features: tuple[str, ...]
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    noise = _mapping(recipe.get("state_noise"), "state_noise")
    if not isinstance(noise.get("enabled"), bool):
        raise RecipeError("state_noise.enabled must be boolean")
    if noise.get("space") not in ALLOWED_NOISE_SPACES:
        raise RecipeError(f"state_noise.space must be one of {sorted(ALLOWED_NOISE_SPACES)}")
    if noise.get("distribution") != "gaussian":
        raise RecipeError("state_noise.distribution must be gaussian")
    by_feature = _mapping(noise.get("std_by_feature"), "state_noise.std_by_feature")
    unknown = sorted(set(by_feature) - set(state_features))
    if unknown:
        raise RecipeError(f"state_noise.std_by_feature contains features absent from policy state: {unknown}")
    scalar = noise.get("std")
    if scalar is not None:
        scalar = _finite_number(scalar, "state_noise.std", minimum=0.0)
    resolved_by_name: dict[str, float] = {}
    if scalar is not None and scalar > 0.0:
        resolved_by_name.update({name: scalar for name in state_features})
    for name, raw_std in by_feature.items():
        std = _finite_number(raw_std, f"state_noise.std_by_feature.{name}", minimum=0.0)
        if std == 0.0:
            resolved_by_name.pop(name, None)
        else:
            resolved_by_name[name] = std
    resolved = [
        (index, resolved_by_name[name])
        for index, name in enumerate(state_features)
        if name in resolved_by_name
    ]
    if noise["enabled"] and not resolved:
        raise RecipeError("enabled state noise requires at least one positive per-feature standard deviation")
    return tuple(index for index, _ in resolved), tuple(std for _, std in resolved)


def validate_dataset(
    recipe: dict[str, Any],
    dataset_host_root: Path,
    state_features: tuple[str, ...],
    action_features: tuple[str, ...],
) -> dict[str, Any]:
    info_path = dataset_host_root / "meta" / "info.json"
    if not info_path.is_file():
        raise RecipeError(f"LeRobot metadata is missing: {info_path}")
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RecipeError(f"cannot read LeRobot metadata {info_path}: {exc}") from exc

    task = _mapping(recipe["task"], "task")
    policy_io = _mapping(recipe["policy_io"], "policy_io")
    if info.get("robot_type") != task["robot_type"]:
        raise RecipeError(
            f"dataset robot_type mismatch: expected {task['robot_type']!r}, got {info.get('robot_type')!r}"
        )
    if float(info.get("fps", 0)) != float(policy_io["fps"]):
        raise RecipeError(f"dataset fps mismatch: expected {policy_io['fps']}, got {info.get('fps')}")

    features = _mapping(info.get("features"), "dataset info.features")
    for key, expected in (("observation.state", state_features), ("action", action_features)):
        feature = _mapping(features.get(key), f"dataset info.features.{key}")
        actual_names = feature.get("names")
        if actual_names != list(expected):
            raise RecipeError(
                f"dataset {key} names/order mismatch:\n"
                f"  expected={list(expected)}\n"
                f"  actual={actual_names}"
            )
        if feature.get("shape") != [len(expected)]:
            raise RecipeError(
                f"dataset {key} shape mismatch: expected {[len(expected)]}, got {feature.get('shape')}"
            )

    expected_camera_keys: set[str] = set()
    for camera in _selected_cameras(recipe):
        key = camera["key"]
        expected_camera_keys.add(key)
        feature = _mapping(features.get(key), f"dataset info.features.{key}")
        expected_shape = [3, camera["height"], camera["width"]]
        if feature.get("shape") != expected_shape:
            raise RecipeError(
                f"dataset camera shape mismatch for {key}: expected {expected_shape}, got {feature.get('shape')}"
            )
    actual_camera_keys = {key for key in features if key.startswith("observation.images.")}
    allowed_camera_keys = {camera["key"] for camera in policy_io["cameras"].values()}
    missing_camera_keys = expected_camera_keys - actual_camera_keys
    unexpected_camera_keys = actual_camera_keys - allowed_camera_keys
    if missing_camera_keys or unexpected_camera_keys:
        raise RecipeError(
            "dataset camera keys do not satisfy the selected inputs/catalog: "
            f"missing_selected={sorted(missing_camera_keys)}, "
            f"unexpected={sorted(unexpected_camera_keys)}, actual={sorted(actual_camera_keys)}"
        )
    return info


def resolve_recipe(
    source: str | Path,
    *,
    dataset_root: str | None = None,
    output_dir: str | None = None,
    validate_data: bool = True,
) -> ResolvedRecipe:
    source_path = Path(source).expanduser().resolve()
    recipe = copy.deepcopy(load_recipe(source_path))
    recipe_dataset = _mapping(recipe.get("dataset"), "dataset")
    if Path(recipe_dataset.get("root", "")).name != str(recipe_dataset.get("repo_id", "")).rsplit("/", 1)[-1]:
        raise RecipeError("dataset.repo_id name must match the final component of dataset.root")
    if dataset_root is not None:
        _mapping(recipe.get("dataset"), "dataset")["root"] = dataset_root
    if output_dir is not None:
        _mapping(recipe.get("training"), "training")["output_dir"] = output_dir

    state_features, action_features, camera_inputs = _validate_policy_io(recipe)
    _validate_policy(recipe)
    _validate_optimization(recipe)
    _validate_training(recipe)
    _validate_simulation(recipe)
    noise_indices, noise_std = _resolve_noise(recipe, state_features)

    dataset_host_root = _container_path_to_host(recipe["dataset"]["root"])
    output_host_root = _container_path_to_host(recipe["training"]["output_dir"])
    if validate_data:
        validate_dataset(recipe, dataset_host_root, state_features, action_features)

    recipe["resolved_policy_io"] = {
        "state_features": list(state_features),
        "action_features": list(action_features),
        "state_dim": len(state_features),
        "action_dim": len(action_features),
        "camera_inputs": list(camera_inputs),
        "camera_keys": [camera["key"] for camera in _selected_cameras(recipe)],
    }
    recipe["source_recipe"] = (
        str(source_path.relative_to(REPO_ROOT))
        if source_path.is_relative_to(REPO_ROOT)
        else str(source_path)
    )
    return ResolvedRecipe(
        source=source_path,
        payload=recipe,
        dataset_host_root=dataset_host_root,
        output_host_root=output_host_root,
        state_features=state_features,
        action_features=action_features,
        camera_inputs=camera_inputs,
        noise_indices=noise_indices,
        noise_std=noise_std,
    )


def _cli_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def _flag(name: str, value: Any) -> str:
    return f"--{name}={_cli_value(value)}"


def build_lerobot_args(resolved: ResolvedRecipe) -> list[str]:
    recipe = resolved.payload
    dataset = recipe["dataset"]
    policy = recipe["policy"]
    optimizer = recipe["optimizer"]
    scheduler = recipe["scheduler"]
    training = recipe["training"]
    image_transforms = _mapping(dataset.get("image_transforms"), "dataset.image_transforms")
    input_features = {
        "observation.state": {"type": "STATE", "shape": [len(resolved.state_features)]},
        **{
            camera["key"]: {
                "type": "VISUAL",
                "shape": [3, camera["height"], camera["width"]],
            }
            for camera in _selected_cameras(recipe)
        },
    }

    args = [
        _flag("policy.type", "act"),
        _flag("policy.n_obs_steps", policy["n_obs_steps"]),
        _flag("policy.input_features", input_features),
        _flag("policy.chunk_size", policy["chunk_size"]),
        _flag("policy.n_action_steps", policy["n_action_steps"]),
        _flag("policy.normalization_mapping", policy["normalization"]),
        _flag("policy.vision_backbone", policy["vision_backbone"]),
        _flag("policy.replace_final_stride_with_dilation", policy["replace_final_stride_with_dilation"]),
        _flag("policy.pre_norm", policy["pre_norm"]),
        _flag("policy.dim_model", policy["dim_model"]),
        _flag("policy.n_heads", policy["n_heads"]),
        _flag("policy.dim_feedforward", policy["dim_feedforward"]),
        _flag("policy.feedforward_activation", policy["feedforward_activation"]),
        _flag("policy.n_encoder_layers", policy["n_encoder_layers"]),
        _flag("policy.n_decoder_layers", policy["n_decoder_layers"]),
        _flag("policy.use_vae", policy["use_vae"]),
        _flag("policy.latent_dim", policy["latent_dim"]),
        _flag("policy.n_vae_encoder_layers", policy["n_vae_encoder_layers"]),
        _flag("policy.dropout", policy["dropout"]),
        _flag("policy.kl_weight", policy["kl_weight"]),
        _flag("policy.device", policy["device"]),
        _flag("policy.use_amp", policy["inference_amp"]),
        _flag("policy.push_to_hub", policy["push_to_hub"]),
        _flag("dataset.repo_id", dataset["repo_id"]),
        _flag("dataset.root", dataset["root"]),
        _flag("dataset.eval_split", dataset["eval_split"]),
        _flag("dataset.use_imagenet_stats", dataset["use_imagenet_stats"]),
        _flag("dataset.video_backend", dataset["video_backend"]),
        _flag("dataset.return_uint8", dataset["return_uint8"]),
        _flag("dataset.image_transforms.enable", image_transforms["enable"]),
        _flag("dataset.image_transforms.max_num_transforms", image_transforms["max_num_transforms"]),
        _flag("dataset.image_transforms.random_order", image_transforms["random_order"]),
        _flag("dataset.image_transforms.tfs", image_transforms["transforms"]),
        _flag("output_dir", training["output_dir"]),
        _flag("job_name", training["job_name"]),
        _flag("resume", training["resume"]),
        _flag("seed", training["seed"]),
        _flag("cudnn_deterministic", training["cudnn_deterministic"]),
        _flag("num_workers", training["num_workers"]),
        _flag("batch_size", training["batch_size"]),
        _flag("prefetch_factor", training["prefetch_factor"]),
        _flag("persistent_workers", training["persistent_workers"]),
        _flag("steps", training["steps"]),
        _flag("env_eval_freq", training["env_eval_freq"]),
        _flag("log_freq", training["log_freq"]),
        _flag("eval_steps", training["eval_steps"]),
        _flag("max_eval_samples", training["max_eval_samples"]),
        _flag("tolerance_s", training["tolerance_s"]),
        _flag("save_checkpoint", training["save_checkpoint"]),
        _flag("save_freq", training["save_freq"]),
        _flag("wandb.enable", recipe["wandb"]["enable"]),
        _flag("wandb.project", recipe["wandb"]["project"]),
    ]
    if policy.get("pretrained_backbone_weights") is not None:
        args.append(_flag("policy.pretrained_backbone_weights", policy["pretrained_backbone_weights"]))
    if policy.get("temporal_ensemble_coeff") is not None:
        args.append(_flag("policy.temporal_ensemble_coeff", policy["temporal_ensemble_coeff"]))
    if policy.get("repo_id"):
        args.append(_flag("policy.repo_id", policy["repo_id"]))
    if dataset.get("episodes") is not None:
        args.append(_flag("dataset.episodes", dataset["episodes"]))
    if training["resume"]:
        args.extend([_flag("config_path", training["resume_config"])])

    if optimizer["mode"] == "policy_preset":
        preset = optimizer["policy_preset"]
        args.extend(
            [
                _flag("use_policy_training_preset", True),
                _flag("policy.optimizer_lr", preset["lr"]),
                _flag("policy.optimizer_weight_decay", preset["weight_decay"]),
                _flag("policy.optimizer_lr_backbone", preset["lr_backbone"]),
            ]
        )
    else:
        custom = optimizer["custom"]
        args.extend(
            [
                _flag("use_policy_training_preset", False),
                _flag("optimizer.type", custom["type"]),
                _flag("optimizer.lr", custom["lr"]),
                _flag("optimizer.weight_decay", custom["weight_decay"]),
                _flag("optimizer.grad_clip_norm", custom["grad_clip_norm"]),
            ]
        )
        if custom["type"] in {"adam", "adamw"}:
            args.extend(
                [
                    _flag("optimizer.betas", custom["betas"]),
                    _flag("optimizer.eps", custom["eps"]),
                ]
            )
        if custom["type"] == "sgd":
            args.extend(
                [
                    _flag("optimizer.momentum", custom.get("momentum", 0.0)),
                    _flag("optimizer.dampening", custom.get("dampening", 0.0)),
                    _flag("optimizer.nesterov", custom.get("nesterov", False)),
                ]
            )
        scheduler_type = scheduler["type"]
        args.append(_flag("scheduler.type", scheduler_type))
        args.append(_flag("scheduler.num_warmup_steps", scheduler["num_warmup_steps"]))
        if scheduler_type == "cosine_decay_with_warmup":
            for field in ("num_decay_steps", "peak_lr", "decay_lr"):
                args.append(_flag(f"scheduler.{field}", scheduler[field]))
        elif scheduler_type == "diffuser":
            args.append(_flag("scheduler.name", scheduler["name"]))

    return args


def build_docker_command(resolved: ResolvedRecipe, container: str) -> list[str]:
    noise_enabled = resolved.payload["state_noise"]["enabled"]
    command = [
        "docker",
        "exec",
        "-e",
        f"ACCELERATE_MIXED_PRECISION={resolved.payload['training']['mixed_precision']}",
        "-e",
        "PYTHONUNBUFFERED=1",
    ]
    if noise_enabled:
        noise_payload = {
            "space": resolved.payload["state_noise"]["space"],
            "indices": resolved.noise_indices,
            "std": resolved.noise_std,
            "features": [resolved.state_features[index] for index in resolved.noise_indices],
        }
        command.extend(["-e", f"CYCLO_STATE_NOISE={json.dumps(noise_payload, separators=(',', ':'))}"])
    command.append(container)
    if noise_enabled:
        command.extend(["python3", str(CONTAINER_NOISE_TRAINER)])
    else:
        command.append("lerobot-train")
    command.extend(build_lerobot_args(resolved))
    return command


def policy_manifest(resolved: ResolvedRecipe) -> dict[str, Any]:
    recipe = resolved.payload
    policy_io = recipe["policy_io"]
    cameras = {
        camera["source"]: {
            "key": camera["key"],
            "width": camera["width"],
            "height": camera["height"],
        }
        for camera in _selected_cameras(recipe)
    }
    return {
        "task": recipe["task"]["id"],
        "robot": recipe["task"]["robot_type"],
        "policy_hz": policy_io["fps"],
        "state": {"names": list(resolved.state_features)},
        "action": {
            "names": list(resolved.action_features),
            "inactive": copy.deepcopy(policy_io["inactive_actions"]),
        },
        "cameras": cameras,
        "simulation": copy.deepcopy(recipe["simulation"]),
    }


def _manifest_text(resolved: ResolvedRecipe) -> str:
    return yaml.safe_dump(policy_manifest(resolved), sort_keys=False, allow_unicode=True)


def _resolved_recipe_text(resolved: ResolvedRecipe) -> str:
    return yaml.safe_dump(resolved.payload, sort_keys=False, allow_unicode=True)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def _write_if_changed(path: Path, content: str) -> None:
    try:
        if path.is_file() and path.read_text(encoding="utf-8") == content:
            return
    except OSError:
        pass
    _atomic_write(path, content)


def _complete_checkpoint(pretrained_model_dir: Path) -> bool:
    checkpoint_dir = pretrained_model_dir.parent
    training_state = checkpoint_dir / "training_state" / "training_step.json"
    required = (
        pretrained_model_dir / "config.json",
        pretrained_model_dir / "model.safetensors",
        pretrained_model_dir / "train_config.json",
        training_state,
    )
    if not all(path.is_file() for path in required):
        return False
    try:
        payload = json.loads(training_state.read_text(encoding="utf-8"))
        return isinstance(payload.get("step"), int) and payload["step"] > 0
    except (OSError, json.JSONDecodeError, AttributeError):
        return False


def finalize_contracts(resolved: ResolvedRecipe) -> list[Path]:
    output_root = resolved.output_host_root
    if not output_root.is_dir():
        raise RecipeError(f"training output does not exist: {output_root}")
    content = _manifest_text(resolved)
    written = [output_root / "cyclo_policy.yaml"]
    _write_if_changed(written[0], content)
    _write_if_changed(output_root / "resolved_recipe.yaml", _resolved_recipe_text(resolved))
    checkpoints_root = output_root / "checkpoints"
    if checkpoints_root.is_dir():
        for checkpoint in sorted(checkpoints_root.iterdir()):
            if not checkpoint.is_dir() or not checkpoint.name.isdigit():
                continue
            pretrained = checkpoint / "pretrained_model"
            if _complete_checkpoint(pretrained):
                destination = pretrained / "cyclo_policy.yaml"
                _write_if_changed(destination, content)
                written.append(destination)
    return written


def _print_plan(resolved: ResolvedRecipe, command: list[str]) -> None:
    recipe = resolved.payload
    io = recipe["policy_io"]
    noise = recipe["state_noise"]
    camera_summary = ", ".join(
        f"{name}={io['cameras'][name]['width']}x{io['cameras'][name]['height']}"
        for name in io["camera_inputs"]
    )
    noise_summary = "off"
    if noise["enabled"]:
        noise_summary = (
            f"{noise['space']} Gaussian on {len(resolved.noise_indices)} named features"
        )
    print(f"Task:       {recipe['task']['id']} ({recipe['task']['robot_type']})")
    print(f"Dataset:    {recipe['dataset']['root']}")
    print(f"Policy I/O: {len(resolved.state_features)}D state -> {len(resolved.action_features)}D action")
    print(f"Components: state={io['state_components']} action={io['action_components']}")
    print(f"Inactive:   {io['inactive_actions'] or 'none'}")
    print(f"Cameras:    {camera_summary}; upright; {io['fps']} Hz")
    print(f"Noise:      {noise_summary}")
    print(f"Output:     {recipe['training']['output_dir']}")
    print("Command:")
    print(shlex.join(command))


def _container_running(container: str) -> bool:
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Running}}", container],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def run_recipe(resolved: ResolvedRecipe, container: str) -> list[Path]:
    if not _container_running(container):
        raise RecipeError(
            f"container {container!r} is not running; start it with docker/container.sh start-lerobot"
        )
    training = resolved.payload["training"]
    if resolved.output_host_root.exists() and not training["resume"]:
        raise RecipeError(
            f"output exists and training.resume=false: {resolved.output_host_root}"
        )
    if resolved.payload["state_noise"]["enabled"]:
        STAGED_NOISE_TRAINER.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(NOISE_TRAINER, STAGED_NOISE_TRAINER)

    command = build_docker_command(resolved, container)
    process = subprocess.Popen(command)
    written: list[Path] = []
    while process.poll() is None:
        if resolved.output_host_root.is_dir():
            written = finalize_contracts(resolved)
        time.sleep(2.0)
    if resolved.output_host_root.is_dir():
        written = finalize_contracts(resolved)
    if process.returncode != 0:
        raise RecipeError(
            f"training exited with status {process.returncode}; complete checkpoints were finalized, "
            "but incomplete checkpoint directories were not marked"
        )
    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "plan", "run", "finalize"))
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--dataset-root", help="Override dataset.root with a /workspace path")
    parser.add_argument("--output-dir", help="Override training.output_dir with a /workspace path")
    parser.add_argument(
        "--container",
        default=os.environ.get("LEROBOT_CONTAINER_NAME", "lerobot_server_s2r"),
    )
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Validate recipe structure only; not allowed for run or finalize",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.skip_dataset and args.command in {"run", "finalize"}:
        raise RecipeError("--skip-dataset is only valid with validate or plan")
    resolved = resolve_recipe(
        args.recipe,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        validate_data=not args.skip_dataset,
    )
    command = build_docker_command(resolved, args.container)
    if args.command == "validate":
        print(f"Valid ACT recipe: {resolved.source}")
        return 0
    if args.command == "plan":
        _print_plan(resolved, command)
        return 0
    if args.command == "finalize":
        written = finalize_contracts(resolved)
    else:
        _print_plan(resolved, command)
        written = run_recipe(resolved, args.container)
    print("Wrote policy contract:")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RecipeError as exc:
        print(f"ACT recipe error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
