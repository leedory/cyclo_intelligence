#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Fail-closed parser for checkpoint-local ``cyclo_policy.yaml`` files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CameraContract:
    source: str
    key: str
    width: int
    height: int


@dataclass(frozen=True)
class SimulationContract:
    environment: str
    randomized_environment: str
    default_reset: str


@dataclass(frozen=True)
class PolicyContract:
    task_id: str
    robot_type: str
    fps: int
    state_features: tuple[str, ...]
    action_features: tuple[str, ...]
    inactive_actions: dict[str, str]
    cameras: tuple[CameraContract, ...]
    simulation: SimulationContract


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"cyclo_policy.yaml: {where} must be a mapping")
    return value


def _strict_keys(value: dict[str, Any], expected: set[str], where: str) -> None:
    actual = set(value)
    if actual != expected:
        raise RuntimeError(
            f"cyclo_policy.yaml: {where} fields mismatch: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _nonempty_string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"cyclo_policy.yaml: {where} must be a non-empty string")
    return value


def _string_list(value: Any, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"cyclo_policy.yaml: {where} must be a non-empty list")
    if not all(isinstance(item, str) and item for item in value):
        raise RuntimeError(f"cyclo_policy.yaml: {where} must contain non-empty strings")
    if len(value) != len(set(value)):
        raise RuntimeError(f"cyclo_policy.yaml: {where} contains duplicates")
    return tuple(value)


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"cyclo_policy.yaml: {where} must be a positive integer")
    return value


def load_policy_contract(model_path: str | Path) -> PolicyContract:
    path = Path(model_path) / "cyclo_policy.yaml"
    if not path.is_file():
        raise RuntimeError(f"checkpoint is missing required policy contract: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"cannot read policy contract {path}: {exc}") from exc

    root = _mapping(payload, "root")
    _strict_keys(
        root,
        {"task", "robot", "policy_hz", "state", "action", "cameras", "simulation"},
        "root",
    )
    task_id = _nonempty_string(root["task"], "task")
    robot_type = _nonempty_string(root["robot"], "robot")
    fps = _positive_int(root["policy_hz"], "policy_hz")
    if fps != 15:
        raise RuntimeError(f"cyclo_policy.yaml: SG2 policy rate must be 15 Hz, got {fps}")

    state = _mapping(root["state"], "state")
    _strict_keys(state, {"names"}, "state")
    state_features = _string_list(state["names"], "state.names")

    action = _mapping(root["action"], "action")
    _strict_keys(action, {"names", "inactive"}, "action")
    action_features = _string_list(action["names"], "action.names")
    inactive = _mapping(action["inactive"], "action.inactive")
    invalid_inactive = {
        name: mode
        for name, mode in inactive.items()
        if not isinstance(name, str) or not name or mode != "hold_current"
    }
    if invalid_inactive:
        raise RuntimeError(
            f"cyclo_policy.yaml: unsupported inactive action entries: {invalid_inactive}"
        )

    cameras_raw = _mapping(root["cameras"], "cameras")
    if not cameras_raw:
        raise RuntimeError("cyclo_policy.yaml: cameras must not be empty")
    cameras: list[CameraContract] = []
    for source, raw in cameras_raw.items():
        source = _nonempty_string(source, "camera source")
        camera = _mapping(raw, f"cameras.{source}")
        _strict_keys(camera, {"key", "width", "height"}, f"cameras.{source}")
        key = _nonempty_string(camera["key"], f"cameras.{source}.key")
        if not key.startswith("observation.images."):
            raise RuntimeError(
                f"cyclo_policy.yaml: cameras.{source}.key must be an observation.images.* key"
            )
        cameras.append(
            CameraContract(
                source=source,
                key=key,
                width=_positive_int(camera["width"], f"cameras.{source}.width"),
                height=_positive_int(camera["height"], f"cameras.{source}.height"),
            )
        )
    if len({camera.key for camera in cameras}) != len(cameras):
        raise RuntimeError("cyclo_policy.yaml: camera policy keys must be unique")

    simulation = _mapping(root["simulation"], "simulation")
    _strict_keys(
        simulation,
        {"environment", "randomized_environment", "default_reset"},
        "simulation",
    )
    default_reset = simulation["default_reset"]
    if default_reset not in {"deterministic", "randomized_evaluation"}:
        raise RuntimeError(
            "cyclo_policy.yaml: simulation.default_reset must be deterministic or "
            "randomized_evaluation"
        )
    simulation_contract = SimulationContract(
        environment=_nonempty_string(simulation["environment"], "simulation.environment"),
        randomized_environment=_nonempty_string(
            simulation["randomized_environment"], "simulation.randomized_environment"
        ),
        default_reset=default_reset,
    )
    return PolicyContract(
        task_id=task_id,
        robot_type=robot_type,
        fps=fps,
        state_features=state_features,
        action_features=action_features,
        inactive_actions=dict(inactive),
        cameras=tuple(cameras),
        simulation=simulation_contract,
    )


def _feature_shape(feature: Any) -> tuple[int, ...]:
    raw = feature.get("shape") if isinstance(feature, dict) else getattr(feature, "shape", ())
    return tuple(raw or ())


def validate_policy_config(contract: PolicyContract, policy_config: Any) -> None:
    input_features = getattr(policy_config, "input_features", None)
    output_features = getattr(policy_config, "output_features", None)
    if not isinstance(input_features, dict) or not isinstance(output_features, dict):
        raise RuntimeError("policy config is missing input_features/output_features")

    state_shape = _feature_shape(input_features.get("observation.state"))
    action_shape = _feature_shape(output_features.get("action"))
    if state_shape != (len(contract.state_features),):
        raise RuntimeError(
            f"policy state shape {state_shape} does not match cyclo_policy.yaml names "
            f"({len(contract.state_features)},)"
        )
    if action_shape != (len(contract.action_features),):
        raise RuntimeError(
            f"policy action shape {action_shape} does not match cyclo_policy.yaml names "
            f"({len(contract.action_features)},)"
        )

    expected_image_keys = {camera.key for camera in contract.cameras}
    actual_image_keys = {key for key in input_features if key.startswith("observation.images.")}
    if actual_image_keys != expected_image_keys:
        raise RuntimeError(
            "policy image keys do not match cyclo_policy.yaml: "
            f"expected={sorted(expected_image_keys)}, actual={sorted(actual_image_keys)}"
        )
    for camera in contract.cameras:
        shape = _feature_shape(input_features[camera.key])
        expected = (3, camera.height, camera.width)
        if shape != expected:
            raise RuntimeError(
                f"policy image shape for {camera.key} is {shape}, expected {expected}"
            )
