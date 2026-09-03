#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0

"""Exact robot I/O resolution for checkpoints carrying ``cyclo_policy.yaml``."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from robot_client import RobotClient

from .policy_contract import CameraContract


logger = logging.getLogger("lerobot_engine")

_ODOM_FEATURES = {
    "linear_x": 0,
    "linear_y": 1,
    "angular_z": 2,
}


@dataclass(frozen=True)
class StateSource:
    feature: str
    kind: str
    group: str | None
    index: int


def init_contract_robot(runtime: Any, robot_type: str) -> None:
    """Attach a robot using only the camera/state/action I/O in its contract."""
    contract = runtime._policy_contract
    camera_sources = tuple(camera.source for camera in contract.cameras)
    runtime._robot = RobotClient(robot_type, camera_names=camera_sources)
    runtime._cameras = resolve_camera_mappings(
        runtime._robot.camera_names,
        contract.cameras,
    )
    runtime._camera_contracts = {
        camera.source: camera for camera in contract.cameras
    }

    groups = runtime._robot._config.get("joint_groups", {})
    sensors = runtime._robot._config.get("sensors", {})
    runtime._state_sources = resolve_state_sources(
        groups,
        contract.state_features,
        has_odom="odom" in sensors,
    )
    runtime._action_keys = resolve_action_keys(
        getattr(runtime._robot, "_action_groups", {}),
        contract.action_features,
        contract.inactive_actions,
    )

    if not wait_for_contract_inputs(
        runtime._robot,
        runtime._cameras,
        runtime._state_sources,
        timeout=10.0,
    ):
        raise RuntimeError(f"robot sensors were not ready for robot_type={robot_type}")
    logger.info(
        "Robot ready from checkpoint contract: cameras=%s state=%s action_groups=%s",
        list(runtime._cameras),
        [source.feature for source in runtime._state_sources],
        runtime._action_keys,
    )


def missing_contract_inputs(
    robot: Any,
    camera_sources: Iterable[str],
    state_sources: Iterable[StateSource],
) -> list[str]:
    """Return only missing observations that the checkpoint actually consumes."""
    sources = tuple(state_sources)
    missing = [
        f"camera:{name}" for name in camera_sources if not robot.is_image_ready(name)
    ]
    joint_groups = dict.fromkeys(
        source.group for source in sources if source.kind == "joint" and source.group
    )
    missing.extend(
        f"joint:{name}" for name in joint_groups if not robot.is_joint_ready(name)
    )
    if any(source.kind == "odom" for source in sources) and not robot.is_sensor_ready("odom"):
        missing.append("sensor:odom")
    return missing


def wait_for_contract_inputs(
    robot: Any,
    camera_sources: Iterable[str],
    state_sources: Iterable[StateSource],
    *,
    timeout: float,
) -> bool:
    """Wait for selected policy observations, not every sensor in the robot profile."""
    cameras = tuple(camera_sources)
    sources = tuple(state_sources)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not missing_contract_inputs(robot, cameras, sources):
            return True
        time.sleep(0.1)
    logger.warning(
        "Timeout waiting for contracted inputs. Missing: %s",
        missing_contract_inputs(robot, cameras, sources),
    )
    return False


def resolve_camera_mappings(
    robot_camera_names: Iterable[str],
    cameras: Iterable[CameraContract],
) -> dict[str, str]:
    available = set(robot_camera_names)
    requested = tuple(cameras)
    missing = [camera.source for camera in requested if camera.source not in available]
    if missing:
        raise RuntimeError(
            "robot camera sources do not match cyclo_policy.yaml: "
            f"missing={missing}, robot={sorted(available)}"
        )
    return {camera.source: camera.key for camera in requested}


def resolve_state_sources(
    groups: Mapping[str, Mapping[str, Any]],
    features: Iterable[str],
    *,
    has_odom: bool,
) -> list[StateSource]:
    resolved: list[StateSource] = []
    for feature in features:
        if feature in _ODOM_FEATURES:
            if not has_odom:
                raise RuntimeError(
                    f"policy state feature {feature!r} requires "
                    "observation.state mobile/odom"
                )
            resolved.append(
                StateSource(feature, "odom", None, _ODOM_FEATURES[feature])
            )
            continue

        candidates: list[tuple[bool, str, int]] = []
        for group_name, config in groups.items():
            if config.get("role") != "follower":
                continue
            joint_names = list(config.get("joint_names", []))
            if feature in joint_names:
                candidates.append(
                    (
                        bool(config.get("parent")),
                        group_name,
                        joint_names.index(feature),
                    )
                )
        synthetic = [candidate for candidate in candidates if candidate[0]]
        if synthetic:
            candidates = synthetic
        if len(candidates) != 1:
            names = [candidate[1] for candidate in candidates]
            raise RuntimeError(
                f"policy state feature {feature!r} has {len(candidates)} "
                f"exact robot sources: {names}"
            )
        _, group_name, index = candidates[0]
        resolved.append(StateSource(feature, "joint", group_name, index))
    return resolved


def resolve_action_keys(
    action_groups: Mapping[str, Mapping[str, Any]],
    features: Iterable[str],
    inactive_actions: Mapping[str, str],
) -> list[str]:
    missing_inactive = sorted(set(inactive_actions) - set(action_groups))
    if missing_inactive:
        raise RuntimeError(
            f"inactive action groups are absent from robot config: {missing_inactive}"
        )

    ordered = tuple(features)
    offset = 0
    active: list[str] = []
    while offset < len(ordered):
        matches: list[tuple[str, list[str]]] = []
        for name, config in action_groups.items():
            joint_names = list(config.get("joint_names", []))
            if (
                joint_names
                and tuple(joint_names)
                == ordered[offset : offset + len(joint_names)]
            ):
                matches.append((name, joint_names))
        if len(matches) != 1:
            raise RuntimeError(
                "policy action names do not resolve to one exact robot action group at "
                f"offset {offset}: remaining={list(ordered[offset:])}, "
                f"matches={[name for name, _ in matches]}"
            )
        name, joint_names = matches[0]
        if name in active:
            raise RuntimeError(f"policy action group {name!r} is repeated")
        if name in inactive_actions:
            raise RuntimeError(
                f"policy action group {name!r} is both active and inactive"
            )
        active.append(name)
        offset += len(joint_names)
    return active
