#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Resolve robot transport fields from exact ``cyclo_policy.yaml`` names."""

from __future__ import annotations

import logging
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


class IoMappingMixin:
    """Robot wiring resolved from the loaded checkpoint contract."""

    def _init_robot(self, robot_type: str) -> None:
        if self._policy_contract is None:
            raise RuntimeError("policy contract must be loaded before robot initialization")
        self._robot = RobotClient(robot_type)

        self._cameras = self._resolve_camera_mappings(
            self._robot.camera_names,
            self._policy_contract.cameras,
        )
        self._camera_contracts = {
            camera.source: camera for camera in self._policy_contract.cameras
        }

        groups = self._robot._config.get("joint_groups", {})
        sensors = self._robot._config.get("sensors", {})
        self._state_sources = self._resolve_state_sources(
            groups,
            self._policy_contract.state_features,
            has_odom="odom" in sensors,
        )
        action_groups = getattr(self._robot, "_action_groups", {})
        self._action_keys = self._resolve_action_keys(
            action_groups,
            self._policy_contract.action_features,
            self._policy_contract.inactive_actions,
        )

        if not self._robot.wait_for_ready(timeout=10.0):
            raise RuntimeError(f"robot sensors were not ready for robot_type={robot_type}")
        logger.info(
            "Robot ready: cameras=%s state=%s action_groups=%s",
            list(self._cameras),
            [source.feature for source in self._state_sources],
            self._action_keys,
        )

    def _teardown_robot(self) -> None:
        if self._robot is not None:
            try:
                self._robot.close()
            except Exception:
                pass
            self._robot = None

    @staticmethod
    def _resolve_camera_mappings(
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

    @staticmethod
    def _resolve_state_sources(
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
                        f"policy state feature {feature!r} requires observation.state mobile/odom"
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
                    candidates.append((bool(config.get("parent")), group_name, joint_names.index(feature)))
            synthetic = [candidate for candidate in candidates if candidate[0]]
            if synthetic:
                candidates = synthetic
            if len(candidates) != 1:
                names = [candidate[1] for candidate in candidates]
                raise RuntimeError(
                    f"policy state feature {feature!r} has {len(candidates)} exact robot sources: {names}"
                )
            _, group_name, index = candidates[0]
            resolved.append(StateSource(feature, "joint", group_name, index))
        return resolved

    @staticmethod
    def _resolve_action_keys(
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
                if joint_names and tuple(joint_names) == ordered[offset : offset + len(joint_names)]:
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
                raise RuntimeError(f"policy action group {name!r} is both active and inactive")
            active.append(name)
            offset += len(joint_names)
        return active
