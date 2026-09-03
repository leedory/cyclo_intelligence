#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""LeRobot engine I/O mapping helpers (IoMappingMixin).

Extracted from ``engine.py`` to keep the core ``LeRobotEngine`` class
focused on the ``InferenceEngine`` API. Mixed into the engine via
multiple inheritance; bind-mounted into the policy container as part
of the ``/app/lerobot_engine/`` package.

Owns:
- ``_init_robot``: create RobotClient + resolve camera / state mappings.
- ``_teardown_robot``: release the RobotClient.
- ``_policy_image_keys``: read the policy's expected image input keys.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable

from .constants import IMAGE_KEY_PREFIX as _IMAGE_KEY_PREFIX

from robot_client import RobotClient


logger = logging.getLogger("lerobot_engine")


_POLICY_IO_MAPPING_FILENAME = "cyclo_policy_io.json"


_CAMERA_SEMANTIC_RE = re.compile(
    r"^cam_(?P<a>left|right|head|wrist)_(?P<b>left|right|head|wrist)$"
)


class IoMappingMixin:
    """Robot wiring — camera / state modality resolution and teardown."""

    def _init_robot(self, robot_type: str) -> None:
        """Create RobotClient + resolve camera / state mappings."""
        self._robot = RobotClient(robot_type)

        # Cameras: only those that match a policy input key
        # ``observation.images.<cam>``. Cameras advertised by the robot
        # but not consumed by the policy are silently ignored — same
        # behavior as GR00TInference.
        policy_image_keys = self._policy_image_keys()
        active = self._resolve_camera_mappings(
            self._robot.camera_names,
            policy_image_keys,
        )
        if not active and policy_image_keys:
            raise RuntimeError(
                "No cameras match the policy's expected input keys: "
                f"policy needs {sorted(policy_image_keys)}, robot has "
                f"{self._robot.camera_names}"
            )
        self._cameras = active

        # State modalities: sorted follower joint groups. We follow the
        # same convention groot uses (sorted modality names map to the
        # training-time concat order). Synthetic per-modality views (with
        # ``parent``) win over their leaf physical group; otherwise the
        # leaf group is used directly.
        groups = self._robot._config.get("joint_groups", {})
        parents = {cfg.get("parent") for cfg in groups.values() if cfg.get("parent")}
        modality_groups = []
        for name, cfg in groups.items():
            if cfg.get("role") != "follower" or not name.startswith("follower_"):
                continue
            if cfg.get("parent"):
                modality_groups.append(name)
            elif name not in parents:
                modality_groups.append(name)
        modalities = sorted(name[len("follower_"):] for name in modality_groups)
        if not modalities:
            raise RuntimeError(
                f"No follower joint groups in robot_type={robot_type}"
            )

        # Mobile is sourced from sensors["odom"] in the new schema —
        # bridge it into observation.state alongside the joint states so
        # policies trained on the legacy physical_ai_server pipeline
        # (with mobile as a 3-vector modality) still see it.
        sensors = self._robot._config.get("sensors", {})
        self._has_mobile_state = "odom" in sensors
        if self._has_mobile_state:
            modalities = sorted(set(modalities) | {"mobile"})

        policy_io = self._load_policy_io_mapping(self._loaded_model_path)
        # Keep checkpoint-specific camera orientation out of the shared robot
        # config. This is intentionally opt-in: absent an entry, every policy
        # continues to use the robot's standard camera metadata.
        self._camera_rotation_overrides = self._resolve_camera_rotation_overrides(
            policy_io,
            active,
        )
        self._state_modalities = self._resolve_policy_modalities(
            modalities,
            self._policy_flat_feature_dim("input_features", "observation.state"),
            policy_io,
            "observation_state_modalities",
            "observation_state_joint_names",
            label="observation.state",
        )
        self._action_keys = self._resolve_policy_modalities(
            modalities,
            self._policy_flat_feature_dim("output_features", "action"),
            policy_io,
            "action_modalities",
            "action_joint_names",
            label="action",
        )

        # Block until at least one frame from each sensor lands. 10 s is
        # generous — typical hardware comes up in <2 s.
        self._robot.wait_for_ready(timeout=10.0)
        logger.info(
            "Robot ready: cameras=%s state_modalities=%s",
            list(self._cameras.keys()),
            self._state_modalities,
        )

    def _teardown_robot(self) -> None:
        if self._robot is not None:
            try:
                self._robot.close()
            except Exception:
                pass
            self._robot = None

    def _policy_image_keys(self) -> set:
        try:
            features = getattr(self._policy.config, "input_features", {}) or {}
            return {k for k in features.keys() if k.startswith(_IMAGE_KEY_PREFIX)}
        except Exception:
            return set()

    @classmethod
    def _resolve_camera_mappings(
        cls,
        robot_camera_names: Iterable[str],
        policy_image_keys: set,
    ) -> Dict[str, str]:
        """Map RobotClient camera names to policy image feature keys.

        The canonical Cyclo camera names are ``cam_<side>_<part>`` such as
        ``cam_left_head``. Some runtime configs or checkpoints may expose
        ``rgb.`` prefixes. Older single-head checkpoints may use
        ``cam_head`` for the left head camera. Exact matches remain preferred.
        """
        camera_names = list(robot_camera_names)
        if not policy_image_keys:
            return {cam: f"{_IMAGE_KEY_PREFIX}{cam}" for cam in camera_names}

        active: Dict[str, str] = {}
        used_policy_keys = set()
        for cam in camera_names:
            exact = f"{_IMAGE_KEY_PREFIX}{cam}"
            candidates = cls._camera_policy_key_candidates(cam)
            matches = sorted(policy_image_keys & candidates)
            if not matches:
                continue

            if exact in matches:
                chosen = exact
            elif len(matches) == 1:
                chosen = matches[0]
            else:
                raise RuntimeError(
                    f"Ambiguous camera mapping for {cam}: matches {matches}"
                )

            if chosen in used_policy_keys:
                raise RuntimeError(
                    f"Policy camera key {chosen} matched multiple robot cameras"
                )
            active[cam] = chosen
            used_policy_keys.add(chosen)

        missing = sorted(policy_image_keys - used_policy_keys)
        if missing:
            raise RuntimeError(
                "Missing camera mappings for policy input keys: "
                f"{missing}; robot has {camera_names}; matched {active}"
            )
        return active

    @staticmethod
    def _load_policy_io_mapping(model_path: str | None) -> dict[str, Any]:
        """Load optional Cyclo joint semantics stored beside a checkpoint.

        LeRobot's ACT config only stores feature dimensions, not the names or
        order of individual joints. A ``cyclo_policy_io.json`` sidecar makes a
        reduced-joint checkpoint unambiguous without changing the shared robot
        configuration. Checkpoint roots may be passed directly, or as the
        standard ``checkpoints/<step>/pretrained_model`` directory.
        """
        if not model_path:
            return {}

        root = Path(model_path)
        model_root = root
        if root.name == "pretrained_model" and root.parent.parent.name == "checkpoints":
            model_root = root.parent.parent.parent
        candidates = [root / _POLICY_IO_MAPPING_FILENAME]
        if model_root != root:
            candidates.append(model_root / _POLICY_IO_MAPPING_FILENAME)
        candidates.append(
            Path(__file__).with_name("model_io_mappings")
            / f"{model_root.name}.json"
        )

        for candidate in candidates:
            if not candidate.is_file():
                continue
            with candidate.open() as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                raise RuntimeError(
                    f"{candidate} must contain a JSON object, got {type(payload).__name__}"
                )
            logger.info("Loaded policy I/O mapping: %s", candidate)
            return payload
        return {}

    @staticmethod
    def _resolve_camera_rotation_overrides(
        policy_io: dict[str, Any],
        active_cameras: Dict[str, str],
    ) -> dict[str, int]:
        """Validate optional per-policy camera rotations.

        ``camera_rotation_deg`` is a map of RobotClient camera name to one of
        0/90/180/270. It exists only in a policy I/O mapping, so it cannot
        change the default camera treatment for unrelated policies.
        """
        declared = policy_io.get("camera_rotation_deg")
        if declared is None:
            return {}
        if not isinstance(declared, dict):
            raise RuntimeError("camera_rotation_deg must be an object")

        overrides: dict[str, int] = {}
        for camera_name, rotation in declared.items():
            if not isinstance(camera_name, str):
                raise RuntimeError("camera_rotation_deg keys must be camera names")
            if camera_name not in active_cameras:
                raise RuntimeError(
                    f"camera_rotation_deg contains inactive camera {camera_name}; "
                    f"active={sorted(active_cameras)}"
                )
            if not isinstance(rotation, int) or rotation not in {0, 90, 180, 270}:
                raise RuntimeError(
                    f"camera_rotation_deg[{camera_name!r}] must be one of 0, 90, 180, 270"
                )
            overrides[camera_name] = rotation
        return overrides

    def _resolve_policy_modalities(
        self,
        available_modalities: Iterable[str],
        expected_dim: int | None,
        policy_io: dict[str, Any],
        modalities_key: str,
        joint_names_key: str,
        *,
        label: str,
    ) -> list[str]:
        """Resolve an ordered model-vector layout from optional sidecar data.

        A dimension alone cannot distinguish ``arm_left`` from ``arm_right``
        when both have the same width. Therefore an incomplete layout is an
        error unless a sidecar declares the intended modalities explicitly.
        """
        available = list(available_modalities)
        declared = policy_io.get(modalities_key)
        if declared is None:
            selected = available
        else:
            if not isinstance(declared, list) or not all(isinstance(item, str) for item in declared):
                raise RuntimeError(f"{modalities_key} must be a list of modality names")
            selected = list(declared)
            if len(set(selected)) != len(selected):
                raise RuntimeError(f"{modalities_key} contains duplicate modalities: {selected}")
            unknown = [item for item in selected if item not in available]
            if unknown:
                raise RuntimeError(
                    f"{modalities_key} contains unavailable robot modalities {unknown}; "
                    f"available={available}"
                )

        widths = {modality: self._modality_width(modality) for modality in selected}
        if any(width <= 0 for width in widths.values()):
            raise RuntimeError(f"Cannot determine widths for {label} modalities: {widths}")
        actual_dim = sum(widths.values())
        if expected_dim is not None and actual_dim != expected_dim:
            source = _POLICY_IO_MAPPING_FILENAME if declared is not None else "robot configuration"
            raise RuntimeError(
                f"{label} expects {expected_dim} values but {source} resolves {actual_dim} "
                f"from {selected}. Add the exact layout to {_POLICY_IO_MAPPING_FILENAME}."
            )

        declared_names = policy_io.get(joint_names_key)
        if declared_names is not None:
            if not isinstance(declared_names, list) or not all(
                isinstance(item, str) for item in declared_names
            ):
                raise RuntimeError(f"{joint_names_key} must be a list of joint names")
            actual_names = [
                name for modality in selected for name in self._modality_joint_names(modality)
            ]
            if declared_names != actual_names:
                raise RuntimeError(
                    f"{joint_names_key} does not match the active robot layout: "
                    f"expected {actual_names}, got {declared_names}"
                )
        return selected

    def _modality_width(self, modality: str) -> int:
        return len(self._modality_joint_names(modality))

    def _modality_joint_names(self, modality: str) -> list[str]:
        """Return a modality's exact model-vector labels in robot-config order."""
        if modality == "mobile":
            cfg = getattr(self._robot, "_action_groups", {}).get("mobile", {})
            return list(cfg.get("joint_names", ["linear_x", "linear_y", "angular_z"]))

        group = f"follower_{modality}"
        cfg = getattr(self._robot, "_config", {}).get("joint_groups", {}).get(group, {})
        return list(cfg.get("joint_names", []))

    def _policy_flat_feature_dim(
        self, feature_collection: str, feature_name: str
    ) -> int | None:
        try:
            features = getattr(self._policy.config, feature_collection, {}) or {}
            feature = features.get(feature_name)
            shape = feature.get("shape") if isinstance(feature, dict) else getattr(feature, "shape", None)
            if not shape:
                return None
            dim = 1
            for item in shape:
                dim *= int(item)
            return dim
        except Exception:
            return None

    @staticmethod
    def _camera_policy_key_candidates(camera_name: str) -> set:
        aliases = {camera_name}
        parts = camera_name.split(".")
        suffix = parts[-1]
        prefixes = parts[:-1]
        aliases.add(suffix)

        semantic_names = {suffix}
        if suffix == "cam_left_head":
            semantic_names.add("cam_head")

        match = _CAMERA_SEMANTIC_RE.match(suffix)
        if match:
            first = match.group("a")
            second = match.group("b")
            side = first if first in {"left", "right"} else second
            part = first if first in {"head", "wrist"} else second
            if side in {"left", "right"} and part in {"head", "wrist"}:
                semantic_names.add(f"cam_{side}_{part}")
                semantic_names.add(f"cam_{part}_{side}")

        for name in semantic_names:
            aliases.add(name)
            aliases.add(f"rgb.{name}")
            if prefixes:
                aliases.add(".".join([*prefixes, name]))

        return {f"{_IMAGE_KEY_PREFIX}{alias}" for alias in aliases}
