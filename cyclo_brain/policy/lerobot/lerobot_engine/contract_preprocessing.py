#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0

"""Strict observation construction for checkpoints with a policy contract."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch

from .constants import STATE_KEY
from .image_preprocessing import prepare_policy_image


def build_contract_observation(runtime: Any, task_instruction: str) -> Dict[str, Any]:
    assert runtime._robot is not None

    images = runtime._robot.get_images(format="rgb")
    if not images:
        return runtime._fail("No camera frames available")
    joint_dict = runtime._robot.get_joint_positions()
    if not joint_dict:
        return runtime._fail("No joint positions available")

    batch: Dict[str, Any] = {}
    for source, policy_key in runtime._cameras.items():
        image = images.get(source)
        if image is None:
            return runtime._fail(f"Missing camera frame: {source}")
        contract = runtime._camera_contracts[source]
        camera_config = runtime._robot._config.get("cameras", {}).get(source, {})
        try:
            image = prepare_policy_image(
                image,
                rotation_deg=camera_config.get("rotation_deg", 0),
            )
        except Exception as exc:
            return runtime._fail(f"Camera preprocessing failed for {source}: {exc}")
        expected_shape = (contract.height, contract.width, 3)
        if image.shape != expected_shape:
            return runtime._fail(
                f"Camera {source} produced shape {image.shape}, expected {expected_shape}"
            )
        tensor = torch.from_numpy(image.copy()).to(torch.float32) / 255.0
        batch[policy_key] = (
            tensor.permute(2, 0, 1).contiguous().unsqueeze(0).to(runtime._device)
        )

    odom = None
    values: List[float] = []
    for source in runtime._state_sources:
        if source.kind == "odom":
            if odom is None:
                odom = runtime._robot.get_odom()
            if odom is None:
                return runtime._fail(
                    f"Missing odom for state feature: {source.feature}"
                )
            mobile = (
                float(odom["linear_velocity"][0]),
                float(odom["linear_velocity"][1]),
                float(odom["angular_velocity"][2]),
            )
            values.append(mobile[source.index])
            continue

        positions = joint_dict.get(source.group)
        if positions is None or source.index >= len(positions):
            return runtime._fail(
                f"Missing exact joint source for {source.feature}: "
                f"group={source.group} index={source.index}"
            )
        values.append(float(positions[source.index]))

    expected = len(runtime._policy_contract.state_features)
    if len(values) != expected:
        return runtime._fail(
            f"State vector mismatch: resolved {len(values)} values, expected {expected}"
        )
    flat_state = np.asarray(values, dtype=np.float32)
    batch[STATE_KEY] = torch.from_numpy(flat_state).unsqueeze(0).to(runtime._device)
    batch["task"] = [task_instruction or ""]
    return batch
