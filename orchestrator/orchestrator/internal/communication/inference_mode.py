#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0

"""Inference output-mode helpers shared by UI and BT command dispatch."""

from __future__ import annotations


SIMULATION_MODE = "simulation"
ISAAC_MODE = "isaac"
ROBOT_MODE = "robot"


def normalize_inference_mode(value) -> str:
    mode = str(value or "").strip().lower()
    if mode in {ROBOT_MODE, "robot_mode", "publish", "publish_to_robot"}:
        return ROBOT_MODE
    if mode in {ISAAC_MODE, "isaac_sim", "isaac_simulation"}:
        return ISAAC_MODE
    return SIMULATION_MODE


def inference_mode_from_task_info(task_info) -> str:
    """Return robot/simulation from TaskInfo fields and tags."""
    mode = getattr(task_info, "inference_mode", "")
    if mode:
        return normalize_inference_mode(mode)

    tags = getattr(task_info, "tags", []) or []
    for tag in tags:
        normalized = str(tag or "").strip().lower()
        if normalized in {"inference_mode:robot", "publish_to_robot:true"}:
            return ROBOT_MODE
        if normalized in {"inference_mode:isaac", "publish_actions:true"}:
            return ISAAC_MODE
        if normalized in {"inference_mode:simulation", "publish_to_robot:false"}:
            return SIMULATION_MODE

    return ROBOT_MODE if bool(getattr(task_info, "publish_to_robot", False)) else SIMULATION_MODE


def publish_to_robot_from_task_info(task_info) -> bool:
    """Return whether the runtime may publish actions to its ROS command topics.

    In the workstation stack ``isaac`` targets Cyclo Lab. The legacy
    ``simulation`` value remains a non-publishing browser-preview mode and is
    never emitted by the workstation deployment UI.
    """
    return inference_mode_from_task_info(task_info) in {ISAAC_MODE, ROBOT_MODE}
