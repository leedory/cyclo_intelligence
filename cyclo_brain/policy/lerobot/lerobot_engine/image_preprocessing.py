#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0

"""Rotation-only image preprocessing for LeRobot inference."""

from __future__ import annotations

import numpy as np


def apply_rotation(image: np.ndarray, rotation_deg: int | float | None) -> np.ndarray:
    """Apply camera rotation metadata using the same direction as OpenCV."""
    rotation = int(rotation_deg or 0) % 360
    if rotation == 0:
        return image
    if rotation == 90:
        return np.rot90(image, k=3)
    if rotation == 180:
        return np.rot90(image, k=2)
    if rotation == 270:
        return np.rot90(image, k=1)
    raise ValueError(f"unsupported camera rotation_deg={rotation_deg}")


def prepare_policy_image(
    image: np.ndarray,
    *,
    rotation_deg: int | float | None = 0,
) -> np.ndarray:
    """Apply only the camera's declared rotation; never resize live frames."""
    return apply_rotation(image, rotation_deg)
